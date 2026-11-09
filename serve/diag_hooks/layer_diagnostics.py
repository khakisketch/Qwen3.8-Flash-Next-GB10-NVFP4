"""Best-effort memory snapshots around pinned vLLM post-load boundaries.

Instrumentation failures are emitted but never replace model exceptions. The
18 GiB watchdog remains authoritative; this module only records observations.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager

_local = threading.local()
_CSF_POSTLOAD = threading.local()
_MIN_RECLAIMABLE_BYTES = 512 * 1024**2

_PROBED_HELPERS = {
    "vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils":
        ("prepare_nvfp4_csf_weights",),
    "b12x.moe.fused_moe.planning": ("prepare_weights",),
    "b12x.moe.fused_moe._csf_preparation": (
        "prepare_csf_block_weights", "prepare_online_csf_weights",
    ),
    "b12x._lib.quant._csf_prepare": (
        "validate_source", "expand_source_scales", "prepare_csf_payload",
    ),
    "b12x.moe._shared.kernels.w4a16.prepare": (
        "_nvfp4_compute_scale_factor", "_prepare_w4a16_packed_weights",
        "prepare_w4a16_modelopt_nvfp4_weights",
    ),
}


def _emit(event, emit=None, **fields):
    record = {"time": time.time(), "event": event, **fields}
    try:
        if emit is not None:
            emit(record)
        else:
            print("QWEN38_MEMDIAG " + json.dumps(record, default=str),
                  file=sys.stderr, flush=True)
    except Exception as exc:
        # Telemetry is observational: a full pipe or a broken test sink must
        # not prevent the instrumented function from running or mask errors.
        try:
            fallback = getattr(sys, "__stderr__", None)
            if fallback is not None:
                fallback.write(
                    f"QWEN38_MEMDIAG telemetry_error={type(exc).__name__}:{exc}\n"
                )
                fallback.flush()
        except Exception:
            pass


def _safe_memory(memory):
    try:
        return dict(memory())
    except Exception as exc:
        return {"memory_snapshot_error": f"{type(exc).__name__}:{exc}"}


def process_rss_bytes(path="/proc/self/status"):
    with open(path, encoding="ascii") as source:
        for line in source:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing from process status")


def start_rss_sampler(*, interval=1.0, stop_event=None, emit=None,
                      sampler=process_rss_bytes):
    """Read only process RSS periodically; never inspect tensors or CUDA."""
    stop = stop_event or threading.Event()

    def sample_loop():
        while True:
            try:
                value = sampler()
                _emit("RSS_SAMPLE", emit, pid=os.getpid(), process_rss_bytes=value)
            except Exception as exc:
                _emit("RSS_SAMPLE_ERROR", emit, pid=os.getpid(),
                      error_type=type(exc).__name__, error=str(exc))
            if stop.wait(interval):
                return

    thread = threading.Thread(target=sample_loop, name="qwen38-rss-sampler", daemon=True)
    thread.start()
    return thread


def wrap_helper(original, helper, *, emit=None, memory=None):
    if memory is None:
        memory = memory_snapshot

    def wrapped(*args, **kwargs):
        scale_factor = helper.endswith("._nvfp4_compute_scale_factor")
        if helper.endswith("._prepare_w4a16_packed_weights"):
            maybe_reclaim_csf_cache("before _prepare_w4a16_packed_weights",
                                    emit=emit, memory=memory)
        if not scale_factor:
            _emit("HELPER_ENTER", emit, helper=helper, **_safe_memory(memory))
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _emit("HELPER_ERROR", emit, helper=helper, **_safe_memory(memory),
                  error_type=type(exc).__name__, error=str(exc))
            raise
        if helper.endswith("._nvfp4_compute_scale_factor"):
            maybe_reclaim_csf_cache("after _nvfp4_compute_scale_factor",
                                    emit=emit, memory=memory)
        if not scale_factor:
            _emit("HELPER_EXIT", emit, helper=helper, **_safe_memory(memory))
        return result
    wrapped._qwen38_memdiag_helper = helper
    return wrapped


def csf_postload_active():
    return bool(getattr(_CSF_POSTLOAD, "active", False))


def maybe_reclaim_csf_cache(boundary, *, active=None, enabled=None,
                            torch_module=None, emit=None, memory=None):
    """Return unused CUDA cache only during opt-in NVFP4-CSF post-load."""
    if active is None:
        active = csf_postload_active()
    if enabled is None:
        enabled = os.environ.get("QWEN38_RECLAIM_CSF_CACHE") == "1"
    if not active or not enabled:
        return False
    if memory is None:
        memory = memory_snapshot
    try:
        if torch_module is None:
            import torch as torch_module
        before = dict(memory())
        allocated = before.get("torch_allocated_bytes")
        reserved = before.get("torch_reserved_bytes")
        if not isinstance(allocated, int) or not isinstance(reserved, int):
            return False
        if reserved - allocated < _MIN_RECLAIMABLE_BYTES:
            return False
        _emit("CACHE_RECLAIM_BEFORE", emit, boundary=boundary, **before)
        torch_module.cuda.empty_cache()
        after = dict(memory())
        _emit("CACHE_RECLAIM_AFTER", emit, boundary=boundary, **after)
        return True
    except Exception as exc:
        _emit("CACHE_RECLAIM_ERROR", emit, boundary=boundary,
              error_type=type(exc).__name__, error=str(exc))
        return False


def install_helper_module(module):
    """Wrap selected helpers and aliases, without inspecting their arguments."""
    names = _PROBED_HELPERS.get(getattr(module, "__name__", ""), ())
    installed = []
    for name in names:
        original = getattr(module, name, None)
        if not callable(original) or getattr(original, "_qwen38_memdiag_helper", None):
            continue
        helper = f"{module.__name__}.{name}"
        wrapped = wrap_helper(original, helper)
        setattr(module, name, wrapped)
        installed.append(name)
        # Patch aliases already imported by a consumer; future imports receive
        # the wrapper directly from the source module.
        for consumer in tuple(sys.modules.values()):
            try:
                namespace = vars(consumer)
            except TypeError:
                continue
            for alias, value in tuple(namespace.items()):
                if value is original:
                    try:
                        setattr(consumer, alias, wrapped)
                    except (AttributeError, TypeError):
                        pass
    if installed:
        _emit("HELPERS_INSTALLED", pid=os.getpid(), module=module.__name__,
              functions=installed)


def emit_install_marker(*, emit=None, **fields):
    _emit("HOOK_INSTALLED", emit, pid=os.getpid(), **fields)


def rewire_base_loader_alias(base_loader, utils):
    """Keep the pinned base_loader import alias on the wrapped callable."""
    base_loader.process_weights_after_loading = utils.process_weights_after_loading
    _emit("LOADER_ALIAS_PATCHED", pid=os.getpid(),
          module="vllm.model_executor.model_loader.base_loader")


def install_quant_class_hooks(cls, *, emit=None, memory=None):
    """Wrap existing and future QuantizeMethodBase implementations."""
    if memory is None:
        memory = memory_snapshot
    if getattr(cls, "_qwen38_memdiag_hooked", False):
        return 0
    original_init_subclass = cls.__dict__.get("__init_subclass__")
    wrapped_count = 0
    base_method = cls.__dict__.get("process_weights_after_loading")
    if base_method is not None and not getattr(base_method, "_qwen38_memdiag", False):
        instrumented = wrap_quant_method(base_method, emit=emit, memory=memory)
        instrumented._qwen38_memdiag = True
        setattr(cls, "process_weights_after_loading", instrumented)
        wrapped_count += 1
    seen = set()

    def visit(parent):
        nonlocal wrapped_count
        for child in parent.__subclasses__():
            if child in seen:
                continue
            seen.add(child)
            own = child.__dict__.get("process_weights_after_loading")
            if own is not None and not getattr(own, "_qwen38_memdiag", False):
                instrumented = wrap_quant_method(own, emit=emit, memory=memory)
                instrumented._qwen38_memdiag = True
                setattr(child, "process_weights_after_loading", instrumented)
                wrapped_count += 1
            visit(child)
    visit(cls)

    @classmethod
    def on_subclass(subclass, **kwargs):
        if original_init_subclass is not None:
            original_init_subclass.__get__(None, cls)(subclass, **kwargs)
        own = subclass.__dict__.get("process_weights_after_loading")
        if own is not None and not getattr(own, "_qwen38_memdiag", False):
            instrumented = wrap_quant_method(own, emit=emit, memory=memory)
            instrumented._qwen38_memdiag = True
            setattr(subclass, "process_weights_after_loading", instrumented)

    cls.__init_subclass__ = on_subclass
    cls._qwen38_memdiag_hooked = True
    return wrapped_count


def _host_available():
    with open("/proc/meminfo", encoding="ascii") as source:
        for line in source:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def memory_snapshot():
    result = {"host_memavailable_bytes": None,
              "torch_allocated_bytes": None, "torch_reserved_bytes": None,
              "process_rss_bytes": None}
    try:
        result["host_memavailable_bytes"] = _host_available()
    except Exception as exc:
        result["host_mem_error"] = f"{type(exc).__name__}:{exc}"
    try:
        result["process_rss_bytes"] = process_rss_bytes()
    except Exception as exc:
        result["process_rss_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import torch
        result["torch_allocated_bytes"] = int(torch.cuda.memory_allocated())
        result["torch_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception as exc:
        result["torch_memory_error"] = f"{type(exc).__name__}:{exc}"
    return result


def parameter_bytes(module):
    total = 0
    try:
        for _, param in module.named_parameters(recurse=True):
            total += int(param.numel()) * int(param.element_size())
        return total
    except Exception:
        return None


def _snapshot(module, memory):
    try:
        result = dict(memory())
    except Exception as exc:
        result = {"snapshot_error": f"{type(exc).__name__}:{exc}"}
    result["parameter_bytes"] = parameter_bytes(module)
    return result


def wrap_quant_method(original, *, emit=None, memory=memory_snapshot):
    def wrapped(method, layer, *args, **kwargs):
        names = getattr(_local, "module_names", {})
        layer_name = names.get(id(layer)) or getattr(layer, "prefix", None) or type(layer).__name__
        identity = {"layer": layer_name, "method": type(method).__name__}
        _emit("ENTER", emit, **identity, **_snapshot(layer, memory))
        previous_csf_scope = getattr(_CSF_POSTLOAD, "active", False)
        if type(method).__name__ == "Nvfp4CsfMoEMethod":
            _CSF_POSTLOAD.active = True
        try:
            result = original(method, layer, *args, **kwargs)
        except BaseException as exc:
            _emit("ERROR", emit, **identity, **_snapshot(layer, memory),
                  error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            _CSF_POSTLOAD.active = previous_csf_scope
        _emit("EXIT", emit, **identity, **_snapshot(layer, memory))
        return result
    return wrapped


def wrap_model_postload(original, *, emit=None, memory=memory_snapshot):
    def wrapped(model, config, target_device, *args, **kwargs):
        _local.module_names = {id(module): name for name, module in model.named_modules()}
        model_hook = getattr(model, "process_weights_after_loading", None)
        if callable(model_hook):
            def instrumented_model_hook(*hook_args, **hook_kwargs):
                _emit("MODEL_HOOK_ENTER", emit, boundary="model.process_weights_after_loading",
                      **_snapshot(model, memory))
                try:
                    value = model_hook(*hook_args, **hook_kwargs)
                except BaseException as exc:
                    _emit("MODEL_HOOK_ERROR", emit, boundary="model.process_weights_after_loading",
                          **_snapshot(model, memory), error_type=type(exc).__name__, error=str(exc))
                    raise
                _emit("MODEL_HOOK_EXIT", emit, boundary="model.process_weights_after_loading",
                      **_snapshot(model, memory))
                return value
            model.process_weights_after_loading = instrumented_model_hook
        _emit("POSTLOAD_ENTER", emit, boundary="retie+conversion+quant+attention+model",
              **_snapshot(model, memory))
        try:
            result = original(model, config, target_device, *args, **kwargs)
        except BaseException as exc:
            _emit("POSTLOAD_ERROR", emit, boundary="retie+conversion+quant+attention+model",
                  **_snapshot(model, memory), error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            _local.module_names = {}
        _emit("POSTLOAD_EXIT", emit, boundary="retie+conversion+quant+attention+model",
              **_snapshot(model, memory))
        return result
    return wrapped


def install_utils_module(utils):
    """Instrument the actual pinned loader module after it has executed."""
    if getattr(utils, "_qwen38_memdiag_installed", False):
        return
    QuantizeMethodBase = utils.QuantizeMethodBase
    original_retie = utils.maybe_retie_word_embeddings
    def retie(model, config):
        _emit("RETIE_ENTER", boundary="maybe_retie_word_embeddings", **_snapshot(model, memory_snapshot))
        try:
            value = original_retie(model, config)
        except BaseException as exc:
            _emit("RETIE_ERROR", boundary="maybe_retie_word_embeddings",
                  **_snapshot(model, memory_snapshot), error_type=type(exc).__name__, error=str(exc))
            raise
        _emit("RETIE_EXIT", boundary="maybe_retie_word_embeddings", **_snapshot(model, memory_snapshot))
        return value
    utils.maybe_retie_word_embeddings = retie
    quant_methods_wrapped = install_quant_class_hooks(QuantizeMethodBase)
    original_context = utils.device_loading_context
    @contextmanager
    def monitored_context(module, target_device):
        names = getattr(_local, "module_names", {})
        layer_name = names.get(id(module)) or getattr(module, "prefix", None) or type(module).__name__
        identity = {"layer": layer_name, "boundary": "device_loading_context"}
        _emit("CONVERSION_ENTER", **identity, **_snapshot(module, memory_snapshot))
        try:
            with original_context(module, target_device) as moved:
                yield moved
        except BaseException as exc:
            _emit("CONVERSION_ERROR", **identity, **_snapshot(module, memory_snapshot),
                  error_type=type(exc).__name__, error=str(exc))
            raise
        _emit("CONVERSION_EXIT", **identity, **_snapshot(module, memory_snapshot))
    utils.device_loading_context = monitored_context
    utils.process_weights_after_loading = wrap_model_postload(utils.process_weights_after_loading)
    utils._qwen38_memdiag_installed = True
    import sys as _sys
    base_loader = _sys.modules.get("vllm.model_executor.model_loader.base_loader")
    if base_loader is not None:
        rewire_base_loader_alias(base_loader, utils)
    emit_install_marker(
        modules_processed="vllm.model_executor.model_loader.utils",
        existing_quant_methods_wrapped=quant_methods_wrapped,
    )
    start_rss_sampler()
    for module_name in _PROBED_HELPERS:
        module = _sys.modules.get(module_name)
        if module is not None:
            install_helper_module(module)

