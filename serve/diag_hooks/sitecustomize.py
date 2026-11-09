"""Child-safe, lazy import hook; imports no torch or vLLM in PID 1."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

_UTILS = "vllm.model_executor.model_loader.utils"
_BASE_LOADER = "vllm.model_executor.model_loader.base_loader"
_PROBED_HELPERS = {
    "vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils",
    "b12x.moe.fused_moe.planning",
    "b12x.moe.fused_moe._csf_preparation",
    "b12x._lib.quant._csf_prepare",
    "b12x.moe._shared.kernels.w4a16.prepare",
}


class _DiagnosticLoader:
    def __init__(self, fullname, delegate):
        self.fullname = fullname
        self.delegate = delegate

    def create_module(self, spec):
        create = getattr(self.delegate, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        if self.fullname == _UTILS:
            from layer_diagnostics import install_utils_module
            install_utils_module(module)
        elif self.fullname == _BASE_LOADER:
            from layer_diagnostics import rewire_base_loader_alias
            utils = sys.modules.get(_UTILS)
            if utils is None or not getattr(utils, "_qwen38_memdiag_installed", False):
                raise RuntimeError("diagnostic loader hook missing before base_loader")
            rewire_base_loader_alias(module, utils)
        elif self.fullname in _PROBED_HELPERS:
            from layer_diagnostics import install_helper_module
            install_helper_module(module)


class VllmDiagnosticFinder(importlib.abc.MetaPathFinder):
    """Wrap only vLLM loader imports once the child actually imports them."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_UTILS, _BASE_LOADER) and fullname not in _PROBED_HELPERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _DiagnosticLoader(fullname, spec.loader)
        return spec


def register_import_hook():
    if not any(isinstance(finder, VllmDiagnosticFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, VllmDiagnosticFinder())


if os.environ.get("QWEN38_MEMORY_DIAGNOSTICS") == "1":
    register_import_hook()
