#!/usr/bin/env python3
"""
Verification for the NVMe-backed PLE runtime.

Test 1 (real data, store level): quantize real shard05 ngram shard_0/shard_1 into a
temp converted dir, build PleStore, and check row lookups match the codec dequant.

Test 2 (module level, tiny config): build a tiny ngram config, quantize a synthetic
[pad_vocab,160] embedding into 4 shard files, run QuantizedNGramEmbedding.forward and
compare against the reference Qwen4ExpTextNGramEmbedding over the same dequantized
table. Both use identical hashing; outputs must match essentially exactly.

Usage: .venv/bin/python runtime/verify_ple_runtime.py [--device cuda|cpu]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantization.ple_int4 import dequantize_affine_group, quantize_affine_group

REAL_SHARD = "/tmp/opencode/qwen-manifest/shard05/model-00005-of-00131.safetensors"
NGRAM_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _quantize_to_file(x, path, base, key):
    d = quantize_affine_group(x, group=32)
    save_file(
        {
            base + ".weight_packed": d["packed"],
            base + ".weight_scales": d["scales"],
            base + ".weight_zero_points": d["zero_points"],
        },
        path,
    )
    return {
        "codec": "affine_uint4_g32",
        "group": 32,
        "orig_shape": d["orig_shape"],
        "scale_dtype": d["scale_dtype"],
        "zero_point_dtype": "int16",
    }


def test_real_rows(device, tmp):
    # ---- build converted dir from real shard_0/shard_1 ----
    conv = os.path.join(tmp, "real")
    os.makedirs(conv)
    manifest = {}
    out_tensors = {}
    outs = []
    with safe_open(REAL_SHARD, framework="pt", device="cpu") as f:
        for si in (0, 1):
            key = f"{NGRAM_PREFIX}.shard_{si}.weight"
            x = f.get_tensor(key)
            base = key[:-len(".weight")]
            d = quantize_affine_group(x, group=32)
            out_tensors[base + ".weight_packed"] = d["packed"]
            out_tensors[base + ".weight_scales"] = d["scales"]
            out_tensors[base + ".weight_zero_points"] = d["zero_points"]
            manifest[key] = {
                "shard_file": "ple_real.safetensors",
                "codec": "affine_uint4_g32", "group": 32,
                "orig_shape": d["orig_shape"], "scale_dtype": d["scale_dtype"],
                "zero_point_dtype": "int16",
            }
            outs.append((x, base))
    save_file(out_tensors, os.path.join(conv, "ple_real.safetensors"))
    with open(os.path.join(conv, "ple_int4_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)

    from runtime.ple_store import PleStore
    store = PleStore(conv, device=device, cache_rows=500_000)

    torch.manual_seed(0)
    n_rows = manifest[f"{NGRAM_PREFIX}.shard_0.weight"]["orig_shape"][0]
    # sample rows across BOTH shards, plus shard boundary edges
    r0 = torch.randperm(2 * n_rows)[:4000]
    edges = torch.tensor([0, 1, n_rows - 1, n_rows, n_rows + 1, 2 * n_rows - 1])
    rows = torch.unique(torch.cat([r0, edges]))
    got = store.lookup(rows.to(torch.long)).float().cpu()

    # reference: dequantize each full shard once, then gather by row
    x0, x1 = outs[0][0], outs[1][0]
    xq0 = dequantize_affine_group(quantize_affine_group(x0, group=32))
    xq1 = dequantize_affine_group(quantize_affine_group(x1, group=32))
    ref_rows = []
    for r in rows.tolist():
        shard = r // n_rows
        off = r % n_rows
        ref_rows.append(xq0[off] if shard == 0 else xq1[off])
    ref = torch.stack(ref_rows).to(torch.bfloat16).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    max_abs = (got - ref).abs().max().item()
    print(f"[real-store] rows={len(rows)} rel_l2={rel:.3e} max_abs={max_abs:.3e}")
    assert rel < 1e-5 and max_abs < 1e-3, (rel, max_abs)
    print("[real-store] PLE store lookup matches affine_uint4_g32 codec: PASS")
    store.close()
    return n_rows


def test_module_tiny(device, tmp, text_cfg):
    torch.manual_seed(1)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    # tiny ngram vocab: padded table ~ few 10k rows
    tiny = Qwen4ExpTextConfig(**{**text_cfg.to_dict(), "ngram_vocab_size_base": 2 ** 10})
    emb_dim = tiny.ple_embed_dim

    # build a reference quantized-dequantized table and 4-shard converted dir
    conv = os.path.join(tmp, "tiny")
    os.makedirs(conv)
    manifest = {}
    shard_files = []
    for si in range(4):
        base = f"{NGRAM_PREFIX}.shard_{si}"
        x = torch.randn(32, 160)  # placeholder; replaced below
        out_file = os.path.join(conv, f"tiny_{si}.safetensors")
        m = _quantize_to_file(x, out_file, base, si)
        manifest[f"{NGRAM_PREFIX}.shard_{si}.weight"] = {"shard_file": os.path.basename(out_file), **m}
        shard_files.append(out_file)
    # redo with real table size: padded_vocab_size
    ref_emb = Qwen4ExpTextNGramEmbedding(tiny, emb_dim, 0, 0)
    padded = ref_emb.ngram_embedding.weight.shape[0]
    table = (torch.randn(padded, 160) * 0.02 + torch.randn(padded, 1) * 0.05)
    rows_per = padded // 4
    for si in range(4):
        x = table[si * rows_per:(si + 1) * rows_per]
        base = f"{NGRAM_PREFIX}.shard_{si}"
        out_file = os.path.join(conv, f"tiny_{si}.safetensors")
        m = _quantize_to_file(x, out_file, base, si)
        manifest[f"{NGRAM_PREFIX}.shard_{si}.weight"] = {"shard_file": os.path.basename(out_file), **m}
        # reference embedding over the dequantized table
        ref_emb.ngram_embedding.weight.data[si * rows_per:(si + 1) * rows_per] = \
            dequantize_affine_group(quantize_affine_group(x, group=32)).to(torch.bfloat16)
    with open(os.path.join(conv, "ple_int4_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)

    from runtime.ple_store import PleStore
    from runtime.patch_qwen4_exp import QuantizedNGramEmbedding
    store = PleStore(conv, device=device, cache_rows=50_000)
    mod = QuantizedNGramEmbedding(tiny, emb_dim, 0, 0, store, device).to(device).eval()

    ids = torch.randint(0, tiny.vocab_size, (2, 17), device=device)
    with torch.no_grad():
        got = mod(input_ids=ids, past_key_values=None).float().cpu()
        ref = ref_emb(input_ids=ids.to("cpu"), past_key_values=None).float().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"[tiny-module] ids={tuple(ids.shape)} padded_vocab={padded} "
          f"output={tuple(got.shape)} rel_l2={rel:.3e}")
    assert got.shape == (2, 17, emb_dim), got.shape
    assert rel < 1e-4, rel
    print("[tiny-module] QuantizedNGramEmbedding.forward == reference: PASS")
    store.close()


def test_integration_small_model(device, real_cfg, tmp):
    """Full-model end-to-end: scaled-down qwen4_exp, PLE module patched to NVMe store."""
    import torch.nn as nn
    from transformers import Qwen4ExpForConditionalGeneration
    from tools.architecture_smoke_test import build_small_config

    torch.manual_seed(0)
    small = build_small_config(real_cfg, layers=4, experts=16)  # >= num_experts_per_tok=10
    model = Qwen4ExpForConditionalGeneration(small).to(torch.bfloat16).to(device).eval()
    print(f"[integration] small model built: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    ids = torch.tensor([[151644, 8948, 198, 2610, 525, 264, 10950, 17847]], device=device)
    with torch.no_grad():
        ref_logits = model(input_ids=ids).logits.float().cpu()

    # locate the PLE embedding, quantize its table into a converted dir
    emb = model.model.language_model.layers[1].ple.ple_embedding
    table = emb.ngram_embedding.weight.data.float()  # [padded, 160]
    padded = table.shape[0]
    n_shards = 4
    rows_per = padded // n_shards
    conv = os.path.join(tmp, "integ")
    os.makedirs(conv)
    manifest = {}
    out_tensors = {}
    for si in range(n_shards):
        base = f"{NGRAM_PREFIX}.shard_{si}"
        d = quantize_affine_group(table[si * rows_per:(si + 1) * rows_per], group=32)
        for suf, t in [("weight_packed", d["packed"]), ("weight_scales", d["scales"]),
                       ("weight_zero_points", d["zero_points"])]:
            out_tensors[base + "." + suf] = t
        manifest[f"{NGRAM_PREFIX}.shard_{si}.weight"] = {
            "shard_file": "integ.safetensors", "codec": "affine_uint4_g32", "group": 32,
            "orig_shape": d["orig_shape"], "scale_dtype": d["scale_dtype"],
            "zero_point_dtype": "int16",
        }
    save_file(out_tensors, os.path.join(conv, "integ.safetensors"))
    with open(os.path.join(conv, "ple_int4_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)

    from runtime.ple_store import PleStore
    from runtime.patch_qwen4_exp import patch_ngram_embedding
    store = PleStore(conv, device=device, cache_rows=20_000)
    n = patch_ngram_embedding(model, store, device=device)
    print(f"[integration] patched {n} ngram embedding(s), padded_vocab={padded}")
    assert n == 1

    with torch.no_grad():
        got_logits = model(input_ids=ids).logits.float().cpu()
    rel = ((got_logits - ref_logits).norm() / ref_logits.norm()).item()
    agree = (got_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    print(f"[integration] forward logits rel_l2={rel:.4f} top1-agree={agree:.2f} "
          f"shape={tuple(got_logits.shape)}")
    assert got_logits.shape == ref_logits.shape
    assert rel < 0.25, rel  # PLE quant ~7% projected through tiny random model
    assert agree >= 0.5, agree

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=4, do_sample=False)
    print(f"[integration] generate OK: {gen[0].tolist()}")
    assert gen.shape[1] == ids.shape[1] + 4
    store.close()
    print("[integration] patched small model forward + generate: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", default="/tmp/opencode/qwen-manifest/config.json")
    args = ap.parse_args()
    print(f"device={args.device}")

    from transformers import Qwen4ExpConfig
    real_cfg = Qwen4ExpConfig.from_pretrained(args.config)

    tmp = tempfile.mkdtemp(prefix="ple_runtime_")
    try:
        test_real_rows(args.device, tmp)
        test_module_tiny(args.device, tmp, real_cfg.text_config)
        test_integration_small_model(args.device, real_cfg, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PLE RUNTIME VERIFICATION PASSED")


if __name__ == "__main__":
    main()