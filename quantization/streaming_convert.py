#!/usr/bin/env python3
"""
Shard-streaming converter for Qwen3.8-Flash-Next.

Reads a local directory of BF16 safetensors shards + model.safetensors.index.json,
applies a per-tensor quantization policy, and writes a transformed checkpoint.
Never materializes the full model: one shard (~3.3GB) is held at a time.

Policies (v1):
  - ple_table     -> affine group-wise UINT4 (group=32) nibble-packed +
                     fp16 per-group scale + int16 per-group zero-point
  - moe_experts   -> block-wise NVFP4 (E2M1FN, block_k=16) packed + fp16 per-block scale
  - everything else -> copied as-is (BF16)

Outputs:
  - model-*.safetensors (transformed)
  - model.safetensors.index.json (rebuilt weight map)
  - config.json (rebuilt, quantization note injected)
  - ple_int4_manifest.json (layout so the runtime can dequant on demand)

--synthetic-test builds a tiny fake checkpoint in a temp dir and exercises the
whole pipeline (round-trip + index consistency) with no network/download.
"""

import argparse
import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantization.experts_nvfp4 import dequantize_experts_nvfp4, quantize_experts_nvfp4
from quantization.ple_int4 import dequantize_affine_group, quantize_affine_group

PLE_SUFFIX = ".ple.ple_embedding.ngram_embedding."
PLE_GROUP = 32
EXPERT_BLOCK_K = 16


def categorize(name: str) -> str:
    if PLE_SUFFIX in name:
        return "ple_table"
    if ".mlp.experts." in name:
        return "moe_experts"
    return "other"


def transform_ple(name: str, tensor: torch.Tensor):
    """Return {new_name: tensor, ...} for a PLE shard tensor (affine UINT4 g32)."""
    q = quantize_affine_group(tensor, group=PLE_GROUP)
    base = name[:-len(".weight")] if name.endswith(".weight") else name
    return {
        base + ".weight_packed": q["packed"],
        base + ".weight_scales": q["scales"],
        base + ".weight_zero_points": q["zero_points"],
    }, {
        "codec": "affine_uint4_g32",
        "group": q["group"],
        "orig_shape": q["orig_shape"],
        "scale_dtype": q["scale_dtype"],
        "zero_point_dtype": "int16",
    }


def transform_experts(name: str, tensor: torch.Tensor):
    """Return {new_name: tensor, ...} for a fused 3D expert tensor (NVFP4 g16)."""
    q = quantize_experts_nvfp4(tensor, block_k=EXPERT_BLOCK_K)
    base = name[:-len(".weight")] if name.endswith(".weight") else name
    return {
        base + ".weight_packed": q["weight_packed"],
        base + ".weight_scales": q["weight_scales"],
    }, {
        "codec": "nvfp4_e2m1fn",
        "block_k": q["block_k"],
        "orig_shape": q["orig_shape"],
        "scale_dtype": q["scale_dtype"],
    }


def transform_tensor(name: str, tensor: torch.Tensor):
    """Apply policy; returns (out_tensors: dict, ple_meta: dict|None)."""
    cat = categorize(name)
    if cat == "ple_table":
        out, meta = transform_ple(name, tensor)
        return out, meta
    if cat == "moe_experts":
        out, meta = transform_experts(name, tensor)
        return out, meta
    return {name: tensor}, None


def convert_shard(src_path: str, out_path: str, ple_manifest: dict, base_dir: str):
    with safe_open(src_path, framework="pt", device="cpu") as f:
        names = list(f.keys())
        out_tensors = {}
        for name in names:
            t = f.get_tensor(name)
            out, meta = transform_tensor(name, t)
            for oname, ot in out.items():
                out_tensors[oname] = ot.contiguous()
            if meta:
                # record manifest relative to the shard file
                ple_manifest[name] = {
                    "shard_file": os.path.basename(out_path),
                    **meta,
                }
    save_file(out_tensors, out_path, metadata={"format": "pt"})
    return list(out_tensors.keys()), names


def convert_dir(src_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(src_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    new_weight_map = {}
    ple_manifest = {}

    for shard in shards:
        src = os.path.join(src_dir, shard)
        out = os.path.join(out_dir, shard)
        out_names, in_names = convert_shard(src, out, ple_manifest, src_dir)
        for n in out_names:
            new_weight_map[n] = shard
        print(f"  {shard}: {len(in_names)} -> {len(out_names)} tensors "
              f"({os.path.getsize(out)/1e6:.1f} MB)")

    new_index = {"metadata": {"total_size": index.get("metadata", {}).get("total_size")},
                 "weight_map": new_weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=1)

    cfg_path = os.path.join(src_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg.setdefault("quantization_config", {})["ple_int4"] = (
            "PLE n-gram table: affine group-wise uint4 g32; MoE experts: nvfp4 e2m1fn "
            "block 16 (see ple_int4_manifest.json)")
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    with open(os.path.join(out_dir, "ple_int4_manifest.json"), "w") as f:
        json.dump(ple_manifest, f, indent=1)
    return new_index, ple_manifest


def synthetic_test():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        out = os.path.join(d, "out")
        os.makedirs(src)

        # two shards: shard A has a PLE tensor + an expert tensor; shard B other.
        ple = torch.randn(1024, 160) * 0.02 + torch.randn(1024, 1) * 0.05
        expert = torch.randn(8, 128, 2560) * 0.02
        other = torch.randn(2560, 2560) * 0.02

        a = {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": ple,
            "model.language_model.layers.1.mlp.experts.down_proj": expert,
        }
        b = {"model.language_model.embed_tokens.weight": other}
        save_file(a, os.path.join(src, "model-00001-of-00002.safetensors"))
        save_file(b, os.path.join(src, "model-00002-of-00002.safetensors"))

        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                list(a)[0]: "model-00001-of-00002.safetensors",
                list(a)[1]: "model-00001-of-00002.safetensors",
                list(b)[0]: "model-00002-of-00002.safetensors",
            },
        }
        with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"]}, f)

        print("[synthetic] converting...")
        new_index, manifest = convert_dir(src, out)

        # verify round-trip on the PLE tensor
        from safetensors import safe_open
        with safe_open(os.path.join(out, "model-00001-of-00002.safetensors"), framework="pt") as f:
            packed = f.get_tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight_packed")
            scales = f.get_tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight_scales")
            zps = f.get_tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight_zero_points")
            epacked = f.get_tensor(
                "model.language_model.layers.1.mlp.experts.down_proj.weight_packed")
            escales = f.get_tensor(
                "model.language_model.layers.1.mlp.experts.down_proj.weight_scales")
        reconst = dequantize_affine_group({
            "packed": packed, "scales": scales, "zero_points": zps,
            "orig_shape": [1024, 160],
        })
        rel = ((ple - reconst).norm() / ple.norm()).item()
        print(f"[synthetic] PLE affine_g32 round-trip rel_l2={rel:.5f}")
        assert rel < 0.2

        ereconst = dequantize_experts_nvfp4({
            "weight_packed": epacked, "weight_scales": escales,
            "orig_shape": [8, 128, 2560], "block_k": 16,
        })
        erel = ((expert - ereconst).norm() / expert.norm()).item()
        print(f"[synthetic] experts nvfp4_e2m1fn round-trip rel_l2={erel:.5f}")
        assert erel < 0.2

        assert new_index["weight_map"] and len(manifest) == 2
        codecs = sorted(m["codec"] for m in manifest.values())
        assert codecs == ["affine_uint4_g32", "nvfp4_e2m1fn"], codecs
        print("[synthetic] SYNTHETIC TEST PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-test", action="store_true")
    ap.add_argument("--src-dir", help="dir with config.json + shards + index")
    ap.add_argument("--out-dir", help="output dir")
    args = ap.parse_args()

    if args.synthetic_test or not args.src_dir:
        return synthetic_test()
    if not os.path.isdir(args.src_dir):
        raise SystemExit(f"src dir not found: {args.src_dir}")
    convert_dir(args.src_dir, args.out_dir)
    print("CONVERSION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())