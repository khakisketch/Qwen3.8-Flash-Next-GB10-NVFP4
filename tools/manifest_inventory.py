#!/usr/bin/env python3
"""
Authoritative tensor manifest + size projection for Qwen3.8-Flash-Next.

Builds a full inventory from the HF shard *headers* (safetensors first-N-bytes
range requests) without downloading the 360GB of weights, then categorizes
tensors and projects on-disk/resident sizes under a candidate quantization
policy.

Outputs:
  - manifest.json          (per-tensor: dtype, shape, bytes, shard)
  - manifest_summary.json  (per-category aggregates + projections)
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
    "FP8": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1,
}

URL = "https://huggingface.co/{model_id}/resolve/main/{fname}"


def http_range(url, nbytes):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_header(data):
    import struct
    hlen = struct.unpack("<Q", data[:8])[0]
    body = data[8:8 + hlen]
    return json.loads(body.decode("utf-8"))


def categorize(name):
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "embed_tokens"
    if ".mlp.experts." in name:
        return "moe_experts"
    if ".mlp.shared_expert" in name or ".mlp.shared_expert_gate" in name:
        return "shared_expert"
    if ".mlp.gate.weight" in name:
        return "router"
    if ".ple.ple_embedding.ngram_embedding" in name:
        return "ple_table"
    if ".ple." in name:
        return "ple_proj"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if "hyper_connection" in name:
        return "hyper_connection"
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--out-dir", default="manifest")
    args = ap.parse_args()

    index = json.loads(http_range(
        URL.format(model_id=args.model_id, fname="model.safetensors.index.json"),
        1 << 20))
    weight_map = index["weight_map"]

    shards = sorted(set(weight_map.values()))
    print(f"[*] {len(weight_map)} tensors across {len(shards)} shards", file=sys.stderr)

    manifest = {}
    errs = []

    def fetch_shard(shard):
        try:
            data = http_range(
                URL.format(model_id=args.model_id, fname=shard), 1 << 16)
            return shard, parse_header(data), None
        except Exception as e:  # noqa: BLE001
            return shard, None, repr(e)

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch_shard, s) for s in shards]
        for fut in as_completed(futs):
            shard, header, err = fut.result()
            if err:
                errs.append(f"{shard}: {err}")
                continue
            shared = header.pop("__metadata__", None) or header.pop("_shared", None)
            for tname, spec in header.items():
                if tname == "__metadata__":
                    continue
                dtype = spec["dtype"]
                shape = spec["shape"]
                b0, b1 = spec["data_offsets"]
                nbytes = b1 - b0
                nparams = 1
                for d in shape:
                    nparams *= d
                manifest[tname] = {
                    "dtype": dtype, "shape": shape, "bytes": nbytes,
                    "params": nparams, "shard": shard, "category": categorize(tname),
                }
            done += 1
            if done % 20 == 0:
                print(f"[*] {done}/{len(shards)} shards indexed", file=sys.stderr, flush=True)

    missing = [t for t in weight_map if t not in manifest]
    for t in missing:
        errs.append(f"missing tensor {t}")

    # aggregate
    cats = defaultdict(lambda: {"params": 0, "bytes": 0, "tensors": 0})
    for t, m in manifest.items():
        c = cats[m["category"]]
        c["params"] += m["params"]
        c["bytes"] += m["bytes"]
        c["tensors"] += 1

    total_params = sum(c["params"] for c in cats.values())
    total_bytes = sum(c["bytes"] for c in cats.values())

    # projection policy (bytes-per-param by category)
    policy = {
        "moe_experts": 0.5,   # NVFP4
        "ple_table": 0.5,     # INT4 row-wise (+scales below)
        "shared_expert": 1.0, # FP8
        "embed_tokens": 2.0,  # BF16
        "lm_head": 2.0,       # BF16
        "router": 2.0,        # BF16
        "linear_attention": 2.0,
        "full_attention": 1.0,  # FP8 candidate
        "hyper_connection": 2.0,
        "ple_proj": 2.0,
        "vision": 2.0,
        "mtp": 0.5 if False else 2.0,  # keep BF16 first pass
        "other": 2.0,
    }
    projected = {}
    for cat, c in cats.items():
        bpp = policy.get(cat, 2.0)
        projected[cat] = c["params"] * bpp

    summary = {
        "model_id": args.model_id,
        "total_params": total_params,
        "total_bytes_bf16": total_bytes,
        "categories": {c: dict(cats[c]) for c in sorted(cats)},
        "projected_bytes": projected,
        "projected_total_bytes": sum(projected.values()),
        "errors": errs,
    }

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out_dir, "manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Category breakdown (BF16) ===")
    for c in sorted(cats, key=lambda x: -cats[x]["bytes"]):
        b = cats[c]["bytes"]
        print(f"  {c:18s} {cats[c]['params']/1e9:8.3f}B params  {b/1e9:8.2f} GB")
    print(f"  {'TOTAL':18s} {total_params/1e9:8.3f}B params  {total_bytes/1e9:8.2f} GB")
    print("\n=== Projected resident/disk bytes under candidate policy ===")
    for c in sorted(projected, key=lambda x: -projected[x]):
        if projected[c] < 1e6:
            continue
        print(f"  {c:18s} {projected[c]/1e9:8.2f} GB  ({policy.get(c,2.0):>3} B/param)")
    print(f"  {'TOTAL (projected)':18s} {sum(projected.values())/1e9:8.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())