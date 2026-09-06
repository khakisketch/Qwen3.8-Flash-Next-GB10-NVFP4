#!/usr/bin/env python3
"""
PLE codec benchmark on real Qwen3.8-Flash-Next n-gram embedding rows.

Compares reconstruction codecs (from ple_codecs.py) on the actual PLE shard
already downloaded, measuring row-wise error plus error propagated through the
real ``key_proj`` projection (head-0 slice). Emits a ranked table and a JSON
summary for downstream decisions.

Data: /tmp/opencode/qwen-manifest/shard05/model-00005-of-00131.safetensors
  ngram_embedding.shard_0.weight  [2500012, 160]
  ple.key_proj.weight             [10240, 2560]  (head 0 = cols 0:160)
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

from quantization.ple_codecs import CODECS, build_all

SHARD = "/tmp/opencode/qwen-manifest/shard05/model-00005-of-00131.safetensors"
NGT_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
KEYPROJ_KEY = "model.language_model.layers.1.ple.key_proj.weight"

PLE_TOTAL_PARAMS = 51.2e9  # for size projection


def load_tensor(path, key):
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def row_cosine(x, xq):
    x = x.float()
    xq = xq.float()
    num = (x * xq).sum(dim=1)
    den = x.norm(dim=1) * xq.norm(dim=1).clamp_min(1e-12)
    return num / den


def rel_l2(x, xq):
    return ((x.float() - xq.float()).norm() / x.float().norm()).item()


def snr_db(x, xq):
    x = x.float()
    num = (x ** 2).mean()
    err = ((x - xq.float()) ** 2).mean()
    return 10.0 * torch.log10(num / err.clamp_min(1e-30)).item()


def sat_frac(xq, x):
    """Fraction of elements reconstructed exactly at the top code (clipping)."""
    top = x.float().abs().amax(dim=1, keepdim=True)
    return (xq.float().abs() >= top * 0.999).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--codecs", default=None,
                    help="comma list; default all")
    ap.add_argument("--proj-samples", type=int, default=20000)
    ap.add_argument("--out", default="/tmp/opencode/qwen-manifest/ple_bench.json")
    args = ap.parse_args()

    x = load_tensor(args.shard, NGT_KEY).float()
    print(f"PLE shard_0: {tuple(x.shape)} dtype-from-BF16")

    # data distribution
    print(f"  data: mean={x.mean().item():.5g} std={x.std().item():.5g} "
          f"min={x.min().item():.5g} max={x.max().item():.5g}")
    row_amax = x.abs().amax(dim=1)
    row_min = x.min(dim=1).values
    row_max = x.max(dim=1).values
    print(f"  row amax: p50={row_amax.quantile(0.5).item():.4g} "
          f"p99={row_amax.quantile(0.99).item():.4g} max={row_amax.max().item():.4g}")
    print(f"  row span(min..max): p50={(row_max-row_min).quantile(0.5).item():.4g}")

    codec_names = args.codecs.split(",") if args.codecs else list(CODECS.keys())
    codecs = {n: CODECS[n]() for n in codec_names}

    results = {}
    print("\n=== Reconstruction (full shard_0) ===")
    print(f"{'codec':18s} {'bytes/el':>8s} {'PLE_GB':>7s} {'rel_l2':>8s} {'snr_dB':>7s} "
          f"{'cos_p50':>7s} {'cos_p95':>7s} {'cos_p99':>7s} {'maxabs':>8s} {'mae':>8s} {'sat%':>6s}")
    for name, (enc, dec, nbytes) in codecs.items():
        s = enc(x)
        xq = dec(s)
        rel = rel_l2(x, xq)
        snr = snr_db(x, xq)
        cos = row_cosine(x, xq)
        maxabs = (x - xq).abs().max().item()
        mae = (x - xq).abs().mean().item()
        sat = sat_frac(xq, x)
        bpe = nbytes(s) / x.numel()
        ple_gb = PLE_TOTAL_PARAMS * bpe / 1e9
        results[name] = dict(
            bytes_per_elem=round(bpe, 4), ple_size_gb=round(ple_gb, 2),
            rel_l2=round(rel, 5), snr_db=round(snr, 2),
            cos_p50=round(cos.quantile(0.5).item(), 5),
            cos_p95=round(cos.quantile(0.05).item(), 5),
            cos_p99=round(cos.quantile(0.01).item(), 5),
            max_abs_err=round(maxabs, 5), mae=round(mae, 5), sat_frac=round(sat * 100, 2),
        )
        print(f"{name:18s} {bpe:8.3f} {ple_gb:7.1f} {rel:8.5f} {snr:7.1f} "
              f"{cos.quantile(0.5).item():7.5f} {cos.quantile(0.05).item():7.5f} "
              f"{cos.quantile(0.01).item():7.5f} {maxabs:8.5f} {mae:8.5f} {sat*100:6.2f}")

    # Projection-aware error through key_proj head-0 slice
    W = load_tensor(args.shard, KEYPROJ_KEY)[:, :160].float()  # [10240, 160]
    print(f"\n=== Projection-aware (key_proj head-0, {args.proj_samples} rows) ===")
    idx = torch.randperm(x.shape[0])[: args.proj_samples]
    xs = x[idx]
    print(f"{'codec':18s} {'proj_rel_l2':>12s}")
    for name in codec_names:
        s = codecs[name][0](xs)
        xq = codecs[name][1](s)
        es = (xs - xq) @ W.T
        ss = xs @ W.T
        pr = (es.norm() / ss.norm().clamp_min(1e-12)).item()
        results[name]["proj_rel_l2"] = round(pr, 5)
        print(f"{name:18s} {pr:12.5f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()