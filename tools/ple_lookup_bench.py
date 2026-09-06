#!/usr/bin/env python3
"""
GB10 PLE lookup / dequant microbenchmark.

Answers the serving-architecture decision: can PLE rows be served from
unified-memory (resident) at decode rate, and how expensive is NVMe as the
cold tier. Measures memory-bound gather+dequant for the three representative
footprints (bf16 / fp8 / int4) and NVMe sequential + random read behaviour.

Run on the DGX Spark (sm_121). Uses the already-downloaded PLE shard.
"""

import os
import random
import time

import torch
from safetensors import safe_open

from quantization.ple_codecs import CODECS
from quantization.ple_int4 import pack_int4

SHARD = "/tmp/opencode/qwen-manifest/shard05/model-00005-of-00131.safetensors"
NGT_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
R, C = 2500012, 160


def timeit(fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def resident_bench():
    print("=== UMA resident gather+dequant ===")
    with safe_open(SHARD, framework="pt", device="cpu") as f:
        bf16 = f.get_tensor(NGT_KEY)  # [R,160] bf16, mmap-backed
    x = bf16.float()

    # build representative encoded forms
    enc_fp8, dec_fp8, _ = CODECS["fp8_row"]()
    s_fp8 = enc_fp8(x)
    fp8 = s_fp8["w"]            # [R,160] float8
    fp8_scale = s_fp8["scale"]  # [R,1] fp16

    # symmetric int4 row-wise (nibble packed), for footprint timing
    q = (x / (x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0)).round().clamp(-8, 7).to(torch.int8)
    int4 = pack_int4(q)         # [R,80] uint8
    scale = (x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0).to(torch.float16)

    torch.manual_seed(0)
    for batch, label in [(16, "decode (16 rows/token)"), (4096, "prefill (4096 rows)")]:
        idx = torch.randint(0, R, (batch,))

        def bench_bf16():
            return torch.index_select(bf16, 0, idx).float()

        def bench_fp8():
            w = torch.index_select(fp8, 0, idx)
            sc = torch.index_select(fp8_scale, 0, idx)
            return w.float() * sc.float()

        def bench_int4():
            w = torch.index_select(int4, 0, idx)
            sc = torch.index_select(scale, 0, idx)
            # unpack + dequant
            lo = w & 0xF
            hi = (w >> 4) & 0xF
            lo = torch.where(lo >= 8, lo - 16, lo)
            hi = torch.where(hi >= 8, hi - 16, hi)
            qq = torch.stack((lo, hi), dim=-1).reshape(batch, -1).float()
            return qq * sc.float()

        t_bf16 = timeit(bench_bf16)
        t_fp8 = timeit(bench_fp8)
        t_i4 = timeit(bench_int4)
        bytes_bf16 = batch * C * 2
        bytes_fp8 = batch * C * 1
        bytes_i4 = batch * (C // 2) * 1
        print(f"  {label}:")
        print(f"    bf16  {t_bf16*1e6:8.1f} us  ({bytes_bf16/1e6:.2f} MB -> {bytes_bf16/1e6/t_bf16:6.1f} GB/s)")
        print(f"    fp8   {t_fp8*1e6:8.1f} us  ({bytes_fp8/1e6:.2f} MB -> {bytes_fp8/1e6/t_fp8:6.1f} GB/s)")
        print(f"    int4  {t_i4*1e6:8.1f} us  ({bytes_i4/1e6:.2f} MB -> {bytes_i4/1e6/t_i4:6.1f} GB/s)")
        print(f"    decode headroom @35tok/s budget={35e-3:.0f}ms/step; "
              f"bf16={t_bf16*1e3:.3f}ms fp8={t_fp8*1e3:.3f}ms int4={t_i4*1e3:.3f}ms")


def nvme_bench():
    import mmap
    print("\n=== NVMe cold-tier (O_DIRECT, bypassing page cache) ===")
    sz = os.path.getsize(SHARD)
    fd = os.open(SHARD, os.O_RDONLY | os.O_DIRECT)
    CHUNK = 1 << 20
    buf = mmap.mmap(-1, CHUNK)  # page-aligned anonymous buffer
    total = 0
    t0 = time.perf_counter()
    while total < (1 << 30):
        os.preadv(fd, [buf], total)
        total += CHUNK
    seq_bw = (1 << 30) / (time.perf_counter() - t0) / 1e9
    print(f"  sequential read BW (O_DIRECT, 1 GiB): {seq_bw:.2f} GB/s")

    random.seed(0)
    n = 2000
    offsets = [random.randrange(0, sz - 4096) & ~4095 for _ in range(n)]
    t0 = time.perf_counter()
    for off in offsets:
        os.preadv(fd, [buf], off)
    dt = time.perf_counter() - t0
    print(f"  random 4 KiB QD1 (O_DIRECT): {n/dt:6.0f} IOPS  {dt/n*1e6:7.1f} us avg"
          f"  ({n/dt*4096/1e6:.2f} MB/s)")
    os.close(fd)
    buf.close()

    print("\n  per-step cold-miss cost (16 rows, <=16 pages):")
    for name, br in [("bf16", 16 * 160 * 2), ("fp8", 16 * 160 * 1), ("int4", 16 * 80 * 1)]:
        print(f"    {name}: {br} bytes/step (spread across <=16 pages)")


if __name__ == "__main__":
    resident_bench()
    nvme_bench()