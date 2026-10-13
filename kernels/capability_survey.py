#!/usr/bin/env python3
"""
Capability survey on sm_121: confirm the 4-bit weight-only GEMM path (0.5 bytes/param).
INT4 (AWQ/GPTQ-style) weight-only is memory-equivalent to FP4 and works on every arch,
so it is the fallback experts path if native FP4 tensor cores are unreachable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@triton.jit
def int4_w_gemm_kernel(a_ptr, w_ptr, s_ptr, z_ptr, c_ptr,
                       M, N, K,
                       GROUP: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w_ptrs = w_ptr + (offs_n[:, None] * (K // 2) + (offs_k // 2)[None, :])
    s_ptrs = s_ptr + (offs_n[:, None] * (K // GROUP) + (offs_k // GROUP)[None, :])
    z_ptrs = z_ptr + (offs_n[:, None] * (K // GROUP) + (offs_k // GROUP)[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        a = tl.load(a_ptrs).to(tl.float32)                 # [BLOCK_M, BLOCK_K]
        w = tl.load(w_ptrs)                                # [BLOCK_N, BLOCK_K//2] uint8
        lo = (w & 0xF).to(tl.float32) - 8.0
        hi = ((w >> 4) & 0xF).to(tl.float32) - 8.0
        wq = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))  # [BLOCK_N, BLOCK_K]

        s = tl.load(s_ptrs).to(tl.float32)                 # [BLOCK_N, BLOCK_K//GROUP]
        z = tl.load(z_ptrs).to(tl.float32)
        s = tl.reshape(tl.broadcast_to(tl.reshape(s, (BLOCK_N, 1, BLOCK_K // GROUP)),
                                       (BLOCK_N, GROUP, BLOCK_K // GROUP)),
                       (BLOCK_N, BLOCK_K))
        z = tl.reshape(tl.broadcast_to(tl.reshape(z, (BLOCK_N, 1, BLOCK_K // GROUP)),
                                       (BLOCK_N, GROUP, BLOCK_K // GROUP)),
                       (BLOCK_N, BLOCK_K))

        wd = wq * s + z
        acc = tl.dot(a, tl.trans(wd), acc)

        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K // 2
        s_ptrs += BLOCK_K // GROUP
        z_ptrs += BLOCK_K // GROUP

    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(c_ptrs, acc.to(tl.float16))


def int4_w_gemm(a, w_packed, scale, zero, GROUP, BM, BN, BK):
    M, K = a.shape
    N = w_packed.shape[0]
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    int4_w_gemm_kernel[grid](a, w_packed, scale, zero, c, M, N, K,
                             GROUP=GROUP, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def quant_int4_group(w, group=128):
    """Weight [N,K] -> packed uint8 [N,K/2] + fp16 scale/zero [N,K//group]."""
    N, K = w.shape
    w = w.float()
    g = K // group
    xb = w.view(N, g, group)
    mn = xb.min(-1, keepdim=True).values
    mx = xb.max(-1, keepdim=True).values
    span = (mx - mn).clamp_min(1e-6)
    scale = (span / 15.0).to(torch.float16)
    zero = (torch.round(-mn / scale) - 7.0).clamp(-8, 7).to(torch.float16)
    q = torch.round(w / scale[..., None] + zero[..., None]).clamp(-8, 7).to(torch.int8)
    lo = (q[..., 0::2] & 0xF).to(torch.uint8)
    hi = (q[..., 1::2] & 0xF).to(torch.uint8)
    return ((hi << 4) | lo), scale.reshape(N, g), zero.reshape(N, g)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"cap={torch.cuda.get_device_capability(0)} triton={triton.__version__}")

    for (M, N, K) in [(128, 256, 2560), (4, 1280, 2560), (1, 640, 2560)]:
        a = (torch.randn(M, K, device=dev) * 0.05).to(torch.bfloat16)
        w = torch.randn(N, K, device=dev) * 0.05
        ref = a.float() @ w.float().T
        w_p, s, z = quant_int4_group(w.cpu())
        try:
            out = int4_w_gemm(a, w_p.to(dev), s.to(dev), z.to(dev), GROUP=128, BM=64, BN=64, BK=128)
            print(f"[int4] M={M} N={N} K={K}: rel_l2={rel_l2(out, ref):.5f}")
        except Exception as e:
            print(f"[int4] M={M} N={N} K={K} FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()