#!/usr/bin/env python3
"""
Phase 1 gate: does GB10 (sm_121) run FP4 (E2M1) block-scaled GEMM natively?

This is the make-or-break experiment that decides whether the KSI-FP4 experts plan
is viable on a single DGX Spark. It answers three questions:

  1. Does `tl.dot_scaled(..., "e2m1", ...)` compile and run on sm_121?
  2. Is it CORRECT (matches an fp32 dequant-reference matmul)?
  3. Is it NATIVE (FP4 tensor-core MMA) or silently emulated (bf16 upcast)?

Run:
    .venv/bin/python kernels/fp4_microproof.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl

from quantization.experts_nvfp4 import E2M1_VALUES, cast_fp4


# ---- KSI E2M1 packing (low nibble = lower K index, matches hardware convention) ----
def pack_e2m1(flat_codes: torch.Tensor, K: int):
    """flat_codes [.., K] uint8 codes 0..15 -> packed [.., K/2] uint8."""
    lo = flat_codes[..., 0::2]
    hi = flat_codes[..., 1::2]
    return ((hi << 4) | lo).to(torch.uint8)


def quantize_k(x: torch.Tensor, block: int = 16):
    """Quantize x [M,K] (or [N,K]) to packed uint8 [.., K/2] + fp8 e4m3 scale [.., K/16].

    NVFP4: per-16-block scale in FP8 E4M3. global_scale folded for simplicity here:
    scale = block_amax / 6.0 then cast to fp8e4m3 (with a global fp32 factor kept aside).
    """
    x = x.float()
    M, K = x.shape
    xb = x.view(M, K // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)  # [M, K/16, 1]
    scale = (amax / 6.0).clamp_min(1e-9)        # [M, K/16, 1]
    scaled = xb / scale
    codes = cast_fp4(scaled).reshape(M, K)
    packed = pack_e2m1(codes, K)
    scale_fp8 = scale.reshape(M, K // block).to(torch.float8_e4m3fn)
    return packed, scale_fp8


def dequant_reference(packed, scale_fp8, dtype=torch.float32):
    M, Khalf = packed.shape
    K = Khalf * 2
    lo = packed.to(torch.int32) & 0x0F
    hi = (packed.to(torch.int32) >> 4) & 0x0F
    codes = torch.stack((lo, hi), dim=-1).reshape(M, K)
    vals = E2M1_VALUES.to(codes.device)[codes.long()]  # [M, K]
    scale = scale_fp8.float()  # [M, K/16]
    scale = scale.repeat_interleave(16, dim=1)  # [M, K]
    return vals * scale


@triton.jit
def fp4_gemm_kernel(
    a_ptr, a_s_ptr, b_ptr, b_s_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_K // 2)  # packed byte columns along K
    offs_ks = tl.arange(0, BLOCK_K // 16)

    # A: [M, K/2] row-major packed; B: [N, K/2] row-major packed (weight stored [N,K])
    a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + offs_c[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * (K // 2) + offs_c[None, :]
    a_s_ptrs = a_s_ptr + offs_m[:, None] * (K // 16) + offs_ks[None, :]
    b_s_ptrs = b_s_ptr + offs_n[:, None] * (K // 16) + offs_ks[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        acc = tl.dot_scaled(a, a_s, "e2m1", b.T, b_s, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        b_ptrs += BLOCK_K // 2
        a_s_ptrs += BLOCK_K // 16
        b_s_ptrs += BLOCK_K // 16

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16))


def run_gemm(a_packed, a_scale, b_packed, b_scale, BLOCK_M, BLOCK_N, BLOCK_K):
    M = a_packed.shape[0]
    N = b_packed.shape[0]
    K = a_packed.shape[1] * 2
    c = torch.empty((M, N), dtype=torch.float16, device=a_packed.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    fp4_gemm_kernel[grid](
        a_packed, a_scale, b_packed, b_scale, c,
        M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"device={dev} cap={torch.cuda.get_device_capability(0)} "
          f"triton={triton.__version__} torch={torch.__version__}")

    M, N, K = 128, 256, 2560  # K=2560 matches expert reduction dim
    a_ref = torch.randn(M, K, device=dev) * 0.05
    b_ref = torch.randn(N, K, device=dev) * 0.05  # weight [N, K]

    a_packed, a_scale = quantize_k(a_ref.cpu())
    b_packed, b_scale = quantize_k(b_ref.cpu())
    a_packed, a_scale = a_packed.to(dev), a_scale.to(dev)
    b_packed, b_scale = b_packed.to(dev), b_scale.to(dev)

    ref = dequant_reference(a_packed, a_scale) @ dequant_reference(b_packed, b_scale).T
    ref = ref  # [M, N] fp32

    for (bm, bn, bk) in [(64, 64, 128), (128, 128, 128), (128, 256, 128)]:
        try:
            out = run_gemm(a_packed, a_scale, b_packed, b_scale, bm, bn, bk)
        except Exception as e:
            print(f"[BLOCK {bm}x{bn}x{bk}] FAILED: {type(e).__name__}: {e}")
            continue
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"[BLOCK {bm}x{bn}x{bk}] rel_l2={rel:.5f}")

    # PTX inspection for native FP4 (tcgen05) vs emulation (bf16 upcast + mma)
    try:
        cache = fp4_gemm_kernel.cache
        key = list(cache[dev].keys())[0]
        ptx = cache[dev][key].asm.get("ptx", "")
        has_tcgen05 = "tcgen05" in ptx
        has_f8f6f4 = ("f8f6f4" in ptx) or ("e2m1" in ptx) or ("f4" in ptx)
        has_bf16 = "bf16" in ptx or "m16n8k16" in ptx
        print("\n=== PTX analysis ===")
        print(f"tcgen05 present: {has_tcgen05}")
        print(f"fp4 marker present: {has_f8f6f4}")
        print(f"bf16 path present: {has_bf16}")
        # print first lines mentioning mma / tcgen05
        for line in ptx.splitlines():
            if ("tcgen05" in line or "mma" in line or "f8f6f4" in line or "e2m1" in line):
                print("  ", line.strip())
    except Exception as e:
        print(f"PTX dump failed: {e}")


if __name__ == "__main__":
    main()