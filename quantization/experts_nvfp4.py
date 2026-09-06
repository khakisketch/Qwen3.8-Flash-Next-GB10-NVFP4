#!/usr/bin/env python3
"""
Block-wise NVFP4 (E2M1FN) quantization for Qwen3.8-Flash-Next MoE experts.

The expert weights are fused 3D tensors [E, K, N] (E experts, K reduction dim,
N output), stored WITHOUT a ``.weight`` suffix on disk. Each block of ``block_k``
rows along K shares one fp16 scale; within a block each element is rounded to the
IEEE-style 4-bit E2M1 encoding used by NVIDIA MXFP4/NVFP4.

E2M1FN representable magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}; the encoding and
rounding rule below are bit-identical to NVIDIA ModelOpt's
``NVFP4QTensor._cast_fp4`` (see nvfp4_tensor.py in the installed modelopt package),
so ModelOpt is the correctness oracle.

Format (self-describing, mirrors quantization/ple_int4.py):
  - weight_packed : uint8 [E, K/2, N], two 4-bit codes per byte,
                    low nibble = even K index, high nibble = odd K index.
  - weight_scales : fp16 [E, K/block_k, N], per-block scale = block amax / E2M1_MAX.
  - orig_shape    : [E, K, N]

Run `python -m quantization.experts_nvfp4 --self-test` to verify on synthetic data.
"""

import argparse
import math

import torch

from quantization.ple_int4 import pack_uint4, unpack_uint4

E2M1_MAX = 6.0

# index 0..7  -> positive {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# index 8..15 -> negative counterparts. Sign is bit 3; low 3 bits are the ordinal.
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_ODD_BOUNDS = torch.tensor([0.75, 1.75, 3.5])


def cast_fp4(x: torch.Tensor) -> torch.Tensor:
    """Cast scaled float32 (|x| already ~<= E2M1_MAX) to E2M1 4-bit codes 0..15.

    Bit-identical to ModelOpt NVFP4QTensor._cast_fp4.
    """
    sign_bit = (x < 0).to(torch.uint8)
    xa = x.abs()
    ord_ = torch.searchsorted(E2M1_BOUNDS, xa, out_int32=True).to(torch.uint8)
    eq = torch.any(xa.unsqueeze(-1) == _ODD_BOUNDS, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord_ + eq


def dequant_codes(codes: torch.Tensor) -> torch.Tensor:
    """Map 4-bit codes back to fp32 values using the E2M1 table."""
    return E2M1_VALUES[codes.long()]


def quantize_experts_nvfp4(x: torch.Tensor, block_k: int = 16,
                           scale_dtype=torch.float16) -> dict:
    """x: [E, K, N] float -> {weight_packed, weight_scales, orig_shape, ...}."""
    x = x.float()
    E, K, N = x.shape
    if K % block_k != 0:
        raise ValueError(f"K dim {K} not divisible by block_k {block_k}")
    g = K // block_k
    xb = x.view(E, g, block_k, N)
    amax = xb.abs().amax(dim=2, keepdim=True)  # [E, g, 1, N]
    scale = (amax / E2M1_MAX).clamp_min(1e-12)
    scaled = xb / scale                       # [E, g, block_k, N]
    q = cast_fp4(scaled).view(E, K, N)
    # pack along K (dim 1): [E, N, K] -> [E, N, K/2] -> back to [E, K/2, N]
    packed = pack_uint4(q.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
    return {
        "weight_packed": packed,
        "weight_scales": scale.view(E, g, N).to(scale_dtype).contiguous(),
        "orig_shape": [E, K, N],
        "block_k": block_k,
        "scale_dtype": str(scale_dtype),
    }


def dequantize_experts_nvfp4(d: dict) -> torch.Tensor:
    # unpack along K (dim 1): [E, K/2, N] -> [E, N, K/2] -> [E, N, K] -> [E, K, N]
    codes = unpack_uint4(d["weight_packed"].permute(0, 2, 1)).permute(0, 2, 1).long()
    E, K, N = d["orig_shape"]
    block_k = d["block_k"]
    g = K // block_k
    codes = codes.view(E, g, block_k, N)
    scale = d["weight_scales"].float().unsqueeze(2)      # [E, g, 1, N]
    vals = E2M1_VALUES[codes]                            # [E, g, block_k, N]
    return (vals * scale).reshape(E, K, N)


def _rel_l2(ref: torch.Tensor, approx: torch.Tensor) -> float:
    ref = ref.float()
    approx = approx.float()
    denom = ref.norm(2)
    if denom == 0:
        return 0.0
    return ((ref - approx).norm(2) / denom).item()


def self_test():
    torch.manual_seed(0)
    for (e, k, n) in [(8, 128, 64), (4, 160, 32), (2, 96, 48)]:
        x = torch.randn(e, k, n) * 0.02 + torch.randn(e, 1, 1) * 0.05
        d = quantize_experts_nvfp4(x, block_k=16)
        xq = dequantize_experts_nvfp4(d)
        rel = _rel_l2(x, xq)
        print(f"shape [{e},{k},{n}] nvfp4_e2m1fn rel_l2={rel:.4g}")
        # packed/scales shapes and reconstruction dtype sanity
        assert tuple(d["weight_packed"].shape) == (e, k // 2, n)
        assert tuple(d["weight_scales"].shape) == (e, k // 16, n)
    # bit-identical cast vs ModelOpt oracle table
    x = torch.tensor([0.1, 0.3, 0.8, 1.2, 1.8, 2.7, 4.0, 5.9,
                     -0.1, -0.3, -0.8, -1.2, -1.8, -2.7, -4.0, -5.9])
    codes = cast_fp4(x)
    vals = E2M1_VALUES[codes.long()]
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0,
                             -0.0, -0.5, -1.0, -1.0, -2.0, -3.0, -4.0, -6.0])
    assert torch.equal(vals, expected), (vals, expected)
    # unsigned nibble round-trip
    u = torch.randint(0, 16, (7, 160), dtype=torch.int32)
    assert torch.equal(unpack_uint4(pack_uint4(u)).to(torch.int32), u)
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        self_test()


if __name__ == "__main__":
    main()