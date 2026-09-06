#!/usr/bin/env python3
"""
Row-wise symmetric INT4 quantization for the PLE (n-gram embedding) table.

Format (self-describing, runtime-friendly):
  - packed : uint8 tensor of shape [R, C/2], two signed INT4 values per byte,
             low nibble = even column index, high nibble = odd column index.
  - scales : per-row scale, dtype configurable (default float16), shape [R, 1].
  - orig_shape : original [R, C] shape (for reconstruction / runtime).

Run `python -m quantization.ple_int4 --self-test` to verify round-trip error
on synthetic data without downloading anything.
"""

import argparse
import math

import torch

INT4_MIN = -8
INT4_MAX = 7


def pack_int4(signed: torch.Tensor) -> torch.Tensor:
    """Pack [..., N] signed ints (N even) into [..., N/2] uint8 nibbles."""
    assert signed.dtype == torch.int8
    shape = signed.shape
    if shape[-1] % 2 != 0:
        raise ValueError("last dim must be even for nibble packing")
    a = signed[..., 0::2].to(torch.int32) & 0xF
    b = signed[..., 1::2].to(torch.int32) & 0xF
    return (b << 4 | a).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4 -> [..., N] int8 in [-8, 7]."""
    packed = packed.to(torch.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF

    def sign_extend(v):
        return torch.where(v >= 8, v - 16, v)

    lo = sign_extend(lo)
    hi = sign_extend(hi)
    out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.int8)


def quantize_rowwise(x: torch.Tensor, scale_dtype=torch.float16) -> dict:
    """x: [R, C] (any float dtype) -> {packed, scales, orig_shape}."""
    x = x.float()
    r, c = x.shape
    if c % 2 != 0:
        raise ValueError(f"column dim {c} must be even for nibble packing")
    amax = x.abs().amax(dim=1, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    scales = (amax / INT4_MAX).clamp_min(eps)
    q = (x / scales).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    return {
        "packed": pack_int4(q),
        "scales": scales.to(scale_dtype).contiguous(),
        "orig_shape": torch.tensor([r, c], dtype=torch.int64),
        "scale_dtype": str(scale_dtype),
    }


def dequantize_rowwise(d: dict) -> torch.Tensor:
    q = unpack_int4(d["packed"]).float()
    scales = d["scales"].float()
    return (q * scales).reshape(tuple(d["orig_shape"].tolist()))


def pack_uint4(q: torch.Tensor) -> torch.Tensor:
    """Pack unsigned ints in [0,15] along last dim (2 per byte, low nibble first)."""
    assert q.dtype in (torch.uint8, torch.int32, torch.int16)
    if q.shape[-1] % 2 != 0:
        raise ValueError("last dim must be even for nibble packing")
    q = q.to(torch.int32) & 0xF
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return ((hi << 4) | lo).to(torch.uint8)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_uint4 -> [..., N] unsigned ints in [0,15]."""
    packed = packed.to(torch.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.uint8)


def quantize_affine_group(x: torch.Tensor, group: int = 32,
                          scale_dtype=torch.float16) -> dict:
    """Group-wise affine UINT4 quantization (production PLE codec).

    x: [R, C] float. Each contiguous group of `group` columns shares a scale
    (fp16) and a zero_point (int16). Reconstruction: (q - zero_point) * scale.
    """
    x = x.float()
    r, c = x.shape
    if c % group != 0:
        raise ValueError(f"column dim {c} not divisible by group {group}")
    g = c // group
    xr = x.view(r, g, group)
    mn = xr.min(dim=-1, keepdim=True).values
    mx = xr.max(dim=-1, keepdim=True).values
    span = (mx - mn).clamp_min(1e-12)
    scale = span / 15.0
    zero_point = (-mn / scale).round()
    q = (xr / scale + zero_point).round().clamp(0, 15).to(torch.uint8)
    return {
        "packed": pack_uint4(q),
        "scales": scale.to(scale_dtype).reshape(r, g),
        "zero_points": zero_point.to(torch.int16).reshape(r, g),
        "orig_shape": [r, c],
        "group": group,
        "scale_dtype": str(scale_dtype),
    }


def dequantize_affine_group(d: dict) -> torch.Tensor:
    q = unpack_uint4(d["packed"]).float()          # [R, G, group]
    scale = d["scales"].float().unsqueeze(-1)      # [R, G, 1]
    zp = d["zero_points"].float().unsqueeze(-1)     # [R, G, 1]
    out = (q - zp) * scale
    return out.reshape(d["orig_shape"])


def _snr_dB(ref: torch.Tensor, approx: torch.Tensor) -> float:
    ref = ref.float().flatten()
    approx = approx.float().flatten()
    num = (ref ** 2).mean()
    err = (ref - approx) ** 2
    if num == 0 or err.sum() == 0:
        return float("inf")
    return 10.0 * math.log10(num.item() / err.mean().item())


def _l2_rel(ref: torch.Tensor, approx: torch.Tensor) -> float:
    ref = ref.float()
    approx = approx.float()
    denom = ref.norm(2)
    if denom == 0:
        return 0.0
    return ((ref - approx).norm(2) / denom).item()


def _metrics(ref: torch.Tensor, approx: torch.Tensor) -> dict:
    abs_err = (ref.float() - approx.float()).abs()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "relative_l2": _l2_rel(ref, approx),
        "snr_dB": _snr_dB(ref, approx),
    }


def self_test():
    torch.manual_seed(0)
    for (r, c) in [(4096, 160), (1000, 64), (7, 160)]:
        x = torch.randn(r, c) * 0.02 + torch.randn(r, 1) * 0.05
        d = quantize_rowwise(x)
        xq = dequantize_rowwise(d)
        m = _metrics(x, xq)
        print(f"shape [{r},{c}] sym_int4 rel_l2={m['relative_l2']:.4g} snr={m['snr_dB']:.1f}dB")
        # affine group
        if c % 32 == 0:
            ad = quantize_affine_group(x, group=32)
            axq = dequantize_affine_group(ad)
            am = _metrics(x, axq)
            print(f"shape [{r},{c}] affine_g32 rel_l2={am['relative_l2']:.4g} snr={am['snr_dB']:.1f}dB")
            assert am["relative_l2"] < m["relative_l2"]
    # unsigned pack round-trip
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