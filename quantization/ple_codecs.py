#!/usr/bin/env python3
"""
PLE (n-gram embedding) codec candidates.

Each codec exposes ``encode(x[, group]) -> dict`` and ``decode(state) -> fp32``
plus ``num_bytes(state)`` for storage accounting. Input ``x`` is a 2D float32
tensor [R, C] representing PLE rows. These are reconstruction-quality codecs
for the embedding *lookup* path (not NVFP4 GEMM formats); final format is chosen
by the measurements in ``ple_codec_bench.py``, not assumed here.
"""

import math

import torch

FP16 = 2
FP32 = 4


def _fp16(x):
    return x.to(torch.float16)


def codec_bf16():
    def encode(x):
        return {"name": "bf16", "w": x.to(torch.bfloat16)}

    def decode(s):
        return s["w"].float()

    def num_bytes(s):
        return s["w"].numel() * 2

    return encode, decode, num_bytes


def codec_fp8_row():
    # row-wise FP8 E4M3 with per-row FP16 scale
    def encode(x):
        amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (amax / 448.0).to(torch.float16)
        q = (x / scale.float()).clamp(-448.0, 448.0)
        w = q.to(torch.float8_e4m3fn)
        return {"name": "fp8_row", "w": w, "scale": scale}

    def decode(s):
        return s["w"].float() * s["scale"].float()

    def num_bytes(s):
        return s["w"].numel() * 1 + s["scale"].numel() * FP16

    return encode, decode, num_bytes


def _codec_int_sym(group=None):
    label = f"sym_int4_g{group}" if group else "sym_int4"

    def encode(x):
        r, c = x.shape
        if group:
            xr = x.view(r, c // group, group)
            amax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (amax / 7.0).to(torch.float16)
            q = (xr / scale.float()).round().clamp(-8, 7).to(torch.int8)
            return {"name": label, "q": q, "scale": scale, "group": group}
        amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (amax / 7.0).to(torch.float16)
        q = (x / scale.float()).round().clamp(-8, 7).to(torch.int8)
        return {"name": label, "q": q, "scale": scale, "group": None}

    def decode(s):
        scale = s["scale"].float()
        q = s["q"].float()
        return (q * scale).reshape(s["q"].shape[0], -1)

    def num_bytes(s):
        return s["q"].numel() * 0.5 + s["scale"].numel() * FP16

    return encode, decode, num_bytes


def _codec_int_affine(group=None, mse=False):
    label = "affine_uint4" + ("_mse" if mse else "") + (f"_g{group}" if group else "")
    group = group or 0  # 0 means whole row

    def _block_quant(xr, dim=-1):
        mn = xr.min(dim=dim, keepdim=True).values
        mx = xr.max(dim=dim, keepdim=True).values
        span = (mx - mn).clamp_min(1e-12)
        scale = span / 15.0
        zp = (mn / scale).round()
        q = (xr / scale - zp).round().clamp(0, 15).to(torch.uint8)
        return q, scale, zp

    def encode(x):
        x = x.float()
        r, c = x.shape
        g = group if group else c
        xr = x.view(r, c // g, g)
        q, scale, zp = _block_quant(xr)
        if mse:
            scale_f = scale.float()
            factors = torch.linspace(0.55, 1.0, 10, device=x.device)
            best_q = q
            best_scale = scale
            best_zp = zp
            best_err = None
            for f in factors:
                s = scale_f * f
                zpf = (xr.min(dim=-1, keepdim=True).values / s).round()
                qf = (xr / s - zpf).round().clamp(0, 15)
                rec = (qf - zpf) * s
                err = ((xr - rec) ** 2).sum(dim=-1)
                if best_err is None:
                    best_err = err
                update = (err < best_err).unsqueeze(-1)
                best_q = torch.where(update, qf, best_q)
                best_scale = torch.where(update, s, best_scale)
                best_zp = torch.where(update, zpf, best_zp)
                best_err = torch.minimum(best_err, err)
            q = best_q
            scale = best_scale
            zp = best_zp
        scale = scale.to(torch.float16)
        zp = zp.to(torch.int16)
        q = q.to(torch.uint8)
        return {"name": label, "q": q, "scale": scale, "zp": zp, "group": g}

    def decode(s):
        q = s["q"].float()
        scale = s["scale"].float()
        zp = s["zp"].float()
        return ((q + zp) * scale).reshape(s["q"].shape[0], -1)

    def num_bytes(s):
        return s["q"].numel() * 0.5 + s["scale"].numel() * FP16 + s["zp"].numel() * 2

    return encode, decode, num_bytes


CODECS = {
    "bf16": codec_bf16,
    "fp8_row": codec_fp8_row,
    "sym_int4": lambda: _codec_int_sym(None),
    "sym_int4_g32": lambda: _codec_int_sym(32),
    "affine_uint4": lambda: _codec_int_affine(None, mse=False),
    "affine_uint4_mse": lambda: _codec_int_affine(None, mse=True),
    "affine_uint4_g32": lambda: _codec_int_affine(32, mse=False),
}


def build_all():
    return {name: factory() for name, factory in CODECS.items()}


def self_test():
    torch.manual_seed(0)
    x = torch.randn(512, 160) * 0.02 + torch.randn(512, 1) * 0.05
    for name, (enc, dec, nbytes) in build_all().items():
        s = enc(x)
        xq = dec(s)
        err = (x - xq).abs().max().item()
        rel = (x - xq).norm() / x.norm()
        assert err < 10 and rel < 0.2, (name, err, rel)
        print(f"  {name:18s} bytes/row_elems={nbytes(s)/x.numel():.3f} "
              f"max_abs={err:.4g} rel_l2={rel:.4g}")


if __name__ == "__main__":
    self_test()
    print("SELF-TEST PASSED")