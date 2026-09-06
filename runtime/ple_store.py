#!/usr/bin/env python3
"""
NVMe-backed PLE (n-gram embedding) lookup engine for the quantized Qwen3.8-Flash-Next.

The 320M-row n-gram table is stored shard-streamed as affine UINT4 g32:
  weight_packed       uint8 [R, 5, 16]
  weight_scales       fp16  [R, 5]
  weight_zero_points  int16 [R, 5]
Rows are laid out contiguously across shards: row = shard * rows_per_shard + offset.
Only the quantized shard tensors live here; everything else stays BF16 in the model.

This module serves row lookups from NVMe (safetensors row slices) with a bounded
LRU hot-row cache, and dequantizes to BF16. It is self-contained (no dependency on
quantization/*) so the runtime stays independent of the offline converter.
"""

import json
import os
import re
from collections import OrderedDict

import torch
from safetensors import safe_open

PLE_CODE = "affine_uint4_g32"
NGRAM_SHARD_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.")


def unpack_and_dequant_block(packed, scales, zero_points, group=32):
    """packed [n,5,16] uint8 + scales/zp [n,5] -> [n, 5*group] fp32 (affine g32)."""
    lo = packed.to(torch.int32) & 0x0F
    hi = (packed.to(torch.int32) >> 4) & 0x0F
    codes = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1, group).float()
    scale = scales.float().unsqueeze(-1)
    zp = zero_points.float().unsqueeze(-1)
    return ((codes - zp) * scale).reshape(packed.shape[0], -1)


class PleStore:
    """Row -> BF16 embedding lookup from quantized PLE shard files + LRU cache."""

    def __init__(self, converted_dir: str, device="cuda", cache_rows: int = 2_000_000):
        self.converted_dir = converted_dir
        self.device = device
        self.cache_rows = cache_rows
        self._handles = {}
        self._shards = {}
        self._load_manifest()
        # LRU: row_id -> dequantized BF16 [160] on self.device
        self._cache = OrderedDict()

    # ---- manifest / file setup ----
    def _load_manifest(self):
        path = os.path.join(self.converted_dir, "ple_int4_manifest.json")
        with open(path) as f:
            manifest = json.load(f)
        self.rows_per_shard = None
        for name, meta in manifest.items():
            if meta.get("codec") != PLE_CODE or not meta.get("orig_shape"):
                continue
            m = NGRAM_SHARD_RE.search(name)
            if not m:
                continue
            shard_idx = int(m.group(1))
            r = meta["orig_shape"][0]
            if self.rows_per_shard is None:
                self.rows_per_shard = r
            elif self.rows_per_shard != r:
                raise ValueError("PLE shards have inconsistent row counts")
            self._shards[shard_idx] = {
                "file": os.path.join(self.converted_dir, meta["shard_file"]),
                "base": name[:-len(".weight")],
                "rows": r,
            }
        if not self._shards:
            raise ValueError("no affine_uint4_g32 ngram shards found in manifest")
        self.n_shards = len(self._shards)

    def _file(self, path):
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    # ---- lookup ----
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, S, H] long -> [B, S, H, 160] BF16 on ids.device."""
        shape = ids.shape
        flat = ids.reshape(-1).tolist()
        n = len(flat)
        if n == 0:
            return torch.empty(*shape, 160, dtype=torch.bfloat16, device=self.device)

        out = torch.empty(n, 160, dtype=torch.bfloat16, device=self.device)
        pos_of = {}
        for i, r in enumerate(flat):
            pos_of.setdefault(r, []).append(i)

        hits = [r for r in pos_of if r in self._cache]
        miss = [r for r in pos_of if r not in self._cache]
        for r in hits:
            v = self._cache[r]
            for i in pos_of[r]:
                out[i] = v

        # group misses by shard (rows unique => offsets unique within a shard)
        from collections import defaultdict
        by_shard = defaultdict(list)  # shard -> [(i, offset)]
        for r in miss:
            by_shard[r // self.rows_per_shard].append((pos_of[r][0], r % self.rows_per_shard))

        for shard, items in by_shard.items():
            rows = self._fetch_rows(shard, items)
            for (i, offset), row in zip(items, rows):
                row_id = shard * self.rows_per_shard + offset
                self._set_cache(row_id, row)
                for pos in pos_of[row_id]:
                    out[pos] = row
        return out.view(*shape, 160)

    # ---- NVMe fetch + dequant ----
    def _fetch_rows(self, shard_idx, items):
        """items: [(i, offset), ...] -> dequantized BF16 rows [len(items), 160]."""
        info = self._shards[shard_idx]
        offsets = sorted(o for _, o in items)
        lo, hi = offsets[0], offsets[-1] + 1
        span = hi - lo
        f = self._file(info["file"])
        if span > max(64, 4 * len(items)):
            # sparse access: read row-by-row to bound read amplification
            packs, scales, zps = [], [], []
            for off in offsets:
                packs.append(f.get_slice(info["base"] + ".weight_packed")[off : off + 1])
                scales.append(f.get_slice(info["base"] + ".weight_scales")[off : off + 1])
                zps.append(f.get_slice(info["base"] + ".weight_zero_points")[off : off + 1])
            packed = torch.cat(packs, dim=0)
            scale = torch.cat(scales, dim=0)
            zp = torch.cat(zps, dim=0)
        else:
            packed = f.get_slice(info["base"] + ".weight_packed")[lo:hi]
            scale = f.get_slice(info["base"] + ".weight_scales")[lo:hi]
            zp = f.get_slice(info["base"] + ".weight_zero_points")[lo:hi]
        deq = unpack_and_dequant_block(packed, scale, zp)
        # map back to requested order
        off_index = {o: k for k, o in enumerate(offsets)}
        order = torch.tensor([off_index[o] for _, o in items], dtype=torch.long)
        return deq[order].to(torch.bfloat16).to(self.device)

    # ---- LRU ----
    def _set_cache(self, row, vec):
        self._cache[row] = vec
        while len(self._cache) > self.cache_rows:
            self._cache.popitem(last=False)

    def cache_size(self) -> int:
        return len(self._cache)

    def close(self):
        self._handles.clear()


class PleEmbeddingProxy(torch.nn.Module):
    """Drop-in replacement for nn.Embedding; routes lookup through a PleStore.

    Exposes `weight.device` (as the original nn.Embedding does) so the parent
    Qwen4ExpTextNGramEmbedding.forward can do its device-placement dance.
    """

    def __init__(self, store: PleStore, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
        self._store = store

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._store.lookup(ids)