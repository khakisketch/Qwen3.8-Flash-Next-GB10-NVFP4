#!/usr/bin/env python3
"""
Quantized PLE runtime adapter for Qwen3.8-Flash-Next (transformers 5.16.1).

Replaces the giant Qwen4ExpTextNGramEmbedding (320M-row nn.Embedding, ~102GB BF16)
with a memory-light subclass whose lookup is served from NVMe via PleStore.
The n-gram hashing logic (shift/eos/mix/xor/prime-head hashing) is copied verbatim
from the pinned transformers implementation so results are bit-identical; only the
final `nn.Embedding` lookup is replaced.

Usage:
    import transformers
    model = transformers.Qwen4ExpForConditionalGeneration.from_pretrained(...)
    store = PleStore("converted_dir", device="cuda")
    patch_ngram_embedding(model, store)
    model.generate(...)  # PLE served from NVMe
"""

import torch
import torch.nn as nn

from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextNGramEmbedding,
    _build_layer_multipliers,
    _find_nth_prime_after,
)

from runtime.ple_store import PleEmbeddingProxy, PleStore


class QuantizedNGramEmbedding(Qwen4ExpTextNGramEmbedding):
    """Same public surface as Qwen4ExpTextNGramEmbedding, NVMe-backed lookup."""

    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index, store: PleStore, device):
        nn.Module.__init__(self)  # skip base __init__ (it allocates the giant embedding)
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.register_buffer(
            "layer_multipliers",
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed),
        )
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(self.head_offsets, dtype=torch.long))

        self._store = store
        self.ngram_embedding = PleEmbeddingProxy(store, device)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids: torch.Tensor, past_key_values=None) -> torch.Tensor:
        input_ids = input_ids.long()
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            if (
                not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                and input_ids.shape[1] < self.context_len
            ):
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
        return self.ngram_embedding(ngram_ids.to(self.ngram_embedding.weight.device)).to(ngram_ids.device).flatten(-2)


def patch_ngram_embedding(model, store: PleStore, device="cuda"):
    """Replace every Qwen4ExpTextNGramEmbedding in `model` with the NVMe-backed one.

    `store` must already be constructed and point at the converted checkpoint.
    Returns the number of modules patched.
    """
    cfg = model.config.text_config
    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen4ExpTextNGramEmbedding) and not isinstance(module, QuantizedNGramEmbedding):
            layer_idx = module.layer_idx
            ple_layer_index = module.ple_layer_index
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            new_mod = QuantizedNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx, ple_layer_index, store, device)
            setattr(parent, name.split(".")[-1], new_mod.to(device))
            patched += 1
    return patched