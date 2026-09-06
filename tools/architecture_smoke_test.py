#!/usr/bin/env python3
"""
Scaled-down forward/generation smoke test for the qwen4_exp reference model.

Verifies the Hugging Face transformers 5.16 implementation of
Qwen4ExpForConditionalGeneration actually runs on this GB10 (sm_121, CUDA 13)
before we download the full 360GB checkpoint. Uses a shrunken config so the
model fits in a few GB of unified memory.

This is a correctness-of-code test, not an accuracy test.
"""

import argparse

import torch
from transformers import (
    Qwen4ExpConfig,
    Qwen4ExpForConditionalGeneration,
)
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def build_small_config(real_cfg: Qwen4ExpConfig, layers: int, experts: int):
    tc = real_cfg.text_config
    small_text = Qwen4ExpTextConfig(
        **{
            k: v
            for k, v in tc.to_dict().items()
            if k not in ("layer_types", "ngram_vocab_size_base", "num_experts")
        }
    )
    assert layers <= len(tc.layer_types)
    small_text.layer_types = tc.layer_types[:layers]
    small_text.num_hidden_layers = layers
    small_text.num_experts = experts
    small_text.ngram_vocab_size_base = 4096
    small_text.ple_layer_ids = [i for i in small_text.ple_layer_ids if i < layers]

    vc = real_cfg.vision_config.to_dict()
    vc["depth"] = 1
    small = Qwen4ExpConfig(
        text_config=small_text,
        vision_config=vc,
        **{k: v for k, v in real_cfg.to_dict().items() if k in ("architectures",)},
    )
    return small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--config", default="/tmp/opencode/qwen-manifest/config.json")
    ap.add_argument("--ref-config", default="Qwen/Qwen3.8-Flash-Next")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    real_cfg = Qwen4ExpConfig.from_pretrained(args.config)
    cfg = build_small_config(real_cfg, args.layers, args.experts)

    model = Qwen4ExpForConditionalGeneration(cfg).to(torch.bfloat16)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"small model: {n/1e6:.1f}M params")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    input_ids = torch.tensor([[151644, 8948, 198, 2610, 525, 264, 10950, 17847]], device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits
    print(f"forward OK: logits {tuple(logits.shape)}")
    token = logits[:, -1].argmax(dim=-1).item()
    print(f"argmax next token: {token}")

    # generation path
    gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(f"generation OK: {gen.shape} -> {gen[0].tolist()}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()