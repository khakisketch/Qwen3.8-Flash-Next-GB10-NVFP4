#!/usr/bin/env python3
"""
BF16 accuracy reference harness for Qwen3.8-Flash-Next.

Loads the *full* 360GB checkpoint and generates a small set of reference
outputs to use as the correctness baseline for the quantized variant.

Because the 180B BF16 model exceeds the 130GB unified memory of the GB10, this
uses device_map auto-dispatch with CPU/RAM spilling and a disk offload folder,
so generation is slow. Use a tiny max_new_tokens.

Usage:
  python scripts/accuracy_baseline.py \
      --model-path original_model \
      --prompts-file configs/reference_prompts.txt \
      --output reference_outputs.json
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts", nargs="*", default=["What is 84 * 3 / 2?"])
    ap.add_argument("--output", default="reference_outputs.json")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--offload-folder", default="offload")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)} "
              f"mem {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tok = AutoTokenizer.from_pretrained(
        args.model_path, revision=args.revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=args.offload_folder,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for prompt in args.prompts:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "output": text, "token_ids": out[0].tolist()})
        print(f"[{len(results)}/{len(args.prompts)}] {prompt!r} -> {text!r}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()