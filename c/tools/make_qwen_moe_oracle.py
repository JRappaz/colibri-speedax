#!/usr/bin/env python3
"""Build a tiny random Qwen2-MoE checkpoint and greedy reference.

This is a local correctness fixture for the Colibrì Qwen backend. It uses the
real Transformers Qwen2-MoE module so tensor names and packed expert shapes match
current HF checkpoints, while keeping dimensions small enough for tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_fixture(out: Path, ref: Path, seed: int = 1234, max_new_tokens: int = 8) -> None:
    import torch
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        decoder_sparse_step=1,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    cfg._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
            else:
                p.zero_()

    prompt = [3, 14, 15, 92, 65, 35]
    ids = torch.tensor([prompt])
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        full = generated[0].tolist()
        logits = model(torch.tensor([full]), use_cache=False).logits[0]
    tf_pred = logits.argmax(-1).tolist()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n")
    ref.write_text(
        json.dumps(
            {
                "model": "tiny-random-qwen2-moe",
                "seed": seed,
                "prompt_ids": prompt,
                "full_ids": full,
                "tf_pred": tf_pred,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny Qwen2-MoE oracle checkpoint")
    parser.add_argument("--out", default="qwen_moe_tiny", help="Output HF checkpoint directory")
    parser.add_argument("--ref", default="ref_qwen_moe.json", help="Output JSON reference")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    build_fixture(Path(args.out), Path(args.ref), seed=args.seed, max_new_tokens=args.max_new_tokens)
    print(f"saved: {args.out} and {args.ref}")


if __name__ == "__main__":
    main()
