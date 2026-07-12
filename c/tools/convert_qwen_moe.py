#!/usr/bin/env python3
"""Convert Qwen1.5-MoE HuggingFace checkpoints to colibri's initial layout.

Milestone 0 keeps the storage contract simple:
  * dense tensors are copied unchanged;
  * routed expert tensors are row-wise quantized and remain keyed by
    (layer_id, expert_id) in their original HF tensor names;
  * config.json is copied with a small colibri_qwen_moe marker.

The runtime backend will stream routed expert tensors by layer/expert using
these names:
  model.layers.<layer>.mlp.experts.<expert>.(gate_proj|up_proj|down_proj).weight

Recent Transformers Qwen2-MoE checkpoints store routed experts packed as:
  model.layers.<layer>.mlp.experts.gate_up_proj  [E, 2I, H]
  model.layers.<layer>.mlp.experts.down_proj     [E, H, I]
The converter unpacks those tensors into the per-expert names above.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

EXPERT_KEY_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
PACKED_EXPERT_KEY_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)$"
)


def parse_expert_key(name: str) -> tuple[int, int, str] | None:
    match = EXPERT_KEY_RE.fullmatch(name)
    if not match:
        return None
    return int(match.group("layer")), int(match.group("expert")), match.group("proj")


def is_expert_weight(name: str) -> bool:
    return parse_expert_key(name) is not None


def parse_packed_expert_key(name: str) -> tuple[int, str] | None:
    match = PACKED_EXPERT_KEY_RE.fullmatch(name)
    if not match:
        return None
    return int(match.group("layer")), match.group("proj")


def expert_weight_name(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def quantize_row(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if bits < 2 or bits > 8:
        raise ValueError("--ebits must be in [2, 8]")
    qmax = (1 << (bits - 1)) - 1
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / float(qmax)
    q = (w_f32 / scales).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scales.squeeze(1).float()


def copy_config(src: Path, out: Path) -> None:
    cfg = json.loads((src / "config.json").read_text())
    cfg.setdefault("colibri_backend", "qwen_moe")
    cfg.setdefault("colibri_expert_key", "layer_expert")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def resolve_source(repo: str | None, model: str | None) -> Path:
    if repo:
        from huggingface_hub import snapshot_download

        print(f"Downloading {repo}...")
        path = snapshot_download(repo, local_files_only=True, max_workers=4)
        if not any(Path(path).glob("*.safetensors")):
            path = snapshot_download(repo, max_workers=4)
        return Path(path)
    assert model is not None
    return Path(model)


def main() -> None:
    try:
        from safetensors.torch import load_file, save_file
        import torch
    except ImportError as exc:
        sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors")

    parser = argparse.ArgumentParser(description="Convert Qwen1.5-MoE HF checkpoint -> colibri Qwen layout")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID")
    src.add_argument("--model", help="Local HF checkpoint directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ebits", type=int, default=4, help="Expert quant bits, 2..8")
    args = parser.parse_args()

    source = resolve_source(args.repo, args.model)
    if not source.is_dir():
        sys.exit(f"Model directory not found: {source}")
    if not (source / "config.json").is_file():
        sys.exit(f"config.json missing in {source}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copy_config(source, out)
    for sidecar in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (source / sidecar).exists():
            shutil.copy2(source / sidecar, out / sidecar)

    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors found in {source}")

    expert_tensors = 0
    expert_objects: set[tuple[int, int]] = set()
    total_expert_f32 = 0
    total_expert_q = 0
    for index, shard in enumerate(shards, start=1):
        print(f"[{index}/{len(shards)}] {shard.name}...", end=" ", flush=True)
        tensors = load_file(str(shard))
        converted = {}
        for name, tensor in tensors.items():
            parsed = parse_expert_key(name)
            if parsed is not None:
                layer, expert, _ = parsed
                q, scales = quantize_row(tensor, args.ebits)
                converted[name] = q
                converted[name + ".qs"] = scales
                expert_tensors += 1
                expert_objects.add((layer, expert))
                total_expert_f32 += tensor.numel() * tensor.element_size()
                total_expert_q += q.numel() + scales.numel() * 4
                continue

            packed = parse_packed_expert_key(name)
            if packed is None:
                converted[name] = tensor
                continue

            layer, packed_proj = packed
            if tensor.dim() != 3:
                sys.exit(f"Expected packed expert tensor {name} to be rank 3, got shape {tuple(tensor.shape)}")
            if packed_proj == "gate_up_proj":
                if tensor.shape[1] % 2:
                    sys.exit(f"Expected even gate/up dimension for {name}, got shape {tuple(tensor.shape)}")
                gate, up = torch.chunk(tensor, 2, dim=1)
                pieces = (("gate_proj", gate), ("up_proj", up))
            else:
                pieces = (("down_proj", tensor),)
            for expert in range(tensor.shape[0]):
                for proj, expert_tensor in pieces:
                    out_name = expert_weight_name(layer, expert, proj)
                    q, scales = quantize_row(expert_tensor[expert].contiguous(), args.ebits)
                    converted[out_name] = q
                    converted[out_name + ".qs"] = scales
                    expert_tensors += 1
                    expert_objects.add((layer, expert))
                    total_expert_f32 += expert_tensor[expert].numel() * expert_tensor.element_size()
                    total_expert_q += q.numel() + scales.numel() * 4
        save_file(converted, str(out / shard.name))
        print("ok")

    index = {
        "backend": "qwen_moe",
        "expert_key": "layer_expert",
        "expert_tensor_count": expert_tensors,
        "expert_object_count": len(expert_objects),
        "expert_objects": [[layer, expert] for layer, expert in sorted(expert_objects)],
        "expert_bits": args.ebits,
    }
    (out / "qwen_experts.json").write_text(json.dumps(index, indent=2) + "\n")
    ratio = total_expert_q / max(total_expert_f32, 1) * 100.0
    print(f"\nDone. {expert_tensors} expert tensors across {len(expert_objects)} (layer, expert) objects.")
    print(f"Expert storage: {total_expert_f32/1e9:.2f} GB -> {total_expert_q/1e9:.2f} GB ({ratio:.1f}%)")
    print(f"Model ready at: {out}")
    print(f"Run scaffold: SNAP={out} ./qwen_moe")


if __name__ == "__main__":
    main()
