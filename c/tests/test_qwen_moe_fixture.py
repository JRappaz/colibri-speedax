import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def have_qwen_fixture_deps() -> bool:
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(have_qwen_fixture_deps(), "requires torch, safetensors, and transformers Qwen2-MoE")
class QwenMoeFixtureTests(unittest.TestCase):
    def test_tiny_qwen_checkpoint_converts_and_passes_layout_check(self):
        subprocess.run(["make", "qwen_moe"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hf = tmp / "hf"
            ref = tmp / "ref_qwen_moe.json"
            converted = tmp / "converted"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "make_qwen_moe_oracle.py"),
                    "--out",
                    str(hf),
                    "--ref",
                    str(ref),
                    "--max-new-tokens",
                    "2",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "convert_qwen_moe.py"),
                    "--model",
                    str(hf),
                    "--out",
                    str(converted),
                    "--ebits",
                    "8",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            index = json.loads((converted / "qwen_experts.json").read_text())
            self.assertEqual(index["expert_tensor_count"], 2 * 8 * 3)
            self.assertEqual(index["expert_object_count"], 2 * 8)
            result = subprocess.run(
                [str(ROOT / ("qwen_moe.exe" if os.name == "nt" else "qwen_moe"))],
                cwd=ROOT,
                env=os.environ | {"SNAP": str(converted)},
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIn("expert_tensors=48", result.stdout)
            self.assertIn("layout check succeeded", result.stderr)

            result = subprocess.run(
                [
                    str(ROOT / ("qwen_moe.exe" if os.name == "nt" else "qwen_moe")),
                    str(ref),
                    "8",
                ],
                cwd=ROOT,
                env=os.environ | {"SNAP": str(converted)},
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIn("Qwen MoE tiny forward", result.stdout)
            self.assertIn("Matching tokens: 2/2", result.stdout)


if __name__ == "__main__":
    unittest.main()
