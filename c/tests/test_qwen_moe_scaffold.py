import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenMoeScaffoldTests(unittest.TestCase):
    def test_scaffold_loads_minimal_qwen_config(self):
        subprocess.run(["make", "qwen_moe"], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "vocab_size": 128,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "norm_topk_prob": True,
            }
            Path(tmp, "config.json").write_text(json.dumps(config))
            env = os.environ | {"SNAP": tmp}
            result = subprocess.run(
                [str(ROOT / ("qwen_moe.exe" if os.name == "nt" else "qwen_moe"))],
                cwd=ROOT,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertIn("Qwen MoE backend scaffold", result.stdout)
        self.assertIn("(layer_id, expert_id)", result.stdout)
        self.assertIn("config-only scaffold check", result.stdout)
        self.assertIn("scaffold/layout check succeeded", result.stderr)


if __name__ == "__main__":
    unittest.main()
