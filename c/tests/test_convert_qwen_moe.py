import unittest

from tools.convert_qwen_moe import is_expert_weight, parse_expert_key


class QwenMoeConverterTests(unittest.TestCase):
    def test_expert_key_preserves_layer_and_expert_identity(self):
        key = "model.layers.12.mlp.experts.34.gate_proj.weight"
        self.assertEqual(parse_expert_key(key), (12, 34, "gate_proj"))
        self.assertTrue(is_expert_weight(key))

    def test_non_expert_key_is_dense(self):
        self.assertIsNone(parse_expert_key("model.layers.12.mlp.gate.weight"))
        self.assertFalse(is_expert_weight("model.embed_tokens.weight"))


if __name__ == "__main__":
    unittest.main()
