from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gemma4_bench as G
import quality_gemma as Q


class AnswerScoringTests(unittest.TestCase):
    def test_final_numeric_answer_not_intermediate_reasoning(self):
        task = {"dataset": "gsm8k", "answer": "1200"}
        self.assertTrue(Q.score(task, "First compute 12 * 100.\nFinal answer: $1,200.00")['correct'])
        self.assertFalse(Q.score(task, "There are 1200 apples but I am unsure.")['correct'])
        self.assertFalse(Q.score(task, "Final answer: 1201")['correct'])
        self.assertFalse(Q.score(task, "Final answer: 1200", ok=False)['correct'])

    def test_multiple_choice_does_not_search_incidental_letters(self):
        task = {"dataset": "arc_challenge", "answer": "B"}
        self.assertTrue(Q.score(task, "A is unlikely.\nFinal answer: (b).")['correct'])
        self.assertFalse(Q.score(task, "B and C both look possible.")['correct'])
        self.assertFalse(Q.score(task, "Final answer: C")['correct'])
        self.assertIsNone(Q.final_answer("", "arc_challenge"))

    def test_failed_operations_stay_in_task_denominator(self):
        tasks = [{"dataset": name, "id": str(i)} for name in ("gsm8k", "arc_challenge") for i in range(2)]
        def sample(correct, ok, text):
            return {"correct": correct, "ok": ok, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        providers = {
            "antfly": {"samples": [sample(True, True, "A"), sample(False, False, ""), sample(True, True, "B"), sample(False, True, "A")]},
            "llama.cpp": {"samples": [sample(False, True, "C"), sample(True, True, "B"), sample(True, True, "B"), sample(True, True, "D")]},
        }
        compared = Q.summarize(tasks, providers)
        self.assertEqual(compared['gsm8k']['scores']['antfly'], {"correct": 1, "denominator": 2, "accuracy": .5})
        self.assertEqual(compared['gsm8k']['accuracy_difference_antfly_minus_reference'], 0)
        self.assertEqual(compared['gsm8k']['paired_denominator'], 2)
        self.assertEqual(compared['gsm8k']['antfly_only_correct'], 1)
        self.assertEqual(compared['gsm8k']['reference_only_correct'], 1)
        self.assertEqual(compared['gsm8k']['text_equality']['denominator'], 1)
        self.assertEqual(compared['arc_challenge']['accuracy_difference_antfly_minus_reference'], -.5)
        self.assertEqual(compared['arc_challenge']['text_equality']['equal'], 1)


class PromptEvidenceTests(unittest.TestCase):
    def test_prompt_fingerprints_preserve_whitespace_and_token_identity(self):
        rendering = '<bos>user\n  literal text\nassistant'
        observed = G.parse_native_prompt(f"load done\nprompt:\n{rendering}\nprompt_token_ids: 2 10 99\noutput")
        self.assertEqual(observed['rendered_prompt_sha256'], hashlib.sha256(rendering.encode()).hexdigest())
        self.assertEqual(observed['prompt_token_count'], 3)
        self.assertNotIn('prompt_token_ids', observed)
        self.assertNotIn('rendered_prompt', observed)
        reference = G.parse_native_prompt(f"prompt:\n{rendering}\nprompt_token_ids: 2 11 99\n")
        comparison = G.compare_prompts(observed, reference)
        self.assertFalse(comparison['diagnostic_token_ids_equal'])
        self.assertTrue(comparison['diagnostic_rendering_equal'])
        self.assertEqual(comparison['http_generation_prompt_parity'], 'unproven')
        changed_spacing = G.parse_native_prompt(f"prompt:\n{rendering.replace('  ', ' ')}\nprompt_token_ids: 2 10 99\n")
        self.assertFalse(G.compare_prompts(observed, changed_spacing)['diagnostic_rendering_equal'])
        self.assertTrue(G.compare_prompts(observed, changed_spacing)['diagnostic_token_ids_equal'])

    def test_counts_or_reference_only_ids_are_not_native_evidence(self):
        native = G.parse_native_prompt('prompt_tokens=3\n')
        comparison = G.compare_prompts(native, G.parse_native_prompt("prompt:\nhello\nprompt_token_ids: 1 2 3\n"))
        self.assertIsNone(comparison['diagnostic_token_ids_equal'])
        self.assertEqual(comparison['http_generation_prompt_parity'], 'unproven')
        self.assertEqual(G.parse_native_prompt('prompt:\nhello\nprompt_token_ids: 1 bad 2\n')['status'], 'unavailable')


class GgufInventoryTests(unittest.TestCase):
    @staticmethod
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data

    def test_mixed_storage_and_head_inventory_survive_large_metadata_arrays(self):
        metadata = self.string('tokens') + struct.pack('<IIQ', 9, 8, 2) + self.string('one') + self.string('two')
        header = b'GGUF' + struct.pack('<IQQ', 3, 3, 1) + metadata
        tensors = b''
        for name, shape, kind in [('blk.0.attn_q.weight', [32, 16], 2), ('output_norm.weight', [16], 0), ('token_embd.weight', [16, 64], 1)]:
            tensors += self.string(name) + struct.pack('<I', len(shape))
            tensors += b''.join(struct.pack('<Q', d) for d in shape) + struct.pack('<IQ', kind, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.gguf'
            path.write_bytes(header + tensors)
            inventory = G.gguf_inventory(path)
        self.assertEqual(inventory['storage_types'], {
            'Q4_0': {'tensors': 1, 'parameters': 512}, 'F32': {'tensors': 1, 'parameters': 16},
            'F16': {'tensors': 1, 'parameters': 1024}})
        self.assertEqual(inventory['embedding_and_output_tensors']['output_norm.weight']['storage_type'], 'F32')
        self.assertEqual(inventory['tensor_count'], 3)

    def test_truncated_checkpoint_is_rejected_not_reported_as_empty_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 0))
            with self.assertRaises(ValueError):
                G.gguf_inventory(path)


if __name__ == '__main__':
    unittest.main()
