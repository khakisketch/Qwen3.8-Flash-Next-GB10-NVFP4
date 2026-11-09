"""Static checks that the installed service and client example stay aligned."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class ServiceConfigTests(unittest.TestCase):
    def test_unit_uses_the_verified_64k_two_sequence_profile(self):
        unit = (ROOT / "serve" / "systemd" / "qwen38-fp4.service.in").read_text()
        self.assertIn("--context 65536", unit)
        self.assertIn("--sequences 2", unit)
        self.assertIn("--kv-cache-memory-bytes 9663676416", unit)
        self.assertIn("--listen-all", unit)
        self.assertIn("--reclaim-csf-cache", unit)

    def test_unit_has_scoped_cleanup_and_bounded_restart(self):
        unit = (ROOT / "serve" / "systemd" / "qwen38-fp4.service.in").read_text()
        self.assertIn("StartLimitIntervalSec=900", unit)
        self.assertIn("StartLimitBurst=5", unit)
        self.assertIn("ExecStartPre=@PROJECT_DIR@/serve/systemd/cleanup-stale.sh", unit)
        self.assertIn("ExecStopPost=@PROJECT_DIR@/serve/systemd/cleanup-stale.sh", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("SuccessExitStatus=130", unit)

    def test_opencode_example_is_v2_and_targets_general_at_local_model(self):
        example = json.loads(
            (ROOT / "serve" / "opencode.example.json").read_text()
        )
        self.assertIn("providers", example)
        self.assertNotIn("provider", example)
        provider = example["providers"]["spark-qwen"]
        self.assertEqual(
            provider["package"], "@opencode/ai/providers/openai-compatible"
        )
        self.assertEqual(provider["settings"]["baseURL"], "http://172.30.1.44:8000/v1")
        model = provider["models"]["nvidia/Qwen3.8-Flash-Next-FP4-Local"]
        self.assertTrue(model["capabilities"]["tools"])
        self.assertEqual(
            model["capabilities"]["reasoning"]["effort"],
            ["low", "medium", "xhigh"],
        )
        self.assertEqual(
            example["agents"]["general"]["model"],
            "spark-qwen/nvidia/Qwen3.8-Flash-Next-FP4-Local",
        )


if __name__ == "__main__":
    unittest.main()
