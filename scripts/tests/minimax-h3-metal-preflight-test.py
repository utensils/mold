#!/usr/bin/env python3
"""CPU-only tests for the offline H3 Metal budget auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "h3_metal_preflight", ROOT / "scripts/minimax-h3-metal-preflight.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
GIB = 1 << 30


def capture():
    return {
        "schema": "mold.h3-metal-budget-snapshot.v1",
        "identities": {
            "source_commit": "a" * 40,
            "candle_commit": "b" * 40,
            "executable_sha256": "c" * 64,
            "request_sha256": "d" * 64,
            "plan_sha256": "e" * 64,
            "budget_sha256": "f" * 64,
        },
        "phase_bytes": {
            f"{phase}_phase_{space}_bytes": 2 * GIB if space == "host" else 3 * GIB
            for phase in runner.PHASES for space in ("host", "device")
        },
        "owner_projection": {"device_bytes": 5 * GIB, "additional_host_bytes": 0},
        "snapshot": {"available_bytes": 28 * GIB, "device_headroom_bytes": 30 * GIB},
        "native_allocation_ceiling_bytes": 8 * GIB,
    }


class BudgetAuditTests(unittest.TestCase):
    def test_phase_set_matches_existing_rust_authority(self):
        source = (ROOT / "crates/mold-inference/src/minimax_h3/private_server.rs").read_text()
        body = source.split("fn private_h3_unified_target_peak_bytes(", 1)[1].split("\n///", 1)[0]
        phases = re.findall(r"budget\.(\w+)_phase_device_bytes", body)
        self.assertEqual(list(runner.PHASES), phases)

    def test_maximum_is_within_a_phase_not_sum_or_independent_peaks(self):
        data = capture()
        data["phase_bytes"]["qwen_encode_phase_host_bytes"] = 6 * GIB
        data["phase_bytes"]["denoise_phase_device_bytes"] = 7 * GIB
        data["owner_projection"]["device_bytes"] = 9 * GIB
        result = runner.audit(data)
        self.assertEqual(result["unified_peak_bytes"], 9 * GIB)
        self.assertEqual(result["binding_phases"], ["qwen_encode", "denoise"])
        self.assertEqual(result["phases"][5]["host_bytes"], 6 * GIB)
        self.assertEqual(result["decision"], "budget_fits_snapshot")
        self.assertFalse(result["launch_ready"])

    def test_wrong_or_double_charged_owner_projection_is_refused(self):
        for key, value in [("device_bytes", 4 * GIB), ("additional_host_bytes", 2 * GIB)]:
            data = capture()
            data["owner_projection"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "owner projection"):
                runner.audit(data)

    def test_missing_extra_and_invalid_phase_bytes_are_refused(self):
        for value in [-1, True, 1.5, "1024", 1 << 64]:
            data = capture()
            data["phase_bytes"]["mux_phase_host_bytes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.audit(data)
        for change in (lambda d: d.pop("mux_phase_host_bytes"), lambda d: d.update(fake_phase_host_bytes=0)):
            data = capture()
            change(data["phase_bytes"])
            with self.assertRaises(ValueError):
                runner.audit(data)

    def test_u64_sum_overflow_and_zero_budget_are_refused(self):
        data = capture()
        data["phase_bytes"]["mux_phase_device_bytes"] = (1 << 64) - 1
        with self.assertRaisesRegex(ValueError, "overflow"):
            runner.audit(data)
        data = capture()
        data["phase_bytes"] = dict.fromkeys(data["phase_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "zero"):
            runner.audit(data)

    def test_capacity_failures_remain_named_and_never_clear_launch(self):
        mutations = [
            ("baseline", lambda d: d["snapshot"].update(available_bytes=23 * GIB)),
            ("host_floor", lambda d: d["snapshot"].update(available_bytes=16 * GIB)),
            ("device_headroom", lambda d: d["snapshot"].update(device_headroom_bytes=4 * GIB)),
            ("native_ceiling", lambda d: d.update(native_allocation_ceiling_bytes=2 * GIB)),
        ]
        for reason, mutate in mutations:
            data = capture()
            mutate(data)
            with self.subTest(reason=reason):
                result = runner.audit(data)
                self.assertEqual(result["decision"], "budget_refused")
                self.assertIn(reason, result["refusals"])
                self.assertFalse(result["launch_ready"])

    def test_native_ceiling_preserves_host_phase_residency_as_well_as_floor(self):
        data = capture()
        data["snapshot"]["available_bytes"] = 24 * GIB
        data["phase_bytes"]["qwen_encode_phase_host_bytes"] = 6 * GIB
        data["owner_projection"]["device_bytes"] = 9 * GIB
        result = runner.audit(data)
        self.assertIn("native_ceiling_host_floor", result["refusals"])

    def test_identity_and_duplicate_json_keys_are_refused(self):
        data = capture()
        data["identities"]["executable_sha256"] = "unknown"
        with self.assertRaises(ValueError):
            runner.audit(data)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.loads('{"schema": 1, "schema": 2}')

    def test_cli_exit_codes_do_not_claim_launch_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.json"
            data = capture()
            for expected in (0, 1, 2):
                if expected == 1:
                    data["snapshot"]["available_bytes"] = GIB
                if expected == 2:
                    data["phase_bytes"].pop("mux_phase_host_bytes")
                path.write_text(json.dumps(data))
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/minimax-h3-metal-preflight.py"), str(path)],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected != 2:
                    self.assertFalse(json.loads(result.stdout)["launch_ready"])
                else:
                    self.assertIn("refused", result.stderr)

    def test_audit_does_not_mutate_capture(self):
        data = capture()
        before = copy.deepcopy(data)
        runner.audit(data)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
