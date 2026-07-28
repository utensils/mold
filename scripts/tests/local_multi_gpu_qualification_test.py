#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("local_multi_gpu_runner", ROOT / "scripts/qualify-local-multi-gpu.py")
validator = load(
    "local_multi_gpu_validator", ROOT / "scripts/validate-local-multi-gpu-report.py"
)


class RunnerPureContracts(unittest.TestCase):
    def test_csv_parser_rejects_truncated_hardware_rows(self):
        with self.assertRaisesRegex(ValueError, "expected 6 CSV columns"):
            runner.parse_csv_lines("0, GPU-a, RTX 3090\n", 6)

    def test_unrelated_real_uuids_have_no_ambiguous_prefix(self):
        self.assertIsNone(
            runner.common_hex_prefix(
                [
                    "cuda:44f80ce523fca5ddac4e133142952997",
                    "cuda:ba027fc579158d586738b7eaafe427b4",
                ]
            )
        )

    def test_models_manifest_detects_metadata_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            model = root / "model.safetensors"
            model.write_bytes(b"before")
            before = runner.models_tree_manifest(root)
            model.write_bytes(b"after-longer")
            self.assertNotEqual(before, runner.models_tree_manifest(root))


class ReportValidationContracts(unittest.TestCase):
    def fixture(self, root: pathlib.Path) -> pathlib.Path:
        binary = root / "mold"
        binary.write_bytes(b"candidate")
        evidence = root / "evidence.json"
        evidence.write_text('{"exact":true}\n', encoding="utf-8")
        uuids = [
            "GPU-44f80ce5-23fc-a5dd-ac4e-133142952997",
            "GPU-ba027fc5-7915-8d58-6738-b7eaafe427b4",
        ]
        checks = {
            name: {
                "status": "passed",
                "summary": f"{name} passed",
                "evidence_labels": ["fixture"],
            }
            for name in validator.REQUIRED_CHECKS
        }
        report = {
            "schema_version": validator.SCHEMA_VERSION,
            "started_at": "2026-07-28T00:00:00Z",
            "finished_at": "2026-07-28T00:01:00Z",
            "hardware_qualified": True,
            "candidate": {
                "path": str(binary),
                "sha256": validator.sha256(binary),
                "version": "mold fixture",
                "server_pid": 123,
            },
            "host": {
                "hostname": "fixture-host",
                "expected_gpu_uuids": uuids,
                "devices": [
                    {
                        "index": index,
                        "uuid": value,
                        "name": "NVIDIA GeForce RTX 3090",
                        "memory_total_mib": 24576,
                        "compute_capability": "8.6",
                        "driver_version": "999.0",
                    }
                    for index, value in enumerate(uuids)
                ],
            },
            "isolation": {
                "bind": "127.0.0.1",
                "port": 17681,
                "mold_home": str(root / "home"),
                "db_path": str(root / "mold.db"),
                "output_dir": str(root / "output"),
                "models_dir": str(root / "models"),
                "preexisting_listener_absent": True,
            },
            "request": {
                "path": str(root / "request.json"),
                "sha256": "a" * 64,
                "model": "sd15:fp16",
                "job_count": 4,
            },
            "checks": checks,
            "evidence": [
                {
                    "label": "fixture",
                    "path": str(evidence),
                    "sha256": validator.sha256(evidence),
                }
            ],
        }
        path = root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_passing_fixture_validates_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            report = self.fixture(root)
            validator.validate(report, require_passing=True)
            (root / "evidence.json").write_text('{"exact":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validator.validate(report, require_passing=True)

    def test_qualified_bit_must_equal_required_checks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.fixture(root)
            report = json.loads(path.read_text())
            report["checks"]["selector_matrix"]["status"] = "failed"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "const|conjunction"):
                validator.validate(path, require_passing=False)


if __name__ == "__main__":
    unittest.main()
