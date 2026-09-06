"""Retention and provenance contracts for the CUDA capture runner (no GPU needed)."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "capture-hunyuan3d-cuda.py"
spec = importlib.util.spec_from_file_location("hunyuan_capture", RUNNER)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.models = self.home / "models"
        self.models.mkdir()
        self.source = self.home / "source.png"
        self.source.write_bytes(b"retained source")
        self.model = self.models / "model.safetensors"
        self.model.write_bytes(b"retained model")

    def run_capture(self, program, **kwargs):
        return capture.capture(
            home=self.home,
            models=self.models,
            evidence_root=self.home / "output" / "verification",
            gpu_uuid="GPU-test",
            command=[sys.executable, "-c", program],
            inputs=[self.source],
            model_files=[self.model],
            expected_outputs=["result.glb"],
            sample_gpu=lambda _: {"used_mib": 8, "utilization_percent": 0},
            **kwargs,
        )

    def test_success_records_identity_and_keeps_originals(self):
        path, code = self.run_capture(
            "import os,pathlib; p=pathlib.Path(os.environ['MOLD_OUTPUT_DIR']); "
            "(p/'result.glb').write_bytes(b'mesh'); print('complete')"
        )
        self.assertEqual(code, 0)
        report = json.loads((path / "run.json").read_text())
        self.assertEqual(report["status"], "captured")
        self.assertEqual(report["gpu_uuid"], "GPU-test")
        self.assertIn("sha256", report["model_files"][0])
        self.assertEqual(report["gpu_board_used_mib_max"], 8)
        self.assertIn("complete", (path / "stdout.log").read_text())
        self.assertEqual(self.source.read_bytes(), b"retained source")
        self.assertEqual(self.model.read_bytes(), b"retained model")

    def test_failure_retains_partial_output_and_exit_status(self):
        path, code = self.run_capture(
            "import os,pathlib,sys; "
            "(pathlib.Path(os.environ['MOLD_OUTPUT_DIR'])/'partial.glb').write_bytes(b'partial'); "
            "print('failure',file=sys.stderr); sys.exit(17)"
        )
        self.assertEqual(code, 17)
        self.assertTrue((path / "partial.glb").is_file())
        self.assertEqual(json.loads((path / "run.json").read_text())["status"], "failed")

    def test_zero_exit_without_deliverable_is_failure(self):
        path, code = self.run_capture("pass")
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads((path / "run.json").read_text())["missing_outputs"], ["result.glb"])

    def test_repeated_runs_never_overwrite(self):
        first, _ = self.run_capture("pass")
        second, _ = self.run_capture("pass")
        self.assertNotEqual(first, second)
        self.assertTrue((first / "run.json").is_file())

    def test_output_traversal_is_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            capture.validate_outputs(["../production.glb"])

    def test_evidence_outside_home_is_rejected(self):
        with self.assertRaises(ValueError):
            capture.validate_roots(self.home, self.models, self.home.parent / "elsewhere")

    def test_missing_inputs_fail_before_launch(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_capture("pass")


if __name__ == "__main__":
    unittest.main()
