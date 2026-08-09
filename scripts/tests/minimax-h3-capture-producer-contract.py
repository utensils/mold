#!/usr/bin/env python3
"""Adversarial contract tests for the external H3 conditioner producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-conditioner.py"
CONFORMANCE_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_CONFORMANCE_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = load_module("minimax_h3_capture_producer_contract", PRODUCER_PATH)
conformance = load_module("minimax_h3_capture_base_contract", CONFORMANCE_PATH)
gpu = load_module("minimax_h3_capture_gpu_contract", GPU_CONFORMANCE_PATH)


def run(arguments: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sample_evidence() -> Any:
    return producer.PrimitiveEvidence(
        token_ids=(2, 3, 151652, 151655, 151653, 5, 7, 11),
        special_presentation=(3, 1, 0, 2, 4, 151652, 151653, 151655, 151656),
        processor_shapes=(2, 1, 3, 256, 256, 3, 2, 4, 4, 64, 64, 3),
        sampled_bfloat16_bits=(0xBF80, 0x0000, 0x3F00, 0x3F80, 0x4000),
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        image_sha256=hashlib.sha256(b"image").hexdigest(),
        video_sha256=hashlib.sha256(b"video").hexdigest(),
        sampled_source_indices=(0, 12, 24, 36),
        sampled_timestamps_millis=(250, 1250),
    )


class FakeConformanceTool:
    LAYER_OUTPUT_SCHEMA_PATH = conformance.LAYER_OUTPUT_SCHEMA_PATH
    BUNDLE_SCHEMA_PATH = conformance.BUNDLE_SCHEMA_PATH

    def __init__(self) -> None:
        self.bundle_validation_calls = 0

    @staticmethod
    def validate_schema(value: Any, schema_path: pathlib.Path) -> None:
        conformance.validate_schema(value, schema_path)

    def validate_bundle(
        self,
        manifest: dict[str, Any],
        fixture_root_value: str,
        bundle_value: str,
        authorization_value: str,
    ) -> None:
        del manifest, authorization_value
        self.bundle_validation_calls += 1
        fixture_root = pathlib.Path(fixture_root_value).resolve(strict=True)
        bundle_path = pathlib.Path(bundle_value).resolve(strict=True)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        for fixture in bundle["fixtures"]:
            evidence = (fixture_root / fixture["relative_path"]).resolve(strict=True)
            self.assert_inside(evidence, fixture_root)
            if hashlib.sha256(evidence.read_bytes()).hexdigest() != fixture["sha256"]:
                raise AssertionError("fixture hash mismatch")

    @staticmethod
    def assert_inside(path: pathlib.Path, parent: pathlib.Path) -> None:
        try:
            path.relative_to(parent)
        except ValueError as error:
            raise AssertionError("fixture escaped root") from error


class CaptureProducerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = conformance.validate_manifest()
        cls.contract = gpu.exact_layer_contracts(cls.manifest)["tokenizer-processor"]

    def build_document(self) -> dict[str, Any]:
        return producer.build_layer_document(
            self.manifest,
            sample_evidence(),
            producer.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "cuda:0",
        )

    def assert_gpu_rejects(self, document: dict[str, Any]) -> None:
        with self.assertRaises(gpu.GpuConformanceFailure):
            gpu.validate_manifest_layer_evidence(document, self.contract, "oracle")

    def test_emits_the_exact_layer_and_bundle_schemas_consumed_by_gpu_runner(
        self,
    ) -> None:
        document = self.build_document()
        conformance.validate_schema(document, conformance.LAYER_OUTPUT_SCHEMA_PATH)
        gpu.validate_manifest_layer_evidence(document, self.contract, "oracle")
        encoded = producer.document_bytes(document)
        bundle = producer.build_bundle(
            self.manifest,
            document,
            encoded,
            "captures/tokenizer-processor/case/oracle.json",
            producer.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "cuda:0",
        )
        conformance.validate_schema(bundle, conformance.BUNDLE_SCHEMA_PATH)
        self.assertEqual(
            bundle["fixtures"][0]["sha256"], hashlib.sha256(encoded).hexdigest()
        )
        self.assertEqual(
            [output["key"] for output in document["outputs"]],
            self.contract["required_measurements"],
        )
        self.assertEqual(
            document["adapter"]["tensor_hash_encoding"],
            gpu.CANONICAL_TENSOR_HASH_ENCODING,
        )

    def test_source_and_component_authorities_are_not_caller_selectable(self) -> None:
        document = self.build_document()
        self.assertEqual(
            document["producer"]["revision"],
            conformance.EXPECTED_REVISIONS["diffusers"],
        )
        self.assertEqual(document["producer"]["source_id"], "diffusers")
        self.assertEqual(
            [item["id"] for item in document["input"]["component_indexes"]],
            list(producer.REQUIRED_COMPONENT_IDS),
        )
        self.assertIn(
            producer.TRANSFORMERS_REVISION,
            json.dumps(producer.build_input_descriptor(sample_evidence())),
        )

    def test_component_authority_drift_is_rejected(self) -> None:
        document = self.build_document()
        document["input"]["component_indexes"][3]["sha256"] = "0" * 64
        self.assert_gpu_rejects(document)

    def test_excluded_acceleration_and_backend_aliases_are_rejected(self) -> None:
        enabled = self.build_document()
        enabled["environment"]["acceleration_policy"]["enabled"].append("sage-attn")
        self.assert_gpu_rejects(enabled)

        backend = self.build_document()
        backend["environment"]["attention_backend"] = "tensorfloat32-kernel"
        self.assert_gpu_rejects(backend)

    def test_measurement_order_dtype_and_policy_cannot_be_weakened(self) -> None:
        reordered = self.build_document()
        reordered["outputs"][0], reordered["outputs"][1] = (
            reordered["outputs"][1],
            reordered["outputs"][0],
        )
        self.assert_gpu_rejects(reordered)

        dtype = self.build_document()
        dtype["outputs"][-1]["dtype"] = "float32"
        self.assert_gpu_rejects(dtype)

        tolerance = self.build_document()
        tolerance["comparison"][-1]["absolute"] = 0.5
        self.assert_gpu_rejects(tolerance)

    def test_canonical_tensor_hashes_bind_typed_little_endian_values(self) -> None:
        integer = producer.integer_tensor_record("token-ids", (1, -2, 3))
        expected_int = hashlib.sha256(
            b"\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\xfe\xff\xff\xff\xff\xff\xff\xff"
            b"\x03\x00\x00\x00\x00\x00\x00\x00"
        ).hexdigest()
        self.assertEqual(integer["content_sha256"], expected_int)

        bf16 = producer.bfloat16_tensor_record("sampled-values", (0x3F80, 0xBF80))
        self.assertEqual(
            bf16["content_sha256"], hashlib.sha256(b"\x80\x3f\x80\xbf").hexdigest()
        )
        self.assertEqual(bf16["statistics"]["min"], -1.0)
        self.assertEqual(bf16["statistics"]["max"], 1.0)

    def test_environment_aliases_fail_closed_before_any_runtime_import(self) -> None:
        for key in (
            "TORCH_COMPILE",
            "TORCHINDUCTOR_CACHE_DIR",
            "MOLD_H3_FP8",
            "SAGE_ATTENTION_BACKEND",
            "NVIDIA_TF32_OVERRIDE",
        ):
            with self.subTest(key=key), self.assertRaises(producer.CaptureFailure):
                producer.reject_excluded_acceleration_environment({key: "1"})
        producer.reject_excluded_acceleration_environment(
            {"TORCH_COMPILE": "false", "UNRELATED": "1"}
        )

    def test_backend_configuration_must_take_effect(self) -> None:
        class FakeCudaRuntime:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def set_device(index: int) -> None:
                self.assertEqual(index, 0)

        class FakeCudaBackends:
            matmul = SimpleNamespace(allow_tf32=True)

            @staticmethod
            def enable_flash_sdp(enabled: bool) -> None:
                del enabled

            @staticmethod
            def enable_mem_efficient_sdp(enabled: bool) -> None:
                del enabled

            @staticmethod
            def enable_cudnn_sdp(enabled: bool) -> None:
                del enabled

            @staticmethod
            def enable_math_sdp(enabled: bool) -> None:
                del enabled

            @staticmethod
            def flash_sdp_enabled() -> bool:
                return True

            @staticmethod
            def mem_efficient_sdp_enabled() -> bool:
                return True

            @staticmethod
            def cudnn_sdp_enabled() -> bool:
                return True

            @staticmethod
            def math_sdp_enabled() -> bool:
                return True

        class FakeTorch:
            cuda = FakeCudaRuntime()
            backends = SimpleNamespace(
                cuda=FakeCudaBackends(),
                cudnn=SimpleNamespace(
                    allow_tf32=True,
                    benchmark=True,
                    deterministic=False,
                ),
            )
            deterministic = False

            @classmethod
            def use_deterministic_algorithms(cls, enabled: bool) -> None:
                cls.deterministic = enabled

            @staticmethod
            def set_float32_matmul_precision(value: str) -> None:
                del value

            @classmethod
            def are_deterministic_algorithms_enabled(cls) -> bool:
                return cls.deterministic

            @staticmethod
            def device(value: str) -> str:
                return value

        with self.assertRaisesRegex(
            producer.CaptureFailure, "attention backend policy drifted"
        ):
            producer.configure_torch(SimpleNamespace(torch=FakeTorch), "cuda:0")

    def test_missing_authorization_environment_fails_before_capture(self) -> None:
        with self.assertRaisesRegex(
            producer.CaptureFailure, "MOLD_H3_FIXTURE_ROOT is required"
        ):
            producer.run_tokenizer_processor_capture({}, "cuda:0")

    def test_authorization_must_bind_the_one_reviewed_external_document(self) -> None:
        class FakeAuthorizationTool:
            def __init__(self, source: pathlib.Path, digest: str) -> None:
                self.source = source
                self.digest = digest

            def validate_authorization(self, path: pathlib.Path) -> dict[str, str]:
                del path
                return {
                    "source_document_path": str(self.source),
                    "source_document_sha256": self.digest,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "review.bin"
            record = root / "authorization.json"
            source.write_bytes(b"reviewed")
            record.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(producer.CaptureFailure):
                producer.validate_reviewed_authorization(
                    FakeAuthorizationTool(source, "0" * 64), record
                )
            accepted = producer.validate_reviewed_authorization(
                FakeAuthorizationTool(
                    source, producer.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256
                ),
                record,
            )
            self.assertEqual(
                accepted["source_document_sha256"],
                producer.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            )

    def test_repository_paths_are_never_valid_external_roots(self) -> None:
        with self.assertRaises(producer.CaptureFailure):
            producer.external_directory("fixture root", str(REPO_ROOT))
        with self.assertRaises(producer.CaptureFailure):
            producer.external_file("authorization record", str(PRODUCER_PATH))

    def test_source_checkout_requires_exact_clean_head_and_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run(["git", "init"], root)
            run(["git", "config", "user.name", "Capture Contract"], root)
            run(["git", "config", "user.email", "capture@example.invalid"], root)
            (root / "source.py").write_text("PIN = 1\n", encoding="utf-8")
            run(["git", "add", "source.py"], root)
            run(["git", "commit", "-m", "test: pin source"], root)
            revision = run(["git", "rev-parse", "HEAD"], root)
            repository = "https://github.com/example/capture-source"
            run(["git", "remote", "add", "origin", repository + ".git"], root)
            producer.verify_git_checkout("source", root, revision, repository)
            (root / "source.py").write_text("PIN = 2\n", encoding="utf-8")
            with self.assertRaises(producer.CaptureFailure):
                producer.verify_git_checkout("source", root, revision, repository)

    def test_model_components_require_hash_and_snapshot_revision_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            components = []
            for position, identifier in enumerate(
                producer.REQUIRED_COMPONENT_IDS, start=1
            ):
                relative_path = f"component/{position}.json"
                data = f"{identifier}\n".encode()
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                metadata = (
                    root
                    / ".cache"
                    / "huggingface"
                    / "download"
                    / f"{relative_path}.metadata"
                )
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(
                    producer.MODEL_REVISION + "\nopaque-etag\n0\n",
                    encoding="utf-8",
                )
                components.append(
                    {
                        "id": identifier,
                        "source_id": "minimax-official-model",
                        "relative_path": relative_path,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            manifest = {"component_indexes": components}
            verified = producer.verify_model_components(root, manifest)
            self.assertEqual(len(verified), len(producer.REQUIRED_COMPONENT_IDS))

            metadata = (
                root
                / ".cache"
                / "huggingface"
                / "download"
                / f"{components[0]['relative_path']}.metadata"
            )
            metadata.write_text("0" * 40 + "\nopaque-etag\n0\n", encoding="utf-8")
            with self.assertRaises(producer.CaptureFailure):
                producer.verify_model_components(root, manifest)
            metadata.write_text(
                producer.MODEL_REVISION + "\nopaque-etag\n0\n",
                encoding="utf-8",
            )
            (root / components[-1]["relative_path"]).write_bytes(b"drift")
            with self.assertRaises(producer.CaptureFailure):
                producer.verify_model_components(root, manifest)

    def test_external_capture_is_exclusive_schema_checked_and_mode_0600(self) -> None:
        document = self.build_document()
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = pathlib.Path(temporary).resolve()
            authorization = fixture_root / "authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            fake = FakeConformanceTool()
            layer_path, bundle_path = producer.write_capture(
                fixture_root,
                authorization,
                self.manifest,
                document,
                "cuda:0",
                fake,
                gpu,
            )
            self.assertEqual(fake.bundle_validation_calls, 1)
            self.assertTrue(layer_path.is_file())
            self.assertTrue(bundle_path.is_file())
            self.assertEqual(stat.S_IMODE(layer_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(bundle_path.stat().st_mode), 0o600)
            with self.assertRaises(producer.CaptureFailure):
                producer.write_capture(
                    fixture_root,
                    authorization,
                    self.manifest,
                    document,
                    "cuda:0",
                    fake,
                    gpu,
                )

    def test_capture_output_cannot_follow_a_symlink_outside_fixture_root(self) -> None:
        document = self.build_document()
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            fixture_root = base / "fixture"
            outside = base / "outside"
            fixture_root.mkdir()
            outside.mkdir()
            (fixture_root / "captures").symlink_to(outside, target_is_directory=True)
            authorization = base / "authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(producer.CaptureFailure):
                producer.write_capture(
                    fixture_root.resolve(),
                    authorization,
                    self.manifest,
                    document,
                    "cuda:0",
                    FakeConformanceTool(),
                    gpu,
                )

    def test_checked_tree_contains_no_production_capture_or_binary_evidence(
        self,
    ) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "tests/fixtures/minimax_h3"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        paths = [pathlib.Path(value.decode("utf-8")) for value in tracked if value]
        self.assertTrue(paths)
        self.assertTrue(all(path.suffix == ".json" for path in paths))
        producer_source = PRODUCER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from safetensors", producer_source)
        self.assertNotIn("safe_open(", producer_source)

    def test_capture_producer_is_not_reachable_from_shipped_runtime_or_release(
        self,
    ) -> None:
        roots = [
            "Cargo.toml",
            "Cargo.lock",
            "flake.nix",
            "crates",
            "apps",
            "desktop",
            "studio",
            "web",
            "scripts/release",
        ]
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", *roots],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        needle = PRODUCER_PATH.name
        references = []
        for value in tracked:
            if not value:
                continue
            path = REPO_ROOT / value.decode("utf-8")
            if path.is_file() and needle in path.read_text(
                encoding="utf-8", errors="ignore"
            ):
                references.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(references, [])
        protected_workflow = (
            REPO_ROOT / ".github" / "workflows" / "minimax-h3-private-conformance.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn(needle, protected_workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
