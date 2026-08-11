#!/usr/bin/env python3
"""Adversarial contract tests for the protected visual-VAE capture leaf."""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import pathlib
import stat
import struct
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-visual-vae.py"
RUST_ADAPTER_PATH = (
    REPO_ROOT / "crates" / "mold-inference" / "src" / "bin" / "h3_visual_vae_capture.rs"
)
MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
)


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = load_module("minimax_h3_visual_vae_capture_contract", PRODUCER_PATH)


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def payload(key: str, dtype: str, values: list[float]) -> producer.TensorPayload:
    encoding = "<f" if dtype == "float32" else "<e"
    encoded = b"".join(struct.pack(encoding, value) for value in values)
    return producer.TensorPayload(key, (len(values),), dtype, encoded)


def payloads() -> tuple[producer.TensorPayload, ...]:
    return tuple(
        payload(key, dtype, [0.0, 0.25, -0.5, 1.0])
        for key, dtype in producer.MEASUREMENT_DTYPES.items()
    )


def raw_document(case: dict) -> dict:
    tensors = []
    for item in payloads():
        tensors.append(
            {
                "key": item.key,
                "shape": list(item.shape),
                "dtype": item.dtype,
                "little_endian_base64": base64.b64encode(item.encoded).decode("ascii"),
            }
        )
    return {
        "schema_version": producer.RAW_OUTPUT_SCHEMA,
        "claim_marker": producer.CAPTURE_MARKER,
        "authorization_document_sha256": "a" * 64,
        "source_revision": "b" * 40,
        "model_revision": producer.MODEL_REVISION,
        "component_index_sha256": producer.VISUAL_INDEX_SHA256,
        "component_fingerprint_sha256": "c" * 64,
        "device": "cuda:0",
        "stored_weight_dtype": "float32",
        "encoder_compute_dtype": "float32",
        "decoder_compute_dtype": "float16",
        "attention_backend": "math",
        "cases": [
            {
                "case_id": case["case_id"],
                "input_sha256": case["input_sha256"],
                "tensors": tensors,
            }
        ],
    }


class VisualCaptureContract(unittest.TestCase):
    def test_case_identity_binds_geometry_tiling_and_seam_plan(self) -> None:
        case = producer.build_case()
        self.assertEqual(case["case_id"], producer.CASE_ID)
        self.assertEqual((case["frames"], case["height"], case["width"]), (22, 320, 320))
        self.assertEqual(
            (case["frames"] - 5) // 17 * 5 + 2,
            producer.LATENT_FRAMES,
        )
        self.assertRegex(case["input_sha256"], r"^[0-9a-f]{64}$")
        original = producer.TILE_OVERLAP
        try:
            producer.TILE_OVERLAP = 65
            self.assertNotEqual(
                producer.build_case()["input_sha256"], case["input_sha256"]
            )
        finally:
            producer.TILE_OVERLAP = original

    def test_manifest_binds_config_index_and_measurement_order(self) -> None:
        contract = producer.layer_contract(manifest())
        self.assertEqual(
            tuple(contract["required_component_indexes"]), producer.COMPONENT_IDS
        )
        self.assertEqual(
            tuple(contract["required_measurements"]), producer.MEASUREMENTS
        )
        changed = manifest()
        layer = next(
            item
            for item in changed["fixture_layers"]
            if item["id"] == producer.LAYER_ID
        )
        layer["required_measurements"] = list(reversed(layer["required_measurements"]))
        with self.assertRaisesRegex(producer.CaptureFailure, "measurement contract"):
            producer.layer_contract(changed)

    def test_manifest_component_hash_drift_is_rejected(self) -> None:
        changed = manifest()
        component = next(
            item
            for item in changed["component_indexes"]
            if item["id"] == "official-video-vae-config"
        )
        component["sha256"] = "0" * 64
        with self.assertRaisesRegex(producer.CaptureFailure, "component authority"):
            producer.component_authorities(changed)

    def test_manifest_pins_every_visual_checkpoint_shard(self) -> None:
        authorities = producer.component_authorities(manifest())
        observed = {item["id"]: item["sha256"] for item in authorities}
        for identifier, _, sha256 in producer.VISUAL_SHARD_AUTHORITIES:
            self.assertEqual(observed[identifier], sha256)

    def test_layer_document_retains_heterogeneous_precision_and_record_only_hashes(
        self,
    ) -> None:
        document = producer.build_layer_document(
            manifest(),
            "oracle",
            producer.DIFFUSERS_REVISION,
            "a" * 64,
            "cuda:0",
            "d" * 64,
            producer.build_case(),
            payloads(),
            {
                "python": "test",
                "torch": "test",
                "numpy": "test",
                "cuda": "test",
                "transformers_revision": producer.TRANSFORMERS_REVISION,
            },
        )
        self.assertEqual(
            [item["dtype"] for item in document["outputs"]],
            [producer.MEASUREMENT_DTYPES[key] for key in producer.MEASUREMENTS],
        )
        policies = {item["key"]: item for item in document["comparison"]}
        self.assertEqual(policies["posterior-seed-42"]["hash_policy"], "exact")
        for key in set(producer.MEASUREMENTS) - {"posterior-seed-42"}:
            self.assertEqual(policies[key]["hash_policy"], "record-only")
            self.assertLessEqual(policies[key]["absolute"], 1 / 64)

    def test_mold_document_cannot_claim_oracle_comparison_policy(self) -> None:
        document = producer.build_layer_document(
            manifest(),
            "mold",
            "b" * 40,
            "a" * 64,
            "cuda:0",
            "d" * 64,
            producer.build_case(),
            payloads(),
            {"unavailable": "contract-test"},
        )
        self.assertNotIn("comparison", document)
        self.assertEqual(
            document["environment"]["dtype"], "official-f32-encode-fp16-decode"
        )

    def test_raw_output_accepts_only_exact_precision_profile_and_order(self) -> None:
        case = producer.build_case()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.json"
            path.write_text(json.dumps(raw_document(case)), encoding="utf-8")
            path.chmod(0o600)
            with mock.patch.object(producer, "validate_payload_shapes") as validate:
                observed = producer.read_raw_output_bytes(
                    path.read_bytes(), "a" * 64, "b" * 40, "cuda:0", case
                )
            validate.assert_called_once_with(observed)
            self.assertEqual(
                tuple(item.key for item in observed), producer.MEASUREMENTS
            )

    def test_reviewed_tensor_shapes_bind_the_22_to_7_round_trip(self) -> None:
        expected = producer.reviewed_payload_shapes()
        payloads = tuple(
            producer.TensorPayload(key, expected[key], dtype, b"")
            for key, dtype in producer.MEASUREMENT_DTYPES.items()
        )
        producer.validate_payload_shapes(payloads)
        def base64_size(item: producer.TensorPayload) -> int:
            raw_bytes = math.prod(item.shape) * (
                2 if item.dtype == "float16" else 4
            )
            return 4 * ((raw_bytes + 2) // 3)

        base64_bytes = sum(base64_size(item) for item in payloads)
        self.assertGreater(base64_bytes, 64 * 1024 * 1024)
        self.assertLess(base64_bytes + 1024 * 1024, producer.MAXIMUM_RAW_OUTPUT_BYTES)
        changed = list(payloads)
        changed[1] = producer.TensorPayload(
            changed[1].key,
            (1, 48, 2, 20, 20),
            changed[1].dtype,
            b"",
        )
        with self.assertRaisesRegex(producer.CaptureFailure, "reviewed visual case"):
            producer.validate_payload_shapes(changed)

    def test_raw_output_rejects_bf16_visual_evidence(self) -> None:
        case = producer.build_case()
        raw = raw_document(case)
        raw["cases"][0]["tensors"][0]["dtype"] = "bfloat16"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(producer.CaptureFailure, "dtype drifted"):
                producer.read_raw_output_bytes(
                    path.read_bytes(), "a" * 64, "b" * 40, "cuda:0", case
                )

    def test_raw_output_rejects_wrong_decoder_precision(self) -> None:
        case = producer.build_case()
        raw = raw_document(case)
        raw["decoder_compute_dtype"] = "float32"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(producer.CaptureFailure, "requested authority"):
                producer.read_raw_output_bytes(
                    path.read_bytes(), "a" * 64, "b" * 40, "cuda:0", case
                )

    def test_raw_output_bytes_are_independent_from_later_path_replacement(self) -> None:
        case = producer.build_case()
        original = json.dumps(raw_document(case)).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.json"
            path.write_bytes(original)
            path.chmod(0o600)
            snapshot, retained = producer.QWEN.SealedFileSnapshot.seal_and_read(
                path, producer.MAXIMUM_RAW_OUTPUT_BYTES
            )
            path.unlink()
            path.write_text('{"replaced":true}', encoding="utf-8")
            path.chmod(0o400)
            with mock.patch.object(producer, "validate_payload_shapes"):
                observed = producer.read_raw_output_bytes(
                    retained, "a" * 64, "b" * 40, "cuda:0", case
                )
            self.assertEqual(tuple(item.key for item in observed), producer.MEASUREMENTS)
            with self.assertRaises(Exception):
                snapshot.revalidate()

    def test_raw_output_rejects_truncated_and_nonfinite_payloads(self) -> None:
        with self.assertRaisesRegex(producer.CaptureFailure, "byte count"):
            producer.decode_raw_tensor(
                {
                    "key": "pixel-normalization",
                    "shape": [2],
                    "dtype": "float32",
                    "little_endian_base64": base64.b64encode(
                        struct.pack("<f", 1.0)
                    ).decode(),
                },
            )
        with self.assertRaisesRegex(producer.CaptureFailure, "non-finite"):
            producer.decode_raw_tensor(
                {
                    "key": "pixel-normalization",
                    "shape": [1],
                    "dtype": "float32",
                    "little_endian_base64": base64.b64encode(
                        struct.pack("<f", float("nan"))
                    ).decode(),
                },
            )

    def test_request_is_create_new_and_sealed_owner_readonly(self) -> None:
        case = producer.build_case()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "request.json"
            snapshot = producer.write_request(path, "a" * 64, "b" * 40, case)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            snapshot.revalidate()
            with self.assertRaisesRegex(Exception, "already exists"):
                producer.write_request(path, "a" * 64, "b" * 40, case)

    def test_missing_environment_fails_before_model_or_runtime_access(self) -> None:
        with mock.patch.object(producer.BASE, "load_capture_runtime") as runtime:
            with self.assertRaisesRegex(
                producer.CaptureFailure, "MOLD_H3_FIXTURE_ROOT"
            ):
                producer.run_capture({}, "oracle", "cuda:0")
            runtime.assert_not_called()

    def test_adapter_hash_binds_rust_and_numerical_implementation(self) -> None:
        oracle = producer.adapter_implementation_sha256("oracle")
        mold = producer.adapter_implementation_sha256("mold")
        self.assertRegex(oracle, r"^[0-9a-f]{64}$")
        self.assertRegex(mold, r"^[0-9a-f]{64}$")
        self.assertNotEqual(oracle, mold)

    def test_rust_adapter_is_cuda_private_and_diffusers_only(self) -> None:
        source = RUST_ADAPTER_PATH.read_text(encoding="utf-8")
        cargo = (REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml").read_text(
            encoding="utf-8"
        )
        block = cargo.split('name = "h3_visual_vae_capture"', 1)[1].split("\n\n", 1)[0]
        self.assertIn(
            'required-features = ["dev-bins", "h3-private-uat", "cuda"]', block
        )
        self.assertIn("validate_diffusers_weight_index", source)
        self.assertIn("DecodeComputePolicy::Reference", source)
        self.assertIn("VisualAttentionBackend::Math", source)
        self.assertIn("Device::new_cuda", source)
        self.assertNotIn("validate_comfy_weight_file", source)
        self.assertNotIn("OfficialComfySource", source)

    def test_private_evidence_path_delegates_to_production_posterior_recipe(
        self,
    ) -> None:
        condition = (
            REPO_ROOT
            / "crates"
            / "mold-candle"
            / "src"
            / "minimax_h3"
            / "visual_condition.rs"
        ).read_text(encoding="utf-8")
        vae = (
            REPO_ROOT
            / "crates"
            / "mold-candle"
            / "src"
            / "minimax_h3"
            / "visual_vae.rs"
        ).read_text(encoding="utf-8")
        self.assertIn("self.official_seed42_sample()?.roundtrip_f32", condition)
        self.assertIn("official_seed42_capture", vae)
        self.assertIn("capture_condition_evidence_unit", vae)
        self.assertIn('#[cfg(feature = "h3-private-uat")]', vae)

    def test_seed42_noise_uses_a_fresh_cpu_generator_before_device_copy(self) -> None:
        condition = (
            REPO_ROOT
            / "crates"
            / "mold-candle"
            / "src"
            / "minimax_h3"
            / "visual_condition.rs"
        ).read_text(encoding="utf-8")
        self.assertIn("let noise = fresh_seed42_noise(self.mean.shape())?;", condition)
        self.assertIn("noise.to_device(self.mean.device())", condition)
        self.assertIn("to_dtype(DType::F16)?", condition)
        self.assertLess(
            condition.index("to_dtype(DType::F16)?"),
            condition.index("normalize_latents(&sample.roundtrip_f32"),
        )

    def test_seam_probe_is_complete_and_bounded(self) -> None:
        source = RUST_ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn("for frame in SEAM_FRAMES", source)
        self.assertIn("for seam in [64usize, 256]", source)
        self.assertIn("let probe_axis = [0usize, 63, 64, 255, 256, 319]", source)
        expected = len(producer.SEAM_FRAMES) * 3 * len(producer.SEAMS) * 6 * 4
        self.assertEqual(expected, 432)

    def test_release_verifier_rejects_visual_capture_marker(self) -> None:
        verifier = REPO_ROOT / "scripts" / "verify-h3-release-exclusion.sh"
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "candidate"
            binary.write_bytes(
                b"mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted\n"
                + producer.CAPTURE_MARKER.encode()
            )
            result = producer.subprocess.run(
                [str(verifier), str(binary)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_checked_tree_contains_no_model_or_capture_payload(self) -> None:
        forbidden_suffixes = {".safetensors", ".ckpt", ".pt", ".pth", ".mp4", ".mov"}
        tracked = producer.subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        offenders = [
            path
            for path in tracked
            if pathlib.Path(path).suffix.lower() in forbidden_suffixes
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
