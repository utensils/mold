#!/usr/bin/env python3
"""Adversarial contract tests for the exact-FP32 AudioVAE capture leaf."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import pathlib
import stat
import struct
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-audio-vae.py"
CONFORMANCE_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
RUST_PATH = (
    REPO_ROOT / "crates" / "mold-inference" / "src" / "bin" / "h3_audio_vae_capture.rs"
)
CARGO_PATH = REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml"
RELEASE_VERIFIER_PATH = REPO_ROOT / "scripts" / "verify-h3-release-exclusion.sh"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


PRODUCER = load_module("minimax_h3_audio_vae_producer_contract", PRODUCER_PATH)
CONFORMANCE = load_module("minimax_h3_audio_vae_conformance_contract", CONFORMANCE_PATH)
GPU = load_module("minimax_h3_audio_vae_gpu_contract", GPU_PATH)


def captured_fixture() -> dict[str, object]:
    case = PRODUCER.build_case()
    inputs = PRODUCER.decode_f32_payload(
        case["waveform_f32_le_base64"],
        [PRODUCER.BATCH, PRODUCER.CHANNELS, PRODUCER.INPUT_SAMPLES],
        "contract waveform",
    )
    decoded: list[float] = []
    for channel in range(PRODUCER.BATCH * PRODUCER.CHANNELS):
        start = channel * PRODUCER.INPUT_SAMPLES
        decoded.extend(inputs[start : start + PRODUCER.INPUT_SAMPLES])
        decoded.extend([0.0] * (PRODUCER.PADDED_SAMPLES - PRODUCER.INPUT_SAMPLES))
    latent_count = (
        PRODUCER.BATCH
        * PRODUCER.LATENT_CHANNELS
        * PRODUCER.CHANNELS
        * PRODUCER.LATENT_ROWS
    )
    latents = [
        struct.unpack("<f", struct.pack("<f", ((index % 31) - 15) / 64.0))[0]
        for index in range(latent_count)
    ]
    return {
        "case_id": case["case_id"],
        "input_sha256": case["input_sha256"],
        "input_shape": [PRODUCER.BATCH, PRODUCER.CHANNELS, PRODUCER.INPUT_SAMPLES],
        "packed_mono_shape": [
            PRODUCER.BATCH * PRODUCER.CHANNELS,
            1,
            PRODUCER.INPUT_SAMPLES,
        ],
        "normalized_latent_shape": [
            PRODUCER.BATCH,
            PRODUCER.LATENT_CHANNELS,
            PRODUCER.CHANNELS,
            PRODUCER.LATENT_ROWS,
        ],
        "normalized_latents": latents,
        "decoded_pcm_shape": [
            PRODUCER.BATCH,
            PRODUCER.CHANNELS,
            PRODUCER.PADDED_SAMPLES,
        ],
        "decoded_pcm": decoded,
        "encode_phases": list(PRODUCER.EXPECTED_ENCODE_PHASES),
        "decode_phases": list(PRODUCER.EXPECTED_DECODE_PHASES),
    }


def raw_output(case: dict[str, object]) -> dict[str, object]:
    captured = captured_fixture()

    def encoded(values: list[float]) -> str:
        return base64.b64encode(
            b"".join(struct.pack("<f", value) for value in values)
        ).decode("ascii")

    return {
        "schema_version": PRODUCER.RAW_OUTPUT_SCHEMA,
        "claim_marker": PRODUCER.CAPTURE_MARKER,
        "authorization_document_sha256": PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
        "source_revision": "a" * 40,
        "model_revision": PRODUCER.MODEL_REVISION,
        "config_sha256": PRODUCER.OFFICIAL_AUDIO_CONFIG_SHA256,
        "checkpoint_sha256": PRODUCER.COMFY_AUDIO_CHECKPOINT_SHA256,
        "device": "cuda:0",
        "dtype": PRODUCER.CANONICAL_DTYPE,
        "tensor_layout": "comfy-folded-weight-norm",
        "sample_rate": PRODUCER.SAMPLE_RATE,
        "samples_per_latent": PRODUCER.SAMPLES_PER_LATENT,
        "latent_rows_per_second": PRODUCER.LATENT_ROWS_PER_SECOND,
        "encode_phases": list(PRODUCER.EXPECTED_ENCODE_PHASES),
        "decode_phases": list(PRODUCER.EXPECTED_DECODE_PHASES),
        "cases": [
            {
                "case_id": case["case_id"],
                "input_sha256": case["input_sha256"],
                "input_shape": captured["input_shape"],
                "packed_mono_shape": captured["packed_mono_shape"],
                "normalized_latent_shape": captured["normalized_latent_shape"],
                "normalized_latents_f32_le_base64": encoded(
                    captured["normalized_latents"]
                ),
                "decoded_pcm_shape": captured["decoded_pcm_shape"],
                "decoded_pcm_f32_le_base64": encoded(captured["decoded_pcm"]),
            }
        ],
    }


class AudioVaeCaptureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = CONFORMANCE.validate_manifest()
        cls.contract = GPU.exact_layer_contracts(cls.manifest)[PRODUCER.LAYER_ID]

    def layer_document(self, role: str = "oracle") -> dict[str, object]:
        return PRODUCER.build_layer_document(
            self.manifest,
            role,
            PRODUCER.DIFFUSERS_REVISION if role == "oracle" else "a" * 40,
            PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "cuda:0",
            PRODUCER.build_case(),
            captured_fixture(),
            {"runtime": f"contract-{role}", "execution": "float32"},
            implementation_sha256=("b" if role == "oracle" else "c") * 64,
        )

    def test_manifest_boundary_is_exact_fp32_and_complete(self) -> None:
        layer = PRODUCER.layer_contract(self.manifest)
        self.assertEqual(layer["authority_tier"], "exact-full-fp32")
        self.assertEqual(layer["required_component_indexes"], [PRODUCER.COMPONENT_ID])
        self.assertEqual(
            tuple(layer["required_measurements"]), PRODUCER.MEASUREMENT_ORDER
        )
        self.assertEqual(
            GPU.MEASUREMENT_DTYPES[PRODUCER.LAYER_ID],
            {
                "stereo-packing": "int64",
                "latent-rate": "float64",
                "waveform-statistics": "float64",
                "phase-polarity": "float64",
                "decoded-pcm": "float32",
                "sampled-values": "float32",
            },
        )
        expected_checkpoints = {
            "oracle": PRODUCER.OFFICIAL_AUDIO_CHECKPOINT_SHA256,
            "mold": PRODUCER.COMFY_AUDIO_CHECKPOINT_SHA256,
        }
        self.assertEqual(
            CONFORMANCE.AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE,
            expected_checkpoints,
        )
        self.assertEqual(
            GPU.AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE,
            expected_checkpoints,
        )

    def test_oracle_and_mold_documents_satisfy_strict_schema(self) -> None:
        for role in ("oracle", "mold"):
            document = self.layer_document(role)
            GPU.validate_manifest_layer_evidence(document, self.contract, role)
            CONFORMANCE.validate_schema(document, CONFORMANCE.LAYER_OUTPUT_SCHEMA_PATH)
            self.assertEqual(
                [record["key"] for record in document["outputs"]],
                list(PRODUCER.MEASUREMENT_ORDER),
            )
            self.assertEqual(document["environment"]["dtype"], "float32")
            self.assertEqual(
                next(
                    item["value"]
                    for item in document["provenance"]
                    if item["key"] == "runtime-sha256"
                ),
                PRODUCER.sha256_json(document["environment"]["runtime"]),
            )

    def test_deterministic_input_binds_every_field_and_channel_order(self) -> None:
        case = PRODUCER.build_case()
        shape, encoded = PRODUCER.deterministic_waveform()
        self.assertEqual(shape, [2, 2, PRODUCER.INPUT_SAMPLES])
        self.assertEqual(case["input_sha256"], PRODUCER.case_identity(case))
        values = [item[0] for item in struct.iter_unpack("<f", encoded)]
        channel = PRODUCER.INPUT_SAMPLES
        self.assertEqual(values[2 * channel + 521], -values[3 * channel + 521])
        for key in ("sample_rate", "batch", "channels", "samples"):
            changed = copy.deepcopy(case)
            changed[key] += 1
            self.assertNotEqual(PRODUCER.case_identity(changed), case["input_sha256"])
        changed = copy.deepcopy(case)
        payload = bytearray(base64.b64decode(changed["waveform_f32_le_base64"]))
        payload[17] ^= 1
        changed["waveform_f32_le_base64"] = base64.b64encode(payload).decode("ascii")
        self.assertNotEqual(PRODUCER.case_identity(changed), case["input_sha256"])

    def test_measurements_retain_packing_timeline_pcm_and_phase(self) -> None:
        outputs = PRODUCER.measurement_outputs(
            PRODUCER.build_case(), captured_fixture()
        )
        records = {record["key"]: record for record in outputs}
        packing = records["stereo-packing"]
        self.assertEqual(
            [sample["value"] for sample in packing["samples"]][-12:],
            [0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 1, 3],
        )
        timeline = [sample["value"] for sample in records["latent-rate"]["samples"]]
        self.assertEqual(
            timeline, [32_000.0, 800.0, 40.0, 2_401.0, 3_200.0, 4.0, 3_200.0]
        )
        self.assertEqual(len(records["decoded-pcm"]["samples"]), 2 * 2 * 3_200)
        self.assertEqual(len(records["sampled-values"]["samples"]), 2 * 32 * 2 * 4)
        phase = [sample["value"] for sample in records["phase-polarity"]["samples"]]
        self.assertEqual(phase[-1], -1.0)
        self.assertAlmostEqual(phase[-2], -1.0, places=6)

    def test_raw_output_rejects_artifact_phase_and_payload_drift(self) -> None:
        case = PRODUCER.build_case()
        mutations = []
        wrong_checkpoint = raw_output(case)
        wrong_checkpoint["checkpoint_sha256"] = "0" * 64
        mutations.append(wrong_checkpoint)
        wrong_phase = raw_output(case)
        wrong_phase["decode_phases"].reverse()
        mutations.append(wrong_phase)
        wrong_shape = raw_output(case)
        wrong_shape["cases"][0]["packed_mono_shape"] = [2, 2, PRODUCER.INPUT_SAMPLES]
        mutations.append(wrong_shape)
        wrong_payload = raw_output(case)
        wrong_payload["cases"][0]["decoded_pcm_f32_le_base64"] = base64.b64encode(
            struct.pack("<f", 0.0)
        ).decode("ascii")
        mutations.append(wrong_payload)
        with tempfile.TemporaryDirectory() as temporary:
            for index, mutation in enumerate(mutations):
                path = pathlib.Path(temporary) / f"raw-{index}.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                path.chmod(0o600)
                with self.assertRaises(PRODUCER.CaptureFailure):
                    PRODUCER.read_raw_output_bytes(
                        path.read_bytes(),
                        PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                        "a" * 40,
                        "cuda:0",
                        case,
                    )

    def test_raw_output_parses_retained_bytes_and_detects_path_replacement(self) -> None:
        case = PRODUCER.build_case()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "raw.json"
            path.write_text(json.dumps(raw_output(case)), encoding="utf-8")
            path.chmod(0o600)
            snapshot, encoded = PRODUCER.seal_and_read_raw_output(path)
            retained = path.with_suffix(".retained")
            path.rename(retained)
            path.write_text("{}", encoding="utf-8")
            path.chmod(0o400)
            PRODUCER.read_raw_output_bytes(
                encoded,
                PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                "a" * 40,
                "cuda:0",
                case,
            )
            with self.assertRaises(PRODUCER.CaptureFailure):
                snapshot.revalidate()

    def test_fp32_numeric_domain_and_layer_dtype_fail_closed(self) -> None:
        document = self.layer_document("oracle")
        wrong_dtype = copy.deepcopy(document)
        wrong_dtype["environment"]["dtype"] = "bfloat16"
        next(item for item in wrong_dtype["provenance"] if item["key"] == "dtype")[
            "value"
        ] = "bfloat16"
        with self.assertRaises(GPU.GpuConformanceFailure):
            GPU.validate_manifest_layer_evidence(wrong_dtype, self.contract, "oracle")
        non_f32 = copy.deepcopy(document)
        decoded = next(
            item for item in non_f32["outputs"] if item["key"] == "decoded-pcm"
        )
        decoded["samples"][0]["value"] = 0.1
        with self.assertRaises(GPU.GpuConformanceFailure):
            GPU.validate_manifest_layer_evidence(non_f32, self.contract, "oracle")

    def test_hash_provenance_requires_structured_checkpoint_adapter_and_runtime(
        self,
    ) -> None:
        mutations = []
        missing_checkpoint = self.layer_document("oracle")
        del missing_checkpoint["input"]["checkpoint_sha256"]
        mutations.append(missing_checkpoint)
        missing_adapter = self.layer_document("oracle")
        del missing_adapter["adapter"]["implementation_sha256"]
        mutations.append(missing_adapter)
        empty_runtime = self.layer_document("oracle")
        empty_runtime["environment"]["runtime"] = {}
        mutations.append(empty_runtime)
        for mutation in mutations:
            with self.assertRaisesRegex(
                GPU.GpuConformanceFailure,
                "lacks canonical structured evidence|checkpoint authority is not exact",
            ):
                GPU.validate_manifest_layer_evidence(mutation, self.contract, "oracle")

    def test_record_only_hashes_allow_toleranced_cross_framework_evidence(self) -> None:
        oracle = self.layer_document("oracle")
        mold = self.layer_document("mold")
        floating = next(
            item for item in mold["outputs"] if item["key"] == "decoded-pcm"
        )
        floating["content_sha256"] = "d" * 64
        fixture = {"tolerance": {"absolute": 0, "relative": 0, "metric": "x"}}
        GPU.validate_oracle_mold_policy_parity(
            oracle, fixture, mold, fixture, self.contract
        )
        exact = next(
            item for item in mold["outputs"] if item["key"] == "stereo-packing"
        )
        exact["content_sha256"] = "e" * 64
        with self.assertRaises(GPU.GpuConformanceFailure):
            GPU.validate_oracle_mold_policy_parity(
                oracle, fixture, mold, fixture, self.contract
            )

        weakened_timeline = self.layer_document("oracle")
        next(
            item
            for item in weakened_timeline["comparison"]
            if item["key"] == "latent-rate"
        )["hash_policy"] = "record-only"
        with self.assertRaisesRegex(
            GPU.GpuConformanceFailure, "bounded protected policy"
        ):
            GPU.validate_manifest_layer_evidence(
                weakened_timeline, self.contract, "oracle"
            )

    def test_role_specific_checkpoint_authorities_compare_and_fail_closed(self) -> None:
        oracle = self.layer_document("oracle")
        mold = self.layer_document("mold")
        CONFORMANCE.compare_layer_outputs(oracle, mold)

        wrong_checkpoint = copy.deepcopy(mold)
        wrong_checkpoint["input"]["checkpoint_sha256"] = "0" * 64
        next(
            item
            for item in wrong_checkpoint["provenance"]
            if item["key"] == "checkpoint-sha256"
        )["value"] = "0" * 64
        with self.assertRaisesRegex(
            CONFORMANCE.ConformanceFailure,
            "checkpoint authority differs",
        ):
            CONFORMANCE.compare_layer_outputs(oracle, wrong_checkpoint)
        with self.assertRaisesRegex(
            GPU.GpuConformanceFailure,
            "checkpoint authority is not exact",
        ):
            GPU.validate_manifest_layer_evidence(
                wrong_checkpoint, self.contract, "mold"
            )

    def test_environment_and_repository_local_fixture_fail_closed(self) -> None:
        with self.assertRaises(PRODUCER.CaptureFailure):
            PRODUCER.required_environment({})
        with self.assertRaises(PRODUCER.BASE.CaptureFailure):
            PRODUCER.BASE.external_directory("fixture root", str(REPO_ROOT))

    def test_shared_capture_dependency_is_hash_bound_and_in_adapter_identity(
        self,
    ) -> None:
        self.assertEqual(
            PRODUCER.BASE.sha256_file(PRODUCER.QWEN_PRODUCER_PATH),
            PRODUCER.QWEN_PRODUCER_SHA256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = pathlib.Path(temporary) / "changed.py"
            changed.write_text("CHANGED = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reviewed capture dependency"):
                PRODUCER.load_bound_module(
                    "minimax_h3_audio_changed_dependency",
                    changed,
                    PRODUCER.QWEN_PRODUCER_SHA256,
                )
        source = PRODUCER_PATH.read_text(encoding="utf-8")
        self.assertIn('"qwen_capture_dependency_sha256": QWEN_PRODUCER_SHA256', source)

    def test_exclusive_evidence_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "evidence.json"
            PRODUCER.BASE.exclusive_write(path, b"first\n")
            with self.assertRaises(PRODUCER.BASE.CaptureFailure):
                PRODUCER.BASE.exclusive_write(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_rust_adapter_is_private_dense_fp32_cuda_only(self) -> None:
        source = RUST_PATH.read_text(encoding="utf-8")
        cargo = CARGO_PATH.read_text(encoding="utf-8")
        self.assertIn("load_validated_audio_vae", source)
        self.assertIn("AudioTensorLayout::ComfyFolded", source)
        self.assertIn("DType::F32", source)
        self.assertIn("Device::new_cuda", source)
        self.assertIn("encode_with_phase_checkpoint", source)
        self.assertIn("decode_with_phase_checkpoint", source)
        self.assertIn("CheckpointSnapshot::authenticate", source)
        self.assertIn("request_snapshot.revalidate", source)
        self.assertIn(PRODUCER.CAPTURE_MARKER, source)
        audio_bin = cargo.split('name = "h3_audio_vae_capture"', 1)[1].split("\n\n", 1)[
            0
        ]
        self.assertIn('required-features = ["dev-bins", "h3-private-uat"]', audio_bin)
        self.assertNotIn("model_registry", source)

    def test_release_verifier_rejects_audio_capture_marker(self) -> None:
        verifier = RELEASE_VERIFIER_PATH.read_text(encoding="utf-8")
        self.assertIn(PRODUCER.CAPTURE_MARKER, verifier)


if __name__ == "__main__":
    unittest.main(verbosity=2)
