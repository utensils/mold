#!/usr/bin/env python3
"""Adversarial contract tests for the exact-BF16 Qwen layer-50 producer."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import stat
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


sys.dont_write_bytecode = True
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-qwen-layer50.py"
CONFORMANCE_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
RUST_PATH = (
    REPO_ROOT
    / "crates"
    / "mold-inference"
    / "src"
    / "bin"
    / "h3_qwen_layer50_capture.rs"
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


PRODUCER = load_module("minimax_h3_qwen_layer50_producer_contract", PRODUCER_PATH)
CONFORMANCE = load_module(
    "minimax_h3_qwen_layer50_conformance_contract", CONFORMANCE_PATH
)
GPU = load_module("minimax_h3_qwen_layer50_gpu_contract", GPU_PATH)


def synthetic_case(case_id: str, multimodal: bool = False) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": case_id,
        "kind": "multimodal" if multimodal else "text-only",
        "token_ids": [11, 12, 13],
        "position_ids": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        "image": None,
        "video": None,
    }
    if multimodal:
        encoded = base64.b64encode(b"\x00\x00" * 4).decode("ascii")
        case["image"] = {
            "shape": [2, 2],
            "grid_thw": [1, 2, 2],
            "bfloat16_le_base64": encoded,
        }
        case["video"] = {
            "shape": [2, 2],
            "grid_thw": [1, 2, 2],
            "bfloat16_le_base64": encoded,
        }
    case["input_sha256"] = PRODUCER.case_identity(case)
    return case


def activation(sequence: int = 3) -> tuple[list[int], list[int]]:
    shape = [1, sequence, 5120]
    values = [0x3F80 + (index % 8) for index in range(sequence * 5120)]
    return shape, values


class ProducerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = CONFORMANCE.validate_manifest()
        cls.contract = GPU.exact_layer_contracts(cls.manifest)[PRODUCER.LAYER_ID]

    def layer_document(self, role: str = "oracle") -> dict[str, object]:
        revision = PRODUCER.DIFFUSERS_REVISION if role == "oracle" else "a" * 40
        return PRODUCER.build_layer_document(
            self.manifest,
            role,
            revision,
            PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "cuda:0",
            synthetic_case(PRODUCER.TEXT_CASE_ID),
            activation(),
        )

    def test_manifest_boundary_is_exact_and_ordered(self) -> None:
        layer = PRODUCER.layer_contract(self.manifest)
        self.assertEqual(layer["authority_tier"], "exact-full-bf16")
        self.assertEqual(
            layer["required_component_indexes"],
            list(PRODUCER.TEXT_ENCODER_COMPONENT_IDS),
        )
        self.assertEqual(
            layer["required_measurements"],
            ["shape", "dtype", "statistics", "sampled-values", "tolerance"],
        )

    def test_oracle_and_mold_documents_satisfy_strict_layer_schema(self) -> None:
        oracle = self.layer_document("oracle")
        mold = self.layer_document("mold")
        GPU.validate_manifest_layer_evidence(oracle, self.contract, "oracle")
        GPU.validate_manifest_layer_evidence(mold, self.contract, "mold")
        CONFORMANCE.validate_schema(oracle, CONFORMANCE.LAYER_OUTPUT_SCHEMA_PATH)
        CONFORMANCE.validate_schema(mold, CONFORMANCE.LAYER_OUTPUT_SCHEMA_PATH)
        CONFORMANCE.validate_layer_output(oracle, "oracle")
        CONFORMANCE.validate_layer_output(mold, "mold")
        self.assertEqual(
            oracle["input"]["component_index_sha256"],
            CONFORMANCE.component_authority_set_sha256(
                oracle["input"]["component_indexes"]
            ),
        )
        self.assertIn("comparison", oracle)
        self.assertNotIn("comparison", mold)
        self.assertEqual(
            [record["key"] for record in oracle["outputs"]],
            self.contract["required_measurements"],
        )

    def test_role_invariant_provenance_matches(self) -> None:
        oracle = self.layer_document("oracle")
        mold = self.layer_document("mold")
        for key in self.contract["role_invariant_provenance"]:
            oracle_value = next(
                item["value"] for item in oracle["provenance"] if item["key"] == key
            )
            mold_value = next(
                item["value"] for item in mold["provenance"] if item["key"] == key
            )
            self.assertEqual(oracle_value, mold_value)

    def test_case_identity_binds_every_prepared_field(self) -> None:
        original = synthetic_case(PRODUCER.MULTIMODAL_CASE_ID, multimodal=True)
        original_hash = original["input_sha256"]
        mutations = []
        changed_token = copy.deepcopy(original)
        changed_token["token_ids"][0] += 1
        mutations.append(changed_token)
        changed_position = copy.deepcopy(original)
        changed_position["position_ids"][1][1] += 1
        mutations.append(changed_position)
        changed_pixel = copy.deepcopy(original)
        changed_pixel["image"]["bfloat16_le_base64"] = base64.b64encode(
            b"\x01\x00" + b"\x00\x00" * 3
        ).decode("ascii")
        mutations.append(changed_pixel)
        changed_grid = copy.deepcopy(original)
        changed_grid["video"]["grid_thw"][0] += 1
        mutations.append(changed_grid)
        changed_kind = copy.deepcopy(original)
        changed_kind["kind"] = "text-only"
        mutations.append(changed_kind)
        for mutated in mutations:
            self.assertNotEqual(PRODUCER.case_identity(mutated), original_hash)

    def test_position_builder_matches_text_and_split_video_runs(self) -> None:
        self.assertEqual(
            PRODUCER.qwen_positions([0, 0, 0], [], []),
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )
        positions = PRODUCER.qwen_positions(
            [0, 2, 2, 2, 2, 0, 2, 2, 2, 2], [], [[2, 4, 4]]
        )
        self.assertEqual(len(positions), 3)
        self.assertTrue(all(len(axis) == 10 for axis in positions))

    def test_position_builder_rejects_grid_run_mismatch(self) -> None:
        with self.assertRaises(PRODUCER.CaptureFailure):
            PRODUCER.qwen_positions([0, 1, 1], [[1, 4, 4]], [])

    def test_activation_sampling_is_complete_and_ordered(self) -> None:
        indexes = PRODUCER.activation_sample_indexes(10_000)
        self.assertEqual(indexes, list(range(10_000)))

    def test_activation_evidence_retains_real_tensor_coordinates(self) -> None:
        shape, bits = activation()
        outputs = PRODUCER.measurement_outputs(shape, bits)
        sampled = next(
            record for record in outputs if record["key"] == "sampled-values"
        )
        self.assertEqual(sampled["shape"], shape)
        self.assertEqual(len(sampled["samples"]), len(bits))
        self.assertEqual(sampled["samples"][0]["index"], [0, 0, 0])
        self.assertEqual(sampled["samples"][-1]["index"], [0, 2, 5119])
        self.assertTrue(all(len(sample["index"]) == 3 for sample in sampled["samples"]))

    def test_protected_policy_rejects_incomplete_activation_coordinates(self) -> None:
        document = self.layer_document("oracle")
        sampled = next(
            record for record in document["outputs"] if record["key"] == "sampled-values"
        )
        sampled["samples"].pop()
        with self.assertRaises(GPU.GpuConformanceFailure):
            GPU.validate_manifest_layer_evidence(document, self.contract, "oracle")

    def test_measurement_output_rejects_shape_payload_drift(self) -> None:
        with self.assertRaises(PRODUCER.CaptureFailure):
            PRODUCER.measurement_outputs([1, 3, 5120], [0x3F80] * 4)

    def test_bundle_is_strict_schema_and_contains_both_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            documents = []
            for case_id in PRODUCER.CASE_IDS:
                case = synthetic_case(
                    case_id, multimodal=case_id == PRODUCER.MULTIMODAL_CASE_ID
                )
                document = PRODUCER.build_layer_document(
                    self.manifest,
                    "oracle",
                    PRODUCER.DIFFUSERS_REVISION,
                    PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                    "cuda:0",
                    case,
                    activation(),
                )
                path = root / f"{case_id}.json"
                documents.append((path, document, PRODUCER.document_bytes(document)))
            bundle = PRODUCER.build_bundle(
                self.manifest,
                "oracle",
                PRODUCER.DIFFUSERS_REVISION,
                PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                "cuda:0",
                root,
                documents,
            )
            CONFORMANCE.validate_schema(bundle, CONFORMANCE.BUNDLE_SCHEMA_PATH)
            self.assertEqual(len(bundle["fixtures"]), 2)
            self.assertEqual(
                [fixture["layer"] for fixture in bundle["fixtures"]],
                [PRODUCER.LAYER_ID, PRODUCER.LAYER_ID],
            )
            for fixture, (_, document, _) in zip(
                bundle["fixtures"], documents, strict=True
            ):
                self.assertEqual(
                    fixture["component_index_sha256"],
                    document["input"]["component_index_sha256"],
                )
                self.assertEqual(
                    fixture["component_index_sha256"],
                    CONFORMANCE.component_authority_set_sha256(
                        fixture["component_indexes"]
                    ),
                )

    def test_raw_output_rejects_authority_and_input_drift(self) -> None:
        case = synthetic_case(PRODUCER.TEXT_CASE_ID)
        shape, bits = activation()
        encoded = base64.b64encode(
            b"".join(int(value).to_bytes(2, "little") for value in bits)
        ).decode("ascii")
        output = {
            "schema_version": PRODUCER.RAW_OUTPUT_SCHEMA,
            "claim_marker": PRODUCER.CAPTURE_MARKER,
            "authorization_document_sha256": PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "source_revision": "a" * 40,
            "model_revision": PRODUCER.MODEL_REVISION,
            "device": "cuda:0",
            "dtype": PRODUCER.CANONICAL_DTYPE,
            "peak_resident_language_layers": 1,
            "cases": [
                {
                    "case_id": case["case_id"],
                    "input_sha256": "0" * 64,
                    "shape": shape,
                    "bfloat16_le_base64": encoded,
                }
            ],
        }
        with self.assertRaises(PRODUCER.CaptureFailure):
            PRODUCER.read_raw_output_bytes(
                json.dumps(output).encode("utf-8"),
                PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                "a" * 40,
                "cuda:0",
                [case],
            )

    def test_environment_and_repository_local_fixture_fail_closed(self) -> None:
        with self.assertRaises(PRODUCER.CaptureFailure):
            PRODUCER.required_environment({})
        with self.assertRaises(PRODUCER.BASE.CaptureFailure):
            PRODUCER.BASE.external_directory("fixture root", str(REPO_ROOT))

    def test_memory_preflight_rejects_low_host_or_device_memory(self) -> None:
        gib = 1024**3
        runtime = SimpleNamespace(
            torch=SimpleNamespace(
                cuda=SimpleNamespace(mem_get_info=mock.Mock(return_value=(16 * gib, 48 * gib)))
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = pathlib.Path(temporary) / "meminfo"
            meminfo.write_text(f"MemAvailable: {95 * 1024**2} kB\n", encoding="utf-8")
            with mock.patch.object(PRODUCER.BASE, "configure_torch", return_value="cuda:0"):
                with self.assertRaises(PRODUCER.CaptureFailure):
                    PRODUCER.memory_preflight(runtime, "cuda:0", meminfo)

                meminfo.write_text(
                    f"MemAvailable: {128 * 1024**2} kB\n", encoding="utf-8"
                )
                runtime.torch.cuda.mem_get_info.return_value = (11 * gib, 48 * gib)
                with self.assertRaises(PRODUCER.CaptureFailure):
                    PRODUCER.memory_preflight(runtime, "cuda:0", meminfo)

                runtime.torch.cuda.mem_get_info.return_value = (16 * gib, 48 * gib)
                PRODUCER.memory_preflight(runtime, "cuda:0", meminfo)

    def test_exclusive_evidence_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "evidence.json"
            PRODUCER.BASE.exclusive_write(path, b"first\n")
            with self.assertRaises(PRODUCER.BASE.CaptureFailure):
                PRODUCER.BASE.exclusive_write(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_staging_is_owner_only_immutable_and_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "stage"
            PRODUCER.private_directory(root)
            path = root / "source.py"
            PRODUCER.write_private_file(path, b"reviewed\n")
            snapshot = PRODUCER.seal_staging(root)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            snapshot.revalidate()
            path.chmod(0o600)
            path.write_bytes(b"changed\n")
            with self.assertRaises(PRODUCER.CaptureFailure):
                snapshot.revalidate()

    def test_shared_source_parent_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_root = PRODUCER.private_source_root(root)
            package = source_root / "diffusers"
            PRODUCER.write_private_file(package / "source.py", b"source\n")

            self.assertEqual(stat.S_IMODE(source_root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(package.stat().st_mode), 0o700)
            PRODUCER.snapshot_files(root)

    def test_coordinated_shard_and_sidecar_tamper_fails_against_manifest_pin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_root = pathlib.Path(temporary)
            identifier, relative, reviewed_sha256 = (
                PRODUCER.TEXT_ENCODER_SHARD_AUTHORITIES[0]
            )
            shard = model_root / relative
            shard.parent.mkdir(parents=True)
            tampered = b"coordinated replacement"
            shard.write_bytes(tampered)
            metadata = (
                model_root
                / ".cache"
                / "huggingface"
                / "download"
                / f"{relative}.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                f"{PRODUCER.MODEL_REVISION}\n{hashlib.sha256(tampered).hexdigest()}\n",
                encoding="utf-8",
            )
            with self.assertRaises(PRODUCER.CaptureFailure):
                PRODUCER.authenticate_text_encoder_shard(
                    model_root,
                    {
                        "id": identifier,
                        "relative_path": relative,
                        "sha256": reviewed_sha256,
                    },
                )

    def test_raw_output_parses_only_the_fd_sealed_bytes(self) -> None:
        case = synthetic_case(PRODUCER.TEXT_CASE_ID)
        shape, bits = activation()
        payload = base64.b64encode(
            b"".join(int(value).to_bytes(2, "little") for value in bits)
        ).decode("ascii")
        original = {
            "schema_version": PRODUCER.RAW_OUTPUT_SCHEMA,
            "claim_marker": PRODUCER.CAPTURE_MARKER,
            "authorization_document_sha256": PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
            "source_revision": "a" * 40,
            "model_revision": PRODUCER.MODEL_REVISION,
            "device": "cuda:0",
            "dtype": PRODUCER.CANONICAL_DTYPE,
            "peak_resident_language_layers": 1,
            "cases": [
                {
                    "case_id": case["case_id"],
                    "input_sha256": case["input_sha256"],
                    "shape": shape,
                    "bfloat16_le_base64": payload,
                }
            ],
        }
        forged = copy.deepcopy(original)
        forged["cases"][0]["input_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "raw.json"
            saved = root / "sealed.json"
            path.write_bytes(PRODUCER.document_bytes(original))
            snapshot, sealed_bytes = PRODUCER.SealedFileSnapshot.seal_and_read(
                path, PRODUCER.MAXIMUM_RAW_OUTPUT_BYTES
            )
            path.rename(saved)
            path.write_bytes(PRODUCER.document_bytes(forged))
            captured = PRODUCER.read_raw_output_bytes(
                sealed_bytes,
                PRODUCER.REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
                "a" * 40,
                "cuda:0",
                [case],
            )
            self.assertEqual(captured[case["case_id"]][0], shape)
            with self.assertRaises(PRODUCER.CaptureFailure):
                snapshot.revalidate()

    def test_rust_adapter_is_dense_private_cuda_only(self) -> None:
        source = RUST_PATH.read_text(encoding="utf-8")
        cargo = CARGO_PATH.read_text(encoding="utf-8")
        self.assertIn("load_streamed_bf16_conditioner", source)
        self.assertNotIn("load_h3_qwen_nvfp4_conditioner", source)
        self.assertNotIn("private_qwen", source)
        self.assertIn("profile.parameter_dtype != DType::BF16", source)
        self.assertIn("language_layer_count() != 50", source)
        self.assertIn("peak_resident_language_layers() != 1", source)
        self.assertIn('required-features = ["dev-bins", "h3-private-uat"]', cargo)
        self.assertIn("Device::new_cuda", source)
        self.assertIn("OpenedInputSnapshot", source)
        self.assertIn("path_snapshot.descriptor_path()", source)
        self.assertIn("sha256_open_file(&self.file)? != self.expected_sha256", source)
        self.assertIn('request_snapshot.revalidate("capture request")', source)
        self.assertIn("MAX_RAW_OUTPUT_BYTES", source)
        self.assertIn("OFFICIAL_TEXT_ENCODER_SHARDS", source)
        self.assertIn("checkpoint_components", source)
        self.assertIn("metadata_sha256 != expected_sha256", source)

    def test_release_verifier_rejects_capture_marker(self) -> None:
        verifier = RELEASE_VERIFIER_PATH.read_text(encoding="utf-8")
        self.assertIn(PRODUCER.CAPTURE_MARKER, verifier)

    def test_repository_snapshot_survives_a_tracked_symlink(self) -> None:
        # `git archive` of the Mold repository carries the tracked
        # `AGENTS.md -> CLAUDE.md` link, so rejecting every non-regular member
        # made the `--role mold` staging path unreachable.
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            payload = b"guidance\n"
            target = tarfile.TarInfo("CLAUDE.md")
            target.size = len(payload)
            archive.addfile(target, io.BytesIO(payload))
            link = tarfile.TarInfo("AGENTS.md")
            link.type = tarfile.SYMTYPE
            link.linkname = "CLAUDE.md"
            archive.addfile(link)
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / "snapshot"
            PRODUCER.extract_archive(buffer.getvalue(), destination)
            materialized = destination / "AGENTS.md"
            self.assertFalse(materialized.is_symlink())
            self.assertEqual(materialized.read_bytes(), b"guidance\n")

    def test_repository_snapshot_rejects_an_escaping_link(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            link = tarfile.TarInfo("AGENTS.md")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../etc/passwd"
            archive.addfile(link)
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / "snapshot"
            with self.assertRaises(PRODUCER.CaptureFailure):
                PRODUCER.extract_archive(buffer.getvalue(), destination)

    def test_repository_snapshot_still_rejects_a_device_member(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            node = tarfile.TarInfo("dev/null")
            node.type = tarfile.CHRTYPE
            archive.addfile(node)
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / "snapshot"
            with self.assertRaises(PRODUCER.CaptureFailure):
                PRODUCER.extract_archive(buffer.getvalue(), destination)

    def test_shard_payload_length_excludes_the_safetensors_header(self) -> None:
        # A Hugging Face shard index reports the summed tensor payload, while a
        # shard file also carries its 8-byte length prefix and JSON header, so
        # comparing whole-file sizes to the index total can never balance.
        header = b'{"__metadata__":{"format":"pt"}}'
        payload = b"\x00" * 512
        encoded = len(header).to_bytes(8, "little") + header + payload
        with tempfile.TemporaryDirectory() as directory:
            shard = pathlib.Path(directory) / "model-00001-of-00002.safetensors"
            shard.write_bytes(encoded)
            self.assertEqual(
                PRODUCER.safetensors_payload_length(shard, len(encoded)),
                len(payload),
            )

    def test_shard_payload_length_rejects_a_truncated_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = pathlib.Path(directory) / "model-00001-of-00002.safetensors"
            shard.write_bytes((1 << 20).to_bytes(8, "little") + b"{}")
            with self.assertRaises(PRODUCER.CaptureFailure):
                PRODUCER.safetensors_payload_length(shard, shard.stat().st_size)

    def test_producer_never_routes_oracle_through_quantized_loader(self) -> None:
        source = PRODUCER_PATH.read_text(encoding="utf-8")
        self.assertIn("get_qwen3vl_prompt_embeds", source)
        self.assertIn("text_encoder_layer=50", source)
        self.assertIn("torch_dtype=torch.bfloat16", source)
        self.assertNotIn("load_h3_qwen_nvfp4_conditioner", source)
        self.assertNotIn("comfy-pruned", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
