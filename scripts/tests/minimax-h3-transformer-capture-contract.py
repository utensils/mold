#!/usr/bin/env python3
"""Adversarial contract tests for the paired private H3 transformer producer."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-transformer.py"
RUST_ADAPTER_PATH = (
    REPO_ROOT / "crates" / "mold-inference" / "src" / "bin" / "h3_transformer_capture.rs"
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


producer = load_module("minimax_h3_transformer_capture_contract", PRODUCER_PATH)
conformance = load_module("minimax_h3_transformer_contract_conformance", producer.CONFORMANCE_PATH)
gpu = load_module("minimax_h3_transformer_contract_gpu", producer.GPU_PATH)


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def payload(key: str, shape: tuple[int, ...], dtype: str) -> producer.TensorPayload:
    width = 2 if dtype == "bfloat16" else 4
    return producer.TensorPayload(key, shape, dtype, bytes(math.prod(shape) * width))


def raw_payloads() -> tuple[producer.TensorPayload, ...]:
    return (
        payload("token-refiner", (1, 2, 5376), "bfloat16"),
        payload("normalized-q", (1, 6, 56, 128), "bfloat16"),
        payload("normalized-k", (1, 6, 56, 128), "bfloat16"),
        payload("rotated-q", (1, 6, 56, 128), "bfloat16"),
        payload("rotated-k", (1, 6, 56, 128), "bfloat16"),
        payload("adaln", (6, 32256), "bfloat16"),
        payload("block-output", (1, 6, 5376), "bfloat16"),
        payload("video-head", (1, 2, 96), "float32"),
        payload("audio-head", (1, 2, 32), "float32"),
    )


def raw_document(task: str, revision: str = "b" * 40) -> dict:
    component = "transformer" if task == "fl2va" else "transformer_ref"
    return {
        "schema_version": producer.RAW_SCHEMA,
        "claim_marker": producer.CAPTURE_MARKER,
        "task": task,
        "component": component,
        "device": "cuda:0",
        "source_revision": revision,
        "authorization_source_sha256": producer.AUTHORIZATION_SHA256,
        "input_sha256": producer.build_case(task)["input_sha256"],
        "authority_set_sha256": producer.AUTHORITY_SET_SHA256,
        "tensors": [
            {
                "key": item.key,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "little_endian_base64": base64.b64encode(item.encoded).decode("ascii"),
            }
            for item in raw_payloads()
        ],
    }


class TransformerCaptureContract(unittest.TestCase):
    def test_task_cases_are_distinct_and_content_addressed(self) -> None:
        fl2va = producer.build_case("fl2va")
        ref2va = producer.build_case("ref2va")
        self.assertEqual(fl2va["adaln_boundary"], "post-silu-bfloat16")
        self.assertNotEqual(fl2va["input_sha256"], ref2va["input_sha256"])
        changed = copy.deepcopy(fl2va)
        changed["positions"][0][0] = 1
        changed.pop("input_sha256")
        observed = producer.sha256_bytes(
            producer.INPUT_DOMAIN + producer.BASE.canonical_json_bytes(changed)
        )
        self.assertNotEqual(observed, fl2va["input_sha256"])

    def test_manifest_authority_is_exactly_all_two_task_artifacts(self) -> None:
        identifiers = producer.component_ids(manifest())
        self.assertEqual(len(identifiers), 32)
        self.assertEqual(len([item for item in identifiers if item.startswith("official-fl2va")]), 16)
        self.assertEqual(len([item for item in identifiers if item.startswith("official-ref2va")]), 16)
        self.assertEqual(
            producer.BASE.component_authority_set_sha256(
                identifiers, producer.BASE.manifest_component_map(manifest())
            ),
            producer.AUTHORITY_SET_SHA256,
        )

    def test_manifest_pins_exact_producer_and_adapter_pins_exact_manifest(self) -> None:
        reviewed = manifest()
        producer_sha256 = producer.BASE.sha256_file(PRODUCER_PATH)
        for layer in producer.LAYERS:
            contract = next(item for item in reviewed["fixture_layers"] if item["id"] == layer)
            self.assertEqual(contract["oracle_adapter"]["implementation_sha256"], producer_sha256)
        manifest_sha256 = producer.BASE.sha256_file(MANIFEST_PATH)
        self.assertIn(manifest_sha256, RUST_ADAPTER_PATH.read_text(encoding="utf-8"))

    def test_manifest_hash_drift_is_rejected(self) -> None:
        changed = manifest()
        component = next(
            item
            for item in changed["component_indexes"]
            if item["id"] == "official-ref2va-transformer-shard-00014-of-00014"
        )
        component["sha256"] = "0" * 64
        with self.assertRaisesRegex(producer.CaptureFailure, "authority set"):
            producer.component_ids(changed)

    def test_authenticated_snapshot_reads_and_rehashes_retained_descriptor(self) -> None:
        encoded = b"reviewed-transformer-component"
        relative = "transformer/config.json"
        reviewed = {
            "component_indexes": [
                {
                    "id": "component",
                    "relative_path": relative,
                    "sha256": producer.sha256_bytes(encoded),
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            component = root / relative
            component.parent.mkdir(parents=True)
            component.write_bytes(encoded)
            metadata = root / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{producer.MODEL_REVISION}\nreviewed\n", encoding="utf-8")
            with mock.patch.object(producer, "component_ids", return_value=("component",)):
                with producer.authenticate_transformer_snapshot(root, reviewed) as snapshot:
                    self.assertEqual(snapshot.retained_bytes(relative, 1024), encoded)
                    component.write_bytes(b"tampered-transformer-component")
                    with self.assertRaisesRegex(producer.CaptureFailure, "changed during execution"):
                        snapshot.revalidate()

    def test_raw_output_is_bound_to_task_source_authorization_and_input(self) -> None:
        revision = "b" * 40
        encoded = producer.document_bytes(raw_document("fl2va", revision))
        observed = producer.read_raw_output(encoded, "fl2va", "cuda:0", revision)
        self.assertEqual(tuple(item.key for item in observed), producer.RAW_KEYS)
        for key, value in (
            ("task", "ref2va"),
            ("source_revision", "c" * 40),
            ("authorization_source_sha256", "0" * 64),
            ("input_sha256", "0" * 64),
        ):
            changed = raw_document("fl2va", revision)
            changed[key] = value
            with self.assertRaisesRegex(producer.CaptureFailure, "requested authority"):
                producer.read_raw_output(
                    producer.document_bytes(changed), "fl2va", "cuda:0", revision
                )

    def test_raw_output_rejects_reordered_or_nonfinite_tensors(self) -> None:
        revision = "b" * 40
        changed = raw_document("fl2va", revision)
        changed["tensors"] = list(reversed(changed["tensors"]))
        with self.assertRaisesRegex(producer.CaptureFailure, "order drifted"):
            producer.read_raw_output(
                producer.document_bytes(changed), "fl2va", "cuda:0", revision
            )
        changed = raw_document("fl2va", revision)
        changed["tensors"][0]["shape"] = [1, 1, 10752]
        with self.assertRaisesRegex(producer.CaptureFailure, "shape or byte count"):
            producer.read_raw_output(
                producer.document_bytes(changed), "fl2va", "cuda:0", revision
            )
        changed = raw_document("fl2va", revision)
        changed["tensors"][-1]["little_endian_base64"] = base64.b64encode(
            struct.pack("<f", float("nan")) * 64
        ).decode("ascii")
        with self.assertRaisesRegex(producer.CaptureFailure, "non-finite"):
            producer.read_raw_output(
                producer.document_bytes(changed), "fl2va", "cuda:0", revision
            )

    def test_partial_rope_retains_leading_q_and_k_axes(self) -> None:
        raw = list(raw_payloads())
        shape = (1, 1, 1, 128)
        q = b"".join(struct.pack("<H", value) for value in range(128))
        k = b"".join(struct.pack("<H", value + 128) for value in range(128))
        raw[3] = producer.TensorPayload("rotated-q", shape, "bfloat16", q)
        raw[4] = producer.TensorPayload("rotated-k", shape, "bfloat16", k)
        partial = producer.measurement_payloads(raw, "transformer-block")[1]
        self.assertEqual(partial.shape, (2, 1, 1, 1, 96))
        self.assertEqual(partial.encoded, q[:192] + k[:192])

    def test_documents_pass_the_protected_validator_for_both_roles_and_tasks(self) -> None:
        reviewed = manifest()
        contracts = gpu.exact_layer_contracts(reviewed)
        for role in ("oracle", "mold"):
            for task in producer.TASKS:
                revision = producer.DIFFUSERS_REVISION if role == "oracle" else "b" * 40
                for layer in producer.LAYERS:
                    document = producer.build_layer_document(
                        reviewed, role, task, layer, revision,
                        producer.AUTHORIZATION_SHA256, "cuda:0", raw_payloads(),
                    )
                    conformance.validate_schema(document, conformance.LAYER_OUTPUT_SCHEMA_PATH)
                    gpu.validate_manifest_layer_evidence(document, contracts[layer], role)

    def test_bundle_tolerance_matches_each_first_measurement_policy(self) -> None:
        reviewed = manifest()
        documents = []
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for layer in producer.LAYERS:
                document = producer.build_layer_document(
                    reviewed, "oracle", "fl2va", layer, producer.DIFFUSERS_REVISION,
                    producer.AUTHORIZATION_SHA256, "cuda:0", raw_payloads(),
                )
                path = root / f"{layer}.json"
                documents.append((path, document, producer.document_bytes(document)))
            bundle = producer.build_bundle(
                reviewed, "oracle", producer.DIFFUSERS_REVISION,
                producer.AUTHORIZATION_SHA256, "cuda:0", root, documents,
            )
            tolerances = {
                fixture["layer"]: fixture["tolerance"]["absolute"]
                for fixture in bundle["fixtures"]
            }
            self.assertEqual(tolerances, {"token-refiner": 0, "transformer-block": 0.015625})

    def test_mold_uses_staged_manifest_and_selective_safetensor_reads(self) -> None:
        source = PRODUCER_PATH.read_text(encoding="utf-8")
        self.assertIn('staged_mold / "tests" / "fixtures"', source)
        self.assertIn("safetensors.safe_open", source)
        self.assertNotIn("safetensors_torch.load_file", source)

    def test_private_adapter_binds_task_and_converts_fused_ffn_order(self) -> None:
        source = RUST_ADAPTER_PATH.read_text(encoding="utf-8")
        for required in (
            "--source-revision",
            "authorization_source_sha256",
            "input_sha256",
            "Tensor::cat(&[gate, value], 0)",
            "sha256_open_file(&component.file)",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
