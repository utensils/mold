#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import struct
import tempfile
import time
import unittest
import zlib
import binascii

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

    def test_two_rtx3090_profile_is_exact(self):
        uuids = [
            "GPU-44f80ce5-23fc-a5dd-ac4e-133142952997",
            "GPU-ba027fc5-7915-8d58-6738-b7eaafe427b4",
        ]
        devices = [
            {
                "index": index,
                "uuid": value,
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_mib": 24576,
                "compute_capability": "8.6",
                "driver_version": "999.0",
            }
            for index, value in enumerate(uuids)
        ]
        runner.validate_hardware_profile(devices, uuids)
        devices[1]["name"] = "NVIDIA B200"
        with self.assertRaisesRegex(ValueError, "RTX 3090"):
            runner.validate_hardware_profile(devices, uuids)
        devices[1]["name"] = "NVIDIA GeForce RTX 3090"
        devices[1]["memory_total_mib"] = 12288
        with self.assertRaisesRegex(ValueError, "24576"):
            runner.validate_hardware_profile(devices, uuids)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            runner.validate_hardware_profile(devices[:1], uuids[:1])

    def test_reserved_service_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "7680"):
            runner.validate_qualification_port(7680)
        runner.validate_qualification_port(17681)

    def test_sandbox_environment_is_allowlisted_and_home_stays_read_only(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = pathlib.Path(raw)
            inherited = {
                "PATH": "/usr/bin",
                "LD_LIBRARY_PATH": "/cuda",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
                "SSH_AUTH_SOCK": "/run/user/1000/ssh",
                "MOLD_HOST": "http://127.0.0.1:7680",
                "HOME": "/home/real",
            }
            env = runner.sandbox_environment(runtime, inherited=inherited)
            self.assertEqual(env["PATH"], "/usr/bin")
            self.assertEqual(env["LD_LIBRARY_PATH"], "/cuda")
            self.assertEqual(env["HOME"], "/home/real")
            self.assertNotIn("DBUS_SESSION_BUS_ADDRESS", env)
            self.assertNotIn("SSH_AUTH_SOCK", env)
            self.assertNotIn("MOLD_HOST", env)

    def test_replan_requires_same_queued_work_to_move(self):
        before = {
            "w1": "cuda:a",
            "w2": "cuda:a",
        }
        after = {
            "w1": "cuda:b",
            "w2": "cuda:b",
        }
        self.assertEqual(
            runner.prove_replanned_work(before, after, "cuda:a"),
            ["w1", "w2"],
        )
        with self.assertRaisesRegex(ValueError, "same queued work"):
            runner.prove_replanned_work(before, {}, "cuda:a")
        with self.assertRaisesRegex(ValueError, "draining device"):
            runner.prove_replanned_work(before, {"w1": "cuda:a", "w2": "cuda:b"}, "cuda:a")

    def test_png_validation_rejects_non_image_and_accepts_exact_dimensions(self):
        def chunk(kind: bytes, payload: bytes) -> bytes:
            crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

        width, height = 1, 1
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + chunk(b"IEND", b"")
        )
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "out.png"
            path.write_bytes(png)
            self.assertEqual(
                runner.validate_png_output(path, width, height),
                {"decoded": True, "width": 1, "height": 1},
            )
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "PNG"):
                runner.validate_png_output(path, width, height)

    def test_candidate_process_commands_share_the_bwrap_helper(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = pathlib.Path(raw)
            command = runner.sandbox_command(runtime, ["/candidate/mold", "version"])
            self.assertEqual(command[0], "bwrap")
            self.assertIn("--ro-bind", command)
            self.assertEqual(command[-2:], ["/candidate/mold", "version"])

    def test_process_group_timeout_is_bounded(self):
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            runner.run_process_group(
                ["/bin/sh", "-c", "sleep 30"],
                env={"PATH": "/usr/bin:/bin"},
                deadline=runner.Deadline(0.05),
            )
        self.assertLess(time.monotonic() - started, 2.0)


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

    def valid_fixture(self, root: pathlib.Path) -> pathlib.Path:
        report_path = root / "passing.json"
        evidence_root = pathlib.Path(str(report_path) + ".d")
        evidence_root.mkdir()
        models_dir = root / "models"
        models_dir.mkdir()
        artifact = models_dir / "model.safetensors"
        artifact.write_bytes(b"exact model")
        binary = root / "mold"
        binary.write_bytes(b"candidate")
        uuids = [
            "GPU-44f80ce5-23fc-a5dd-ac4e-133142952997",
            "GPU-ba027fc5-7915-8d58-6738-b7eaafe427b4",
        ]
        devices = [
            {
                "index": index,
                "uuid": value,
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_mib": 24576,
                "compute_capability": "8.6",
                "driver_version": "999.0",
            }
            for index, value in enumerate(uuids)
        ]
        api_devices = [
            {
                "id": f"cuda:{index}",
                "ordinal": index,
                "nvml_uuid": value,
                "name": "NVIDIA GeForce RTX 3090",
                "desired_enabled": True,
                "admin_state": "enabled",
                "schedulable": True,
            }
            for index, value in enumerate(uuids)
        ]
        evidence_items = []

        def add(label: str, value, kind: str = "json") -> pathlib.Path:
            suffix = {"json": ".json", "jsonl": ".jsonl", "text": ".txt"}[kind]
            path = evidence_root / f"{label}{suffix}"
            if kind == "json":
                path.write_text(json.dumps(value), encoding="utf-8")
            elif kind == "jsonl":
                path.write_text(
                    "\n".join(json.dumps(row) for row in value) + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(value, encoding="utf-8")
            evidence_items.append(
                {
                    "label": label,
                    "path": str(path),
                    "sha256": validator.sha256(path),
                    "kind": kind,
                }
            )
            return path

        def png_bytes() -> bytes:
            def chunk(kind: bytes, payload: bytes) -> bytes:
                crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
                return (
                    struct.pack(">I", len(payload))
                    + kind
                    + payload
                    + struct.pack(">I", crc)
                )

            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
                + chunk(b"IEND", b"")
            )

        request = {
            "prompt": "qualification",
            "model": "sd15:fp16",
            "width": 1,
            "height": 1,
            "seed": 10,
            "batch_size": 1,
            "output_format": "png",
        }
        request_path = add("normalized-request", request)
        artifacts = [
            {
                "path": str(artifact),
                "sha256": validator.sha256(artifact),
                "size": artifact.stat().st_size,
            }
        ]
        versioned = {"evidence_schema_version": validator.EVIDENCE_SCHEMA_VERSION}
        add(
            "model-artifacts",
            {**versioned, "model": "sd15:fp16", "artifacts": artifacts},
        )
        add(
            "source-provenance",
            {**versioned, "commit": "a" * 40, "source_root": str(root)},
        )
        add(
            "candidate-version",
            {
                **versioned,
                "binary": str(binary),
                "binary_sha256": validator.sha256(binary),
                "version": "mold fixture",
                "sandboxed": True,
                "argv": [
                    "bwrap",
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--",
                    str(binary),
                    "version",
                ],
            },
        )
        inventory = "\n".join(
            f"{row['index']}, {row['uuid']}, {row['name']}, "
            f"{row['memory_total_mib']}, {row['compute_capability']}, "
            f"{row['driver_version']}"
            for row in devices
        )
        add("nvidia-inventory", inventory + "\n", "text")
        initial = {
            "devices": {"devices": api_devices},
            "status": {
                "hostname": "fixture-host",
                "gpus": [
                    {"ordinal": 0, "name": "NVIDIA GeForce RTX 3090"},
                    {"ordinal": 1, "name": "NVIDIA GeForce RTX 3090"},
                ]
            },
            "resources": {
                "gpus": [
                    {"ordinal": 0, "name": "NVIDIA GeForce RTX 3090"},
                    {"ordinal": 1, "name": "NVIDIA GeForce RTX 3090"},
                ]
            },
            "queue": {"plan": {}},
            "models": [{"name": "sd15:fp16", "downloaded": True}],
            "capabilities": {
                "devices": {
                    "available": True,
                    "lifecycle": True,
                    "planned_lanes": True,
                },
                "dispatch": {"v2_authoritative": True, "active_mode": "v2"},
            },
        }
        add("initial-api-projection", initial)
        add(
            "client-projection",
            {
                **versioned,
                "server_pid": 123,
                "gpu_list": {
                    "devices": [{"id": "cuda:0"}, {"id": "cuda:1"}]
                },
                "stderr": "",
                "argv": [
                    "bwrap",
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--",
                    str(binary),
                    "gpu",
                    "list",
                    "--json",
                ],
            },
        )
        add("parallel-runtime-samples", [{"server_pid": 123}], "jsonl")
        output_results = []
        for index in range(1, 5):
            output = evidence_root / f"parallel-output-{index}.png"
            output.write_bytes(png_bytes())
            evidence_items.append(
                {
                    "label": f"parallel-output-{index}",
                    "path": str(output),
                    "sha256": validator.sha256(output),
                    "kind": "png",
                }
            )
            ordinal = [0, 1, 1, 1][index - 1]
            output_results.append(
                {
                    "index": index,
                    "prompt": f"qualification variation {index}",
                    "seed": 9 + index,
                    "status": 200,
                    "content_type": "image/png",
                    "gpu_ordinal": ordinal,
                    "gpu_uuid": uuids[ordinal],
                    "work_id": f"w{index}",
                    "decoded": {"decoded": True, "width": 1, "height": 1},
                    "path": str(output),
                    "size": output.stat().st_size,
                    "sha256": validator.sha256(output),
                    "headers": {},
                }
            )
        add(
            "parallel-results",
            {
                **versioned,
                "server_pid": 123,
                "results": output_results,
                "observed_active_uuids": uuids,
                "observed_compute_uuids": uuids,
                "decisive_samples": [
                    {
                        "server_pid": 123,
                        "active": [
                            {
                                "device_id": "cuda:0",
                                "gpu_uuid": uuids[0],
                                "work_id": "w1",
                            },
                            {
                                "device_id": "cuda:1",
                                "gpu_uuid": uuids[1],
                                "work_id": "w2",
                            },
                        ],
                        "compute_apps": [
                            {"pid": 123, "gpu_uuid": uuids[0]},
                            {"pid": 123, "gpu_uuid": uuids[1]},
                        ],
                    }
                ],
                "execution_bindings": {
                    "w1": uuids[0],
                    "w2": uuids[1],
                    "w3": uuids[1],
                    "w4": uuids[1],
                },
                "disabled_id": "cuda:0",
                "disable_response": {
                    "status": 202,
                    "body": {"admin_state": "draining"},
                },
                "drained": {"admin_state": "disabled"},
                "reenabled": {"admin_state": "enabled"},
                "pre_disable_assignments": {
                    "w3": "cuda:0",
                    "w4": "cuda:0",
                },
                "post_disable_assignments": {
                    "w3": "cuda:1",
                    "w4": "cuda:1",
                },
                "replanned_work_ids": ["w3", "w4"],
            },
        )
        add(
            "queued-cancellation",
            {
                **versioned,
                "server_pid": 123,
                "job_id": "job-cancel",
                "work_id": "work-cancel",
                "cancel_status": 204,
                "resume_status": 200,
                "stream_http_status": 200,
                "typed_cancelled": True,
                "never_active": True,
                "output_tree_unchanged": True,
                "queue_was_paused": True,
                "stream_tail": 'event: error\ndata: {"error":"cancelled"}\n\n',
                "queue_before_cancel": {
                    "paused": True,
                    "entries": [{"id": "job-cancel", "state": "queued"}],
                    "plan": {
                        "work_items": [
                            {
                                "parent_id": "job-cancel",
                                "work_id": "work-cancel",
                                "activity_phase": "queued",
                            }
                        ]
                    },
                },
                "devices_before_cancel": {
                    "devices": [
                        {"active_work_id": None},
                        {"active_work_id": None},
                    ]
                },
                "queue_after": {"entries": [], "plan": {"work_items": []}},
                "devices_after": {
                    "devices": [
                        {"active_work_id": None},
                        {"active_work_id": None},
                    ]
                },
                "output_tree_before": [],
                "output_tree_after": [],
            },
        )
        add(
            "all-disabled-maintenance",
            {
                **versioned,
                "server_pid": 123,
                "status": 503,
                "body": "maintenance",
                "devices": {
                    "devices": [
                        {"admin_state": "disabled"},
                        {"admin_state": "disabled"},
                    ]
                },
            },
        )
        mapping = {"cuda:0": uuids[0], "cuda:1": uuids[1]}
        add(
            "restart-persistence",
            {
                **versioned,
                "old_pid": 123,
                "new_pid": 124,
                "disabled_id": "cuda:0",
                "disabled_uuid": uuids[0],
                "before_mapping": mapping,
                "after_mapping": mapping,
                "persisted_devices": {
                    "devices": [
                        {"id": "cuda:0", "desired_enabled": False},
                        {"id": "cuda:1", "desired_enabled": True},
                    ]
                },
                "restored_devices": {
                    "devices": [
                        {"admin_state": "enabled"},
                        {"admin_state": "enabled"},
                    ]
                },
            },
        )
        add(
            "legacy-rollback",
            {
                **versioned,
                "server_pid": 125,
                "device_mapping": mapping,
                "patch_status": 409,
                "patch_body": {},
                "snapshot": {
                    "capabilities": {
                        "dispatch": {
                            "active_mode": "legacy",
                            "v2_authoritative": False,
                        },
                        "devices": {"lifecycle": False},
                    }
                },
            },
        )
        selector_labels = [
            "empty",
            "all",
            "ordinal-one",
            "stable-id",
            "nvidia-uuid",
            "none",
            "reordered-stable",
            "missing",
        ]
        add(
            "selector-matrix",
            {
                **versioned,
                "scenarios": [
                    (
                        {
                            "label": label,
                            "exit_code": 1,
                            "argv": [
                                "bwrap",
                                "--die-with-parent",
                                "--ro-bind",
                                "/",
                                "/",
                                "--",
                                str(binary),
                                "serve",
                            ],
                        }
                        if label == "missing"
                        else {
                            "label": label,
                            "pid": 200 + index,
                            "argv": [
                                "bwrap",
                                "--die-with-parent",
                                "--ro-bind",
                                "/",
                                "/",
                                "--",
                                str(binary),
                                "serve",
                            ],
                        }
                    )
                    for index, label in enumerate(selector_labels)
                ],
            },
        )
        add(
            "ambiguous-selector-source-contract",
            {
                **versioned,
                "exit_code": 0,
                "hardware_claimed": False,
                "argv": [
                    "bwrap",
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--",
                    "/bin/bash",
                    "-lc",
                    "true",
                ],
            },
        )
        add("models-tree-before", [])
        add("models-tree-after", [])
        for label in ("primary-command", "restart-command", "legacy-command"):
            add(
                label,
                {
                    "argv": [
                        "bwrap",
                        "--die-with-parent",
                        "--ro-bind",
                        "/",
                        "/",
                        "--",
                        str(binary),
                        "serve",
                    ],
                    "environment": {},
                },
            )
        add("restart-server-log", "restart\n", "text")
        add("legacy-server-log", "legacy\n", "text")
        checks = {
            name: {
                "status": "passed",
                "summary": f"{name} passed",
                "evidence_labels": sorted(validator.CHECK_EVIDENCE[name]),
            }
            for name in validator.REQUIRED_CHECKS
        }
        report = {
            "schema_version": validator.SCHEMA_VERSION,
            "qualification_profile": validator.QUALIFICATION_PROFILE,
            "source_commit": "a" * 40,
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
                "devices": devices,
            },
            "isolation": {
                "bind": "127.0.0.1",
                "port": 17681,
                "mold_home": str(evidence_root / "home"),
                "db_path": str(evidence_root / "mold.db"),
                "output_dir": str(evidence_root / "output"),
                "models_dir": str(models_dir),
                "preexisting_listener_absent": True,
            },
            "request": {
                "path": str(request_path),
                "sha256": validator.sha256(request_path),
                "model": "sd15:fp16",
                "job_count": 4,
                "artifacts": artifacts,
            },
            "checks": checks,
            "evidence": evidence_items,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

    def test_fabricated_passing_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            report = self.fixture(root)
            with self.assertRaisesRegex(
                ValueError, "mandatory|request|qualification_profile"
            ):
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

    def test_typed_passing_fixture_validates_and_raw_tampering_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.valid_fixture(root)
            validator.validate(path, require_passing=True)
            output = pathlib.Path(str(path) + ".d") / "parallel-output-1.png"
            output.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validator.validate(path, require_passing=True)

    def test_rehashed_semantic_tampering_still_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.valid_fixture(root)
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "parallel-results"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["results"][0]["gpu_uuid"] = payload["results"][1]["gpu_uuid"]
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordinal|UUID"):
                validator.validate(path, require_passing=True)

    def test_rehashed_cancellation_claim_without_typed_event_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.valid_fixture(root)
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "queued-cancellation"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["typed_cancelled"] = False
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cancellation"):
                validator.validate(path, require_passing=True)

    def test_rehashed_cancellation_claim_with_active_work_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.valid_fixture(root)
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "queued-cancellation"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["queue_before_cancel"]["plan"]["work_items"][0][
                "activity_phase"
            ] = "active"
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "queued|active|cancellation"):
                validator.validate(path, require_passing=True)

    def test_rehashed_selector_without_read_only_sandbox_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.valid_fixture(root)
            report = json.loads(path.read_text())
            item = next(
                row for row in report["evidence"] if row["label"] == "selector-matrix"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["scenarios"][0]["argv"] = [
                "bwrap",
                "--die-with-parent",
                "--",
                report["candidate"]["path"],
                "serve",
            ]
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sandbox|Bubblewrap|read-only"):
                validator.validate(path, require_passing=True)


if __name__ == "__main__":
    unittest.main()
