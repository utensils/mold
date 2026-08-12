#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import struct
import sys
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

    def test_artifact_anchor_expands_to_complete_companion_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            models = pathlib.Path(raw)
            model = models / "selected" / "model.safetensors"
            companion = models / "selected" / "vae.safetensors"
            model.parent.mkdir()
            model.write_bytes(b"model")
            companion.write_bytes(b"vae")
            artifacts, roots = runner.collect_model_artifacts(models, [model])
            self.assertEqual(
                {row["path"] for row in artifacts},
                {str(model), str(companion)},
            )
            self.assertEqual(roots, [str(model.parent)])

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

    def test_embedded_git_sha_must_be_exact_source_prefix(self):
        source = "b65deba7af3a38e46681dbe5c8e19ffb63f1e672"
        for embedded in ("b65deba7", source):
            self.assertTrue(runner.embedded_git_sha_matches_source(embedded, source))
            self.assertTrue(validator.embedded_git_sha_matches_source(embedded, source))
        for embedded in (
            "",
            "b65deb",
            "deadbeef",
            "B65DEBA7",
            source + "0",
            "unknown",
        ):
            self.assertFalse(runner.embedded_git_sha_matches_source(embedded, source))
            self.assertFalse(validator.embedded_git_sha_matches_source(embedded, source))

    def test_initial_queue_allows_idle_before_first_v2_plan(self):
        runner.validate_initial_queue({"entries": []})
        runner.validate_initial_queue(
            {"entries": [], "plan": {"plan_version": 0, "work_items": []}}
        )
        for invalid in (
            {},
            {"entries": "not-a-list"},
            {"entries": [{"id": "queued-without-plan"}]},
            {"entries": [], "plan": "not-an-object"},
        ):
            with self.assertRaisesRegex(ValueError, "queue|plan|entries"):
                runner.validate_initial_queue(invalid)

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
            self.assertIn("--unshare-pid", command)
            self.assertNotIn(["--dev-bind", "/dev", "/dev"], [
                command[index : index + 3] for index in range(len(command) - 2)
            ])
            self.assertEqual(command[-2:], ["/candidate/mold", "version"])

    def test_device_node_collection_excludes_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as raw:
            alias = pathlib.Path(raw) / "render-alias"
            alias.symlink_to("/dev/null")

            selected = runner.direct_char_device_nodes(
                [pathlib.Path("/dev/null"), alias]
            )

            self.assertEqual(selected, [pathlib.Path("/dev/null")])

    def test_process_group_timeout_is_bounded(self):
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            runner.run_process_group(
                # The sleeper is this interpreter by absolute path, not
                # `sh -c "sleep 30"`: a distribution without `/bin/sleep` or
                # `/usr/bin/sleep` (NixOS, and any non-FHS host) exits the
                # shell immediately with 127, so the deadline never expires
                # and the assertion below fails for a reason that has nothing
                # to do with the timeout it covers.
                [sys.executable, "-c", "import time; time.sleep(30)"],
                env={"PATH": "/usr/bin:/bin"},
                deadline=runner.Deadline(0.05),
            )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_file_reads_hashes_and_tree_walks_honor_expired_deadline(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = root / "payload"
            path.write_bytes(b"payload")
            deadline = runner.Deadline(0.001)
            time.sleep(0.01)
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                runner.read_file_text(path, deadline)
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                runner.models_tree_manifest(root, deadline)
            evidence = runner.Evidence(root / "evidence")
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                evidence.existing("payload", path, "text", deadline)
            self.assertFalse(evidence.contains("payload"))


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

    def synthetic_semantic_fixture(self, root: pathlib.Path) -> pathlib.Path:
        """Build invented validator input; this is never hardware evidence."""
        report_path = root / "passing.json"
        evidence_root = pathlib.Path(str(report_path) + ".d")
        evidence_root.mkdir()
        models_dir = root / "models"
        models_dir.mkdir()
        artifact = models_dir / "model.safetensors"
        artifact.write_bytes(b"exact model")
        binary = root / "mold"
        binary.write_bytes(b"candidate")
        inherited_home = "/home/synthetic"
        server_inner = [
            str(binary),
            "serve",
            "--bind",
            "127.0.0.1",
            "--port",
            "17681",
            "--models-dir",
            str(models_dir),
            "--queue-size",
            "64",
            "--log-format",
            "json",
        ]

        def sandbox_argv(
            inner: list[str], runtime: pathlib.Path | None = None
        ) -> list[str]:
            runtime = runtime or evidence_root / "primary-runtime"
            return [
                "bwrap",
                "--die-with-parent",
                "--unshare-user",
                "--disable-userns",
                "--unshare-pid",
                "--unshare-ipc",
                "--new-session",
                "--cap-drop",
                "ALL",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--bind",
                str(runtime),
                str(runtime),
                "--bind",
                str(runtime / "tmp"),
                "/tmp",
                "--",
                *inner,
            ]

        def isolated_environment(
            runtime: pathlib.Path, *, client: bool = False
        ) -> dict[str, str | None]:
            for relative in (
                "tmp",
                "cache/xdg",
                "cache/cuda",
                "cache/huggingface",
                "config",
                "data",
                "home",
                "output",
            ):
                (runtime / relative).mkdir(parents=True, exist_ok=True)
            (runtime / "mold.db").touch()
            return {
                "HOME": inherited_home,
                "TMPDIR": str(runtime / "tmp"),
                "XDG_CACHE_HOME": str(runtime / "cache" / "xdg"),
                "XDG_CONFIG_HOME": str(runtime / "config"),
                "XDG_DATA_HOME": str(runtime / "data"),
                "CUDA_CACHE_PATH": str(runtime / "cache" / "cuda"),
                "HF_HOME": str(runtime / "cache" / "huggingface"),
                "MOLD_HOME": str(runtime / "home"),
                "MOLD_DB_PATH": str(runtime / "mold.db"),
                "MOLD_OUTPUT_DIR": str(runtime / "output"),
                "MOLD_HOST": "http://127.0.0.1:17681" if client else None,
            }

        def process_identity(pid: int) -> dict[str, object]:
            binary_stat = binary.stat()
            return {
                "pid": pid,
                "start_ticks": 1000 + pid,
                "executable": str(binary),
                "executable_device": binary_stat.st_dev,
                "executable_inode": binary_stat.st_ino,
                "binary_sha256": validator.sha256(binary),
                "pid_namespace": "pid:[1234]",
                "nspid": [pid, 1],
            }
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
            {
                **versioned,
                "model": "sd15:fp16",
                "artifacts": artifacts,
                "artifact_roots": [str(models_dir)],
            },
        )
        add(
            "source-provenance",
            {**versioned, "commit": "a" * 40, "source_root": str(root)},
        )
        isolated_environment(evidence_root / "candidate-probe-runtime")
        add(
            "candidate-version",
            {
                **versioned,
                "binary": str(binary),
                "binary_sha256": validator.sha256(binary),
                "version": "mold fixture",
                "sandboxed": True,
                "argv": sandbox_argv(
                    [str(binary), "version"],
                    evidence_root / "candidate-probe-runtime",
                ),
                "environment": {
                    "HOME": inherited_home,
                    "TMPDIR": str(
                        evidence_root / "candidate-probe-runtime" / "tmp"
                    ),
                    "XDG_CACHE_HOME": str(
                        evidence_root
                        / "candidate-probe-runtime"
                        / "cache"
                        / "xdg"
                    ),
                    "XDG_CONFIG_HOME": str(
                        evidence_root / "candidate-probe-runtime" / "config"
                    ),
                    "XDG_DATA_HOME": str(
                        evidence_root / "candidate-probe-runtime" / "data"
                    ),
                    "MOLD_HOST": None,
                },
                "build_identity": {
                    "source_commit": "a" * 40,
                    "binary_sha256": validator.sha256(binary),
                    "version": "mold fixture",
                },
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
                "git_sha": "a" * 40,
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
                "candidate_identity": process_identity(123),
                "gpu_list": {
                    "devices": [{"id": "cuda:0"}, {"id": "cuda:1"}]
                },
                "stderr": "",
                "argv": sandbox_argv(
                    [str(binary), "gpu", "list", "--json"]
                ),
                "environment": isolated_environment(
                    evidence_root / "primary-runtime", client=True
                ),
            },
        )
        raw_samples = [
            {
                "at": "2026-07-28T00:00:10Z",
                "server_pid": 123,
                "candidate_identity": process_identity(123),
                "devices": [
                    {
                        **api_devices[0],
                        "activity": "running",
                        "active_work_id": "w1",
                    },
                    {
                        **api_devices[1],
                        "activity": "running",
                        "active_work_id": "w2",
                    },
                ],
                "queue": {
                    "plan": {
                        "plan_version": 1,
                        "work_items": [
                            {
                                "work_id": "w3",
                                "planned_device_id": "cuda:0",
                                "activity_phase": "queued",
                            },
                            {
                                "work_id": "w4",
                                "planned_device_id": "cuda:0",
                                "activity_phase": "queued",
                            },
                        ]
                    }
                },
                "compute_apps": [
                    {"pid": 123, "gpu_uuid": uuids[0]},
                    {"pid": 123, "gpu_uuid": uuids[1]},
                ],
                "compute_apps_raw": "",
                "reserved_service_connections": [],
            },
            {
                "at": "2026-07-28T00:00:20Z",
                "server_pid": 123,
                "candidate_identity": process_identity(123),
                "devices": [
                    {
                        **api_devices[0],
                        "activity": "idle",
                        "active_work_id": None,
                        "admin_state": "disabled",
                        "desired_enabled": False,
                        "schedulable": False,
                    },
                    {
                        **api_devices[1],
                        "activity": "running",
                        "active_work_id": "w3",
                    },
                ],
                "queue": {
                    "plan": {
                        "plan_version": 2,
                        "work_items": [
                            {
                                "work_id": "w3",
                                "planned_device_id": "cuda:1",
                                "activity_phase": "active",
                            },
                            {
                                "work_id": "w4",
                                "planned_device_id": "cuda:1",
                                "activity_phase": "queued",
                            },
                        ]
                    }
                },
                "compute_apps": [{"pid": 123, "gpu_uuid": uuids[1]}],
                "compute_apps_raw": "",
                "reserved_service_connections": [],
            },
            {
                "at": "2026-07-28T00:00:30Z",
                "server_pid": 123,
                "candidate_identity": process_identity(123),
                "devices": [
                    {
                        **api_devices[0],
                        "activity": "idle",
                        "active_work_id": None,
                        "admin_state": "disabled",
                        "desired_enabled": False,
                        "schedulable": False,
                    },
                    {
                        **api_devices[1],
                        "activity": "running",
                        "active_work_id": "w4",
                    },
                ],
                "queue": {
                    "plan": {
                        "plan_version": 2,
                        "work_items": [
                            {
                                "work_id": "w4",
                                "planned_device_id": "cuda:1",
                                "activity_phase": "active",
                            }
                        ]
                    }
                },
                "compute_apps": [{"pid": 123, "gpu_uuid": uuids[1]}],
                "compute_apps_raw": "",
                "reserved_service_connections": [],
            },
        ]
        add("parallel-runtime-samples", raw_samples, "jsonl")
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
                    "headers": {
                        "content-type": "image/png",
                        "x-mold-seed-used": str(9 + index),
                        "x-mold-gpu": str(ordinal),
                    },
                }
            )
        add(
            "parallel-results",
            {
                **versioned,
                "server_pid": 123,
                "candidate_identity": process_identity(123),
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
                "disabled_uuid": uuids[0],
                "disable_response": {
                    "status": 202,
                    "body": {
                        "id": "cuda:0",
                        "nvml_uuid": uuids[0],
                        "admin_state": "draining",
                    },
                },
                "drained": {
                    "id": "cuda:0",
                    "nvml_uuid": uuids[0],
                    "admin_state": "disabled",
                },
                "reenabled": {
                    "id": "cuda:0",
                    "nvml_uuid": uuids[0],
                    "admin_state": "enabled",
                },
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
                "pause_response": {"status": 200, "body": {"paused": True}},
                "cancel_status": 204,
                "resume_status": 200,
                "stream_http_status": 200,
                "typed_cancelled": True,
                "never_active": True,
                "output_tree_unchanged": True,
                "queue_was_paused": True,
                "stream_tail": 'event: error\ndata: {"error":"cancelled"}\n\n',
                "request": {
                    **request,
                    "prompt": "qualification queued cancellation",
                },
                "terminal_observed_at": "2026-07-28T00:00:40Z",
                "resume_at": "2026-07-28T00:00:41Z",
                "queue_before_cancel": {
                    "entries": [{"id": "job-cancel", "state": "queued"}],
                    "plan": {
                        "work_items": [
                            {
                                "parent_id": "job-cancel",
                                "work_id": "work-cancel",
                                "activity_phase": "blocked",
                                "blocked_reason": "queue_paused",
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
                "queue_after_cancel_before_resume": {
                    "paused": True,
                    "entries": [],
                    "plan": {"work_items": []},
                },
                "devices_after_cancel_before_resume": {
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
                        {
                            "id": "cuda:0",
                            "nvml_uuid": uuids[0],
                            "admin_state": "disabled",
                        },
                        {
                            "id": "cuda:1",
                            "nvml_uuid": uuids[1],
                            "admin_state": "disabled",
                        },
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
                "old_process_identity": process_identity(123),
                "new_process_identity": process_identity(124),
                "persisted_devices": {
                    "devices": [
                        {
                            "id": "cuda:0",
                            "nvml_uuid": uuids[0],
                            "desired_enabled": False,
                        },
                        {
                            "id": "cuda:1",
                            "nvml_uuid": uuids[1],
                            "desired_enabled": True,
                        },
                    ]
                },
                "restored_devices": {
                    "devices": [
                        {
                            "id": "cuda:0",
                            "nvml_uuid": uuids[0],
                            "desired_enabled": True,
                            "admin_state": "enabled",
                        },
                        {
                            "id": "cuda:1",
                            "nvml_uuid": uuids[1],
                            "desired_enabled": True,
                            "admin_state": "enabled",
                        },
                    ]
                },
            },
        )
        add(
            "legacy-rollback",
            {
                **versioned,
                "server_pid": 125,
                "process_identity": process_identity(125),
                "device_mapping": mapping,
                "patch_target": "cuda:0",
                "patch_status": 409,
                "patch_body": {
                    "code": "DEVICE_LIFECYCLE_MODE_CONFLICT",
                    "error": "disabling a live GPU requires an authoritative scheduler V2 runtime",
                },
                "snapshot": {
                    "capabilities": {
                        "dispatch": {
                            "active_mode": "legacy",
                            "v2_authoritative": False,
                        },
                        "devices": {"lifecycle": False},
                    },
                    "devices": {"devices": api_devices},
                },
            },
        )
        selector_specs = [
            ("empty", None, set(uuids)),
            ("all", "all", set(uuids)),
            ("ordinal-one", "1", {uuids[1]}),
            ("stable-id", "cuda:0", {uuids[0]}),
            ("nvidia-uuid", uuids[1], {uuids[1]}),
            ("none", "none", set()),
            ("reordered-stable", "cuda:0", {uuids[0]}),
        ]
        selector_scenarios = []
        for index, (label, selector_value, selected_uuids) in enumerate(
            selector_specs
        ):
            inner = list(server_inner)
            if selector_value is not None:
                inner.extend(["--gpus", selector_value])
            environment = isolated_environment(
                evidence_root / "selector-runtimes" / label
            )
            if label == "empty":
                environment["MOLD_GPUS"] = ""
            if label == "reordered-stable":
                environment["CUDA_VISIBLE_DEVICES"] = f"{uuids[1]},{uuids[0]}"
            selector_scenarios.append(
                {
                    "label": label,
                    "selector": selector_value,
                    "expected_uuids": sorted(selected_uuids),
                    "pid": 200 + index,
                    "process_identity": process_identity(200 + index),
                    "argv": sandbox_argv(
                        inner, evidence_root / "selector-runtimes" / label
                    ),
                    "environment": environment,
                    "devices": {
                        "devices": [
                            {
                                **device,
                                "admin_state": (
                                    "enabled"
                                    if device["nvml_uuid"] in selected_uuids
                                    else "startup_excluded"
                                ),
                            }
                            for device in api_devices
                        ]
                    },
                }
            )
        selector_scenarios.append(
            {
                "label": "missing",
                "selector": "GPU-ffffffff",
                "expected_uuids": [],
                "exit_code": 1,
                "expected_failure": "did not match",
                "argv": sandbox_argv(
                    [*server_inner, "--gpus", "GPU-ffffffff"],
                    evidence_root / "selector-runtimes" / "missing",
                ),
                "environment": isolated_environment(
                    evidence_root / "selector-runtimes" / "missing"
                ),
            }
        )
        add(
            "selector-matrix",
            {
                **versioned,
                "scenarios": selector_scenarios,
            },
        )
        add(
            "ambiguous-selector-source-contract",
            {
                **versioned,
                "exit_code": 0,
                "hardware_claimed": False,
                "command": "true",
                "cwd": str(root),
                "argv": sandbox_argv(
                    ["/bin/bash", "-lc", "true"],
                    evidence_root / "selector-contract-runtime",
                ),
                "environment": {
                    key: value
                    for key, value in isolated_environment(
                        evidence_root / "selector-contract-runtime"
                    ).items()
                    if key
                    in {
                        "HOME",
                        "TMPDIR",
                        "XDG_CACHE_HOME",
                        "XDG_CONFIG_HOME",
                        "XDG_DATA_HOME",
                        "MOLD_HOST",
                    }
                },
            },
        )
        add("models-tree-before", [])
        add("models-tree-after", [])
        for label in ("primary-command", "restart-command", "legacy-command"):
            add(
                label,
                {
                    "argv": sandbox_argv(server_inner),
                    "environment": isolated_environment(
                        evidence_root / "primary-runtime"
                    ),
                },
            )
        add("restart-server-log", "restart\n", "text")
        add("legacy-server-log", "legacy\n", "text")
        live_service = {
            "port": 7680,
            "listeners": [
                {
                    "family": "tcp",
                    "local_port": 7680,
                    "remote_port": 0,
                    "state": "0A",
                    "inode": "42",
                }
            ],
            "owners": [
                {
                    "pid": 42,
                    "start_ticks": 100,
                    "executable": "/synthetic/live-mold",
                    "socket_inodes": ["42"],
                }
            ],
        }
        add("live-service-before", live_service)
        add("live-service-after", live_service)
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
                "mold_home": str(evidence_root / "primary-runtime" / "home"),
                "db_path": str(evidence_root / "primary-runtime" / "mold.db"),
                "output_dir": str(evidence_root / "primary-runtime" / "output"),
                "models_dir": str(models_dir),
                "inherited_home": inherited_home,
                "preexisting_listener_absent": True,
            },
            "request": {
                "path": str(request_path),
                "sha256": validator.sha256(request_path),
                "model": "sd15:fp16",
                "job_count": 4,
                "artifacts": artifacts,
                "artifact_roots": [str(models_dir)],
            },
            "checks": checks,
            "evidence": evidence_items,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

    def mutate_evidence(self, report_path, label, mutate):
        report = json.loads(report_path.read_text())
        item = next(row for row in report["evidence"] if row["label"] == label)
        evidence_path = pathlib.Path(item["path"])
        if item["kind"] == "jsonl":
            payload = [
                json.loads(line)
                for line in evidence_path.read_text().splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(evidence_path.read_text())
        mutate(payload)
        if item["kind"] == "jsonl":
            evidence_path.write_text(
                "".join(json.dumps(row) + "\n" for row in payload),
                encoding="utf-8",
            )
        else:
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
        item["sha256"] = validator.sha256(evidence_path)
        report_path.write_text(json.dumps(report), encoding="utf-8")

    def relocate_runtime(self, payload, runtime: pathlib.Path) -> None:
        for relative in (
            "tmp",
            "cache/xdg",
            "cache/cuda",
            "cache/huggingface",
            "config",
            "data",
            "home",
            "output",
        ):
            (runtime / relative).mkdir(parents=True, exist_ok=True)
        (runtime / "mold.db").touch()
        argv = payload["argv"]
        first_bind = argv.index("--bind")
        argv[first_bind + 1] = str(runtime)
        argv[first_bind + 2] = str(runtime)
        second_bind = argv.index("--bind", first_bind + 1)
        argv[second_bind + 1] = str(runtime / "tmp")
        replacements = {
            "TMPDIR": runtime / "tmp",
            "XDG_CACHE_HOME": runtime / "cache" / "xdg",
            "XDG_CONFIG_HOME": runtime / "config",
            "XDG_DATA_HOME": runtime / "data",
            "CUDA_CACHE_PATH": runtime / "cache" / "cuda",
            "HF_HOME": runtime / "cache" / "huggingface",
            "MOLD_HOME": runtime / "home",
            "MOLD_DB_PATH": runtime / "mold.db",
            "MOLD_OUTPUT_DIR": runtime / "output",
        }
        for key, path in replacements.items():
            if key in payload["environment"]:
                payload["environment"][key] = str(path)

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

    def test_synthetic_semantic_fixture_validates_and_raw_tampering_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            validator.validate(path, require_passing=True)
            output = pathlib.Path(str(path) + ".d") / "parallel-output-1.png"
            output.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validator.validate(path, require_passing=True)

    def test_rehashed_semantic_tampering_still_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
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
            path = self.synthetic_semantic_fixture(root)
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
            path = self.synthetic_semantic_fixture(root)
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
            path = self.synthetic_semantic_fixture(root)
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

    def test_raw_samples_cannot_be_empty_or_disconnected_from_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "parallel-runtime-samples"
            )
            evidence_path = pathlib.Path(item["path"])
            evidence_path.write_text("", encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample|execution|work"):
                validator.validate(path, require_passing=True)

    def test_nonexistent_disabled_lane_cannot_satisfy_replan(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row for row in report["evidence"] if row["label"] == "parallel-results"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["disabled_id"] = "cuda:ghost"
            payload["pre_disable_assignments"] = {
                work_id: "cuda:ghost"
                for work_id in payload["pre_disable_assignments"]
            }
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disable|inventory|lane"):
                validator.validate(path, require_passing=True)

    def test_writable_root_with_post_separator_ro_bind_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row for row in report["evidence"] if row["label"] == "primary-command"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["argv"] = [
                "bwrap",
                "--die-with-parent",
                "--bind",
                "/",
                "/",
                "--",
                report["candidate"]["path"],
                "serve",
                "--ro-bind",
                "/",
                "/",
            ]
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "read-only|writable|root"):
                validator.validate(path, require_passing=True)

    def test_command_environment_cannot_escape_isolation_or_target_7680(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row for row in report["evidence"] if row["label"] == "primary-command"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["environment"] = {
                "HOME": "/home/real",
                "MOLD_HOME": "/home/real/.mold",
                "MOLD_DB_PATH": "/home/real/.mold/mold.db",
                "MOLD_OUTPUT_DIR": "/home/real/.mold/output",
                "MOLD_HOST": "http://127.0.0.1:7680",
            }
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "environment|isolation|7680|HOME"
            ):
                validator.validate(path, require_passing=True)

    def test_ci_filter_tracks_qualification_documentation(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("docs/qualification/README.md", workflow)

    def test_selector_results_must_match_selector_semantics(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row for row in report["evidence"] if row["label"] == "selector-matrix"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            scenario = next(
                row for row in payload["scenarios"] if row["label"] == "none"
            )
            scenario["devices"]["devices"][0]["admin_state"] = "enabled"
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selector|UUID"):
                validator.validate(path, require_passing=True)

    def test_live_service_identity_must_be_unchanged(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "live-service-after"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["owners"][0]["start_ticks"] += 1
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "live-service"):
                validator.validate(path, require_passing=True)

    def test_cancellation_terminal_event_must_precede_resume(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            report = json.loads(path.read_text())
            item = next(
                row
                for row in report["evidence"]
                if row["label"] == "queued-cancellation"
            )
            evidence_path = pathlib.Path(item["path"])
            payload = json.loads(evidence_path.read_text())
            payload["resume_at"] = "2026-07-28T00:00:39Z"
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            item["sha256"] = validator.sha256(evidence_path)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cancellation"):
                validator.validate(path, require_passing=True)

    def test_cancellation_uses_pause_response_and_typed_plan_blocker(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))

            def mutate(payload):
                payload["pause_response"] = {
                    "status": 200,
                    "body": {"paused": True},
                }
                payload["queue_before_cancel"].pop("paused", None)
                work = payload["queue_before_cancel"]["plan"]["work_items"][0]
                work["activity_phase"] = "blocked"
                work["blocked_reason"] = "queue_paused"

            self.mutate_evidence(path, "queued-cancellation", mutate)
            validator.validate(path, require_passing=True)

    def test_device_bind_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))

            def mutate(payload):
                separator = payload["argv"].index("--")
                payload["argv"][separator:separator] = [
                    "--dev-bind-try",
                    "/dev/dri/../../home/killswitch",
                    "/dev/dri/../../home/killswitch",
                ]

            self.mutate_evidence(path, "primary-command", mutate)
            with self.assertRaisesRegex(ValueError, "device|normalized|canonical"):
                validator.validate(path, require_passing=True)

    def test_runtime_symlink_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            report = json.loads(path.read_text())
            runtime = pathlib.Path(report["isolation"]["mold_home"]).parent
            runtime.mkdir(parents=True, exist_ok=True)
            alias = root / "runtime-alias"
            alias.symlink_to(runtime)

            def mutate(payload):
                argv = payload["argv"]
                index = argv.index("--bind")
                argv[index + 1] = str(alias)
                argv[index + 2] = str(alias)
                next_index = argv.index("--bind", index + 1)
                argv[next_index + 1] = str(alias / "tmp")

            self.mutate_evidence(path, "primary-command", mutate)
            with self.assertRaisesRegex(ValueError, "symlink|canonical|normalized"):
                validator.validate(path, require_passing=True)

    def test_candidate_version_runtime_cannot_escape_evidence_root(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            self.mutate_evidence(
                path,
                "candidate-version",
                lambda payload: self.relocate_runtime(
                    payload, root / "external-candidate-runtime"
                ),
            )
            with self.assertRaisesRegex(ValueError, "runtime|evidence|candidate"):
                validator.validate(path, require_passing=True)

    def test_selector_contract_rejects_sibling_prefix_runtime(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            sibling = pathlib.Path(str(path) + ".d-sibling") / "contract"
            self.mutate_evidence(
                path,
                "ambiguous-selector-source-contract",
                lambda payload: self.relocate_runtime(payload, sibling),
            )
            with self.assertRaisesRegex(ValueError, "runtime|evidence|contract"):
                validator.validate(path, require_passing=True)

    def test_probe_runtime_cannot_reuse_another_role_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            other_role = pathlib.Path(str(path) + ".d") / "selector-contract-runtime"
            self.mutate_evidence(
                path,
                "candidate-version",
                lambda payload: self.relocate_runtime(payload, other_role),
            )
            with self.assertRaisesRegex(ValueError, "runtime|role|candidate"):
                validator.validate(path, require_passing=True)

    def test_selector_runtime_cannot_reuse_another_scenario_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)

            def mutate(payload):
                scenario = next(
                    row for row in payload["scenarios"] if row["label"] == "all"
                )
                other = pathlib.Path(str(path) + ".d") / "selector-runtimes" / "empty"
                self.relocate_runtime(scenario, other)

            self.mutate_evidence(path, "selector-matrix", mutate)
            with self.assertRaisesRegex(ValueError, "runtime|role|selector"):
                validator.validate(path, require_passing=True)

    def test_client_runtime_cannot_reuse_probe_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.synthetic_semantic_fixture(root)
            probe = pathlib.Path(str(path) + ".d") / "candidate-probe-runtime"
            self.mutate_evidence(
                path,
                "client-projection",
                lambda payload: self.relocate_runtime(payload, probe),
            )
            with self.assertRaisesRegex(ValueError, "runtime|role|client"):
                validator.validate(path, require_passing=True)

    def test_empty_live_service_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(path, "live-service-before", lambda value: value.clear())
            self.mutate_evidence(path, "live-service-after", lambda value: value.clear())
            with self.assertRaisesRegex(ValueError, "live-service|listener"):
                validator.validate(path, require_passing=True)

    def test_reversed_runtime_timeline_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(
                path, "parallel-runtime-samples", lambda value: value.reverse()
            )
            with self.assertRaisesRegex(ValueError, "timeline|timestamp|plan"):
                validator.validate(path, require_passing=True)

    def test_replan_requires_strictly_advanced_plan_version(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))

            def mutate(payload):
                for row in payload:
                    row["queue"]["plan"]["plan_version"] = 1

            self.mutate_evidence(path, "parallel-runtime-samples", mutate)
            with self.assertRaisesRegex(ValueError, "replan|plan version|advanced"):
                validator.validate(path, require_passing=True)

    def test_missing_reserved_socket_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(
                path,
                "parallel-runtime-samples",
                lambda value: [
                    row.pop("reserved_service_connections") for row in value
                ],
            )
            with self.assertRaisesRegex(ValueError, "reserved|socket|connection"):
                validator.validate(path, require_passing=True)

    def test_empty_device_snapshots_are_rejected(self):
        mutations = (
            ("all-disabled-maintenance", "devices"),
            ("restart-persistence", "restored_devices"),
        )
        for label, field in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw:
                path = self.synthetic_semantic_fixture(pathlib.Path(raw))
                self.mutate_evidence(
                    path,
                    label,
                    lambda value, field=field: value[field].update(devices=[]),
                )
                with self.assertRaisesRegex(ValueError, "device|inventory|mapping"):
                    validator.validate(path, require_passing=True)

    def test_restart_identity_must_match_candidate_inode(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(
                path,
                "restart-persistence",
                lambda value: value["new_process_identity"].update(
                    executable_inode=value["new_process_identity"][
                        "executable_inode"
                    ]
                    + 1
                ),
            )
            with self.assertRaisesRegex(ValueError, "identity|inode|executable"):
                validator.validate(path, require_passing=True)

    def test_legacy_patch_body_is_exactly_mode_conflict(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(
                path,
                "legacy-rollback",
                lambda value: value["patch_body"].update(code="CONFLICT"),
            )
            with self.assertRaisesRegex(ValueError, "legacy|rollback"):
                validator.validate(path, require_passing=True)

    def test_selector_contract_environment_cannot_target_service(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))
            self.mutate_evidence(
                path,
                "ambiguous-selector-source-contract",
                lambda value: value["environment"].update(
                    MOLD_HOST="http://127.0.0.1:7680"
                ),
            )
            with self.assertRaisesRegex(ValueError, "environment|host"):
                validator.validate(path, require_passing=True)

    def test_selector_cannot_omit_excluded_devices(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.synthetic_semantic_fixture(pathlib.Path(raw))

            def mutate(payload):
                scenario = next(
                    row for row in payload["scenarios"] if row["label"] == "none"
                )
                scenario["devices"]["devices"] = []

            self.mutate_evidence(path, "selector-matrix", mutate)
            with self.assertRaisesRegex(ValueError, "selector|inventory|device"):
                validator.validate(path, require_passing=True)


if __name__ == "__main__":
    unittest.main()
