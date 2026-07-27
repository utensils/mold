#!/usr/bin/env python3
"""Adversarial unit and verifier integration tests for embedded PTX parsing."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "scripts/probe-cuda-embedded-ptx.py"
VERIFIER_PATH = REPO_ROOT / "scripts/verify-cuda-release-binary.sh"
SEAL_PATH = REPO_ROOT / "scripts/seal-cuda-ptx-manifest.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("mold_ptx_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load embedded PTX probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmbeddedPtxParserContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probe_module = load_probe_module()
        test_target = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        test_target.mkdir(parents=True, exist_ok=True)
        cls.temp_dir_context = tempfile.TemporaryDirectory(
            prefix="cuda-ptx-contract-",
            dir=test_target,
        )
        cls.temp_dir = Path(cls.temp_dir_context.name)
        cls.fake_bin = cls.temp_dir / "bin"
        cls.fake_bin.mkdir()
        cls._write_executable(
            cls.fake_bin / "readelf",
            "#!/usr/bin/env bash\n"
            "echo '  Machine: Advanced Micro Devices X86-64'\n",
        )
        cls._write_executable(
            cls.fake_bin / "ldd",
            "#!/usr/bin/env bash\n"
            "echo 'linux-vdso.so.1'\n",
        )
        cls.verifier_env = dict(os.environ)
        cls.verifier_env["PATH"] = (
            f"{cls.fake_bin}:{cls.verifier_env.get('PATH', '')}"
        )
        cls.verifier_env["TMPDIR"] = str(cls.temp_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_context.cleanup()

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def compile_unsealed_fixture(self, name: str, ptx_bytes: bytes) -> Path:
        path = self.temp_dir / name
        c_source = self.temp_dir / f"{name}.c"
        byte_literals = ",".join(f"0x{byte:02x}" for byte in ptx_bytes)
        c_source.write_text(
            "#include <string.h>\n"
            f"__attribute__((used)) static const unsigned char generated_ptx[] = {{{byte_literals}}};\n"
            '__attribute__((used)) static const char nvml_symbol[] = "nvmlDeviceGetCount_v2";\n'
            "int main(int argc, char **argv) {\n"
            "  volatile const void *keep[] = {generated_ptx, nvml_symbol};\n"
            "  (void)keep;\n"
            '  return argc == 2 && strcmp(argv[1], "version") == 0 ? 0 : 1;\n'
            "}\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["cc", "-O0", "-o", str(path), str(c_source)],
            check=True,
            capture_output=True,
            text=True,
        )
        return path

    def fixture(self, name: str, ptx: str) -> Path:
        ptx_bytes = f"{ptx.rstrip()}\n".encode()
        path = self.compile_unsealed_fixture(name, ptx_bytes)

        inventory = self.probe_module.extract_entry_modules(ptx_bytes)
        if (
            not inventory["malformed_modules"]
            and not inventory["incomplete_modules"]
            and len(inventory["observed_targets"]) == 1
        ):
            compute_cap = str(inventory["observed_targets"][0]).removeprefix(
                "sm_"
            )
            build_root = self.temp_dir / f"{name}-build"
            output_dir = build_root / "candle-kernels-fixture" / "out"
            output_dir.mkdir(parents=True)
            (output_dir / "fixture.ptx").write_bytes(ptx_bytes)
            (output_dir / "ptx.rs").write_text(
                'pub const FIXTURE: &str = include_str!(concat!(env!("OUT_DIR"), "/fixture.ptx"));\n',
                encoding="utf-8",
            )
            (output_dir.parent / "output").write_text(
                f"cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}\n",
                encoding="utf-8",
            )
            seal = subprocess.run(
                [
                    str(SEAL_PATH),
                    str(path),
                    compute_cap,
                    str(build_root),
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "TMPDIR": str(self.temp_dir)},
            )
            if seal.returncode != 0:
                raise AssertionError(seal.stderr)
        return path

    def test_unsealed_shell_heredoc_ptx_is_not_qualification_evidence(
        self,
    ) -> None:
        fixture = self.temp_dir / "unsealed-heredoc"
        self._write_executable(
            fixture,
            "#!/usr/bin/env bash\n"
            "# nvmlDeviceGetCount_v2\n"
            '[[ "${1:-}" == version ]] && exit 0\n'
            ": <<'PTX'\n"
            ".version 8.0\n"
            ".target sm_86\n"
            ".address_size 64\n"
            ".visible .entry unrelated() { ret; }\n"
            "PTX\n",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_extracting_sealed_sections_does_not_mutate_the_input_artifact(
        self,
    ) -> None:
        fixture = self.temp_dir / "probe-input-is-immutable"
        fixture.write_bytes(b"exact release artifact")
        extracted = self.temp_dir / "probe-input-section"
        fake_objcopy = self.temp_dir / "objcopy-mutates-in-place-without-output"
        self._write_executable(
            fake_objcopy,
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'section_path="${2#*=}"\n'
            'printf "section bytes" > "$section_path"\n'
            'if [[ "$#" -lt 4 ]]; then\n'
            '  printf "mutated" >> "$3"\n'
            "else\n"
            '  cp "$3" "$4"\n'
            "fi\n",
        )
        before = hashlib.sha256(fixture.read_bytes()).hexdigest()
        previous_objcopy = os.environ.get("OBJCOPY")
        os.environ["OBJCOPY"] = str(fake_objcopy)
        try:
            self.assertEqual(
                self.probe_module.extract_elf_section(
                    fixture, ".mold_cuda_ptx", extracted
                ),
                b"section bytes",
            )
        finally:
            if previous_objcopy is None:
                os.environ.pop("OBJCOPY", None)
            else:
                os.environ["OBJCOPY"] = previous_objcopy
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), before)

    def test_unrelated_expected_target_cannot_replace_generated_kernel_ptx(
        self,
    ) -> None:
        unrelated = (
            b".version 8.0\n.target sm_86\n.address_size 64\n"
            b".visible .entry unrelated() { ret; }\n"
        )
        actual_kernel = (
            b".version 8.0\n.target sm_86\n.address_size 64\n"
            b".visible .entry actual_kernel() { ret; }\n"
        )
        fixture = self.compile_unsealed_fixture(
            "unrelated-instead-of-kernel", unrelated
        )
        build_root = self.temp_dir / "wrong-kernel-build"
        output_dir = build_root / "candle-kernels-fixture" / "out"
        output_dir.mkdir(parents=True)
        (output_dir / "actual.ptx").write_bytes(actual_kernel)
        (output_dir / "ptx.rs").write_text(
            'pub const ACTUAL: &str = include_str!(concat!(env!("OUT_DIR"), "/actual.ptx"));\n',
            encoding="utf-8",
        )
        (output_dir.parent / "output").write_text(
            "cargo:rustc-env=CUDA_COMPUTE_CAP=86\n", encoding="utf-8"
        )
        seal = subprocess.run(
            [str(SEAL_PATH), str(fixture), "86", str(build_root)],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "TMPDIR": str(self.temp_dir)},
        )
        self.assertNotEqual(seal.returncode, 0, seal.stdout)
        self.assertIn("no generated candle-kernels PTX", seal.stderr)
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_partially_overlapping_manifest_ranges_are_rejected(self) -> None:
        fixture = self.fixture(
            "overlapping-manifest-ranges",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry real() { ret; }
""",
        )
        manifest_path = self.temp_dir / "overlapping-manifest.json"
        rodata_path = self.temp_dir / "overlapping-rodata.bin"
        subprocess.run(
            [
                "objcopy",
                "--dump-section",
                f".mold_cuda_ptx={manifest_path}",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["objcopy", "--dump-section", f".rodata={rodata_path}", str(fixture)],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original = manifest["modules"][0]
        overlapping_offset = original["rodata_offset"] + 1
        overlapping_length = original["byte_length"] - 1
        rodata = rodata_path.read_bytes()
        overlapping_bytes = rodata[
            overlapping_offset : overlapping_offset + overlapping_length
        ]
        manifest["modules"].append(
            {
                "source_name": "overlapping.ptx",
                "rodata_offset": overlapping_offset,
                "byte_length": overlapping_length,
                "sha256": hashlib.sha256(overlapping_bytes).hexdigest(),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "objcopy",
                "--update-section",
                f".mold_cuda_ptx={manifest_path}",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def probe(self, fixture: Path, compute_cap: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(PROBE_PATH),
                str(fixture),
                str(compute_cap),
                "--extract-only",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def verifier(
        self, fixture: Path, compute_cap: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(VERIFIER_PATH), str(fixture), str(compute_cap)],
            cwd=REPO_ROOT,
            env=self.verifier_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_probe_and_verifier_accept(self, fixture: Path, compute_cap: int) -> None:
        probe = self.probe(fixture, compute_cap)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        verifier = self.verifier(fixture, compute_cap)
        self.assertEqual(verifier.returncode, 0, verifier.stderr)

    def assert_probe_and_verifier_reject(self, fixture: Path, compute_cap: int) -> None:
        probe = self.probe(fixture, compute_cap)
        self.assertNotEqual(probe.returncode, 0, probe.stdout)
        verifier = self.verifier(fixture, compute_cap)
        self.assertNotEqual(verifier.returncode, 0, verifier.stdout)

    def test_unit_inventory_ignores_comments_and_comment_braces(self) -> None:
        data = b"""
/* .version 7.0 */
.version 8.0 // .version 9.0
/*
.target sm_89
*/
.target sm_86 /* .target sm_89 */
// .address_size 32
.address_size 64
// .visible .entry fake_line() { }
/* .visible .entry fake_block() { } */
.visible .entry real() {
  // }
  /* } */
  ret;
}
"""
        inventory = self.probe_module.extract_entry_modules(data)
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertEqual(len(inventory["candidates"]), 1)
        self.assertEqual(inventory["malformed_modules"], [])
        self.assertEqual(inventory["incomplete_modules"], [])

    def test_generated_preamble_bounds_unrelated_static_comment_text(self) -> None:
        data = b"""
unrelated concatenated static string /* not PTX and never closed
// Generated by NVIDIA NVVM Compiler
//
.version 8.0
.target sm_86
.address_size 64
.visible .entry real() { ret; }
"""
        inventory = self.probe_module.extract_entry_modules(data)
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertEqual(len(inventory["candidates"]), 1)
        self.assertEqual(inventory["malformed_modules"], [])
        self.assertEqual(inventory["incomplete_modules"], [])

    def test_static_property_text_is_not_a_raw_version_directive(self) -> None:
        data = b"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry real() { ret; }
\0const current = state.version; const message = "unclosed static text
"""
        inventory = self.probe_module.extract_entry_modules(data)
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertEqual(len(inventory["candidates"]), 1)
        self.assertEqual(inventory["malformed_modules"], [])
        self.assertEqual(inventory["incomplete_modules"], [])

    def test_inline_and_multiline_comments_are_accepted(self) -> None:
        fixture = self.fixture(
            "inline-comments",
            """
/* .version 7.0 */
.version 8.0 // .version 9.0
/* first line
.target sm_89
last line */
.target sm_86 /* .target sm_89 */
// .address_size 32
.address_size 64 /* .address_size 32 */
// .visible .entry fake_line() { }
/* .visible .entry fake_block() { } */
.visible .entry real() {
  // }
  /* } */
  ret;
}
""",
        )
        self.assert_probe_and_verifier_accept(fixture, 86)

    def test_block_comment_target_cannot_spoof_real_target(self) -> None:
        fixture = self.fixture(
            "block-target-spoof",
            """
.version 8.0
/*
.target sm_86
*/
.target sm_89
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)
        self.assert_probe_and_verifier_accept(fixture, 89)

    def test_line_comment_target_cannot_spoof_real_target(self) -> None:
        fixture = self.fixture(
            "line-target-spoof",
            """
.version 8.0
// .target sm_86
.target sm_89
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)
        self.assert_probe_and_verifier_accept(fixture, 89)

    def test_commented_version_cannot_create_a_module(self) -> None:
        fixture = self.fixture(
            "version-spoof",
            """
/*
.version 8.0
.target sm_86
.address_size 64
.visible .entry fake() { ret; }
*/
.version 8.0
.target sm_89
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)
        self.assert_probe_and_verifier_accept(fixture, 89)

    def test_duplicate_same_target_is_rejected(self) -> None:
        fixture = self.fixture(
            "duplicate-same",
            """
.version 8.0
.target sm_86
.target sm_86
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_duplicate_different_targets_are_rejected(self) -> None:
        fixture = self.fixture(
            "duplicate-different",
            """
.version 8.0
.target sm_86
.target sm_89
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_target_operand_cannot_smuggle_a_second_architecture(self) -> None:
        fixture = self.fixture(
            "target-tail-second-architecture",
            """
.version 8.0
.target sm_86, sm_100
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_valid_target_platform_options_are_accepted(self) -> None:
        fixture = self.fixture(
            "target-platform-options",
            """
.version 8.0
.target sm_86, texmode_unified, debug
.address_size 64
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_accept(fixture, 86)

    def test_second_target_after_first_entry_is_rejected(self) -> None:
        fixture = self.fixture(
            "target-after-first-entry",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry first() { ret; }
.target sm_100
""",
        )
        inventory = self.probe_module.extract_entry_modules(fixture.read_bytes())
        self.assertEqual(
            inventory["malformed_modules"][0]["reason"],
            "duplicate target directives",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_commented_address_is_not_an_address_directive(self) -> None:
        fixture = self.fixture(
            "address-spoof",
            """
.version 8.0
.target sm_86
/* .address_size 64 */
.visible .entry real() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_commented_entry_is_not_an_entry_directive(self) -> None:
        fixture = self.fixture(
            "entry-spoof",
            """
.version 8.0
.target sm_86
.address_size 64
/* .visible .entry fake() { ret; } */
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_unclosed_block_comment_is_rejected(self) -> None:
        fixture = self.fixture(
            "unclosed-comment",
            """
.version 8.0
.target sm_86
/*
.address_size 64
.visible .entry fake() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_unclosed_comment_after_good_module_cannot_hide_later_target(self) -> None:
        fixture = self.fixture(
            "good-then-unclosed-comment-hides-later-target",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry first() { ret; }
/* unclosed
.version 8.0
.target sm_100
.address_size 64
.visible .entry hidden() { ret; }
""",
        )
        inventory = self.probe_module.extract_entry_modules(fixture.read_bytes())
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertTrue(
            any(
                module.get("hidden_version_offsets")
                for module in inventory["incomplete_modules"]
            )
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_lexical_error_after_complete_candidate_is_inventory_visible(self) -> None:
        fixture = self.fixture(
            "good-then-unclosed-comment",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry first() { ret; }
/* unclosed
""",
        )
        inventory = self.probe_module.extract_entry_modules(fixture.read_bytes())
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertEqual(
            [module["reason"] for module in inventory["incomplete_modules"]],
            ["block comment is not closed"],
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_generated_preamble_region_cannot_silently_drop_hidden_version(
        self,
    ) -> None:
        fixture = self.fixture(
            "generated-preamble-then-unclosed-comment",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry first() { ret; }
// Generated by NVIDIA NVVM Compiler
/*
.version 8.0
.target sm_100
.address_size 64
.visible .entry hidden() { ret; }
""",
        )
        inventory = self.probe_module.extract_entry_modules(fixture.read_bytes())
        self.assertEqual(inventory["observed_targets"], ["sm_86"])
        self.assertTrue(
            any(
                module.get("hidden_version_offsets")
                for module in inventory["incomplete_modules"]
            )
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_line_comment_entry_decoy_cannot_hide_later_module(self) -> None:
        fixture = self.fixture(
            "entry-decoy-before-later-target",
            """
.version 8.0
.target sm_86
.address_size 64
.visible .entry first() { ret; }
.version 8.0
// .entry decoy
.target sm_90
.address_size 64
.visible .entry later() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)
        self.assert_probe_and_verifier_reject(fixture, 90)

    def test_target_beyond_sixteen_kib_is_inventory_visible(self) -> None:
        padding = "\n".join(f".file {index} \"padding-{index}\"" for index in range(900))
        self.assertGreater(len(padding), 16 * 1024)
        fixture = self.fixture(
            "large-header",
            f"""
.version 8.0
{padding}
.target sm_86
.address_size 64
.visible .entry real() {{ ret; }}
""",
        )
        self.assert_probe_and_verifier_accept(fixture, 86)

    def test_complete_entry_without_target_is_rejected(self) -> None:
        fixture = self.fixture(
            "missing-target",
            """
.version 8.0
.address_size 64
.visible .entry no_target() { ret; }
""",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_version_region_without_entry_is_inventory_visible(self) -> None:
        ptx = """
.version 8.0
.address_size 64
"""
        inventory = self.probe_module.extract_entry_modules(ptx.encode())
        self.assertEqual(len(inventory["incomplete_modules"]), 1)
        self.assertEqual(
            inventory["incomplete_modules"][0]["reason"],
            "missing entry",
        )
        fixture = self.fixture("missing-entry-and-target", ptx)
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_malformed_version_directive_is_rejected(self) -> None:
        fixture = self.fixture(
            "malformed-version",
            """
.version 8.x
.target sm_86
.address_size 64
.visible .entry malformed_version() { ret; }
""",
        )
        inventory = self.probe_module.extract_entry_modules(fixture.read_bytes())
        self.assertEqual(len(inventory["malformed_modules"]), 1)
        self.assertEqual(
            inventory["malformed_modules"][0]["reason"],
            "malformed version directive",
        )
        self.assert_probe_and_verifier_reject(fixture, 86)

    def test_version_requires_integer_major_and_minor_operands(self) -> None:
        invalid_operands = (
            "8",
            ".8",
            "8.",
            "8.0.1",
            "8..0",
            "8.0suffix",
            "+8.0",
            "-8.0",
        )
        for index, operand in enumerate(invalid_operands):
            with self.subTest(operand=operand):
                fixture = self.fixture(
                    f"malformed-version-boundary-{index}",
                    f"""
.version {operand}
.target sm_86
.address_size 64
.visible .entry malformed_version() {{ ret; }}
""",
                )
                inventory = self.probe_module.extract_entry_modules(
                    fixture.read_bytes()
                )
                self.assertEqual(len(inventory["malformed_modules"]), 1)
                self.assertEqual(
                    inventory["malformed_modules"][0]["reason"],
                    "malformed version directive",
                )
                self.assert_probe_and_verifier_reject(fixture, 86)

    def test_version_accepts_spacing_and_crlf_with_major_minor(self) -> None:
        fixture = self.fixture(
            "version-spacing-crlf",
            "\t.version\t8.0   \r\n"
            "\t.target sm_86\r\n"
            "\t.address_size 64\r\n"
            "\t.visible .entry real() { ret; }\r\n",
        )
        self.assert_probe_and_verifier_accept(fixture, 86)


if __name__ == "__main__":
    unittest.main()
