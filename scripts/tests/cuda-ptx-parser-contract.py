#!/usr/bin/env python3
"""Adversarial unit and verifier integration tests for embedded PTX parsing."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "scripts/probe-cuda-embedded-ptx.py"
VERIFIER_PATH = REPO_ROOT / "scripts/verify-cuda-release-binary.sh"


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
        cls.temp_dir_context = tempfile.TemporaryDirectory()
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_context.cleanup()

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def fixture(self, name: str, ptx: str) -> Path:
        path = self.temp_dir / name
        self._write_executable(
            path,
            "#!/usr/bin/env bash\n"
            "# nvmlDeviceGetCount_v2\n"
            '[[ "${1:-}" == version ]] && exit 0\n'
            "exit 1\n"
            ": <<'PTX'\n"
            f"{ptx.rstrip()}\n"
            "PTX\n",
        )
        return path

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


if __name__ == "__main__":
    unittest.main()
