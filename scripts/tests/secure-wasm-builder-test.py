#!/usr/bin/env python3
"""Exercise the canonical browser builder without installing a Rust toolchain."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SecureWasmBuilderTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="mold wasm builder ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/build-secure-wasm.sh", self.root / "scripts")
        (self.root / "Cargo.lock").write_text(
            '[[package]]\nname = "wasm-bindgen"\nversion = "0.2.118"\n'
        )
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        CARGO_HOME=str(self.root / "cargo home"),
                        MOLD_WASM_TOOLCHAIN="1.98.0", MOCK_RUST_HOST="x86_64-unknown-linux-gnu",
                        MOCK_RUST_VERSION="1.98.0", MOCK_BUILD_ROOT=str(self.root))
        self.tool("rustc", """
import os, sys
if '--version' in sys.argv:
    print('rustc ' + os.environ['MOCK_RUST_VERSION'] + ' (88d9e12ae 2026-08-18)')
elif '-vV' in sys.argv:
    print('host: ' + os.environ['MOCK_RUST_HOST'])
    print('commit-hash: 88d9e12ae178fab0fb5cc050a94da85685d449ea')
elif '--print' in sys.argv:
    print(os.environ['MOCK_BUILD_ROOT'] + '/rust toolchain')
else:
    raise SystemExit(2)
""")
        self.tool("cargo", """
import json, os, sys
from pathlib import Path
Path('cargo-called.json').write_text(json.dumps({
    'args': sys.argv[1:], 'flags': os.environ['CARGO_ENCODED_RUSTFLAGS'].split(chr(31))}))
""")
        self.tool("wasm-bindgen", """
import sys
from pathlib import Path
if '--version' in sys.argv:
    print('wasm-bindgen 0.2.118')
else:
    Path('generator-called').touch()
""")
        self.tool("git", """
from pathlib import Path
Path('verification-called').touch()
""")

    def tool(self, name, code):
        path = self.bin / name
        path.write_text("#!/usr/bin/env python3\n" + code)
        path.chmod(0o755)

    def run_builder(self):
        return subprocess.run(["bash", str(self.root / "scripts/build-secure-wasm.sh"), "--check"],
                              env=self.env, capture_output=True, text=True, check=False)

    def test_noncanonical_hosts_refuse_before_build_or_generation(self):
        for host in ["aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-musl"]:
            with self.subTest(host=host):
                self.env["MOCK_RUST_HOST"] = host
                result = self.run_builder()
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("x86_64-unknown-linux-gnu", result.stderr)
                self.assertFalse((self.root / "cargo-called.json").exists())
                self.assertFalse((self.root / "generator-called").exists())

    def test_canonical_build_preserves_spaced_paths_and_exact_byte_check(self):
        self.env["CARGO_ENCODED_RUSTFLAGS"] = "--cfg\x1fambient_should_not_survive"
        result = self.run_builder()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call = json.loads((self.root / "cargo-called.json").read_text())
        self.assertIn("--locked", call["args"])
        self.assertEqual(call["flags"], [
            "--cfg", 'getrandom_backend="wasm_js"',
            f"--remap-path-prefix={self.root}=/mold",
            f"--remap-path-prefix={self.root}/cargo home=/cargo",
            f"--remap-path-prefix={self.root}/rust toolchain/lib/rustlib/src/rust="
            "/rustc/88d9e12ae178fab0fb5cc050a94da85685d449ea",
        ])
        self.assertTrue((self.root / "generator-called").exists())
        self.assertTrue((self.root / "verification-called").exists())

    def test_wrong_compiler_refuses_before_build(self):
        self.env["MOCK_RUST_VERSION"] = "1.99.0"
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rust 1.98.0", result.stderr)
        self.assertFalse((self.root / "cargo-called.json").exists())


if __name__ == "__main__":
    unittest.main()
