#!/usr/bin/env python3
"""Rebuild on native source changes without watching generated build output."""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
IOS = ROOT / "apps/ios"


def snapshot():
    roots = [IOS / "Sources", IOS / "Tests"]
    for package in (ROOT / "apps/shared/Packages").iterdir():
        roots += [package / "Sources", package / "Tests"]
    files = [IOS / "project.yml", IOS / "Makefile"]
    files += list((ROOT / "apps/shared/Packages").glob("*/Package.swift"))
    files += [p for root in roots for p in root.rglob("*") if p.is_file()]
    return {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()}


def main():
    command = ["make", "-C", str(IOS), "run", *sys.argv[1:]]
    previous = None
    try:
        while True:
            current = snapshot()
            if current != previous:
                previous = current
                if subprocess.run(command).returncode:
                    print("Build failed; watching for the next edit.", flush=True)
                else:
                    print("Watching native iOS and shared Swift sources. Ctrl-C to stop.", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
