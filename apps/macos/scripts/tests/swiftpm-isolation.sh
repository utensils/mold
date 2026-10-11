#!/usr/bin/env bash
# A missing runner-global SwiftPM repository must not block local artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
from pathlib import Path
import shlex
import subprocess

expected = str(Path.cwd() / "build/DerivedData/SourcePackages")
for config in ("Debug", "Release"):
    for target in ("build", "test"):
        output = subprocess.check_output([
            "make", "-n", target, f"CONFIG={config}",
            "MOLD_DEV_SIGN_IDENTITY=", "BUILD_NUMBER=1", "MARKETING_VERSION=0.0.0",
        ], text=True)
        lines = output.replace("\\\n", " ").splitlines()
        command = next(line for line in lines if "xcodebuild -project" in line)
        args = shlex.split(command.split("xcodebuild ", 1)[1])
        assert "-disablePackageRepositoryCache" in args, (
            f"{config} {target} still uses the runner-global SwiftPM repository cache"
        )
        option = "-clonedSourcePackagesDirPath"
        assert option in args and args[args.index(option) + 1] == expected, (
            f"{config} {target} must isolate packages under DerivedData for Sparkle appcast discovery"
        )
print("SwiftPM package isolation ok")
PY
