#!/usr/bin/env python3
"""Build/deploy Lambda relay code and private shell assets; never rotate secrets."""
import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
import time
import zipfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


class AwsError(RuntimeError):
    def __init__(self, service, operation, code="Unknown"):
        self.code = code
        super().__init__(f"AWS {service} {operation} failed ({code}); response omitted")


def aws(args, options):
    result = subprocess.run(["aws", "--profile", options.profile, "--region", options.region,
                             "--no-cli-pager", *args], capture_output=True, text=True)
    if result.returncode:
        match = re.search(r"An error occurred \(([A-Za-z0-9_.-]+)\)", result.stderr)
        raise AwsError(args[0], args[1], match.group(1) if match else "Unknown")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def private_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(content)


def read_token(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("Token file must be an owner-only regular file")
        token = source.read(4097).strip()
    if not token or len(token) > 4096 or any(c.isspace() for c in token):
        raise RuntimeError("Token file contains an invalid token")
    return token


def token_preflight(options):
    result = {}
    for role in ("host", "bridge"):
        path = getattr(options, f"{role}_token_file")
        local = read_token(path) if path.exists() or path.is_symlink() else None
        try:
            remote = aws(["ssm", "get-parameter", "--name", f"/mold/relay/{role}-token",
                          "--with-decryption"], options)["Parameter"]["Value"]
        except AwsError as error:
            if error.code != "ParameterNotFound":
                raise
            remote = None
        if local is not None and remote is not None and local != remote:
            raise RuntimeError(f"Existing {role} token differs; refusing rotation")
        result[role] = {"token": remote or local or secrets.token_hex(32),
                        "remote": remote is not None, "local": local is not None}
    if secrets.compare_digest(result["host"]["token"], result["bridge"]["token"]):
        raise RuntimeError("Host and bridge tokens must have different values")
    return result


def provision_token(options, role, path, prepared=None):
    if prepared is None:
        # Standalone compatibility; deployment always preflights both roles together.
        local = read_token(path) if path.exists() or path.is_symlink() else None
        try:
            remote = aws(["ssm", "get-parameter", "--name", f"/mold/relay/{role}-token", "--with-decryption"], options)["Parameter"]["Value"]
        except AwsError as error:
            if error.code != "ParameterNotFound":
                raise
            remote = None
        if local is not None and remote is not None and local != remote:
            raise RuntimeError(f"Existing {role} token differs; refusing rotation")
        prepared = {"token": remote or local or secrets.token_hex(32), "remote": remote is not None, "local": local is not None}
    if not prepared["remote"]:
        with tempfile.TemporaryDirectory(prefix="mold-relay-secret-") as directory:
            request = Path(directory) / "request.json"
            private_write(request, json.dumps({"Name": f"/mold/relay/{role}-token", "Type": "SecureString",
                                              "Value": prepared["token"], "Description": f"Mold relay {role} authentication"}))
            aws(["ssm", "put-parameter", "--cli-input-json", f"file://{request}"], options)
    if not prepared["local"]:
        private_write(path, prepared["token"] + "\n")


def retain_rollback(options, packages):
    manifest = {"profile": options.profile, "region": options.region, "functions": {}}
    for role, package in packages.items():
        response = aws(["lambda", "get-function", "--function-name", f"mold-relay-{role}"], options)
        config = response["Configuration"]
        destination = options.output / f"before-{role}.zip"
        try:
            # Presigned AWS URL stays in memory; exceptions must never expose it.
            with urllib.request.urlopen(response["Code"]["Location"], timeout=30) as source:
                data = source.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise RuntimeError("Rollback artifact exceeds download bound")
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                target.write(data)
        except Exception:
            raise RuntimeError(f"Could not retain {role} rollback code; URL and response omitted") from None
        if package_hash(destination) != config["CodeSha256"]:
            raise RuntimeError(f"{role} rollback artifact hash mismatch")
        manifest["functions"][role] = {"before": config["CodeSha256"], "deployed": package_hash(package)}
    private_write(options.output / "rollback.json", json.dumps(manifest))
    return manifest


def rollback(options, directory):
    manifest = json.loads((directory / "rollback.json").read_text())
    if manifest["profile"] != options.profile or manifest["region"] != options.region:
        raise RuntimeError("Rollback profile/region differs from retained deployment")
    # Preflight every function before changing either; never overwrite unrelated code.
    for role, entry in manifest["functions"].items():
        if role not in ("router", "frontend"):
            raise RuntimeError("Invalid rollback role")
        package = directory / f"before-{role}.zip"
        if package_hash(package) != entry["before"]:
            raise RuntimeError("Rollback ZIP hash mismatch")
        current = aws(["lambda", "get-function-configuration", "--function-name", f"mold-relay-{role}"], options)
        if current["CodeSha256"] not in (entry["before"], entry["deployed"]):
            raise RuntimeError("Concurrent code change detected; refusing rollback")
    for role, entry in manifest["functions"].items():
        deploy_function(options, role, directory / f"before-{role}.zip", allowed_hashes=(entry["before"], entry["deployed"]))


def build_packages(directory):
    packages = {}
    for role in ("router", "frontend"):
        output = directory / role
        output.mkdir()
        result = subprocess.run(["bun", "build", str(ROOT / "relay/aws" / f"{role}.mjs"),
                                 "--target=node", "--outfile", str(output / "index.mjs")],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Bun {role} build failed; build output omitted")
        package = directory / f"{role}.zip"
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            entry = zipfile.ZipInfo("index.mjs", (2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (output / "index.mjs").read_bytes())
        packages[role] = package
    return packages


def package_hash(package):
    return base64.b64encode(hashlib.sha256(package.read_bytes()).digest()).decode()


def deploy_function(options, role, package, allowed_hashes=None):
    name = f"mold-relay-{role}"
    before = aws(["lambda", "get-function-configuration", "--function-name", name], options)
    if before.get("LastUpdateStatus") != "Successful" or before.get("State") != "Active":
        raise RuntimeError(f"{name} is not ready for a code update")
    if allowed_hashes is not None and before["CodeSha256"] not in allowed_hashes:
        raise RuntimeError("Concurrent code change detected; refusing rollback")
    expected = package_hash(package)
    if before["CodeSha256"] == expected:
        return
    aws(["lambda", "update-function-code", "--function-name", name,
         "--revision-id", before["RevisionId"], "--zip-file", f"fileb://{package}"], options)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        after = aws(["lambda", "get-function-configuration", "--function-name", name], options)
        if after.get("LastUpdateStatus") == "Failed":
            raise RuntimeError(f"{name} code update failed; diagnostic values omitted")
        if after.get("LastUpdateStatus") == "Successful":
            if after["CodeSha256"] != expected:
                raise RuntimeError(f"{name} deployed ZIP hash mismatch")
            # Reject concurrent configuration changes; this tool never writes configuration.
            for key in ("Runtime", "Handler", "Role", "Timeout", "MemorySize", "Architectures", "Environment"):
                if before.get(key) != after.get(key):
                    raise RuntimeError(f"{name} configuration changed concurrently ({key})")
            print(f"{name}: verified ZIP sha256 {expected}")
            return
        time.sleep(2)
    raise RuntimeError(f"{name} update status unconfirmed; inspect before retrying")


def shell_files(options):
    root = options.shell.resolve()
    if not (root / "index.html").is_file():
        raise RuntimeError("Shell must contain a built index.html")
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise RuntimeError("Shell assets must not contain symlinks")
        if source.is_file():
            yield source, source.relative_to(root).as_posix()


def upload_shell(options):
    for source, relative in shell_files(options):
        aws(["s3api", "put-object", "--bucket", options.bucket, "--key", f"shell/{relative}",
                 "--body", str(source), "--content-type", mimetypes.guess_type(relative)[0] or "application/octet-stream",
                 "--server-side-encryption", "AES256"], options)
    print("Private shell/ assets uploaded; other bucket prefixes preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev.urandom.io")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--bucket", default="mold-relay-042506291754-us-west-2")
    parser.add_argument("--shell", type=Path, default=ROOT / "web/dist")
    parser.add_argument("--host-token-file", type=Path)
    parser.add_argument("--bridge-token-file", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rollback", type=Path, help="Restore retained predeployment ZIPs with concurrent-code guards")
    parser.add_argument("--output", type=Path, help="Fresh retained artifact directory")
    options = parser.parse_args()
    if options.rollback:
        rollback(options, options.rollback)
        return
    if options.output is None:
        parser.error("--output is required for build/deploy")
    if not options.build_only and (options.host_token_file is None or options.bridge_token_file is None):
        parser.error("Deployment requires separate owner-only host and bridge token files")
    if not options.build_only and options.host_token_file.resolve() == options.bridge_token_file.resolve():
        parser.error("Host and bridge token paths must differ")
    options.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    packages = build_packages(options.output)
    for role, package in packages.items():
        print(f"{role} ZIP sha256 {package_hash(package)}")
    if options.build_only:
        return
    # Validate shell before any cloud mutation.
    list(shell_files(options))
    tokens = token_preflight(options)
    manifest = retain_rollback(options, packages)
    for role in ("host", "bridge"):
        provision_token(options, role, getattr(options, f"{role}_token_file"), tokens[role])
    for role, package in packages.items():
        deploy_function(options, role, package, allowed_hashes=(manifest["functions"][role]["before"],))
    upload_shell(options)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
