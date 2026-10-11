#!/usr/bin/env python3
"""Deploy E2EE-only Lambda relay code or provision public host admission with AWS IAM."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
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


def protocol_contract(package, role):
    try:
        with zipfile.ZipFile(package) as archive:
            if archive.getinfo("mold-relay-contract.json").file_size > 1024:
                return False
            raw = archive.read("mold-relay-contract.json")
        return len(raw) <= 1024 and json.loads(raw) == {"protocol": 3, "role": role, "security": "noise-only"}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def provision_host(options):
    if not re.fullmatch(r"[a-f0-9]{64}", options.host_key or ""):
        raise RuntimeError("Host signing public key must be exactly 64 lowercase hex characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}", options.principal or ""):
        raise RuntimeError("Principal ID must be a bounded stable operator-owned identifier")
    namespace = hashlib.sha256(bytes.fromhex(options.host_key)).hexdigest()[:32]
    key = {"id": {"S": "v3-registration#" + namespace}}
    result = aws(["dynamodb", "get-item", "--table-name", options.table, "--consistent-read",
                  "--key", json.dumps(key), "--projection-expression", "document, revision"], options)
    existing = result.get("Item")
    if options.disable and not existing:
        raise RuntimeError("Cannot disable an unregistered host")
    document = {"namespace": namespace, "signingKey": options.host_key,
                "principalId": options.principal, "enabled": not options.disable}
    if existing:
        old = json.loads(existing["document"]["S"])
        if any(old.get(k) != document[k] for k in ("namespace", "signingKey", "principalId")):
            raise RuntimeError("Registration belongs to a different identity/principal; refusing reassignment")
        if old.get("enabled") == document["enabled"]:
            print(f"Host {namespace}: registration already matches")
            return
    revision = int(existing["revision"]["N"]) if existing else 0
    request = {"TableName": options.table, "Item": {**key, "document": {"S": json.dumps(document)},
               "revision": {"N": str(revision + 1)}},
               "ConditionExpression": "revision = :previous" if existing else "attribute_not_exists(id)"}
    if existing:
        request["ExpressionAttributeValues"] = {":previous": {"N": str(revision)}}
    # No TTL: only an IAM operator/account authority can revoke registration.
    with tempfile.TemporaryDirectory(prefix="mold-relay-registration-") as directory:
        path = Path(directory) / "request.json"
        private_write(path, json.dumps(request))
        aws(["dynamodb", "put-item", "--cli-input-json", f"file://{path}"], options)
    print(f"Host {namespace}: {'disabled' if options.disable else 'registered'}")


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
        manifest["functions"][role] = {"before": config["CodeSha256"], "deployed": package_hash(package), "before_secure_protocol": protocol_contract(destination, role)}
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
        if not entry.get("before_secure_protocol") or not protocol_contract(package, role):
            raise RuntimeError("Pre-cutover or unmarked code cannot be restored by secure relay rollback")
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
            contract = zipfile.ZipInfo("mold-relay-contract.json", (2020, 1, 1, 0, 0, 0))
            contract.compress_type = zipfile.ZIP_DEFLATED
            contract.external_attr = 0o644 << 16
            archive.writestr(contract, json.dumps({"protocol": 3, "role": role, "security": "noise-only"}, sort_keys=True))
        packages[role] = package
    return packages


def package_hash(package):
    return base64.b64encode(hashlib.sha256(package.read_bytes()).digest()).decode()


def deploy_function(options, role, package, allowed_hashes=None):
    name = f"{getattr(options, 'function_prefix', 'mold-relay')}-{role}"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev.urandom.io")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("action", nargs="?", choices=("deploy", "provision"), default="deploy")
    parser.add_argument("--table", default="mold-relay-sessions")
    parser.add_argument("--host-key", help="Public Ed25519 signing key; never a private key or API credential")
    parser.add_argument("--principal", help="Stable operator/account identifier for this registered host")
    parser.add_argument("--disable", action="store_true", help="Disable an existing matching host registration")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rollback", type=Path, help="Restore retained predeployment ZIPs with concurrent-code guards")
    parser.add_argument("--output", type=Path, help="Fresh retained artifact directory")
    options = parser.parse_args()
    if options.action == "provision":
        if options.build_only or options.rollback or options.output:
            parser.error("Provisioning cannot be combined with code deployment options")
        provision_host(options)
        return
    if options.host_key or options.principal or options.disable:
        parser.error("Host registration options require the provision action")
    if options.rollback:
        rollback(options, options.rollback)
        return
    if options.output is None:
        parser.error("--output is required for build/deploy")
    options.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    packages = build_packages(options.output)
    for role, package in packages.items():
        print(f"{role} ZIP sha256 {package_hash(package)}")
    if options.build_only:
        return
    manifest = retain_rollback(options, packages)
    for role, package in packages.items():
        deploy_function(options, role, package, allowed_hashes=(manifest["functions"][role]["before"],))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
