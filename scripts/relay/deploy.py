#!/usr/bin/env python3
"""Deploy a reviewed static relay artifact through AWS SSM, without secret logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile
import time


def aws(args, options):
    result = subprocess.run(
        ["aws", "--profile", options.profile, "--region", options.region,
         "--no-cli-pager", *args], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        # Commands may contain API responses; never forward captured output.
        raise RuntimeError(f"AWS {args[0]} {args[1]} failed; inspect the service separately")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def private_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse symlinks and world-readable existing files rather than truncating them.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(content)


def provision_token(options):
    if options.token_file.exists():
        raise RuntimeError("Token destination already exists; choose a fresh owner-only path")
    # SecureString is deliberately outside Terraform state and instance user-data.
    listing = aws(["ssm", "describe-parameters", "--parameter-filters",
                   json.dumps([{"Key": "Name", "Option": "Equals", "Values": [options.parameter]}])], options)
    if listing.get("Parameters"):
        token = aws(["ssm", "get-parameter", "--name", options.parameter,
                     "--with-decryption"], options)["Parameter"]["Value"]
    else:
        token = secrets.token_hex(32)
        with tempfile.TemporaryDirectory(prefix="mold-relay-token-") as directory:
            request = Path(directory) / "request.json"
            private_write(request, json.dumps({"Name": options.parameter,
                          "Type": "SecureString", "Value": token,
                          "Description": "Mold relay host enrollment; not a Mold API key"}))
            aws(["ssm", "put-parameter", "--cli-input-json", f"file://{request}"], options)
    private_write(options.token_file, token + "\n")
    print("Enrollment token saved to the requested owner-only file; value omitted.")


def deploy(options):
    artifact = options.artifact.resolve()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    key = f"mold-relay/releases/{digest}/mold-relay"
    aws(["s3api", "put-object", "--bucket", options.bucket, "--key", key,
         "--body", str(artifact), "--server-side-encryption", "AES256"], options)
    remote_file = f"/tmp/mold-relay-{digest}"
    source = shlex.quote(f"s3://{options.bucket}/{key}")
    commands = [
        "set -eu",
        f"aws --region {shlex.quote(options.region)} s3 cp {source} {remote_file} --only-show-errors",
        f"printf '%s  %s\\n' {digest} {remote_file} | sha256sum -c -",
        f"chmod 755 {remote_file}",
        f"{remote_file} --help >/dev/null",
        "rm -f /usr/local/bin/mold-relay.previous",
        "if test -f /usr/local/bin/mold-relay; then cp /usr/local/bin/mold-relay /usr/local/bin/mold-relay.previous; fi",
        f"install -m 755 {remote_file} /usr/local/bin/mold-relay.next",
        "mv /usr/local/bin/mold-relay.next /usr/local/bin/mold-relay",
        "if systemctl restart mold-relay && sleep 2 && systemctl is-active --quiet mold-relay && curl --fail --silent http://127.0.0.1:7681/_mold/relay/health; then rm -f /usr/local/bin/mold-relay.previous; else systemctl stop mold-relay; if test -f /usr/local/bin/mold-relay.previous; then mv /usr/local/bin/mold-relay.previous /usr/local/bin/mold-relay; systemctl restart mold-relay; else rm -f /usr/local/bin/mold-relay; fi; exit 1; fi",
        f"rm -f {remote_file}",
    ]
    request = {
        "InstanceIds": [options.instance_id], "DocumentName": "AWS-RunShellScript",
        "Comment": f"Deploy reviewed Mold relay {digest[:12]}",
        "Parameters": {"commands": commands, "executionTimeout": ["300"]},
    }
    with tempfile.TemporaryDirectory(prefix="mold-relay-deploy-") as directory:
        path = Path(directory) / "request.json"
        private_write(path, json.dumps(request))
        command = aws(["ssm", "send-command", "--cli-input-json", f"file://{path}"], options)["Command"]["CommandId"]
    deadline = time.monotonic() + 330
    while time.monotonic() < deadline:
        time.sleep(3)
        try:
            result = aws(["ssm", "get-command-invocation", "--command-id", command,
                          "--instance-id", options.instance_id], options)
        except RuntimeError:
            continue  # SSM invocation appears asynchronously.
        status = result["Status"]
        if status == "Success":
            print(f"Deployed artifact sha256:{digest}; gateway service health verified.")
            return
        if status in {"Cancelled", "Failed", "TimedOut", "Cancelling"}:
            raise RuntimeError(f"SSM deployment {command} ended {status}; inspect its nonsecret output")
    raise RuntimeError(f"SSM deployment {command} did not finish within 330 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev.urandom.io")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bucket", default="dev-urandom-io-devops")
    parser.add_argument("--parameter", default="/mold/relay/enrollment-token")
    parser.add_argument("--provision-token", action="store_true",
                        help="Create/retrieve enrollment SecureString and save an owner-only local file")
    parser.add_argument("--token-file", type=Path)
    options = parser.parse_args()
    if options.provision_token and options.token_file is None:
        parser.error("--provision-token requires --token-file")
    if not options.artifact.is_file():
        parser.error("--artifact must name a reviewed built binary")
    if options.provision_token:
        provision_token(options)
    deploy(options)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
