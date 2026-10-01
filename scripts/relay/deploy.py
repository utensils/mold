#!/usr/bin/env python3
"""Deploy a reviewed static relay artifact through AWS SSM, without secret logs."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile
import time


class AwsError(RuntimeError):
    def __init__(self, service, operation, code="Unknown"):
        self.code = code
        super().__init__(f"AWS {service} {operation} failed ({code}); inspect the service separately")


def aws(args, options):
    result = subprocess.run(
        ["aws", "--profile", options.profile, "--region", options.region,
         "--no-cli-pager", *args], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        # Commands may contain API responses; never forward captured output.
        match = re.search(r"An error occurred \(([A-Za-z0-9_.-]+)\)", result.stderr)
        raise AwsError(args[0], args[1], match.group(1) if match else "Unknown")
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


def remote_script(options, digest, key):
    source = shlex.quote(f"s3://{options.bucket}/{key}")
    region = shlex.quote(options.region)
    # /run is root-owned. Keep staging and the deployment lock in an owner-only
    # directory; no predictable file in a shared /tmp and no parallel swaps.
    return f"""set -eu
install -d -m 700 /run/mold-relay-deploy
exec 9>/run/mold-relay-deploy/lock
flock -w 60 9
work=$(mktemp -d /run/mold-relay-deploy/staging.XXXXXX)
changed=false
committed=false
check_gateway() {{
  attempt=0
  while [ "$attempt" -lt 15 ]; do
    if systemctl is-active --quiet mold-relay && curl --max-time 2 --fail --silent http://127.0.0.1:7681/_mold/relay/health >/dev/null; then return 0; fi
    attempt=$((attempt + 1))
    sleep 2
  done
  return 1
}}
finish() {{
  result=$?
  trap - EXIT HUP INT TERM
  if [ "$changed" = true ] && [ "$committed" = false ]; then
    systemctl stop mold-relay || true
    if [ -f "$work/previous" ]; then
      install -m 755 "$work/previous" /usr/local/bin/mold-relay.next
      mv /usr/local/bin/mold-relay.next /usr/local/bin/mold-relay
      if ! systemctl restart mold-relay || ! check_gateway; then
        printf 'Rollback service health could not be verified.\\n' >&2
      fi
    else
      rm -f /usr/local/bin/mold-relay
      printf 'First deployment failed; service is stopped and can be retried.\\n' >&2
    fi
  fi
  rm -rf "$work"
  exit "$result"
}}
trap finish EXIT
trap 'exit 1' HUP INT TERM
aws --region {region} s3 cp {source} "$work/mold-relay" --only-show-errors
printf '%s  %s\\n' {digest} "$work/mold-relay" | sha256sum -c -
chmod 755 "$work/mold-relay"
"$work/mold-relay" --help >/dev/null
if [ -f /usr/local/bin/mold-relay ]; then cp /usr/local/bin/mold-relay "$work/previous"; fi
install -m 755 "$work/mold-relay" /usr/local/bin/mold-relay.next
changed=true
mv /usr/local/bin/mold-relay.next /usr/local/bin/mold-relay
systemctl restart mold-relay
check_gateway
committed=true
"""


def deploy(options):
    artifact = options.artifact.resolve()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    key = f"mold-relay/releases/{digest}/mold-relay"
    aws(["s3api", "put-object", "--bucket", options.bucket, "--key", key,
         "--body", str(artifact), "--server-side-encryption", "AES256"], options)
    request = {
        "InstanceIds": [options.instance_id], "DocumentName": "AWS-RunShellScript",
        "Comment": f"Deploy reviewed Mold relay {digest[:12]}",
        "TimeoutSeconds": 60,
        "Parameters": {"commands": [remote_script(options, digest, key)], "executionTimeout": ["300"]},
    }
    with tempfile.TemporaryDirectory(prefix="mold-relay-deploy-") as directory:
        path = Path(directory) / "request.json"
        private_write(path, json.dumps(request))
        command = aws(["ssm", "send-command", "--cli-input-json", f"file://{path}"], options)["Command"]["CommandId"]
    # Covers bounded delivery + execution + response propagation. Only the
    # documented not-yet-visible invocation is retryable; AccessDenied is fatal.
    deadline = time.monotonic() + 390
    while time.monotonic() < deadline:
        time.sleep(3)
        try:
            result = aws(["ssm", "get-command-invocation", "--command-id", command,
                          "--instance-id", options.instance_id], options)
        except AwsError as error:
            if error.code == "InvocationDoesNotExist":
                continue
            raise
        status = result["Status"]
        if status == "Success":
            print(f"Deployed artifact sha256:{digest}; gateway service health verified.")
            return
        if status in {"Cancelled", "Failed", "TimedOut", "Cancelling"}:
            raise RuntimeError(f"SSM deployment {command} ended {status}; inspect its nonsecret output")
    # Stop our own outstanding command instead of allowing a late silent swap.
    aws(["ssm", "cancel-command", "--command-id", command,
         "--instance-ids", options.instance_id], options)
    cancellation_deadline = time.monotonic() + 60
    while time.monotonic() < cancellation_deadline:
        time.sleep(3)
        result = aws(["ssm", "get-command-invocation", "--command-id", command,
                      "--instance-id", options.instance_id], options)
        status = result["Status"]
        if status == "Success":
            print(f"Deployed artifact sha256:{digest}; SSM confirmed completion while cancellation was requested.")
            return
        if status in {"Cancelled", "Failed", "TimedOut"}:
            raise RuntimeError(f"SSM deployment {command} exceeded its deadline and ended {status}; it can be retried after inspecting rollback health")
    raise RuntimeError(f"SSM deployment {command} cancellation is unconfirmed; do not retry until its terminal state is verified")


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
