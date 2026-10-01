#!/usr/bin/env python3
"""Credential safety checks for the deployment helper; never contacts AWS."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("relay_deploy", Path(__file__).parents[1] / "relay/deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentSafety(unittest.TestCase):
    def test_private_file_creation_refuses_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token"
            deploy.private_write(token, "fixture-secret")
            self.assertEqual(os.stat(token).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                deploy.private_write(token, "replacement")
            link = Path(directory) / "link"
            link.symlink_to(token)
            with self.assertRaises(FileExistsError):
                deploy.private_write(link, "replacement")
            self.assertEqual(token.read_text(), "fixture-secret")

    def test_deployment_uses_private_staging_lock_and_bounded_health_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.write_bytes(b"fixture-artifact")
            options = SimpleNamespace(artifact=artifact, bucket="fixture", region="us-west-2", instance_id="i-fixture")
            requests = []
            def fake_aws(args, _options):
                if args[:2] == ["ssm", "send-command"]:
                    request = Path(args[args.index("--cli-input-json") + 1].removeprefix("file://"))
                    requests.append(__import__("json").loads(request.read_text()))
                    return {"Command": {"CommandId": "fixture-command"}}
                return {"Status": "Success"}
            with patch.object(deploy, "aws", side_effect=fake_aws), patch.object(deploy.time, "sleep"):
                deploy.deploy(options)
            command = "\n".join(requests[0]["Parameters"]["commands"])
            self.assertIn("mktemp -d", command)
            self.assertIn("flock", command)
            self.assertIn("check_gateway", command)
            self.assertNotIn("/tmp/mold-relay-", command)

    def test_failed_health_check_restores_previous_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            tools = root / "tools"
            tools.mkdir()
            old = binaries / "mold-relay"
            old.write_text("#!/bin/sh\necho old\n")
            old.chmod(0o755)
            artifact = root / "artifact"
            artifact.write_text("#!/bin/sh\necho new\n")
            artifact.chmod(0o755)
            stubs = {
                "aws": 'cp "$FIXTURE_ARTIFACT" "$6"',
                "systemctl": "exit 0",
                "sleep": "exit 0",
                "flock": "exit 0",
                "curl": 'test "$("$FIXTURE_BIN/mold-relay")" = old',
            }
            for name, body in stubs.items():
                path = tools / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            options = SimpleNamespace(bucket="fixture", region="us-west-2")
            digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
            script = deploy.remote_script(options, digest, "fixture")
            script = script.replace("/usr/local/bin", str(binaries)).replace("/run/mold-relay-deploy", str(root / "private"))
            environment = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                           "FIXTURE_ARTIFACT": str(artifact), "FIXTURE_BIN": str(binaries)}
            result = subprocess.run(["sh", "-c", script], env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(subprocess.check_output([str(old)], text=True).strip(), "old")
            self.assertFalse(list((root / "private").glob("staging.*")))

    def test_access_denied_poll_is_fatal_instead_of_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.write_bytes(b"fixture")
            options = SimpleNamespace(artifact=artifact, bucket="fixture", region="us-west-2", instance_id="i-fixture")
            polls = []
            def fake_aws(args, _options):
                if args[:2] == ["ssm", "send-command"]:
                    return {"Command": {"CommandId": "fixture-command"}}
                if args[:2] == ["ssm", "get-command-invocation"]:
                    polls.append(1)
                    raise deploy.AwsError("ssm", "get-command-invocation", "AccessDeniedException")
                return {}
            with patch.object(deploy, "aws", side_effect=fake_aws), patch.object(deploy.time, "sleep"):
                with self.assertRaises(deploy.AwsError):
                    deploy.deploy(options)
            self.assertEqual(len(polls), 1)

    def test_aws_error_never_includes_response_or_credentials(self):
        result = subprocess.CompletedProcess([], 1, "fixture-secret", "fixture-secret")
        with patch.object(deploy.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError) as failure:
                deploy.aws(["ssm", "get-parameter"], SimpleNamespace(profile="fixture", region="us-west-2"))
        self.assertNotIn("fixture-secret", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
