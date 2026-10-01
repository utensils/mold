#!/usr/bin/env python3
"""Deployment contract tests; no AWS calls or real secrets."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("relay_deploy", Path(__file__).parents[1] / "relay/deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentSafety(unittest.TestCase):
    def options(self, **kwargs):
        return SimpleNamespace(profile="fixture", region="us-west-2", **kwargs)

    def test_private_file_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            deploy.private_write(path, "fixture")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(deploy.read_token(path), "fixture")
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                deploy.read_token(link)
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                deploy.read_token(path)

    def test_missing_token_uses_private_json_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host"
            calls = []
            def fake(args, _):
                calls.append(args)
                if args[1] == "get-parameter":
                    raise deploy.AwsError("ssm", "get-parameter", "ParameterNotFound")
                request = Path(args[-1].removeprefix("file://"))
                self.assertEqual(request.stat().st_mode & 0o777, 0o600)
                body = json.loads(request.read_text())
                self.assertEqual(body["Name"], "/mold/relay/host-token")
                self.assertEqual(body["Type"], "SecureString")
                self.assertNotIn(body["Value"], " ".join(args))
                self.assertNotIn("Overwrite", body)
                return {}
            with patch.object(deploy, "aws", side_effect=fake):
                deploy.provision_token(self.options(), "host", path)
            self.assertEqual(len(calls), 2)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_existing_token_mismatch_never_rotates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host"
            deploy.private_write(path, "local")
            with patch.object(deploy, "aws", return_value={"Parameter": {"Value": "remote"}}) as call:
                with self.assertRaises(RuntimeError):
                    deploy.provision_token(self.options(), "host", path)
                self.assertEqual(call.call_count, 1)

    def test_code_update_revision_hash_and_configuration_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "router.zip"
            package.write_bytes(b"fixture ZIP")
            before = {"State": "Active", "LastUpdateStatus": "Successful", "CodeSha256": "old", "RevisionId": "revision", "Handler": "index.handler"}
            after = {**before, "CodeSha256": deploy.package_hash(package)}
            with patch.object(deploy, "aws", side_effect=[before, {}, after]) as call:
                deploy.deploy_function(self.options(), "router", package)
                args = call.call_args_list[1].args[0]
                self.assertEqual(args[:2], ["lambda", "update-function-code"])
                self.assertIn("revision", args)
                self.assertNotIn("update-function-configuration", args)

    def test_shell_upload_only_prefix_with_mime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("fixture")
            with patch.object(deploy, "aws", return_value={}) as call:
                deploy.upload_shell(self.options(shell=root, bucket="fixture"))
                args = call.call_args.args[0]
                self.assertEqual(args[args.index("--key") + 1], "shell/index.html")
                self.assertEqual(args[args.index("--content-type") + 1], "text/html")


if __name__ == "__main__":
    unittest.main()
