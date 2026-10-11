#!/usr/bin/env python3
"""Deployment contract tests; no AWS calls or real secrets."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("relay_deploy", Path(__file__).parents[1] / "relay/deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentSafety(unittest.TestCase):
    def options(self, **kwargs):
        return SimpleNamespace(profile="fixture", region="us-west-2", **kwargs)




    def package(self, path, role):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("index.mjs", "fixture")
            archive.writestr("mold-relay-contract.json", json.dumps({"protocol": 3, "role": role, "security": "noise-only"}))

    def test_registration_uses_public_key_derived_namespace_iam_and_conditional_create(self):
        calls = []
        def fake(args, _):
            calls.append(args)
            if args[1] == "get-item":
                return {}
            request = json.loads(Path(args[-1].removeprefix("file://")).read_text())
            row = json.loads(request["Item"]["document"]["S"])
            self.assertEqual(row["signingKey"], "12" * 32)
            self.assertEqual(row["principalId"], "operator-one")
            self.assertTrue(row["enabled"])
            self.assertEqual(request["Item"]["id"]["S"], "v3-registration#" + deploy.hashlib.sha256(bytes.fromhex("12" * 32)).hexdigest()[:32])
            self.assertNotIn("expiresAt", request["Item"])
            self.assertEqual(request["ConditionExpression"], "attribute_not_exists(id)")
            return {}
        with patch.object(deploy, "aws", side_effect=fake):
            deploy.provision_host(self.options(host_key="12" * 32, principal="operator-one", disable=False, table="fixture"))
        self.assertEqual([x[:2] for x in calls], [["dynamodb", "get-item"], ["dynamodb", "put-item"]])

    def test_registration_never_reassigns_owner_or_accepts_private_key_shapes(self):
        opts = self.options(host_key="12" * 32, principal="new-owner", disable=False, table="fixture")
        existing = {"Item": {"document": {"S": json.dumps({"principalId": "old-owner"})}, "revision": {"N": "1"}}}
        with patch.object(deploy, "aws", return_value=existing) as call:
            with self.assertRaisesRegex(RuntimeError, "refusing reassignment"):
                deploy.provision_host(opts)
            self.assertEqual(call.call_count, 1)
        opts.host_key = "12" * 64
        with patch.object(deploy, "aws") as call:
            with self.assertRaises(RuntimeError):
                deploy.provision_host(opts)
            call.assert_not_called()

    def test_rollback_refuses_plaintext_pre_cutover_before_any_aws_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "before-router.zip"
            old.write_bytes(b"old plaintext code")
            deploy.private_write(root / "rollback.json", json.dumps({"profile": "fixture", "region": "us-west-2", "functions": {"router": {"before": deploy.package_hash(old), "deployed": "new"}}}))
            with patch.object(deploy, "aws", return_value={"CodeSha256": deploy.package_hash(old), "State": "Active", "LastUpdateStatus": "Successful"}) as call:
                with self.assertRaisesRegex(RuntimeError, "Pre-cutover"):
                    deploy.rollback(self.options(), root)
                call.assert_not_called()

    def test_no_plaintext_token_or_cloud_shell_provisioning_remains(self):
        for function in ["read_token", "token_preflight", "provision_token", "shell_files", "upload_shell"]:
            self.assertFalse(hasattr(deploy, function), function)

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


    def test_rollback_refuses_concurrent_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "before-router.zip"
            self.package(package, "router")
            deploy.private_write(root / "rollback.json", json.dumps({"profile": "fixture", "region": "us-west-2", "functions": {"router": {"before": deploy.package_hash(package), "deployed": "expected", "before_secure_protocol": True}}}))
            with patch.object(deploy, "aws", return_value={"CodeSha256": "concurrent"}) as call:
                with self.assertRaises(RuntimeError):
                    deploy.rollback(self.options(), root)
                self.assertEqual(call.call_count, 1)

    def test_retained_code_hash_verified_and_url_never_saved(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "router.zip"
            package.write_bytes(b"new")
            old = root / "old.zip"
            old.write_bytes(b"old")
            response = {"Configuration": {"CodeSha256": deploy.package_hash(old)}, "Code": {"Location": "https://fixture.invalid/private-presigned"}}
            with patch.object(deploy, "aws", return_value=response), patch.object(deploy.urllib.request, "urlopen", return_value=BytesIO(b"old")):
                deploy.retain_rollback(self.options(output=root), {"router": package})
            self.assertEqual((root / "before-router.zip").stat().st_mode & 0o777, 0o600)
            self.assertNotIn("presigned", (root / "rollback.json").read_text())



if __name__ == "__main__":
    unittest.main()
