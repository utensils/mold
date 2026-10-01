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

    def test_aws_error_never_includes_response_or_credentials(self):
        result = subprocess.CompletedProcess([], 1, "fixture-secret", "fixture-secret")
        with patch.object(deploy.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError) as failure:
                deploy.aws(["ssm", "get-parameter"], SimpleNamespace(profile="fixture", region="us-west-2"))
        self.assertNotIn("fixture-secret", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
