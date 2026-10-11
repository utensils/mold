#!/usr/bin/env python3
"""Isolated staging resource and IAM contracts; no AWS calls."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
import json
sys.path.insert(0, str(Path(__file__).parents[1] / "relay"))
import staging

class StagingBounds(unittest.TestCase):
    def test_only_temporary_named_lambda_websocket_state_and_logs(self):
        resources = staging.template()["Resources"]
        self.assertEqual(resources["Router"]["Properties"]["FunctionName"], "mold-relay-v3-uat-router")
        self.assertEqual(resources["Sessions"]["Properties"]["TableName"], "mold-relay-v3-uat-sessions")
        self.assertEqual(resources["Router"]["Properties"]["ReservedConcurrentExecutions"], 32)
        self.assertEqual(resources["Router"]["Properties"]["Timeout"], 10)
        self.assertEqual(resources["Logs"]["Properties"]["RetentionInDays"], 1)
        self.assertTrue(all(r["Type"] not in ("AWS::EC2::Instance", "AWS::Route53::RecordSet") for r in resources.values()))
        bucket=resources['Ciphertext']['Properties']
        self.assertTrue(all(bucket['PublicAccessBlockConfiguration'].values()))
        self.assertEqual(bucket['LifecycleConfiguration']['Rules'][0]['Prefix'],'ciphertext/v3/')
        self.assertNotIn('*',bucket['CorsConfiguration']['CorsRules'][0]['AllowedOrigins'])

    def test_update_refuses_mismatched_manifest_before_any_aws_call(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "resources.json").write_text(json.dumps({"profile":"other", "region":"us-west-2", "stackId":"arn:aws:cloudformation:us-west-2:123:stack/mold-relay-v3-uat/example"}))
            with mock.patch.object(sys, "argv", ["staging", "update", "--output", folder]), mock.patch.object(staging.deploy, "aws") as aws:
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    staging.main()
                aws.assert_not_called()

    def test_update_refuses_unmarked_code_before_any_aws_call(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "resources.json").write_text(json.dumps({"profile":"dev.urandom.io", "region":"us-west-2", "stackId":"arn:aws:cloudformation:us-west-2:123:stack/mold-relay-v3-uat/example"}))
            with mock.patch.object(sys, "argv", ["staging", "update", "--package", str(Path(folder,"missing.zip")), "--output", folder]), mock.patch.object(staging.deploy, "aws") as aws:
                with self.assertRaisesRegex(RuntimeError, "verified protocol-3"):
                    staging.main()
                aws.assert_not_called()

    def test_runtime_cannot_write_registration_or_read_legacy_content(self):
        statements = staging.template()["Resources"]["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        actions = [a for s in statements for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])]
        self.assertFalse(any(a.startswith(("ssm:", "lambda:")) for a in actions))
        objects=next(s for s in statements if 's3:PutObject' in s['Action'])
        self.assertEqual(objects['Resource']['Fn::Sub'],'${Ciphertext.Arn}/ciphertext/v3/*')
        self.assertNotIn('s3:ListBucket',actions)
        writes = next(s for s in statements if "dynamodb:PutItem" in s["Action"])
        self.assertNotIn("v3-registration#*", writes["Condition"]["ForAllValues:StringLike"]["dynamodb:LeadingKeys"])

if __name__ == "__main__":
    unittest.main()
