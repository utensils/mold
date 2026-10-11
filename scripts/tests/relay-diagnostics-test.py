#!/usr/bin/env python3
"""Read-only diagnostics contracts: fixtures only, no AWS calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("relay_diagnostics", Path(__file__).parents[1] / "relay/diagnose.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticsSafety(unittest.TestCase):
    def test_host_id_rejects_query_injection(self):
        for value in ["host", "a" * 31, "a" * 32 + "' | fields @message", "A" * 32]:
            with self.assertRaises(ValueError):
                diag.host_key(value)
        self.assertEqual(diag.host_key("a" * 32), "v3-host#" + "a" * 32)

    def test_current_host_scope_never_queries_all_hosts(self):
        self.assertIsNone(diag.diagnostic_query([]))
        query = diag.diagnostic_query(["abc=", "xyz="])
        self.assertIn("abc=", query)
        self.assertIn("xyz=", query)
        self.assertIn("stats count()", query)
        self.assertNotIn("fields @message", query)
        with self.assertRaises(ValueError):
            diag.diagnostic_query(["x/ | fields @message"])

    def test_host_summary_excludes_credentials_and_connection_ids(self):
        result = diag.host_summary({"connectionId": "private-id", "sid": "private-session",
            "expiresAt": 110, "guests": {"guest": {"expiresAt": 105}, "old": {"expiresAt": 99}},
            "token": "never-output"}, 100)
        self.assertEqual(result, {"present": True, "ready": True, "lease_seconds_remaining": 10,
                                  "live_guests": 1})
        self.assertNotIn("private", json.dumps(result))

    def test_logs_return_only_allowlisted_aggregate_fields(self):
        rows = [[{"field": "event", "value": "failure"},
                 {"field": "reason", "value": "secret-token"},
                 {"field": "count", "value": "3"},
                 {"field": "@message", "value": "authorization secret"}]]
        self.assertEqual(diag.log_summary(rows), [{"event": "failure", "reason": "other", "count": 3}])

    def test_aws_mutation_refused_before_process_execution(self):
        with patch.object(diag.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                diag.aws(["lambda", "update-function-code"], SimpleNamespace(profile="fixture", region="us-west-2"))
            run.assert_not_called()

    def test_aws_failure_never_echoes_raw_response(self):
        result = SimpleNamespace(returncode=1, stdout="secret", stderr="An error occurred (Denied) secret")
        with patch.object(diag.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Denied") as caught:
                diag.aws(["lambda", "get-function-configuration"], SimpleNamespace(profile="fixture", region="us-west-2"))
            self.assertNotIn("secret", str(caught.exception))

    def test_metric_summary_preserves_sum_and_percentile_meaning(self):
        value = diag.metric_summary({"Label": "frontend Duration p95", "Values": [5, 10], "StatusCode": "Complete"})
        self.assertEqual(value["maximum_period_value"], 10)
        self.assertNotIn("sum", value)
        value = diag.metric_summary({"Label": "frontend Throttles Sum", "Values": [5, 10], "StatusCode": "Complete"})
        self.assertEqual(value["sum"], 15)


if __name__ == "__main__":
    unittest.main()
