#!/usr/bin/env python3
"""Retry selection against the public xcresulttool TestNode JSON contract."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ios_retry', ROOT / 'apps/ios/scripts/run-uitests.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(*cases):
    return {'testNodes': [{'nodeType': 'UI test bundle', 'name': 'MoldCompanionUITests', 'children': list(cases)}]}


def case(name='HiddenCollectionTests/testExtraSmall', result='Failed'):
    return {'nodeType': 'Test Case', 'result': result,
            'nodeIdentifierURL': f'test://com.apple.xcode/MoldCompanion/MoldCompanionUITests/{name}'}


class RetrySelection(unittest.TestCase):
    def test_public_failed_leaf_ids_only_and_deduplicated(self):
        self.assertEqual(module.failed_tests(report(case(), case(), case('ShellAccessibilityTests/testDefaultLarge', 'Passed'))),
                         ['MoldCompanionUITests/HiddenCollectionTests/testExtraSmall'])

    def test_no_failures_unknown_result_or_bad_id_fail_closed(self):
        for value in ({}, report(case(result='Passed')), report(case(result='unknown')),
                      report({'nodeType': 'Test Case', 'result': 'Failed'}), report(case('wrong/path/extra'))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.failed_tests(value)

    def test_unparseable_failed_sibling_cannot_be_partially_retried(self):
        for sibling in ({'nodeType': 'Test Suite', 'result': 'Failed', 'name': 'Infrastructure failure'},
                        {'nodeType': 'Test Suite', 'result': 'unknown', 'name': 'Interrupted'},
                        {'nodeType': 'Test Case', 'result': 'Failed'}):
            with self.subTest(sibling=sibling), self.assertRaises(ValueError):
                module.failed_tests(report(case(), sibling))

    def test_retry_requires_every_requested_test_to_run_and_pass(self):
        failures = ['MoldCompanionUITests/HiddenCollectionTests/testExtraSmall']
        module.verify_retry(report(case(result='Passed')), failures)
        for value in (report(), report(case()), report(case(result='Skipped')),
                      report(case(result='Passed'), case('ShellAccessibilityTests/testDefaultLarge', 'Passed'))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.verify_retry(value, failures)

    def test_retry_removes_full_suite_selectors_and_targets_only_failure(self):
        cmd = ['xcodebuild', 'test', '-only-testing:MoldCompanionUITests', '-destination', 'fixture']
        retried = module.retry_command(cmd, ['MoldCompanionUITests/HiddenCollectionTests/testExtraSmall'])
        self.assertNotIn('-only-testing:MoldCompanionUITests', retried)
        self.assertIn('-only-testing:MoldCompanionUITests/HiddenCollectionTests/testExtraSmall', retried)
        self.assertEqual(retried.count('xcodebuild'), 1)

    def test_retry_refuses_failure_outside_selected_shard(self):
        with self.assertRaises(ValueError):
            module.retry_command(['xcodebuild', 'test', '-only-testing:MoldCompanionUITests/ShellAccessibilityTests'],
                                 ['MoldCompanionUITests/HiddenCollectionTests/testExtraSmall'])


if __name__ == '__main__':
    unittest.main()
