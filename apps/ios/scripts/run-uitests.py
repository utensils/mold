#!/usr/bin/env python3
"""Run one audit selection, retrying only identified failed tests once.

Uses Xcode's public `xcresulttool get test-results tests` TestNode JSON schema;
never asks xcodebuild to repeat a complete UI test target after one failure.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def test_results(report):
    outcomes = {}

    def visit(node):
        if not isinstance(node, dict):
            raise ValueError('Malformed test node')
        if node.get('result') == 'unknown':
            raise ValueError('Incomplete report node')
        failed_count = 0
        if node.get('nodeType') == 'Test Case':
            result = node.get('result')
            if result not in ('Passed', 'Failed', 'Skipped', 'Expected Failure'):
                raise ValueError('Incomplete or unknown test result')
            identifier = node.get('nodeIdentifierURL', '')
            match = re.fullmatch(r'test://[^/]+/[^/]+/(MoldCompanionUITests)/([A-Za-z_]\w*)/([A-Za-z_]\w*)', identifier)
            if not match:
                raise ValueError('Test has no valid public test identifier')
            test = '/'.join(match.groups())
            if test in outcomes and outcomes[test] != result:
                raise ValueError('Conflicting repeated test results')
            outcomes[test] = result
            failed_count = int(result == 'Failed')
        for child in node.get('children', []):
            failed_count += visit(child)
        if node.get('result') == 'Failed' and not failed_count:
            raise ValueError('Failed report node has no identifiable failed test')
        return failed_count

    if not isinstance(report, dict) or not isinstance(report.get('testNodes'), list):
        raise ValueError('Missing public testNodes report')
    for node in report['testNodes']:
        visit(node)
    return outcomes


def failed_tests(report):
    failures = sorted(test for test, result in test_results(report).items() if result == 'Failed')
    if not failures:
        raise ValueError('No identifiable failed tests; refusing an infrastructure retry')
    return failures


def verify_retry(report, failures):
    outcomes = test_results(report)
    if set(outcomes) != set(failures) or any(result != 'Passed' for result in outcomes.values()):
        raise ValueError('Retry did not run and pass exactly the requested failed tests')


def read_report(results):
    report = subprocess.run(['xcrun', 'xcresulttool', 'get', 'test-results', 'tests', '--compact',
                             '--path', str(results) + '.xcresult'], check=True, text=True, capture_output=True)
    return json.loads(report.stdout)


def retry_command(command, failures):
    selections = [arg.removeprefix('-only-testing:') for arg in command if arg.startswith('-only-testing:')]
    if not selections or any(not any(test == selected or test.startswith(selected + '/') for selected in selections) for test in failures):
        raise ValueError('Failure outside the requested test selection')
    return [arg for arg in command if not arg.startswith('-only-testing:')] + [f'-only-testing:{test}' for test in failures]


def run(command, results):
    bundle = Path(str(results) + '.xcresult')
    if bundle.exists():
        shutil.rmtree(bundle)
    with Path(str(results) + '.log').open('w') as log:
        process = subprocess.Popen(command + ['-resultBundlePath', str(bundle)], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if 'error:' in line or re.search(r'TEST (SUCCEEDED|FAILED)', line):
                print(line, end='', flush=True)
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command or any(arg.startswith(('-retry-tests', '-test-iterations', '-test-repetition')) for arg in command):
        parser.error('Provide a single-pass xcodebuild command without implicit repetition')
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    # A fresh successful pass must not upload a stale retry from an older run.
    retry_results = Path(str(results) + '-retry')
    stale_bundle = Path(str(retry_results) + '.xcresult')
    if stale_bundle.exists():
        shutil.rmtree(stale_bundle)
    Path(str(retry_results) + '.log').unlink(missing_ok=True)
    status = run(command, results)
    if status == 0:
        return 0
    if status != 65:  # xcodebuild's test-failure status; never retry interruption/tool errors.
        return status
    try:
        failures = failed_tests(read_report(results))
        retry = retry_command(command, failures)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(f'error: Cannot identify failed tests safely: {error}', file=sys.stderr)
        return status
    print('Retrying failed tests once: ' + ', '.join(failures), flush=True)
    status = run(retry, retry_results)
    if status != 0:
        return status
    try:
        verify_retry(read_report(retry_results), failures)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(f'error: Failed-only retry has no complete passing proof: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
