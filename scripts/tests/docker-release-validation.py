#!/usr/bin/env python3
"""Exercise candidate selection and the fail-closed Docker release gate."""

import fnmatch
import os
import re
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / '.github/workflows/docker-validation.yml').read_text()


def run_step(name):
    section = WORKFLOW.split(f'- name: {name}\n', 1)[1]
    lines = section.split('        run: |\n', 1)[1].splitlines()
    body = []
    for line in lines:
        if line.strip() and not line.startswith('          '):
            break
        body.append(line)
    return textwrap.dedent('\n'.join(body))


class DockerReleaseValidation(unittest.TestCase):
    def select(self, event, tagged):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / 'Cargo.toml').write_text('[workspace.package]\nversion = "1.2.3"\n')
            for args in [('init', '-q'), ('add', 'Cargo.toml'),
                         ('-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                          'commit', '-qm', 'fixture')]:
                subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)
            if tagged:
                subprocess.run(['git', 'tag', 'v1.2.3'], cwd=cwd, check=True)
            output = cwd / 'output'
            subprocess.run(['bash', '-euc', run_step('Select validation targets')],
                           cwd=cwd, check=True, capture_output=True,
                           env={**os.environ, 'EVENT_NAME': event, 'GITHUB_OUTPUT': str(output)})
            return dict(line.split('=', 1) for line in output.read_text().splitlines())

    def test_pr_builds_only_for_container_inputs(self):
        pr = WORKFLOW.split('  pull_request:', 1)[1].split('  push:', 1)[0]
        patterns = re.findall(r"^      - '([^']+)'", pr, re.M)
        cases = {
            'scripts/tests/ios-uitest-runner.sh': False,
            'scripts/tests/ci-routing-contract.sh': False,
            'scripts/tests/docker-release-validation.py': False,
            '.github/workflows/docker-validation.yml': False,
            '.github/workflows/ci.yml': False,
            'apps/ios/Sources/LibraryView.swift': False,
            'Dockerfile': True,
            'docker/start.sh': True,
            'crates/mold-server/src/lib.rs': True,
            'web/src/main.ts': True,
            'relay/aws/package.json': True,
            'scripts/seal-cuda-ptx-manifest.py': True,
            'scripts/probe-cuda-embedded-ptx.py': True,
            'scripts/verify-h3-release-exclusion.sh': True,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(any(fnmatch.fnmatchcase(path, p) for p in patterns), expected)
        push = WORKFLOW.split('  push:', 1)[1].split('  workflow_dispatch:', 1)[0]
        self.assertNotIn('paths:', push)

    def test_pr_builds_both_feature_paths_even_with_existing_tag(self):
        result = self.select('pull_request', True)
        self.assertEqual(result['build'], 'true')
        self.assertEqual(result['targets'], '["89","120"]')

    def test_unpublished_main_candidate_builds_all_targets(self):
        result = self.select('push', False)
        self.assertEqual(result['targets'], '["80","86","89","90","100","120"]')
        self.assertEqual(result['version'], '1.2.3')

    def test_published_main_does_not_rebuild(self):
        self.assertEqual(self.select('push', True)['build'], 'false')

    def test_manual_validation_builds_all_targets(self):
        self.assertEqual(self.select('workflow_dispatch', True)['build'], 'true')

    def test_aggregate_rejects_failed_cancelled_or_skipped_candidate_builds(self):
        for candidate, required, build, success in [
            ('success', 'true', 'success', True),
            ('success', 'true', 'failure', False),
            ('success', 'true', 'cancelled', False),
            ('success', 'true', 'skipped', False),
            ('failure', '', 'skipped', False),
            ('cancelled', '', 'skipped', False),
            ('success', 'false', 'skipped', True),
        ]:
            with self.subTest(candidate=candidate, required=required, build=build):
                result = subprocess.run(
                    ['bash', '-euc', run_step('Require every selected target to pass')],
                    env={**os.environ, 'CANDIDATE_RESULT': candidate,
                         'BUILD_REQUIRED': required, 'BUILD_RESULT': build},
                    capture_output=True)
                self.assertEqual(result.returncode == 0, success)

    def test_tag_waits_for_exact_sha_and_mints_fresh_token(self):
        release = (ROOT / '.github/workflows/release-plz.yml').read_text().split(
            '  release-plz-release:', 1)[1]
        self.assertIn('docker-validation.yml|push|Docker validation complete', release)
        self.assertIn('--commit "$GITHUB_SHA"', release)
        self.assertIn('timeout-minutes: 210', release)
        self.assertIn('deadline=$((SECONDS + 10800))', release)
        self.assertIn('"$SECONDS" -ge "$deadline"', release)
        self.assertLess(release.index('Docker validation complete'),
                        release.index('- name: Generate app token'))
        self.assertLess(release.index('- name: Generate app token'),
                        release.index('- name: Run release-plz release'))

    def test_protoc_is_installed_in_builder_not_only_runtime(self):
        builder = (ROOT / 'Dockerfile').read_text().split(' AS builder', 1)[1].split('\nFROM ', 1)[0]
        packages = builder.split('apt-get install', 1)[1].split('&& break', 1)[0]
        self.assertIn('protobuf-compiler', packages)
        self.assertIn('protoc --version', builder)

    def test_both_matrices_preserve_sibling_results(self):
        release = (ROOT / '.github/workflows/release.yml').read_text().split('  docker:', 1)[1]
        self.assertIn('fail-fast: false', release.split('    steps:', 1)[0])
        self.assertIn('fail-fast: false', WORKFLOW)
        self.assertIn('push: false', WORKFLOW)
        self.assertIn("cache-to: ${{ github.event_name == 'push' &&", WORKFLOW)


if __name__ == '__main__':
    unittest.main()
