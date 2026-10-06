#!/usr/bin/env python3
"""Exercise conservative native iOS audit routing with complete local Git diffs."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audit_scope', ROOT / 'apps/ios/scripts/ci-audit-scope.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditScope(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.com')
        for name in ['apps/ios/Sources/App/Existing.swift', 'apps/ios/Sources/Widgets/Existing.swift']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('old\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True)

    def change(self, *names):
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('changed\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'change')
        return self.git('rev-parse', 'HEAD').strip()

    def test_widget_docs_and_exact_static_contracts_skip_app_audit(self):
        head = self.change('apps/ios/Sources/Widgets/GenerationLiveActivity.swift', 'apps/ios/README.md',
                           'changelog.d/ios-widget.md', 'scripts/tests/ios-notification-branding.sh',
                           '.github/workflows/ios-native.yml', 'apps/ios/scripts/ci-audit-scope.py',
                           'scripts/tests/ios-native-ci-scope.py', 'scripts/tests/ci-routing-contract.sh')
        for event in ['pull_request', 'push']:
            self.assertFalse(module.requires_audit(self.root, event, self.base, head))

    def test_app_shared_ui_tests_build_inputs_and_unknown_paths_force_audit(self):
        for name in ['apps/ios/Sources/App/Library.swift', 'apps/ios/Sources/Shared/Snapshot.swift',
                     'apps/shared/Packages/MoldClient/Sources/API.swift',
                     'apps/ios/Tests/CompanionUITests/NewTests.swift', 'apps/ios/Makefile',
                     'apps/ios/project.yml', 'apps/ios/Sources/Widgets/Info.plist',
                     'apps/ios/Sources/Widgets/Widget.entitlements', 'apps/ios/scripts/run-uitests.py',
                     '.github/workflows/release.yml', 'scripts/tests/something-new.py']:
            with self.subTest(name=name):
                head = self.change(name)
                self.assertTrue(module.requires_audit(self.root, 'pull_request', self.base, head))
                self.git('reset', '--hard', self.base)

    def test_workspace_version_triggers_native_delivery_without_ui_audits(self):
        head = self.change('Cargo.toml')
        for event in ['pull_request', 'push']:
            self.assertFalse(module.requires_audit(self.root, event, self.base, head))
        workflow = (ROOT / '.github/workflows/ios-native.yml').read_text()
        push, pull_request = workflow.split('  pull_request:', 1)
        self.assertIn('- "Cargo.toml"', push)
        self.assertIn('- "Cargo.toml"', pull_request.split('  workflow_dispatch:', 1)[0])

    def test_mixed_widget_and_app_changes_require_full_audit(self):
        head = self.change('apps/ios/Sources/Widgets/GenerationLiveActivity.swift',
                           'apps/ios/Sources/App/Library.swift')
        self.assertTrue(module.requires_audit(self.root, 'pull_request', self.base, head))

    def test_dispatch_invalid_missing_empty_and_nonancestor_diffs_force_audit(self):
        head = self.change('apps/ios/Sources/Widgets/GenerationLiveActivity.swift')
        for event, base, target in [('workflow_dispatch', self.base, head),
                                    ('pull_request', '', head), ('push', '0' * 40, head),
                                    ('pull_request', 'missing', head), ('push', self.base, 'missing'),
                                    ('pull_request', head, self.base), ('push', head, head)]:
            with self.subTest(event=event, base=base, target=target):
                self.assertTrue(module.requires_audit(self.root, event, base, target))

    def test_rename_cannot_hide_app_source_in_widget_allowlist(self):
        self.git('mv', 'apps/ios/Sources/App/Existing.swift', 'apps/ios/Sources/Widgets/Moved.swift')
        self.git('commit', '-qm', 'rename')
        self.assertTrue(module.requires_audit(self.root, 'pull_request', self.base, self.git('rev-parse', 'HEAD').strip()))

    def test_large_diff_checks_every_path(self):
        docs = [f'apps/ios/docs/{number}.md' for number in range(350)]
        head = self.change(*docs, 'zzz/app-ui.swift')
        self.assertTrue(module.requires_audit(self.root, 'pull_request', self.base, head))


if __name__ == '__main__':
    unittest.main()
