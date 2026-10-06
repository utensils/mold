#!/usr/bin/env python3
"""Select app UI audits conservatively; unrelated Widget surfaces need their own UAT."""
import argparse
from pathlib import Path
import subprocess

# These files cannot change the app UI exercised by CompanionUITests. Widget
# Swift changes still compile in the always-on unit lane and require Widget UAT.
STATIC_PATHS = {
    # The native Swift app reads only its marketing version from this Rust
    # manifest. Lint/unit builds validate the stamp before TestFlight delivery.
    'Cargo.toml',
    'README.md',
    'apps/ios/README.md',
    '.claude/rules/ios-native.md',
    '.claude/rules/release-ci.md',
    '.github/workflows/ios-native.yml',
    '.github/workflows/testflight-ios-native.yml',
    'apps/ios/scripts/ci-audit-scope.py',
    'scripts/tests/ios-native-ci-scope.py',
    'scripts/tests/ci-routing-contract.sh',
    'scripts/tests/ios-notification-branding.sh',
}


def unrelated_to_app_audit(path):
    if path in STATIC_PATHS:
        return True
    return ((path.startswith('apps/ios/Sources/Widgets/') and path.endswith('.swift'))
            or (path.startswith('apps/ios/docs/') and path.endswith('.md'))
            or (path.startswith('website/') and path.endswith('.md'))
            or (path.startswith('changelog.d/') and path.endswith('.md')))


def requires_audit(root, event, base, head):
    if event not in ('pull_request', 'push') or not base or not head:
        return True
    try:
        # Full checkout plus local Git avoids API diff limits. A stale base,
        # initial push, missing ref, or unusable diff defaults to the full audit.
        subprocess.run(['git', 'merge-base', '--is-ancestor', base, head], cwd=root,
                       check=True, capture_output=True)
        diff = subprocess.check_output(['git', 'diff', '--no-renames', '--name-only', '-z',
                                        base, head], cwd=root, stderr=subprocess.PIPE)
        paths = diff.decode('utf-8').rstrip('\0').split('\0') if diff else []
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return True
    return not paths or any(not unrelated_to_app_audit(path) for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', required=True)
    parser.add_argument('--base', default='')
    parser.add_argument('--head', default='')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print('audit=' + str(requires_audit(root, args.event, args.base, args.head)).lower())


if __name__ == '__main__':
    main()
