#!/usr/bin/env python3
"""Temporary build-first policy: inspect active YAML separately from retained jobs."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]

def active(name):
    return '\n'.join(line for line in (root / '.github/workflows' / name).read_text().splitlines() if not line.lstrip().startswith('#'))

def require(condition, message):
    if not condition:
        raise AssertionError(message)

ci = active('ci.yml')
require('cargo check --locked --workspace' in ci, 'basic workspace build check missing')
for job in ('rust-default', 'cuda-typecheck', 'metal-check', 'coverage', 'linux-cpu', 'nix-web', 'windows-rust'):
    require(not re.search(r'^  ' + job + r':', ci, re.M), f'{job} still active')
    require('#   ' + job + ':' in (root / '.github/workflows/ci.yml').read_text(), f'{job} restoration definition missing')
require('cargo nextest' not in ci and 'cargo clippy' not in ci, 'expensive Rust gates still active')
require('cargo fmt --all -- --check' in ci and 'actionlint' in ci, 'basic static checks missing')
require('docker://rhysd/actionlint' not in ci, 'static lint depends on rate-limited Docker Hub pulls')
require('bash scripts/ci/install-actionlint.sh' in ci, 'pinned native actionlint installer missing')
require('bash scripts/tests/actionlint-native.sh' in ci, 'external actionlint analyzers are not verified')
desktop = active('desktop.yml')
desktop_frontend = re.search(r'^            frontend:\n(.*?)(?=^            \w+:)', desktop, re.M | re.S)
require(desktop_frontend is not None, 'desktop frontend classifier missing')
for directory in ('composables', 'mobile'):
    require(f"'desktop/src/{directory}/**'" in desktop_frontend.group(1),
            f'desktop frontend classifier omits {directory}')
desktop_push = desktop.split('  push:', 1)[1].split('  pull_request:', 1)[0]
require('"desktop/src/composables/**"' in desktop_push, 'desktop main trigger omits composables')
require('"desktop/src/mobile/**"' not in desktop_push, 'mobile-only changes trigger desktop artifact nightlies')
for name in ('desktop.yml', 'ios.yml', 'macos-native.yml', 'ios-native.yml'):
    require('cargo clippy' not in active(name), f'{name} retains duplicate clippy builds')
require('needs: [desktop-nightly]' in active('desktop.yml'), 'desktop publish waits for non-build gates')
require('needs: [macos-native-nightly]' in active('macos-native.yml'), 'native macOS publish waits for non-build gates')
require('Run Android 35 app and instrumentation tests' not in active('android.yml'), 'Android emulator still blocks builds')
require('Build Android validation APK' in active('android.yml'), 'Android build removed')
require('Wait for exact Apple and Docker candidate delivery' not in active('release-plz.yml'), 'duplicate candidate builds still block tagging')
require(not re.search(r'^  (pull_request|push):', active('docker-validation.yml'), re.M), 'automatic duplicate Docker builds still active')
for name in ('desktop-distribution.yml', 'macos-native-distribution.yml', 'release.yml', 'testflight-ios-native.yml', 'testflight-ios.yml', 'windows-nightly.yml', 'nix-cache.yml'):
    require('jobs:' in active(name), f'{name} shipping builds removed')
print('PASS: temporary build-first routing and retained restoration definitions')
