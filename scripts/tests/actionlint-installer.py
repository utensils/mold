#!/usr/bin/env python3
"""A corrupt download must never reach extraction or installation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class InstallerTest(unittest.TestCase):
    def test_checksum_failure_stops_before_extracting(self):
        installer = ROOT / 'scripts/ci/install-actionlint.sh'
        self.assertTrue(installer.is_file(), 'native installer missing')
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            mocks = scratch / 'bin'
            mocks.mkdir()
            curl = mocks / 'curl'
            curl.write_text('#!/bin/bash\nwhile [[ $# -gt 0 ]]; do\n  if [[ $1 == --output ]]; then printf corrupt > "$2"; exit 0; fi\n  shift\ndone\nexit 2\n')
            tar = mocks / 'tar'
            tar.write_text('#!/bin/bash\ntouch "$ACTIONLINT_TEST_EXTRACTED"\nexit 2\n')
            curl.chmod(0o755)
            tar.chmod(0o755)
            extracted = scratch / 'extracted'
            destination = scratch / 'installed'
            result = subprocess.run(['bash', str(installer), str(destination)],
                                    env={**os.environ, 'PATH': f'{mocks}:{os.environ["PATH"]}',
                                         'ACTIONLINT_TEST_EXTRACTED': str(extracted)},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum', result.stderr.lower())
            self.assertFalse(extracted.exists())
            self.assertFalse((destination / 'actionlint').exists())


if __name__ == '__main__':
    unittest.main()
