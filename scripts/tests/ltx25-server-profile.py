#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'validate-ltx25-server-profile.py'
spec = importlib.util.spec_from_file_location('profile_contract', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ProfileContract(unittest.TestCase):
    def test_execution_matches_physical_gpu_and_declared_profile(self):
        profiles = {'default': {'MOLD_ATTN': 'math'}, 'flash': {'MOLD_ATTN': 'flash'},
                    'qmatmul': {'MOLD_ATTN': 'math', 'MOLD_LTX2_QMATMUL': '1'}}
        actual = {'CUDA_VISIBLE_DEVICES': 'GPU-selected', 'MOLD_ATTN': 'math'}
        module.validate_environment(actual, 'GPU-selected', profiles, 'default')
        for changed in [dict(actual, CUDA_VISIBLE_DEVICES='GPU-other'),
                        dict(actual, MOLD_ATTN='flash'),
                        dict(actual, MOLD_LTX2_QMATMUL='1'),
                        {'CUDA_VISIBLE_DEVICES': 'GPU-selected'}]:
            with self.assertRaises(ValueError):
                module.validate_environment(changed, 'GPU-selected', profiles, 'default')
        module.validate_environment(dict(actual, MOLD_LTX2_QMATMUL='1'),
                                    'GPU-selected', profiles, 'qmatmul')

if __name__ == '__main__':
    unittest.main()
