#!/usr/bin/env python3
"""Bind qualification inputs to the actual scratch server process."""
import json
from pathlib import Path
import sys


def validate_environment(actual, gpu_uuid, profiles, profile):
    if actual.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('scratch server CUDA_VISIBLE_DEVICES does not match the selected GPU')
    expected = profiles[profile]
    for name in {name for values in profiles.values() for name in values}:
        if actual.get(name) != expected.get(name):
            raise ValueError(f'scratch server {name} does not match profile {profile}')


if __name__ == '__main__':
    pid, gpu_uuid, matrix, profile = sys.argv[1:]
    entries = Path(f'/proc/{int(pid)}/environ').read_bytes().split(b'\0')
    actual = dict(entry.decode().split('=', 1) for entry in entries if b'=' in entry)
    profiles = json.loads(Path(matrix).read_text())['common']['profiles']
    try:
        validate_environment(actual, gpu_uuid, profiles, profile)
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
