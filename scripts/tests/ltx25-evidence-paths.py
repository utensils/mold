#!/usr/bin/env python3
"""Path identity contracts for evidence reached through a bind mount."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "ltx25_validator", Path(__file__).resolve().parents[1] / "validate-ltx25-cuda-report.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

root = Path("/storage/mold/output/campaign")
alias = Path("/mnt/mold/output/campaign")
# A bind mount preserves inode identity but realpath does not rewrite its prefix.
original_samefile = Path.samefile

def samefile(left, right):
    if {str(left), str(right)} == {str(root), str(alias)}:
        return True
    return original_samefile(left, right)

with patch.object(Path, "samefile", samefile):
    assert module.inside(alias / "rows/case/output.mp4", root), "bind alias refused"
    assert not module.inside(alias.parent / "different/output.mp4", root)
    assert not module.inside(Path("/unrelated/output.mp4"), root)
print("LTX-2.5 evidence path identity contract OK")
