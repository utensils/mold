#!/usr/bin/env python3
"""Every `#[cfg]` arm of one function must declare the same visibility.

A function written once per platform (`#[cfg(target_os = "macos")] fn f` beside
`#[cfg(not(target_os = "macos"))] fn f`) is only ever compiled one arm at a
time, so the arm the author's machine skips is unchecked until some other
machine builds it. #1744 made the macOS arm of `kernel_pressure_level`
`pub(crate)` for a new caller and left the fallback arm private: every macOS
build and every PR check passed, and `main` went red on the first Linux build
of that feature set, which is push-only.

This reads the source instead of compiling it, so it answers on any host and
for every feature combination at once. Only VISIBILITY is compared: a fallback
arm legitimately renames its parameters to `_name` and a fallback `main` drops
its return type, but no caller can be meant to reach one arm and not the other.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("crates", "desktop/src-tauri", "apps/mobile/src-tauri")
SKIP_PARTS = {"target", "vendor", "node_modules", "gen"}

FN_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<visibility>pub(?:\([^)]*\))?\s+)?"
    r"(?:(?:const|async|unsafe|extern\s+\"[^\"]*\")\s+)*fn\s+(?P<name>\w+)"
)
SCOPE_LINE = re.compile(r"^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:impl|mod|trait)\b")


def cfg_gated_functions(source: str):
    lines = source.splitlines()
    open_attribute: str | None = None  # a `#[cfg(any(` still waiting for its `)]`
    scopes: list[tuple[int, int]] = []  # (indent, line number of the scope header)
    pending_cfg: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if open_attribute is not None:
            open_attribute += " " + stripped
            if open_attribute.count("[") > open_attribute.count("]"):
                continue
            stripped, open_attribute = open_attribute, None
        elif stripped.startswith("#[") and stripped.count("[") > stripped.count("]"):
            open_attribute = stripped
            continue
        indent = len(line) - len(line.lstrip())
        while scopes and scopes[-1][0] >= indent and not stripped.startswith("}"):
            scopes.pop()
        if stripped.startswith("#[cfg(") or stripped.startswith("#[cfg_attr("):
            if stripped.startswith("#[cfg("):
                pending_cfg = stripped
            continue
        if stripped.startswith("#["):
            continue
        if SCOPE_LINE.match(line):
            scopes.append((indent, index))
        found = FN_LINE.match(line)
        if found and pending_cfg is not None:
            scope = scopes[-1][1] if scopes and scopes[-1][0] < indent else -1
            visibility = " ".join((found["visibility"] or "private").split())
            yield (scope, found["name"]), index + 1, pending_cfg, visibility
        pending_cfg = None


SELF_TEST_MISMATCH = """
impl Sampler {
    #[cfg(target_os = "macos")]
    pub(crate) fn level() -> u32 { 1 }

    /// The fallback arm.
    #[cfg(not(
        target_os = "macos"
    ))]
    #[allow(dead_code)]
    fn level() -> u32 { 0 }
}
"""
SELF_TEST_AGREEMENT = SELF_TEST_MISMATCH.replace("    fn level", "    pub(crate) fn level")


def visibilities(source: str) -> list[set[str]]:
    groups: dict[tuple[int, str], set[str]] = defaultdict(set)
    for key, _, _, visibility in cfg_gated_functions(source):
        groups[key].add(visibility)
    return list(groups.values())


def self_test() -> None:
    """A guard that has quietly stopped matching is worse than no guard."""
    if visibilities(SELF_TEST_MISMATCH) != [{"pub(crate)", "private"}]:
        raise SystemExit("FAIL: the scanner no longer sees a visibility mismatch across cfg arms")
    if visibilities(SELF_TEST_AGREEMENT) != [{"pub(crate)"}]:
        raise SystemExit("FAIL: the scanner reports a mismatch between identical cfg arms")


def main() -> int:
    self_test()
    failures: list[str] = []
    arms_checked = 0
    for root in SOURCE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.rs")):
            if SKIP_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            groups: dict[tuple[int, str], list[tuple[int, str, str]]] = defaultdict(list)
            for key, line, cfg, visibility in cfg_gated_functions(
                path.read_text(encoding="utf-8", errors="replace")
            ):
                groups[key].append((line, cfg, visibility))
            for (_, name), arms in groups.items():
                if len(arms) < 2:
                    continue
                arms_checked += len(arms)
                if len({visibility for _, _, visibility in arms}) == 1:
                    continue
                relative = path.relative_to(REPO_ROOT)
                detail = "\n".join(
                    f"    {relative}:{line}  {visibility:<12} {cfg}" for line, cfg, visibility in arms
                )
                failures.append(f"`{name}` is not equally visible across its cfg arms:\n{detail}")
    if failures:
        print("FAIL: cfg arms of one function must share one visibility\n", file=sys.stderr)
        print("\n\n".join(failures), file=sys.stderr)
        print(
            "\nOnly one arm compiles on any one machine. Give every arm the same visibility, or"
            " write the function once and put the cfg inside its body.",
            file=sys.stderr,
        )
        return 1
    if arms_checked == 0:
        print("FAIL: found no cfg-paired functions at all; the scanner is broken", file=sys.stderr)
        return 1
    print(f"PASS: cfg-arm-visibility ({arms_checked} arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
