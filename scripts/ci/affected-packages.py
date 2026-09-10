#!/usr/bin/env python3
"""Decide which workspace packages a change can affect, for diff-scoped CI.

Usage:
  affected-packages.py --base SHA --head SHA        # paths from `git diff`
  affected-packages.py --paths-from FILE            # one path per line ("-" = stdin)
  affected-packages.py --all                        # the whole workspace (main pushes)

Prints, one per line, `key=value` pairs suitable for `$GITHUB_OUTPUT`:
  packages=<cargo package selector args>   e.g. `-p mold-ai-core -p mold-ai`
                                           or `--workspace`
  scope=workspace|packages|none            `none` means no Rust package is touched
  names=<space-separated package names>    empty for `none`, every member for `workspace`
  features=<pkg/feature,...>               the optional-feature union, limited to the
                                           selected packages (cargo refuses the rest)

Rules (deliberately conservative -- a wrong answer here hides a red test):
  * A path under `crates/<dir>/` maps to that crate's package. Every workspace
    member that depends on it -- transitively, dev-dependencies included, which
    is what `cargo metadata`'s resolve graph reports -- is affected too.
  * Any path that can change how EVERYTHING compiles or is tested widens the
    answer to the whole workspace: the root `Cargo.toml`/`Cargo.lock`, `.cargo/`,
    the toolchain pin, `.config/nextest.toml`, this script, `.github/workflows/`,
    `vendor/`, any `build.rs` or `build_support/`, and `tests/fixtures/` (crates
    `include_bytes!` from there).
  * `include_str!` reaching ACROSS crates is real in this repo (mold-server pins
    mold-cli and mold-inference source text; mold-cli pins mold-server's). Those
    are dependency edges the Cargo graph does not know about, so they are listed
    explicitly in EXTRA_EDGES; the contract test fails if a new one appears
    that is not listed.
  * Everything else (docs, website, frontends, changelog fragments) touches no
    Rust package.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

WORKSPACE_WIDE = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
    "clippy.toml",
    "rustfmt.toml",
    "deny.toml",
    ".config/nextest.toml",
    "scripts/ci/affected-packages.py",
)
WORKSPACE_WIDE_PREFIXES = (
    ".cargo/",
    ".github/workflows/",
    "vendor/",
    "tests/fixtures/",
)

# Source-text pins: `include_str!("../../<crate>/...")` from one crate into another.
# The INCLUDING crate is affected when the INCLUDED crate's directory changes.
EXTRA_EDGES = {
    # included crate dir -> including package names
    "crates/mold-catalog": ("mold-ai-server",),
    "crates/mold-cli": ("mold-ai-server",),
    "crates/mold-inference": ("mold-ai-server", "mold-ai"),
    "crates/mold-server": ("mold-ai",),
}


def changed_paths(args):
    if args.paths_from:
        source = sys.stdin if args.paths_from == "-" else open(args.paths_from)
        return [line.strip() for line in source if line.strip()]
    out = subprocess.run(
        ["git", "diff", "--name-only", args.base, args.head],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def workspace_graph():
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    members = set(meta["workspace_members"])
    by_id = {p["id"]: p for p in meta["packages"] if p["id"] in members}
    dir_to_name = {}
    for pkg in by_id.values():
        rel = Path(pkg["manifest_path"]).resolve().relative_to(REPO_ROOT).parent
        dir_to_name[str(rel)] = pkg["name"]
    # Reverse edges among members from the DECLARED dependencies, so optional
    # deps behind a feature (`mold-ai`'s `tui = ["dep:mold-tui"]`) and
    # dev-dependencies count: the resolve graph would drop the optional ones
    # under default features and hide a real dependent.
    member_names = {pkg["name"] for pkg in by_id.values()}
    dependents = {name: set() for name in member_names}
    for pkg in by_id.values():
        for dep in pkg["dependencies"]:
            if dep["name"] in member_names:
                dependents[dep["name"]].add(pkg["name"])
    return dir_to_name, dependents


def is_workspace_wide(path):
    if path in WORKSPACE_WIDE:
        return True
    if path.startswith(WORKSPACE_WIDE_PREFIXES):
        return True
    name = path.rsplit("/", 1)[-1]
    return name == "build.rs" or "/build_support/" in path


def affected(paths, dir_to_name, dependents):
    direct = set()
    for path in paths:
        if is_workspace_wide(path):
            return "workspace", set(dependents)
        m = re.match(r"^(crates/[^/]+)/", path)
        if not m:
            continue
        crate_dir = m.group(1)
        if crate_dir in dir_to_name:
            direct.add(dir_to_name[crate_dir])
        for including in EXTRA_EDGES.get(crate_dir, ()):
            direct.add(including)
    if not direct:
        return "none", set()
    result = set()
    stack = list(direct)
    while stack:
        name = stack.pop()
        if name in result:
            continue
        result.add(name)
        stack.extend(dependents.get(name, ()))
    if result == set(dependents):
        return "workspace", result
    return "packages", result


# The optional features a pull request turns on so gated code (PuLID, mDNS,
# mesh preparation, the private H3 record runtime, every `mold` CLI feature) is
# linted and tested in the SAME build as everything else instead of one
# recompile per feature. `mold-ai-inference/h3` is deliberately absent: it
# flips `private_h3_record_runtime()` off, and the record-runtime arm is what
# the private H3 foundations exercise; `main` covers `h3` in its own step.
# Cargo refuses a `pkg/feature` for a package outside the selection, so the
# list is filtered to the selected packages.
FEATURE_UNION = {
    "mold-ai-core": ("pulid",),
    "mold-ai-inference": (
        "pulid",
        "mesh-texture",
        "mesh-matting",
        "mesh-delight",
        "h3-private-uat",
    ),
    "mold-ai-server": ("pulid", "mdns"),
    "mold-ai": (
        "preview",
        "discord",
        "expand",
        "tui",
        "webp",
        "mp4",
        "mdns",
        "pulid",
        "metrics",
    ),
}


def feature_flags(names):
    return ",".join(
        f"{pkg}/{feature}"
        for pkg in sorted(FEATURE_UNION)
        if pkg in names
        for feature in sorted(FEATURE_UNION[pkg])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--paths-from")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.paths_from and not (args.base and args.head):
        parser.error("give --base and --head, --paths-from, or --all")
    dir_to_name, dependents = workspace_graph()
    if args.all:
        scope, names = "workspace", set(dependents)
    else:
        scope, names = affected(changed_paths(args), dir_to_name, dependents)
    ordered = sorted(names)
    if scope == "workspace":
        selector = "--workspace"
    elif scope == "packages":
        selector = " ".join(f"-p {n}" for n in ordered)
    else:
        selector = ""
    print(f"scope={scope}")
    print(f"packages={selector}")
    print(f"names={' '.join(ordered)}")
    print(f"features={feature_flags(names)}")


if __name__ == "__main__":
    main()
