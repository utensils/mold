#!/usr/bin/env bash
# Generate test coverage report using cargo-llvm-cov.
#
# Usage:
#   ./scripts/coverage.sh          # Print summary to terminal
#   ./scripts/coverage.sh --html   # Generate HTML report in target/coverage/html/
#
# Requires: cargo-llvm-cov (cargo install cargo-llvm-cov) + llvm-tools-preview
# In the Nix devshell, LLVM tools are auto-detected from the Nix store.

set -euo pipefail

# Auto-detect LLVM tools from Nix store if not already set
if [ -z "${LLVM_COV:-}" ]; then
    LLVM_COV=$(find /nix/store -maxdepth 3 -name "llvm-cov" 2>/dev/null | head -1 || true)
fi
if [ -z "${LLVM_PROFDATA:-}" ]; then
    LLVM_PROFDATA=$(find /nix/store -maxdepth 3 -name "llvm-profdata" 2>/dev/null | head -1 || true)
fi

if [ -z "$LLVM_COV" ] || [ -z "$LLVM_PROFDATA" ]; then
    echo "Error: llvm-cov/llvm-profdata not found."
    echo "Install with: rustup component add llvm-tools-preview"
    exit 1
fi

export LLVM_COV LLVM_PROFDATA

if [ "${1:-}" = "--html" ]; then
    cargo llvm-cov --workspace --html --no-cfg-coverage --output-dir target/coverage
    echo "Coverage report: target/coverage/html/index.html"
else
    cargo llvm-cov --workspace --no-cfg-coverage --skip-functions
fi
