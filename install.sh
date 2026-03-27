#!/bin/sh
# Install mold — local AI image generation CLI
# https://github.com/utensils/mold
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
#
# Options (via environment):
#   MOLD_INSTALL_DIR  — install directory (default: ~/.local/bin)
#   MOLD_VERSION      — release tag (default: latest)

set -e

REPO="utensils/mold"
VERSION="${MOLD_VERSION:-latest}"
INSTALL_DIR="${MOLD_INSTALL_DIR:-$HOME/.local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Linux)
        case "${ARCH}" in
            x86_64)  ASSET="mold-x86_64-unknown-linux-gnu-cuda.tar.gz" ;;
            *)       echo "Error: unsupported Linux architecture: ${ARCH}" >&2; exit 1 ;;
        esac
        ;;
    Darwin)
        case "${ARCH}" in
            arm64)   ASSET="mold-aarch64-apple-darwin.tar.gz" ;;
            *)       echo "Error: unsupported macOS architecture: ${ARCH}" >&2; exit 1 ;;
        esac
        ;;
    *)
        echo "Error: unsupported OS: ${OS}" >&2
        exit 1
        ;;
esac

URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

echo "Installing mold (${VERSION}) for ${OS}/${ARCH}..."
echo "  from: ${URL}"
echo "  to:   ${INSTALL_DIR}/mold"

# Create install directory
mkdir -p "${INSTALL_DIR}"

# Download, extract, and install
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

curl -fsSL "${URL}" -o "${TMPDIR}/${ASSET}"
tar -xzf "${TMPDIR}/${ASSET}" -C "${TMPDIR}"
install -m 755 "${TMPDIR}/mold" "${INSTALL_DIR}/mold"

# macOS: remove quarantine attribute if present
if [ "${OS}" = "Darwin" ]; then
    xattr -d com.apple.quarantine "${INSTALL_DIR}/mold" 2>/dev/null || true
fi

echo ""
echo "mold installed to ${INSTALL_DIR}/mold"

# Check if install dir is in PATH
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo ""
        echo "Add ${INSTALL_DIR} to your PATH:"
        echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
        echo ""
        echo "Or add it to your shell profile (~/.bashrc, ~/.zshrc, etc.)"
        ;;
esac

"${INSTALL_DIR}/mold" version
