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
#   MOLD_CUDA_ARCH    — force GPU architecture: sm89 or sm120 (default: auto-detect)

set -e

REPO="utensils/mold"
VERSION="${MOLD_VERSION:-latest}"
INSTALL_DIR="${MOLD_INSTALL_DIR:-$HOME/.local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

# Detect NVIDIA GPU compute capability on Linux
detect_cuda_arch() {
    # Allow explicit override
    if [ -n "${MOLD_CUDA_ARCH}" ]; then
        echo "${MOLD_CUDA_ARCH}"
        return
    fi

    # Query GPU compute capability via nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
        if [ -n "${CC}" ]; then
            # Parse major version: "8.9" -> 8, "12.0" -> 12
            MAJOR="$(echo "${CC}" | cut -d. -f1)"
            if [ "${MAJOR}" -ge 12 ] 2>/dev/null; then
                echo "sm120"
                return
            elif [ "${MAJOR}" -ge 8 ] 2>/dev/null; then
                echo "sm89"
                return
            else
                echo "Error: GPU compute capability ${CC} is too old (requires >= 8.0, RTX 30-series or newer)" >&2
                exit 1
            fi
        fi
    fi

    # No nvidia-smi or no GPU detected — fall back to sm89 with warning
    echo "Warning: could not detect GPU architecture (nvidia-smi not found or no GPU)." >&2
    echo "  Defaulting to sm89 (RTX 40-series). Set MOLD_CUDA_ARCH=sm120 for RTX 50-series." >&2
    echo "sm89"
}

case "${OS}" in
    Linux)
        case "${ARCH}" in
            x86_64)
                CUDA_ARCH="$(detect_cuda_arch)"
                ASSET="mold-x86_64-unknown-linux-gnu-cuda-${CUDA_ARCH}.tar.gz"
                ;;
            *)
                echo "Error: unsupported Linux architecture: ${ARCH}" >&2
                exit 1
                ;;
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
