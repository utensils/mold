#!/bin/sh
# Install mold — local AI image generation CLI
# https://github.com/utensils/mold
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
#
# By default the script installs the latest tagged release from
# https://github.com/utensils/mold/releases/latest.
#
# Options (via environment):
#   MOLD_INSTALL_DIR  — install directory (default: ~/.local/bin)
#   MOLD_VERSION      — release tag to install (default: resolved from the
#                       GitHub "latest release" redirect). Pin to a specific
#                       tag like "v0.8.0" to reproduce an older install.
#   MOLD_CUDA_ARCH    — force GPU architecture: sm89 or sm120 (default: auto-detect)

set -e

REPO="utensils/mold"
# Empty / "latest" means "whatever /releases/latest currently redirects to".
# We resolve this to a concrete tag name below so the status line reports the
# real version and the download URL is deterministic.
VERSION="${MOLD_VERSION:-latest}"
INSTALL_DIR="${MOLD_INSTALL_DIR:-$HOME/.local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

# Resolve MOLD_VERSION=latest (the default) to the tag GitHub currently
# considers the latest release. We follow the redirect from
# `/releases/latest` (which points at `/releases/tag/<tag>`) rather than
# calling the JSON API so no auth token is ever needed.
resolve_latest_tag() {
    LATEST_URL="https://github.com/${REPO}/releases/latest"
    RESOLVED_URL="$(curl -fsSILo /dev/null -w '%{url_effective}' "${LATEST_URL}" 2>/dev/null || true)"

    # Expected: https://github.com/<owner>/<repo>/releases/tag/<tag>
    case "${RESOLVED_URL}" in
        */releases/tag/*)
            RESOLVED_TAG="${RESOLVED_URL##*/tag/}"
            RESOLVED_TAG="${RESOLVED_TAG%%/*}"
            ;;
        *)
            RESOLVED_TAG=""
            ;;
    esac

    if [ -z "${RESOLVED_TAG}" ]; then
        echo "Error: could not resolve the latest mold release from ${LATEST_URL}" >&2
        echo "  Set MOLD_VERSION=<tag> to install a specific version," >&2
        echo "  e.g. MOLD_VERSION=v0.9.0 curl ... | sh" >&2
        exit 1
    fi
    echo "${RESOLVED_TAG}"
}

if [ "${VERSION}" = "latest" ] || [ -z "${VERSION}" ]; then
    VERSION="$(resolve_latest_tag)"
fi

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
            # Parse major.minor: "8.9" -> 8 9, "12.0" -> 12 0
            MAJOR="$(echo "${CC}" | cut -d. -f1)"
            MINOR="$(echo "${CC}" | cut -d. -f2)"
            MINOR="${MINOR:-0}"

            if [ "${MAJOR}" -ge 12 ] 2>/dev/null; then
                echo "sm120"
                return
            elif [ "${MAJOR}" -eq 8 ] 2>/dev/null && [ "${MINOR}" -ge 9 ] 2>/dev/null; then
                # Ada Lovelace (8.9) — native sm_89 binary
                echo "sm89"
                return
            elif [ "${MAJOR}" -eq 8 ] 2>/dev/null; then
                # Ampere (8.0, 8.6) — sm_89 binary works via PTX JIT but
                # flash-attention kernels won't run natively. Still the
                # best available binary; core inference works.
                echo "Warning: GPU compute capability ${CC} (Ampere) — using sm89 binary." >&2
                echo "  Core inference works via PTX JIT. Flash-attention may be unavailable." >&2
                echo "sm89"
                return
            elif [ "${MAJOR}" -ge 9 ] 2>/dev/null; then
                # Hopper (9.0) or other post-Ada pre-Blackwell — use sm89 with JIT
                echo "Warning: GPU compute capability ${CC} (Hopper/post-Ada) — using sm89 binary." >&2
                echo "  Core inference works via PTX JIT. A native sm90 binary is not available." >&2
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

# Fallback: older releases used unsuffixed asset name for the single Linux binary
LEGACY_ASSET="mold-x86_64-unknown-linux-gnu-cuda.tar.gz"
LEGACY_URL="https://github.com/${REPO}/releases/download/${VERSION}/${LEGACY_ASSET}"

echo "Installing mold ${VERSION} for ${OS}/${ARCH}..."
echo "  to:   ${INSTALL_DIR}/mold"

# Create install directory
mkdir -p "${INSTALL_DIR}"

# Download, extract, and install
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

if ! curl -fsSL "${URL}" -o "${TMPDIR}/${ASSET}" 2>/dev/null; then
    # Try legacy unsuffixed name (releases before multi-arch support)
    if [ "${OS}" = "Linux" ] && curl -fsSL "${LEGACY_URL}" -o "${TMPDIR}/${LEGACY_ASSET}" 2>/dev/null; then
        ASSET="${LEGACY_ASSET}"
    else
        echo "Error: failed to download ${URL}" >&2
        exit 1
    fi
fi
tar -xzf "${TMPDIR}/${ASSET}" -C "${TMPDIR}"
install -m 755 "${TMPDIR}/mold" "${INSTALL_DIR}/mold"

# macOS: remove quarantine attribute if present
if [ "${OS}" = "Darwin" ]; then
    xattr -d com.apple.quarantine "${INSTALL_DIR}/mold" 2>/dev/null || true
fi

echo ""
echo "mold ${VERSION} installed to ${INSTALL_DIR}/mold"

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
