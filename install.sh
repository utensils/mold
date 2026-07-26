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
#   MOLD_CUDA_ARCH    — force GPU architecture: sm86, sm89, sm100, or sm120
#                       (default: auto-detect)

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
        echo "  e.g. curl ... | MOLD_VERSION=v0.10.0 sh" >&2
        exit 1
    fi
    echo "${RESOLVED_TAG}"
}

if [ "${VERSION}" = "latest" ] || [ -z "${VERSION}" ]; then
    VERSION="$(resolve_latest_tag)"
fi

# Print the compute capability of every CUDA-visible GPU or MIG instance.
# NVIDIA's management API sees the whole machine, so apply
# CUDA_VISIBLE_DEVICES here instead of silently inspecting only the first
# physical card.
visible_compute_caps() {
    INVENTORY="$1"
    MIG_LISTING="$2"
    NORMALIZED_INVENTORY="$(
        printf '%s\n' "${INVENTORY}" | awk -F',' '
            /^[[:space:]]*$/ { next }
            {
                if (NF != 3) {
                    print "Error: malformed nvidia-smi inventory row: " $0 > "/dev/stderr"
                    exit 1
                }
                idx = $1; uuid = $2; cap = $3
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", cap)
                if (idx == "" || uuid == "" || cap == "") {
                    print "Error: malformed nvidia-smi inventory row: " $0 > "/dev/stderr"
                    exit 1
                }
                print idx "," uuid "," cap
            }
        '
    )" || return 1
    NORMALIZED_MIG_INVENTORY="$(
        printf '%s\n' "${MIG_LISTING}" | awk '
            /^[[:space:]]*GPU [0-9]+:/ {
                uuid = $0
                sub(/^.*\(UUID:[[:space:]]*/, "", uuid)
                sub(/\).*$/, "", uuid)
                parent = (uuid ~ /^GPU-/) ? uuid : ""
                next
            }
            /^[[:space:]]*MIG / {
                uuid = $0
                sub(/^.*\(UUID:[[:space:]]*/, "", uuid)
                sub(/\).*$/, "", uuid)
                if (parent != "" && uuid ~ /^MIG-/) {
                    print uuid "," parent
                }
            }
        '
    )"

    if [ "${CUDA_VISIBLE_DEVICES+x}" != x ]; then
        printf '%s\n' "${NORMALIZED_INVENTORY}" | awk -F',' 'NF == 3 { print $3 }'
        return
    fi

    TRIMMED_VISIBILITY="$(
        printf '%s' "${CUDA_VISIBLE_DEVICES}" \
            | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
    )"
    case "${TRIMMED_VISIBILITY}" in
        ""|-1) return ;;
    esac

    SELECTORS="$(
        printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '
            {
                for (i = 1; i <= NF; i++) {
                    selector = $i
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", selector)
                    if (selector == "") {
                        print "Error: CUDA_VISIBLE_DEVICES contains an empty selector" > "/dev/stderr"
                        exit 1
                    }
                    print selector
                }
            }
        '
    )" || return 1

    SELECTED_UUIDS="|"
    while IFS= read -r SELECTOR; do
        if printf '%s\n' "${SELECTOR}" | grep -q '^MIG-'; then
            MATCHED_PARENTS="$(printf '%s\n' "${NORMALIZED_MIG_INVENTORY}" \
                | awk -F',' -v selector="${SELECTOR}" '
                    $1 == selector || index($1, selector) == 1 { print $2 }
                ')"
            MATCHES="$(printf '%s\n' "${MATCHED_PARENTS}" \
                | while IFS= read -r PARENT_UUID; do
                    [ -n "${PARENT_UUID}" ] || continue
                    printf '%s\n' "${NORMALIZED_INVENTORY}" \
                        | awk -F',' -v parent="${PARENT_UUID}" '
                            $2 == parent { print $2 "," $3 }
                        '
                done)"
        else
            MATCHES="$(printf '%s\n' "${NORMALIZED_INVENTORY}" | awk -F',' -v selector="${SELECTOR}" '
                {
                    idx = $1; uuid = $2; cap = $3
                    if (idx == selector || uuid == selector ||
                        (selector ~ /^GPU-/ && index(uuid, selector) == 1)) {
                        print uuid "," cap
                    }
                }
            ')"
        fi
        MATCH_COUNT="$(printf '%s\n' "${MATCHES}" | awk 'NF { count++ } END { print count + 0 }')"
        if [ "${MATCH_COUNT}" -ne 1 ]; then
            echo "Error: CUDA_VISIBLE_DEVICES selector '${SELECTOR}' matched ${MATCH_COUNT} CUDA devices." >&2
            echo "  Use an unambiguous GPU or MIG UUID, or unset CUDA_VISIBLE_DEVICES." >&2
            exit 1
        fi
        MATCH_UUID="${MATCHES%%,*}"
        MATCH_CAP="${MATCHES#*,}"
        case "${SELECTED_UUIDS}" in
            *"|${MATCH_UUID}|"*) ;;
            *)
                printf '%s\n' "${MATCH_CAP}"
                SELECTED_UUIDS="${SELECTED_UUIDS}${MATCH_UUID}|"
                ;;
        esac
    done <<EOF
${SELECTORS}
EOF
}

select_cuda_arch_for_caps() {
    CAPS="$1"
    COUNT=0
    ALL_AMPERE_ARCHIVE=1
    HAS_SM86=0
    ALL_SM100=1
    ALL_SM120=1

    for CC in ${CAPS}; do
        COUNT=$((COUNT + 1))
        if ! printf '%s\n' "${CC}" | grep -Eq '^[0-9]+\.[0-9]+$'; then
            echo "Error: no published Mold CUDA archive is proven compatible with visible compute capability '${CC}'." >&2
            echo "  Use CUDA_VISIBLE_DEVICES to expose supported devices, or make a source build with a verified CUDA_COMPUTE_CAP target list." >&2
            return 1
        fi
        MAJOR="${CC%%.*}"
        MINOR="${CC#*.}"
        MAJOR="$(printf '%s\n' "${MAJOR}" | sed 's/^0*//')"
        MINOR="$(printf '%s\n' "${MINOR}" | sed 's/^0*//')"
        [ -n "${MAJOR}" ] || MAJOR=0
        [ -n "${MINOR}" ] || MINOR=0
        if ! awk -v major="${MAJOR}" -v minor="${MINOR}" \
            'BEGIN { exit !(major <= 4294967295 && minor <= 4294967295) }'; then
            echo "Error: no published Mold CUDA archive is proven compatible with visible compute capability '${CC}'." >&2
            echo "  Use CUDA_VISIBLE_DEVICES to expose supported devices, or make a source build with a verified CUDA_COMPUTE_CAP target list." >&2
            return 1
        fi
        case "${MAJOR}.${MINOR}" in
            8.6)
                HAS_SM86=1
                ALL_SM100=0
                ALL_SM120=0
                ;;
            8.9)
                ALL_SM100=0
                ALL_SM120=0
                ;;
            10.*)
                ALL_AMPERE_ARCHIVE=0
                ALL_SM120=0
                ;;
            12.*)
                ALL_AMPERE_ARCHIVE=0
                ALL_SM100=0
                ;;
            *)
                echo "Error: no published Mold CUDA archive is proven compatible with visible compute capability '${CC}'." >&2
                echo "  Use CUDA_VISIBLE_DEVICES to expose supported devices, or make a source build with a verified CUDA_COMPUTE_CAP target list." >&2
                return 1
                ;;
        esac
    done

    if [ "${COUNT}" -eq 0 ]; then
        echo "Error: no CUDA GPU is visible." >&2
        echo "  Set MOLD_CUDA_ARCH only when intentionally installing for another machine." >&2
        return 1
    elif [ "${ALL_AMPERE_ARCHIVE}" -eq 1 ]; then
        if [ "${HAS_SM86}" -eq 1 ]; then
            echo "sm86"
        else
            echo "sm89"
        fi
    elif [ "${ALL_SM100}" -eq 1 ]; then
        echo "sm100"
    elif [ "${ALL_SM120}" -eq 1 ]; then
        echo "sm120"
    else
        echo "Error: no published Mold CUDA archive is proven compatible with every visible GPU (${CAPS})." >&2
        echo "  Use CUDA_VISIBLE_DEVICES to expose one compatible architecture family," >&2
        echo "  or make a source build with a verified CUDA_COMPUTE_CAP target list." >&2
        return 1
    fi
}

# Detect NVIDIA GPU compute capability on Linux.
detect_cuda_arch() {
    case "${MOLD_CUDA_ARCH:-}" in
        ""|sm86|sm89|sm100|sm120) ;;
        *)
            echo "Error: unsupported MOLD_CUDA_ARCH=${MOLD_CUDA_ARCH}" >&2
            echo "  Expected one of: sm86, sm89, sm100, sm120" >&2
            exit 1
            ;;
    esac

    INVENTORY=""
    MIG_LISTING=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        INVENTORY="$(nvidia-smi \
            --query-gpu=index,uuid,compute_cap \
            --format=csv,noheader,nounits 2>/dev/null || true)"
        MIG_LISTING="$(nvidia-smi -L 2>/dev/null || true)"
    fi

    if [ -z "${INVENTORY}" ]; then
        if [ -n "${MOLD_CUDA_ARCH:-}" ]; then
            echo "Warning: nvidia-smi inventory is unavailable; trusting explicit MOLD_CUDA_ARCH=${MOLD_CUDA_ARCH}." >&2
            echo "${MOLD_CUDA_ARCH}"
            return
        fi
        echo "Error: could not inspect NVIDIA GPUs with nvidia-smi." >&2
        echo "  Set MOLD_CUDA_ARCH only when intentionally installing for a known target." >&2
        exit 1
    fi

    CAPS="$(visible_compute_caps "${INVENTORY}" "${MIG_LISTING}")" || exit 1
    if [ -z "${CAPS}" ] && [ -n "${MOLD_CUDA_ARCH:-}" ]; then
        echo "Warning: CUDA_VISIBLE_DEVICES exposes no GPU; trusting explicit MOLD_CUDA_ARCH=${MOLD_CUDA_ARCH} for another machine." >&2
        echo "${MOLD_CUDA_ARCH}"
        return
    fi
    SELECTED="$(select_cuda_arch_for_caps "${CAPS}")" || exit 1
    if [ -n "${MOLD_CUDA_ARCH:-}" ] \
        && [ "${MOLD_CUDA_ARCH}" != "${SELECTED}" ]; then
        echo "Error: MOLD_CUDA_ARCH=${MOLD_CUDA_ARCH} is not compatible with the complete visible fleet (selected ${SELECTED})." >&2
        echo "  Hide incompatible devices with CUDA_VISIBLE_DEVICES or use a source build." >&2
        exit 1
    fi
    echo "${MOLD_CUDA_ARCH:-${SELECTED}}"
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

# Historical unsuffixed releases used the same target as today's sm89 asset.
LEGACY_ASSET="mold-x86_64-unknown-linux-gnu-cuda.tar.gz"
LEGACY_URL="https://github.com/${REPO}/releases/download/${VERSION}/${LEGACY_ASSET}"

echo "Installing mold ${VERSION} for ${OS}/${ARCH}..."
echo "  to:   ${INSTALL_DIR}/mold"

# Create install directory
mkdir -p "${INSTALL_DIR}"

# Download, extract, and install.
MOLD_INSTALL_TMP="$(mktemp -d)"
trap 'rm -rf "${MOLD_INSTALL_TMP}"' EXIT

download_release_file() {
    DOWNLOAD_URL="$1"
    DOWNLOAD_PATH="$2"
    HTTP_STATUS="$(
        curl --silent --show-error --location \
            --retry 3 --retry-delay 1 --retry-connrefused \
            --output "${DOWNLOAD_PATH}" \
            --write-out '%{http_code}' \
            "${DOWNLOAD_URL}"
    )" || {
        echo "Error: transient/network failure downloading ${DOWNLOAD_URL}; no compatibility fallback was attempted." >&2
        return 1
    }
    case "${HTTP_STATUS}" in
        200) return 0 ;;
        404)
            rm -f "${DOWNLOAD_PATH}"
            return 44
            ;;
        *)
            rm -f "${DOWNLOAD_PATH}"
            echo "Error: ${DOWNLOAD_URL} returned HTTP ${HTTP_STATUS}; no compatibility fallback was attempted." >&2
            return 1
            ;;
    esac
}

DOWNLOAD_STATUS=0
download_release_file "${URL}" "${MOLD_INSTALL_TMP}/${ASSET}" || DOWNLOAD_STATUS=$?
if [ "${DOWNLOAD_STATUS}" -eq 44 ]; then
    # Only sm89 may use the old unsuffixed name for the same target. Native
    # sm86/sm100/sm120 assets have no safe compatibility substitute.
    if [ "${OS}" = "Linux" ] && [ "${CUDA_ARCH}" = "sm89" ]; then
        DOWNLOAD_STATUS=0
        download_release_file "${LEGACY_URL}" "${MOLD_INSTALL_TMP}/${LEGACY_ASSET}" \
            || DOWNLOAD_STATUS=$?
        if [ "${DOWNLOAD_STATUS}" -eq 0 ]; then
            echo "Warning: ${VERSION} predates suffixed CUDA assets; using its same-release legacy archive." >&2
            ASSET="${LEGACY_ASSET}"
        fi
    fi
fi
if [ "${DOWNLOAD_STATUS}" -ne 0 ]; then
    echo "Error: failed to download a compatible ${VERSION} release archive." >&2
    exit 1
fi

SUMS_URL="https://github.com/${REPO}/releases/download/${VERSION}/SHA256SUMS"
download_release_file "${SUMS_URL}" "${MOLD_INSTALL_TMP}/SHA256SUMS" || {
    echo "Error: ${VERSION} cannot be installed safely because SHA256SUMS is unavailable." >&2
    exit 1
}
EXPECTED_SHA="$(
    awk -v asset="${ASSET}" '
        {
            candidate = $2
            sub(/^\*/, "", candidate)
            sub(/^\.\//, "", candidate)
        }
        candidate == asset {
            if (found++) exit 2
            print $1
        }
        END {
            if (found != 1) exit 1
        }
    ' "${MOLD_INSTALL_TMP}/SHA256SUMS"
)" || {
    echo "Error: SHA256SUMS does not contain exactly one checksum for ${ASSET}." >&2
    exit 1
}
case "${EXPECTED_SHA}" in
    *[!0-9A-Fa-f]*|"")
        echo "Error: SHA256SUMS contains an invalid checksum for ${ASSET}." >&2
        exit 1
        ;;
esac
if [ "${#EXPECTED_SHA}" -ne 64 ]; then
    echo "Error: SHA256SUMS contains an invalid checksum length for ${ASSET}." >&2
    exit 1
fi
EXPECTED_SHA="$(printf '%s' "${EXPECTED_SHA}" | tr 'A-F' 'a-f')"
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA="$(sha256sum "${MOLD_INSTALL_TMP}/${ASSET}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA="$(shasum -a 256 "${MOLD_INSTALL_TMP}/${ASSET}" | awk '{print $1}')"
else
    echo "Error: sha256sum or shasum is required to verify the release archive." >&2
    exit 1
fi
if [ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]; then
    echo "Error: SHA-256 mismatch for ${ASSET}." >&2
    echo "  expected: ${EXPECTED_SHA}" >&2
    echo "  actual:   ${ACTUAL_SHA}" >&2
    exit 1
fi

tar -xzf "${MOLD_INSTALL_TMP}/${ASSET}" -C "${MOLD_INSTALL_TMP}"
install -m 755 "${MOLD_INSTALL_TMP}/mold" "${INSTALL_DIR}/mold"

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
