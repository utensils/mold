#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE="$ROOT/apps/mobile/src-tauri"
ACTION="${1:-dev}"
shift || true

# Keep the large Android toolchain and mutable caches off the internal disk.
# Developers can override the root when their external volume has another path.
ANDROID_ROOT="${MOLD_ANDROID_ROOT:-/Volumes/ExternalStorage/Android}"
export ANDROID_HOME="${MOLD_ANDROID_SDK_ROOT:-$ANDROID_ROOT/sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_USER_HOME="${MOLD_ANDROID_USER_HOME:-$ANDROID_ROOT/user-home}"
export ANDROID_AVD_HOME="${MOLD_ANDROID_AVD_HOME:-$ANDROID_ROOT/avd}"
export GRADLE_USER_HOME="${MOLD_ANDROID_GRADLE_HOME:-$ANDROID_ROOT/gradle}"

STUDIO_APP="${MOLD_ANDROID_STUDIO_APP:-$ANDROID_ROOT/Android Studio.app}"
if [[ -z "${JAVA_HOME:-}" && -d "$STUDIO_APP/Contents/jbr/Contents/Home" ]]; then
  export JAVA_HOME="$STUDIO_APP/Contents/jbr/Contents/Home"
fi

NDK_VERSION="${MOLD_ANDROID_NDK_VERSION:-27.0.12077973}"
export NDK_HOME="${NDK_HOME:-$ANDROID_HOME/ndk/$NDK_VERSION}"
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

AVD_NAME="${MOLD_ANDROID_AVD:-Mold_API_37}"
SYSTEM_IMAGE="${MOLD_ANDROID_SYSTEM_IMAGE:-system-images;android-37.0;google_apis;arm64-v8a}"

android_cli() {
  local name="$1"
  local bundled="$ANDROID_HOME/cmdline-tools/latest/bin/$name"
  if [[ -x "$bundled" ]]; then
    echo "$bundled"
  else
    command -v "$name"
  fi
}

link_command_line_tools() {
  if [[ ! -x "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" || -L "$ANDROID_HOME/cmdline-tools/latest" ]]; then
    local sdkmanager_path cli_root
    sdkmanager_path="$(command -v sdkmanager)" || {
      echo "Install Android SDK Command-line Tools first (brew install --cask android-commandlinetools)." >&2
      exit 1
    }
    cli_root="$(cd "$(dirname "$(realpath "$sdkmanager_path")")/.." && pwd)"
    mkdir -p "$ANDROID_HOME/cmdline-tools"
    if [[ -L "$ANDROID_HOME/cmdline-tools/latest" ]]; then
      unlink "$ANDROID_HOME/cmdline-tools/latest"
    fi
    if [[ ! -e "$ANDROID_HOME/cmdline-tools/latest" ]]; then
      # avdmanager derives the SDK root from its own location, so a symlink
      # back to Homebrew's prefix cannot see external-storage system images.
      cp -R "$cli_root" "$ANDROID_HOME/cmdline-tools/latest"
    fi
  fi
}

require_android_toolchain() {
  [[ -d "$ANDROID_HOME" ]] || {
    echo "Android SDK not found at $ANDROID_HOME" >&2
    echo "Set MOLD_ANDROID_ROOT or install it with scripts/android.sh setup." >&2
    exit 1
  }
  [[ -d "$NDK_HOME" ]] || {
    echo "Android NDK $NDK_VERSION not found at $NDK_HOME" >&2
    exit 1
  }
  command -v java >/dev/null || { echo "Java 21 is required." >&2; exit 1; }
  command -v adb >/dev/null || { echo "Android platform-tools are required." >&2; exit 1; }
}

create_avd() {
  mkdir -p "$ANDROID_USER_HOME" "$ANDROID_AVD_HOME" "$GRADLE_USER_HOME"
  if ! "$ANDROID_HOME/emulator/emulator" -list-avds | grep -qx "$AVD_NAME"; then
    link_command_line_tools
    echo no | "$(android_cli avdmanager)" create avd \
      --force \
      --name "$AVD_NAME" \
      --package "$SYSTEM_IMAGE" \
      --device pixel_9_pro
  fi
}

case "$ACTION" in
  setup)
    mkdir -p "$ANDROID_HOME" "$ANDROID_USER_HOME" "$ANDROID_AVD_HOME" "$GRADLE_USER_HOME"
    link_command_line_tools
    set +o pipefail
    yes | "$(android_cli sdkmanager)" --sdk_root="$ANDROID_HOME" --licenses >/dev/null
    license_status="${PIPESTATUS[1]}"
    set -o pipefail
    [[ "$license_status" -eq 0 ]] || exit "$license_status"
    "$(android_cli sdkmanager)" --sdk_root="$ANDROID_HOME" \
      platform-tools emulator \
      'platforms;android-36' 'platforms;android-37.0' \
      'build-tools;36.0.0' 'build-tools;37.0.0' \
      "ndk;$NDK_VERSION" "$SYSTEM_IMAGE"
    rustup target add \
      aarch64-linux-android armv7-linux-androideabi \
      i686-linux-android x86_64-linux-android
    create_avd
    ;;
  doctor)
    require_android_toolchain
    create_avd
    echo "Android Studio: $STUDIO_APP"
    echo "Android SDK:    $ANDROID_HOME"
    echo "Android NDK:    $NDK_HOME"
    echo "Android AVD:    $ANDROID_AVD_HOME/$AVD_NAME"
    echo "Gradle cache:   $GRADLE_USER_HOME"
    java -version
    adb version | head -1
    emulator_revision="$(sed -n 's/^Pkg.Revision=//p' "$ANDROID_HOME/emulator/source.properties")"
    echo "Android Emulator version $emulator_revision"
    ;;
  init)
    require_android_toolchain
    cd "$MOBILE"
    cargo tauri android init --ci "$@"
    ;;
  emulator)
    require_android_toolchain
    create_avd
    if adb devices | grep -q '^emulator-.*device$'; then
      echo "An Android emulator is already running."
      exit 0
    fi
    exec "$ANDROID_HOME/emulator/emulator" -avd "$AVD_NAME" "$@"
    ;;
  check)
    require_android_toolchain
    cd "$MOBILE"
    cargo tauri android build --debug --apk --target aarch64 --ci "$@"
    ;;
  dev)
    require_android_toolchain
    cd "$MOBILE"
    cargo tauri android dev "$@"
    ;;
  run)
    require_android_toolchain
    cd "$MOBILE"
    cargo tauri android run "$@"
    ;;
  build)
    require_android_toolchain
    cd "$MOBILE"
    cargo tauri android build --aab --target aarch64 --target armv7 --ci "$@"
    ;;
  studio)
    require_android_toolchain
    [[ -d "$STUDIO_APP" ]] || { echo "Android Studio not found at $STUDIO_APP" >&2; exit 1; }
    open "$STUDIO_APP" --args "$MOBILE/gen/android"
    ;;
  *)
    echo "usage: scripts/android.sh {setup|doctor|init|emulator|check|dev|run|build|studio} [args...]" >&2
    exit 2
    ;;
esac
