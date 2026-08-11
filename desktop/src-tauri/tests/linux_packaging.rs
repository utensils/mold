#[test]
fn appimage_ci_installs_tauri_linuxdeploy_runtime() {
    let workflow = include_str!("../../../.github/workflows/desktop.yml");
    let linux_job = workflow
        .split("desktop-linux:")
        .nth(1)
        .expect("desktop workflow must define the Linux job");

    assert!(
        linux_job
            .split_whitespace()
            .any(|package| package == "xdg-utils"),
        "Tauri linuxdeploy requires /usr/bin/xdg-open from xdg-utils"
    );
    assert!(
        linux_job
            .split_whitespace()
            .any(|package| package == "openbox"),
        "the Xvfb launch needs a window manager for Tauri's maximize/show lifecycle"
    );
    assert!(
        linux_job
            .split_whitespace()
            .any(|package| package == "dbus-x11"),
        "the AppImage smoke needs a D-Bus session for native desktop plugins"
    );
}

#[test]
fn appimage_ci_reserves_distribution_work_for_main() {
    let workflow = include_str!("../../../.github/workflows/desktop.yml");
    assert!(
        workflow.contains("Install CUDA toolkit for AppImage")
            && workflow.contains("Build optimized CUDA AppImage")
            && workflow.matches("if: github.event_name == 'push'").count() >= 8,
        "CUDA distribution and smoke steps must be limited to main pushes"
    );
    assert!(
        workflow.contains("if: github.event_name == 'push'")
            && workflow.contains("mold-desktop-linux-appimage-sm89"),
        "large optimized AppImage uploads must be limited to main pushes"
    );
    assert!(
        workflow.contains("xdotool search --onlyvisible --pid")
            && workflow.contains("xdotool search --onlyvisible --name '.'")
            && workflow.contains("openbox >/tmp/mold-openbox.log")
            && workflow.contains("dbus-run-session -- bash"),
        "the isolated Xvfb smoke must not depend on an exact window title"
    );
}

#[test]
fn appimage_excludes_the_host_cuda_driver() {
    let workflow = include_str!("../../../.github/workflows/desktop.yml");
    assert!(
        workflow.contains("../scripts/prepare-desktop-linuxdeploy.sh"),
        "Linux CI must install the linuxdeploy CUDA-driver exclusion wrapper"
    );
    // The wrapper has to trigger the CI that actually exercises it, which is
    // the main push that builds the AppImage. It deliberately does NOT trigger
    // pull requests: #947 made desktop PR feedback format-only on Linux, so a
    // PR run builds no AppImage and a wrapper-only PR trigger would start a
    // job that cannot exercise the change. Asserted per trigger block rather
    // than as a whole-file count, which said "twice" without saying where.
    let (push_filters, rest) = workflow
        .split_once("\n  pull_request:")
        .expect("desktop.yml must declare a pull_request trigger");
    let (pull_request_filters, _) = rest
        .split_once("\n  workflow_dispatch:")
        .expect("desktop.yml must declare a workflow_dispatch trigger");
    const WRAPPER_FILTER: &str = "- \"scripts/prepare-desktop-linuxdeploy.sh\"";
    assert!(
        push_filters.contains(WRAPPER_FILTER),
        "wrapper changes must trigger the main push that builds the AppImage"
    );
    assert!(
        !pull_request_filters.contains(WRAPPER_FILTER),
        "desktop pull requests are format-only and build no AppImage, so the wrapper \
         must not trigger one"
    );
    assert!(
        workflow.contains("XDG_CACHE_HOME: /tmp/mold-desktop-cache-${{ github.run_id }}"),
        "Linux packaging must not replace tools in Tauri's shared user cache"
    );
    assert!(
        workflow.contains("Verify bundled CUDA libraries")
            && workflow.contains("unexpected bundled NVIDIA driver")
            && workflow.contains("missing bundled $library"),
        "CI must inspect the completed AppImage's CUDA library contents"
    );
    assert!(
        workflow.contains("libcuda.so.1") && workflow.contains("GITHUB_ENV"),
        "GPU-less CI packaging and launch need a host-side CUDA driver stub"
    );

    let script_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/prepare-desktop-linuxdeploy.sh"
    );
    let script = std::fs::read_to_string(script_path)
        .expect("Linux bundles must provide the linuxdeploy wrapper setup script");
    assert!(
        script.contains("--exclude-library=libcuda.so*"),
        "AppImages must use the host NVIDIA driver instead of bundling its toolkit stub"
    );
    assert!(
        script.contains("--appimage-extract-and-run"),
        "recursive linuxdeploy plugin calls must not require FUSE"
    );
    assert!(
        script.contains("sha256sum --check"),
        "downloaded linuxdeploy executables must be integrity checked"
    );
    assert!(
        script.contains("linuxdeploy-plugin-gtk.sh")
            && script.contains("linuxdeploy-plugin-gstreamer.sh")
            && script.contains("recursive linuxdeploy call"),
        "linuxdeploy plugins must preserve the CUDA-driver exclusion"
    );
}

#[test]
fn nix_desktop_packages_follow_the_workspace_version() {
    let flake = include_str!("../../../flake.nix");
    assert!(
        flake.contains("workspaceVersion =")
            && flake.matches("version = workspaceVersion;").count() >= 2,
        "desktop Nix packages must not drift from release-plz's workspace version"
    );
}

#[test]
fn nix_desktop_cuda_targets_keep_b200_server_only() {
    let flake = include_str!("../../../flake.nix");
    assert!(
        flake.contains("mold-desktop-sm86 = mkMoldDesktop \"86\";"),
        "RTX 3090/A40 needs a native sm_86 desktop package"
    );
    assert!(
        flake.contains("mold-desktop-sm120 = mkMoldDesktop \"120\";"),
        "the existing RTX 50-series desktop package must remain available"
    );
    assert!(
        !flake.contains("mold-desktop-sm100"),
        "B200 is a remote server target, not a desktop package"
    );
}
