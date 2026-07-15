#[test]
fn appimage_ci_installs_tauri_linuxdeploy_runtime() {
    let workflow = include_str!("../../../.github/workflows/desktop.yml");
    let linux_job = workflow
        .split("- name: Install Tauri and media dependencies")
        .nth(1)
        .expect("desktop workflow must define the Linux dependency step");
    let apt_packages = linux_job
        .split("sudo apt-get install -y")
        .nth(1)
        .expect("desktop workflow must install Linux packaging dependencies")
        .split("- uses: Jimver/cuda-toolkit")
        .next()
        .expect("CUDA setup must follow Linux packaging dependencies");

    assert!(
        apt_packages
            .split_whitespace()
            .any(|package| package == "xdg-utils"),
        "Tauri linuxdeploy requires /usr/bin/xdg-open from xdg-utils"
    );
}

#[test]
fn appimage_excludes_the_host_cuda_driver() {
    let workflow = include_str!("../../../.github/workflows/desktop.yml");
    assert!(
        workflow.contains("../scripts/prepare-desktop-linuxdeploy.sh"),
        "Linux CI must install the linuxdeploy CUDA-driver exclusion wrapper"
    );
    assert_eq!(
        workflow
            .matches("- \"scripts/prepare-desktop-linuxdeploy.sh\"")
            .count(),
        2,
        "wrapper changes must trigger desktop CI for pushes and pull requests"
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
