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
