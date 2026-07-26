//! Self-update command — downloads and installs the latest mold release from GitHub.

use std::io::Read as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::theme;
use mold_core::cuda_distribution::{
    is_release_arch, release_arch_for_compute_caps, release_arch_supports_compute_caps,
    visible_compute_capabilities_with_mig,
};

const GITHUB_REPO: &str = "utensils/mold";
const GITHUB_API_BASE: &str = "https://api.github.com";

// ── GitHub API types ────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Clone, serde::Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

// ── Version comparison ──────────────────────────────────────────────────────

/// Parse a version string like "0.6.1" or "v0.6.1" into (major, minor, patch).
fn parse_version(v: &str) -> Option<(u32, u32, u32)> {
    let v = v.strip_prefix('v').unwrap_or(v);
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// Returns true if `remote` is strictly newer than `current`.
fn is_newer(current: &str, remote: &str) -> bool {
    match (parse_version(current), parse_version(remote)) {
        (Some(c), Some(r)) => r > c,
        _ => false,
    }
}

// ── Platform detection ──────────────────────────────────────────────────────

/// Detect the correct GitHub release asset name for this platform.
fn detect_asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("macos", "aarch64") => Ok("mold-aarch64-apple-darwin.tar.gz".to_string()),
        ("linux", "x86_64") => {
            let cuda_arch = detect_cuda_arch()?;
            Ok(format!(
                "mold-x86_64-unknown-linux-gnu-cuda-{cuda_arch}.tar.gz"
            ))
        }
        _ => bail!("unsupported platform: {os}/{arch}"),
    }
}

/// Detect CUDA GPU architecture via nvidia-smi, env override, or default.
fn detect_cuda_arch() -> Result<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let inventory = match output {
        Ok(out) if out.status.success() => Some(String::from_utf8_lossy(&out.stdout).to_string()),
        _ => None,
    };
    let mig_listing = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).to_string());
    detect_cuda_arch_from_inventory(
        std::env::var("MOLD_CUDA_ARCH").ok().as_deref(),
        inventory.as_deref(),
        mig_listing.as_deref(),
        std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
    )
}

fn detect_cuda_arch_from_inventory(
    override_arch: Option<&str>,
    inventory_csv: Option<&str>,
    mig_listing: Option<&str>,
    cuda_visible_devices: Option<&str>,
) -> Result<String> {
    if let Some(arch) = override_arch {
        if !is_release_arch(arch) {
            bail!("unsupported MOLD_CUDA_ARCH={arch}; expected sm86, sm89, sm100, or sm120");
        }
        if let Some(inventory) = inventory_csv {
            let caps = visible_compute_capabilities_with_mig(
                inventory,
                mig_listing.unwrap_or_default(),
                cuda_visible_devices,
            )
            .map_err(anyhow::Error::msg)?;
            if !caps.is_empty() {
                release_arch_supports_compute_caps(arch, &caps).map_err(|error| {
                    anyhow::anyhow!("MOLD_CUDA_ARCH={arch} is incompatible: {error}")
                })?;
            }
        }
        return Ok(arch.to_string());
    }

    let inventory = inventory_csv.context(
        "could not inspect NVIDIA GPUs with nvidia-smi; set MOLD_CUDA_ARCH only when \
         intentionally installing for a known target",
    )?;
    let caps = visible_compute_capabilities_with_mig(
        inventory,
        mig_listing.unwrap_or_default(),
        cuda_visible_devices,
    )
    .map_err(anyhow::Error::msg)?;
    release_arch_for_compute_caps(&caps)
        .map(str::to_string)
        .map_err(anyhow::Error::msg)
}

// ── Package manager detection ───────────────────────────────────────────────

/// Check if the binary path looks like it was installed by a package manager.
/// Returns a hint string if so.
fn detect_package_manager(exe_path: &Path) -> Option<String> {
    let path_str = exe_path.to_string_lossy();
    if path_str.contains("/nix/store/") {
        Some("nix flake update".to_string())
    } else if path_str.contains("/Cellar/") || path_str.contains("/homebrew/") {
        Some("brew upgrade mold".to_string())
    } else if cfg!(target_os = "linux")
        && (path_str.starts_with("/usr/bin/") || path_str.starts_with("/usr/sbin/"))
    {
        // Bail unconditionally — any `/usr/bin` or `/usr/sbin` install on
        // Linux is system-managed (Arch's usrmerge symlinks /usr/sbin →
        // /usr/bin, so `which` can resolve to either). /usr/local/bin is
        // intentionally excluded — that's the conventional install.sh
        // target and unowned by any package manager.
        //
        // Hint text is distro-specific: only Arch / Arch-derivative users
        // would recognise `paru`, so we read /etc/os-release to pick the
        // right wording. Fedora/Debian/openSUSE/etc. get a generic hint.
        let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
        if is_arch_linux(&os_release) {
            Some(
                "paru -Syu mold-ai-bin (or mold-ai, depending on which AUR package you installed)"
                    .to_string(),
            )
        } else {
            Some("update via your distro's package manager (apt/dnf/zypper/etc.)".to_string())
        }
    } else {
        None
    }
}

/// Heuristic for Arch / Arch-derivative detection from /etc/os-release content.
/// Matches `ID=arch` or any `ID_LIKE=…arch…` (Manjaro, EndeavourOS, Garuda…).
fn is_arch_linux(os_release: &str) -> bool {
    os_release.lines().any(|line| {
        let line = line.trim();
        line == "ID=arch"
            || line == "ID=\"arch\""
            || (line.starts_with("ID_LIKE=") && line.contains("arch"))
    })
}

// ── SHA-256 checksum verification ───────────────────────────────────────────

/// Parse a SHA256SUMS file and verify the checksum for `asset_name` against `data`.
fn verify_checksum(sums_content: &str, asset_name: &str, data: &[u8]) -> Result<()> {
    let matches = sums_content
        .lines()
        .filter_map(|line| {
            // Format: "{hash}  {filename}" (two-space separator, sha256sum convention)
            let (hash, name) = line.split_once("  ")?;
            let name = name.trim().trim_start_matches('*').trim_start_matches("./");
            if name == asset_name {
                Some(hash.trim().to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let [expected] = matches.as_slice() else {
        if matches.is_empty() {
            bail!("asset {asset_name} not found in SHA256SUMS");
        }
        bail!("duplicate entries for asset {asset_name} in SHA256SUMS");
    };
    anyhow::ensure!(
        expected.len() == 64
            && expected
                .chars()
                .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character)),
        "invalid SHA-256 checksum for {asset_name}"
    );

    let mut hasher = Sha256::new();
    hasher.update(data);
    let actual = format!("{:x}", hasher.finalize());

    if actual != *expected {
        bail!(
            "SHA-256 checksum mismatch for {asset_name}\n  expected: {expected}\n  actual:   {actual}"
        );
    }

    Ok(())
}

// ── Tarball extraction ──────────────────────────────────────────────────────

/// Extract the `mold` binary from a .tar.gz archive.
fn extract_binary_from_tarball(data: &[u8]) -> Result<Vec<u8>> {
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.file_name().map(|n| n == "mold").unwrap_or(false) {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            return Ok(buf);
        }
    }

    bail!("'mold' binary not found in release archive")
}

// ── Binary self-replacement ─────────────────────────────────────────────────

/// Replace the running binary with new contents. Returns the path that was replaced.
fn replace_binary(new_binary: &[u8], exe_path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let exe_dir = exe_path
        .parent()
        .context("cannot determine binary directory")?;

    let pid = std::process::id();
    let tmp_path = exe_dir.join(format!(".mold-update-{pid}"));
    let backup_path = exe_path.with_extension("old");

    // Write new binary to temp file
    std::fs::write(&tmp_path, new_binary).context("failed to write new binary to temp file")?;
    std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o755))
        .context("failed to set permissions on new binary")?;

    // Atomic swap: current -> backup, then temp -> current
    std::fs::rename(exe_path, &backup_path).context("failed to move current binary to backup")?;

    if let Err(e) = std::fs::rename(&tmp_path, exe_path) {
        // Recovery: restore from backup
        let _ = std::fs::rename(&backup_path, exe_path);
        let _ = std::fs::remove_file(&tmp_path);
        bail!("failed to install new binary: {e}");
    }

    // Clean up backup
    let _ = std::fs::remove_file(&backup_path);

    // macOS: remove quarantine attribute
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("xattr")
            .args(["-d", "com.apple.quarantine"])
            .arg(exe_path)
            .output();
    }

    Ok(())
}

// ── HTTP helpers ────────────────────────────────────────────────────────────

/// Build a reqwest client with appropriate headers for the GitHub API.
fn build_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT,
        "application/vnd.github+json".parse().expect("valid header"),
    );

    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {token}")
                .parse()
                .context("invalid GITHUB_TOKEN")?,
        );
    }

    reqwest::Client::builder()
        .user_agent(format!("mold/{}", mold_core::build_info::VERSION))
        .default_headers(headers)
        .build()
        .context("failed to build HTTP client")
}

/// Fetch the latest non-prerelease GitHub release.
async fn fetch_latest_release(client: &reqwest::Client) -> Result<GitHubRelease> {
    let url = format!("{GITHUB_API_BASE}/repos/{GITHUB_REPO}/releases/latest");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("failed to connect to GitHub API")?;

    if resp.status() == reqwest::StatusCode::FORBIDDEN {
        bail!(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN to authenticate:\n  \
             export GITHUB_TOKEN=$(gh auth token)"
        );
    }

    if !resp.status().is_success() {
        bail!("GitHub API returned {}", resp.status());
    }

    resp.json()
        .await
        .context("failed to parse GitHub release response")
}

/// Fetch a specific release by tag name.
async fn fetch_release_by_tag(client: &reqwest::Client, tag: &str) -> Result<GitHubRelease> {
    // Normalise: accept both "v0.7.0" and "0.7.0"
    let tag = if tag.starts_with('v') {
        tag.to_string()
    } else {
        format!("v{tag}")
    };

    let url = format!("{GITHUB_API_BASE}/repos/{GITHUB_REPO}/releases/tags/{tag}");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("failed to connect to GitHub API")?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!("release {tag} not found on GitHub");
    }

    if resp.status() == reqwest::StatusCode::FORBIDDEN {
        bail!(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN to authenticate:\n  \
             export GITHUB_TOKEN=$(gh auth token)"
        );
    }

    if !resp.status().is_success() {
        bail!("GitHub API returned {}", resp.status());
    }

    resp.json()
        .await
        .context("failed to parse GitHub release response")
}

/// Download a release asset with a progress bar.
async fn download_asset(client: &reqwest::Client, url: &str, size: u64) -> Result<Vec<u8>> {
    let resp = client
        .get(url)
        .header(reqwest::header::ACCEPT, "application/octet-stream")
        .send()
        .await
        .context("failed to download release asset")?;

    if !resp.status().is_success() {
        bail!("download failed with HTTP {}", resp.status());
    }

    let pb = ProgressBar::new(size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "  {{spinner:.{style}}} Downloading {{bar:30.{style}/dim}} \
                 {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})",
                style = theme::SPINNER_STYLE
            ))
            .expect("valid template")
            .progress_chars("━╸─"),
    );

    let mut data = Vec::with_capacity(size as usize);
    let mut stream = resp;

    while let Some(chunk) = stream
        .chunk()
        .await
        .context("error reading download stream")?
    {
        pb.inc(chunk.len() as u64);
        data.extend_from_slice(&chunk);
    }

    pb.finish_and_clear();
    Ok(data)
}

fn select_release_asset<'a>(
    assets: &'a [GitHubAsset],
    desired_name: &str,
    linux_x86_64: bool,
) -> Option<&'a GitHubAsset> {
    let exact = assets.iter().find(|asset| asset.name == desired_name);
    if exact.is_some() || !linux_x86_64 {
        return exact;
    }
    if !desired_name.ends_with("-sm89.tar.gz") {
        return None;
    }

    // Only the historical unsuffixed artifact may substitute for sm89: it was
    // the old name of that same release target. Never substitute an artifact
    // compiled above a visible GPU's compute capability.
    let legacy = "mold-x86_64-unknown-linux-gnu-cuda.tar.gz";
    assets.iter().find(|asset| asset.name == legacy)
}

// ── Main command ────────────────────────────────────────────────────────────

pub async fn run(check: bool, force: bool, version: Option<String>) -> Result<()> {
    let current = mold_core::build_info::VERSION;
    eprintln!("{} Current version: {current}", theme::icon_info());
    eprintln!("{} Checking for updates...", theme::icon_info());

    let client = build_client()?;

    let release = match &version {
        Some(tag) => fetch_release_by_tag(&client, tag).await?,
        None => fetch_latest_release(&client).await?,
    };

    let remote_version = release
        .tag_name
        .strip_prefix('v')
        .unwrap_or(&release.tag_name);

    // Version comparison
    if !force {
        if remote_version == current {
            eprintln!("{} Already up to date ({current})", theme::icon_done());
            return Ok(());
        }

        if version.is_none() && !is_newer(current, remote_version) {
            eprintln!(
                "{} Current version ({current}) is newer than latest release ({remote_version})",
                theme::icon_done()
            );
            return Ok(());
        }
    }

    let action = if is_newer(current, remote_version) {
        "Updating"
    } else if remote_version == current {
        "Reinstalling"
    } else {
        "Downgrading"
    };

    // --check: report availability and exit (no write access needed)
    if check {
        if is_newer(current, remote_version) {
            eprintln!(
                "{} New version available: {remote_version} (current: {current})",
                theme::icon_info()
            );
        } else {
            eprintln!(
                "{} Version {remote_version} is available (current: {current})",
                theme::icon_info()
            );
        }
        return Ok(());
    }

    // From here on we will write to disk — validate install location
    let exe_path = std::env::current_exe()?.canonicalize()?;
    if let Some(pkg_hint) = detect_package_manager(&exe_path) {
        eprintln!(
            "{} mold is installed at {}, which is read-only.",
            theme::icon_fail(),
            exe_path.display()
        );
        eprintln!(
            "  {} update via your package manager instead (e.g. {pkg_hint})",
            theme::prefix_hint()
        );
        std::process::exit(1);
    }

    if let Some(exe_dir) = exe_path.parent() {
        let test_path = exe_dir.join(format!(".mold-update-test-{}", std::process::id()));
        match std::fs::write(&test_path, b"") {
            Ok(()) => {
                let _ = std::fs::remove_file(&test_path);
            }
            Err(_) => {
                bail!(
                    "no write permission to {}. Try running with sudo or \
                     set MOLD_INSTALL_DIR to a writable location and reinstall.",
                    exe_dir.display()
                );
            }
        }
    }

    eprintln!(
        "{} {action}: {current} -> {remote_version}",
        theme::icon_info()
    );

    // Detect correct asset (with legacy fallback for pre-multi-arch releases)
    let asset_name = detect_asset_name()?;

    let asset = select_release_asset(
        &release.assets,
        &asset_name,
        cfg!(all(target_os = "linux", target_arch = "x86_64")),
    )
    .with_context(|| {
        format!(
            "release {} has no asset matching {asset_name}",
            release.tag_name
        )
    })?;
    if asset.name != asset_name {
        eprintln!(
            "{} Release {} predates {}; using compatibility asset {}",
            theme::prefix_warning(),
            release.tag_name,
            asset_name,
            asset.name
        );
    }

    let sums_asset = release
        .assets
        .iter()
        .find(|a| a.name == "SHA256SUMS")
        .context("release has no SHA256SUMS file")?;

    // Download archive and checksums
    let archive_data = download_asset(&client, &asset.browser_download_url, asset.size).await?;

    let sums_resp = client
        .get(&sums_asset.browser_download_url)
        .send()
        .await
        .context("failed to download SHA256SUMS")?;
    if !sums_resp.status().is_success() {
        anyhow::bail!(
            "failed to download SHA256SUMS: server returned HTTP {}",
            sums_resp.status()
        );
    }
    let sums_content = sums_resp
        .text()
        .await
        .context("failed to read SHA256SUMS")?;

    // Verify checksum (use the matched asset's actual name, which may be the legacy name)
    verify_checksum(&sums_content, &asset.name, &archive_data)?;
    eprintln!("{} Checksum verified (SHA-256)", theme::icon_info());

    // Extract binary
    let binary = extract_binary_from_tarball(&archive_data)?;

    // Replace binary
    replace_binary(&binary, &exe_path)?;

    eprintln!(
        "{} {action} complete: mold {remote_version} ({})",
        theme::icon_done(),
        exe_path.display()
    );

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    // ── Version comparison ──────────────────────────────────────────────

    #[test]
    fn test_parse_version_valid() {
        assert_eq!(parse_version("0.6.1"), Some((0, 6, 1)));
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("10.20.30"), Some((10, 20, 30)));
    }

    #[test]
    fn test_parse_version_invalid() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("1.2"), None);
        assert_eq!(parse_version("1.2.3.4"), None);
        assert_eq!(parse_version("abc"), None);
        assert_eq!(parse_version("1.2.x"), None);
    }

    #[test]
    fn test_is_newer_basic() {
        assert!(is_newer("0.6.0", "0.7.0"));
        assert!(is_newer("0.6.1", "0.6.2"));
        assert!(!is_newer("0.6.1", "0.6.1"));
        assert!(!is_newer("1.0.0", "0.6.1"));
    }

    #[test]
    fn test_is_newer_with_v_prefix() {
        assert!(is_newer("0.6.0", "v0.7.0"));
        assert!(is_newer("v0.6.0", "0.7.0"));
        assert!(is_newer("v0.6.0", "v0.7.0"));
        assert!(!is_newer("v0.7.0", "v0.6.0"));
    }

    #[test]
    fn test_is_newer_major_bump() {
        assert!(is_newer("0.9.9", "1.0.0"));
        assert!(is_newer("0.99.99", "1.0.0"));
        assert!(!is_newer("1.0.0", "0.99.99"));
    }

    // ── Platform detection ──────────────────────────────────────────────

    #[test]
    fn test_detect_asset_name_current_platform() {
        let name = detect_asset_name();
        // This test just verifies it returns a valid result on the current platform
        assert!(name.is_ok(), "detect_asset_name failed: {name:?}");
        let name = name.unwrap();
        assert!(name.starts_with("mold-"));
        assert!(name.ends_with(".tar.gz"));
    }

    // ── Tarball extraction ──────────────────────────────────────────────

    fn make_test_tarball(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        let tar_data = builder.into_inner().unwrap();

        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&tar_data).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn test_extract_binary_from_tarball() {
        let expected = b"fake-mold-binary-content-12345";
        let archive = make_test_tarball(&[("mold", expected)]);
        let result = extract_binary_from_tarball(&archive).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn cuda_release_architecture_matches_published_gpu_families() {
        assert!(release_arch_for_compute_caps(["8.0"]).is_err());
        assert_eq!(release_arch_for_compute_caps(["8.6"]).unwrap(), "sm86");
        assert_eq!(release_arch_for_compute_caps(["8.9"]).unwrap(), "sm89");
        assert!(release_arch_for_compute_caps(["9.0"]).is_err());
        assert_eq!(release_arch_for_compute_caps(["10.0"]).unwrap(), "sm100");
        assert_eq!(release_arch_for_compute_caps(["10.3"]).unwrap(), "sm100");
        assert_eq!(release_arch_for_compute_caps(["12.0"]).unwrap(), "sm120");
        assert_eq!(release_arch_for_compute_caps(["12.1"]).unwrap(), "sm120");
        assert!(release_arch_for_compute_caps(["not-a-version"]).is_err());
        assert!(is_release_arch("sm100"));
        assert!(!is_release_arch("../unexpected"));
    }

    #[test]
    fn updater_selects_for_the_full_visible_fleet_in_any_order() {
        let first = "\
0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 8.6
1, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, 8.9
";
        let reversed = "\
0, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, 8.9
1, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 8.6
";
        assert_eq!(
            detect_cuda_arch_from_inventory(None, Some(first), None, None).unwrap(),
            "sm86"
        );
        assert_eq!(
            detect_cuda_arch_from_inventory(None, Some(reversed), None, None).unwrap(),
            "sm86"
        );
        assert_eq!(
            detect_cuda_arch_from_inventory(
                None,
                Some(first),
                None,
                Some("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
            )
            .unwrap(),
            "sm86"
        );
    }

    #[test]
    fn updater_rejects_incompatible_mixed_fleets_and_bad_overrides() {
        let mixed = "\
0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 8.6
1, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, 10.0
";
        let error = detect_cuda_arch_from_inventory(None, Some(mixed), None, None).unwrap_err();
        assert!(error.to_string().contains("source build"));
        let error =
            detect_cuda_arch_from_inventory(Some("sm86"), Some(mixed), None, None).unwrap_err();
        assert!(error.to_string().contains("MOLD_CUDA_ARCH"));
    }

    #[test]
    fn updater_selects_parent_architecture_for_visible_mig_instance() {
        let inventory = "\
0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 10.0
";
        let listing = "\
GPU 0: NVIDIA B200 (UUID: GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa)
  MIG 1g.23gb Device 0: (UUID: MIG-11111111-1111-1111-1111-111111111111)
";
        assert_eq!(
            detect_cuda_arch_from_inventory(
                None,
                Some(inventory),
                Some(listing),
                Some("MIG-11111111-1111-1111-1111-111111111111")
            )
            .unwrap(),
            "sm100"
        );
        assert!(
            detect_cuda_arch_from_inventory(None, Some(inventory), None, Some("MIG-unknown"))
                .is_err()
        );
    }

    #[test]
    fn pre_phase_g_release_fails_closed_when_native_asset_is_absent() {
        let release: GitHubRelease = serde_json::from_str(include_str!(
            "../../tests/fixtures/release-pre-phase-g.json"
        ))
        .unwrap();

        for desired in [
            "mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz",
            "mold-x86_64-unknown-linux-gnu-cuda-sm100.tar.gz",
        ] {
            assert!(
                select_release_asset(&release.assets, desired, true).is_none(),
                "must not substitute sm89 for {desired}"
            );
        }
        let without_sm120 = release
            .assets
            .iter()
            .filter(|asset| !asset.name.ends_with("-sm120.tar.gz"))
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            select_release_asset(
                &without_sm120,
                "mold-x86_64-unknown-linux-gnu-cuda-sm120.tar.gz",
                true
            )
            .is_none(),
            "an old release must not claim forward compatibility with sm120"
        );
    }

    #[test]
    fn ancient_release_never_substitutes_unsuffixed_for_native_asset() {
        let release: GitHubRelease =
            serde_json::from_str(include_str!("../../tests/fixtures/release-ancient.json"))
                .unwrap();
        assert!(select_release_asset(
            &release.assets,
            "mold-x86_64-unknown-linux-gnu-cuda-sm100.tar.gz",
            true,
        )
        .is_none());
        assert!(select_release_asset(
            &release.assets,
            "mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz",
            true,
        )
        .is_some());
    }

    #[test]
    fn test_extract_tarball_with_extra_files() {
        let expected = b"the-real-mold";
        let archive = make_test_tarball(&[
            ("README.md", b"readme content"),
            ("mold", expected),
            ("LICENSE", b"license content"),
        ]);
        let result = extract_binary_from_tarball(&archive).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_extract_tarball_missing_binary() {
        let archive = make_test_tarball(&[("not-mold", b"something else")]);
        let result = extract_binary_from_tarball(&archive);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in release archive"));
    }

    // ── Checksum verification ───────────────────────────────────────────

    #[test]
    fn test_verify_checksum_match() {
        let data = b"hello world";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = format!("{:x}", hasher.finalize());

        let sums = format!("{hash}  test-file.tar.gz\n");
        assert!(verify_checksum(&sums, "test-file.tar.gz", data).is_ok());
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let sums =
            "0000000000000000000000000000000000000000000000000000000000000000  test.tar.gz\n";
        let result = verify_checksum(sums, "test.tar.gz", b"actual data");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_verify_checksum_missing_asset() {
        let sums = "abcdef1234567890  other-file.tar.gz\n";
        let result = verify_checksum(sums, "missing-file.tar.gz", b"data");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_verify_checksum_multi_line() {
        let data_a = b"file-a-content";
        let data_b = b"file-b-content";

        let mut hasher_a = Sha256::new();
        hasher_a.update(data_a);
        let hash_a = format!("{:x}", hasher_a.finalize());

        let mut hasher_b = Sha256::new();
        hasher_b.update(data_b);
        let hash_b = format!("{:x}", hasher_b.finalize());

        let sums = format!("{hash_a}  file-a.tar.gz\n{hash_b}  file-b.tar.gz\n");

        assert!(verify_checksum(&sums, "file-a.tar.gz", data_a).is_ok());
        assert!(verify_checksum(&sums, "file-b.tar.gz", data_b).is_ok());
    }

    #[test]
    fn test_verify_checksum_accepts_release_workflow_relative_path() {
        let data = b"release bytes";
        let expected = format!("{:x}", Sha256::digest(data));
        let sums = format!("{expected}  ./mold-test.tar.gz\n");
        verify_checksum(&sums, "mold-test.tar.gz", data).unwrap();
    }

    #[test]
    fn test_verify_checksum_rejects_duplicate_asset_entries() {
        let data = b"release bytes";
        let expected = format!("{:x}", Sha256::digest(data));
        let sums = format!(
            "{expected}  mold-test.tar.gz\n{}  ./mold-test.tar.gz\n",
            "0".repeat(64)
        );
        let error = verify_checksum(&sums, "mold-test.tar.gz", data).unwrap_err();
        assert!(error.to_string().contains("duplicate"), "{error:#}");
    }

    // ── Package manager detection ───────────────────────────────────────

    #[test]
    fn test_detect_nix_store() {
        let path = Path::new("/nix/store/abc123-mold/bin/mold");
        assert_eq!(
            detect_package_manager(path),
            Some("nix flake update".to_string())
        );
    }

    #[test]
    fn test_detect_homebrew() {
        let path = Path::new("/opt/homebrew/Cellar/mold/0.6.1/bin/mold");
        assert_eq!(
            detect_package_manager(path),
            Some("brew upgrade mold".to_string())
        );
    }

    #[test]
    fn test_detect_local_bin() {
        let path = Path::new("/home/user/.local/bin/mold");
        assert_eq!(detect_package_manager(path), None);
    }

    #[test]
    fn test_detect_usr_local() {
        let path = Path::new("/usr/local/bin/mold");
        assert_eq!(detect_package_manager(path), None);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_detect_system_managed_usr_bin() {
        // Bail-out is unconditional on Linux /usr/bin — the hint text
        // varies by distro but the bail itself protects every Linux
        // install from a self-update writing to a system path.
        let hint = detect_package_manager(Path::new("/usr/bin/mold"));
        assert!(hint.is_some());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_detect_system_managed_usr_sbin() {
        // Arch's usrmerge symlinks /usr/sbin → /usr/bin.
        let hint = detect_package_manager(Path::new("/usr/sbin/mold"));
        assert!(hint.is_some());
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn test_usr_bin_is_unmanaged_off_linux() {
        // On macOS /usr/bin is system territory but pacman doesn't exist
        // there; classify as None so `update` doesn't print Linux hints
        // to a Darwin user.
        assert_eq!(detect_package_manager(Path::new("/usr/bin/mold")), None);
    }

    // ── Arch detection from /etc/os-release ─────────────────────────────

    #[test]
    fn test_is_arch_linux_canonical() {
        assert!(is_arch_linux("NAME=\"Arch Linux\"\nID=arch\n"));
    }

    #[test]
    fn test_is_arch_linux_id_quoted() {
        // Some distros quote the ID value; both forms are valid per
        // freedesktop os-release(5).
        assert!(is_arch_linux("ID=\"arch\"\n"));
    }

    #[test]
    fn test_is_arch_linux_via_id_like() {
        // Manjaro / EndeavourOS / Garuda — derivatives where users
        // still use paru/yay against the AUR.
        assert!(is_arch_linux("ID=manjaro\nID_LIKE=arch\n"));
        assert!(is_arch_linux("ID=endeavouros\nID_LIKE=\"arch\"\n"));
    }

    #[test]
    fn test_is_not_arch_linux() {
        assert!(!is_arch_linux("ID=fedora\n"));
        assert!(!is_arch_linux("ID=debian\nID_LIKE=\"\"\n"));
        assert!(!is_arch_linux("ID=ubuntu\nID_LIKE=debian\n"));
        assert!(!is_arch_linux(
            "ID=opensuse-tumbleweed\nID_LIKE=\"suse opensuse\"\n"
        ));
        // Empty file (some minimal containers don't ship os-release)
        assert!(!is_arch_linux(""));
    }

    // ── Binary replacement ──────────────────────────────────────────────

    #[test]
    fn test_replace_binary_roundtrip() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let exe_path = dir.path().join("mold");

        // Create a fake "current" binary
        std::fs::write(&exe_path, b"old-binary-content").unwrap();
        std::fs::set_permissions(&exe_path, std::fs::Permissions::from_mode(0o755)).unwrap();

        // Replace it
        let new_content = b"new-binary-content-v2";
        replace_binary(new_content, &exe_path).unwrap();

        // Verify new content
        let actual = std::fs::read(&exe_path).unwrap();
        assert_eq!(actual, new_content);

        // Verify permissions
        let perms = std::fs::metadata(&exe_path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o755);

        // Verify backup was cleaned up
        assert!(!exe_path.with_extension("old").exists());
    }

    #[test]
    fn test_replace_binary_no_leftover_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let exe_path = dir.path().join("mold");
        std::fs::write(&exe_path, b"original").unwrap();

        replace_binary(b"updated", &exe_path).unwrap();

        // No .mold-update-* temp files should remain
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".mold-update-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
    }
}
