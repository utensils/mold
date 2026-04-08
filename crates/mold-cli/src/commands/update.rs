//! Self-update command — downloads and installs the latest mold release from GitHub.

use std::io::Read as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::theme;

const GITHUB_REPO: &str = "utensils/mold";
const GITHUB_API_BASE: &str = "https://api.github.com";

// ── GitHub API types ────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(serde::Deserialize)]
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
            let cuda_arch = detect_cuda_arch();
            Ok(format!(
                "mold-x86_64-unknown-linux-gnu-cuda-{cuda_arch}.tar.gz"
            ))
        }
        _ => bail!("unsupported platform: {os}/{arch}"),
    }
}

/// Detect CUDA GPU architecture via nvidia-smi, env override, or default.
fn detect_cuda_arch() -> String {
    if let Ok(arch) = std::env::var("MOLD_CUDA_ARCH") {
        return arch;
    }

    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let cc = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let parts: Vec<&str> = cc.split('.').collect();
            if let Some(major) = parts.first().and_then(|s| s.parse::<u32>().ok()) {
                if major >= 12 {
                    return "sm120".to_string();
                }
            }
            "sm89".to_string()
        }
        _ => {
            eprintln!(
                "{} Could not detect GPU architecture, defaulting to sm89",
                theme::prefix_warning()
            );
            "sm89".to_string()
        }
    }
}

// ── Package manager detection ───────────────────────────────────────────────

/// Check if the binary path looks like it was installed by a package manager.
/// Returns a hint string if so.
fn detect_package_manager(exe_path: &Path) -> Option<&'static str> {
    let path_str = exe_path.to_string_lossy();
    if path_str.contains("/nix/store/") {
        Some("nix flake update")
    } else if path_str.contains("/Cellar/") || path_str.contains("/homebrew/") {
        Some("brew upgrade mold")
    } else {
        None
    }
}

// ── SHA-256 checksum verification ───────────────────────────────────────────

/// Parse a SHA256SUMS file and verify the checksum for `asset_name` against `data`.
fn verify_checksum(sums_content: &str, asset_name: &str, data: &[u8]) -> Result<()> {
    let expected = sums_content
        .lines()
        .find_map(|line| {
            // Format: "{hash}  {filename}" (two-space separator, sha256sum convention)
            let (hash, name) = line.split_once("  ")?;
            if name.trim() == asset_name {
                Some(hash.trim().to_string())
            } else {
                None
            }
        })
        .with_context(|| format!("asset {asset_name} not found in SHA256SUMS"))?;

    let mut hasher = Sha256::new();
    hasher.update(data);
    let actual = format!("{:x}", hasher.finalize());

    if actual != expected {
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

// ── Main command ────────────────────────────────────────────────────────────

pub async fn run(check: bool, force: bool, version: Option<String>) -> Result<()> {
    let current = mold_core::build_info::VERSION;
    eprintln!("{} Current version: {current}", theme::icon_info());

    // Check if installed via package manager
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

    // Check write permission
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

    eprintln!(
        "{} {action}: {current} -> {remote_version}",
        theme::icon_info()
    );

    // Detect correct asset
    let asset_name = detect_asset_name()?;

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .with_context(|| {
            format!(
                "release {} has no asset matching {asset_name}",
                release.tag_name
            )
        })?;

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
    let sums_content = sums_resp
        .text()
        .await
        .context("failed to read SHA256SUMS")?;

    // Verify checksum
    verify_checksum(&sums_content, &asset_name, &archive_data)?;
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

    // ── Package manager detection ───────────────────────────────────────

    #[test]
    fn test_detect_nix_store() {
        let path = Path::new("/nix/store/abc123-mold/bin/mold");
        assert_eq!(detect_package_manager(path), Some("nix flake update"));
    }

    #[test]
    fn test_detect_homebrew() {
        let path = Path::new("/opt/homebrew/Cellar/mold/0.6.1/bin/mold");
        assert_eq!(detect_package_manager(path), Some("brew upgrade mold"));
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
