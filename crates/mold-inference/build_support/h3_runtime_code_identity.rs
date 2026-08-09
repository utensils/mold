#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.private-runtime-code.v1\0";
const PRIVATE_SERVER_RELATIVE_PATH: &str = "crates/mold-inference/src/minimax_h3/private_server.rs";
const REVIEWED_ALLOWLIST_DECLARATION: &[u8] = b"const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256";
const NORMALIZED_ALLOWLIST: &[u8] =
    b"const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256: &[&str] = &[\n    /* reviewed values excluded from runtime-code identity */\n];";

// This is the complete local dependency closure of mold-server. Cargo.lock
// binds all registry and git dependencies. Keep this list explicit so adding a
// new local server dependency fails review visibly rather than silently
// weakening the campaign identity.
const LOCAL_RUNTIME_CRATES: &[&str] = &[
    "mold-core",
    "mold-catalog",
    "mold-db",
    "mold-scheduler",
    "mold-candle",
    "mold-inference",
    "mold-server",
];

const EXCLUDED_CRATE_DIRECTORIES: &[&str] = &["benches", "examples", "tests", "target"];

const BUILD_ENVIRONMENT_KEYS: &[&str] = &[
    "AR",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_INCREMENTAL",
    "CUDA_COMPUTE_CAP",
    "CUDA_HOME",
    "CUDA_NVCC_FLAGS",
    "CUDA_PATH",
    "DEBUG",
    "HOST",
    "LD",
    "LDFLAGS",
    "MACOSX_DEPLOYMENT_TARGET",
    "NVCC",
    "NVCC_APPEND_FLAGS",
    "NVCC_CCBIN",
    "NVCC_PREPEND_FLAGS",
    "OPT_LEVEL",
    "PROFILE",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTFLAGS",
    "TARGET",
];

// `CUDA_HOME` and friends are captured above, but only as strings. A toolkit
// upgraded in place under the same prefix leaves every captured string and
// `rustc -vV` identical while producing a different compiled executable, so the
// campaign identity must bind what the native compilers report about
// themselves, not merely where they live.
//
// Identifying a *different* nvcc than the one that compiled the kernels would
// be worse than identifying none, so this mirrors `cudaforge`'s own search
// order exactly (`cudaforge-0.1.6/src/toolkit.rs:108-135`): the `NVCC`
// override, then `PATH`, then `CUDA_HOME`, then `CUDA_PATH` on Windows.
// cudaforge hands the host compiler to `nvcc -ccbin` from `NVCC_CCBIN`
// (`cudaforge-0.1.6/src/builder.rs:393`), so that variable wins here too.
const CUDA_HOME_KEY: &str = "CUDA_HOME";
const CUDA_PATH_KEY: &str = "CUDA_PATH";
const NVCC_OVERRIDE_KEY: &str = "NVCC";
const NVCC_HOST_COMPILER_KEY: &str = "NVCC_CCBIN";

pub const CANONICAL_H3_SERVER_FEATURES: &[&str] = &[
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
    "CARGO_FEATURE_H3_PRIVATE_UAT",
    "CARGO_FEATURE_MP4",
    "CARGO_FEATURE_NVML",
];

pub fn collect_runtime_inputs(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    validate_local_dependency_closure(workspace_root)?;
    let mut inputs = Vec::new();
    for relative in ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        collect_required_file(&workspace_root.join(relative), &mut inputs)?;
    }
    for crate_name in LOCAL_RUNTIME_CRATES {
        collect_crate_inputs(&workspace_root.join("crates").join(crate_name), &mut inputs)?;
    }
    inputs.sort();
    inputs.dedup();
    Ok(inputs)
}

pub fn collect_build_environment() -> Result<Vec<(String, String)>, String> {
    let mut environment = std::env::vars_os()
        .filter_map(|(key, value)| {
            let key = key.into_string().ok()?;
            should_capture_environment_key(&key).then_some((key, value))
        })
        .map(|(key, value)| {
            value
                .into_string()
                .map(|value| (key.clone(), value))
                .map_err(|_| format!("runtime build environment {key} is not UTF-8"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .arg("-vV")
        .output()
        .map_err(|error| format!("failed to execute rustc -vV: {error}"))?;
    if !output.status.success() {
        return Err(format!("rustc -vV failed with status {}", output.status));
    }
    let verbose_version = String::from_utf8(output.stdout)
        .map_err(|_| "rustc -vV output is not UTF-8".to_string())?;
    environment.push((
        "RUSTC_VERBOSE_VERSION".into(),
        verbose_version.trim_end().into(),
    ));
    environment.push((
        "MOLD_H3_CANONICAL_SERVER_FEATURES".into(),
        CANONICAL_H3_SERVER_FEATURES.join(","),
    ));
    environment.extend(collect_native_toolchain_identity()?.0);
    environment.sort();
    if environment.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("runtime build environment contains duplicate keys".into());
    }
    Ok(environment)
}

/// Identity of the native compilers that turn this source into the measured
/// executable, plus the binaries whose in-place replacement must invalidate the
/// build script.
///
/// CUDA is part of the canonical campaign feature set, so an unresolvable or
/// unrunnable `nvcc` fails the build rather than silently producing an identity
/// that cannot distinguish two toolkits.
pub fn collect_native_toolchain_identity() -> Result<NativeToolchainIdentity, String> {
    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    native_toolchain_identity_for(cuda, resolve_cuda_compiler(), host_compiler_command())
}

/// Environment entries to hash, paired with the binaries to watch.
pub type NativeToolchainIdentity = (Vec<(String, String)>, Vec<PathBuf>);

pub fn native_toolchain_identity_for(
    cuda: bool,
    nvcc: Option<PathBuf>,
    host_compiler: PathBuf,
) -> Result<NativeToolchainIdentity, String> {
    let mut entries = Vec::new();
    let mut watched = Vec::new();
    if !cuda {
        entries.push(("MOLD_H3_NATIVE_CUDA_TOOLCHAIN".into(), "absent".into()));
        return Ok((entries, watched));
    }
    let nvcc = nvcc.ok_or_else(|| {
        "private H3 CUDA campaign builds must resolve nvcc to bind the toolkit identity".to_string()
    })?;
    // Which nvcc `PATH` selected is itself an input: two toolkits can report
    // the same version banner from different prefixes.
    entries.push((
        "MOLD_H3_NVCC_PATH".into(),
        path_identity(&nvcc, "nvcc")?.into(),
    ));
    entries.push((
        "MOLD_H3_NVCC_VERSION".into(),
        command_identity(&nvcc, "--version")?,
    ));
    watched.push(nvcc);
    // nvcc delegates to a host compiler, so its upgrade changes the executable
    // just as a toolkit upgrade does.
    entries.push((
        "MOLD_H3_HOST_COMPILER_PATH".into(),
        path_identity(&host_compiler, "host compiler")?.into(),
    ));
    entries.push((
        "MOLD_H3_HOST_COMPILER_VERSION".into(),
        command_identity(&host_compiler, "--version")?,
    ));
    if host_compiler.is_absolute() {
        watched.push(host_compiler);
    }
    Ok((entries, watched))
}

fn path_identity<'a>(program: &'a Path, label: &str) -> Result<&'a str, String> {
    program
        .to_str()
        .ok_or_else(|| format!("resolved {label} path is not UTF-8"))
}

fn command_identity(program: &Path, argument: &str) -> Result<String, String> {
    let output = Command::new(program)
        .arg(argument)
        .output()
        .map_err(|error| {
            format!(
                "failed to execute {} {argument}: {error}",
                program.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "{} {argument} failed with status {}",
            program.display(),
            output.status
        ));
    }
    let mut reported = String::from_utf8(output.stdout)
        .map_err(|_| format!("{} {argument} output is not UTF-8", program.display()))?;
    if reported.trim().is_empty() {
        reported = String::from_utf8(output.stderr)
            .map_err(|_| format!("{} {argument} output is not UTF-8", program.display()))?;
    }
    let reported = reported.trim_end().to_string();
    if reported.is_empty() {
        return Err(format!(
            "{} {argument} reported no version",
            program.display()
        ));
    }
    Ok(reported)
}

/// Mirror of `cudaforge::toolkit::find_nvcc`, so the identity describes the
/// compiler that actually built the kernels rather than a different one that
/// happens to be installed.
fn resolve_cuda_compiler() -> Option<PathBuf> {
    resolve_cuda_compiler_from(
        std::env::var_os(NVCC_OVERRIDE_KEY).map(PathBuf::from),
        search_path_for("nvcc"),
        std::env::var_os(CUDA_HOME_KEY).map(PathBuf::from),
        std::env::var_os(CUDA_PATH_KEY).map(PathBuf::from),
        &|path| path.exists(),
    )
}

/// Precedence half of [`resolve_cuda_compiler`], split out so the order can be
/// tested without mutating this process's environment.
pub fn resolve_cuda_compiler_from(
    nvcc_override: Option<PathBuf>,
    path_hit: Option<PathBuf>,
    cuda_home: Option<PathBuf>,
    cuda_path: Option<PathBuf>,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    if let Some(candidate) = nvcc_override {
        if exists(&candidate) {
            return Some(candidate);
        }
    }
    if let Some(candidate) = path_hit {
        return Some(candidate);
    }
    if let Some(home) = cuda_home {
        let candidate = home.join("bin").join("nvcc");
        if exists(&candidate) {
            return Some(candidate);
        }
    }
    if let Some(path) = cuda_path {
        let candidate = path.join("bin").join("nvcc.exe");
        if exists(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn host_compiler_command() -> PathBuf {
    // cudaforge passes NVCC_CCBIN to `nvcc -ccbin`, so it, not CC, decides
    // which host compiler compiles the device code's host halves.
    for key in [NVCC_HOST_COMPILER_KEY, "CC"] {
        if let Some(value) = std::env::var_os(key).filter(|value| !value.is_empty()) {
            let candidate = PathBuf::from(value);
            if candidate.is_absolute() {
                return candidate;
            }
            if let Some(found) = candidate.to_str().and_then(search_path_for) {
                return found;
            }
            return candidate;
        }
    }
    search_path_for("cc").unwrap_or_else(|| PathBuf::from("cc"))
}

/// Absolute path of the first `PATH` entry holding `program`.
///
/// A bare command name would still yield a version string, but it could not be
/// watched for in-place replacement, which is the point of resolving it.
fn search_path_for(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
}

pub fn validate_canonical_h3_server_features() -> Result<(), String> {
    if std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none() {
        return Ok(());
    }
    let actual = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("CARGO_FEATURE_") && key != "CARGO_FEATURE_DEFAULT")
        .collect::<Vec<_>>();
    validate_canonical_h3_server_feature_keys(&actual)
}

pub fn validate_canonical_h3_server_feature_keys(actual: &[String]) -> Result<(), String> {
    let actual = actual.iter().cloned().collect::<BTreeSet<_>>();
    let expected = CANONICAL_H3_SERVER_FEATURES
        .iter()
        .map(|key| (*key).to_string())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "private H3 server features differ from the canonical campaign build: expected {expected:?}, actual {actual:?}"
        ));
    }
    Ok(())
}

pub fn canonical_h3_server_feature_rerun_keys() -> impl Iterator<Item = &'static str> {
    CANONICAL_H3_SERVER_FEATURES
        .iter()
        .copied()
        .chain(["CARGO_FEATURE_DEFAULT"])
}

pub fn build_environment_rerun_keys() -> Vec<String> {
    let mut keys = BUILD_ENVIRONMENT_KEYS
        .iter()
        .map(|key| (*key).to_string())
        .chain(["RUSTC".to_string()])
        .chain(
            std::env::vars_os()
                .filter_map(|(key, _)| key.into_string().ok())
                .filter(|key| should_capture_environment_key(key)),
        )
        .collect::<Vec<_>>();
    keys.sort();
    keys.dedup();
    keys
}

fn should_capture_environment_key(key: &str) -> bool {
    BUILD_ENVIRONMENT_KEYS.contains(&key)
        || key.starts_with("CARGO_CFG_")
        || key.starts_with("CARGO_FEATURE_")
        || key.starts_with("CARGO_PROFILE_")
        || key.starts_with("CARGO_TARGET_")
            && (key.ends_with("_LINKER") || key.ends_with("_RUNNER") || key.ends_with("_RUSTFLAGS"))
}

fn validate_local_dependency_closure(workspace_root: &Path) -> Result<(), String> {
    let crates_root = workspace_root.join("crates");
    let canonical_crates_root = fs::canonicalize(&crates_root)
        .map_err(|error| format!("failed to resolve {}: {error}", crates_root.display()))?;
    let mut pending = vec!["mold-server".to_string()];
    let mut discovered = BTreeSet::new();
    while let Some(crate_name) = pending.pop() {
        if !discovered.insert(crate_name.clone()) {
            continue;
        }
        let crate_root = crates_root.join(&crate_name);
        let manifest_path = crate_root.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .map_err(|error| format!("failed to read {}: {error}", manifest_path.display()))?;
        for path_dependency in manifest_path_assignments(&manifest) {
            let target = crate_root.join(path_dependency);
            if !target.is_dir() {
                continue;
            }
            let canonical_target = fs::canonicalize(&target)
                .map_err(|error| format!("failed to resolve {}: {error}", target.display()))?;
            let relative = canonical_target
                .strip_prefix(&canonical_crates_root)
                .map_err(|_| {
                    format!(
                        "local runtime dependency {} escapes the workspace crates directory",
                        target.display()
                    )
                })?;
            let mut components = relative.components();
            let dependency = components
                .next()
                .and_then(|component| component.as_os_str().to_str())
                .ok_or_else(|| {
                    format!("local runtime dependency {} is invalid", target.display())
                })?;
            if components.next().is_some() {
                return Err(format!(
                    "local runtime dependency {} is not a crate root",
                    target.display()
                ));
            }
            pending.push(dependency.to_string());
        }
    }

    let expected = LOCAL_RUNTIME_CRATES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    if discovered != expected {
        return Err(format!(
            "reviewed local runtime dependency closure differs: expected {expected:?}, discovered {discovered:?}"
        ));
    }
    Ok(())
}

fn manifest_path_assignments(manifest: &str) -> Vec<&str> {
    let mut paths = Vec::new();
    for line in manifest.lines() {
        let line = line.split('#').next().unwrap_or_default();
        let mut search_from = 0_usize;
        while let Some(relative) = line[search_from..].find("path") {
            let index = search_from + relative;
            let before_is_boundary = index == 0
                || !line.as_bytes()[index - 1].is_ascii_alphanumeric()
                    && line.as_bytes()[index - 1] != b'_';
            let after = &line[index + 4..];
            let after_is_boundary = after
                .as_bytes()
                .first()
                .is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_');
            let assignment = after.trim_start();
            if before_is_boundary && after_is_boundary && assignment.starts_with('=') {
                let value = assignment[1..].trim_start();
                if let Some(quote) = value.as_bytes().first().copied() {
                    if quote == b'"' || quote == b'\'' {
                        if let Some(end) =
                            value.as_bytes()[1..].iter().position(|byte| *byte == quote)
                        {
                            paths.push(&value[1..end + 1]);
                        }
                    }
                }
            }
            search_from = index + 4;
        }
    }
    paths
}

fn collect_required_file(path: &Path, inputs: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "runtime identity input {} is a symbolic link",
            path.display()
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(format!(
            "runtime identity input {} is not a regular file",
            path.display()
        ));
    }
    inputs.push(path.to_path_buf());
    Ok(())
}

fn collect_crate_inputs(directory: &Path, inputs: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| format!("failed to inspect {}: {error}", directory.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "runtime identity directory {} is a symbolic link",
            directory.display()
        ));
    }
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "runtime identity path {} is not a directory",
            directory.display()
        ));
    }

    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to enumerate {}: {error}", directory.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "runtime identity entry {} is a symbolic link",
                path.display()
            ));
        }
        if file_type.is_dir() {
            let excluded = entry
                .file_name()
                .to_str()
                .is_some_and(|name| EXCLUDED_CRATE_DIRECTORIES.contains(&name));
            if !excluded {
                collect_crate_inputs(&path, inputs)?;
            }
        } else if file_type.is_file() {
            inputs.push(path);
        } else {
            return Err(format!(
                "runtime identity entry {} is not a regular file or directory",
                path.display()
            ));
        }
    }
    Ok(())
}

pub fn identity_for_workspace(workspace_root: &Path, inputs: &[PathBuf]) -> Result<String, String> {
    identity_for_workspace_and_environment(workspace_root, inputs, &[])
}

pub fn identity_for_workspace_and_environment(
    workspace_root: &Path,
    inputs: &[PathBuf],
    environment: &[(String, String)],
) -> Result<String, String> {
    let mut entries = Vec::with_capacity(inputs.len());
    for path in inputs {
        let relative = path.strip_prefix(workspace_root).map_err(|_| {
            format!(
                "runtime identity input {} escapes {}",
                path.display(),
                workspace_root.display()
            )
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| format!("runtime identity path {} is not UTF-8", path.display()))?;
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("failed to inspect runtime input {relative}: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(format!(
                "runtime identity input {relative} is not a regular file"
            ));
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read runtime input {relative}: {error}"))?;
        entries.push((relative.to_string(), bytes));
    }
    identity_for_entries_and_environment(&entries, environment)
}

pub fn identity_for_entries(entries: &[(String, Vec<u8>)]) -> Result<String, String> {
    identity_for_entries_and_environment(entries, &[])
}

pub fn identity_for_entries_and_environment(
    entries: &[(String, Vec<u8>)],
    environment: &[(String, String)],
) -> Result<String, String> {
    let mut ordered = entries.to_vec();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    if ordered.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("runtime identity inputs contain duplicate paths".into());
    }
    let mut digest = Sha256::new();
    digest.update(IDENTITY_DOMAIN);
    digest.update((ordered.len() as u64).to_le_bytes());
    for (path, bytes) in ordered {
        let normalized = normalize_input(&path, &bytes)?;
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update((normalized.len() as u64).to_le_bytes());
        digest.update(&normalized);
    }
    let mut environment = environment.to_vec();
    environment.sort();
    if environment.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("runtime build environment contains duplicate keys".into());
    }
    digest.update(b"build-environment\0");
    digest.update((environment.len() as u64).to_le_bytes());
    for (key, value) in environment {
        digest.update((key.len() as u64).to_le_bytes());
        digest.update(key.as_bytes());
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn normalize_input(relative_path: &str, bytes: &[u8]) -> Result<Vec<u8>, String> {
    if relative_path != PRIVATE_SERVER_RELATIVE_PATH {
        return Ok(bytes.to_vec());
    }
    let declaration = find_once(bytes, REVIEWED_ALLOWLIST_DECLARATION)?;
    let suffix = &bytes[declaration..];
    let terminator_relative = suffix
        .windows(2)
        .position(|window| window == b"];")
        .ok_or_else(|| "reviewed runtime allowlist has no terminator".to_string())?;
    let terminator = declaration + terminator_relative + 2;
    let mut normalized = Vec::with_capacity(bytes.len());
    normalized.extend_from_slice(&bytes[..declaration]);
    normalized.extend_from_slice(NORMALIZED_ALLOWLIST);
    normalized.extend_from_slice(&bytes[terminator..]);
    Ok(normalized)
}

fn find_once(haystack: &[u8], needle: &[u8]) -> Result<usize, String> {
    let matches = haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, window)| (window == needle).then_some(index))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err("reviewed runtime allowlist declaration is missing".into()),
        _ => Err("reviewed runtime allowlist declaration is duplicated".into()),
    }
}
