//! CUDA distribution routing shared by cloud providers and the self-updater.
//!
//! Native image targets are selected only from explicit GPU families. Generic
//! `Blackwell` is ambiguous between datacenter sm_100 and consumer sm_120, so
//! it deliberately falls back to the forward-compatible default sm_89 image.
//! Generic `Ampere` uses sm_80, the lowest Ampere target.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaImageTarget {
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm100,
    Sm120,
}

impl CudaImageTarget {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Sm80 => "-sm80",
            Self::Sm86 => "-sm86",
            Self::Sm89 => "",
            Self::Sm90 => "-sm90",
            Self::Sm100 => "-sm100",
            Self::Sm120 => "-sm120",
        }
    }

    pub const fn manifest_key(self) -> &'static str {
        match self {
            Self::Sm80 => "sm80",
            Self::Sm86 => "sm86",
            Self::Sm89 => "sm89",
            Self::Sm90 => "sm90",
            Self::Sm100 => "sm100",
            Self::Sm120 => "sm120",
        }
    }
}

fn words(name: &str) -> Vec<String> {
    name.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| word.to_ascii_lowercase())
        .collect()
}

fn has_word(words: &[String], needle: &str) -> bool {
    words.iter().any(|word| word == needle)
}

fn has_adjacent(words: &[String], first: &str, second: &str) -> bool {
    words
        .windows(2)
        .any(|window| window[0] == first && window[1] == second)
}

fn has_rtx_30_series(words: &[String]) -> bool {
    words.windows(2).any(|window| {
        window[0] == "rtx"
            && matches!(
                window[1].as_str(),
                "30" | "3050" | "3060" | "3070" | "3080" | "3090"
            )
    })
}

fn has_rtx_50_series(words: &[String]) -> bool {
    words.windows(2).any(|window| {
        window[0] == "rtx"
            && matches!(
                window[1].as_str(),
                "50" | "5050" | "5060" | "5070" | "5080" | "5090"
            )
    })
}

/// Select the published container target for a provider GPU display name.
pub fn image_target_for_gpu_name(name: &str) -> CudaImageTarget {
    let words = words(name);

    if ["b200", "gb200", "b300", "gb300"]
        .iter()
        .any(|model| has_word(&words, model))
    {
        CudaImageTarget::Sm100
    } else if has_adjacent(&words, "rtx", "pro") || has_rtx_50_series(&words) {
        CudaImageTarget::Sm120
    } else if ["h100", "h200", "gh200"]
        .iter()
        .any(|model| has_word(&words, model))
        || has_word(&words, "hopper")
    {
        CudaImageTarget::Sm90
    } else if ["a100", "a30"].iter().any(|model| has_word(&words, model)) {
        CudaImageTarget::Sm80
    } else if [
        "a10", "a10g", "a40", "a4000", "a4500", "a5000", "a5500", "a6000", "a16", "a2", "3050",
        "3060", "3070", "3080", "3090",
    ]
    .iter()
    .any(|model| has_word(&words, model))
        || has_rtx_30_series(&words)
    {
        CudaImageTarget::Sm86
    } else if has_word(&words, "ampere") {
        CudaImageTarget::Sm80
    } else {
        // Ada names and unknown or ambiguous families retain the established
        // sm_89 image. In particular, never guess generic Blackwell as sm_120.
        CudaImageTarget::Sm89
    }
}

/// Build a rolling (`latest*`) or human-readable versioned image tag.
pub fn image_tag_for_gpu_name(name: &str, version: &str) -> String {
    let base = match version.trim().trim_start_matches('v') {
        "" | "latest" => "latest",
        version => version,
    };
    format!("{base}{}", image_target_for_gpu_name(name).suffix())
}

/// Image tag base embedded by release CI.
///
/// Main-branch rolling binaries set this to `latest`; tagged official builds
/// set the exact tag. Source, crates.io, and Nix builds intentionally use the
/// mutable rolling channel unless their build system explicitly supplies an
/// official distribution version. A local Cargo package version is not
/// evidence that the matching container was published.
pub fn distribution_image_version() -> &'static str {
    option_env!("MOLD_DISTRIBUTION_IMAGE_VERSION").unwrap_or("latest")
}

pub const OFFICIAL_IMAGE_REPOSITORY: &str = "ghcr.io/utensils/mold";

#[derive(Debug, serde::Deserialize)]
struct ContainerDigestManifest {
    schema_version: String,
    release_tag: String,
    source_sha: String,
    repository: String,
    targets: std::collections::BTreeMap<String, ContainerDigestTarget>,
}

#[derive(Debug, serde::Deserialize)]
struct ContainerDigestTarget {
    tag: String,
    digest: String,
}

fn normalized_stable_tag(version: &str) -> Option<String> {
    let version = version.trim();
    let bare = version.strip_prefix('v').unwrap_or(version);
    let fields = bare.split('.').collect::<Vec<_>>();
    let valid = fields.len() == 3
        && fields
            .iter()
            .all(|field| !field.is_empty() && field.chars().all(|ch| ch.is_ascii_digit()));
    valid.then(|| format!("v{bare}"))
}

pub fn is_exact_image_digest_reference(repository: &str) -> bool {
    let Some((name, digest)) = repository.split_once("@sha256:") else {
        return false;
    };
    !name.is_empty()
        && !name.contains('@')
        && digest.len() == 64
        && digest.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn image_reference_from_digest_manifest(
    repository: &str,
    gpu_name: &str,
    version: &str,
    manifest_json: &str,
) -> anyhow::Result<String> {
    let release_tag = normalized_stable_tag(version)
        .ok_or_else(|| anyhow::anyhow!("stable distribution version is invalid: {version}"))?;
    let manifest: ContainerDigestManifest = serde_json::from_str(manifest_json)
        .map_err(|error| anyhow::anyhow!("invalid container digest manifest: {error}"))?;
    anyhow::ensure!(
        manifest.schema_version == "mold.container-digests.v1",
        "unsupported container digest manifest schema"
    );
    anyhow::ensure!(
        manifest.release_tag == release_tag,
        "container digest manifest release mismatch"
    );
    anyhow::ensure!(
        manifest.source_sha.len() == 40
            && manifest.source_sha.chars().all(|ch| ch.is_ascii_hexdigit()),
        "container digest manifest contains an invalid source commit"
    );
    let build_sha = crate::build_info::GIT_SHA.trim();
    if build_sha != "unknown" && !build_sha.is_empty() {
        anyhow::ensure!(
            manifest
                .source_sha
                .to_ascii_lowercase()
                .starts_with(&build_sha.to_ascii_lowercase()),
            "container digest manifest source does not match this Mold build"
        );
    }
    anyhow::ensure!(
        manifest.repository == repository,
        "container digest manifest repository mismatch"
    );
    let target = image_target_for_gpu_name(gpu_name);
    let entry = manifest.targets.get(target.manifest_key()).ok_or_else(|| {
        anyhow::anyhow!("container digest manifest omits {}", target.manifest_key())
    })?;
    let expected_tag = image_tag_for_gpu_name(gpu_name, version);
    anyhow::ensure!(
        entry.tag == expected_tag,
        "container digest manifest tag mismatch for {}",
        target.manifest_key()
    );
    anyhow::ensure!(
        entry.digest.len() == 71
            && entry.digest.starts_with("sha256:")
            && entry.digest[7..].chars().all(|ch| ch.is_ascii_hexdigit()),
        "container digest manifest contains an invalid digest"
    );
    Ok(format!(
        "{repository}@{}",
        entry.digest.to_ascii_lowercase()
    ))
}

/// Resolve the exact image reference submitted to a cloud provider.
///
/// Rolling `latest*` builds intentionally use mutable tags. Official stable
/// builds fetch the target-to-digest manifest attached to that exact GitHub
/// release and submit `repository@sha256:...`; a missing or malformed manifest
/// fails closed. Custom repositories must already be supplied as an explicit
/// digest because Mold cannot authenticate their tag policy.
pub async fn resolve_distribution_image_reference(
    repository: &str,
    gpu_name: &str,
    version: &str,
) -> anyhow::Result<String> {
    let version = version.trim();
    if is_exact_image_digest_reference(repository) {
        return Ok(repository.to_string());
    }
    if version.is_empty() || version == "latest" {
        return Ok(format!(
            "{repository}:{}",
            image_tag_for_gpu_name(gpu_name, "latest")
        ));
    }
    anyhow::ensure!(
        repository == OFFICIAL_IMAGE_REPOSITORY,
        "stable provisioning from a custom image repository requires an explicit @sha256 digest"
    );
    let release_tag = normalized_stable_tag(version)
        .ok_or_else(|| anyhow::anyhow!("unsupported stable distribution version: {version}"))?;
    let url = format!(
        "https://github.com/utensils/mold/releases/download/{release_tag}/mold-container-digests.json"
    );
    let response = reqwest::Client::builder()
        .user_agent(format!("mold/{}", crate::build_info::VERSION))
        .build()?
        .get(&url)
        .send()
        .await
        .map_err(|error| anyhow::anyhow!("fetch stable container digest manifest: {error}"))?;
    anyhow::ensure!(
        response.status().is_success(),
        "stable container digest manifest returned HTTP {}; refusing mutable tag fallback",
        response.status()
    );
    let manifest_json = response.text().await?;
    image_reference_from_digest_manifest(repository, gpu_name, version, &manifest_json)
}

/// Select one of the CUDA release tarballs for an exact compute capability.
///
/// Release tarballs exist only for targets that CI builds and verifies
/// explicitly. PTX is forward-compatible, never backward-compatible, so a
/// target above the device's compute capability must not be selected.
pub fn release_arch_for_compute_cap(compute_cap: &str) -> Option<&'static str> {
    let (major, minor) = parse_compute_capability(compute_cap)?;

    match (major, minor) {
        (8, 6) => Some("sm86"),
        (10, _) => Some("sm100"),
        (12, _) => Some("sm120"),
        (8, 9) => Some("sm89"),
        _ => None,
    }
}

pub fn is_release_arch(arch: &str) -> bool {
    matches!(arch, "sm86" | "sm89" | "sm100" | "sm120")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReleaseCompatibilityClass {
    Sm86,
    Sm89,
    Sm100,
    Sm120,
}

fn parse_compute_capability(value: &str) -> Option<(u32, u32)> {
    let (major, minor) = value.trim().split_once('.')?;
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn compute_capability_class(value: &str) -> Option<ReleaseCompatibilityClass> {
    let (major, minor) = parse_compute_capability(value)?;
    match (major, minor) {
        (8, 6) => Some(ReleaseCompatibilityClass::Sm86),
        (8, 9) => Some(ReleaseCompatibilityClass::Sm89),
        (10, _) => Some(ReleaseCompatibilityClass::Sm100),
        (12, _) => Some(ReleaseCompatibilityClass::Sm120),
        _ => None,
    }
}

fn incompatible_fleet_message(caps: &[String]) -> String {
    format!(
        "no published Mold CUDA archive is proven compatible with every visible GPU \
         (compute capabilities: {}). Use CUDA_VISIBLE_DEVICES to expose one compatible \
         architecture family, or make a source build for the intended fleet with \
         CUDA_COMPUTE_CAP set to a verified fat-binary target list",
        caps.join(", ")
    )
}

/// Select one release archive that is proven compatible with every visible
/// device. Input order never affects the result and there is no 1/2-GPU limit.
///
/// A mixed 8.6/8.9 fleet uses the sm86 floor artifact: release CI verifies
/// exact embedded sm86 PTX, which both Ampere and the forward-compatible Ada
/// driver can JIT. Families without a published floor archive fail closed.
/// Datacenter and consumer Blackwell targets are deliberately not mixed
/// because their release artifacts are distinct.
pub fn release_arch_for_compute_caps<I, S>(caps: I) -> Result<&'static str, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut values = caps
        .into_iter()
        .map(|value| value.as_ref().trim().to_string())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    if values.is_empty() {
        return Err("no CUDA GPU is visible; set MOLD_CUDA_ARCH only when intentionally installing for another machine".to_string());
    }

    let classes = values
        .iter()
        .map(|value| compute_capability_class(value))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| incompatible_fleet_message(&values))?;

    if classes.iter().all(|class| {
        matches!(
            class,
            ReleaseCompatibilityClass::Sm86 | ReleaseCompatibilityClass::Sm89
        )
    }) {
        return Ok(if classes.contains(&ReleaseCompatibilityClass::Sm86) {
            "sm86"
        } else {
            "sm89"
        });
    }
    if classes
        .iter()
        .all(|class| *class == ReleaseCompatibilityClass::Sm100)
    {
        return Ok("sm100");
    }
    if classes
        .iter()
        .all(|class| *class == ReleaseCompatibilityClass::Sm120)
    {
        return Ok("sm120");
    }

    Err(incompatible_fleet_message(&values))
}

/// Validate an explicit archive override against the complete visible fleet.
pub fn release_arch_supports_compute_caps<I, S>(arch: &str, caps: I) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let selected = release_arch_for_compute_caps(caps)?;
    if arch == selected {
        Ok(())
    } else {
        Err(format!(
            "MOLD_CUDA_ARCH={arch} is not the archive selected for the complete visible fleet \
             ({selected}); hide incompatible devices with CUDA_VISIBLE_DEVICES or use a source build"
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NvidiaInventoryRow {
    index: String,
    uuid: String,
    compute_capability: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NvidiaMigRow {
    uuid: String,
    parent_uuid: String,
}

fn uuid_from_listing_line(line: &str) -> Option<&str> {
    let value = line.rsplit_once("(UUID: ")?.1.strip_suffix(')')?;
    (!value.is_empty()).then_some(value)
}

fn parse_mig_inventory(listing: &str) -> Vec<NvidiaMigRow> {
    let mut parent_uuid = None;
    let mut rows = Vec::new();
    for line in listing.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("GPU ") {
            parent_uuid = uuid_from_listing_line(trimmed)
                .filter(|uuid| uuid.starts_with("GPU-"))
                .map(str::to_string);
        } else if trimmed.starts_with("MIG ") {
            if let (Some(uuid), Some(parent_uuid)) =
                (uuid_from_listing_line(trimmed), parent_uuid.as_ref())
            {
                if uuid.starts_with("MIG-") {
                    rows.push(NvidiaMigRow {
                        uuid: uuid.to_string(),
                        parent_uuid: parent_uuid.clone(),
                    });
                }
            }
        }
    }
    rows
}

/// Parse `nvidia-smi --query-gpu=index,uuid,compute_cap` and apply the
/// process-level CUDA visibility boundary. This intentionally fails on unknown
/// selectors rather than accidentally selecting a hidden sibling.
pub fn visible_compute_capabilities(
    inventory_csv: &str,
    cuda_visible_devices: Option<&str>,
) -> Result<Vec<String>, String> {
    visible_compute_capabilities_with_mig(inventory_csv, "", cuda_visible_devices)
}

/// Apply CUDA visibility using physical GPU identifiers or MIG compute-instance
/// UUIDs from `nvidia-smi -L`. A MIG instance inherits its parent GPU's compute
/// capability for distribution artifact selection.
pub fn visible_compute_capabilities_with_mig(
    inventory_csv: &str,
    mig_listing: &str,
    cuda_visible_devices: Option<&str>,
) -> Result<Vec<String>, String> {
    let rows = inventory_csv
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let columns = line.split(',').map(str::trim).collect::<Vec<_>>();
            if columns.len() != 3
                || columns[0].is_empty()
                || columns[1].is_empty()
                || columns[2].is_empty()
            {
                return Err(format!("malformed nvidia-smi inventory row: {line}"));
            }
            Ok(NvidiaInventoryRow {
                index: columns[0].to_string(),
                uuid: columns[1].to_string(),
                compute_capability: columns[2].to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let selected = match cuda_visible_devices {
        None => rows,
        Some(value) if value.trim().is_empty() || value.trim() == "-1" => Vec::new(),
        Some(value) => {
            let mig_rows = parse_mig_inventory(mig_listing);
            let mut selected = Vec::new();
            for selector in value.split(',').map(str::trim) {
                if selector.is_empty() {
                    return Err("CUDA_VISIBLE_DEVICES contains an empty selector".to_string());
                }
                let matches = if selector.starts_with("MIG-") {
                    mig_rows
                        .iter()
                        .filter(|row| row.uuid == selector || row.uuid.starts_with(selector))
                        .filter_map(|mig| rows.iter().find(|row| row.uuid == mig.parent_uuid))
                        .collect::<Vec<_>>()
                } else {
                    rows.iter()
                        .filter(|row| {
                            row.index == selector
                                || row.uuid == selector
                                || (selector.starts_with("GPU-") && row.uuid.starts_with(selector))
                        })
                        .collect::<Vec<_>>()
                };
                if matches.len() != 1 {
                    return Err(format!(
                        "CUDA_VISIBLE_DEVICES selector {selector:?} matched {} CUDA devices; \
                         use an unambiguous GPU or MIG UUID",
                        matches.len()
                    ));
                }
                if !selected
                    .iter()
                    .any(|row: &&NvidiaInventoryRow| row.uuid == matches[0].uuid)
                {
                    selected.push(matches[0]);
                }
            }
            selected.into_iter().cloned().collect()
        }
    };

    Ok(selected
        .into_iter()
        .map(|row| row.compute_capability)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_explicit_provider_names_and_conservative_generic_names() {
        let cases = [
            ("A30", CudaImageTarget::Sm80),
            ("A100 80GB", CudaImageTarget::Sm80),
            ("Ampere", CudaImageTarget::Sm80),
            ("A10", CudaImageTarget::Sm86),
            ("A10G", CudaImageTarget::Sm86),
            ("A40", CudaImageTarget::Sm86),
            ("NVIDIA A40 Ampere", CudaImageTarget::Sm86),
            ("RTX A4000", CudaImageTarget::Sm86),
            ("RTX A4500", CudaImageTarget::Sm86),
            ("RTX A5000", CudaImageTarget::Sm86),
            ("RTX A5500", CudaImageTarget::Sm86),
            ("RTX A6000", CudaImageTarget::Sm86),
            ("A16", CudaImageTarget::Sm86),
            ("A2", CudaImageTarget::Sm86),
            ("GeForce RTX 3060 Laptop GPU", CudaImageTarget::Sm86),
            ("GeForce RTX 3050 Laptop GPU", CudaImageTarget::Sm86),
            ("GeForce RTX 3050 Ti", CudaImageTarget::Sm86),
            ("GeForce RTX 3070 Ti", CudaImageTarget::Sm86),
            ("GeForce RTX 3080 Ti", CudaImageTarget::Sm86),
            ("GeForce RTX 3090", CudaImageTarget::Sm86),
            ("GeForce RTX 3090 Ampere", CudaImageTarget::Sm86),
            ("H100", CudaImageTarget::Sm90),
            ("H200", CudaImageTarget::Sm90),
            ("B200", CudaImageTarget::Sm100),
            ("GB200", CudaImageTarget::Sm100),
            ("B300", CudaImageTarget::Sm100),
            ("GB300", CudaImageTarget::Sm100),
            ("RTX PRO 6000 Blackwell", CudaImageTarget::Sm120),
            ("NVIDIA RTX PRO 1000 Blackwell", CudaImageTarget::Sm120),
            ("nvidia rtx-pro-2000 blackwell", CudaImageTarget::Sm120),
            ("RTX PRO 4000 Blackwell", CudaImageTarget::Sm120),
            ("RTX PRO 4500 Blackwell", CudaImageTarget::Sm120),
            ("RTX PRO 5000 Blackwell", CudaImageTarget::Sm120),
            ("RTX PRO 6000 Blackwell Max-Q", CudaImageTarget::Sm120),
            ("GeForce RTX 5050", CudaImageTarget::Sm120),
            ("geforce rtx-5060", CudaImageTarget::Sm120),
            ("GeForce RTX 5060 Ti", CudaImageTarget::Sm120),
            ("GeForce RTX 5070", CudaImageTarget::Sm120),
            ("NVIDIA GeForce RTX-5070-Ti", CudaImageTarget::Sm120),
            ("GeForce RTX 5080", CudaImageTarget::Sm120),
            ("GeForce RTX 5090", CudaImageTarget::Sm120),
            ("Blackwell", CudaImageTarget::Sm89),
        ];

        for (name, expected) in cases {
            assert_eq!(image_target_for_gpu_name(name), expected, "{name}");
        }
    }

    #[test]
    fn tags_are_versioned_or_rolling_with_the_same_suffix() {
        assert_eq!(image_tag_for_gpu_name("A40", "latest"), "latest-sm86");
        assert_eq!(image_tag_for_gpu_name("A40", "v0.20.2"), "0.20.2-sm86");
        assert_eq!(image_tag_for_gpu_name("B300", "0.20.2"), "0.20.2-sm100");
        assert_eq!(image_tag_for_gpu_name("Blackwell", "0.20.2"), "0.20.2");
        assert!(!distribution_image_version().is_empty());
        assert!(is_exact_image_digest_reference(&format!(
            "registry.example/mold@sha256:{}",
            "a".repeat(64)
        )));
        assert!(!is_exact_image_digest_reference(
            "registry.example/mold@sha256:not-a-digest"
        ));
    }

    #[test]
    fn stable_image_reference_is_pinned_to_the_exact_target_digest() {
        let digest = format!("sha256:{}", "a".repeat(64));
        let source_sha = format!("{:0<40}", crate::build_info::GIT_SHA);
        let manifest = serde_json::json!({
            "schema_version": "mold.container-digests.v1",
            "release_tag": "v0.20.2",
            "source_sha": source_sha,
            "repository": OFFICIAL_IMAGE_REPOSITORY,
            "targets": {
                "sm120": {
                    "tag": "0.20.2-sm120",
                    "digest": digest
                }
            }
        });
        assert_eq!(
            image_reference_from_digest_manifest(
                OFFICIAL_IMAGE_REPOSITORY,
                "GeForce RTX 5070 Ti",
                "v0.20.2",
                &manifest.to_string()
            )
            .unwrap(),
            format!("{}@sha256:{}", OFFICIAL_IMAGE_REPOSITORY, "a".repeat(64))
        );
    }

    #[test]
    fn stable_image_reference_rejects_wrong_or_incomplete_manifests() {
        let base = serde_json::json!({
            "schema_version": "mold.container-digests.v1",
            "release_tag": "v0.20.2",
            "source_sha": format!("{:0<40}", crate::build_info::GIT_SHA),
            "repository": OFFICIAL_IMAGE_REPOSITORY,
            "targets": {}
        });
        assert!(image_reference_from_digest_manifest(
            OFFICIAL_IMAGE_REPOSITORY,
            "B200",
            "v0.20.2",
            &base.to_string()
        )
        .is_err());

        let mut wrong_release = base;
        wrong_release["release_tag"] = serde_json::json!("v0.20.3");
        assert!(image_reference_from_digest_manifest(
            OFFICIAL_IMAGE_REPOSITORY,
            "RTX 5090",
            "v0.20.2",
            &wrong_release.to_string()
        )
        .is_err());
    }

    #[test]
    fn release_arch_matrix_is_explicit() {
        let cases = [
            ("8.0", None),
            ("8.6", Some("sm86")),
            ("8.9", Some("sm89")),
            ("9.0", None),
            ("10.0", Some("sm100")),
            ("10.3", Some("sm100")),
            ("12.0", Some("sm120")),
            ("12.1", Some("sm120")),
        ];
        for (compute_cap, expected) in cases {
            assert_eq!(release_arch_for_compute_cap(compute_cap), expected);
        }
        assert_eq!(release_arch_for_compute_cap("11.0"), None);
        assert_eq!(release_arch_for_compute_cap("garbage"), None);
        assert_eq!(release_arch_for_compute_cap("010.003"), Some("sm100"));
        for malformed in [
            "10.garbage",
            "12.future",
            "10.",
            ".0",
            "10.0.0",
            "10x0",
            "sm86",
            "+10.0",
            "10.+0",
            "4294967296.0",
            "10.4294967296",
        ] {
            assert_eq!(release_arch_for_compute_cap(malformed), None, "{malformed}");
            assert!(
                release_arch_for_compute_caps([malformed]).is_err(),
                "{malformed}"
            );
        }
    }

    #[test]
    fn fleet_release_arch_is_order_independent_and_supports_arbitrary_n() {
        assert_eq!(
            release_arch_for_compute_caps(["8.6", "8.6"]).unwrap(),
            "sm86"
        );
        assert_eq!(
            release_arch_for_compute_caps(["8.9", "8.6"]).unwrap(),
            "sm86"
        );
        assert_eq!(
            release_arch_for_compute_caps(["8.6", "8.9"]).unwrap(),
            "sm86"
        );
        assert_eq!(
            release_arch_for_compute_caps(std::iter::repeat_n("10.0", 64)).unwrap(),
            "sm100"
        );
        assert_eq!(
            release_arch_for_compute_caps(std::iter::repeat_n("12.0", 8)).unwrap(),
            "sm120"
        );
    }

    #[test]
    fn fleet_release_arch_fails_closed_for_incompatible_mixed_families() {
        for caps in [
            vec!["8.6", "10.0"],
            vec!["8.0"],
            vec!["9.0"],
            vec!["8.9", "9.0"],
            vec!["10.0", "12.0"],
            vec!["8.9", "12.0"],
            vec!["8.6", "garbage"],
        ] {
            let error = release_arch_for_compute_caps(caps).unwrap_err();
            assert!(
                error.contains("source build") && error.contains("CUDA_COMPUTE_CAP"),
                "{error}"
            );
        }
        assert!(release_arch_for_compute_caps(Vec::<&str>::new()).is_err());
        assert!(release_arch_supports_compute_caps("sm89", ["8.6", "8.6"]).is_err());
        assert!(release_arch_supports_compute_caps("sm86", ["8.6", "8.9"]).is_ok());
        assert!(release_arch_supports_compute_caps("sm89", ["8.6", "8.9"]).is_err());
    }

    #[test]
    fn visible_inventory_respects_cuda_visible_devices_and_order() {
        let inventory = "\
0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 8.6
1, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, 8.9
2, GPU-cccccccc-cccc-cccc-cccc-cccccccccccc, 10.0
";
        assert_eq!(
            visible_compute_capabilities(inventory, None).unwrap(),
            ["8.6", "8.9", "10.0"]
        );
        assert_eq!(
            visible_compute_capabilities(inventory, Some("1,0")).unwrap(),
            ["8.9", "8.6"]
        );
        assert_eq!(
            visible_compute_capabilities(
                inventory,
                Some("GPU-cccccccc-cccc-cccc-cccc-cccccccccccc")
            )
            .unwrap(),
            ["10.0"]
        );
        assert!(visible_compute_capabilities(inventory, Some("-1"))
            .unwrap()
            .is_empty());
        assert!(visible_compute_capabilities(inventory, Some("MIG-unknown")).is_err());
        assert_eq!(
            visible_compute_capabilities(inventory, Some("0,0")).unwrap(),
            ["8.6"]
        );
        for visibility in [",", ",0", "0,", "0,,1"] {
            assert!(
                visible_compute_capabilities(inventory, Some(visibility)).is_err(),
                "{visibility:?}"
            );
        }

        for malformed_inventory in [
            "0,GPU-two-columns",
            "0,GPU-too-many,8.6,extra",
            "0,,8.6",
            "0,GPU-empty-cap,",
            "0,GPU-valid,8.6\nmalformed-hidden-row",
        ] {
            assert!(
                visible_compute_capabilities(malformed_inventory, None).is_err(),
                "{malformed_inventory:?}"
            );
            assert!(
                visible_compute_capabilities(malformed_inventory, Some("0")).is_err(),
                "{malformed_inventory:?}"
            );
        }
    }

    #[test]
    fn visible_inventory_maps_mig_uuids_to_parent_compute_capability() {
        let inventory = "\
0, GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, 9.0
1, GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, 10.0
";
        let listing = "\
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa)
  MIG 1g.10gb Device 0: (UUID: MIG-11111111-1111-1111-1111-111111111111)
  MIG 1g.10gb Device 1: (UUID: MIG-12222222-2222-2222-2222-222222222222)
GPU 1: NVIDIA B200 (UUID: GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb)
  MIG 1g.23gb Device 0: (UUID: MIG-GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb/1/0)
";
        assert_eq!(
            visible_compute_capabilities_with_mig(
                inventory,
                listing,
                Some("MIG-GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb/1/0")
            )
            .unwrap(),
            ["10.0"]
        );
        assert_eq!(
            visible_compute_capabilities_with_mig(
                inventory,
                listing,
                Some("MIG-11111111-1111-1111-1111-111111111111")
            )
            .unwrap(),
            ["9.0"]
        );
        assert_eq!(
            visible_compute_capabilities_with_mig(inventory, listing, Some("MIG-12")).unwrap(),
            ["9.0"]
        );
        assert!(visible_compute_capabilities_with_mig(inventory, listing, Some("MIG-1")).is_err());
        assert!(visible_compute_capabilities_with_mig(inventory, "", Some("MIG-unknown")).is_err());
    }
}
