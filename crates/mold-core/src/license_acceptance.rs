//! Explicit, recorded acceptance of third-party model licenses.
//!
//! A few manifest files are published under terms mold cannot accept on the
//! user's behalf — most notably the InsightFace antelopev2 pretrained models,
//! which are licensed for non-commercial research only. Mold does not bundle
//! those weights and refuses to download them until the user has recorded an
//! acceptance, so an automatic or server-side auto-pull can never silently
//! acquire them.
//!
//! The acceptance record is an authorization record, not a model artifact:
//! it is written owner-only (`0o600`) via a temp-file + rename, exactly like
//! the server's catalog credentials. (The "model storage permissions
//! invariant" deliberately does not apply here — it protects runnable model
//! files from group/other mode-bit rejection, never authorization evidence.)
//!
//! An acceptance is bound to the exact license text mold showed. If the
//! pinned `sha256` changes because the upstream terms changed, the old
//! record no longer counts and the user must accept again.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// File name of the acceptance record under `$MOLD_HOME`.
pub const ACCEPTANCE_FILE: &str = "license-acceptances.json";

/// A third-party license the user must accept before mold will download the
/// files it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThirdPartyLicense {
    /// Stable identifier used by `mold pull --accept-license <id>` and by the
    /// on-disk record. Never renamed — a rename invalidates every acceptance.
    pub id: &'static str,
    /// Human-readable name for prompts and error messages.
    pub name: &'static str,
    /// Canonical URL of the license text mold pinned.
    pub url: &'static str,
    /// SHA-256 of the exact license text fetched from `url`. An acceptance
    /// recorded against a different digest does not count.
    pub sha256: &'static str,
    /// One-sentence statement of the restriction the user is accepting.
    pub summary: &'static str,
}

/// InsightFace pretrained models (the antelopev2 pack PuLID conditions on).
///
/// InsightFace splits its terms: the code is MIT with no usage limitation,
/// but "the training data containing the annotation (and the models trained
/// with these data) are available for non-commercial research purposes only",
/// and that applies to both manually downloaded and auto-downloaded models
/// (InsightFace README, "License" section).
///
/// `sha256` is the digest of the README as fetched on 2026-08-21 from the
/// pinned raw URL. A change upstream invalidates existing acceptances so the
/// user is re-shown the terms.
pub const INSIGHTFACE_ANTELOPEV2: ThirdPartyLicense = ThirdPartyLicense {
    id: "insightface-antelopev2",
    name: "InsightFace pretrained models (antelopev2)",
    url: "https://raw.githubusercontent.com/deepinsight/insightface/master/README.md",
    sha256: "84606d9ab37a38606b12c10d96172c6343768d2ef72c802a16482e476f8baf22",
    summary: "InsightFace pretrained models (antelopev2: scrfd_10g_bnkps, glintr100) are licensed for non-commercial research purposes only.",
};

/// Every license mold knows how to gate on.
pub const THIRD_PARTY_LICENSES: &[&ThirdPartyLicense] = &[&INSIGHTFACE_ANTELOPEV2];

/// Resolve a license by its stable id.
pub fn license_by_id(id: &str) -> Option<&'static ThirdPartyLicense> {
    THIRD_PARTY_LICENSES
        .iter()
        .copied()
        .find(|license| license.id == id)
}

/// The license covering one manifest file, if that file is gated on
/// acceptance.
///
/// Keyed on `(manifest name, hf filename)` rather than on the repository so a
/// mirror that also hosts unrestricted files is not over-gated.
pub fn license_for_manifest_file(
    manifest_name: &str,
    hf_filename: &str,
) -> Option<&'static ThirdPartyLicense> {
    if manifest_name == crate::manifest::PULID_FLUX_MANIFEST
        && matches!(hf_filename, "scrfd_10g_bnkps.onnx" | "glintr100.onnx")
    {
        return Some(&INSIGHTFACE_ANTELOPEV2);
    }
    None
}

/// True when any file in `manifest` is gated on an unaccepted license.
pub fn manifest_requires_license(
    manifest: &crate::manifest::ModelManifest,
) -> Option<&'static ThirdPartyLicense> {
    manifest
        .files
        .iter()
        .find_map(|file| license_for_manifest_file(&manifest.name, &file.hf_filename))
}

/// Path of the acceptance record inside a Mold data root.
pub fn acceptance_path(mold_home: &Path) -> PathBuf {
    mold_home.join(ACCEPTANCE_FILE)
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct AcceptanceRecord {
    #[serde(default)]
    acceptances: Vec<Acceptance>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Acceptance {
    id: String,
    url: String,
    sha256: String,
    accepted_at_unix_ms: u64,
}

fn load(mold_home: &Path) -> AcceptanceRecord {
    match fs::read(acceptance_path(mold_home)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => AcceptanceRecord::default(),
    }
}

/// True when `license` has been accepted against its CURRENT pinned text.
///
/// A record whose `sha256` differs from the const does not count: the terms
/// mold would show today are not the terms the user agreed to.
pub fn is_accepted(mold_home: &Path, license: &ThirdPartyLicense) -> bool {
    load(mold_home).acceptances.iter().any(|accepted| {
        accepted.id == license.id
            && accepted.sha256.eq_ignore_ascii_case(license.sha256)
            && accepted.url == license.url
    })
}

/// True when the license with `id` is known and accepted.
pub fn is_accepted_by_id(mold_home: &Path, id: &str) -> bool {
    license_by_id(id).is_some_and(|license| is_accepted(mold_home, license))
}

/// Record the user's acceptance of `license`, replacing any stale record for
/// the same id.
///
/// Written owner-only through a temp file and an atomic rename so a partially
/// written record can never be read as an acceptance.
pub fn record_acceptance(mold_home: &Path, license: &ThirdPartyLicense) -> io::Result<()> {
    let mut record = load(mold_home);
    record.acceptances.retain(|entry| entry.id != license.id);
    record.acceptances.push(Acceptance {
        id: license.id.to_string(),
        url: license.url.to_string(),
        sha256: license.sha256.to_string(),
        accepted_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis() as u64)
            .unwrap_or_default(),
    });
    write_owner_only(&acceptance_path(mold_home), &record)
}

fn write_owner_only(path: &Path, record: &AcceptanceRecord) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("license acceptance path has no parent"))?;
    fs::create_dir_all(parent)?;

    let bytes = serde_json::to_vec_pretty(record).map_err(io::Error::other)?;
    let tmp = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// The actionable message shown when a gated download is refused.
///
/// Names the restriction, the exact text the user would be accepting, and the
/// one command that records acceptance — a refusal that does not say how to
/// proceed is a dead end.
pub fn acceptance_required_message(model: &str, license: &ThirdPartyLicense) -> String {
    format!(
        "{model} includes files under a license that must be accepted before download.\n\n  {}\n  {}\n  Terms: {}\n\nReview the terms, then accept explicitly:\n\n  mold pull {model} --accept-license {}\n",
        license.name, license.summary, license.url, license.id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antelopev2_files_are_gated_and_nothing_else_is() {
        assert_eq!(
            license_for_manifest_file("pulid-flux", "scrfd_10g_bnkps.onnx"),
            Some(&INSIGHTFACE_ANTELOPEV2)
        );
        assert_eq!(
            license_for_manifest_file("pulid-flux", "glintr100.onnx"),
            Some(&INSIGHTFACE_ANTELOPEV2)
        );
        assert_eq!(
            license_for_manifest_file("pulid-flux", "pulid_flux_v0.9.1.safetensors"),
            None
        );
        assert_eq!(
            license_for_manifest_file("pulid-flux", "EVA02_CLIP_L_336_psz14_s6B.pt"),
            None
        );
        // Same filename under a different manifest is not gated.
        assert_eq!(
            license_for_manifest_file("flux-dev:q4", "glintr100.onnx"),
            None
        );
    }

    #[test]
    fn acceptance_round_trips_and_stale_hashes_do_not_count() {
        let home = tempfile::tempdir().unwrap();
        assert!(!is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));

        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        assert!(is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));
        assert!(is_accepted_by_id(home.path(), "insightface-antelopev2"));

        // Terms changed upstream: the stored acceptance no longer applies.
        let revised = ThirdPartyLicense {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..INSIGHTFACE_ANTELOPEV2
        };
        assert!(!is_accepted(home.path(), &revised));
    }

    #[test]
    fn re_accepting_replaces_rather_than_appends() {
        let home = tempfile::tempdir().unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        let record = load(home.path());
        assert_eq!(record.acceptances.len(), 1);
    }

    #[test]
    fn unknown_license_id_is_never_accepted() {
        let home = tempfile::tempdir().unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        assert!(!is_accepted_by_id(home.path(), "not-a-license"));
    }

    #[cfg(unix)]
    #[test]
    fn acceptance_record_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let home = tempfile::tempdir().unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        let mode = fs::metadata(acceptance_path(home.path()))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "acceptance record must be owner-only");
    }

    #[test]
    fn refusal_message_names_the_terms_and_the_command() {
        let message = acceptance_required_message("pulid-flux", &INSIGHTFACE_ANTELOPEV2);
        assert!(message.contains("non-commercial research"));
        assert!(message.contains(INSIGHTFACE_ANTELOPEV2.url));
        assert!(message.contains("mold pull pulid-flux --accept-license insightface-antelopev2"));
    }

    #[test]
    fn corrupt_record_is_not_an_acceptance() {
        let home = tempfile::tempdir().unwrap();
        fs::write(acceptance_path(home.path()), b"{not json").unwrap();
        assert!(!is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));
    }
}
