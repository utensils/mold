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
    /// **Immutable** URL of the exact license text mold pinned — a
    /// content-addressed upstream revision, never a moving branch ref.
    ///
    /// This is load-bearing, not cosmetic. A `.../master/README.md` URL is
    /// mutable: upstream can rewrite the terms, and mold would go on
    /// recording a hard-coded digest of text nobody can fetch any more while
    /// showing the user a link whose contents no longer match. Old
    /// acceptances would keep passing and the "changed terms require
    /// re-acceptance" promise would be hollow. Pinning the revision makes
    /// `(url, sha256)` a closed pair: the bytes at this URL cannot change, so
    /// the digest cannot go stale, and adopting new terms is necessarily a
    /// visible edit to this constant.
    pub url: &'static str,
    /// SHA-256 of the exact license text served at [`Self::url`]. Verified at
    /// pin time; an acceptance recorded against a different digest does not
    /// count.
    pub sha256: &'static str,
    /// Human-readable landing page for the project's current terms.
    ///
    /// Shown alongside [`Self::url`] so a user can read today's terms in
    /// context, but deliberately NOT part of the accepted identity — its
    /// contents move, which is exactly what `url` must not do.
    pub canonical: &'static str,
    /// One-sentence statement of the restriction the user is accepting.
    pub summary: &'static str,
}

impl ThirdPartyLicense {
    /// The identity an acceptance is bound to: the immutable text location
    /// and its digest.
    ///
    /// Re-pinning either half — a new upstream revision, or a corrected
    /// digest — invalidates every stored acceptance, which is the whole point
    /// of binding both.
    pub fn accepted_identity(&self) -> (&'static str, &'static str) {
        (self.url, self.sha256)
    }
}

/// InsightFace pretrained models (the antelopev2 pack PuLID conditions on).
///
/// InsightFace splits its terms: the code is MIT with no usage limitation,
/// but "the training data containing the annotation (and the models trained
/// with these data) are available for non-commercial research purposes only",
/// and that applies to both manually downloaded and auto-downloaded models
/// (InsightFace README, "License" section).
///
/// The terms are pinned to upstream commit
/// `7fadd420c2351d0ffa8cac403421c1a3ed733365`, whose `README.md` was fetched
/// and verified on 2026-08-21 to hash to `sha256` below. GitHub serves a
/// commit-addressed raw URL immutably, so that digest can never go stale
/// against its own URL — re-pinning to a later revision is a deliberate edit
/// here, and it invalidates every recorded acceptance so users are re-shown
/// whatever the new text says.
pub const INSIGHTFACE_ANTELOPEV2: ThirdPartyLicense = ThirdPartyLicense {
    id: "insightface-antelopev2",
    name: "InsightFace pretrained models (antelopev2)",
    url: "https://raw.githubusercontent.com/deepinsight/insightface/7fadd420c2351d0ffa8cac403421c1a3ed733365/README.md",
    sha256: "84606d9ab37a38606b12c10d96172c6343768d2ef72c802a16482e476f8baf22",
    canonical: "https://github.com/deepinsight/insightface#license",
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

/// Manifest names that cannot be downloaded until `license` is accepted.
///
/// Derived from the manifest registry rather than hard-coded, so adding a
/// gated file to a manifest cannot leave `GET /api/licenses` telling users the
/// wrong thing about what they are unblocking.
pub fn manifests_requiring(license: &ThirdPartyLicense) -> Vec<String> {
    crate::manifest::known_manifests()
        .iter()
        .filter(|manifest| {
            manifest.files.iter().any(|file| {
                license_for_manifest_file(&manifest.name, &file.hf_filename)
                    .is_some_and(|found| found.id == license.id)
            })
        })
        .map(|manifest| manifest.name.clone())
        .collect()
}

/// Every known license plus this root's acceptance state, for `GET /api/licenses`.
pub fn license_statuses(mold_home: &Path) -> Vec<crate::types::ThirdPartyLicenseStatus> {
    THIRD_PARTY_LICENSES
        .iter()
        .map(|license| crate::types::ThirdPartyLicenseStatus {
            id: license.id.to_string(),
            name: license.name.to_string(),
            url: license.url.to_string(),
            canonical: license.canonical.to_string(),
            sha256: license.sha256.to_string(),
            summary: license.summary.to_string(),
            accepted: is_accepted(mold_home, license),
            required_by: manifests_requiring(license),
        })
        .collect()
}

/// The machine-readable refusal payload for `license`.
pub fn refusal(license: &ThirdPartyLicense) -> crate::types::LicenseRefusal {
    crate::types::LicenseRefusal {
        id: license.id.to_string(),
        name: license.name.to_string(),
        url: license.url.to_string(),
        canonical: license.canonical.to_string(),
        sha256: license.sha256.to_string(),
        summary: license.summary.to_string(),
    }
}

/// An `accept_licenses` entry naming a license this build does not know.
///
/// Rejected rather than ignored: silently dropping an unrecognised id would
/// let a client believe it had accepted something, then fail the download with
/// a refusal that looks like a server bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLicenseId(pub String);

impl std::fmt::Display for UnknownLicenseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let known = THIRD_PARTY_LICENSES
            .iter()
            .map(|license| license.id)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "unknown license id '{}'. Known licenses: {known}",
            self.0
        )
    }
}

impl std::error::Error for UnknownLicenseId {}

/// Resolve and VERIFY every acceptance, then record them all.
///
/// Two things are checked before anything is written, for the whole list:
/// that the id is known, and that the `(url, sha256)` the caller displayed is
/// the document this build pins. The second check is the point of the whole
/// struct — the caller may be a client on a different Mold release, and
/// recording its consent against terms of OUR choosing would store agreement
/// to text the user never read.
///
/// Nothing is written until every entry passes, so a rejected request leaves a
/// root the caller can describe as untouched.
pub fn record_acceptances(
    mold_home: &Path,
    acceptances: &[crate::types::LicenseAcceptance],
) -> Result<Vec<&'static ThirdPartyLicense>, RecordAcceptancesError> {
    let mut resolved = Vec::with_capacity(acceptances.len());
    for acceptance in acceptances {
        let license = license_by_id(&acceptance.id).ok_or_else(|| {
            RecordAcceptancesError::Unknown(UnknownLicenseId(acceptance.id.clone()))
        })?;
        if !acceptance.matches(license.url, license.sha256) {
            return Err(RecordAcceptancesError::TermsMismatch(license));
        }
        resolved.push(license);
    }
    for license in &resolved {
        record_acceptance(mold_home, license).map_err(RecordAcceptancesError::Io)?;
    }
    Ok(resolved)
}

/// Failure modes of [`record_acceptances`].
#[derive(Debug)]
pub enum RecordAcceptancesError {
    /// The request named a license this build does not know — a client error.
    Unknown(UnknownLicenseId),
    /// The caller accepted a DIFFERENT revision of a license this build knows.
    /// Carries our pinned license so the refusal can show the caller what it
    /// would have to accept instead.
    TermsMismatch(&'static ThirdPartyLicense),
    /// The record could not be written — a server error.
    Io(io::Error),
}

impl std::fmt::Display for RecordAcceptancesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(error) => write!(f, "{error}"),
            Self::TermsMismatch(license) => write!(
                f,
                "the accepted terms for '{}' are not the ones this server pins. \
                 This server pins {} (sha256 {}). Review those terms and accept again.",
                license.id, license.url, license.sha256
            ),
            Self::Io(error) => write!(f, "failed to record license acceptance: {error}"),
        }
    }
}

impl std::error::Error for RecordAcceptancesError {}

/// The acceptance payload for `license` as this build pins it.
///
/// Used by the CLI's older-server fallback, and by tests.
pub fn acceptance_for(license: &ThirdPartyLicense) -> crate::types::LicenseAcceptance {
    crate::types::LicenseAcceptance {
        id: license.id.to_string(),
        url: license.url.to_string(),
        sha256: license.sha256.to_string(),
    }
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
/// The record must match the license's whole [`ThirdPartyLicense::accepted_identity`]
/// — BOTH the immutable text URL and the digest — not just the id. Either
/// half moving means the terms mold would show today are not the terms the
/// user agreed to, so a Mold release that re-pins the license to a newer
/// upstream revision asks every user again rather than silently inheriting
/// consent given for different text. Checking only the digest would let a
/// re-pin whose text happens to be byte-identical slip through, and checking
/// only the URL would miss a corrected digest.
pub fn is_accepted(mold_home: &Path, license: &ThirdPartyLicense) -> bool {
    let (url, sha256) = license.accepted_identity();
    load(mold_home).acceptances.iter().any(|accepted| {
        accepted.id == license.id
            && accepted.sha256.eq_ignore_ascii_case(sha256)
            && accepted.url == url
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
    let (url, sha256) = license.accepted_identity();
    let mut record = load(mold_home);
    record.acceptances.retain(|entry| entry.id != license.id);
    record.acceptances.push(Acceptance {
        id: license.id.to_string(),
        url: url.to_string(),
        sha256: sha256.to_string(),
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
        "{model} includes files under a license that must be accepted before download.\n\n  {}\n  {}\n  Terms (pinned): {}\n  Project terms: {}\n\nReview the terms, then accept explicitly:\n\n  mold pull {model} --accept-license {}\n",
        license.name, license.summary, license.url, license.canonical, license.id
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

    /// The other half of the binding: a Mold release that re-pins the license
    /// to a NEWER upstream revision must ask again, even in the degenerate
    /// case where the new revision's text hashes the same. Consent was given
    /// for one exact document, identified by where it lives as well as what
    /// it says.
    #[test]
    fn re_pinning_to_a_new_revision_invalidates_the_acceptance() {
        let home = tempfile::tempdir().unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        assert!(is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));

        let repinned = ThirdPartyLicense {
            url: "https://raw.githubusercontent.com/deepinsight/insightface/0000000000000000000000000000000000000000/README.md",
            ..INSIGHTFACE_ANTELOPEV2
        };
        assert!(
            !is_accepted(home.path(), &repinned),
            "a new pinned revision must require a fresh acceptance"
        );

        // And accepting the re-pin replaces rather than accumulates, so the
        // superseded revision does not linger as a second standing consent.
        record_acceptance(home.path(), &repinned).unwrap();
        assert!(is_accepted(home.path(), &repinned));
        assert!(!is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));
        assert_eq!(load(home.path()).acceptances.len(), 1);
    }

    /// The guarantee this whole design rests on: the accepted URL must be
    /// content-addressed. A branch ref such as `.../master/README.md` can be
    /// rewritten upstream, which would leave the pinned digest describing
    /// text nobody can fetch while old acceptances kept passing.
    #[test]
    fn every_pinned_license_url_is_immutable() {
        for license in THIRD_PARTY_LICENSES {
            let url = license.url;
            assert!(
                !url.contains("/master/") && !url.contains("/main/") && !url.contains("/HEAD/"),
                "{} pins a mutable branch ref: {url}",
                license.id
            );
            // GitHub raw URLs are immutable only when the ref is a full
            // 40-hex commit SHA.
            let looks_commit_pinned = url.split('/').any(|segment| {
                segment.len() == 40 && segment.chars().all(|c| c.is_ascii_hexdigit())
            });
            assert!(
                looks_commit_pinned,
                "{} must pin an exact upstream commit: {url}",
                license.id
            );
            assert_eq!(
                license.sha256.len(),
                64,
                "{} must pin a full sha256",
                license.id
            );
            assert_ne!(
                license.url, license.canonical,
                "{} must keep the browsable page separate from the accepted identity",
                license.id
            );
        }
    }

    #[test]
    fn re_accepting_replaces_rather_than_appends() {
        let home = tempfile::tempdir().unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        record_acceptance(home.path(), &INSIGHTFACE_ANTELOPEV2).unwrap();
        let record = load(home.path());
        assert_eq!(record.acceptances.len(), 1);
    }

    /// The consent-integrity rule: a caller may only have its acceptance
    /// recorded against the document it actually displayed.
    #[test]
    fn recording_requires_the_terms_the_caller_displayed() {
        let home = tempfile::tempdir().unwrap();
        let ours = &INSIGHTFACE_ANTELOPEV2;

        let honest = [acceptance_for(ours)];
        record_acceptances(home.path(), &honest).unwrap();
        assert!(is_accepted(home.path(), ours));

        // A caller on a different release, resolving the same id to its own
        // pinned revision.
        let fresh = tempfile::tempdir().unwrap();
        let other_revision = crate::types::LicenseAcceptance {
            id: ours.id.to_string(),
            url: "https://raw.githubusercontent.com/deepinsight/insightface/0000000000000000000000000000000000000000/README.md".to_string(),
            sha256: ours.sha256.to_string(),
        };
        let error = record_acceptances(fresh.path(), &[other_revision]).unwrap_err();
        assert!(matches!(
            error,
            RecordAcceptancesError::TermsMismatch(license) if license.id == ours.id
        ));
        assert!(
            !is_accepted(fresh.path(), ours),
            "a mismatched acceptance must record nothing"
        );

        // Same URL, different digest, is equally a mismatch — the URL alone
        // is not the identity.
        let wrong_digest = crate::types::LicenseAcceptance {
            id: ours.id.to_string(),
            url: ours.url.to_string(),
            sha256: "0".repeat(64),
        };
        assert!(matches!(
            record_acceptances(fresh.path(), &[wrong_digest]).unwrap_err(),
            RecordAcceptancesError::TermsMismatch(_)
        ));
        assert!(!is_accepted(fresh.path(), ours));
    }

    #[test]
    fn digest_casing_is_not_part_of_the_identity() {
        let home = tempfile::tempdir().unwrap();
        let ours = &INSIGHTFACE_ANTELOPEV2;
        let upper = crate::types::LicenseAcceptance {
            id: ours.id.to_string(),
            url: ours.url.to_string(),
            sha256: ours.sha256.to_ascii_uppercase(),
        };
        record_acceptances(home.path(), &[upper]).unwrap();
        assert!(is_accepted(home.path(), ours));
    }

    /// A rejected entry must not leave earlier entries in the same request
    /// applied — the refusal describes a root that was not touched.
    #[test]
    fn a_mismatch_late_in_the_list_writes_nothing_at_all() {
        let home = tempfile::tempdir().unwrap();
        let ours = &INSIGHTFACE_ANTELOPEV2;
        let bad = crate::types::LicenseAcceptance {
            id: ours.id.to_string(),
            url: ours.url.to_string(),
            sha256: "0".repeat(64),
        };
        let error = record_acceptances(home.path(), &[acceptance_for(ours), bad]).unwrap_err();
        assert!(matches!(error, RecordAcceptancesError::TermsMismatch(_)));
        assert!(!is_accepted(home.path(), ours));
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
        assert!(message.contains(INSIGHTFACE_ANTELOPEV2.canonical));
        assert!(message.contains("mold pull pulid-flux --accept-license insightface-antelopev2"));
    }

    #[test]
    fn corrupt_record_is_not_an_acceptance() {
        let home = tempfile::tempdir().unwrap();
        fs::write(acceptance_path(home.path()), b"{not json").unwrap();
        assert!(!is_accepted(home.path(), &INSIGHTFACE_ANTELOPEV2));
    }
}
