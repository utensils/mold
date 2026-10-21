//! Private admission records never appear as accepted jobs before input ownership.

use crate::reference_uploads::{ConsumptionReceiptError, InputConsumptionClaim};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

const MAX_PENDING_BYTES: u64 = 24 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RequiredInputClaim {
    pub kind: String,
    pub request_scope_sha256: String,
    pub credential_identity: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PendingAdmission {
    version: u32,
    instance_id: String,
    durable_owner: String,
    pub job: mold_core::video_job::VideoJob,
    pub request_sha256: String,
    pub manifest_json: String,
    pub manifest_sha256: String,
    pub required_claims: Vec<RequiredInputClaim>,
    #[serde(skip)]
    claimed: Vec<String>,
}

impl PendingAdmission {
    pub fn new(
        instance: &str,
        owner: &str,
        row: mold_db::video_jobs::StoredVideoJob,
        required_claims: Vec<RequiredInputClaim>,
    ) -> Self {
        Self {
            version: 1,
            instance_id: instance.into(),
            durable_owner: owner.into(),
            job: row.job,
            request_sha256: row.request_sha256,
            manifest_json: row.manifest_json,
            manifest_sha256: row.manifest_sha256,
            required_claims,
            claimed: Vec::new(),
        }
    }
    pub fn ready(&self) -> bool {
        self.required_claims
            .iter()
            .all(|required| self.claimed.contains(&required.kind))
    }
    pub fn row(&self) -> mold_db::video_jobs::StoredVideoJob {
        mold_db::video_jobs::StoredVideoJob {
            job: self.job.clone(),
            request_sha256: self.request_sha256.clone(),
            manifest_json: self.manifest_json.clone(),
            manifest_sha256: self.manifest_sha256.clone(),
            revision: 0,
            cancel_requested: false,
        }
    }
}

fn claim_name(kind: &str) -> Result<&'static str> {
    match kind {
        "reference_upload" => Ok("admission-reference-claim.json"),
        "gallery_reuse" => Ok("admission-gallery-claim.json"),
        _ => anyhow::bail!("unsupported input ownership kind"),
    }
}

/// A rename may have made the receipt visible even when directory sync fails.
/// The caller must invalidate its token on Published(error), never return it to
/// the pool while this operation can already prove ownership.
fn write_json(
    directory: &Path,
    name: &str,
    value: &impl Serialize,
) -> std::result::Result<(), ConsumptionReceiptError> {
    let before = |error: anyhow::Error| {
        ConsumptionReceiptError::NotPublished(crate::routes::ApiError::internal(error.to_string()))
    };
    let mut temporary =
        tempfile::NamedTempFile::new_in(directory).map_err(|error| before(error.into()))?;
    serde_json::to_writer(&mut temporary, value).map_err(|error| before(error.into()))?;
    temporary.flush().map_err(|error| before(error.into()))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| before(error.into()))?;
    temporary
        .persist(directory.join(name))
        .map_err(|error| before(error.error.into()))?;
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|error| {
            ConsumptionReceiptError::Published(crate::routes::ApiError::internal(error.to_string()))
        })
}

fn read_json<T: serde::de::DeserializeOwned>(
    directory: &Path,
    name: &str,
    maximum: u64,
) -> Result<T> {
    let mut file = mold_core::secure_file::open_regular_file_no_follow(&directory.join(name))?;
    ensure!(
        file.metadata()?.len() <= maximum,
        "admission metadata exceeds its budget"
    );
    let mut bytes = Vec::new();
    (&mut file).take(maximum + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "admission metadata grew beyond its budget"
    );
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn write_pending(directory: &Path, pending: &PendingAdmission) -> Result<()> {
    ensure!(
        pending.required_claims.len() <= 2,
        "too many admission claims"
    );
    for required in &pending.required_claims {
        claim_name(&required.kind)?;
    }
    ensure!(
        serde_json::to_vec(pending)?.len() as u64 <= MAX_PENDING_BYTES,
        "admission metadata exceeds its budget"
    );
    write_json(directory, "admission.json", pending).map_err(|error| match error {
        ConsumptionReceiptError::NotPublished(error)
        | ConsumptionReceiptError::Published(error) => anyhow::anyhow!(error.error),
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundClaim {
    operation_id: String,
    request_sha256: String,
    manifest_sha256: String,
    claim: InputConsumptionClaim,
}

pub(crate) fn record_claim(
    directory: &Path,
    pending: &PendingAdmission,
    claim: &InputConsumptionClaim,
) -> std::result::Result<(), ConsumptionReceiptError> {
    let validate = || -> Result<&str> {
        ensure!(
            claim.instance_id == pending.instance_id,
            "input installation changed"
        );
        ensure!(
            claim.session_sha256.len() == 64
                && claim
                    .session_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "invalid input session digest"
        );
        ensure!(
            pending
                .required_claims
                .iter()
                .any(|required| required.kind == claim.kind
                    && required.request_scope_sha256 == claim.request_scope_sha256
                    && required.credential_identity == claim.credential_identity),
            "input claim does not match the pending operation"
        );
        claim_name(&claim.kind)
    };
    let name = validate().map_err(|error| {
        ConsumptionReceiptError::NotPublished(crate::routes::ApiError::validation(
            error.to_string(),
        ))
    })?;
    let bound = BoundClaim {
        operation_id: pending.job.id.clone(),
        request_sha256: pending.request_sha256.clone(),
        manifest_sha256: pending.manifest_sha256.clone(),
        claim: claim.clone(),
    };
    if directory.join(name).try_exists().map_err(|error| {
        ConsumptionReceiptError::NotPublished(crate::routes::ApiError::internal(error.to_string()))
    })? {
        let previous: BoundClaim = read_json(directory, name, 4096).map_err(|error| {
            ConsumptionReceiptError::NotPublished(crate::routes::ApiError::internal(
                error.to_string(),
            ))
        })?;
        if serde_json::to_vec(&previous).ok() != serde_json::to_vec(&bound).ok() {
            return Err(ConsumptionReceiptError::NotPublished(
                crate::routes::ApiError::validation(
                    "Input ownership is already bound to a different session",
                ),
            ));
        }
    }
    write_json(directory, name, &bound)
}

pub(crate) fn read_pending(
    directory: &Path,
    instance: &str,
    owner: &str,
) -> Result<Option<PendingAdmission>> {
    if !directory.join("admission.json").try_exists()? {
        return Ok(None);
    }
    let mut pending: PendingAdmission = read_json(directory, "admission.json", MAX_PENDING_BYTES)?;
    ensure!(
        pending.version == 1 && pending.instance_id == instance && pending.durable_owner == owner,
        "pending admission owner changed"
    );
    ensure!(
        pending.required_claims.len() <= 2,
        "too many admission claims"
    );
    for required in &pending.required_claims {
        let name = claim_name(&required.kind)?;
        if !directory.join(name).try_exists()? {
            continue;
        }
        let bound: BoundClaim = read_json(directory, name, 4096)?;
        ensure!(
            bound.operation_id == pending.job.id
                && bound.request_sha256 == pending.request_sha256
                && bound.manifest_sha256 == pending.manifest_sha256,
            "input claim operation changed"
        );
        ensure!(
            bound.claim.instance_id == instance
                && bound.claim.kind == required.kind
                && bound.claim.request_scope_sha256 == required.request_scope_sha256
                && bound.claim.credential_identity == required.credential_identity,
            "input claim scope changed"
        );
        pending.claimed.push(required.kind.clone());
    }
    crate::video_job_manifest::read_manifest(&pending.row())
        .context("pending manifest authentication failed")?;
    Ok(Some(pending))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_admission_requires_bound_durable_claims_before_publication() {
        let directory = tempfile::tempdir().unwrap();
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.id = uuid::Uuid::new_v4().to_string();
        let scopes = vec![RequiredInputClaim {
            kind: "reference_upload".into(),
            request_scope_sha256: "a".repeat(64),
            credential_identity: "process-key".into(),
        }];
        let pending = PendingAdmission::new("instance", "durable-key", row, scopes);
        write_pending(directory.path(), &pending).unwrap();
        assert!(!read_pending(directory.path(), "instance", "durable-key")
            .unwrap()
            .unwrap()
            .ready());
        let mut claim = crate::reference_uploads::InputConsumptionClaim {
            kind: "reference_upload".into(),
            session_sha256: "b".repeat(64),
            request_scope_sha256: "a".repeat(64),
            credential_identity: "wrong-key".into(),
            instance_id: "instance".into(),
        };
        assert!(record_claim(directory.path(), &pending, &claim).is_err());
        claim.credential_identity = "process-key".into();
        assert!(record_claim(directory.path(), &pending, &claim).is_ok());
        claim.session_sha256 = "c".repeat(64);
        assert!(record_claim(directory.path(), &pending, &claim).is_err());
        assert!(read_pending(directory.path(), "instance", "durable-key")
            .unwrap()
            .unwrap()
            .ready());
        assert!(read_pending(directory.path(), "other-instance", "durable-key").is_err());
        assert!(read_pending(directory.path(), "instance", "other-owner").is_err());
    }
}
