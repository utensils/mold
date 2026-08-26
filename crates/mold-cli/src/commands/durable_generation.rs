//! CLI adapters for the shared durable-generation state machine.

use anyhow::{Context, Result};
use mold_core::{GenerateRequest, GenerationBatchAuthority, GenerationBatchChild, MoldClient};

pub(crate) use mold_core::durable_generation::{
    CanonicalGenerationEvent, CanonicalGenerationReport, CanonicalRetrySubmission,
};

#[derive(Debug)]
pub(crate) struct CanonicalGenerationArtifact {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub request: GenerateRequest,
    pub metadata: mold_core::OutputMetadata,
}

#[derive(Debug)]
pub(crate) struct CanonicalHydratedArtifact {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub metadata: mold_core::OutputMetadata,
}

fn channel_observer(
    events: &tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>,
) -> impl Fn(CanonicalGenerationEvent) + Send + Sync + '_ {
    |event| {
        let _ = events.send(event);
    }
}

pub(crate) async fn try_canonical_generation(
    client: &MoldClient,
    requests: &[GenerateRequest],
) -> Result<Option<CanonicalGenerationReport>> {
    mold_core::durable_generation::try_canonical_generation(client, requests).await
}

pub(crate) async fn try_canonical_generation_observed(
    client: &MoldClient,
    requests: &[GenerateRequest],
    events: Option<&tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>>,
) -> Result<Option<CanonicalGenerationReport>> {
    match events {
        Some(events) => {
            let observer = channel_observer(events);
            mold_core::durable_generation::try_canonical_generation_observed(
                client,
                requests,
                Some(&observer),
            )
            .await
        }
        None => mold_core::durable_generation::try_canonical_generation(client, requests).await,
    }
}

pub(crate) async fn retry_canonical_child(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    job_id: &str,
) -> Result<CanonicalRetrySubmission> {
    mold_core::durable_generation::retry_canonical_child(client, authority, job_id).await
}

pub(crate) async fn hydrate_canonical_artifact(
    client: &MoldClient,
    child: &GenerationBatchChild,
) -> Result<CanonicalHydratedArtifact> {
    let job_id = child.job_id.clone();
    let result = child
        .result
        .clone()
        .context("canonical generation completed without a gallery result")?;
    let filename = result
        .filename
        .or(result.original_filename)
        .context("canonical generation completed without a gallery filename")?;
    let bytes = client
        .get_gallery_image(&filename)
        .await
        .with_context(|| format!("could not hydrate accepted output {filename}"))?;
    let metadata = client
        .list_gallery()
        .await
        .with_context(|| format!("could not read metadata for accepted output {filename}"))?
        .into_iter()
        .find(|item| item.filename == filename)
        .with_context(|| format!("accepted output {filename} is missing from the gallery index"))?
        .metadata;
    if metadata.job_id.as_deref() != Some(job_id.as_str()) {
        anyhow::bail!("accepted output {filename} does not belong to durable job {job_id}");
    }
    Ok(CanonicalHydratedArtifact {
        bytes,
        filename,
        metadata,
    })
}

pub(crate) async fn try_canonical_singleton_artifact(
    client: &MoldClient,
    request: &GenerateRequest,
) -> Result<Option<CanonicalGenerationArtifact>> {
    let Some(report) = try_canonical_generation(client, std::slice::from_ref(request)).await?
    else {
        return Ok(None);
    };
    if !report.failures.is_empty() {
        let mut failures = report.failures;
        if !report.admitted_client_ids.is_empty() {
            failures.push(format!(
                "accepted client ids: {}",
                report.admitted_client_ids.join(", ")
            ));
        }
        anyhow::bail!(failures.join("; "));
    }
    let outcome = report
        .outcomes
        .into_iter()
        .find(|outcome| outcome.child.state == mold_core::GenerationBatchChildState::Complete)
        .context("canonical singleton completed without a successful child")?;
    if outcome.authority.client_batch_id != outcome.client_batch_id {
        anyhow::bail!("canonical singleton outcome lost its admission authority");
    }
    let artifact = hydrate_canonical_artifact(client, &outcome.child).await?;
    Ok(Some(CanonicalGenerationArtifact {
        bytes: artifact.bytes,
        filename: artifact.filename,
        request: outcome.request,
        metadata: artifact.metadata,
    }))
}

#[cfg(test)]
pub(crate) async fn admit_for_test(
    client: &MoldClient,
    request: &mold_core::GenerationBatchAdmissionRequest,
) -> Result<mold_core::GenerationBatchStatus> {
    mold_core::durable_generation::admit_canonical_batch(client, request).await
}
