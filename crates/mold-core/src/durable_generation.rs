//! Canonical protocol-v2 generation admission and reconciliation.
//!
//! This module owns transport-independent orchestration. Callers adapt its
//! events into their own progress UI and hydrate completed gallery artifacts
//! only after the durable state machine has settled.

use crate::{
    client::{is_missing_endpoint_error, is_transient_request_error},
    GenerateRequest, GenerationBatchAdmissionRequest, GenerationBatchAuthority,
    GenerationBatchChild, GenerationBatchChildState, GenerationBatchStatus, GenerationRetryRequest,
    MoldClient,
};
use anyhow::{Context, Result};
use std::time::Duration;

#[derive(Debug)]
pub struct CanonicalGenerationOutcome {
    pub authority: GenerationBatchAuthority,
    pub client_batch_id: String,
    /// Zero-based position in the original request slice, independent of
    /// transport chunking.
    pub request_offset: u32,
    pub request: GenerateRequest,
    pub child: GenerationBatchChild,
}

#[derive(Debug, Default)]
pub struct CanonicalGenerationReport {
    pub authorities: Vec<GenerationBatchAuthority>,
    pub admitted_client_ids: Vec<String>,
    pub outcomes: Vec<CanonicalGenerationOutcome>,
    /// Admission, identity, or reconciliation failures that are not already
    /// represented by a terminal child outcome.
    pub orchestration_failures: Vec<String>,
    /// Complete user-facing failure set, including terminal child outcomes.
    pub failures: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum CanonicalGenerationEvent {
    Admitted {
        authority: GenerationBatchAuthority,
        status: GenerationBatchStatus,
        request_offset: u32,
    },
    Snapshot {
        authority: GenerationBatchAuthority,
        status: GenerationBatchStatus,
        request_offset: u32,
    },
    ReconcileDelayed {
        authority: GenerationBatchAuthority,
        error: String,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum CanonicalRetrySubmission {
    Accepted,
    Ambiguous {
        error: String,
        /// The child's `revision` as the caller last observed it, captured
        /// BEFORE the POST that was lost. Reconciliation compares against
        /// this to tell "my retry landed and the child re-held" from "my
        /// retry never arrived" — two facts a `held` state alone conflates.
        /// `0` means the caller had no revision authority (an older server),
        /// and reconciliation degrades to the state-only rule.
        observed_revision: u64,
    },
}

pub type CanonicalGenerationObserver<'a> = dyn Fn(CanonicalGenerationEvent) + Send + Sync + 'a;

fn new_client_batch_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

async fn admit_recovering_ambiguity(
    client: &MoldClient,
    request: &GenerationBatchAdmissionRequest,
) -> Result<GenerationBatchStatus> {
    match client.admit_generation_batch(request).await {
        Ok(status) => Ok(status),
        Err(error) if is_missing_endpoint_error(&error) => Err(error).context(
            "this host does not serve POST /api/generation-batches, which is the only \
             generation admission path",
        ),
        Err(error) if is_transient_request_error(&error) => {
            const LOOKUP_ATTEMPTS: u32 = 5;
            let mut last_lookup_error = None;
            let mut saw_missing = false;
            for attempt in 0..LOOKUP_ATTEMPTS {
                match client
                    .generation_batch_by_client_id(&request.client_batch_id)
                    .await
                {
                    Ok(Some(status)) => return Ok(status),
                    Ok(None) => saw_missing = true,
                    Err(lookup) => last_lookup_error = Some(lookup),
                }
                if attempt + 1 < LOOKUP_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(250 * u64::from(attempt + 1))).await;
                }
            }
            Err(error.context(format!(
                "generation-batch admission is uncertain for client id {}; no durable admission became visible after {LOOKUP_ATTEMPTS} attempts{}{}",
                request.client_batch_id,
                if saw_missing { "; the idempotency key was not found" } else { "" },
                last_lookup_error
                    .map(|lookup| format!("; last lookup error: {lookup}"))
                    .unwrap_or_default()
            )))
        }
        Err(error) => Err(error),
    }
}

fn child_is_settled(state: &GenerationBatchChildState) -> bool {
    matches!(
        state,
        GenerationBatchChildState::Complete
            | GenerationBatchChildState::Failed
            | GenerationBatchChildState::Cancelled
            | GenerationBatchChildState::Held
    )
}

fn observe(observer: Option<&CanonicalGenerationObserver<'_>>, event: CanonicalGenerationEvent) {
    if let Some(observer) = observer {
        observer(event);
    }
}

async fn wait_for_batch(
    client: &MoldClient,
    mut status: GenerationBatchStatus,
    authority: &GenerationBatchAuthority,
    request_offset: u32,
    observer: Option<&CanonicalGenerationObserver<'_>>,
) -> Result<GenerationBatchStatus> {
    authority
        .validate_status(&status)
        .map_err(anyhow::Error::msg)?;
    loop {
        if status
            .children
            .iter()
            .all(|child| child_is_settled(&child.state))
        {
            return Ok(status);
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        // The OUTER settle loop is deliberately unbounded — a queued job may
        // legitimately wait hours. Only this transient-error retry is bounded,
        // so a host that has genuinely gone away reports it instead of spinning
        // in silence forever.
        let mut transient_attempts = 0_u32;
        let next = loop {
            match client.generation_batch(&status.id).await {
                Ok(Some(next)) => break next,
                Ok(None) => anyhow::bail!("generation batch {} disappeared", status.id),
                Err(error) if is_transient_request_error(&error) => {
                    transient_attempts += 1;
                    observe(
                        observer,
                        CanonicalGenerationEvent::ReconcileDelayed {
                            authority: authority.clone(),
                            error: error.to_string(),
                        },
                    );
                    if transient_attempts >= RECONCILE_TRANSIENT_ATTEMPTS {
                        return Err(error).context(format!(
                            "generation batch {} could not be reconciled after {RECONCILE_TRANSIENT_ATTEMPTS} transient failures",
                            status.id
                        ));
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                Err(error) => return Err(error),
            }
        };
        authority
            .validate_status(&next)
            .map_err(anyhow::Error::msg)?;
        observe(
            observer,
            CanonicalGenerationEvent::Snapshot {
                authority: authority.clone(),
                status: next.clone(),
                request_offset,
            },
        );
        status = next;
    }
}

/// Submit one retry, carrying the revision the caller last saw for the child
/// so an interrupted response can still be reconciled exactly.
pub async fn retry_canonical_child(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    job_id: &str,
    observed_revision: u64,
) -> Result<CanonicalRetrySubmission> {
    match client
        .retry_queue_job(&GenerationRetryRequest::from_authority(authority, job_id))
        .await
    {
        Ok(()) => Ok(CanonicalRetrySubmission::Accepted),
        Err(error) if is_transient_request_error(&error) => {
            Ok(CanonicalRetrySubmission::Ambiguous {
                error: error.to_string(),
                observed_revision,
            })
        }
        Err(error) => Err(error),
    }
}

async fn read_canonical_authority(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    observer: Option<&CanonicalGenerationObserver<'_>>,
) -> Result<GenerationBatchStatus> {
    let mut attempts = 0_u32;
    let status = loop {
        attempts += 1;
        let exhausted = attempts >= RECONCILE_TRANSIENT_ATTEMPTS;
        match client.generation_batch(&authority.batch_id).await {
            Ok(Some(status)) => break status,
            Ok(None) => {
                let error = format!("generation batch {} is not visible yet", authority.batch_id);
                observe(
                    observer,
                    CanonicalGenerationEvent::ReconcileDelayed {
                        authority: authority.clone(),
                        error: error.clone(),
                    },
                );
                if exhausted {
                    anyhow::bail!(
                        "generation batch {} did not become visible after {RECONCILE_TRANSIENT_ATTEMPTS} attempts: {error}",
                        authority.batch_id
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(error) if is_transient_request_error(&error) => {
                observe(
                    observer,
                    CanonicalGenerationEvent::ReconcileDelayed {
                        authority: authority.clone(),
                        error: error.to_string(),
                    },
                );
                if exhausted {
                    return Err(error).context(format!(
                        "generation batch {} could not be read after {RECONCILE_TRANSIENT_ATTEMPTS} transient failures",
                        authority.batch_id
                    ));
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(error) => return Err(error),
        }
    };
    authority
        .validate_status(&status)
        .map_err(anyhow::Error::msg)?;
    Ok(status)
}

pub async fn reconcile_canonical_authority_observed(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    observer: Option<&CanonicalGenerationObserver<'_>>,
) -> Result<GenerationBatchStatus> {
    let status = read_canonical_authority(client, authority, observer).await?;
    observe(
        observer,
        CanonicalGenerationEvent::Snapshot {
            authority: authority.clone(),
            status: status.clone(),
            request_offset: 0,
        },
    );
    wait_for_batch(client, status, authority, 0, observer).await
}

/// Bound on transient-failure retries inside one reconciliation read. Sibling
/// of [`AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS`]; the outer settle loop stays
/// unbounded on purpose, because queued work may legitimately wait hours.
const RECONCILE_TRANSIENT_ATTEMPTS: u32 = 5;

const AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS: usize = 5;

/// Reconcile a retry whose POST response was lost without publishing the old
/// Held snapshot as fresh authority.
///
/// `observed_revision` is the child's `revision` from before the lost POST.
/// A child whose revision has advanced past it was retried — even if it is
/// held again, which is a legitimate outcome the state alone cannot express.
/// A child still sitting at that exact revision when the bounded attempts run
/// out was NOT retried, and this returns an error with the retry fence still
/// held rather than republishing the pre-retry snapshot as fresh authority.
///
/// `observed_revision == 0` means the caller had no revision authority (a
/// server predating the column). The bounded loop then keeps its original
/// behaviour and publishes on exhaustion, because with no version to compare
/// there is no evidence either way and hanging forever is worse.
pub async fn reconcile_ambiguous_retry_observed(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    job_id: &str,
    observed_revision: u64,
    observer: Option<&CanonicalGenerationObserver<'_>>,
) -> Result<GenerationBatchStatus> {
    for attempt in 0..AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
        let status = match client.generation_batch(&authority.batch_id).await {
            Ok(Some(status)) => {
                authority
                    .validate_status(&status)
                    .map_err(anyhow::Error::msg)?;
                status
            }
            Ok(None) => {
                let error = format!(
                    "retry outcome is not confirmed; generation batch {} is not visible",
                    authority.batch_id
                );
                observe(
                    observer,
                    CanonicalGenerationEvent::ReconcileDelayed {
                        authority: authority.clone(),
                        error: error.clone(),
                    },
                );
                if attempt + 1 == AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
                    anyhow::bail!(
                        "ambiguous retry remains unconfirmed after {} bounded attempts; retry lock retained: {error}",
                        AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            Err(error) if is_transient_request_error(&error) => {
                let error = error.to_string();
                observe(
                    observer,
                    CanonicalGenerationEvent::ReconcileDelayed {
                        authority: authority.clone(),
                        error: error.clone(),
                    },
                );
                if attempt + 1 == AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS {
                    anyhow::bail!(
                        "ambiguous retry remains unconfirmed after {} bounded attempts; retry lock retained: {error}",
                        AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            Err(error) => return Err(error),
        };
        let child = status
            .children
            .iter()
            .find(|child| child.job_id == job_id)
            .with_context(|| format!("generation batch lost durable job {job_id} after retry"))?;
        // A revision past the pre-POST one proves the retry landed, whatever
        // state the child is in now: a re-held child at a higher revision was
        // retried and held again for a fresh reason.
        let advanced = observed_revision > 0 && child.revision > observed_revision;
        let last_attempt = attempt + 1 == AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS;
        if advanced || child.state != GenerationBatchChildState::Held {
            observe(
                observer,
                CanonicalGenerationEvent::Snapshot {
                    authority: authority.clone(),
                    status: status.clone(),
                    request_offset: 0,
                },
            );
            return wait_for_batch(client, status, authority, 0, observer).await;
        }
        if last_attempt {
            // Still held at exactly the revision we submitted against: the
            // retry did not land. Releasing the fence here would republish a
            // pre-retry snapshot as fresh authority while the original POST
            // may yet commit, so refuse and keep the fence.
            if observed_revision > 0 {
                anyhow::bail!(
                    "ambiguous retry did not reach durable job {job_id}: it remains held at revision {} after {} bounded attempts; retry lock retained",
                    child.revision,
                    AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
                );
            }
            // No revision authority to compare against (older server). Keep
            // the original behaviour rather than hanging on no evidence.
            observe(
                observer,
                CanonicalGenerationEvent::Snapshot {
                    authority: authority.clone(),
                    status: status.clone(),
                    request_offset: 0,
                },
            );
            return wait_for_batch(client, status, authority, 0, observer).await;
        }
        observe(
            observer,
            CanonicalGenerationEvent::ReconcileDelayed {
                authority: authority.clone(),
                error: format!("retry outcome is not confirmed; durable job {job_id} remains held"),
            },
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    unreachable!("ambiguous retry confirmation loop always returns")
}

/// The user-facing failure sentence for a child that did not complete, or
/// `None` for a completed one. Exposed so a client that resumed held children
/// can re-derive its failure set from the settled children.
pub fn child_failure(client_batch_id: &str, child: &GenerationBatchChild) -> Option<String> {
    terminal_failure(client_batch_id, child)
}

/// A child the host parked because the MODEL ITSELF is absent — the one hold a
/// client can repair by pulling and retrying. Read off the typed `error_code`,
/// never the sentence.
pub fn is_missing_model_hold(child: &GenerationBatchChild) -> bool {
    child.state == GenerationBatchChildState::Held
        && matches!(
            child.error_code.as_deref(),
            Some(crate::SSE_ERROR_CODE_MODEL_NOT_FOUND) | Some(crate::SSE_ERROR_CODE_UNKNOWN_MODEL)
        )
}

/// Read an admitted batch afresh and wait for every child to settle — the
/// re-wait a client performs after retrying held children.
pub async fn wait_for_settled_batch(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
) -> Result<GenerationBatchStatus> {
    let status = client
        .generation_batch(&authority.batch_id)
        .await?
        .with_context(|| format!("generation batch {} disappeared", authority.batch_id))?;
    wait_for_batch(client, status, authority, 0, None).await
}

fn terminal_failure(client_batch_id: &str, child: &GenerationBatchChild) -> Option<String> {
    if child.state == GenerationBatchChildState::Complete {
        return None;
    }
    let detail = child.error.as_deref().unwrap_or(match child.state {
        GenerationBatchChildState::Held => "dependency preparation stopped",
        GenerationBatchChildState::Cancelled => "cancelled without server detail",
        _ => "failed without server detail",
    });
    let retry = if child.state == GenerationBatchChildState::Held && child.retryable == Some(true) {
        format!(
            "; durable job {} is retryable after correcting the cause",
            child.job_id
        )
    } else {
        String::new()
    };
    Some(format!(
        "generation batch child {} for client id {client_batch_id} is {}: {detail}{retry}",
        child.index,
        match child.state {
            GenerationBatchChildState::Held => "held",
            GenerationBatchChildState::Cancelled => "cancelled",
            _ => "failed",
        }
    ))
}

fn push_orchestration_failure(report: &mut CanonicalGenerationReport, error: String) {
    report.orchestration_failures.push(error.clone());
    report.failures.push(error);
}

fn validate_terminal_children(
    client_batch_id: &str,
    children: &[GenerationBatchChild],
    expected: usize,
) -> Result<(), String> {
    if children.len() != expected {
        return Err(format!(
            "accepted client id {client_batch_id} returned {} children, expected {expected}",
            children.len()
        ));
    }
    let mut seen = vec![false; expected];
    for child in children {
        let Some(offset) = child
            .index
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < expected)
        else {
            return Err(format!(
                "accepted client id {client_batch_id} returned invalid child index {}",
                child.index
            ));
        };
        if std::mem::replace(&mut seen[offset], true) {
            return Err(format!(
                "accepted client id {client_batch_id} duplicated child index {}",
                child.index
            ));
        }
    }
    Ok(())
}

pub async fn canonical_generation(
    client: &MoldClient,
    requests: &[GenerateRequest],
) -> Result<CanonicalGenerationReport> {
    canonical_generation_observed(client, requests, None).await
}

/// Run the canonical admission, chunking, ambiguity recovery and durable
/// reconciliation state machine.
///
/// This is the only generation path. A host that does not serve it, or that
/// cannot represent one of these requests, is an error naming what it lacks —
/// never a silent downgrade to a second pipeline.
pub async fn canonical_generation_observed(
    client: &MoldClient,
    requests: &[GenerateRequest],
    observer: Option<&CanonicalGenerationObserver<'_>>,
) -> Result<CanonicalGenerationReport> {
    let capabilities = client
        .server_capabilities()
        .await
        .context("could not read generation admission capabilities")?;
    let limit = capabilities
        .canonical_generation_batch_limit(requests)
        .map_err(|refusal| anyhow::anyhow!("{refusal}"))?;

    let mut report = CanonicalGenerationReport::default();
    let mut admitted = Vec::new();
    for (chunk_index, chunk) in requests.chunks(limit).enumerate() {
        let request_offset = (chunk_index * limit) as u32;
        let client_batch_id = new_client_batch_id();
        let admission = GenerationBatchAdmissionRequest {
            client_batch_id: client_batch_id.clone(),
            requests: chunk.to_vec(),
        };
        match admit_recovering_ambiguity(client, &admission).await {
            Ok(status) => {
                let authority =
                    match GenerationBatchAuthority::from_admission(&status, &client_batch_id) {
                        Ok(authority) => authority,
                        Err(error) => {
                            push_orchestration_failure(&mut report, error);
                            break;
                        }
                    };
                report
                    .admitted_client_ids
                    .push(status.client_batch_id.clone());
                report.authorities.push(authority.clone());
                observe(
                    observer,
                    CanonicalGenerationEvent::Admitted {
                        authority: authority.clone(),
                        status: status.clone(),
                        request_offset,
                    },
                );
                admitted.push((request_offset, admission.requests, status, authority));
            }
            Err(error) => {
                push_orchestration_failure(
                    &mut report,
                    format!(
                    "generation-batch admission failed for client id {client_batch_id}: {error:#}"
                ),
                );
                break;
            }
        }
    }

    for (request_offset, chunk, initial_status, authority) in admitted {
        let client_batch_id = initial_status.client_batch_id.clone();
        let status = match wait_for_batch(
            client,
            initial_status,
            &authority,
            request_offset,
            observer,
        )
        .await
        {
            Ok(status) => status,
            Err(error) => {
                push_orchestration_failure(
                    &mut report,
                    format!("could not reconcile accepted client id {client_batch_id}: {error:#}"),
                );
                continue;
            }
        };
        if let Err(error) =
            validate_terminal_children(&client_batch_id, &status.children, chunk.len())
        {
            push_orchestration_failure(&mut report, error);
            continue;
        }
        for child in status.children {
            let request = &chunk[child.index.saturating_sub(1) as usize];
            if let Some(error) = terminal_failure(&client_batch_id, &child) {
                report.failures.push(error);
            }
            report.outcomes.push(CanonicalGenerationOutcome {
                authority: authority.clone(),
                client_batch_id: client_batch_id.clone(),
                request_offset,
                request: request.clone(),
                child,
            });
        }
    }
    Ok(report)
}

/// Admit one caller-supplied idempotency chunk with the same ambiguity
/// recovery used by the higher-level orchestration.
pub async fn admit_canonical_batch(
    client: &MoldClient,
    request: &GenerationBatchAdmissionRequest,
) -> Result<GenerationBatchStatus> {
    admit_recovering_ambiguity(client, request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    fn request(prompt: &str) -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": prompt,
            "model": "flux-dev:q4",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap()
    }

    async fn mount_capabilities(server: &MockServer, max_outputs: u32) {
        let mut capabilities = crate::ServerCapabilities::default();
        capabilities.queue.heterogeneous_batch_max_outputs = Some(max_outputs);
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
            .mount(server)
            .await;
    }

    fn authority() -> GenerationBatchAuthority {
        GenerationBatchAuthority {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "accepted-id".into(),
        }
    }

    struct MissingThenAdmitted {
        calls: Arc<AtomicUsize>,
    }

    impl Respond for MissingThenAdmitted {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(404)
            } else {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "batch-1",
                    "client_batch_id": "recovery-key",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{"index": 1, "job_id": "job-1", "state": "accepted"}]
                }))
            }
        }
    }

    #[tokio::test]
    async fn admission_ambiguity_retries_a_missing_lookup_until_commit_is_visible() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{"))
            .expect(1)
            .mount(&server)
            .await;
        let calls = Arc::new(AtomicUsize::new(0));
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/by-client/recovery-key"))
            .respond_with(MissingThenAdmitted {
                calls: calls.clone(),
            })
            .expect(2)
            .mount(&server)
            .await;

        let status = admit_canonical_batch(
            &MoldClient::new(&server.uri()),
            &GenerationBatchAdmissionRequest {
                client_batch_id: "recovery-key".into(),
                requests: vec![request("one")],
            },
        )
        .await
        .unwrap();

        assert_eq!(status.id, "batch-1");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn terminal_child_validation_rejects_missing_and_duplicate_indices() {
        let child = |index| GenerationBatchChild {
            index,
            job_id: format!("job-{index}"),
            state: GenerationBatchChildState::Complete,
            error: None,
            error_code: None,
            retryable: None,
            created_at_ms: 0,
            updated_at_ms: 0,
            revision: 1,
            completed_at_ms: Some(1),
            terminal_error: None,
            result: None,
        };
        let cases = [
            (vec![child(1)], 2, "returned 1 children, expected 2"),
            (vec![child(1), child(1)], 2, "duplicated child index 1"),
        ];

        for (children, expected, detail) in cases {
            let error = validate_terminal_children("client-1", &children, expected).unwrap_err();
            assert!(error.contains(detail), "unexpected error: {error}");
        }
    }

    #[tokio::test]
    async fn transient_capability_failure_fails_closed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;

        let error = canonical_generation(&MoldClient::new(&server.uri()), &[request("one")])
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("could not read generation admission capabilities"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn admission_rejects_a_mismatched_client_identity() {
        let server = MockServer::start().await;
        mount_capabilities(&server, 64).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": "different-client-id",
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "job-1",
                    "state": "accepted"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let report = canonical_generation(&MoldClient::new(&server.uri()), &[request("one")])
            .await
            .unwrap();

        assert!(report.authorities.is_empty());
        assert!(report.outcomes.is_empty());
        assert_eq!(report.failures.len(), 1);
        assert!(
            report.failures[0].contains("returned client id"),
            "unexpected failure: {}",
            report.failures[0]
        );
    }

    #[tokio::test]
    async fn polling_rejects_a_replacement_server_instance() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": "accepted-id",
                "instance_id": "instance-2",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "job-1",
                    "state": "complete",
                    "result": {"filename": "wrong.png"}
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let initial = GenerationBatchStatus {
            id: "batch-1".into(),
            client_batch_id: "accepted-id".into(),
            instance_id: "instance-1".into(),
            durable: true,
            children: vec![GenerationBatchChild {
                index: 1,
                job_id: "job-1".into(),
                state: GenerationBatchChildState::Accepted,
                error: None,
                error_code: None,
                retryable: None,
                created_at_ms: 0,
                updated_at_ms: 0,
                revision: 1,
                completed_at_ms: None,
                terminal_error: None,
                result: None,
            }],
        };
        let authority = GenerationBatchAuthority::from_admission(&initial, "accepted-id").unwrap();

        let error = wait_for_batch(
            &MoldClient::new(&server.uri()),
            initial,
            &authority,
            0,
            None,
        )
        .await
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("instance changed from instance-1 to instance-2"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn report_preserves_every_terminal_child_error() {
        let server = MockServer::start().await;
        mount_capabilities(&server, 64).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(|request: &wiremock::Request| {
                let admission = request
                    .body_json::<GenerationBatchAdmissionRequest>()
                    .unwrap();
                ResponseTemplate::new(202).set_body_json(serde_json::json!({
                    "id": "batch-1",
                    "client_batch_id": admission.client_batch_id,
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [
                        {
                            "index": 1,
                            "job_id": "job-1",
                            "state": "held",
                            "error": "missing license",
                            "retryable": true
                        },
                        {
                            "index": 2,
                            "job_id": "job-2",
                            "state": "failed",
                            "error": "bad weights"
                        }
                    ]
                }))
            })
            .expect(1)
            .mount(&server)
            .await;

        let report = canonical_generation(
            &MoldClient::new(&server.uri()),
            &[request("one"), request("two")],
        )
        .await
        .unwrap();

        assert_eq!(report.failures.len(), 2);
        for detail in ["missing license", "bad weights", "retryable"] {
            assert!(
                report.failures.iter().any(|error| error.contains(detail)),
                "missing {detail:?} in {:?}",
                report.failures
            );
        }
    }

    #[tokio::test]
    async fn canonical_orchestration_chunks_and_preserves_global_offsets() {
        let server = MockServer::start().await;
        mount_capabilities(&server, 2).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(|request: &wiremock::Request| {
                let admission = request
                    .body_json::<GenerationBatchAdmissionRequest>()
                    .unwrap();
                let children = admission
                    .requests
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        serde_json::json!({
                            "index": index + 1,
                            "job_id": format!("{}-{index}", admission.client_batch_id),
                            "state": "complete",
                            "result": {"filename": format!("{index}.png")}
                        })
                    })
                    .collect::<Vec<_>>();
                ResponseTemplate::new(202).set_body_json(serde_json::json!({
                    "id": format!("batch-{}", admission.client_batch_id),
                    "client_batch_id": admission.client_batch_id,
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": children
                }))
            })
            .expect(2)
            .mount(&server)
            .await;
        let admitted_offsets = Mutex::new(Vec::new());
        let observer = |event| {
            if let CanonicalGenerationEvent::Admitted { request_offset, .. } = event {
                admitted_offsets.lock().unwrap().push(request_offset);
            }
        };

        let report = canonical_generation_observed(
            &MoldClient::new(&server.uri()),
            &[request("one"), request("two"), request("three")],
            Some(&observer),
        )
        .await
        .unwrap();

        assert_eq!(*admitted_offsets.lock().unwrap(), vec![0, 2]);
        assert_eq!(
            report
                .outcomes
                .iter()
                .map(|outcome| outcome.request_offset + outcome.child.index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(report.failures.is_empty());
    }

    #[tokio::test]
    async fn ambiguous_retry_waits_for_delayed_commit_without_publishing_stale_hold() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut status_reads = 0;
            for _ in 0..4 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request);
                if head.starts_with("POST /api/queue/job-1/retry") {
                    socket.shutdown().await.unwrap();
                    continue;
                }
                assert!(head.starts_with("GET /api/generation-batches/batch-1"));
                status_reads += 1;
                let child = match status_reads {
                    1 => serde_json::json!({
                        "index": 1, "job_id": "job-1", "state": "held",
                        "error": "old dependency failure", "retryable": true
                    }),
                    2 => serde_json::json!({
                        "index": 1, "job_id": "job-1", "state": "accepted"
                    }),
                    _ => serde_json::json!({
                        "index": 1, "job_id": "job-1", "state": "complete",
                        "result": {"filename": "settled.png"}
                    }),
                };
                let body = serde_json::json!({
                    "id": "batch-1",
                    "client_batch_id": "accepted-id",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [child]
                })
                .to_string();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let client = MoldClient::new(&base);
        assert!(matches!(
            retry_canonical_child(&client, &authority(), "job-1", 0)
                .await
                .unwrap(),
            CanonicalRetrySubmission::Ambiguous { .. }
        ));
        let events = Mutex::new(Vec::new());
        let observer = |event| events.lock().unwrap().push(event);

        let status =
            reconcile_ambiguous_retry_observed(&client, &authority(), "job-1", 0, Some(&observer))
                .await
                .unwrap();

        assert_eq!(
            status.children[0].state,
            GenerationBatchChildState::Complete
        );
        let events = events.into_inner().unwrap();
        assert!(matches!(
            events.first(),
            Some(CanonicalGenerationEvent::ReconcileDelayed { error, .. })
                if error.contains("remains held")
        ));
        assert!(!events.iter().any(|event| matches!(
            event,
            CanonicalGenerationEvent::Snapshot { status, .. }
                if status.children[0].state == GenerationBatchChildState::Held
        )));
        server.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn ambiguous_retry_bounds_repeated_transient_status_failures() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(503))
            .expect(AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS as u64)
            .mount(&server)
            .await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let client = MoldClient::new(&server.uri());
        let observer = move |event| observed.lock().unwrap().push(event);
        let error =
            reconcile_ambiguous_retry_observed(&client, &authority(), "job-1", 0, Some(&observer))
                .await
                .unwrap_err();
        assert!(
            error.to_string().contains(
                "ambiguous retry remains unconfirmed after 5 bounded attempts; retry lock retained"
            ),
            "unexpected error: {error:#}"
        );
        assert_eq!(
            events
                .lock()
                .unwrap()
                .iter()
                .filter(|event| matches!(event, CanonicalGenerationEvent::ReconcileDelayed { .. }))
                .count(),
            AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
        );
    }

    struct TransientMissingHeldThenComplete {
        calls: Arc<AtomicUsize>,
    }

    impl Respond for TransientMissingHeldThenComplete {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            match self.calls.fetch_add(1, Ordering::SeqCst) {
                0 => ResponseTemplate::new(500),
                1 => ResponseTemplate::new(429),
                2 => ResponseTemplate::new(404),
                3 => ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "batch-1",
                    "client_batch_id": "accepted-id",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{
                        "index": 1,
                        "job_id": "job-1",
                        "state": "held",
                        "error": "old dependency failure",
                        "retryable": true
                    }]
                })),
                _ => ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "batch-1",
                    "client_batch_id": "accepted-id",
                    "instance_id": "instance-1",
                    "durable": true,
                    "children": [{
                        "index": 1,
                        "job_id": "job-1",
                        "state": "complete",
                        "result": {"filename": "settled.png"}
                    }]
                })),
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn ambiguous_retry_recovers_after_transient_missing_and_held_reads() {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(TransientMissingHeldThenComplete {
                calls: calls.clone(),
            })
            .expect(AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS as u64)
            .mount(&server)
            .await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let client = MoldClient::new(&server.uri());
        let observer = move |event| observed.lock().unwrap().push(event);
        let status =
            reconcile_ambiguous_retry_observed(&client, &authority(), "job-1", 0, Some(&observer))
                .await
                .unwrap();
        assert_eq!(
            status.children[0].state,
            GenerationBatchChildState::Complete
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
        );
        assert!(!events.lock().unwrap().iter().any(|event| matches!(
            event,
            CanonicalGenerationEvent::Snapshot { status, .. }
                if status.children[0].state == GenerationBatchChildState::Held
        )));
    }

    // ── Ambiguous-retry revision fence ────────────────────────────────────────
    // An interrupted retry POST leaves the client unable to tell whether the
    // retry landed. `held` alone cannot answer it: a retry that landed and
    // was immediately re-held for a fresh reason presents identically to one
    // that never arrived. The child's revision separates them.

    fn held_batch(revision: u64) -> serde_json::Value {
        serde_json::json!({
            "id": "batch-1",
            "client_batch_id": "accepted-id",
            "instance_id": "instance-1",
            "durable": true,
            "children": [{
                "index": 1,
                "job_id": "job-1",
                "state": "held",
                "error": "dependency unavailable",
                "retryable": true,
                "created_at_ms": 10,
                "updated_at_ms": 20,
                "revision": revision,
                "completed_at_ms": null
            }]
        })
    }

    #[tokio::test(start_paused = true)]
    async fn an_unadvanced_revision_keeps_the_retry_fence_instead_of_republishing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(held_batch(4)))
            .mount(&server)
            .await;
        let client = MoldClient::new(&server.uri());
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let observer = move |event| observed.lock().unwrap().push(event);

        let error =
            reconcile_ambiguous_retry_observed(&client, &authority(), "job-1", 4, Some(&observer))
                .await
                .unwrap_err();

        assert!(
            error.to_string().contains("remains held at revision 4"),
            "unexpected error: {error:#}"
        );
        assert!(
            error.to_string().contains("retry lock retained"),
            "the fence must be reported as retained: {error:#}"
        );
        // Republishing the pre-retry snapshot as fresh authority is the exact
        // defect this fence exists to prevent.
        assert!(!events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, CanonicalGenerationEvent::Snapshot { .. })));
    }

    #[tokio::test(start_paused = true)]
    async fn an_advanced_revision_confirms_the_retry_even_while_still_held() {
        // The retry landed and the job was held again for a new reason. That
        // is a real outcome, not an unconfirmed retry.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(held_batch(5)))
            .mount(&server)
            .await;
        let client = MoldClient::new(&server.uri());
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let observer = move |event| observed.lock().unwrap().push(event);

        let status =
            reconcile_ambiguous_retry_observed(&client, &authority(), "job-1", 4, Some(&observer))
                .await
                .unwrap();

        assert_eq!(status.children[0].state, GenerationBatchChildState::Held);
        assert_eq!(status.children[0].revision, 5);
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, CanonicalGenerationEvent::Snapshot { .. })));
    }
}
