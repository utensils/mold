use anyhow::{Context, Result};
use mold_core::{
    GenerateRequest, GenerationBatchAdmissionRequest, GenerationBatchAuthority,
    GenerationBatchChild, GenerationBatchChildState, GenerationBatchStatus, GenerationRetryRequest,
    MoldClient,
};
use std::time::Duration;

#[derive(Debug)]
pub(crate) struct CanonicalGenerationOutcome {
    pub authority: GenerationBatchAuthority,
    pub client_batch_id: String,
    pub request: GenerateRequest,
    pub child: GenerationBatchChild,
}

#[derive(Debug, Default)]
pub(crate) struct CanonicalGenerationReport {
    pub authorities: Vec<GenerationBatchAuthority>,
    pub admitted_client_ids: Vec<String>,
    pub outcomes: Vec<CanonicalGenerationOutcome>,
    pub failures: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct CanonicalGenerationArtifact {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub request: GenerateRequest,
    pub metadata: mold_core::OutputMetadata,
}

#[derive(Debug, Clone)]
pub(crate) enum CanonicalGenerationEvent {
    Admitted {
        authority: GenerationBatchAuthority,
        status: GenerationBatchStatus,
    },
    Snapshot {
        authority: GenerationBatchAuthority,
        status: GenerationBatchStatus,
    },
    ReconcileDelayed {
        authority: GenerationBatchAuthority,
        error: String,
    },
}

#[derive(Debug)]
pub(crate) struct CanonicalHydratedArtifact {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub metadata: mold_core::OutputMetadata,
}

fn new_client_batch_id() -> String {
    let bytes = rand::random::<u128>().to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-4{:01x}{:02x}-{:01x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6] & 0x0f,
        bytes[7], (bytes[8] & 0x3f) | 0x80, bytes[9], bytes[10], bytes[11], bytes[12],
        bytes[13], bytes[14], bytes[15]
    )
}

fn admission_may_have_committed(error: &anyhow::Error) -> bool {
    if MoldClient::is_connection_error(error) {
        return true;
    }
    error.downcast_ref::<reqwest::Error>().is_some_and(|error| {
        error.is_timeout() || error.is_body() || error.is_decode() || error.is_request()
    })
}

enum CanonicalAdmission {
    Admitted(GenerationBatchStatus),
    MissingEndpoint,
}

async fn admit_recovering_ambiguity(
    client: &MoldClient,
    request: &GenerationBatchAdmissionRequest,
) -> Result<CanonicalAdmission> {
    match client.admit_generation_batch(request).await {
        Ok(status) => Ok(CanonicalAdmission::Admitted(status)),
        Err(error) if mold_core::client::is_missing_endpoint_error(&error) => {
            Ok(CanonicalAdmission::MissingEndpoint)
        }
        Err(error) if admission_may_have_committed(&error) => {
            const LOOKUP_ATTEMPTS: u32 = 5;
            let mut last_lookup_error = None;
            for attempt in 0..LOOKUP_ATTEMPTS {
                match client
                    .generation_batch_by_client_id(&request.client_batch_id)
                    .await
                {
                    Ok(Some(status)) => return Ok(CanonicalAdmission::Admitted(status)),
                    Ok(None) => {
                        return Err(error.context(format!(
                            "generation-batch admission is uncertain for client id {}; the host did not retain that idempotency key",
                            request.client_batch_id
                        )));
                    }
                    Err(lookup) => last_lookup_error = Some(lookup),
                }
                if attempt + 1 < LOOKUP_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(250 * u64::from(attempt + 1))).await;
                }
            }
            Err(error.context(format!(
                "generation-batch admission is uncertain for client id {}; idempotency lookup failed after {LOOKUP_ATTEMPTS} attempts: {}",
                request.client_batch_id,
                last_lookup_error
                    .map(|lookup| lookup.to_string())
                    .unwrap_or_else(|| "unknown lookup error".to_string())
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

async fn wait_for_batch(
    client: &MoldClient,
    mut status: GenerationBatchStatus,
    authority: &GenerationBatchAuthority,
    events: Option<&tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>>,
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
        let next = loop {
            match client.generation_batch(&status.id).await {
                Ok(Some(next)) => break next,
                Ok(None) => {
                    anyhow::bail!("generation batch {} disappeared", status.id)
                }
                Err(error) if reconciliation_error_is_retryable(&error) => {
                    send_event(
                        events,
                        CanonicalGenerationEvent::ReconcileDelayed {
                            authority: authority.clone(),
                            error: error.to_string(),
                        },
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                Err(error) => return Err(error),
            }
        };
        authority
            .validate_status(&next)
            .map_err(anyhow::Error::msg)?;
        send_event(
            events,
            CanonicalGenerationEvent::Snapshot {
                authority: authority.clone(),
                status: next.clone(),
            },
        );
        status = next;
    }
}

pub(crate) async fn retry_canonical_child(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    job_id: &str,
) -> Result<()> {
    client
        .retry_queue_job(&GenerationRetryRequest::from_authority(authority, job_id))
        .await
}

pub(crate) async fn reconcile_canonical_authority_observed(
    client: &MoldClient,
    authority: &GenerationBatchAuthority,
    events: Option<&tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>>,
) -> Result<GenerationBatchStatus> {
    let status = loop {
        match client.generation_batch(&authority.batch_id).await {
            Ok(Some(status)) => break status,
            Ok(None) => anyhow::bail!("generation batch {} disappeared", authority.batch_id),
            Err(error) if reconciliation_error_is_retryable(&error) => {
                send_event(
                    events,
                    CanonicalGenerationEvent::ReconcileDelayed {
                        authority: authority.clone(),
                        error: error.to_string(),
                    },
                );
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(error) => return Err(error),
        }
    };
    authority
        .validate_status(&status)
        .map_err(anyhow::Error::msg)?;
    send_event(
        events,
        CanonicalGenerationEvent::Snapshot {
            authority: authority.clone(),
            status: status.clone(),
        },
    );
    wait_for_batch(client, status, authority, events).await
}

fn reconciliation_error_is_retryable(error: &anyhow::Error) -> bool {
    if MoldClient::is_connection_error(error) {
        return true;
    }
    error.chain().any(|cause| {
        cause.downcast_ref::<reqwest::Error>().is_some_and(|error| {
            error.is_timeout()
                || error.is_connect()
                || error
                    .status()
                    .is_some_and(|status| status.is_server_error() || status.as_u16() == 429)
        })
    })
}

fn send_event(
    events: Option<&tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>>,
    event: CanonicalGenerationEvent,
) {
    if let Some(events) = events {
        let _ = events.send(event);
    }
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

/// Try the canonical protocol-v2 transport. `Ok(None)` is reserved for an
/// explicitly old or request-ineligible host; transient capability failures
/// fail closed and never silently select the attached endpoint.
pub(crate) async fn try_canonical_generation(
    client: &MoldClient,
    requests: &[GenerateRequest],
) -> Result<Option<CanonicalGenerationReport>> {
    try_canonical_generation_observed(client, requests, None).await
}

pub(crate) async fn try_canonical_generation_observed(
    client: &MoldClient,
    requests: &[GenerateRequest],
    events: Option<&tokio::sync::mpsc::UnboundedSender<CanonicalGenerationEvent>>,
) -> Result<Option<CanonicalGenerationReport>> {
    let capabilities = match client.server_capabilities().await {
        Ok(capabilities) => capabilities,
        Err(error) if mold_core::client::is_missing_endpoint_error(&error) => return Ok(None),
        Err(error) => {
            return Err(error).context("could not read generation admission capabilities")
        }
    };
    let Some(limit) = capabilities.canonical_generation_batch_limit(requests) else {
        return Ok(None);
    };

    let mut report = CanonicalGenerationReport::default();
    let mut admitted = Vec::new();
    for chunk in requests.chunks(limit) {
        let client_batch_id = new_client_batch_id();
        let admission = GenerationBatchAdmissionRequest {
            client_batch_id: client_batch_id.clone(),
            requests: chunk.to_vec(),
        };
        match admit_recovering_ambiguity(client, &admission).await {
            Ok(CanonicalAdmission::Admitted(status)) => {
                let authority =
                    match GenerationBatchAuthority::from_admission(&status, &client_batch_id) {
                        Ok(authority) => authority,
                        Err(error) => {
                            report.failures.push(error);
                            break;
                        }
                    };
                report
                    .admitted_client_ids
                    .push(status.client_batch_id.clone());
                report.authorities.push(authority.clone());
                send_event(
                    events,
                    CanonicalGenerationEvent::Admitted {
                        authority: authority.clone(),
                        status: status.clone(),
                    },
                );
                admitted.push((admission.requests, status, authority));
            }
            Ok(CanonicalAdmission::MissingEndpoint) if admitted.is_empty() => return Ok(None),
            Ok(CanonicalAdmission::MissingEndpoint) => {
                report.failures.push(format!(
                    "generation-batch endpoint disappeared after accepting client ids: {}",
                    report.admitted_client_ids.join(", ")
                ));
                break;
            }
            Err(error) => {
                report.failures.push(format!(
                    "generation-batch admission failed for client id {client_batch_id}: {error:#}"
                ));
                break;
            }
        }
    }

    for (chunk, initial_status, authority) in admitted {
        let client_batch_id = initial_status.client_batch_id.clone();
        let status = match wait_for_batch(client, initial_status, &authority, events).await {
            Ok(status) => status,
            Err(error) => {
                report.failures.push(format!(
                    "could not reconcile accepted client id {client_batch_id}: {error:#}"
                ));
                continue;
            }
        };
        for child in status.children {
            let Some(request) = chunk.get(child.index.saturating_sub(1) as usize) else {
                report.failures.push(format!(
                    "accepted client id {client_batch_id} returned invalid child index {}",
                    child.index
                ));
                continue;
            };
            if let Some(error) = terminal_failure(&client_batch_id, &child) {
                report.failures.push(error);
            }
            report.outcomes.push(CanonicalGenerationOutcome {
                authority: authority.clone(),
                client_batch_id: client_batch_id.clone(),
                request: request.clone(),
                child,
            });
        }
    }
    Ok(Some(report))
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

/// Canonical singleton transport shared by callers that need the rendered
/// bytes rather than the full durable reconciliation report.
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
        .find(|outcome| outcome.child.state == GenerationBatchChildState::Complete)
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
    request: &GenerationBatchAdmissionRequest,
) -> Result<GenerationBatchStatus> {
    match admit_recovering_ambiguity(client, request).await? {
        CanonicalAdmission::Admitted(status) => Ok(status),
        CanonicalAdmission::MissingEndpoint => {
            anyhow::bail!("generation-batch endpoint is unavailable")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    async fn mount_capabilities(server: &MockServer) {
        let mut capabilities = mold_core::ServerCapabilities::default();
        capabilities.queue.heterogeneous_batch = true;
        capabilities.queue.durable_batch_outcomes = true;
        capabilities.queue.admission_protocol_version = Some(2);
        capabilities.queue.heterogeneous_batch_max_outputs = Some(64);
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn transient_capability_failure_never_selects_legacy_transport() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        let error = try_canonical_generation(&MoldClient::new(&server.uri()), &[request("one")])
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("could not read generation admission capabilities"));
    }

    #[tokio::test]
    async fn definite_first_admission_404_selects_legacy_transport() {
        let server = MockServer::start().await;
        mount_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;

        let result = try_canonical_generation(&MoldClient::new(&server.uri()), &[request("one")])
            .await
            .unwrap();
        assert!(result.is_none());
        let requests = server.received_requests().await.unwrap();
        assert!(!requests
            .iter()
            .any(|request| request.url.path().contains("/by-client/")));
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
                "children": [{"index": 1, "job_id": "job-1", "state": "complete", "result": {"filename": "wrong.png"}}]
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
                retryable: None,
                created_at_ms: 0,
                updated_at_ms: 0,
                completed_at_ms: None,
                terminal_error: None,
                result: None,
            }],
        };
        let authority = GenerationBatchAuthority::from_admission(&initial, "accepted-id").unwrap();
        let error = wait_for_batch(&MoldClient::new(&server.uri()), initial, &authority, None)
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
    async fn reports_every_terminal_child_error() {
        let server = MockServer::start().await;
        mount_capabilities(&server).await;
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
                        {"index": 1, "job_id": "job-1", "state": "held", "error": "missing license", "retryable": true},
                        {"index": 2, "job_id": "job-2", "state": "failed", "error": "bad weights"}
                    ]
                }))
            })
            .expect(1)
            .mount(&server)
            .await;

        let report = try_canonical_generation(
            &MoldClient::new(&server.uri()),
            &[request("one"), request("two")],
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(report.failures.len(), 2);
        assert!(report
            .failures
            .iter()
            .any(|error| error.contains("missing license")));
        assert!(report
            .failures
            .iter()
            .any(|error| error.contains("bad weights")));
        assert!(report
            .failures
            .iter()
            .any(|error| error.contains("retryable")));
    }
}
