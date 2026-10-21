//! Explicit new attempts. The original parent and its progress never change.
use anyhow::{ensure, Context, Result};
use mold_core::{video_job::VideoJob, MoldClient};
fn restart_capability(caps: &mold_core::ServerCapabilities) -> Result<()> {
    ensure!(
        caps.video_jobs
            .as_ref()
            .is_some_and(|c| c.supported && c.literal_prompts && c.restart),
        "Update this machine before restarting or sending duration videos"
    );
    Ok(())
}
pub(crate) async fn restart(
    source: &MoldClient,
    destination: &MoldClient,
    id: &str,
    source_instance: &str,
) -> Result<()> {
    restart_with_receipt(
        source,
        destination,
        id,
        source_instance,
        |operation, instance| super::video_receipt::store(operation, destination.host(), instance),
    )
    .await
}
async fn restart_with_receipt(
    source: &MoldClient,
    destination: &MoldClient,
    id: &str,
    source_instance: &str,
    persist: impl FnOnce(&str, &str) -> Result<()>,
) -> Result<()> {
    restart_capability(&source.capabilities().await?)?;
    restart_capability(&destination.capabilities().await?)?;
    let target_instance = destination
        .server_status()
        .await?
        .instance_id
        .context("Destination has no installation identity")?;
    let options = source.video_restart_options(id, source_instance).await?;
    let permission = if source_instance == target_instance {
        &options.restart
    } else {
        &options.send
    };
    ensure!(
        permission.supported,
        "{}",
        permission
            .reason
            .as_deref()
            .unwrap_or("This video cannot start again here")
    );
    let exported = source
        .export_video_restart(
            id,
            &mold_core::job_restart::RestartExportRequest {
                source_instance_id: source_instance.into(),
                source_revision: options.source_revision,
                destination_instance_id: target_instance.clone(),
            },
        )
        .await?;
    let prefix = if let Some(prefix) = &exported.prefix {
        if prefix.retained_upload.is_none() {
            ensure!(
                source
                    .capabilities()
                    .await?
                    .video_jobs
                    .is_some_and(|c| c.source_transfer)
                    && destination
                        .capabilities()
                        .await?
                        .video_jobs
                        .is_some_and(|c| c.source_transfer),
                "Update both machines before transferring the original video prefix"
            );
        }
        Some(
            source
                .transfer_video_restart_prefix(
                    id,
                    source_instance,
                    prefix,
                    destination,
                    &target_instance,
                    &uuid::Uuid::new_v4().to_string(),
                )
                .await?,
        )
    } else {
        None
    };
    let plan = destination
        .validate_video_job_json_on_instance(
            &exported.request_json,
            &target_instance,
            prefix.as_ref(),
        )
        .await?;
    ensure!(
        !plan.validation_revision.is_empty(),
        "Destination returned no video validation proof"
    );
    let operation = uuid::Uuid::new_v4().to_string();
    persist(&operation, &target_instance)?;
    println!("New video attempt operation: {operation}");
    std::io::Write::flush(&mut std::io::stdout())?;
    let created=destination.create_video_job_json_on_instance(&exported.request_json,&operation,&target_instance,&plan.validation_revision,prefix.as_ref()).await
        .with_context(||format!("Acceptance may be uncertain. Original {id} is unchanged. Reconcile operation {operation} on the destination before starting another attempt"))?;
    print_accepted(&created, destination);
    Ok(())
}
fn print_accepted(job: &VideoJob, destination: &MoldClient) {
    println!(
        "Started {} on {}. Original remains recoverable.",
        job.id,
        destination.host()
    );
}
/// An imported identity never triggers private export or media preparation.
pub(crate) async fn reconcile(destination: &MoldClient, operation: &str) -> Result<()> {
    let receipt = super::video_receipt::read(operation, destination.host())?;
    let actual = destination
        .server_status()
        .await?
        .instance_id
        .context("Destination has no installation identity")?;
    ensure!(
        receipt.instance_id == actual,
        "Original destination was replaced; no operation changed"
    );
    if receipt.chain {
        let job = destination.get_chain_job(operation).await?;
        ensure!(
            job.summary.id == operation,
            "Machine returned another operation"
        );
        println!("{}: {:?}", job.summary.id, job.summary.state);
    } else {
        restart_capability(&destination.capabilities().await?)?;
        let job = destination
            .get_video_job(operation, &receipt.instance_id)
            .await?;
        print_accepted(&job, destination);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_bytes, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    // Wiremock's generic header matcher splits comma-delimited lists, but this
    // protocol field is one opaque JSON value. Match the actual header bytes.
    fn source_upload_header(value: &str) -> impl wiremock::Match {
        let expected = value.to_owned();
        move |request: &wiremock::Request| {
            request
                .headers
                .get("x-mold-video-source-upload")
                .is_some_and(|actual| actual.as_bytes() == expected.as_bytes())
        }
    }
    async fn capabilities(server: &MockServer, supported: bool) {
        let caps = mold_core::ServerCapabilities {
            video_jobs: supported.then_some(mold_core::video_job::VideoJobCapabilities {
                supported: true,
                literal_prompts: true,
                restart: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(caps))
            .mount(server)
            .await;
    }
    #[tokio::test]
    async fn legacy_destination_is_refused_before_private_export() {
        let source = MockServer::start().await;
        let target = MockServer::start().await;
        capabilities(&source, true).await;
        capabilities(&target, false).await;
        assert!(restart_with_receipt(
            &MoldClient::new(&source.uri()),
            &MoldClient::new(&target.uri()),
            "original",
            "source",
            |_, _| panic!("legacy destination wrote receipt")
        )
        .await
        .is_err());
        assert_eq!(source.received_requests().await.unwrap().len(), 1);
        assert_eq!(target.received_requests().await.unwrap().len(), 1);
    }
    #[tokio::test]
    async fn lost_admission_keeps_exact_raw_request_prefix_grant_and_published_receipt() {
        let source = MockServer::start().await;
        let target = MockServer::start().await;
        capabilities(&source, true).await;
        capabilities(&target, true).await;
        Mock::given(method("GET")).and(path("/api/status")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"test","models_loaded":[],"gpu_info":null,"uptime_secs":1,"instance_id":"same"}))).mount(&target).await;
        Mock::given(method("GET")).and(path("/api/video-jobs/original/restart-options")).and(header("x-mold-destination-instance","same"))
          .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"source_instance_id":"same","source_revision":"revision","restart":{"supported":true},"send":{"supported":true}}))).expect(1).mount(&source).await;
        let raw = r#"{"generation":{"model":"installed-video","prompt":"literal\\nscene","width":768,"height":768,"steps":2,"guidance":1,"seed":18446744073709551615},"duration_ms":"4458","source":{"filename":"original.mp4","expected_archive_identity_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}"#;
        serde_json::from_str::<mold_core::video_job::VideoJobRequest>(raw).unwrap();
        let upload = mold_core::video_job::VideoSourceUploadRef {
            handle: uuid::Uuid::new_v4().to_string(),
            sha256: "a".repeat(64),
            size_bytes: 10,
        };
        Mock::given(method("POST")).and(path("/api/video-jobs/original/restart-export")).and(header("x-mold-destination-instance","same"))
          .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"source_instance_id":"same","source_revision":"revision","destination_instance_id":"same","request_json":raw,"prefix":{"filename":"original.mp4","archive_identity_sha256":"b".repeat(64),"sha256":upload.sha256,"size_bytes":"10","source_ref":"opaque","retained_upload":upload,"upload_spec_json":"opaque local spec; not uploaded"}}))).expect(1).mount(&source).await;
        let plan = serde_json::json!({"validation_revision":"proof","model":"installed-video","width":768,"height":768,"fps":24,"single_clip_max_frames":107,"overlap_frames":1,"join_mode":"continuous","clip_count":"2","part_count":"2","added_frames":"107","source_frames":"107","total_frames":"214","added_duration_ms":"4458","total_duration_ms":"8916","warnings":[]});
        let upload_header = serde_json::to_string(&upload).unwrap();
        Mock::given(method("POST"))
            .and(path("/api/video-jobs/validate"))
            .and(header("x-mold-destination-instance", "same"))
            .and(source_upload_header(&upload_header))
            .and(body_bytes(raw.as_bytes().to_vec()))
            .respond_with(ResponseTemplate::new(200).set_body_json(plan))
            .expect(1)
            .mount(&target)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/video-jobs"))
            .and(header("x-mold-destination-instance", "same"))
            .and(header("x-mold-video-validation-revision", "proof"))
            .and(source_upload_header(&upload_header))
            .and(body_bytes(raw.as_bytes().to_vec()))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&target)
            .await;
        let receipt = std::cell::RefCell::new(None);
        let error = restart_with_receipt(
            &MoldClient::new(&source.uri()),
            &MoldClient::new(&target.uri()),
            "original",
            "same",
            |id, instance| {
                *receipt.borrow_mut() = Some((id.to_string(), instance.to_string()));
                Ok(())
            },
        )
        .await
        .unwrap_err();
        let target_requests = target.received_requests().await.unwrap();
        let validation = target_requests
            .iter()
            .find(|r| r.url.path() == "/api/video-jobs/validate")
            .unwrap();
        assert_eq!(
            validation.body,
            raw.as_bytes(),
            "opaque validation body changed"
        );
        assert_eq!(
            validation
                .headers
                .get("x-mold-destination-instance")
                .unwrap()
                .to_str()
                .unwrap(),
            "same"
        );
        assert_eq!(
            validation
                .headers
                .get("x-mold-video-source-upload")
                .unwrap()
                .to_str()
                .unwrap(),
            upload_header,
            "prefix grant changed"
        );
        let source_paths: Vec<_> = source
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|r| (r.method.to_string(), r.url.path().to_string()))
            .collect();
        let target_paths: Vec<_> = target
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|r| (r.method.to_string(), r.url.path().to_string()))
            .collect();
        assert!(
            error.to_string().contains("Acceptance may be uncertain"),
            "{error:#}; source {source_paths:?}; target {target_paths:?}"
        );
        let (id, instance) = receipt.into_inner().unwrap();
        assert_eq!(instance, "same");
        uuid::Uuid::parse_str(&id).unwrap();
        let calls = target.received_requests().await.unwrap();
        let posted = calls
            .iter()
            .find(|r| r.url.path() == "/api/video-jobs")
            .unwrap();
        assert_eq!(
            posted
                .headers
                .get("x-mold-operation-id")
                .unwrap()
                .to_str()
                .unwrap(),
            id
        );
        assert!(!calls
            .iter()
            .any(|r| r.method == "DELETE" || r.url.path().contains("video-source-uploads")));
        assert!(!source.received_requests().await.unwrap().iter().any(|r| r
            .url
            .path()
            .ends_with("cancel")
            || r.url.path().ends_with("resume")));
    }
}
