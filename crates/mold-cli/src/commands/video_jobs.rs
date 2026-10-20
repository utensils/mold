//! Arbitrary-duration video CLI contracts.

#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct VideoJobFlags {
    #[arg(skip)]
    pub dry_run: bool,
    /// Extend this exact video filename from the destination's Library.
    #[arg(long, requires = "duration", conflicts_with_all = ["local", "extend", "video"], help_heading = "Video")]
    pub extend_library: Option<String>,
    /// Known operation UUID to reconcile after an uncertain submission.
    #[arg(
        long,
        requires = "duration",
        conflicts_with = "local",
        help_heading = "Video"
    )]
    pub operation_id: Option<uuid::Uuid>,
    /// Return after durable admission; monitor with `mold jobs show ID`.
    #[arg(
        long,
        requires = "duration",
        conflicts_with = "local",
        help_heading = "Video"
    )]
    pub detach: bool,
}

pub(crate) struct VideoJobOptions {
    pub duration_ms: u64,
    pub flags: VideoJobFlags,
}

/// Parse seconds directly into the wire's exact millisecond integer.
pub(crate) fn parse_duration_ms(value: &str) -> Result<u64, String> {
    let invalid =
        || "duration must be positive seconds with at most three decimal places".to_owned();
    let (seconds, fraction) = value
        .split_once('.')
        .map_or((value, None), |(whole, fraction)| (whole, Some(fraction)));
    if seconds.is_empty() || !seconds.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    let seconds = seconds.parse::<u64>().map_err(|_| invalid())?;
    let fractional_ms = match fraction {
        None => 0,
        Some(text)
            if !text.is_empty()
                && text.len() <= 3
                && text.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            text.parse::<u64>().map_err(|_| invalid())? * 10u64.pow(3 - text.len() as u32)
        }
        Some(_) => return Err(invalid()),
    };
    seconds
        .checked_mul(1000)
        .and_then(|whole| whole.checked_add(fractional_ms))
        .filter(|total| *total > 0)
        .ok_or_else(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_are_exact_positive_milliseconds_without_float_rounding() {
        for (text, expected) in [
            ("30", 30_000),
            ("4.458", 4458),
            ("0.001", 1),
            ("1.2", 1200),
            ("18446744073709551.615", u64::MAX),
        ] {
            assert_eq!(parse_duration_ms(text).unwrap(), expected);
        }
        for text in [
            "0",
            "0.000",
            "-1",
            "NaN",
            "inf",
            "1e3",
            "1.0001",
            "18446744073709551.616",
            "",
            ".1",
            "1.",
            " 30",
            "+3",
        ] {
            assert!(parse_duration_ms(text).is_err(), "{text}");
        }
    }
}

/// Capture the server installation before private validation or admission.
async fn destination(client: &mold_core::MoldClient, extending: bool) -> anyhow::Result<String> {
    let capabilities = client.capabilities().await?;
    let caps = capabilities.video_jobs.as_ref();
    anyhow::ensure!(
        caps.is_some_and(|caps| caps.supported
            && caps.literal_prompts
            && (!extending || caps.library_extend)),
        "Update this machine before making videos with a requested duration"
    );
    client
        .server_status()
        .await?
        .instance_id
        .filter(|instance| !instance.is_empty())
        .ok_or_else(|| anyhow::anyhow!("This machine did not identify its installation"))
}

pub(crate) async fn admit(
    client: &mold_core::MoldClient,
    generation: &mold_core::GenerateRequest,
    options: &VideoJobOptions,
    before_submission: impl FnOnce(),
) -> anyhow::Result<Option<(mold_core::video_job::VideoJob, String)>> {
    use crate::output::status;
    use anyhow::Context;
    use std::io::Write;
    let instance = destination(client, options.flags.extend_library.is_some()).await?;
    if let Some(id) = options.flags.operation_id {
        // A supplied identity observes the original attempt. It must never
        // turn an absent/partially claimed admission into another generation.
        let job = client.get_video_job(&id.to_string(), &instance).await
            .context("Could not reconcile this operation. Inspect or cancel its exact ID before starting a new attempt")?;
        return Ok(Some((job, instance)));
    }
    let mut request = mold_core::video_job::VideoJobRequest {
        generation: generation.clone(),
        duration_ms: options.duration_ms,
        source: options.flags.extend_library.as_ref().map(|filename| {
            mold_core::video_job::VideoJobSource {
                filename: filename.clone(),
                expected_archive_identity_sha256: None,
                retained_upload: None,
            }
        }),
        scenes: None,
        validation_revision: None,
    };
    let plan = client.validate_video_job(&request, &instance).await?;
    anyhow::ensure!(
        !plan.validation_revision.is_empty(),
        "Machine returned no video validation proof"
    );
    status!("{} clips joined into one video; single-generation limit: {} frames at {} FPS; overlap: {} frames ({:?})",
        plan.clip_count, plan.single_clip_max_frames, plan.fps, plan.overlap_frames, plan.join_mode);
    for warning in &plan.warnings {
        status!("{warning}");
    }
    if options.flags.dry_run {
        return Ok(None);
    }
    request.validation_revision = Some(plan.validation_revision);
    let operation = uuid::Uuid::new_v4().to_string();
    eprintln!("Video operation: {operation}");
    std::io::stderr().flush()?;
    before_submission();
    let job = client.create_video_job(&request, &operation, &instance).await
        .with_context(|| format!("Submission outcome may be uncertain. Reconcile with --operation-id {operation}; do not start a duplicate"))?;
    Ok(Some((job, instance)))
}

pub(crate) async fn observe(
    client: &mold_core::MoldClient,
    mut job: mold_core::video_job::VideoJob,
    instance: &str,
    detach: bool,
    output: Option<&str>,
) -> anyhow::Result<()> {
    use crate::output::status;
    use anyhow::Context;
    if detach {
        status!(
            "Queued video {} on {}. Use mold jobs show {} to monitor it.",
            job.id,
            client.host(),
            job.id
        );
        return Ok(());
    }
    let mut shown = None;
    while !job.state.is_terminal() {
        let progress = (job.state, job.completed_clips);
        if shown != Some(progress) {
            status!(
                "Video {}: {:?}, {}/{} clips",
                job.id,
                job.state,
                job.completed_clips,
                job.plan.clip_count
            );
            shown = Some(progress);
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        job = client
            .get_video_job(&job.id, instance)
            .await
            .with_context(|| {
                format!(
                    "Video remains on {}. Reconnect with --operation-id {}",
                    client.host(),
                    job.id
                )
            })?;
        if job.state == mold_core::video_job::VideoJobState::Paused {
            anyhow::bail!(
                "Video {} is paused. Resume it with mold jobs resume {}",
                job.id,
                job.id
            );
        }
    }
    anyhow::ensure!(
        job.state == mold_core::video_job::VideoJobState::Completed,
        "Video {} is {:?}: {}. Inspect with mold jobs show {}",
        job.id,
        job.state,
        job.error_detail
            .as_deref()
            .or(job.error.as_deref())
            .unwrap_or("no additional detail"),
        job.id
    );
    let filename = job
        .output_filename
        .context("Completed video has no published Library filename")?;
    status!("Saved in the Library on {}: {}", client.host(), filename);
    if let Some(output) = output {
        crate::commands::library_download::download_published(client, &filename, output).await?;
    }
    Ok(())
}

pub(crate) async fn supported_instance(
    client: &mold_core::MoldClient,
) -> anyhow::Result<Option<String>> {
    if !client
        .capabilities()
        .await?
        .video_jobs
        .as_ref()
        .is_some_and(|caps| caps.supported && caps.literal_prompts)
    {
        return Ok(None);
    }
    Ok(Some(
        client
            .server_status()
            .await?
            .instance_id
            .filter(|id| !id.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Machine did not identify its installation"))?,
    ))
}

pub(crate) async fn lookup(
    client: &mold_core::MoldClient,
    id: &str,
) -> anyhow::Result<Option<(mold_core::video_job::VideoJob, String)>> {
    if !uuid::Uuid::parse_str(id).is_ok_and(|parsed| parsed.to_string() == id) {
        return Ok(None);
    }
    let Some(instance) = supported_instance(client).await? else {
        return Ok(None);
    };
    Ok(client
        .find_video_job(id, &instance)
        .await?
        .map(|job| (job, instance)))
}

#[cfg(test)]
mod transport_tests {
    use super::*;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use wiremock::matchers::{header, method, path};

    fn request() -> mold_core::GenerateRequest {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({"model":"h3-pdmd2:bf16","prompt":"base","width":768,"height":768,"steps":4,"seed":u64::MAX,"output_format":"mp4"})).unwrap();
        request.prompt = r"literal\nscene".into();
        request
    }
    fn plan() -> serde_json::Value {
        serde_json::json!({"validation_revision":"proof","model":"h3-pdmd2:bf16","width":768,"height":768,"fps":24,"single_clip_max_frames":107,"overlap_frames":1,"join_mode":"continuous","clip_count":"14","part_count":"14","added_frames":"1440","source_frames":"0","total_frames":"1440","added_duration_ms":"60000","total_duration_ms":"60000","warnings":[]})
    }
    async fn server() -> MockServer {
        let server = MockServer::start().await;
        let caps = mold_core::ServerCapabilities { video_jobs: Some(mold_core::video_job::VideoJobCapabilities { supported:true, literal_prompts:true, library_extend:true, ..Default::default() }), ..Default::default() };
        Mock::given(method("GET")).and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(caps)).mount(&server).await;
        Mock::given(method("GET")).and(path("/api/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"test","models_loaded":[],"gpu_info":null,"uptime_secs":1,"instance_id":"captured"}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/api/video-jobs/validate"))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(200).set_body_json(plan())).mount(&server).await;
        server
    }
    #[tokio::test]
    async fn duration_dry_run_validates_without_admission_or_claiming_inputs() {
        let server = server().await;
        let options = VideoJobOptions { duration_ms:60_000, flags:VideoJobFlags { dry_run:true, ..Default::default() } };
        let result = admit(&mold_core::MoldClient::new(&server.uri()), &request(), &options, || panic!("dry run claimed media")).await.unwrap();
        assert!(result.is_none());
        assert_eq!(server.received_requests().await.unwrap().iter().filter(|request| request.url.path() == "/api/video-jobs").count(), 0);
    }
    #[tokio::test]
    async fn ambiguous_admission_keeps_inputs_and_never_reposts() {
        let server = server().await;
        Mock::given(method("POST")).and(path("/api/video-jobs"))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(503)).expect(1).mount(&server).await;
        let options = VideoJobOptions { duration_ms:60_000, flags:Default::default() };
        let mut retained = false;
        let error = admit(&mold_core::MoldClient::new(&server.uri()), &request(), &options, || retained = true).await.unwrap_err();
        assert!(retained);
        assert!(error.to_string().contains("--operation-id"));
        let requests = server.received_requests().await.unwrap();
        let posted = requests.iter().find(|request| request.url.path() == "/api/video-jobs").unwrap();
        let body: mold_core::video_job::VideoJobRequest = serde_json::from_slice(&posted.body).unwrap();
        assert_eq!(body.generation.seed, Some(u64::MAX));
        assert_eq!(body.generation.prompt, r"literal\nscene");
        assert_eq!(body.duration_ms, 60_000);
        assert_eq!(body.validation_revision.as_deref(), Some("proof"));
    }
    #[tokio::test]
    async fn known_unknown_operation_never_starts_a_new_generation() {
        let server = server().await;
        let id = uuid::Uuid::new_v4();
        Mock::given(method("GET")).and(path(format!("/api/video-jobs/{id}")))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({"code":"VIDEO_JOB_NOT_FOUND"}))).mount(&server).await;
        let options = VideoJobOptions { duration_ms:60_000, flags:VideoJobFlags { operation_id:Some(id), ..Default::default() } };
        assert!(admit(&mold_core::MoldClient::new(&server.uri()), &request(), &options, || panic!("reconciliation submitted")).await.is_err());
        assert!(!server.received_requests().await.unwrap().iter().any(|request| request.method == "POST"));
    }
}
