//! Arbitrary-duration video CLI contracts.

#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct VideoJobFlags {
    #[arg(skip)]
    pub expected_instance: Option<String>,
    #[arg(skip)]
    pub extension_recipe: Option<mold_core::GenerateRequest>,
    #[arg(skip)]
    pub recipe_overrides: std::collections::BTreeSet<&'static str>,
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
pub(crate) async fn destination(
    client: &mold_core::MoldClient,
    extending: bool,
) -> anyhow::Result<String> {
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

/// Original scalar defaults fill only fields the new command did not override.
/// Newly prepared inputs, seeds, filing and prompt provenance remain untouched.
pub(crate) fn apply_extension_recipe(
    prepared: &mut mold_core::GenerateRequest,
    flags: &VideoJobFlags,
) {
    let Some(base) = flags.extension_recipe.clone() else {
        return;
    };
    macro_rules! restore {($($field:ident),*$(,)?)=>{$(if !flags.recipe_overrides.contains(stringify!($field)){prepared.$field=base.$field;})*};}
    restore!(
        model,
        width,
        height,
        steps,
        guidance,
        negative_prompt,
        scheduler,
        cfg_plus,
        strength,
        reference_weight,
        true_cfg,
        cfg_start_step,
        control_model,
        control_scale,
        loras,
        fps,
        enable_audio,
        video_only,
        pipeline,
        ic_lora_control,
        spatial_upscale,
        temporal_upscale,
        guidance_overrides,
        sample_shift,
        distill_strength_high,
        distill_strength_low
    );
}
pub(crate) async fn library_recipe(
    client: &mold_core::MoldClient,
    filename: &str,
) -> anyhow::Result<mold_core::GenerateRequest> {
    use anyhow::Context;
    let row = client
        .gallery_video_source(filename)
        .await?
        .context("Choose an existing video from this machine's Library")?;
    anyhow::ensure!(
        row.filename == filename
            && row.trashed_at.is_none()
            && (row.format == Some(mold_core::OutputFormat::Mp4)
                || (row.format.is_none() && filename.to_ascii_lowercase().ends_with(".mp4"))),
        "Choose an active Library MP4"
    );
    anyhow::ensure!(!row.metadata_synthetic,"This older print has no recoverable recipe; open Extend in an app and choose a video model");
    mold_core::video_extension::extension_generation(&row.metadata, "").map_err(anyhow::Error::msg)
}

/// This is the first branch of `run`, before catalog lookup or opening input files.
pub(crate) async fn reconcile(
    client: &mold_core::MoldClient,
    id: uuid::Uuid,
    detach: bool,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let receipt = super::video_receipt::read(&id.to_string(), client.host())?;
    anyhow::ensure!(
        !receipt.chain,
        "This receipt belongs to an authored part; use mold jobs show or restart"
    );
    let instance = destination(client, false).await?;
    anyhow::ensure!(
        instance == receipt.instance_id,
        "The original video machine was replaced; no operation was changed"
    );
    let job = client
        .get_video_job(&id.to_string(), &receipt.instance_id)
        .await?;
    observe(client, job, &receipt.instance_id, detach, output).await
}

pub(crate) async fn admit(
    client: &mold_core::MoldClient,
    generation: &mold_core::GenerateRequest,
    options: &VideoJobOptions,
    before_submission: impl FnOnce(),
) -> anyhow::Result<Option<(mold_core::video_job::VideoJob, String)>> {
    admit_with_receipt(
        client,
        generation,
        options,
        before_submission,
        |id, instance| super::video_receipt::store(id, client.host(), instance),
    )
    .await
}
async fn admit_with_receipt(
    client: &mold_core::MoldClient,
    generation: &mold_core::GenerateRequest,
    options: &VideoJobOptions,
    before_submission: impl FnOnce(),
    persist: impl FnOnce(&str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<Option<(mold_core::video_job::VideoJob, String)>> {
    use crate::output::status;
    use anyhow::Context;
    use std::io::Write;
    let instance = destination(client, options.flags.extend_library.is_some()).await?;
    anyhow::ensure!(
        options
            .flags
            .expected_instance
            .as_ref()
            .is_none_or(|expected| expected == &instance),
        "The selected video machine was replaced during preparation; nothing was submitted"
    );
    if let Some(id) = options.flags.operation_id {
        // A supplied identity observes the original attempt. It must never
        // turn an absent/partially claimed admission into another generation.
        let receipt = super::video_receipt::read(&id.to_string(), client.host())?;
        anyhow::ensure!(
            receipt.instance_id == instance,
            "The original video machine was replaced"
        );
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
    persist(&operation, &instance)?;
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
            "Video {}: {:?} on {}. Use mold jobs show {} to monitor it.",
            job.id,
            job.state,
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
    if let Some(receipt) = super::video_receipt::read_optional(id, client.host())? {
        anyhow::ensure!(
            receipt.instance_id == instance,
            "The original operation destination was replaced"
        );
        if receipt.chain {
            return Ok(None);
        }
    }
    Ok(client
        .find_video_job(id, &instance)
        .await?
        .map(|job| (job, instance)))
}

#[cfg(test)]
mod transport_tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
        let caps = mold_core::ServerCapabilities {
            video_jobs: Some(mold_core::video_job::VideoJobCapabilities {
                supported: true,
                literal_prompts: true,
                library_extend: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(caps))
            .mount(&server)
            .await;
        Mock::given(method("GET")).and(path("/api/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"test","models_loaded":[],"gpu_info":null,"uptime_secs":1,"instance_id":"captured"}))).mount(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/video-jobs/validate"))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(200).set_body_json(plan()))
            .mount(&server)
            .await;
        server
    }
    #[test]
    fn duration_extension_preserves_original_recipe_explicit_overrides_and_fresh_inputs() {
        let mut original = request();
        original.steps = 9;
        original.width = 1024;
        original.height = 768;
        original.fps = Some(24);
        original.scheduler = Some(mold_core::Scheduler::Euler);
        original.frames = None;
        original.seed = None;
        let mut prepared = request();
        prepared.steps = 4;
        prepared.source_image = Some(vec![1, 2]);
        prepared.prompt = r"added\night".into();
        prepared.title = Some("New title".into());
        let flags = VideoJobFlags {
            extension_recipe: Some(original),
            recipe_overrides: std::collections::BTreeSet::from(["steps"]),
            ..Default::default()
        };
        apply_extension_recipe(&mut prepared, &flags);
        assert_eq!(prepared.steps, 4);
        assert_eq!((prepared.width, prepared.height), (1024, 768));
        assert_eq!(prepared.scheduler, Some(mold_core::Scheduler::Euler));
        assert_eq!(prepared.source_image, Some(vec![1, 2]));
        assert_eq!(prepared.prompt, r"added\night");
        assert_eq!(prepared.title.as_deref(), Some("New title"));
        assert_eq!(prepared.seed, Some(u64::MAX));
    }
    #[tokio::test]
    async fn failed_receipt_write_never_claims_inputs_or_posts_generation() {
        let server = server().await;
        let options = VideoJobOptions {
            duration_ms: 60000,
            flags: Default::default(),
        };
        let result = admit_with_receipt(
            &mold_core::MoldClient::new(&server.uri()),
            &request(),
            &options,
            || panic!("failed receipt claimed media"),
            |_, _| anyhow::bail!("disk unavailable"),
        )
        .await;
        assert!(result.is_err());
        assert!(!server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .any(|r| r.url.path() == "/api/video-jobs"));
    }
    #[tokio::test]
    async fn replaced_preparation_owner_never_validates_or_submits() {
        let server = server().await;
        let options = VideoJobOptions {
            duration_ms: 60000,
            flags: VideoJobFlags {
                expected_instance: Some("original".into()),
                ..Default::default()
            },
        };
        assert!(admit_with_receipt(
            &mold_core::MoldClient::new(&server.uri()),
            &request(),
            &options,
            || panic!("wrong owner claimed media"),
            |_, _| Ok(())
        )
        .await
        .is_err());
        assert!(!server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .any(|r| r.method == "POST"));
    }
    #[tokio::test]
    async fn duration_dry_run_validates_without_admission_or_claiming_inputs() {
        let server = server().await;
        let options = VideoJobOptions {
            duration_ms: 60_000,
            flags: VideoJobFlags {
                dry_run: true,
                ..Default::default()
            },
        };
        let result = admit_with_receipt(
            &mold_core::MoldClient::new(&server.uri()),
            &request(),
            &options,
            || panic!("dry run claimed media"),
            |_, _| Ok(()),
        )
        .await
        .unwrap();
        assert!(result.is_none());
        assert_eq!(
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .filter(|request| request.url.path() == "/api/video-jobs")
                .count(),
            0
        );
    }
    #[tokio::test]
    async fn ambiguous_admission_keeps_inputs_and_never_reposts() {
        let server = server().await;
        Mock::given(method("POST"))
            .and(path("/api/video-jobs"))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        let options = VideoJobOptions {
            duration_ms: 60_000,
            flags: Default::default(),
        };
        let mut retained = false;
        let error = admit_with_receipt(
            &mold_core::MoldClient::new(&server.uri()),
            &request(),
            &options,
            || retained = true,
            |_, _| Ok(()),
        )
        .await
        .unwrap_err();
        assert!(retained);
        assert!(error.to_string().contains("--operation-id"));
        let requests = server.received_requests().await.unwrap();
        let posted = requests
            .iter()
            .find(|request| request.url.path() == "/api/video-jobs")
            .unwrap();
        let body: mold_core::video_job::VideoJobRequest =
            serde_json::from_slice(&posted.body).unwrap();
        assert_eq!(body.generation.seed, Some(u64::MAX));
        assert_eq!(body.generation.prompt, r"literal\nscene");
        assert_eq!(body.duration_ms, 60_000);
        assert_eq!(body.validation_revision.as_deref(), Some("proof"));
    }
    #[tokio::test]
    async fn known_unknown_operation_never_starts_a_new_generation() {
        let server = server().await;
        let id = uuid::Uuid::new_v4();
        Mock::given(method("GET"))
            .and(path(format!("/api/video-jobs/{id}")))
            .respond_with(
                ResponseTemplate::new(404)
                    .set_body_json(serde_json::json!({"code":"VIDEO_JOB_NOT_FOUND"})),
            )
            .mount(&server)
            .await;
        let options = VideoJobOptions {
            duration_ms: 60_000,
            flags: VideoJobFlags {
                operation_id: Some(id),
                ..Default::default()
            },
        };
        assert!(admit_with_receipt(
            &mold_core::MoldClient::new(&server.uri()),
            &request(),
            &options,
            || panic!("reconciliation submitted"),
            |_, _| Ok(())
        )
        .await
        .is_err());
        assert!(!server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .any(|request| request.method == "POST"));
    }
}
