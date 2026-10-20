use super::*;
use mold_core::{GenerationReference, GenerationReferenceProvenance, SourceImageCapability};
use serde_json::json;

fn model() -> mold_core::ModelInfoExtended {
    serde_json::from_value(json!({
        "name":"minimax-h3-fl2va:fasth3-v2-int8","family":"minimax-h3","size_gb":8,
        "is_loaded":false,"last_used":null,"hf_repo":"test","downloaded":true,
        "default_steps":9,"default_guidance":0.0,"default_width":768,"default_height":512,
        "description":"test","default_frames":209,"default_fps":24,"min_frames":107,
        "frame_step":17,"frame_offset":5,"supports_sequence":true,"source_image":"optional"
    }))
    .unwrap()
}
fn limits() -> VideoLimits {
    serde_json::from_value(json!({"model":model().info.name,"supports_sequence":true,
        "frames_per_clip_cap":345,"frames_per_clip_recommended":209,"supports_audio":true,
        "authoring":{"motion_tail_frames":1,"min_frames":107,"frame_step":17,"frame_offset":5}}
    ))
    .unwrap()
}
#[test]
fn legacy_part_fixture_uses_the_shared_model_grid() {
    let request = legacy_fixture_request(
        &model(),
        &limits(),
        "First",
        "Next",
        Some(5.0),
        None,
        None,
        None,
    )
    .unwrap();
    assert!(!request.ephemeral);
    assert_eq!(request.motion_tail_frames, 1);
    assert_eq!(request.fps, 24);
    assert_eq!(request.enable_audio, Some(true));
    assert_eq!(request.stages.len(), 2);
    assert_eq!(request.stages[1].frames % 17, 5);
    assert_eq!(request.stages[0].prompt, "First");
    assert_eq!(request.stages[1].prompt, "Next");
}
#[test]
fn legacy_grid_refuses_invalid_clip_lengths() {
    assert!(
        legacy_fixture_request(&model(), &limits(), "", "Next", None, None, None, None).is_err()
    );
    assert!(legacy_fixture_request(
        &model(),
        &limits(),
        "First",
        "Next",
        Some(f64::NAN),
        None,
        None,
        None
    )
    .is_err());
    assert!(legacy_fixture_request(
        &model(),
        &limits(),
        "First",
        "Next",
        Some(100.0),
        None,
        None,
        None
    )
    .is_err());
}
#[test]
fn operation_identity_is_stable_and_uses_the_whole_interaction_id() {
    let id = operation_id(1);
    assert_eq!(id, operation_id(1));
    assert_ne!(id, operation_id(1 + (1u64 << 32)));
    assert_eq!(id.len(), 36);
    assert_eq!(id.chars().filter(|ch| *ch == '-').count(), 4);
}
#[test]
fn video_commands_are_small_and_mutations_require_guild_admin() {
    let command = video();
    for name in ["make", "extend", "continue", "status", "resume", "cancel"] {
        assert!(command.subcommands.iter().any(|sub| sub.name == name));
    }
    for name in ["extend", "continue", "status", "resume", "cancel"] {
        let sub = command
            .subcommands
            .iter()
            .find(|sub| sub.name == name)
            .unwrap();
        assert!(sub.guild_only);
        assert!(sub
            .required_permissions
            .contains(poise::serenity_prelude::Permissions::MANAGE_GUILD));
    }
    assert!(command
        .subcommands
        .iter()
        .all(|sub| sub.parameters.len() < 10));
}
#[test]
fn default_make_needs_one_prompt_and_optional_duration_and_scene() {
    let command = video();
    let make = command
        .subcommands
        .iter()
        .find(|s| s.name == "make")
        .unwrap();
    let required: Vec<_> = make
        .parameters
        .iter()
        .filter(|p| p.required)
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(required, ["prompt"]);
    assert!(make
        .parameters
        .iter()
        .any(|p| p.name == "duration" && !p.required));
    assert!(make
        .parameters
        .iter()
        .any(|p| p.name == "next_prompt" && !p.required));
    for name in ["status", "resume", "cancel"] {
        let action = command.subcommands.iter().find(|s| s.name == name).unwrap();
        assert!(action
            .parameters
            .iter()
            .any(|p| p.name == "legacy_part" && !p.required));
    }
}

fn image_bytes() -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(16, 16)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}
fn source(request: &ChainRequest) -> ChainJobDetail {
    serde_json::from_value(json!({
        "id":"original-job","state":"completed","model":request.model,"stage_count":2,
        "current_stage":2,"created_at_unix_ms":1,"updated_at_unix_ms":2,"ephemeral":false,
        "stages":[],"retakes":[],"amends":[],
        "finalizes":[{"output":"final/one.mp4","gallery_filename":"one.mp4","at_unix_ms":1,"stage_seeds":["42","42"]},
            {"output":"final/two.mp4","gallery_filename":"two.mp4","at_unix_ms":2,"stage_seeds":["42","42"]}],
        "script":mold_core::ChainScript::from(request),
    })).unwrap()
}
#[test]
fn ref2va_repeats_inline_image_and_continuation_recalls_it() {
    let mut model = model();
    model.info.name = "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step".into();
    model.defaults.default_frames = Some(124);
    let mut limits = limits();
    limits.model = model.info.name.clone();
    limits.authoring.as_mut().unwrap().motion_tail_frames = 0;
    limits.frames_per_clip_recommended = 124;
    let request = legacy_fixture_request(
        &model,
        &limits,
        "First",
        "Next",
        None,
        None,
        None,
        Some(image_bytes()),
    )
    .unwrap();
    assert!(request
        .stages
        .iter()
        .all(|stage| stage.source_image.is_none()));
    let refs = request.stages[0].generation_references.clone().unwrap();
    assert_eq!(
        request.stages[1].generation_references.as_ref(),
        Some(&refs)
    );
    assert!(matches!(
        refs[0].media(),
        GenerationReferenceAuthority::Inline { .. }
    ));
    let next = continue_request(&source(&request), &model, &limits, "New part", None).unwrap();
    assert_eq!(next.stages[0].generation_references.as_ref(), Some(&refs));
    assert_eq!(next.motion_tail_frames, 0);
}
#[test]
fn continued_recipe_uses_source_take_and_frozen_geometry_not_live_defaults() {
    let request = legacy_fixture_request(
        &model(),
        &limits(),
        "First",
        "Next",
        None,
        Some(768),
        Some(512),
        None,
    )
    .unwrap();
    let mut source = source(&request);
    source.script.chain.seed = Some(u64::MAX);
    let mut changed = model();
    changed.defaults.default_width = 1344;
    changed.defaults.default_fps = Some(30);
    changed.defaults.default_steps = 99;
    let next = continue_request(&source, &changed, &limits(), "Continue", None).unwrap();
    assert_eq!(
        (next.width, next.height, next.fps, next.steps),
        (768, 512, 24, 9)
    );
    assert_eq!(next.seed, None);
    assert_eq!(
        next.continue_from,
        Some(ChainContinuation {
            job_id: "original-job".into(),
            finalize_take: 2
        })
    );
    assert!(next.stages[0].source_image.is_none());
    source.summary.state = ChainJobState::Running;
    assert!(continue_request(&source, &changed, &limits(), "Continue", None).is_err());
}
#[test]
fn scrubbed_reference_authority_is_never_dispatched() {
    let mut request =
        legacy_fixture_request(&model(), &limits(), "First", "Next", None, None, None, None)
            .unwrap();
    request.stages[1].generation_references = Some(vec![GenerationReference::Image {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: GenerationReferenceProvenance::default(),
        mime_type: "image/png".into(),
        width: 16,
        height: 16,
    }]);
    assert!(continue_request(&source(&request), &model(), &limits(), "Next", None).is_err());
}
#[test]
fn quota_refunds_preflight_but_preserves_ambiguous_dispatch_charge() {
    let state = crate::state::BotState::new(
        mold_core::MoldClient::new("http://fixture"),
        crate::state::BotConfig {
            cooldown_seconds: 0,
            allowed_roles: crate::access::AllowedRoles::parse(None),
            daily_quota: Some(1),
        },
    );
    state.quotas.consume(7, Some(1)).unwrap();
    drop(GenerationCharge::new(&state, 7));
    assert_eq!(state.quotas.usage(7), 0);
    state.quotas.consume(7, Some(1)).unwrap();
    let mut charge = GenerationCharge::new(&state, 7);
    charge.attempting_admission();
    drop(charge);
    assert_eq!(state.quotas.usage(7), 1);
}

#[test]
fn job_ids_are_opaque_bounded_identifiers_never_url_paths() {
    for invalid in [
        "",
        "../job",
        "..",
        "job/next",
        "job?x=1",
        "job#fragment",
        "job name",
        "job\\next",
    ] {
        assert!(validate_job_id(invalid).is_err(), "{invalid:?}");
    }
    assert!(validate_job_id(&"a".repeat(129)).is_err());
    assert!(validate_job_id("47c19f75-4414-4971-b8f3-a9b1826bb129").is_ok());
    assert!(validate_job_id("legacy_job-1").is_ok());
}

#[test]
fn continued_source_fit_needs_affirmative_host_support() {
    let request =
        legacy_fixture_request(&model(), &limits(), "First", "Next", None, None, None, None)
            .unwrap();
    let mut source = source(&request);
    source.script.chain.source_fit = Some(json!({"mode":"contain","alignX":0.5,"alignY":0.5}));
    assert!(continue_request(&source, &model(), &limits(), "Next", None).is_err());
    let supported: VideoLimits = serde_json::from_value(json!({"model":model().info.name,"supports_sequence":true,
        "frames_per_clip_cap":345,"frames_per_clip_recommended":209,"supports_audio":true,
        "authoring":{"motion_tail_frames":1,"min_frames":107,"frame_step":17,"frame_offset":5,"source_fit_supported":true}})).unwrap();
    let next = continue_request(&source, &model(), &supported, "Next", None).unwrap();
    assert_eq!(next.source_fit, source.script.chain.source_fit);
}

fn tail_for(model: &ModelInfoExtended, limits: &VideoLimits) -> u32 {
    limits
        .authoring
        .as_ref()
        .map(|value| value.motion_tail_frames)
        .unwrap_or_else(|| {
            mold_core::validation::chain_motion_tail_frames_for_model(
                &model.info.family,
                &model.info.name,
                model.source_image,
                mold_core::chain::DEFAULT_MOTION_TAIL_FRAMES,
            )
        })
}

#[allow(clippy::too_many_arguments)]
fn legacy_fixture_request(
    model: &ModelInfoExtended,
    limits: &VideoLimits,
    prompt: &str,
    next_prompt: &str,
    length: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    image: Option<Vec<u8>>,
) -> Result<ChainRequest> {
    ensure!(
        limits.supports_sequence && limits.model == model.info.name,
        "Choose a model that supports video clips on this machine."
    );
    ensure!(
        model.runtime_available != Some(false),
        "This model is not runnable on this machine."
    );
    ensure!(
        !prompt.trim().is_empty() && !next_prompt.trim().is_empty(),
        "Write both clip prompts."
    );
    let fps = model.defaults.default_fps.unwrap_or(24).max(1);
    let tail = tail_for(model, limits);
    let frames = clip_frames(
        model,
        limits,
        fps,
        length,
        limits.frames_per_clip_recommended,
        tail,
    )?;
    let mut request: ChainRequest = serde_json::from_value(serde_json::json!({
        "model":model.info.name,"width":width.unwrap_or(model.defaults.default_width),
        "height":height.unwrap_or(model.defaults.default_height),"fps":fps,
        "steps":model.defaults.default_steps,"guidance":model.defaults.default_guidance,
        "motion_tail_frames":tail,"enable_audio":limits.supports_audio,
        "output_format":"mp4","output_mode":"sequence",
        "stages":[{"prompt":prompt,"frames":frames},{"prompt":next_prompt,"frames":frames}]
    }))?;
    let task = mold_core::minimax_h3::task_for_model(&model.info.name);
    if task == Some(mold_core::minimax_h3::Task::Ref2va) {
        let bytes =
            image.context("This reference model needs an image attachment, used in both clips.")?;
        let format = image::guess_format(&bytes)?;
        let (width, height) = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()?
            .into_dimensions()?;
        ensure!(
            matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg),
            "Use a PNG or JPEG reference image."
        );
        ensure!(
            bytes.len() <= 16 * 1024 * 1024,
            "The repeated image must fit within the 32 MiB part budget."
        );
        let reference = GenerationReference::Image {
            media: GenerationReferenceAuthority::Inline { data: bytes },
            provenance: GenerationReferenceProvenance::default(),
            mime_type: if format == image::ImageFormat::Png {
                "image/png"
            } else {
                "image/jpeg"
            }
            .into(),
            width,
            height,
        };
        for stage in &mut request.stages {
            stage.generation_references = Some(vec![reference.clone()]);
        }
    } else {
        let contract = mold_core::validation::source_image_capability_for_engine(
            Some(&model.info.family),
            model.source_image,
        );
        ensure!(
            !(contract == Some(SourceImageCapability::Unsupported) && image.is_some()),
            "This model does not accept an opening image."
        );
        ensure!(
            !(contract == Some(SourceImageCapability::Required) && image.is_none()),
            "This model needs an opening image."
        );
        request.stages[0].source_image = image;
    }
    Ok(request)
}
