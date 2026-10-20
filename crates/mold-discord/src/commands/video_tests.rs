use super::*;
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
fn two_prompts_make_durable_audio_work_with_model_grid() {
    let request = make_request(
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
fn invalid_length_or_empty_prompt_cannot_take_gpu_work() {
    assert!(make_request(&model(), &limits(), "", "Next", None, None, None, None).is_err());
    assert!(make_request(
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
    assert!(make_request(
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
    for name in ["make", "continue", "status", "resume", "cancel"] {
        assert!(command.subcommands.iter().any(|sub| sub.name == name));
    }
    for name in ["continue", "status", "resume", "cancel"] {
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
    let request = make_request(
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
    let request = make_request(
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
        make_request(&model(), &limits(), "First", "Next", None, None, None, None).unwrap();
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
        make_request(&model(), &limits(), "First", "Next", None, None, None, None).unwrap();
    let mut source = source(&request);
    source.script.chain.source_fit = Some(json!({"mode":"contain","alignX":0.5,"alignY":0.5}));
    assert!(continue_request(&source, &model(), &limits(), "Next", None).is_err());
    let supported: VideoLimits = serde_json::from_value(json!({"model":model().info.name,"supports_sequence":true,
        "frames_per_clip_cap":345,"frames_per_clip_recommended":209,"supports_audio":true,
        "authoring":{"motion_tail_frames":1,"min_frames":107,"frame_step":17,"frame_offset":5,"source_fit_supported":true}})).unwrap();
    let next = continue_request(&source, &model(), &supported, "Next", None).unwrap();
    assert_eq!(next.source_fit, source.script.chain.source_fit);
}
