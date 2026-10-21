use super::*;
fn model() -> mold_core::ModelInfoExtended {
    serde_json::from_value(serde_json::json!({"name":"future-video:installed","family":"future-video","size_gb":8,"is_loaded":false,"hf_repo":"fixture","downloaded":true,"default_steps":2,"default_guidance":1.0,"default_width":768,"default_height":512,"description":"fixture","default_frames":107,"default_fps":24,"source_image":"optional","supports_sequence":false,"supports_audio":true})).unwrap()
}
#[test]
fn one_prompt_defaults_to_the_short_recipe_without_local_chain_or_model_allowlists() {
    let model = model();
    let request =
        make_parent_request(&model, r"literal\night", None, None, None, None, None, None).unwrap();
    assert_eq!(request.duration_ms, 4458);
    assert!(request.scenes.is_none());
    assert_eq!(request.generation.prompt, r"literal\night");
    assert_eq!(request.generation.model, model.info.name);
    assert_eq!(
        request.generation.output_format,
        Some(mold_core::OutputFormat::Mp4)
    );
    assert_eq!(request.generation.batch_size, 1);
}
#[test]
fn requested_duration_is_exact_and_not_limited_to_one_generation() {
    assert_eq!(parse_seconds("18446744073709551.615").unwrap(), u64::MAX);
    for invalid in [
        "0",
        "NaN",
        "-1",
        "1.",
        "1.0001",
        "18446744073709551.616",
        "1e5",
    ] {
        assert!(parse_seconds(invalid).is_err(), "{invalid}");
    }
    let request = make_parent_request(
        &model(),
        "First",
        Some("Second"),
        Some("30.001"),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let scenes = request.scenes.unwrap();
    assert_eq!(scenes.len(), 2);
    assert_eq!(scenes.iter().map(|s| s.duration_ms).sum::<u64>(), 30001);
    assert_eq!(scenes[0].prompt, "First");
    assert_eq!(scenes[1].prompt, "Second");
}
#[test]
fn timing_uses_the_selected_recipe_for_request_and_added_duration() {
    let mut selected = model();
    let manifest = mold_core::manifest::find_manifest("ltx-2-19b-distilled:fp8").unwrap();
    let mut profile = mold_core::generation_profile::generation_profile_for_manifest(manifest);
    for recipe in &mut profile.recipes {
        recipe.defaults.frames = Some(125);
        recipe.defaults.fps = Some(25);
    }
    selected.generation_profile = Some(profile);
    let make = make_parent_request(&selected, "New", None, None, None, None, None, None).unwrap();
    assert_eq!(
        (
            make.duration_ms,
            make.generation.frames,
            make.generation.fps
        ),
        (5000, Some(125), Some(25))
    );
    let source:mold_core::GalleryImage=serde_json::from_value(serde_json::json!({
        "filename":"original.mp4","timestamp":1,"format":"mp4",
        "metadata":{"model":"future-video:installed","prompt":"old","width":768,"height":512,"steps":2,"guidance":1.0,"seed":1,"fps":24,"frames":10000,"version":"test"}
    })).unwrap();
    let extended = extension_request(&source, "original.mp4", "Added", None, &selected).unwrap();
    assert_eq!(extended.duration_ms, 5208);
    assert_eq!(extended.generation.fps, Some(24));
}
#[test]
fn receipt_binds_the_exact_job_and_installation_without_a_bot_memory_map() {
    let id = "01234567-89ab-4cde-8123-456789abcdef";
    let instance = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
    let receipt = video_receipt(id, instance).unwrap();
    assert_eq!(receipt_parts(&receipt).unwrap(), (id, instance));
    for bad in [
        id,
        "../job@machine",
        "x@y@z",
        "",
        "01234567-89ab-4cde-8123-456789abcdef@invalid",
    ] {
        assert!(receipt_parts(bad).is_err());
    }
    assert!(validate_receipt_owner(&receipt, "replacement").is_err());
    assert!(validate_receipt_owner(&receipt, instance).is_ok());
}
#[test]
fn extension_uses_exact_live_source_original_recipe_and_added_time_only() {
    let source:mold_core::GalleryImage=serde_json::from_value(serde_json::json!({
            "filename":"original.mp4","timestamp":1,"format":"mp4",
            "metadata":{"model":"future-video:installed","prompt":"old","width":768,"height":512,"steps":2,"guidance":1.0,"seed":18446744073709551615u64,"fps":24,"frames":10000,"version":"test"}
        })).unwrap();
    let request =
        extension_request(&source, "original.mp4", r"new\night", Some("30"), &model()).unwrap();
    assert_eq!(request.duration_ms, 30000);
    assert_eq!(request.generation.prompt, r"new\night");
    assert!(request.generation.frames.is_none());
    assert!(request.generation.seed.is_none());
    assert_eq!(request.source.unwrap().filename, "original.mp4");
    assert!(extension_request(&source, "other.mp4", "new", None, &model()).is_err());
    assert!(extension_request(&source, "original.mp4", " ", None, &model()).is_err());
    let mut trashed = source.clone();
    trashed.trashed_at = Some(1);
    assert!(extension_request(&trashed, "original.mp4", "new", None, &model()).is_err());
}
#[test]
fn displayed_receipt_stays_copyable_after_message_sanitization() {
    let receipt = "01234567-89ab-4cde-8123-456789abcdef@aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
    let message = receipt_message(&format!("Video ID: {receipt}\n@everyone"));
    assert!(message.contains(receipt));
    let mentions =
        serde_json::to_value(receipt_reply("@everyone <@123>").allowed_mentions.unwrap()).unwrap();
    assert_eq!(mentions["parse"], serde_json::json!([]));
    assert_eq!(mentions["users"], serde_json::json!([]));
    assert_eq!(mentions["roles"], serde_json::json!([]));
    assert_eq!(mentions["replied_user"], false);

    assert!(receipt_parts(message.split_whitespace().nth(2).unwrap()).is_ok());
}
#[tokio::test]
async fn receipt_delivery_must_succeed_before_any_admission() {
    let called = std::cell::Cell::new(false);
    let result: anyhow::Result<()> =
        admit_after_receipt(async { anyhow::bail!("Discord unavailable") }, || async {
            called.set(true);
            Ok(())
        })
        .await;
    assert!(result.is_err());
    assert!(!called.get());
    admit_after_receipt(async { Ok(()) }, || async {
        called.set(true);
        Ok(())
    })
    .await
    .unwrap();
    assert!(called.get());
}
#[test]
fn positive_capability_requires_literal_parent_semantics() {
    let mut caps = mold_core::ServerCapabilities::default();
    assert!(require_parent_capability(&caps, false).is_err());
    caps.video_jobs = Some(mold_core::video_job::VideoJobCapabilities {
        supported: true,
        literal_prompts: true,
        library_extend: false,
        source_transfer: false,
        restart: false,
    });
    caps.video_jobs.as_mut().unwrap().literal_prompts = false;
    assert!(require_parent_capability(&caps, false).is_err());
    caps.video_jobs.as_mut().unwrap().literal_prompts = true;
    assert!(require_parent_capability(&caps, false).is_ok());
    assert!(require_parent_capability(&caps, true).is_err());
}
