//! Serde round-trip plus default tests for `DeviceRef`, `DevicePlacement`,
//! `AdvancedPlacement`. Kept in a sibling file because `types.rs` is already
//! 2100+ lines.
use super::{AdvancedPlacement, DevicePlacement, DeviceRef};

#[test]
fn device_ref_auto_default() {
    assert_eq!(DeviceRef::default(), DeviceRef::Auto);
}

#[test]
fn device_ref_auto_round_trip() {
    let json = serde_json::to_string(&DeviceRef::Auto).unwrap();
    assert_eq!(json, r#"{"kind":"auto"}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::Auto);
}

#[test]
fn device_ref_cpu_round_trip() {
    let json = serde_json::to_string(&DeviceRef::Cpu).unwrap();
    assert_eq!(json, r#"{"kind":"cpu"}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::Cpu);
}

#[test]
fn device_ref_gpu_round_trip() {
    let json = serde_json::to_string(&DeviceRef::gpu(2)).unwrap();
    assert_eq!(json, r#"{"kind":"gpu","ordinal":2}"#);
    let back: DeviceRef = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DeviceRef::gpu(2));
}

#[test]
fn device_placement_defaults_to_all_auto() {
    let dp = DevicePlacement::default();
    assert_eq!(dp.text_encoders, DeviceRef::Auto);
    assert!(dp.advanced.is_none());
}

#[test]
fn device_placement_serializes_tier1_only_without_advanced() {
    let dp = DevicePlacement {
        text_encoders: DeviceRef::Cpu,
        advanced: None,
    };
    let json = serde_json::to_value(&dp).unwrap();
    assert_eq!(json["text_encoders"]["kind"], "cpu");
    assert!(json.get("advanced").is_none() || json["advanced"].is_null());
}

#[test]
fn device_placement_round_trip_with_advanced() {
    let dp = DevicePlacement {
        text_encoders: DeviceRef::gpu(0),
        advanced: Some(AdvancedPlacement {
            transformer: DeviceRef::gpu(1),
            vae: DeviceRef::Cpu,
            clip_l: Some(DeviceRef::Auto),
            clip_g: None,
            t5: Some(DeviceRef::gpu(0)),
            qwen: None,
        }),
    };
    let json = serde_json::to_string(&dp).unwrap();
    let back: DevicePlacement = serde_json::from_str(&json).unwrap();
    assert_eq!(back.text_encoders, DeviceRef::gpu(0));
    let adv = back.advanced.unwrap();
    assert_eq!(adv.transformer, DeviceRef::gpu(1));
    assert_eq!(adv.vae, DeviceRef::Cpu);
    assert_eq!(adv.clip_l, Some(DeviceRef::Auto));
    assert_eq!(adv.clip_g, None);
    assert_eq!(adv.t5, Some(DeviceRef::gpu(0)));
}

#[test]
fn advanced_placement_defaults_to_auto_pair() {
    let adv = AdvancedPlacement::default();
    assert_eq!(adv.transformer, DeviceRef::Auto);
    assert_eq!(adv.vae, DeviceRef::Auto);
    assert!(adv.clip_l.is_none());
    assert!(adv.clip_g.is_none());
    assert!(adv.t5.is_none());
    assert!(adv.qwen.is_none());
}

#[test]
fn generate_request_placement_round_trips() {
    use super::GenerateRequest;
    let req = GenerateRequest {
        prompt: "a cat".into(),
        negative_prompt: None,
        model: "flux-dev:q4".into(),
        width: 1024,
        height: 1024,
        steps: 20,
        guidance: 3.5,
        seed: Some(7),
        batch_size: 1,
        output_format: super::OutputFormat::Png,
        embed_metadata: None,
        scheduler: None,
        source_image: None,
        edit_images: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: Some(DevicePlacement {
            text_encoders: DeviceRef::Cpu,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                vae: DeviceRef::Auto,
                t5: Some(DeviceRef::Cpu),
                ..Default::default()
            }),
        }),
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: GenerateRequest = serde_json::from_str(&json).unwrap();
    let p = back.placement.unwrap();
    assert_eq!(p.text_encoders, DeviceRef::Cpu);
    let adv = p.advanced.unwrap();
    assert_eq!(adv.transformer, DeviceRef::gpu(1));
    assert_eq!(adv.t5, Some(DeviceRef::Cpu));
}

#[test]
fn generate_request_without_placement_is_none() {
    use super::GenerateRequest;
    let json = r#"{
        "prompt": "a cat",
        "model": "flux-dev:q4",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
        "batch_size": 1,
        "strength": 0.75
    }"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert!(req.placement.is_none());
}
