use mold_core::{Config, Scheduler};

/// Capabilities and defaults derived from a model's family.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Whether the model supports negative prompts (CFG-based models).
    pub supports_negative_prompt: bool,
    /// Whether the model supports scheduler selection.
    pub supports_scheduler: bool,
    /// Whether the model supports img2img.
    pub supports_img2img: bool,
    /// Whether the model accepts a source/reference image.
    pub supports_source_image: bool,
    /// Whether this exact model accepts ordered MiniMax H3 references.
    pub supports_references: bool,
    /// Whether the model uses a denoising strength control.
    pub supports_strength: bool,
    /// Whether the model supports a mask image.
    pub supports_mask: bool,
    /// Whether the model supports ControlNet.
    pub supports_controlnet: bool,
    /// Whether the model supports LoRA adapters.
    pub supports_lora: bool,
    /// Whether the model is a video model (supports frames/fps params).
    pub supports_video: bool,
    /// Whether the model can emit synchronized audio with video.
    pub supports_audio: bool,
    /// Whether synchronized audio is inherent and cannot be disabled.
    pub audio_required: bool,
    /// Whether the model supports LTX-2 latent spatial/temporal upscaling.
    pub supports_video_upscale: bool,
    /// Whether the model takes Wan's request-level flow shift (#782).
    pub supports_flow_shift: bool,
    /// Default scheduler for UNet-based models.
    pub default_scheduler: Option<Scheduler>,
}

/// Determine model capabilities from family name.
pub fn capabilities_for_family(family: &str) -> ModelCapabilities {
    if mold_core::minimax_h3::is_family(family) {
        // Static form-shape knowledge only. H3 remains hidden and every H3
        // identity remains server-gated, so recognizing aliases here cannot
        // make the family selectable or runnable.
        return ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: true,
            supports_audio: true,
            audio_required: true,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        };
    }
    match family {
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: true,
            supports_img2img: true,
            supports_source_image: true,
            supports_references: false,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: true,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sdxl" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: true,
            supports_img2img: true,
            supports_source_image: true,
            supports_references: false,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: Some(Scheduler::Ddim),
        },
        "sd3" | "sd3.5" | "stable-diffusion-3" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "wuerstchen" | "wuerstchen-v2" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "flux" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: true,
            supports_source_image: true,
            supports_references: false,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: false, // ControlNet only supported on SD1.5
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "flux2" | "flux.2" | "flux2-klein" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "z-image" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "qwen-image" | "qwen_image" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "qwen-image-edit" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: true,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "ltx-video" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: true,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        "ltx2" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: true,
            supports_audio: true,
            audio_required: false,
            supports_video_upscale: true,
            supports_flow_shift: false,
            default_scheduler: None,
        },
        // Wan differs from both LTX entries on three axes, and the catch-all
        // below gets every one of them wrong: it is the only video family that
        // *wants* a negative prompt (its checkpoints ship a tuned default and
        // CFG is live above guidance 1.0), it conditions on a single source
        // image, and it has no audio branch at all.
        "wan" => ModelCapabilities {
            supports_negative_prompt: true,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: true,
            supports_references: false,
            // Wan's image conditioning is a first-frame anchor, not a
            // strength-weighted blend, so a strength slider would imply a
            // control the engine does not read.
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: true,
            supports_video: true,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: true,
            default_scheduler: None,
        },
        _ => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: false,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_video: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            default_scheduler: None,
        },
    }
}

/// Refine family capabilities with the selected checkpoint's additive
/// `/api/models[].supports_audio` and `[].source_image` facts. Older servers
/// omit the fields, so `None` preserves the family-level LTX-2 capability and
/// the Wan name heuristic; a current server's explicit value hides a control
/// the checkpoint cannot honor, or keeps one it cannot run without.
pub fn capabilities_for_model(
    family: &str,
    model: &str,
    advertised_audio_support: Option<bool>,
    advertised_guidance: Option<mold_core::GuidanceCapabilities>,
    advertised_source_image: Option<mold_core::SourceImageCapability>,
) -> ModelCapabilities {
    let mut caps = capabilities_for_family(family);
    caps.supports_references = mold_core::minimax_h3::is_family(family)
        && mold_core::minimax_h3::task_for_model(model)
            == Some(mold_core::minimax_h3::Task::Ref2va);
    // H3 has no unconditional/CFG branch. Contradictory checkpoint metadata
    // may not expose an editor for conditioning the model cannot consume.
    if !mold_core::minimax_h3::is_family(family) {
        caps.supports_negative_prompt = advertised_guidance
            .unwrap_or_else(|| mold_core::GuidanceCapabilities::for_recipe(family, model, None))
            .supports_negative_prompt;
    }
    // Checkpoint metadata can narrow optional audio (LTX-2), but may never
    // weaken H3's inherent synchronized-audio contract.
    if advertised_audio_support == Some(false) && !caps.audio_required {
        caps.supports_audio = false;
    }
    if family == "wan" {
        caps.supports_source_image = wan_model_takes_source_image(model);
    }
    // The advertised contract is the server's own classification of this
    // checkpoint, so it outranks both the family default and the name
    // heuristic above. H3 stays out: its form shape is static and
    // compliance-gated, and contradictory metadata may not reshape it.
    if !mold_core::minimax_h3::is_family(family) {
        match advertised_source_image {
            Some(mold_core::SourceImageCapability::Unsupported) => {
                caps.supports_source_image = false;
            }
            // A required source is still a source: keep the row so the user
            // has somewhere to satisfy the contract.
            Some(
                mold_core::SourceImageCapability::Optional
                | mold_core::SourceImageCapability::Required,
            ) => {
                caps.supports_source_image = true;
            }
            None => {}
        }
    }
    caps
}

/// Reject a request the selected checkpoint's advertised source-image
/// contract (#772) already refuses, before it costs a queue slot, a UMT5
/// encode, and an expert load. Returns the server's own admission wording so
/// the message is identical wherever the rejection lands.
///
/// `None` — an older server, or a checkpoint the server could not classify —
/// enforces nothing: the engine remains the authority and its late error is no
/// worse than today's behavior. Only Wan checkpoints advertise a non-optional
/// contract, which is why both messages name the family.
///
/// `has_source` counts first/last-frame keyframes as well as a source image
/// (#779), matching admission: both carry source frames, so either satisfies
/// a required contract and either is refused by a T2V-only checkpoint. The
/// TUI Create form has no keyframe control, so its callers pass the source
/// image alone.
pub fn source_image_contract_error(
    capability: Option<mold_core::SourceImageCapability>,
    has_source: bool,
) -> Option<&'static str> {
    match capability {
        Some(mold_core::SourceImageCapability::Unsupported) if has_source => Some(
            "this Wan checkpoint is text-to-video only and does not accept a source image \
             or keyframes — remove them, or pick an I2V-capable checkpoint such as \
             wan22-ti2v-5b or wan22-i2v-a14b",
        ),
        Some(mold_core::SourceImageCapability::Required) if !has_source => Some(
            "this Wan I2V checkpoint needs a source image; supply one, or pick a \
             text-to-video checkpoint such as wan22-t2v-a14b",
        ),
        _ => None,
    }
}

/// Whether a Wan checkpoint reads a source image, from its name.
///
/// This is the older-server fallback only: a current server advertises the
/// contract per model in `/api/models[].source_image`, classified from the
/// checkpoint's own headers, and [`capabilities_for_model`] prefers it.
///
/// The family as a whole conditions on images, but individual checkpoints do
/// not: `WanEngine::build_image_conditioning` refuses one outright on a
/// text-to-video checkpoint ("this Wan checkpoint is text-to-video only"), and
/// *requires* one on a 36-channel image-to-video checkpoint. Advertising the
/// field family-wide offered an image on `wan21-t2v-1.3b` and
/// `wan22-t2v-a14b` that the engine then rejected at generate time.
///
/// `i2v` is the discriminator and it subsumes `ti2v`, so one test covers both
/// the 36-channel I2V checkpoints and TI2V-5B's latent-inpaint path. A name
/// matching neither is treated as text-to-video: the overwhelming majority of
/// community Wan fine-tunes are T2V, and withholding a control is recoverable
/// where offering a rejected one is not.
fn wan_model_takes_source_image(model: &str) -> bool {
    model.to_ascii_lowercase().contains("i2v")
}

/// Resolve the family string for a given model name using the config and manifest.
pub fn family_for_model(model_name: &str, config: &Config) -> String {
    if mold_core::minimax_h3::task_for_model(model_name).is_some() {
        return mold_core::minimax_h3::FAMILY.to_string();
    }
    let model_cfg = config.resolved_model_config(model_name);
    model_cfg
        .family
        .clone()
        .or_else(|| mold_core::manifest::find_manifest(model_name).map(|m| m.family.clone()))
        .unwrap_or_else(|| "flux".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_does_not_support_controlnet() {
        let caps = capabilities_for_family("flux");
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn h3_references_are_exposed_only_for_an_explicit_ref2va_identity() {
        let ref2va = capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::REF2VA_COMFY,
            None,
            None,
            None,
        );
        let fl2va = capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::FL2VA_COMFY,
            None,
            None,
            None,
        );
        assert!(ref2va.supports_references);
        assert!(!fl2va.supports_references);
        assert_eq!(
            family_for_model(mold_core::minimax_h3::REF2VA_COMFY, &Config::default()),
            mold_core::minimax_h3::FAMILY
        );
        assert!(!capabilities_for_family("flux").supports_references);
    }

    #[test]
    fn sd15_supports_controlnet() {
        let caps = capabilities_for_family("sd15");
        assert!(caps.supports_controlnet);
    }

    #[test]
    fn sd15_supports_negative_and_scheduler() {
        let caps = capabilities_for_family("sd15");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_scheduler);
        assert!(caps.default_scheduler.is_some());
    }

    #[test]
    fn flux_supports_lora_and_img2img() {
        let caps = capabilities_for_family("flux");
        assert!(caps.supports_lora);
        assert!(caps.supports_img2img);
    }

    #[test]
    fn lora_capabilities_match_server_gate() {
        for family in [
            "flux",
            "flux2",
            "ltx2",
            "sd15",
            "sd3",
            "sdxl",
            "qwen-image",
            "qwen-image-edit",
            "wan",
            "z-image",
        ] {
            assert!(
                capabilities_for_family(family).supports_lora,
                "{family} should expose LoRA controls",
            );
        }
        assert!(!capabilities_for_family("wuerstchen").supports_lora);
        assert!(!capabilities_for_family("ltx-video").supports_lora);
    }

    /// Wan is the TUI's only video family that keeps the negative prompt and
    /// takes a source image, and the only one with no audio at all. Without
    /// its own entry it fell to the catch-all, which hides Frames/FPS — so a
    /// wan model in the Create form offered no way to set a clip length.
    #[test]
    fn wan_shows_video_rows_without_ltx_only_controls() {
        let caps = capabilities_for_family("wan");
        assert!(caps.supports_video, "Frames/FPS rows are gated on this");
        assert!(caps.supports_lora);
        assert!(caps.supports_source_image);
        // The negative prompt is live: Wan ships a tuned default and CFG runs
        // whenever guidance exceeds 1.0.
        assert!(caps.supports_negative_prompt);

        // LTX-only rows must stay hidden. Audio in particular is not a
        // degraded control for wan — its checkpoints carry no audio VAE or
        // vocoder, and the server rejects the request before denoising.
        assert!(!caps.supports_audio);
        assert!(!caps.supports_video_upscale);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
        assert!(!caps.supports_mask);
        // First-frame anchoring, not a strength-weighted blend.
        assert!(!caps.supports_strength);

        // The per-model refinement must not resurrect audio for wan the way
        // an advertised `supports_audio` can for LTX-2.
        assert!(
            !capabilities_for_model("wan", "wan22-t2v-a14b:q5", Some(true), None, None)
                .supports_audio
        );
    }

    /// The family conditions on images; individual checkpoints do not. A
    /// text-to-video checkpoint offered the source-image field would have it
    /// rejected by `build_image_conditioning` at generate time, after the
    /// user picked a file. This is the older-server path, where the name is
    /// all the TUI has.
    #[test]
    fn wan_source_image_follows_the_selected_checkpoint() {
        for model in [
            "wan22-i2v-a14b:q5",
            "wan22-i2v-a14b:q8",
            // TI2V-5B conditions by pinning latent frame 0; `i2v` subsumes it.
            "wan22-ti2v-5b:fp16",
        ] {
            assert!(
                capabilities_for_model("wan", model, None, None, None).supports_source_image,
                "{model} takes a source image"
            );
        }

        for model in [
            "wan21-t2v-1.3b:bf16",
            "wan22-t2v-a14b:q5",
            // An unrecognized name defaults to text-to-video.
            "cv:12345",
        ] {
            assert!(
                !capabilities_for_model("wan", model, None, None, None).supports_source_image,
                "{model} is text-to-video; the engine rejects an image"
            );
        }

        // Other families are untouched by the wan-specific narrowing.
        assert!(
            capabilities_for_model("sdxl", "sdxl:fp16", None, None, None).supports_source_image
        );
    }

    /// A current server classifies the checkpoint from its own headers, which
    /// is the only way an installed `cv:`/`hf:` I2V fine-tune with an
    /// unhelpful name gets its Source row — and the only way a T2V fine-tune
    /// named `...i2v-style...` loses one it cannot use.
    #[test]
    fn advertised_source_image_contract_outranks_the_name_heuristic() {
        use mold_core::SourceImageCapability::{Optional, Required, Unsupported};

        for capability in [Optional, Required] {
            assert!(
                capabilities_for_model("wan", "cv:12345", None, None, Some(capability))
                    .supports_source_image,
                "{capability:?} keeps the Source row on a name the heuristic reads as T2V"
            );
        }
        assert!(
            !capabilities_for_model(
                "wan",
                "some-i2v-flavored-t2v",
                None,
                None,
                Some(Unsupported)
            )
            .supports_source_image
        );

        // H3's static, compliance-gated form shape is never reshaped by
        // checkpoint metadata.
        assert!(
            !capabilities_for_model(
                mold_core::minimax_h3::FAMILY,
                mold_core::minimax_h3::FL2VA_COMFY,
                None,
                None,
                Some(Required),
            )
            .supports_source_image
        );
    }

    /// The pre-dispatch contract check. Identical table in `mold-ai` and
    /// `mold-ai-discord`; all three must agree with the server's admission
    /// gate, whose wording they reuse verbatim.
    #[test]
    fn source_image_contract_rejects_exactly_what_admission_rejects() {
        use mold_core::SourceImageCapability::{Optional, Required, Unsupported};

        for (capability, has_source, rejected) in [
            // Unknown contract — older server, or a checkpoint the server
            // could not classify — enforces nothing.
            (None, false, false),
            (None, true, false),
            (Some(Optional), false, false),
            (Some(Optional), true, false),
            (Some(Unsupported), false, false),
            (Some(Unsupported), true, true),
            (Some(Required), false, true),
            (Some(Required), true, false),
        ] {
            assert_eq!(
                source_image_contract_error(capability, has_source).is_some(),
                rejected,
                "{capability:?} with has_source={has_source}"
            );
        }

        assert!(source_image_contract_error(Some(Unsupported), true)
            .is_some_and(|message| message.contains("text-to-video only")));
        assert!(source_image_contract_error(Some(Required), false)
            .is_some_and(|message| message.contains("needs a source image")));
    }

    #[test]
    fn zimage_supports_lora_without_cfg_controls() {
        let caps = capabilities_for_family("z-image");
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_img2img);
        assert!(!caps.supports_controlnet);
        assert!(caps.supports_lora);
    }

    #[test]
    fn flux2_supports_lora_without_cfg_controls() {
        let caps = capabilities_for_family("flux2");
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_img2img);
        assert!(!caps.supports_controlnet);
        assert!(caps.supports_lora);
    }

    #[test]
    fn checkpoint_audio_fact_narrows_optional_audio_but_not_mandatory_h3_audio() {
        assert!(capabilities_for_model("ltx2", "ltx-2-dev", None, None, None).supports_audio);
        assert!(capabilities_for_model("ltx2", "ltx-2-dev", Some(true), None, None).supports_audio);
        assert!(
            !capabilities_for_model("ltx2", "ltx-2-dev", Some(false), None, None).supports_audio
        );
        assert!(
            !capabilities_for_model("ltx-video", "ltx-video-dev", Some(true), None, None)
                .supports_audio
        );
        let mandatory_h3 = capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::FL2VA_COMFY,
            Some(false),
            None,
            None,
        );
        assert!(mandatory_h3.supports_audio);
        assert!(mandatory_h3.audio_required);
    }

    #[test]
    fn ltx_negative_prompt_support_tracks_the_checkpoint_recipe() {
        assert!(
            capabilities_for_model("ltx2", "ltx-2.3-22b-dev:fp8", None, None, None)
                .supports_negative_prompt
        );
        assert!(
            !capabilities_for_model("ltx2", "ltx-2.3-22b-distilled:fp8", None, None, None)
                .supports_negative_prompt
        );
        assert!(
            capabilities_for_model(
                "ltx-video",
                "ltx-video-0.9.8-13b-dev:bf16",
                None,
                None,
                None
            )
            .supports_negative_prompt
        );
        assert!(
            !capabilities_for_model(
                "ltx2",
                "hf:opaque/checkpoint",
                None,
                Some(mold_core::GuidanceCapabilities::FIXED_ONE),
                None,
            )
            .supports_negative_prompt
        );
    }

    #[test]
    fn unknown_family_defaults_to_minimal() {
        let caps = capabilities_for_family("unknown-model");
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn sdxl_supports_negative_and_scheduler_but_not_controlnet() {
        let caps = capabilities_for_family("sdxl");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
        assert!(caps.supports_lora);
    }

    #[test]
    fn qwen_image_supports_negative_only() {
        let caps = capabilities_for_family("qwen-image");
        assert!(caps.supports_negative_prompt);
        assert!(!caps.supports_scheduler);
        assert!(!caps.supports_controlnet);
    }

    #[test]
    fn qwen_image_edit_supports_source_image_without_img2img_controls() {
        let caps = capabilities_for_family("qwen-image-edit");
        assert!(caps.supports_negative_prompt);
        assert!(caps.supports_source_image);
        assert!(!caps.supports_strength);
        assert!(!caps.supports_mask);
    }

    #[test]
    fn ltx_video_supports_video() {
        let caps = capabilities_for_family("ltx-video");
        assert!(caps.supports_video);
        assert!(!caps.supports_negative_prompt);
    }

    #[test]
    fn ltx2_supports_video() {
        let caps = capabilities_for_family("ltx2");
        assert!(caps.supports_video);
        assert!(!caps.supports_negative_prompt);
    }

    #[test]
    fn h3_exposes_only_its_static_synchronized_av_fields() {
        for family in [mold_core::minimax_h3::FAMILY, "minimax_h3", "minimaxh3"] {
            let caps = capabilities_for_family(family);
            assert!(caps.supports_video);
            assert!(caps.supports_audio);
            assert!(caps.audio_required);
            assert!(!caps.supports_negative_prompt);
            assert!(!caps.supports_scheduler);
            assert!(!caps.supports_img2img);
            assert!(!caps.supports_source_image);
            assert!(!caps.supports_strength);
            assert!(!caps.supports_mask);
            assert!(!caps.supports_controlnet);
            assert!(!caps.supports_lora);
            assert!(!caps.supports_video_upscale);
        }

        let contradictory = capabilities_for_model(
            "minimax_h3",
            mold_core::minimax_h3::FL2VA_COMFY,
            Some(false),
            Some(mold_core::GuidanceCapabilities::ADJUSTABLE_CFG),
            None,
        );
        assert!(contradictory.audio_required);
        assert!(contradictory.supports_audio);
        assert!(!contradictory.supports_negative_prompt);
    }
}
