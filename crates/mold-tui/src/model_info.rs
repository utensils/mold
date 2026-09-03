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
    /// Whether this exact checkpoint accepts a PuLID face-identity reference.
    /// Derived only from the server's additive `/api/models[].supports_identity`
    /// — absent means false, because a build that cannot execute identity
    /// conditioning refuses the request rather than rendering a faceless print.
    pub supports_identity: bool,
    /// Whether the model is a video model (supports frames/fps params).
    pub supports_video: bool,
    /// Whether this exact downloaded checkpoint can infer clip duration when
    /// `GenerateRequest.frames` is absent. Family recognition cannot enable it.
    pub supports_duration_prediction: bool,
    /// Whether the model can emit synchronized audio with video.
    pub supports_audio: bool,
    /// Whether synchronized audio is inherent and cannot be disabled.
    pub audio_required: bool,
    /// Whether the model supports LTX-2 latent spatial/temporal upscaling.
    pub supports_video_upscale: bool,
    /// Whether the model takes Wan's request-level flow shift (#782).
    pub supports_flow_shift: bool,
    /// The 3-D controls the selected recipe accepts, copied verbatim from
    /// its generation profile (`capabilities.mesh`). `Some` is what shows the
    /// Octree / Iso threshold / Target faces rows and pins the output format
    /// to GLB; `None` means `GenerateRequest.mesh` is refused here. Never
    /// derived from the family name: the profile is the one authority, and a
    /// family-only catalog (no profile) offers no mesh rows at all rather
    /// than inventing bounds the server might not honour.
    pub mesh: Option<mold_core::MeshCapabilitiesProfile>,
    /// The ordered reference-image contract the selected recipe advertises
    /// (`capabilities.reference_images`), copied verbatim. `Some` is what
    /// shows the References row; `None` means this model has no reference
    /// protocol and `GenerateRequest.edit_images` is refused here. Distinct
    /// from `supports_references`, which is MiniMax H3's uploaded Ref2VA
    /// group — a different field on a different wire.
    pub reference_images: Option<mold_core::generation_profile::ReferenceImagesProfile>,
    /// Default scheduler for UNet-based models.
    pub default_scheduler: Option<Scheduler>,
}

impl ModelCapabilities {
    /// The reference contract the Create form may offer a **References row**
    /// for — the ONE gate every surface that draws, opens, or commits that
    /// row reads.
    ///
    /// Narrower than `reference_images` on purpose: a `primary_is_target`
    /// recipe (qwen-image-edit) advertises the contract, but its FIRST
    /// `edit_images` entry is the image being edited and that image reaches
    /// the request through the Source well — `backend::build_request` takes
    /// the `primary_is_target` arm and never reads
    /// `params.edit_image_paths`. A References row there is worse than
    /// missing: filling it ships nothing AND, because the relation is
    /// `Replaces`, parks the Source row, so the user loses the edit Target
    /// and Generate is refused. A target-first recipe therefore keeps
    /// exactly the Source-well flow it had before the row existed.
    ///
    /// The CAPABILITY is untouched — `reference_images` still carries the
    /// contract for anything that needs to reason about it (the img2img
    /// gate in [`capabilities_for_model`] reads it directly).
    pub fn reference_images_row(
        &self,
    ) -> Option<&mold_core::generation_profile::ReferenceImagesProfile> {
        self.reference_images
            .as_ref()
            .filter(|profile| !profile.primary_is_target)
    }

    /// Whether an attached reference group leaves no room for a source image
    /// on the selected recipe.
    ///
    /// Read from the recipe's `source_relation`, never from the family name.
    /// `Replaces` never reads `source_image` at all and `Exclusive` renders
    /// from one or the other, so both park the Source row while references
    /// are attached; the reserved `Combines` would keep both. Gated on
    /// [`Self::reference_images_row`] so a recipe with no References row has
    /// nothing that could park the Source well in the first place.
    pub fn reference_group_replaces_source(&self) -> bool {
        self.reference_images_row().is_some_and(|profile| {
            !matches!(
                profile.source_relation,
                mold_core::generation_profile::ReferenceSourceRelation::Combines
            )
        })
    }
}

/// Determine model capabilities from family name.
pub fn capabilities_for_family(family: &str) -> ModelCapabilities {
    if mold_core::minimax_h3::is_family(family) {
        // Static form-shape knowledge only. Compact weights may be downloaded,
        // but every H3 runtime identity remains server-gated, so recognizing
        // aliases here cannot make the family runnable.
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
            supports_identity: false,
            supports_video: true,
            supports_duration_prediction: false,
            supports_audio: true,
            audio_required: true,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
            default_scheduler: None,
        },
        // The family answer: every flux2-klein* tier's engine
        // (`crates/mold-inference/src/flux2/pipeline.rs` img2img path) and
        // generation profile support img2img/strength/mask. flux2-dev does
        // not, because its reference protocol REPLACES the source image
        // rather than running alongside it — `capabilities_for_model` turns
        // these four off there by reading the resolved `reference_images`
        // relation, not by name-testing "dev".
        "flux2" | "flux.2" | "flux2-klein" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: true,
            supports_source_image: true,
            supports_references: false,
            supports_strength: true,
            supports_mask: true,
            supports_controlnet: false,
            supports_lora: true,
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: true,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: true,
            supports_duration_prediction: false,
            supports_audio: true,
            audio_required: false,
            supports_video_upscale: true,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: true,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: true,
            mesh: None,
            reference_images: None,
            default_scheduler: None,
        },
        // A source image is REQUIRED here, not optional: it is the family's
        // only conditioning. Falling through to the all-false default below
        // would hide the Create form's Source image row entirely and leave the
        // TUI unable to submit a request the server would accept — the one
        // control this family cannot do without. Everything else genuinely is
        // false: no text encoder, so no negative prompt; no canvas; no
        // strength, mask, ControlNet, LoRA or scheduler.
        "hunyuan3d" | "hunyuan-3d" => ModelCapabilities {
            supports_negative_prompt: false,
            supports_scheduler: false,
            supports_img2img: false,
            supports_source_image: true,
            supports_references: false,
            supports_strength: false,
            supports_mask: false,
            supports_controlnet: false,
            supports_lora: false,
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
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
            supports_identity: false,
            supports_video: false,
            supports_duration_prediction: false,
            supports_audio: false,
            audio_required: false,
            supports_video_upscale: false,
            supports_flow_shift: false,
            mesh: None,
            reference_images: None,
            default_scheduler: None,
        },
    }
}

/// Refine family capabilities with the selected checkpoint's additive
/// `/api/models[].supports_audio`, `[].source_image`, and
/// `[].supports_identity` facts. Older servers
/// omit the fields, so `None` preserves the family-level LTX-2 capability and
/// the Wan name heuristic; a current server's explicit value hides a control
/// the checkpoint cannot honor, or keeps one it cannot run without.
///
/// The generation profile is layered on afterwards by
/// [`apply_recipe_capabilities`]; this function reads only `/api/models`
/// facts.
pub fn capabilities_for_model(
    family: &str,
    model: &str,
    advertised_audio_support: Option<bool>,
    advertised_guidance: Option<mold_core::GuidanceCapabilities>,
    advertised_source_image: Option<mold_core::SourceImageCapability>,
    advertised_identity: Option<bool>,
) -> ModelCapabilities {
    let mut caps = capabilities_for_family(family);
    // Identity is advertised-only. There is no family default and no name
    // heuristic: `supports_identity` is true exactly when this server both
    // links the PuLID contract and can execute it, so an absent field is
    // "no" rather than "unknown" — offering the control against a server
    // that would refuse it is worse than withholding it.
    caps.supports_identity = advertised_identity == Some(true);
    caps.supports_references = mold_core::minimax_h3::is_family(family)
        && mold_core::minimax_h3::task_for_model(model)
            == Some(mold_core::minimax_h3::Task::Ref2va);
    // The static fallback for a host that advertises no profile at all.
    // `apply_recipe_capabilities` replaces it with the selected recipe's own
    // block when one is available — a remote host is the authority on what
    // it will accept — so this answers only the question "what does a
    // PRE-CONTRACT host honour", and it is deliberately narrower than the
    // core function's full answer.
    caps.reference_images = statically_visible_reference_images(
        mold_core::generation_profile::reference_images_for_recipe(family, model),
    );
    // A live reference protocol whose relation is `Replaces` never reads
    // `source_image` — admission refuses it outright ("... uses edit_images
    // instead of source_image") — so the Source row and its img2img/
    // strength/mask siblings describe a path this checkpoint cannot take.
    // Read the resolved relation, never a name test on "dev", so flux2-dev
    // loses all four while flux2-klein's `Exclusive` protocol (renders from
    // ONE of source image or references, never both) keeps them. Skip a
    // `primary_is_target` protocol (qwen-image-edit): there the first
    // reference IS the source/target image, so the Source row is how that
    // image reaches the request rather than a control this checkpoint lacks.
    if let Some(profile) = caps.reference_images.as_ref() {
        if profile.source_relation
            == mold_core::generation_profile::ReferenceSourceRelation::Replaces
            && !profile.primary_is_target
        {
            caps.supports_img2img = false;
            caps.supports_source_image = false;
            caps.supports_strength = false;
            caps.supports_mask = false;
        }
    }
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

/// Keep a reference block only while it describes a protocol this model
/// actually has. A `Hidden` block is the contract's way of saying "no
/// reference editing here, and here is the sentence to show instead"; the
/// form has nowhere to put that sentence, so it becomes `None` and the row
/// simply does not exist.
fn visible_reference_images(
    profile: mold_core::generation_profile::ReferenceImagesProfile,
) -> Option<mold_core::generation_profile::ReferenceImagesProfile> {
    (profile.mode != mold_core::ControlMode::Hidden).then_some(profile)
}

/// The same answer, narrowed to what a host that predates the
/// `reference_images` contract will actually honour.
///
/// Reference editing is only OLDER than the contract where the engine
/// already took `edit_images` before the block existed: FLUX.2 [dev] and
/// qwen-image-edit. FLUX.2 [klein]'s reference protocol shipped WITH the
/// contract, so a host old enough to advertise no profile is old enough to
/// refuse Klein references at admission — offering the row there is a
/// guaranteed failed render. `Exclusive` is exactly that new protocol, so
/// the static seed withholds it and waits for a recipe to say otherwise;
/// `apply_recipe_capabilities` restores the full answer the moment the host
/// carries the block, which every current server does.
///
/// The browser's legacy fallback draws the same line
/// (`studio/lib/legacyRecipeRules.ts`, `legacyReferenceImages`); the two must
/// not disagree about which models a pre-contract host supports.
fn statically_visible_reference_images(
    profile: mold_core::generation_profile::ReferenceImagesProfile,
) -> Option<mold_core::generation_profile::ReferenceImagesProfile> {
    visible_reference_images(profile).filter(|profile| {
        profile.source_relation != mold_core::generation_profile::ReferenceSourceRelation::Exclusive
    })
}

/// Layer the resolved generation profile's capability block over
/// [`capabilities_for_model`]'s answer. Applied LAST, so the profile outranks
/// every family and guidance fallback.
///
/// The profile is the single authority for the 3-D controls: a `mesh` block
/// is copied onto the form verbatim, and on such a recipe the strength, mask,
/// and negative-prompt rows follow `supports_strength`, `mask.mode`, and
/// `negative_prompt.mode` rather than the family arm — exactly what
/// `validate_request_against_recipe` enforces, so the form can never offer a
/// knob admission refuses. The three row gates are read only on a recipe
/// that carries the mesh block, because such a recipe is new enough to have
/// authored all of them: `supports_strength` defaults to `false` on an older
/// server's profile, and reading that `false` for SD img2img would hide a
/// row that works. `None` (no profile at all) changes nothing.
pub fn apply_recipe_capabilities(
    caps: &mut ModelCapabilities,
    recipe: Option<&mold_core::GenerationCapabilitiesProfile>,
) {
    caps.mesh = recipe.and_then(|recipe| recipe.mesh.clone());
    // `reference_images` is `Option` on the wire and its ABSENCE means an
    // older server — one whose flux2-dev and qwen-image-edit references still
    // work. So a profile that does not carry the block leaves
    // `capabilities_for_model`'s static answer standing; only a profile that
    // does carry one outranks it.
    if let Some(reference_images) = recipe.and_then(|recipe| recipe.reference_images.clone()) {
        caps.reference_images = visible_reference_images(reference_images);
    }
    if let Some(recipe) = recipe.filter(|recipe| recipe.mesh.is_some()) {
        caps.supports_strength = recipe.supports_strength;
        // Two capabilities in the family table, one signal in the profile:
        // `supports_img2img` means "a denoise-strength img2img pass", which
        // is exactly what `supports_strength` advertises. A mesh recipe
        // conditions on its source image without a denoise pass, so both
        // are false there; a profile that ever advertises strength on a
        // mesh recipe is describing that pass and turns both on.
        caps.supports_img2img = recipe.supports_strength;
        caps.supports_mask = recipe.mask.mode != mold_core::ControlMode::Hidden;
        caps.supports_negative_prompt =
            recipe.negative_prompt.mode != mold_core::ControlMode::Hidden;
    }
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
///
/// The wording follows `family`: the contract is advertised per checkpoint
/// for more than one family, and a 3-D model's "needs a source image" is a
/// different instruction (attach one on the Source row) from Wan's (pick a
/// text-to-video checkpoint instead).
pub fn source_image_contract_error(
    capability: Option<mold_core::SourceImageCapability>,
    has_source: bool,
    family: &str,
) -> Option<&'static str> {
    let wan = family == "wan";
    let mesh = family == mold_core::manifest::HUNYUAN3D_FAMILY;
    match capability {
        Some(mold_core::SourceImageCapability::Unsupported) if has_source && wan => Some(
            "this Wan checkpoint is text-to-video only and does not accept a source image \
             or keyframes — remove them, or pick an I2V-capable checkpoint such as \
             wan22-ti2v-5b or wan22-i2v-a14b",
        ),
        Some(mold_core::SourceImageCapability::Unsupported) if has_source => Some(
            "this model does not accept a source image — clear the Source row (x), or \
             pick a checkpoint that conditions on one",
        ),
        Some(mold_core::SourceImageCapability::Required) if !has_source && mesh => Some(
            "this 3-D model reconstructs a source image; attach one on the Source row \
             (Enter)",
        ),
        Some(mold_core::SourceImageCapability::Required) if !has_source && wan => Some(
            "this Wan I2V checkpoint needs a source image; supply one, or pick a \
             text-to-video checkpoint such as wan22-t2v-a14b",
        ),
        Some(mold_core::SourceImageCapability::Required) if !has_source => {
            Some("this model needs a source image; attach one on the Source row (Enter)")
        }
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
            None,
        );
        let fl2va = capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::FL2VA_COMFY,
            None,
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

        // The 3-D family needs its own arm, not the all-false default: the
        // source image is its ONLY conditioning, so hiding that row would
        // leave the TUI unable to submit a request the server accepts.
        let mesh = capabilities_for_family("hunyuan3d");
        assert!(mesh.supports_source_image, "source image is required");
        assert!(!mesh.supports_img2img, "there is no denoise strength here");
        assert!(!mesh.supports_negative_prompt, "no text encoder at all");
        assert!(!mesh.supports_video);
        assert!(!mesh.supports_lora);
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
            !capabilities_for_model("wan", "wan22-t2v-a14b:q5", Some(true), None, None, None)
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
                capabilities_for_model("wan", model, None, None, None, None).supports_source_image,
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
                !capabilities_for_model("wan", model, None, None, None, None).supports_source_image,
                "{model} is text-to-video; the engine rejects an image"
            );
        }

        // Other families are untouched by the wan-specific narrowing.
        assert!(
            capabilities_for_model("sdxl", "sdxl:fp16", None, None, None, None)
                .supports_source_image
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
                capabilities_for_model("wan", "cv:12345", None, None, Some(capability), None)
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
                Some(Unsupported),
                None,
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
                None,
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
            for family in ["wan", "hunyuan3d", "flux"] {
                assert_eq!(
                    source_image_contract_error(capability, has_source, family).is_some(),
                    rejected,
                    "{capability:?} with has_source={has_source} for {family}"
                );
            }
        }

        assert!(source_image_contract_error(Some(Unsupported), true, "wan")
            .is_some_and(|message| message.contains("text-to-video only")));
        assert!(source_image_contract_error(Some(Required), false, "wan")
            .is_some_and(|message| message.contains("needs a source image")));
    }

    /// The refusal names the family's own remedy: a mesh reconstructs its
    /// source image (attach one), Wan can swap to a T2V checkpoint, and any
    /// other family gets a neutral sentence with no Wan checkpoint names.
    #[test]
    fn source_image_contract_wording_follows_the_family() {
        use mold_core::SourceImageCapability::{Required, Unsupported};
        let mesh = source_image_contract_error(
            Some(Required),
            false,
            mold_core::manifest::HUNYUAN3D_FAMILY,
        )
        .unwrap();
        assert!(
            mesh.contains("3-D model reconstructs a source image"),
            "{mesh}"
        );
        assert!(mesh.contains("Source row"), "{mesh}");
        assert!(!mesh.contains("Wan"), "{mesh}");

        let wan = source_image_contract_error(Some(Required), false, "wan").unwrap();
        assert!(wan.contains("Wan I2V checkpoint"), "{wan}");
        assert!(wan.contains("wan22-t2v-a14b"), "{wan}");

        let other = source_image_contract_error(Some(Required), false, "ltx2").unwrap();
        assert!(other.contains("needs a source image"), "{other}");
        assert!(other.contains("Source row"), "{other}");
        assert!(!other.to_ascii_lowercase().contains("wan"), "{other}");

        let other_unsupported =
            source_image_contract_error(Some(Unsupported), true, "flux").unwrap();
        assert!(
            !other_unsupported.to_ascii_lowercase().contains("wan"),
            "{other_unsupported}"
        );
        assert!(
            other_unsupported.contains("Source row"),
            "{other_unsupported}"
        );
        let wan_unsupported = source_image_contract_error(Some(Unsupported), true, "wan").unwrap();
        assert!(
            wan_unsupported.contains("text-to-video only"),
            "{wan_unsupported}"
        );
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
        // The family answer is true: the engine
        // (`crates/mold-inference/src/flux2/pipeline.rs`) and every
        // flux2-klein* recipe support img2img/strength/mask.
        // `capabilities_for_model` turns them off for flux2-dev, whose
        // reference protocol REPLACES the source image, but the family-level
        // default must stay the Klein answer so a catalog-only lookup with
        // no resolved model still offers the row.
        assert!(caps.supports_img2img);
        assert!(caps.supports_source_image);
        assert!(caps.supports_strength);
        assert!(caps.supports_mask);
        assert!(!caps.supports_controlnet);
        assert!(caps.supports_lora);
    }

    #[test]
    fn checkpoint_audio_fact_narrows_optional_audio_but_not_mandatory_h3_audio() {
        assert!(capabilities_for_model("ltx2", "ltx-2-dev", None, None, None, None).supports_audio);
        assert!(
            capabilities_for_model("ltx2", "ltx-2-dev", Some(true), None, None, None)
                .supports_audio
        );
        assert!(
            !capabilities_for_model("ltx2", "ltx-2-dev", Some(false), None, None, None)
                .supports_audio
        );
        assert!(
            !capabilities_for_model("ltx-video", "ltx-video-dev", Some(true), None, None, None)
                .supports_audio
        );
        let mandatory_h3 = capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::FL2VA_COMFY,
            Some(false),
            None,
            None,
            None,
        );
        assert!(mandatory_h3.supports_audio);
        assert!(mandatory_h3.audio_required);
    }

    #[test]
    fn ltx_negative_prompt_support_tracks_the_checkpoint_recipe() {
        assert!(
            capabilities_for_model("ltx2", "ltx-2.3-22b-dev:fp8", None, None, None, None)
                .supports_negative_prompt
        );
        assert!(
            !capabilities_for_model("ltx2", "ltx-2.3-22b-distilled:fp8", None, None, None, None)
                .supports_negative_prompt
        );
        assert!(
            capabilities_for_model(
                "ltx-video",
                "ltx-video-0.9.8-13b-dev:bf16",
                None,
                None,
                None,
                None,
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
            None,
        );
        assert!(contradictory.audio_required);
        assert!(contradictory.supports_audio);
        assert!(!contradictory.supports_negative_prompt);
    }

    /// The local catalog's mesh recipe, exactly as `/api/models` serves it.
    fn mesh_recipe() -> mold_core::GenerationCapabilitiesProfile {
        let catalog = mold_core::build_model_catalog(&Config::default(), None, false);
        catalog
            .iter()
            .find(|entry| entry.name == mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL)
            .and_then(|entry| entry.generation_profile.as_ref())
            .and_then(|profile| profile.default_recipe())
            .map(|recipe| recipe.capabilities.clone())
            .expect("the built-in catalog carries the Hunyuan3D profile")
    }

    /// The recipe a CURRENT server hands back for one catalog model — the
    /// built-in generation profile, which always carries the
    /// `reference_images` block.
    fn recipe_for(model: &str) -> mold_core::GenerationCapabilitiesProfile {
        let catalog = mold_core::build_model_catalog(&Config::default(), None, false);
        catalog
            .iter()
            .find(|entry| entry.name == model)
            .and_then(|entry| entry.generation_profile.as_ref())
            .and_then(|profile| profile.default_recipe())
            .map(|recipe| recipe.capabilities.clone())
            .unwrap_or_else(|| panic!("the built-in catalog carries a profile for {model}"))
    }

    /// The profile, not the family name, decides the 3-D rows and the three
    /// gates a mesh recipe turns off. Reading the SAME profile with
    /// `supports_strength` flipped must flip the row, or the form would be
    /// carrying a family allowlist under another name.
    #[test]
    fn a_mesh_recipe_profile_drives_the_mesh_strength_mask_and_negative_rows() {
        let recipe = mesh_recipe();
        assert!(recipe.mesh.is_some(), "the fixture must be a mesh recipe");

        let mut caps = capabilities_for_family("hunyuan3d");
        apply_recipe_capabilities(&mut caps, Some(&recipe));
        let mesh = caps.mesh.as_ref().expect("mesh block copied verbatim");
        assert_eq!(
            mesh.octree_resolutions,
            mold_core::validation::MESH_OCTREE_RESOLUTIONS.to_vec()
        );
        assert_eq!(
            mesh.octree_default,
            mold_core::validation::MESH_DEFAULT_OCTREE_RESOLUTION
        );
        assert!(!caps.supports_strength);
        assert!(!caps.supports_mask);
        assert!(!caps.supports_negative_prompt);
        assert!(caps.supports_source_image, "the image is the conditioning");

        let mut flipped = recipe.clone();
        flipped.supports_strength = true;
        flipped.mask.mode = mold_core::ControlMode::Adjustable;
        flipped.negative_prompt.mode = mold_core::ControlMode::Adjustable;
        let mut caps = capabilities_for_family("hunyuan3d");
        apply_recipe_capabilities(&mut caps, Some(&flipped));
        assert!(
            caps.supports_strength,
            "the profile, not the family, answers"
        );
        assert!(caps.supports_mask);
        assert!(caps.supports_negative_prompt);
    }

    /// A raster profile never grows mesh rows, and an older server's profile
    /// (no mesh block, `supports_strength` defaulted to false) must not hide
    /// the SD strength row that works there.
    #[test]
    fn a_raster_profile_leaves_the_family_rows_alone() {
        let mut raster = mesh_recipe();
        raster.mesh = None;
        raster.supports_strength = false;
        let mut caps = capabilities_for_family("sd15");
        apply_recipe_capabilities(&mut caps, Some(&raster));
        assert!(caps.mesh.is_none());
        assert!(caps.supports_strength, "legacy predicate survives");
        assert!(caps.supports_mask);

        let mut no_profile = capabilities_for_family("sd15");
        apply_recipe_capabilities(&mut no_profile, None);
        assert!(no_profile.mesh.is_none());
        assert!(no_profile.supports_strength);
    }

    /// The References row is the recipe's `reference_images` block and
    /// nothing else. Klein and [dev] both have the protocol and differ only
    /// in what it does to the Source row; a family without it has no row at
    /// all; and H3's own uploaded group stays a separate capability.
    #[test]
    fn the_reference_row_follows_the_recipes_reference_block() {
        let mut klein = capabilities_for_model("flux2", "flux2-klein:bf16", None, None, None, None);
        // Klein's protocol shipped WITH the contract, so it takes a recipe
        // to reach the form — see
        // `klein_references_wait_for_a_server_that_advertises_them`.
        apply_recipe_capabilities(&mut klein, Some(&recipe_for("flux2-klein:bf16")));
        let profile = klein
            .reference_images
            .as_ref()
            .expect("Klein advertises the reference protocol");
        assert_eq!(
            profile.source_relation,
            mold_core::generation_profile::ReferenceSourceRelation::Exclusive
        );
        assert!(!profile.primary_is_target);
        assert!(
            crate::ui::create_form::section_fields(
                crate::ui::create_form::AdvSection::Source,
                &klein
            )
            .contains(&crate::app::ParamField::ReferenceImages),
            "the block is what puts the row on the form"
        );

        let dev = capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None);
        assert_eq!(
            dev.reference_images
                .as_ref()
                .expect("flux2-dev advertises it too")
                .source_relation,
            mold_core::generation_profile::ReferenceSourceRelation::Replaces
        );

        let qwen = capabilities_for_model(
            "qwen-image-edit",
            "qwen-image-edit-2511:q4",
            None,
            None,
            None,
            None,
        );
        assert!(
            qwen.reference_images
                .as_ref()
                .expect("the edit family advertises it")
                .primary_is_target,
            "the first image is the edit target"
        );

        let flux = capabilities_for_model("flux", "flux-dev:q4", None, None, None, None);
        assert!(
            flux.reference_images.is_none(),
            "a Hidden block is no row, not an empty one"
        );
        assert!(
            !flux.supports_references,
            "H3's uploaded group is a different capability"
        );
    }

    /// Klein's reference protocol is `Exclusive`: the engine
    /// (`crates/mold-inference/src/flux2/pipeline.rs` img2img path) and
    /// every flux2-klein* recipe support img2img/strength/mask beside the
    /// References row, so `capabilities_for_model` must keep all four on.
    #[test]
    fn klein_keeps_img2img_strength_and_mask_beside_references() {
        let mut klein = capabilities_for_model("flux2", "flux2-klein:bf16", None, None, None, None);
        apply_recipe_capabilities(&mut klein, Some(&recipe_for("flux2-klein:bf16")));
        assert!(klein.supports_img2img);
        assert!(klein.supports_source_image);
        assert!(klein.supports_strength);
        assert!(klein.supports_mask);
        assert_eq!(
            klein
                .reference_images
                .as_ref()
                .expect("Klein advertises the reference protocol")
                .source_relation,
            mold_core::generation_profile::ReferenceSourceRelation::Exclusive
        );
    }

    /// [dev]'s reference protocol REPLACES the source image: its
    /// `source_relation` is `Replaces`, admission refuses `source_image`
    /// outright ("flux2-dev uses edit_images instead of source_image"), and
    /// the generation profile advertises `supports_strength: false` and
    /// `mask.mode: "hidden"`. `capabilities_for_model` must read that
    /// resolved relation — never a name test on "dev" — and turn all four
    /// controls off.
    #[test]
    fn flux2_dev_hides_img2img_strength_and_mask_because_references_replace_the_source() {
        let dev = capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None);
        assert!(!dev.supports_img2img);
        assert!(!dev.supports_source_image);
        assert!(!dev.supports_strength);
        assert!(!dev.supports_mask);
        assert_eq!(
            dev.reference_images
                .as_ref()
                .expect("flux2-dev advertises the reference protocol too")
                .source_relation,
            mold_core::generation_profile::ReferenceSourceRelation::Replaces
        );
    }

    /// A profile that carries the block outranks the static answer; one that
    /// does NOT carry it is an older server whose reference support still
    /// works, so the static answer stands rather than being read as a
    /// refusal. Asked of flux2-dev, because it is the static answer that has
    /// to survive here and Klein deliberately has none.
    #[test]
    fn a_profile_without_the_block_leaves_the_static_reference_answer_standing() {
        let mut recipe = mesh_recipe();
        recipe.reference_images = None;
        let mut caps = capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None);
        apply_recipe_capabilities(&mut caps, Some(&recipe));
        assert!(
            caps.reference_images.is_some(),
            "absence is an older server, never a refusal"
        );

        recipe.reference_images = Some(mold_core::generation_profile::ReferenceImagesProfile {
            mode: mold_core::ControlMode::Hidden,
            required: false,
            max_count: Some(0),
            primary_is_target: false,
            source_relation: mold_core::generation_profile::ReferenceSourceRelation::Replaces,
            max_pixels_single: None,
            max_pixels_multi: None,
            reason: Some(
                mold_core::generation_profile::REFERENCE_IMAGES_UNSUPPORTED_REASON.to_string(),
            ),
        });
        let mut caps = capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None);
        apply_recipe_capabilities(&mut caps, Some(&recipe));
        assert!(
            caps.reference_images.is_none(),
            "a host that says it cannot take references is believed"
        );
    }

    /// A target-first recipe (qwen-image-edit) keeps EXACTLY the Source-well
    /// flow it had before the References row existed.
    ///
    /// Its first `edit_images` entry is the image being edited, and that
    /// image reaches the request through the Source row — `build_request`
    /// takes the `primary_is_target` arm and never reads
    /// `params.edit_image_paths`. A References row there is dead: filling it
    /// ships nothing AND clears the Source row (the relation is `Replaces`),
    /// so the user loses the edit Target and Generate is refused.
    #[test]
    fn a_target_first_recipe_keeps_its_source_flow_instead_of_a_references_row() {
        let qwen = capabilities_for_model(
            "qwen-image-edit",
            "qwen-image-edit-2511:q4",
            None,
            None,
            None,
            None,
        );
        assert!(
            qwen.reference_images
                .as_ref()
                .expect("the edit family still advertises the contract")
                .primary_is_target,
            "the capability is unchanged; only the ROW is withheld"
        );
        assert!(
            qwen.reference_images_row().is_none(),
            "no References row on a target-first recipe"
        );
        let qwen_fields = crate::ui::create_form::section_fields(
            crate::ui::create_form::AdvSection::Source,
            &qwen,
        );
        assert!(
            !qwen_fields.contains(&crate::app::ParamField::ReferenceImages),
            "the row would destroy the edit Target it cannot replace"
        );
        assert!(
            qwen_fields.contains(&crate::app::ParamField::SourceImage),
            "the Source well IS how the edit Target reaches the request"
        );
        assert!(
            !qwen.reference_group_replaces_source(),
            "with no row to fill, nothing parks the Source well"
        );

        // Every recipe whose references are a group of their own keeps the row.
        let dev = capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None);
        assert!(dev.reference_images_row().is_some());
        assert!(crate::ui::create_form::section_fields(
            crate::ui::create_form::AdvSection::Source,
            &dev
        )
        .contains(&crate::app::ParamField::ReferenceImages));

        let mut klein = capabilities_for_model("flux2", "flux2-klein:bf16", None, None, None, None);
        apply_recipe_capabilities(&mut klein, Some(&recipe_for("flux2-klein:bf16")));
        assert!(klein.reference_images_row().is_some());
        assert!(crate::ui::create_form::section_fields(
            crate::ui::create_form::AdvSection::Source,
            &klein
        )
        .contains(&crate::app::ParamField::ReferenceImages));
    }

    /// Klein's references are a NEW contract: a host that predates it has
    /// flux2-klein engines that refuse `edit_images` at admission. The
    /// static seed is what a TUI shows against such a host, so it must
    /// answer `None` for Klein — exactly as the browser's legacy fallback
    /// (`studio/lib/legacyRecipeRules.ts` `legacyReferenceImages`) does.
    /// flux2-dev and qwen-image-edit are the exception: their engines always
    /// took `edit_images`, so an older host honours those rows.
    #[test]
    fn klein_references_wait_for_a_server_that_advertises_them() {
        let klein = capabilities_for_model("flux2", "flux2-klein:bf16", None, None, None, None);
        assert!(
            klein.reference_images.is_none(),
            "no recipe means an older host, which refuses Klein references"
        );

        let mut current =
            capabilities_for_model("flux2", "flux2-klein:bf16", None, None, None, None);
        apply_recipe_capabilities(&mut current, Some(&recipe_for("flux2-klein:bf16")));
        assert_eq!(
            current
                .reference_images
                .as_ref()
                .expect("a recipe carrying the block is the authority")
                .source_relation,
            mold_core::generation_profile::ReferenceSourceRelation::Exclusive
        );

        // The two whose engines always accepted the group keep the static
        // answer, so an older host still offers their rows.
        assert!(
            capabilities_for_model("flux2", "flux2-dev:bf16", None, None, None, None)
                .reference_images
                .is_some()
        );
        assert!(capabilities_for_model(
            "qwen-image-edit",
            "qwen-image-edit-2511:q4",
            None,
            None,
            None,
            None
        )
        .reference_images
        .is_some());
    }
}
