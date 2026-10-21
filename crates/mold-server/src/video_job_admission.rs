//! Admission for durable duration parents. Public prompts retain literal JSON semantics.

use crate::routes::ApiError;
use mold_core::{
    chain::{ChainGenerationControls, ChainRequest},
    video_job::{VideoJobPlan, VideoJobRequest, VideoJoinMode},
    video_job_schedule::VideoJobSchedule,
    video_plan::ClipEnvelope,
};

#[cfg(test)]
fn plan_template(
    request: &VideoJobRequest,
    family: &str,
    source_image: Option<mold_core::SourceImageCapability>,
    source_frames: u64,
) -> Result<(VideoJobPlan, ClipEnvelope, ChainRequest), ApiError> {
    plan_template_with_defaults(request, family, source_image, source_frames, None)
}

fn plan_template_with_defaults(
    request: &VideoJobRequest,
    family: &str,
    source_image: Option<mold_core::SourceImageCapability>,
    source_frames: u64,
    recipe_default_frames: Option<u32>,
) -> Result<(VideoJobPlan, ClipEnvelope, ChainRequest), ApiError> {
    request.validate_shape().map_err(ApiError::validation)?;
    let generation = &request.generation;
    if generation.batch_size > 1 {
        return Err(ApiError::validation(
            "A duration parent produces one movie; submit each prepared batch sibling separately.",
        ));
    }
    if generation.audio_file.is_some()
        || generation.audio_file_path.is_some()
        || generation.source_video.is_some()
        || generation.source_video_path.is_some()
        || generation.extend_video.is_some()
        || generation.extend_video_path.is_some()
    {
        return Err(ApiError::validation("Conditioned audio/video needs an explicit timeline; use Library Extend for a published prefix."));
    }
    if generation.id_image.is_some()
        || generation.id_images.is_some()
        || generation.edit_images.is_some()
        || generation.mask_image.is_some()
        || generation.control_image.is_some()
        || generation.mesh.is_some()
    {
        return Err(ApiError::validation("Duration video accepts first/last frames and typed references; identity, mask, ControlNet and mesh inputs cannot be applied to this timeline."));
    }
    if generation.expand == Some(true)
        || generation.upscale_model.is_some()
        || generation.gif_preview
    {
        return Err(ApiError::validation("Prepare prompt expansion and post-generation upscaling separately before using duration video."));
    }
    if generation
        .output_format
        .is_some_and(|format| format != mold_core::OutputFormat::Mp4)
    {
        return Err(ApiError::validation(
            "Duration video publishes an MP4 movie; choose MP4 output.",
        ));
    }
    if generation.retake_range.is_some() || generation.extend_overlap_frames.is_some() {
        return Err(ApiError::validation("Duration video manages its own overlap and timeline; omit retake_range and extend_overlap_frames."));
    }
    if generation.save_to_gallery == Some(false) {
        return Err(ApiError::validation(
            "Duration video is saved to Library; enable Gallery output.",
        ));
    }
    let canonical = mold_core::manifest::resolve_model_name(&generation.model);
    let defaults =
        mold_core::manifest::find_manifest(&canonical).map(|manifest| &manifest.defaults);
    let fps = generation
        .fps
        .or_else(|| defaults.and_then(|defaults| defaults.fps))
        .unwrap_or(24);
    if fps == 0 || fps > 240 {
        return Err(ApiError::validation("fps must be between 1 and 240"));
    }
    // Use the installed recipe's ordinary single-render envelope. The family
    // ceiling can be much larger than its qualified default (H3: 345 vs 124).
    // generation.frames may be a restored movie's total and is never this cap.
    let limits = crate::chain_limits::compute_limits(
        &generation.model,
        family,
        "",
        recipe_default_frames.or_else(|| defaults.and_then(|defaults| defaults.frames)),
        Some(fps),
    );
    let clip_cap = limits
        .frames_per_clip_recommended
        .min(limits.frames_per_clip_cap);
    if !limits.supports_sequence {
        return Err(ApiError::validation(
            limits
                .sequence_unsupported_reason
                .unwrap_or_else(|| "This installed recipe cannot render video clips".into()),
        ));
    }
    let authoring =
        crate::chain_limits::authoring_capabilities(family, &generation.model, source_image);
    if request.source.is_some()
        && authoring.motion_tail_frames > 0
        && (generation.source_image.is_some()
            || generation
                .keyframes
                .as_ref()
                .is_some_and(|frames| frames.iter().any(|frame| frame.frame == 0)))
    {
        return Err(ApiError::validation("Library Extend continues from the video's visible ending. Remove the first-frame input; an optional closing frame can still guide the ending."));
    }
    let overlap = if authoring.motion_tail_frames == 0 {
        8.min(clip_cap.saturating_sub(1))
    } else {
        authoring.motion_tail_frames
    };
    if request.source.is_some()
        && authoring.motion_tail_frames == 0
        && source_frames > 0
        && source_frames < u64::from(overlap)
    {
        return Err(ApiError::validation("This video is shorter than the default blend. Choose a source with at least eight visible frames, or a recipe with continuous frame conditioning."));
    }
    let envelope = ClipEnvelope {
        maximum_frames: clip_cap,
        minimum_frames: authoring.min_frames,
        frame_step: authoring.frame_step,
        frame_offset: authoring.frame_offset,
        overlap_frames: overlap,
    };
    if authoring.motion_tail_frames == 0 && (request.source.is_none() || source_frames > 0) {
        if let Some(scenes) = &request.scenes {
            let mut elapsed_ms = 0u64;
            for scene in scenes.iter().take(scenes.len().saturating_sub(1)) {
                elapsed_ms += scene.duration_ms; // validate_shape checked the full sum.
                let visible = u128::from(source_frames)
                    + (u128::from(elapsed_ms) * u128::from(fps)).div_ceil(1000);
                if visible < u128::from(overlap) {
                    return Err(ApiError::validation("Each blended scene boundary needs at least eight visible frames before it. Lengthen the opening scene or use continuous frame conditioning."));
                }
            }
        }
    }
    let schedule = VideoJobSchedule::new(request, envelope, fps, source_frames)
        .map_err(|error| ApiError::validation(error.to_string()))?;
    let total_frames = source_frames
        .checked_add(schedule.added_frames())
        .ok_or_else(|| ApiError::validation("total frame count overflow"))?;
    let total_duration_ms =
        u64::try_from((u128::from(total_frames) * 1000).div_ceil(u128::from(fps)))
            .map_err(|_| ApiError::validation("duration exceeds its representation"))?;
    let plan = VideoJobPlan {
        validation_revision: String::new(),
        model: generation.model.clone(),
        width: generation.width,
        height: generation.height,
        fps,
        single_clip_max_frames: envelope.maximum_frames,
        overlap_frames: overlap,
        join_mode: if authoring.motion_tail_frames == 0 {
            VideoJoinMode::BlendedShots
        } else {
            VideoJoinMode::Continuous
        },
        clip_count: schedule.clip_count(),
        part_count: schedule.part_count(),
        added_frames: schedule.added_frames(),
        source_frames,
        total_frames,
        added_duration_ms: request.duration_ms,
        total_duration_ms,
        warnings: Vec::new(),
    };
    // Decode the ordinary chain shape without invoking legacy prompt escape
    // normalization. All original controls remain bound into this template.
    let mut template:ChainRequest=serde_json::from_value(serde_json::json!({"model":generation.model,"width":generation.width,"height":generation.height,"fps":fps,"steps":generation.steps,"guidance":generation.guidance,"motion_tail_frames":authoring.motion_tail_frames,"stages":[{"prompt":generation.prompt,"negative_prompt":generation.negative_prompt,"frames":envelope.maximum_frames}]})).map_err(|error|ApiError::internal(error.to_string()))?;
    template.generation_controls = Some(ChainGenerationControls::from_generation(generation));
    template.seed = generation.seed;
    template.strength = generation.strength;
    template.offload = generation.offload;
    template.placement = generation.placement.clone();
    template.source_fit = generation.source_fit.clone();
    template.enable_audio = generation.enable_audio;
    template.title = generation.title.clone();
    template.tags = generation.tags.clone();
    template.collection = generation.collection.clone();
    template.original_prompt = generation.original_prompt.clone();
    template.prompt_transform = generation.prompt_transform.clone();
    template.batch_id = generation.batch_id.clone();
    template.batch_index = generation.batch_index;
    template.batch_count = generation.batch_count;
    template.output_mode = Some(mold_core::GenerationOutputMode::OneShot);
    template.stages[0].source_image = generation.source_image.clone();
    template.stages[0].keyframes = generation.keyframes.clone();
    template.stages[0].generation_references = generation.references.clone();
    Ok((plan, envelope, template))
}

pub(crate) struct PreparedVideo {
    pub request: VideoJobRequest,
    pub template: ChainRequest,
    pub plan: VideoJobPlan,
    pub envelope: ClipEnvelope,
    pub frozen: Option<mold_core::chain_job::FrozenChainModel>,
    pub frozen_loras: Vec<crate::video_job_loras::FrozenVideoLora>,
    pub source: Option<(std::fs::File, crate::video_source::RetainedVideoSource)>,
    pub required_claims: Vec<crate::video_job_pending::RequiredInputClaim>,
}

/// A read-only proof is derived from exact authenticated pixels and current
/// installed recipe authority. Upload IDs never become the durable input.
pub(crate) async fn prepare(
    state: &crate::state::AppState,
    mut request: VideoJobRequest,
    headers: &axum::http::HeaderMap,
    authenticated: Option<&axum::Extension<crate::auth::ApiKeyAuthenticated>>,
    auth: Option<&axum::Extension<crate::auth::AuthState>>,
    planning_only: bool,
) -> Result<PreparedVideo, ApiError> {
    use sha2::{Digest, Sha256};
    request.validate_shape().map_err(ApiError::validation)?;
    validate_input_budget(&request.generation)?;
    let mut required_claims = Vec::new();
    if !planning_only && headers.contains_key(crate::gallery_source_media::REUSE_SESSION_HEADER) {
        required_claims.push(crate::video_job_pending::RequiredInputClaim {
            kind: "gallery_reuse".into(),
            request_scope_sha256: crate::gallery_source_media::request_sha256(&request.generation)?,
            credential_identity: crate::gallery_source_media::reuse_credential_identity(
                authenticated,
            ),
        });
    }
    let authority =
        crate::routes_chain::resolve_chain_model_authority(state, &request.generation.model)
            .await?;
    let family =
        crate::routes_chain::resolve_chain_family(&authority.config, &request.generation.model);
    mold_core::validate_generation_dimensions_for_model(
        &request.generation.model,
        request.generation.width,
        request.generation.height,
        Some(&family),
        mold_core::validation::ltx2_spatial_composition(&request.generation.model, None),
    )
    .map_err(ApiError::validation)?;
    let contract = crate::routes_chain::resolved_chain_source_image_contract(
        &authority.config,
        &request.generation.model,
        &family,
    );
    let recipe_default_frames = authority
        .config
        .resolved_model_config(&request.generation.model)
        .effective_frames();
    let fps = plan_template_with_defaults(&request, &family, contract, 0, recipe_default_frames)?
        .2
        .fps;
    let mut source = match &request.source {
        None => None,
        Some(source) => Some(match &source.retained_upload {
            Some(upload) => crate::video_source_upload::open(state, upload, source).await?,
            None if planning_only => {
                crate::video_source::open_preview(
                    state,
                    &source.filename,
                    source.expected_archive_identity_sha256.as_deref(),
                )
                .await?
            }
            None => {
                crate::video_source::open(
                    state,
                    &source.filename,
                    source.expected_archive_identity_sha256.as_deref(),
                )
                .await?
            }
        }),
    };
    let mut warnings = Vec::new();
    let source_frames = if let Some((file, binding)) = source.as_mut() {
        if planning_only {
            match (binding.metadata.frames, binding.metadata.fps) {
                (Some(frames), Some(source_fps)) if source_fps > 0 => u64::try_from(
                    (u128::from(frames) * u128::from(fps)).div_ceil(u128::from(source_fps)),
                )
                .map_err(|_| ApiError::validation("source duration overflow"))?,
                _ => {
                    warnings
                        .push("The source presentation will be verified before generation.".into());
                    0
                }
            }
        } else {
            inspect_source_frames(
                file.try_clone()
                    .map_err(|error| ApiError::internal(error.to_string()))?,
                fps,
            )
            .await?
        }
    } else {
        0
    };
    if !planning_only {
        if let Some((_, binding)) = source.as_ref() {
            inherit_source_references(
                state,
                binding,
                &mut request.generation,
                crate::gallery_source_media::private_media_authorized(auth, authenticated),
            )
            .await?;
        }
        crate::gallery_source_media::preview_hydrate_reuse_session(
            state,
            authenticated,
            auth,
            headers,
            &mut request.generation,
            32 * 1024 * 1024,
        )
        .await?;
        let identity = crate::reference_uploads::ReferenceIdentity::resolve(
            authenticated.map(|value| &value.0),
            auth.map(|value| &value.0),
            state.instance_id.as_str(),
        );
        if request
            .generation
            .references
            .iter()
            .flatten()
            .any(|reference| {
                matches!(
                    reference.media(),
                    mold_core::GenerationReferenceAuthority::Upload { .. }
                )
            })
        {
            required_claims.push(crate::video_job_pending::RequiredInputClaim {
                kind: "reference_upload".into(),
                request_scope_sha256: state.reference_uploads.scope_sha256(&request.generation)?,
                credential_identity: identity
                    .as_ref()
                    .ok_or_else(|| {
                        ApiError::validation("Reference upload ownership is unavailable")
                    })?
                    .identity_str(),
            });
        }
        let roots = state.config.read().await.resolved_media_roots();
        let staged = state
            .reference_uploads
            .preview_request(identity.as_ref(), &mut request.generation, &roots, None)
            .await?;
        if let Some(staged) = staged {
            hydrate_staged(&mut request.generation, &staged)?;
        }
        validate_input_budget(&request.generation)?;
    }
    if !planning_only {
        crate::routes::require_server_generation_request_activation(
            state,
            &request.generation,
            Some(&family),
        )
        .await?;
        warnings.extend(crate::routes::resolve_collection_reference(
            crate::video_jobs::database(state)?,
            &mut request.generation.collection,
        ));
        crate::routes::plan_builtin_ltx2_camera_controls(state, &request.generation).await?;
    }
    let (mut plan, envelope, mut template) = plan_template_with_defaults(
        &request,
        &family,
        contract,
        source_frames,
        recipe_default_frames,
    )?;
    plan.warnings = warnings;
    if !planning_only {
        // Validate one legal clip, retaining the original endpoint locator in
        // the logical request rather than treating a long slider as one clip.
        let mut validation = request.generation.clone();
        if let Some(first) = request.scenes.as_ref().and_then(|scenes| scenes.first()) {
            validation.prompt = first.prompt.clone();
        }
        validation.frames = Some(envelope.maximum_frames);
        validation.fps = Some(fps);
        validation.output_format = Some(mold_core::OutputFormat::Mp4);
        if let Some(keyframes) = validation.keyframes.as_mut() {
            let last = request
                .generation
                .frames
                .and_then(|frames| frames.checked_sub(1));
            for frame in keyframes {
                if frame.frame != 0 {
                    if Some(frame.frame) != last {
                        return Err(ApiError::validation("Duration video supports first and last frames; intermediate keyframes need an explicit scene boundary."));
                    }
                    frame.frame = envelope.maximum_frames - 1;
                }
            }
        }
        if let Some((file, _)) = source.as_ref() {
            if template.motion_tail_frames > 0 {
                validation.source_image = Some(
                    source_boundary_png(
                        file.try_clone()
                            .map_err(|error| ApiError::internal(error.to_string()))?,
                        plan.width,
                        plan.height,
                    )
                    .await?,
                );
                if let Some(keyframes) = validation.keyframes.as_mut() {
                    keyframes.retain(|frame| frame.frame != 0);
                }
            }
        }
        mold_core::validation::validate_generate_request_with_family(&validation, Some(&family))
            .map_err(ApiError::validation)?;
        let mut validation_template = template.clone();
        validation_template.stages[0].prompt = validation.prompt;
        validation_template.stages[0].source_image = validation.source_image;
        validation_template.stages[0].keyframes = validation.keyframes;
        crate::routes_chain::validate_and_normalize_chain_family(
            &authority.config,
            &mut validation_template,
        )?;
        template.motion_tail_frames = validation_template.motion_tail_frames;
        template.enable_audio = validation_template.enable_audio;
    }
    let frozen_loras = if planning_only {
        Vec::new()
    } else {
        crate::routes::materialize_chain_camera_controls(state, &authority.config, &template)
            .await?;
        crate::video_job_loras::capture(&authority.config, &request.generation)
            .map_err(|error| ApiError::validation(error.to_string()))?
    };
    let frozen = if planning_only {
        None
    } else {
        Some(crate::routes_chain::freeze_chain_model(
            authority,
            &request.generation.model,
        )?)
    };
    if !planning_only {
        let mut canonical = request.clone();
        canonical.validation_revision = None;
        let binding = source.as_ref().map(|(_, binding)| {
            (
                binding.authority_kind,
                &binding.archive_identity_sha256,
                &binding.checksum_sha256,
                binding.size_bytes,
            )
        });
        let bytes = serde_json::to_vec(&(
            state.instance_id.as_str(),
            canonical,
            &frozen,
            &frozen_loras,
            &plan,
            binding,
        ))
        .map_err(|error| ApiError::internal(error.to_string()))?;
        plan.validation_revision = format!("{:x}", Sha256::digest(bytes));
    }
    Ok(PreparedVideo {
        request,
        template,
        plan,
        envelope,
        frozen,
        frozen_loras,
        source,
        required_claims,
    })
}

#[cfg(feature = "mp4")]
async fn inspect_source_frames(file: std::fs::File, fps: u32) -> Result<u64, ApiError> {
    tokio::task::spawn_blocking(move || {
        let facts = mold_inference::video_assembly_frames::inspect(file, &mut || Ok(())).map_err(
            |error| {
                ApiError::validation(format!("Source presentation cannot be extended: {error:#}"))
            },
        )?;
        if facts.fps_numerator == 0 {
            return Err(ApiError::validation("Source frame clock is invalid"));
        }
        u64::try_from(
            (u128::from(facts.frames) * u128::from(fps) * u128::from(facts.fps_denominator))
                .div_ceil(u128::from(facts.fps_numerator)),
        )
        .map_err(|_| ApiError::validation("source duration overflow"))
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
}
#[cfg(not(feature = "mp4"))]
async fn inspect_source_frames(_: std::fs::File, _: u32) -> Result<u64, ApiError> {
    Err(ApiError::validation(
        "This server was built without MP4 support",
    ))
}

fn validate_input_budget(request: &mold_core::GenerateRequest) -> Result<(), ApiError> {
    let mut total = 0_u64;
    for bytes in request
        .source_image
        .iter()
        .chain(request.keyframes.iter().flatten().map(|frame| &frame.image))
        .chain(request.references.iter().flatten().filter_map(
            |reference| match reference.media() {
                mold_core::GenerationReferenceAuthority::Inline { data } => Some(data),
                _ => None,
            },
        ))
    {
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| ApiError::validation("Input size overflow"))?;
        if total > 32 * 1024 * 1024 {
            return Err(ApiError::validation(
                "Original video inputs exceed the 32 MiB portable media budget",
            ));
        }
    }
    Ok(())
}

fn hydrate_staged(
    request: &mut mold_core::GenerateRequest,
    staged: &crate::reference_uploads::StagedReferences,
) -> Result<(), ApiError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let references = request
        .references
        .as_mut()
        .ok_or_else(|| ApiError::internal("Resolved references are missing"))?;
    if references.len() != staged.entries().len() {
        return Err(ApiError::internal("Resolved reference inventory changed"));
    }
    let mut remaining = 32_u64 * 1024 * 1024;
    for (reference, entry) in references.iter_mut().zip(staged.entries()) {
        let file = mold_core::secure_file::open_regular_file_no_follow(&entry.path)
            .map_err(|_| ApiError::validation("Original reference media is unavailable"))?;
        let mut bytes = Vec::new();
        file.take(remaining + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| ApiError::validation("Could not read original reference media"))?;
        if bytes.len() as u64 > remaining {
            return Err(ApiError::validation(
                "Original references exceed the 32 MiB portable media budget",
            ));
        }
        remaining -= bytes.len() as u64;
        if format!("{:x}", Sha256::digest(&bytes)) != entry.metadata.sha256 {
            return Err(ApiError::validation("Original reference media changed"));
        }
        use mold_core::GenerationReference::*;
        let media = match reference {
            Image { media, .. }
            | NamedImage { media, .. }
            | Video { media, .. }
            | Audio { media, .. }
            | Mesh { media, .. } => media,
        };
        *media = mold_core::GenerationReferenceAuthority::Inline { data: bytes };
    }
    Ok(())
}

/// The archive's typed descriptors select retained bytes; matching names or
/// downloaded metadata alone never authorize a Library reference.
async fn inherit_source_references(
    state: &crate::state::AppState,
    binding: &crate::video_source::RetainedVideoSource,
    generation: &mut mold_core::GenerateRequest,
    authorized: bool,
) -> Result<(), ApiError> {
    if generation.references.is_some()
        || mold_core::minimax_h3::task_for_model(&generation.model)
            != Some(mold_core::minimax_h3::Task::Ref2va)
    {
        return Ok(());
    }
    if binding.authority_kind == mold_core::video_job::VideoSourceAuthorityKind::LegacySnapshot {
        return Err(ApiError::validation("This legacy video has no authenticated original reference archive. Attach fresh references before extending."));
    }
    if !authorized {
        return Err(ApiError::validation("Authentication is required to reuse the original reference media. Attach fresh references to extend this video."));
    }
    let (descriptors, stage_index) = source_reference_descriptors(&binding.metadata)?;
    if descriptors.len() > 64 {
        return Err(ApiError::validation(
            "Original reference inventory exceeds the retained member budget",
        ));
    }
    let state = state.clone();
    let binding = binding.clone();
    let descriptors = descriptors.to_vec();
    let references = tokio::task::spawn_blocking(move || {
        inherited_reference_bytes(&state, &binding, descriptors, stage_index)
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))??;
    generation.references = Some(references);
    Ok(())
}

fn inherited_reference_bytes(
    state: &crate::state::AppState,
    binding: &crate::video_source::RetainedVideoSource,
    descriptors: Vec<mold_core::GenerationReferenceMetadata>,
    stage_index: Option<usize>,
) -> Result<Vec<mold_core::GenerationReference>, ApiError> {
    let resolved = crate::gallery_source_media::resolve_members(state, &binding.filename)?
        .ok_or_else(|| ApiError::validation("Original reference media is no longer in Library"))?;
    if resolved.archive_identity_sha256 != binding.archive_identity_sha256 || resolved.corrupt {
        return Err(ApiError::validation(
            "Original source publication changed; refresh Library before extending",
        ));
    }
    let mut selected = Vec::with_capacity(descriptors.len());
    let mut total = 0_u64;
    for (index, descriptor) in descriptors.into_iter().enumerate() {
        let role = stage_index
            .map(|stage| format!("stage_reference:{stage}:{index}"))
            .unwrap_or_else(|| "references".into());
        let candidates = resolved
            .members
            .iter()
            .filter(|member| {
                member.member.content_sha256.as_deref() == Some(descriptor.sha256.as_str())
                    && member.member.role == role
            })
            .collect::<Vec<_>>();
        let member=candidates.first().ok_or_else(||ApiError::validation("Original reference bytes were not retained. Attach fresh references before extending."))?;
        total = total
            .checked_add(member.member.size_bytes)
            .ok_or_else(|| ApiError::validation("Original reference size overflow"))?;
        if total > 32 * 1024 * 1024 {
            return Err(ApiError::validation(
                "Original references exceed the 32 MiB portable media budget",
            ));
        }
        selected.push((descriptor, *member));
    }
    let lifecycle = state
        .queue_journal
        .queue_media_lifecycle()
        .ok_or_else(|| ApiError::validation("Original retained references are unavailable"))?;
    let mut remaining = 32 * 1024 * 1024;
    let mut references = Vec::with_capacity(selected.len());
    for (descriptor, member) in selected {
        let bytes = lifecycle
            .gallery_member_bytes_bounded(
                member.media_set.clone(),
                member.pin_id.clone(),
                member.index,
                remaining,
            )
            .map_err(|_| ApiError::validation("Original reference bytes are unavailable"))?;
        remaining -= bytes.len() as u64;
        let bytes = crate::gallery_source_media::media_content(&member.member.role, bytes)?;
        use sha2::{Digest, Sha256};
        if format!("{:x}", Sha256::digest(&bytes)) != descriptor.sha256 {
            return Err(ApiError::validation("Original reference identity changed"));
        }
        let mut value = serde_json::to_value(&descriptor)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        value["media"] = serde_json::json!({"authority":"descriptor"});
        value["provenance"] = serde_json::json!({"name":descriptor.name,"sha256":descriptor.sha256,"crop":descriptor.crop});
        let mut reference: mold_core::GenerationReference =
            serde_json::from_value(value).map_err(|_| {
                ApiError::validation(
                    "Original reference metadata is incomplete; attach fresh references",
                )
            })?;
        set_reference_bytes(&mut reference, bytes);
        references.push(reference);
    }
    Ok(references)
}

fn set_reference_bytes(reference: &mut mold_core::GenerationReference, bytes: Vec<u8>) {
    use mold_core::GenerationReference::*;
    let media = match reference {
        Image { media, .. }
        | NamedImage { media, .. }
        | Video { media, .. }
        | Audio { media, .. }
        | Mesh { media, .. } => media,
    };
    *media = mold_core::GenerationReferenceAuthority::Inline { data: bytes };
}

#[cfg(feature = "mp4")]
async fn source_boundary_png(
    file: std::fs::File,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ApiError> {
    tokio::task::spawn_blocking(move || {
        let facts =
            mold_inference::video_assembly_frames::inspect(file.try_clone()?, &mut || Ok(()))?;
        let mut tail = mold_inference::video_assembly_frames::tail(
            file,
            facts.frames,
            1,
            facts.width,
            facts.height,
            &mut || Ok(()),
        )?;
        let frame = tail
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Source has no visible boundary frame"))?;
        let scale = (f64::from(width) / f64::from(frame.width()))
            .min(f64::from(height) / f64::from(frame.height()));
        let fitted = image::imageops::resize(
            &frame,
            (f64::from(frame.width()) * scale).round().max(1.) as u32,
            (f64::from(frame.height()) * scale).round().max(1.) as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let mut canvas = image::RgbImage::new(width, height);
        image::imageops::overlay(
            &mut canvas,
            &fitted,
            i64::from((width - fitted.width()) / 2),
            i64::from((height - fitted.height()) / 2),
        );
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(canvas).write_to(&mut encoded, image::ImageFormat::Png)?;
        Ok::<_, anyhow::Error>(encoded.into_inner())
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(|error| ApiError::validation(format!("Could not inspect source boundary: {error:#}")))
}
#[cfg(not(feature = "mp4"))]
async fn source_boundary_png(_: std::fs::File, _: u32, _: u32) -> Result<Vec<u8>, ApiError> {
    Err(ApiError::validation(
        "This server was built without MP4 support",
    ))
}

fn source_reference_descriptors(
    metadata: &mold_core::OutputMetadata,
) -> Result<(&[mold_core::GenerationReferenceMetadata], Option<usize>), ApiError> {
    if let Some(references) = metadata
        .references
        .as_deref()
        .filter(|references| !references.is_empty())
    {
        return Ok((references, metadata.video_job.as_ref().map(|_| 0)));
    }
    if let Some(chain) = metadata.chain.as_ref() {
        if let Some(stage) = chain
            .stages
            .last()
            .filter(|stage| !stage.generation_references.is_empty())
        {
            return Ok((&stage.generation_references, Some(chain.stages.len() - 1)));
        }
    }
    Err(ApiError::validation(
        "Original reference media is unavailable. Attach references before extending this video.",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_uses_recipe_default_envelope_instead_of_model_ceiling_or_requested_frames() {
        for (model, family, expected) in [
            (
                mold_core::minimax_h3::FL2VA_FASTH3_V2_INT8,
                "minimax-h3",
                124,
            ),
            ("ltx-video-0.9.6:bf16", "ltx-video", 25),
        ] {
            let mut request: VideoJobRequest = serde_json::from_value(serde_json::json!({
                "generation":{"model":model,"prompt":"scene","width":512,"height":512,"steps":8,"frames":99999},
                "duration_ms":"60000"
            })).unwrap();
            let (plan, envelope, _) = plan_template(&request, family, None, 0).unwrap();
            assert_eq!(envelope.maximum_frames, expected, "{model}");
            assert_eq!(plan.single_clip_max_frames, expected);
            if family == "ltx-video" {
                let (_, bounded, _) =
                    plan_template_with_defaults(&request, family, None, 0, Some(30)).unwrap();
                assert_eq!(
                    bounded.maximum_frames, 25,
                    "sidecar default must snap to its family grid"
                );
            }
            request.generation.frames = Some(1);
            assert_eq!(
                plan_template(&request, family, None, 0)
                    .unwrap()
                    .1
                    .maximum_frames,
                expected
            );
        }
    }

    #[test]
    fn source_reference_selection_uses_only_the_actual_final_authored_scene() {
        let generation: mold_core::GenerateRequest=serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8","prompt":"scene","width":512,"height":512,"steps":4})).unwrap();
        let mut metadata =
            mold_core::OutputMetadata::from_generate_request(&generation, 1, None, "test");
        let reference = |name: &str| serde_json::json!({"kind":"image","index":0,"name":name,"sha256":"a".repeat(64),"mime_type":"image/png","width":512,"height":512});
        metadata.chain=Some(serde_json::from_value(serde_json::json!({"stage_count":2,"motion_tail_frames":0,"stages":[{"prompt":"one","frames":107,"transition":"cut","loras":[],"generation_references":[reference("old")],"keyframes":[]},{"prompt":"two","frames":107,"transition":"cut","loras":[],"generation_references":[reference("final")],"keyframes":[]}]})).unwrap());
        let (references, stage) = source_reference_descriptors(&metadata).unwrap();
        assert_eq!(stage, Some(1));
        assert_eq!(references[0].name.as_deref(), Some("final"));
        metadata.chain.as_mut().unwrap().stages[1]
            .generation_references
            .clear();
        assert!(source_reference_descriptors(&metadata).is_err());
    }

    #[test]
    fn duration_plan_rejects_meaningful_unsupported_inputs_before_scrubbing() {
        let mut request:VideoJobRequest=serde_json::from_value(serde_json::json!({"generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"scene","width":512,"height":512,"steps":4},"duration_ms":"30000"})).unwrap();
        request.generation.audio_file = Some(vec![1]);
        assert!(plan_template(&request, "ltx2", None, 0).is_err());
        request.generation.audio_file = None;
        request.generation.mask_image = Some(vec![1]);
        assert!(plan_template(&request, "ltx2", None, 0).is_err());
        request.generation.mask_image = None;
        request.generation.source_video_path = Some("local-only.mp4".into());
        assert!(plan_template(&request, "ltx2", None, 0).is_err());
        request.generation.source_video_path = None;
        request.generation.batch_size = 2;
        assert!(plan_template(&request, "ltx2", None, 0).is_err());
    }

    #[test]
    fn continuous_library_extension_refuses_a_second_opening_authority() {
        let mut request: VideoJobRequest = serde_json::from_value(serde_json::json!({
            "generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"scene","width":512,"height":512,"steps":4},
            "duration_ms":"30000", "source":{"filename":"original.mp4"}
        })).unwrap();
        request.generation.source_image = Some(vec![1]);
        assert!(plan_template(
            &request,
            "ltx2",
            Some(mold_core::SourceImageCapability::Optional),
            97
        )
        .is_err());
        request.generation.source_image = None;
        request.generation.keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 0,
            image: vec![1],
            name: None,
        }]);
        assert!(plan_template(
            &request,
            "ltx2",
            Some(mold_core::SourceImageCapability::Optional),
            97
        )
        .is_err());
    }

    #[test]
    fn blended_scene_boundaries_require_enough_visible_frames_before_inference() {
        let mut request: VideoJobRequest = serde_json::from_value(serde_json::json!({
            "generation":{"model":"wan2.1-t2v-1.3b","prompt":"scene","width":512,"height":512,"steps":4,"fps":24},
            "duration_ms":"1100", "scenes":[{"prompt":"brief","duration_ms":"100"},{"prompt":"next","duration_ms":"1000"}]
        })).unwrap();
        assert!(plan_template(&request, "wan", None, 0).is_err());
        request.source =
            Some(serde_json::from_value(serde_json::json!({"filename":"prefix.mp4"})).unwrap());
        assert!(plan_template(&request, "wan", None, 8).is_ok());
        request.source = None;
        request.scenes = None;
        request.duration_ms = 100;
        assert!(plan_template(&request, "wan", None, 0).is_ok());
    }

    #[test]
    fn blended_library_extension_requires_a_full_visible_fade() {
        let request:VideoJobRequest=serde_json::from_value(serde_json::json!({"generation":{"model":"wan2.1-t2v-1.3b","prompt":"scene","width":512,"height":512,"steps":4},"duration_ms":"30000","source":{"filename":"original.mp4"}})).unwrap();
        assert!(plan_template(&request, "wan", None, 1).is_err());
    }

    #[test]
    fn duration_preserves_output_target_upscaling_controls() {
        let mut request:VideoJobRequest=serde_json::from_value(serde_json::json!({"generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"scene","width":512,"height":512,"steps":4},"duration_ms":"30000"})).unwrap();
        request.generation.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X2);
        assert!(plan_template(&request, "ltx2", None, 0).is_ok());
        request.generation.spatial_upscale = None;
        request.generation.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        assert!(plan_template(&request, "ltx2", None, 0).is_ok());
    }

    #[test]
    fn duration_template_retains_literal_controls_and_uses_safe_clip_cap() {
        let request: mold_core::video_job::VideoJobRequest = serde_json::from_value(serde_json::json!({"generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"a\\npath","width":512,"height":512,"steps":4,"seed":18446744073709551615_u64,"frames":99999,"guidance":2.5,"pipeline":"distilled"},"duration_ms":"30000"})).unwrap();
        let (plan, envelope, template) = plan_template(
            &request,
            "ltx2",
            Some(mold_core::SourceImageCapability::Optional),
            0,
        )
        .unwrap();
        assert_eq!(plan.single_clip_max_frames, 97);
        assert_eq!(template.stages[0].prompt, "a\\npath");
        assert_eq!(template.seed, Some(u64::MAX));
        assert_eq!(
            template.generation_controls.unwrap().pipeline,
            request.generation.pipeline
        );
        assert!(plan.clip_count > 1);
        assert_eq!(envelope.overlap_frames, 17);
        assert!(plan.validation_revision.is_empty());
    }
}
