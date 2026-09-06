//! Read-only chain planning for `POST /api/generate/chain/validate`.
//!
//! Chained video generation itself is durable chain jobs
//! (`POST /api/chain-jobs` + `GET /api/chain-jobs/{id}/events`). The
//! synchronous and SSE compatibility endpoints that ran one as a hidden
//! ephemeral job are gone; what stays here is the normalization, model-freeze
//! and VRAM-estimate work a client uses to inspect a plan without creating
//! one.

use axum::{extract::State, Json};
use mold_core::chain::{ChainRequest, ChainValidationResponse, TransitionMode};

use crate::model_manager::ExistingModelAuthority;
use crate::routes::ApiError;
use crate::state::AppState;

fn chain_freeze_error(model: &str, error: impl std::fmt::Display) -> ApiError {
    ApiError::validation(format!(
        "cannot freeze concrete chain model companions for '{model}': {error}"
    ))
}

pub(crate) async fn resolve_chain_model_authority(
    state: &AppState,
    model: &str,
) -> Result<ExistingModelAuthority, ApiError> {
    // Preserve the request-local snapshot when it already resolves. Tests,
    // embedded callers, and catalog overlays may intentionally carry
    // authority that is newer than (or absent from) bootstrap storage.
    {
        let config = state.config.read().await;
        require_chain_model_artifact_activation(&config, model, None, None)?;
        if let Some(authority) =
            crate::model_manager::resolve_existing_model_authority(model, &config)
                .map_err(|error| chain_freeze_error(model, error.error))?
        {
            require_chain_model_artifact_activation(
                &authority.config,
                model,
                Some(&authority.paths),
                None,
            )?;
            return Ok(authority);
        }
    }

    // Inventory/component endpoints reload bootstrap configuration before
    // reporting a model as installed. Retry from that same fresh authority
    // only when the in-memory snapshot has no concrete paths, so a long-lived
    // server can self-heal without replacing valid request-local authority.
    let config = crate::model_manager::refresh_config(state).await;
    require_chain_model_artifact_activation(&config, model, None, None)?;
    let authority = crate::model_manager::resolve_existing_model_authority(model, &config)
        .map_err(|error| chain_freeze_error(model, error.error))?
        .ok_or_else(|| {
            chain_freeze_error(
                model,
                format!("model '{model}' has no concrete local artifact paths"),
            )
        })?;
    require_chain_model_artifact_activation(
        &authority.config,
        model,
        Some(&authority.paths),
        None,
    )?;
    Ok(authority)
}

pub(crate) fn freeze_chain_model(
    authority: ExistingModelAuthority,
    model: &str,
) -> Result<mold_core::chain_job::FrozenChainModel, ApiError> {
    require_chain_model_artifact_activation(
        &authority.config,
        model,
        Some(&authority.paths),
        None,
    )?;
    let frozen = crate::execution_plan::freeze_chain_model_with_paths(
        &authority.config,
        model,
        authority.paths,
    )
    .map_err(|error| chain_freeze_error(model, error))?;
    require_chain_model_artifact_activation(&authority.config, model, None, Some(&frozen))?;
    Ok(frozen)
}

fn require_chain_artifact_path_activation(
    path: &std::path::Path,
    artifact_root: &std::path::Path,
    family: Option<&str>,
) -> Result<(), ApiError> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    mold_core::require_model_artifact_activation(path, Some(artifact_root), family)
        .map_err(ApiError::model_activation)?;

    // A neutral-looking symlink or relative path may still resolve into a
    // gated artifact tree. Check the concrete target against the concrete
    // trusted root as well; missing files remain covered by the raw identity
    // above and are diagnosed separately by ordinary model resolution.
    if let Ok(canonical_path) = std::fs::canonicalize(path) {
        let canonical_root =
            std::fs::canonicalize(artifact_root).unwrap_or_else(|_| artifact_root.to_path_buf());
        mold_core::require_model_artifact_activation(
            &canonical_path,
            Some(&canonical_root),
            family,
        )
        .map_err(ApiError::model_activation)?;
    }
    Ok(())
}

fn require_chain_config_artifact_activation(
    config: &mold_core::ModelConfig,
    artifact_root: &std::path::Path,
    fallback_family: Option<&str>,
) -> Result<(), ApiError> {
    let family = config.family.as_deref().or(fallback_family);
    for path in config.all_file_paths() {
        require_chain_artifact_path_activation(std::path::Path::new(&path), artifact_root, family)?;
    }
    Ok(())
}

fn require_chain_manifest_artifact_activation(
    manifest: &mold_core::manifest::ModelManifest,
    artifact_root: &std::path::Path,
) -> Result<(), ApiError> {
    let family = Some(manifest.family.as_str());
    mold_core::require_registered_manifest_activation(manifest)
        .map_err(ApiError::model_activation)?;
    for file in &manifest.files {
        let storage_path = mold_core::manifest::storage_path(manifest, file);
        require_chain_artifact_path_activation(&storage_path, artifact_root, family)?;
        require_chain_artifact_path_activation(
            &artifact_root.join(storage_path),
            artifact_root,
            family,
        )?;
    }
    Ok(())
}

/// Fail closed over every model/config/manifest identity that can become the
/// immutable authority of a chain job. Both raw and canonicalized artifact
/// paths are checked before they can be frozen into a durable manifest.
fn require_chain_model_artifact_activation(
    config: &mold_core::Config,
    model: &str,
    paths: Option<&mold_core::ModelPaths>,
    frozen: Option<&mold_core::chain_job::FrozenChainModel>,
) -> Result<(), ApiError> {
    let artifact_root = config.resolved_models_dir();
    let canonical_model = mold_core::manifest::resolve_model_name(model);
    let manifest = mold_core::manifest::find_manifest(&canonical_model);
    let configured = config.resolved_model_config(model);
    let family = configured
        .family
        .as_deref()
        .or_else(|| manifest.map(|entry| entry.family.as_str()));

    mold_core::require_model_activation(model, family).map_err(ApiError::model_activation)?;
    mold_core::require_model_activation(&canonical_model, family)
        .map_err(ApiError::model_activation)?;
    require_chain_config_artifact_activation(&configured, &artifact_root, family)?;

    if let Some(paths) = paths {
        for path in paths.all_file_paths() {
            require_chain_artifact_path_activation(path, &artifact_root, family)?;
        }
    }

    if let Some(frozen) = frozen {
        let frozen_family = frozen.config.family.as_deref().or(family);
        mold_core::require_model_activation(model, frozen_family)
            .map_err(ApiError::model_activation)?;
        if !frozen.runtime_model_id.is_empty() {
            mold_core::require_model_activation(&frozen.runtime_model_id, frozen_family)
                .map_err(ApiError::model_activation)?;
        }
        require_chain_config_artifact_activation(&frozen.config, &artifact_root, frozen_family)?;
    }

    if let Some(manifest) = manifest {
        require_chain_manifest_artifact_activation(manifest, &artifact_root)?;
    }
    Ok(())
}

/// Extend the primary-model fence to request-local executable artifacts.
/// Stage LoRA paths are not part of `ModelPaths`, so they must be checked
/// independently before validation, control downloads, persistence, or a
/// persisted job's transition back to `Queued`.
pub(crate) fn require_chain_artifact_activation(
    config: &mold_core::Config,
    request: &ChainRequest,
    paths: Option<&mold_core::ModelPaths>,
    frozen: Option<&mold_core::chain_job::FrozenChainModel>,
) -> Result<(), ApiError> {
    require_chain_model_artifact_activation(config, &request.model, paths, frozen)?;
    let canonical_model = mold_core::manifest::resolve_model_name(&request.model);
    let family = frozen
        .and_then(|snapshot| snapshot.config.family.clone())
        .or_else(|| config.resolved_model_config(&request.model).family)
        .or_else(|| {
            mold_core::manifest::find_manifest(&canonical_model).map(|entry| entry.family.clone())
        });
    let artifact_root = config.resolved_models_dir();

    for stage in &request.stages {
        if let Some(model) = stage.model.as_deref() {
            mold_core::require_model_activation(model, family.as_deref())
                .map_err(ApiError::model_activation)?;
        }
        for lora in &stage.loras {
            let path = std::path::Path::new(&lora.path);
            require_chain_artifact_path_activation(path, &artifact_root, family.as_deref())?;

            // Built-in camera controls materialize a hidden manifest before
            // the job is persisted. Include that manifest and its target path
            // in this same pre-mutation authority boundary.
            if let Some(alias) = lora.path.strip_prefix("camera-control:") {
                if let Ok(preset) = mold_core::ltx2_camera::resolve_camera_control_preset(alias) {
                    if let Some(manifest) =
                        mold_core::manifest::find_manifest(preset.download_model)
                    {
                        require_chain_manifest_artifact_activation(manifest, &artifact_root)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate the chain request's model family and apply family-specific
/// fixups. Returns `Err` with a 422 response if the family doesn't support
/// chain generation. Mutates `req.motion_tail_frames` for families that lack
/// latent context handoff (currently `ltx-video`) so the stitch layer doesn't
/// trim independent fresh frames at Smooth boundaries.
pub(crate) fn validate_chain_build_features(_req: &ChainRequest) -> Result<(), ApiError> {
    #[cfg(not(feature = "mp4"))]
    if _req.enable_audio == Some(true) {
        return Err(ApiError::validation(
            "chain audio requires a mold build with the mp4 feature; rebuild with `--features mp4` or remove `enable_audio: true`",
        ));
    }
    Ok(())
}

/// The family a chain request will actually render as.
///
/// The sidecar-derived `ModelConfig` an installed `cv:` / `hf:` checkpoint
/// carries is the authority; the built-in manifest is the fallback. Reading
/// only the manifest leaves a catalog wan checkpoint unclassified, and an
/// unclassified family means the LTX `8k+1` grid (#783).
pub(crate) fn resolve_chain_family(config: &mold_core::Config, model: &str) -> String {
    let resolved_model = config.resolved_model_config(model);
    let canonical_model = mold_core::manifest::resolve_model_name(model);
    resolved_model
        .family
        .clone()
        .or_else(|| {
            mold_core::manifest::find_manifest(&canonical_model).map(|model| model.family.clone())
        })
        .unwrap_or_default()
}

pub(crate) fn validate_and_normalize_chain_family(
    config: &mold_core::Config,
    req: &mut ChainRequest,
) -> Result<(), ApiError> {
    require_chain_artifact_activation(config, req, None, None)?;
    let resolved_model = config.resolved_model_config(&req.model);
    let canonical_model = mold_core::manifest::resolve_model_name(&req.model);
    let manifest = mold_core::manifest::find_manifest(&canonical_model);
    let family = resolve_chain_family(config, &req.model);
    mold_core::require_model_activation(
        &req.model,
        (!family.is_empty()).then_some(family.as_str()),
    )
    .map_err(ApiError::model_activation)?;
    if mold_core::minimax_h3::is_family(&family) {
        return Err(ApiError::with_code(
            "MiniMax H3 chains remain disabled until reference media have durable recovery authority",
            "MINIMAX_H3_CHAIN_RECOVERY_UNSUPPORTED",
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    validate_chain_build_features(req)?;
    // A clip is one generation, so the composed ceiling applies here exactly
    // as it does to a single shot. `resolved_model.spatial_upscaler` is the
    // config-supplied override; the shared resolver reads the manifest, so a
    // model that only has the component through config keeps the conservative
    // single-pass ceiling rather than being admitted on a path the engine
    // might not take.
    let composition = if family == "ltx2" {
        mold_core::validation::ltx2_spatial_composition(&req.model, None)
    } else {
        mold_core::validation::Ltx2SpatialComposition::SinglePass
    };
    mold_core::validate_generation_dimensions_for_model(
        &req.model,
        req.width,
        req.height,
        (!family.is_empty()).then_some(family.as_str()),
        composition,
    )
    .map_err(ApiError::validation)?;

    // Only reject early when we positively know the family is non-chain-capable.
    // An empty family means the model isn't in the manifest yet (catalog
    // synth, mock test, etc.) — let it through; the engine's
    // `as_chain_renderer()` check will still fire if it really can't render.
    if !family.is_empty() && crate::chain_limits::family_cap(&family).is_none() {
        return Err(ApiError::validation(format!(
            "model '{}' (family '{}') does not support chained video generation",
            req.model, family
        )));
    }
    if !family.is_empty() {
        let sequence = crate::chain_limits::sequence_support(
            &req.model,
            &family,
            resolved_model.spatial_upscaler.is_some()
                || manifest.is_some_and(|model| {
                    model.files.iter().any(|file| {
                        file.component == mold_core::manifest::ModelComponent::SpatialUpscaler
                    })
                }),
        );
        if !sequence.supported {
            return Err(ApiError::validation(sequence.reason.unwrap_or_else(|| {
                format!("model '{}' cannot render sequences", req.model)
            })));
        }
    }
    // Each stage is denoised as one generation, so it is bound by the family's
    // SINGLE-REQUEST ceiling — LTX-2's 20 s runtime budget, 481 frames at
    // 24 fps. That is deliberately looser than the `frames_per_clip_cap`
    // `/api/capabilities/chain-limits` advertises, which is the model's
    // routing clip size: `mold run --clip-frames` is documented to go all the
    // way to the model's real budget and submits chains here, so admission
    // must not refuse what the CLI is allowed to ask for. This used to be
    // neither advertised nor enforced, so an over-budget clip only failed once
    // the stage reached the engine.
    if let Some(cap) = crate::chain_limits::family_cap_at_fps(&family, req.fps) {
        if let Some((idx, stage)) = req
            .stages
            .iter()
            .enumerate()
            .find(|(_, stage)| stage.frames > cap)
        {
            return Err(ApiError::validation(format!(
                "stage {idx} asks for {} frames; the single-request ceiling for '{}' at {} fps is \
                 {cap} frames",
                stage.frames, family, req.fps,
            )));
        }
    }
    // Every stage is denoised as one generation, so its frame count has to sit
    // on the family's own grid. Wan is `4k+1` where the LTX families are
    // `8k+1`; catching it here names the stage, instead of failing mid-chain
    // once the stage reaches the engine's request validator.
    if let (Some(step), Some(offset)) = (
        mold_core::validation::frame_step_for_family(&family),
        mold_core::validation::frame_offset_for_family(&family),
    ) {
        if let Some((idx, stage)) = req
            .stages
            .iter()
            .enumerate()
            .find(|(_, stage)| stage.frames % step != offset % step)
        {
            return Err(ApiError::validation(format!(
                "stage {idx} asks for {} frames; '{family}' clips must be {step}k+{offset}",
                stage.frames,
            )));
        }
    }
    // Wan probes the resolved checkpoint's own headers FIRST, exactly as
    // `/api/models` and single-generation admission do: `ModelPaths` honors
    // config/env path overrides, so the artifacts actually loaded can differ
    // from the manifest's task structure. The manifest is the cold fallback
    // for a model that is not downloaded yet. Other families have no header
    // probe and their manifest contract binds directly.
    let manifest_contract = manifest.and_then(|model| model.defaults.source_image);
    let contract = if family == "wan" {
        let probed = mold_core::ModelPaths::resolve(&req.model, config).and_then(|paths| {
            mold_inference::wan_source_image_capability(&paths.transformer, &paths.vae)
        });
        mold_core::SourceImageCapability::resolve(manifest_contract, probed)
    } else {
        manifest_contract
    };
    let contract = mold_core::validation::source_image_capability_for_engine(
        (!family.is_empty()).then_some(family.as_str()),
        contract,
    );

    // Stage 0 is the only clip that can carry an opening image; every
    // continuation is seeded by the seam. Admission never asked the
    // checkpoint whether it could accept one — `enforce_source_image_
    // capability` covered single generations and the placement preview but
    // not this path — so a wan I2V sequence with no opening image was
    // admitted and then died after the UMT5 encode and both expert loads had
    // been paid for, and a T2V sequence carrying one was admitted too (#783).
    //
    // The two arms read different sets on purpose. `Required` asks whether
    // stage 0 can be seeded at all, so only the opening image counts — every
    // continuation is seeded by the seam. `Unsupported` asks whether an image
    // was supplied anywhere the engine has no channel for, so a per-stage
    // image on a continuation has to count too; a script may attach one to
    // any stage (`source_image_path`).
    // The top-level `source_image` is the auto-expand form's opening image and
    // `normalise` only projects it onto stage 0 when `stages` is empty — with
    // explicit stages it is cleared. Counting it either way admitted a mixed
    // form whose image is discarded before execution, so the I2V model loaded
    // and then failed on an unseeded stage 0.
    let opening_image = (req.stages.is_empty() && req.source_image.is_some())
        || req.stages.first().is_some_and(|s| s.source_image.is_some());
    let any_image = opening_image || req.stages.iter().any(|s| s.source_image.is_some());
    let family_hint = (!family.is_empty()).then_some(family.as_str());
    let has_source = match contract {
        Some(mold_core::SourceImageCapability::Unsupported) => any_image,
        _ => opening_image,
    };
    if let Some(message) = mold_core::validation::source_image_contract_violation(
        family_hint,
        &req.model,
        contract,
        has_source,
    ) {
        return Err(ApiError::validation(message));
    }

    // A ONE-SHOT long video is a chain only because the model cannot render it
    // in one pass, so admission owes the caller the same answer the CLI router
    // gives: a text-only wan tier or the legacy LTX-Video engine hands nothing
    // across a clip boundary, and every stage re-derives the scene from the
    // same prompt and seed. The "longer" video is the same clip repeated with a
    // visible reset at each seam, paid for at full GPU price. `mold-core` owns
    // the decision and the sentence so the CLI, the Studio router, and this 422
    // cannot drift (#1508 landed it on the CLI alone, and the web Studio kept
    // submitting the same job here).
    //
    // Scoped to `ephemeral`: an AUTHORED sequence that repeats stages is what
    // its author asked for, and admission must not second-guess it. A client
    // that omits the field (serde default `false`) is therefore exempt by
    // design, and that is safe because every one-shot auto-chainer sets it —
    // the CLI (`commands/chain.rs`), the web (`useGenerateStream.ts`), desktop
    // (`stores/generation.ts`), and the iPhone (via `buildAutoChainRequest`).
    // The TUI never builds an ephemeral chain at all.
    if req.ephemeral && matches!(family.as_str(), "wan" | "ltx-video") {
        // Derived from the STAGES when there are stages: the largest stage is
        // the clip the caller actually rendered with, and it is the only form
        // of the answer that survives `normalise`, which clears `clip_frames`
        // along with the rest of the auto-expand sugar. Reading `clip_frames`
        // alone made this door contradict itself — `validate_chain` preflights
        // before AND after normalising, so a `{total_frames: 97,
        // clip_frames: 97}` body on a 73-frame-routing tier passed the first
        // call and was refused by the second, i.e. validate refusing what
        // `create_chain_job` (one call, pre-normalise) admits. The CLI needs
        // the same derivation for a different reason: it normalises
        // client-side, so its ephemeral chain arrives with stages and no
        // `clip_frames` at all.
        //
        // The caller's own clip size still wins on the auto-expand form —
        // `--clip-frames` is documented to go up to the family budget, and
        // refusing a request that would have been ONE clip would be a false
        // refusal.
        let clip_frames = req
            .stages
            .iter()
            .map(|stage| stage.frames)
            .max()
            .or(req.clip_frames)
            .or_else(|| mold_core::chain::routing_clip_frames(&family, &req.model))
            .unwrap_or(mold_core::chain::LTX2_DEFAULT_CLIP_FRAMES);
        let clip_frames = crate::chain_limits::family_cap_at_fps(&family, req.fps)
            .map_or(clip_frames, |cap| clip_frames.min(cap));
        // Both shapes reach this door: the Studio and the auto-expand sugar
        // send `total_frames`, the CLI's own router sends the expanded stages.
        let total_frames = if req.stages.is_empty() {
            req.total_frames.unwrap_or(0)
        } else {
            req.stages.iter().map(|stage| stage.frames).sum()
        };
        if let Some(message) = mold_core::chain::text_only_auto_chain_refusal(
            Some(family.as_str()),
            &req.model,
            contract,
            total_frames,
            clip_frames,
        ) {
            return Err(ApiError::validation(message));
        }
    }

    if family == "wan" {
        // Wan has no latent motion tail. Its seam re-renders exactly the one
        // frame the continuation was seeded with, and only an image-conditioned
        // checkpoint can be seeded at all — so the tail is 1 or 0, never the
        // LTX-shaped value a client may have carried over.
        // One authority, shared with the forced-local CLI path: the two had
        // drifted, and only this one normalized (#783).
        let normalized = mold_core::validation::chain_motion_tail_frames_for_family(
            &family,
            contract,
            req.motion_tail_frames,
        );
        if req.motion_tail_frames != normalized {
            tracing::debug!(
                model = %req.model,
                original = req.motion_tail_frames,
                normalized,
                "wan carries one seeded frame at most; normalizing motion_tail_frames"
            );
            req.motion_tail_frames = normalized;
        }
    }
    if family == "ltx-video" && req.motion_tail_frames > 0 {
        // LtxVideoEngine has no img2vid path, so the carry tail can't anchor
        // the next stage's denoise. Zero motion_tail makes Smooth boundaries
        // collapse to clean concatenation at the stitch layer (Smooth is
        // implemented as `next_clip.skip(motion_tail)` — with 0, no skip).
        tracing::debug!(
            model = %req.model,
            original = req.motion_tail_frames,
            "ltx-video has no context handoff; forcing motion_tail_frames=0"
        );
        req.motion_tail_frames = 0;
    }
    // Audio is only emitted by AV-capable families (currently LTX-2 / LTX-2.3).
    // Reject `enable_audio: true` for video-only families (e.g. ltx-video) at
    // the wire boundary so users get a clear upfront error instead of
    // silently muted output. Empty `family` (mock / catalog-synth models) is
    // permissive — the engine's renderer abstraction is the final gate.
    if req.enable_audio == Some(true)
        && !family.is_empty()
        && !crate::chain_limits::family_supports_audio(&family)
    {
        return Err(ApiError::validation(format!(
            "model '{}' (family '{}') does not support chain audio; \
             remove `enable_audio: true` or pick an LTX-2 / LTX-2.3 model",
            req.model, family
        )));
    }
    // The audio default is RESOLVED here, once, for every durable chain job —
    // `POST /api/chain-jobs`, an amend candidate, and
    // `POST /api/generate/chain/validate` all pass through this door. Unset
    // means the recipe's own answer (`resolve_enable_audio`), which is ON
    // wherever the family can deliver sound: an LTX-2 sequence now renders
    // with audio exactly like the same model's one-shot, which has defaulted
    // ON for MP4 at the engine since the flag existed. The two had diverged,
    // and a sequence was the silent one.
    //
    // Resolving to an EXPLICIT value is the point: every downstream reader —
    // the stage-request builder, `finalize_job`'s mux gate, and
    // `stage_cache_ready` — asks `== Some(true)`, so a persisted manifest
    // carries the answer rather than re-deriving it. A manifest written
    // before this door resolved keeps its `None` and stays silent, which is
    // the only correct answer for stages already rendered without sound.
    //
    // An empty family (mock / catalog-synth models) is left alone for the
    // same reason the refusal above skips it: nothing here classified it.
    if !family.is_empty() {
        // A build with no mp4 muxer cannot deliver a chain's audio at all,
        // and the refusal in `validate_chain_build_features` only covers an
        // EXPLICIT ask. The default must not mint the request that build
        // would then fail to finish.
        #[cfg(feature = "mp4")]
        let deliverable = crate::chain_limits::family_supports_audio(&family)
            // MP4 is the only container a stitched chain can carry a track
            // in, exactly as `ltx2::execution::wants_audio_output` reads it
            // for a one-shot. A gif/webp chain would render every stage's
            // audio at full GPU cost and then transcode it away.
            && req.output_format == mold_core::OutputFormat::Mp4;
        #[cfg(not(feature = "mp4"))]
        let deliverable = false;
        req.enable_audio = Some(mold_core::generation_profile::resolve_enable_audio(
            req.enable_audio,
            deliverable,
        ));
    }
    Ok(())
}

/// `POST /api/generate/chain/validate` — normalize and validate a chain
/// without creating a job, starting downloads, or touching an inference
/// engine.
#[utoipa::path(
    post,
    path = "/api/generate/chain/validate",
    tag = "generation",
    request_body = mold_core::ChainRequest,
    responses(
        (status = 200, description = "Normalized chain plan", body = mold_core::ChainValidationResponse),
        (status = 422, description = "Invalid request or unsupported model"),
    )
)]
pub async fn validate_chain(
    State(state): State<AppState>,
    Json(mut req): Json<ChainRequest>,
) -> Result<Json<ChainValidationResponse>, ApiError> {
    req.normalize_prompt_newlines();
    let opening_transition = req.stages.first().map(|stage| stage.transition);
    let requested_motion_tail = req.motion_tail_frames;
    let mut warnings = Vec::new();

    // The same authority submission uses. Reading `state.config` directly
    // left an installed `cv:` / `hf:` wan checkpoint with an empty family,
    // which skipped the wan tail normalization, the family cap, the grid
    // check, and the sequence-support check — so `mold chain validate` and the
    // submission that follows it disagreed about exactly the models the
    // sequence work targets (#783).
    let authority = resolve_chain_model_authority(&state, &req.model).await?;
    validate_and_normalize_chain_family(&authority.config, &mut req)?;
    let family = resolve_chain_family(&authority.config, &req.model);
    if req.motion_tail_frames != requested_motion_tail {
        warnings.push(format!(
            "The selected model normalized motion_tail_frames from {requested_motion_tail} to {}.",
            req.motion_tail_frames
        ));
    }

    let mut req = req
        .normalise_with_family(Some(&family))
        .map_err(|error| ApiError::validation(error.to_string()))?;
    if opening_transition.is_some_and(|transition| transition != TransitionMode::Smooth) {
        warnings.push("The opening clip transition was normalized to smooth.".to_string());
    }
    {
        let before = req.motion_tail_frames;
        validate_and_normalize_chain_family(&authority.config, &mut req)?;
        if req.motion_tail_frames != before && req.motion_tail_frames != requested_motion_tail {
            warnings.push(format!(
                "The selected model normalized motion_tail_frames from {before} to {}.",
                req.motion_tail_frames
            ));
        }
    }
    let mut generate = req.synthetic_generate_request(
        mold_core::OutputFormat::Mp4,
        req.estimated_total_frames(),
        req.fps,
    );
    generate.loras = Some(
        req.stages
            .iter()
            .flat_map(|stage| stage.loras.iter())
            .map(|lora| mold_core::LoraWeight {
                path: lora.path.clone(),
                scale: lora.scale,

                expert: None,
            })
            .collect(),
    );
    crate::routes::plan_builtin_ltx2_camera_controls(&state, &generate).await?;

    let vram_estimate = chain_vram_estimate(&state, &req).await;
    Ok(Json(
        ChainValidationResponse::from_normalized(&req, warnings).with_vram_estimate(vram_estimate),
    ))
}

/// Advisory peak-VRAM estimate for a normalized chain.
///
/// Stages execute strictly one at a time, so the chain's peak is the *max*
/// over stages, never their sum. Each stage is priced through the same
/// `build_stage_generate_request` the runner dispatches, so the number matches
/// what admission will later re-derive.
///
/// Returns `None` — never a guess — when the model is not downloaded or no
/// device sample exists. Validation is a pure normalization endpoint and must
/// not become download-gated, and `fits` must never be fabricated.
async fn chain_vram_estimate(
    state: &AppState,
    req: &ChainRequest,
) -> Option<mold_core::chain::VramEstimate> {
    let base_seed = req.seed.unwrap_or(0);
    let mut worst_case_bytes = 0u64;
    let mut fits = true;

    for (idx, stage) in req.stages.iter().enumerate() {
        let stage_seed = stage
            .seed_offset
            .map_or(base_seed, |offset| base_seed ^ offset);
        let stage_request =
            crate::chain_job_runner::build_stage_generate_request(stage, req, stage_seed, idx);
        let estimate =
            crate::model_manager::estimate_generation_memory(state, &stage_request).await;
        let Ok(estimate) = estimate else {
            // Model not downloaded yet, or paths unresolvable. Validate still
            // answers; it just cannot price the run.
            return None;
        };
        // A queued chain runs stages serially, after earlier GPU work releases
        // its allocations. Use the same stable physical-capacity estimate as
        // the one-shot badge; live free VRAM would make this advisory verdict
        // flip simply because another queued clip is currently denoising.
        let stage_peak = estimate.capacity_peak_memory_bytes?;
        let stage_fits = estimate.fits_device_capacity?;
        worst_case_bytes = worst_case_bytes.max(stage_peak);
        fits &= stage_fits;
    }

    Some(mold_core::chain::VramEstimate {
        worst_case_bytes,
        fits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use mold_core::chain::{ChainRequest, ChainStage, TransitionMode};
    use mold_core::OutputFormat;

    fn req(format: OutputFormat) -> ChainRequest {
        ChainRequest {
            offload: None,
            collection: None,
            tags: None,
            title: None,
            model: "ltx-2-19b-distilled:mock".into(),
            stages: vec![
                ChainStage {
                    prompt: "stage zero".into(),
                    frames: 9,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Smooth,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
                ChainStage {
                    prompt: "stage one".into(),
                    frames: 17,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: Some(7),
                    transition: TransitionMode::Cut,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
            ],
            motion_tail_frames: 0,
            width: 64,
            height: 64,
            fps: 12,
            seed: Some(42),
            steps: 4,
            guidance: 3.0,
            strength: 1.0,
            output_format: format,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
            ephemeral: false,
        }
    }

    /// Peak is the max over stages, never their sum: stages run one at a
    /// time, so no two working sets are ever co-resident. Summing would make
    /// a long sequence look infeasible on any card.
    ///
    /// The fixture has no model on disk, so the estimate is *withheld* rather
    /// than guessed — and validation still answers. `fits` must never be
    /// fabricated, and validate must not become download-gated.
    #[tokio::test]
    async fn chain_validation_withholds_vram_estimate_it_cannot_price() {
        let state = AppState::for_tests();
        let chain = req(OutputFormat::Mp4)
            .normalise()
            .expect("fixture chain must normalise");

        assert!(
            chain_vram_estimate(&state, &chain).await.is_none(),
            "an unpriceable chain must yield no estimate, never a fabricated one"
        );
    }

    #[tokio::test]
    async fn chain_preflight_admits_an_ltx2_two_stage_checkpoint() {
        // A dev checkpoint selects the two-stage pipeline, which now renders
        // sequence clips: the stage-2 pass already re-encodes conditioning at
        // its own pixel shape, and the carry is decoded RGB.
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-2.3-22b-dev:fp8".into();

        let config = state.config.read().await;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("a two-stage LTX-2 checkpoint must pass chain preflight");
    }

    /// The door is where a chain's audio default is resolved, and an LTX-2
    /// sequence that never mentions the field renders WITH sound — the same
    /// answer the model's one-shot has always given for MP4.
    #[cfg(feature = "mp4")]
    #[tokio::test]
    async fn chain_preflight_resolves_an_unset_audio_default_on_for_ltx2() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-2.3-22b-dev:fp8".into();
        assert_eq!(request.enable_audio, None, "the fixture asks for nothing");

        let config = state.config.read().await;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("an LTX-2 chain passes preflight");
        assert_eq!(
            request.enable_audio,
            Some(true),
            "an unset flag on an audio family must resolve to sound"
        );
    }

    /// A build with no mp4 muxer cannot deliver a chain's audio at all, so
    /// the default must not mint a request that build would fail to finish —
    /// `validate_chain_build_features` only refuses an EXPLICIT ask, and it
    /// runs before this door.
    #[cfg(not(feature = "mp4"))]
    #[tokio::test]
    async fn chain_preflight_resolves_audio_off_without_the_mp4_muxer() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-2.3-22b-dev:fp8".into();

        let config = state.config.read().await;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("an LTX-2 chain passes preflight");
        assert_eq!(request.enable_audio, Some(false));
    }

    /// A container that cannot carry a track resolves OFF. The stitched print
    /// would transcode the audio away, so defaulting it on would pay full GPU
    /// cost per stage for nothing.
    #[cfg(feature = "mp4")]
    #[tokio::test]
    async fn chain_preflight_resolves_audio_off_for_a_silent_container() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Gif);
        request.model = "ltx-2.3-22b-dev:fp8".into();

        let config = state.config.read().await;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("an LTX-2 chain passes preflight");
        assert_eq!(request.enable_audio, Some(false));
    }

    /// A family with no audio decode path resolves OFF, never `Some(true)` —
    /// which its own admission would refuse by name.
    #[test]
    fn chain_preflight_resolves_an_unset_audio_default_off_for_a_silent_family() {
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-video-2b:fp16".into();
        let mut config = mold_core::Config::default();
        config.models.insert(
            request.model.clone(),
            mold_core::ModelConfig {
                family: Some("ltx-video".into()),
                ..Default::default()
            },
        );

        validate_and_normalize_chain_family(&config, &mut request)
            .expect("an ltx-video chain passes preflight");
        assert_eq!(
            request.enable_audio,
            Some(false),
            "ltx-video has no audio decode path; the default must stay silent"
        );
    }

    /// An explicit `false` survives the door. This is the silent-regression
    /// guard: with the default flipped, a client that means "no sound" must
    /// say so, and the door must not overwrite it.
    #[tokio::test]
    async fn chain_preflight_keeps_an_explicit_audio_refusal() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-2.3-22b-dev:fp8".into();
        request.enable_audio = Some(false);

        let config = state.config.read().await;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("an LTX-2 chain passes preflight");
        assert_eq!(request.enable_audio, Some(false));
    }

    #[tokio::test]
    async fn chain_preflight_still_rejects_a_family_that_does_not_chain() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "flux-dev:q4".into();

        let config = state.config.read().await;
        let error = validate_and_normalize_chain_family(&config, &mut request)
            .expect_err("a still-image family must be rejected at the API boundary");
        assert!(
            error.error.contains("flux"),
            "the rejection must name the family, got: {}",
            error.error
        );
    }

    fn ltx2_chain_config(model: &str) -> mold_core::Config {
        let mut config = mold_core::Config::default();
        config.models.insert(
            model.to_string(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..Default::default()
            },
        );
        config
    }

    #[test]
    fn chain_artifact_gate_ignores_a_gated_name_in_the_trusted_models_root() {
        let dir = tempfile::tempdir().unwrap();
        let models_root = dir.path().join("minimax-h3-uat/models");
        let ordinary = models_root.join("ordinary-ltx2");
        std::fs::create_dir_all(&ordinary).unwrap();
        let transformer = ordinary.join("transformer.safetensors");
        let vae = ordinary.join("vae.safetensors");
        let lora = ordinary.join("style.safetensors");
        for path in [&transformer, &vae, &lora] {
            std::fs::write(path, b"ordinary fixture").unwrap();
        }
        let mut config = mold_core::Config {
            models_dir: models_root.display().to_string(),
            ..mold_core::Config::default()
        };
        config.models.insert(
            "ordinary-chain-model".into(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                lora: Some(lora.display().to_string()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "ordinary-chain-model".into();
        request.stages[0].loras.push(mold_core::chain::LoraSpec {
            path: lora.display().to_string(),
            scale: 1.0,
            name: None,
        });

        require_chain_artifact_activation(&config, &request, None, None)
            .expect("the trusted root's name is storage placement, not model identity");
    }

    #[test]
    fn chain_preflight_rejects_ltx2_dimensions_aligned_only_to_16() {
        let mut request = req(OutputFormat::Mp4);
        request.width = 1008;
        request.height = 704;
        let config = ltx2_chain_config(&request.model);

        let error = validate_and_normalize_chain_family(&config, &mut request)
            .expect_err("LTX-2 chain must reject a 16px-only canvas");

        assert!(error.error.contains("multiples of 32"), "{}", error.error);
        assert!(error.error.contains("ltx2"), "{}", error.error);
    }

    #[test]
    fn chain_preflight_accepts_custom_ltx2_dimensions_aligned_to_32() {
        let mut request = req(OutputFormat::Mp4);
        request.width = 1056;
        request.height = 736;
        let config = ltx2_chain_config(&request.model);

        validate_and_normalize_chain_family(&config, &mut request)
            .expect("custom 32px-aligned LTX-2 canvas should be admitted");
    }

    /// Dimension admission on the chain route is per model, not per family:
    /// `wan22-ti2v-5b`'s 2.2 VAE needs a 32px grid, so a 16px-only canvas is
    /// named as off-grid instead of surviving to a later check.
    #[test]
    fn chain_preflight_rejects_wan22_ti2v_5b_16px_canvas() {
        let mut request = req(OutputFormat::Mp4);
        request.model = "wan22-ti2v-5b".into();
        request.width = 1280;
        request.height = 720;
        let config = mold_core::Config::default();

        let error = validate_and_normalize_chain_family(&config, &mut request)
            .expect_err("the 5B must reject a canvas off its 32px grid");
        assert!(error.error.contains("multiples of 32"), "{}", error.error);
    }

    /// Wan chain stages must sit on wan's own `4k+1` grid (#783).
    ///
    /// The default request's 9- and 17-frame stages are on `8k+1` and happen
    /// to be on `4k+1` too, so this uses a value that separates them: 13 is a
    /// valid wan clip and an invalid LTX one, and 12 is invalid for both.
    #[test]
    fn chain_preflight_holds_wan_stages_to_the_4k_plus_1_grid() {
        let mut request = req(OutputFormat::Mp4);
        request.model = "wan22-ti2v-5b".into();
        request.width = 1280;
        request.height = 704;
        request.stages[0].frames = 13;
        request.stages[1].frames = 12;
        let config = mold_core::Config::default();

        let error = validate_and_normalize_chain_family(&config, &mut request)
            .expect_err("12 is off wan's 4k+1 grid");
        assert!(error.error.contains("4k+1"), "{}", error.error);
        assert!(error.error.contains("stage 1"), "{}", error.error);

        // 13 alone is admitted — it is on wan's grid even though it is off the
        // LTX grid the check used to hardcode.
        request.stages[1].frames = 13;
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("13-frame wan clips are on the 4k+1 grid");
    }

    /// Wan carries at most the one frame its continuation was seeded with, so
    /// a client's LTX-shaped motion tail is normalized rather than honoured.
    /// A text-to-video checkpoint carries nothing at all.
    #[test]
    fn chain_preflight_normalizes_the_wan_motion_tail() {
        let config = mold_core::Config::default();

        // TI2V-5B conditions on an image (latent inpaint), so it keeps one
        // frame of seam — never the 17 an LTX-shaped client would send.
        let mut conditioned = req(OutputFormat::Mp4);
        conditioned.model = "wan22-ti2v-5b".into();
        conditioned.width = 1280;
        conditioned.height = 704;
        conditioned.motion_tail_frames = 17;
        validate_and_normalize_chain_family(&config, &mut conditioned).expect("admitted");
        assert_eq!(
            conditioned.motion_tail_frames,
            mold_inference::wan::pipeline::WAN_HANDOFF_DUPLICATED_FRAMES,
        );

        // A text-to-video checkpoint has no conditioning channel, so nothing
        // crosses its seam.
        let mut t2v = req(OutputFormat::Mp4);
        t2v.model = "wan21-t2v-1.3b".into();
        t2v.width = 832;
        t2v.height = 480;
        t2v.motion_tail_frames = 17;
        validate_and_normalize_chain_family(&config, &mut t2v).expect("admitted");
        assert_eq!(t2v.motion_tail_frames, 0);
    }

    /// Chain admission holds a checkpoint to its own conditioning contract
    /// (#783).
    ///
    /// The tail was normalized here since #936, but the contract that decides
    /// the tail was never enforced: `enforce_source_image_capability` lives on
    /// the single-generation path and the placement preview, and nothing on
    /// the chain path called it. So a wan A14B I2V sequence with no opening
    /// image was admitted and then died after the UMT5 encode and both expert
    /// loads were paid for, and a text-to-video sequence carrying an opening
    /// image was admitted too. This is the acceptance criterion "T2V-only
    /// checkpoints offer Cut/Crossfade only or are excluded, per advertised
    /// capability".
    #[test]
    fn chain_preflight_holds_a_checkpoint_to_its_source_image_contract() {
        let config = mold_core::Config::default();
        let opening_image = |request: &mut ChainRequest, image: Option<Vec<u8>>| {
            request.stages[0].source_image = image;
        };

        // A14B I2V denoises a 36-channel mask-plus-image concat: without an
        // image there is nothing to seed stage 0 with.
        let mut required = req(OutputFormat::Mp4);
        required.model = "wan22-i2v-a14b:q5".into();
        required.width = 832;
        required.height = 480;
        opening_image(&mut required, None);
        let error = validate_and_normalize_chain_family(&config, &mut required)
            .expect_err("a Required checkpoint cannot open a sequence unseeded");
        assert!(
            error.error.to_lowercase().contains("source image"),
            "got: {}",
            error.error
        );

        // The same checkpoint with an opening image is admitted.
        let mut seeded = req(OutputFormat::Mp4);
        seeded.model = "wan22-i2v-a14b:q5".into();
        seeded.width = 832;
        seeded.height = 480;
        opening_image(&mut seeded, Some(vec![1, 2, 3]));
        validate_and_normalize_chain_family(&config, &mut seeded).expect("admitted");
        assert_eq!(
            seeded.motion_tail_frames,
            mold_inference::wan::pipeline::WAN_HANDOFF_DUPLICATED_FRAMES
        );

        // A text-to-video checkpoint has no conditioning channel at all, so
        // an attached opening image is refused rather than ignored.
        let mut unsupported = req(OutputFormat::Mp4);
        unsupported.model = "wan21-t2v-1.3b".into();
        unsupported.width = 832;
        unsupported.height = 480;
        opening_image(&mut unsupported, Some(vec![1, 2, 3]));
        assert!(
            validate_and_normalize_chain_family(&config, &mut unsupported).is_err(),
            "a T2V checkpoint has no channel for an opening image"
        );

        // And without one it is admitted, carrying nothing across its seams.
        let mut plain_t2v = req(OutputFormat::Mp4);
        plain_t2v.model = "wan21-t2v-1.3b".into();
        plain_t2v.width = 832;
        plain_t2v.height = 480;
        opening_image(&mut plain_t2v, None);
        validate_and_normalize_chain_family(&config, &mut plain_t2v).expect("admitted");
        assert_eq!(plain_t2v.motion_tail_frames, 0);

        // A script can attach an image to any stage, so an unsupported
        // checkpoint is refused for one on a continuation too — the two arms
        // read different sets: `Required` asks only whether stage 0 can be
        // seeded, because every continuation is seeded by the seam.
        let mut late_image = req(OutputFormat::Mp4);
        late_image.model = "wan21-t2v-1.3b".into();
        late_image.width = 832;
        late_image.height = 480;
        assert!(
            late_image.stages.len() > 1,
            "this case needs a continuation to attach to"
        );
        late_image.stages[1].source_image = Some(vec![1, 2, 3]);
        assert!(
            validate_and_normalize_chain_family(&config, &mut late_image).is_err(),
            "an image on a continuation is still an image the engine cannot take"
        );

        // The mirror: a Required checkpoint seeded only on a continuation is
        // still unseeded where it matters.
        let mut late_only = req(OutputFormat::Mp4);
        late_only.model = "wan22-i2v-a14b:q5".into();
        late_only.width = 832;
        late_only.height = 480;
        opening_image(&mut late_only, None);
        late_only.stages[1].source_image = Some(vec![1, 2, 3]);
        assert!(validate_and_normalize_chain_family(&config, &mut late_only).is_err());

        // The top-level `source_image` belongs to the auto-expand form.
        // `normalise` projects it onto stage 0 only when `stages` is empty and
        // clears it otherwise, so counting it alongside explicit stages
        // admitted a request whose image is discarded before execution — the
        // I2V model loaded, then failed on an unseeded stage 0.
        let mut mixed_form = req(OutputFormat::Mp4);
        mixed_form.model = "wan22-i2v-a14b:q5".into();
        mixed_form.width = 832;
        mixed_form.height = 480;
        opening_image(&mut mixed_form, None);
        mixed_form.source_image = Some(vec![1, 2, 3]);
        assert!(
            !mixed_form.stages.is_empty(),
            "the defect needs explicit stages beside the top-level image"
        );
        assert!(
            validate_and_normalize_chain_family(&config, &mut mixed_form).is_err(),
            "an image `normalise` is about to discard cannot satisfy the contract"
        );

        // The auto-expand form still seeds stage 0 from that same field.
        let mut auto_expand = req(OutputFormat::Mp4);
        auto_expand.model = "wan22-i2v-a14b:q5".into();
        auto_expand.width = 832;
        auto_expand.height = 480;
        auto_expand.stages = Vec::new();
        auto_expand.prompt = Some("a balloon lifts off".into());
        auto_expand.total_frames = Some(106);
        auto_expand.clip_frames = Some(53);
        auto_expand.source_image = Some(vec![1, 2, 3]);
        validate_and_normalize_chain_family(&config, &mut auto_expand).expect("admitted");
    }

    /// A ONE-SHOT long video with no context handoff is refused at this door.
    ///
    /// The bug: a 259-frame one-shot submitted from the web Studio on
    /// `wan21-t2v-1.3b:turbo` was admitted here as an ephemeral three-stage
    /// chain (121/121/17, every stage on the same seed with a zero seam) and
    /// the delivered video reset at both boundaries — the second one a hard
    /// cut to a different composition. The tier declares
    /// `source_image: Unsupported`, so there is nothing to hand across a clip
    /// boundary and each stage re-derives the scene from the same prompt.
    ///
    /// The refusal is scoped to `ephemeral`: an AUTHORED sequence that repeats
    /// stages is what its author asked for, and admission must not second-guess
    /// it. The sentence is `mold_core::chain::text_only_auto_chain_refusal`'s,
    /// so the CLI, the Studio router, and this 422 all read the same.
    #[test]
    fn chain_preflight_refuses_a_text_only_wan_one_shot_auto_chain() {
        let config = mold_core::Config::default();
        let one_shot = |model: &str, width: u32, height: u32, ephemeral: bool| {
            let mut request = req(OutputFormat::Mp4);
            request.model = model.into();
            request.width = width;
            request.height = height;
            request.motion_tail_frames = 0;
            request.stages = Vec::new();
            request.prompt = Some("a village at dusk".into());
            request.total_frames = Some(259);
            request.clip_frames = None;
            request.ephemeral = ephemeral;
            request
        };

        let mut refused = one_shot("wan21-t2v-1.3b:turbo", 832, 480, true);
        let error = validate_and_normalize_chain_family(&config, &mut refused)
            .expect_err("a text-only tier cannot be auto-chained into a longer video");
        assert_eq!(
            error.error,
            mold_core::chain::text_only_auto_chain_refusal(
                Some("wan"),
                "wan21-t2v-1.3b:turbo",
                Some(mold_core::SourceImageCapability::Unsupported),
                259,
                121,
            )
            .expect("core refuses this one"),
            "the 422 must render mold-core's sentence verbatim"
        );
        assert_eq!(
            error.code, "VALIDATION_ERROR",
            "a client mistake, not a hold"
        );

        // The same request AUTHORED is still admitted: repeated stages are
        // what the author asked for.
        let mut authored = one_shot("wan21-t2v-1.3b:turbo", 832, 480, false);
        validate_and_normalize_chain_family(&config, &mut authored)
            .expect("an authored wan sequence is untouched by the one-shot rule");

        // Legacy LTX-Video has the same deterministic failure mode: the
        // engine ignores carry while prompt and seed stay fixed. The family
        // itself is authoritative even if a custom entry has no manifest.
        let mut legacy = one_shot("ltx-video-0.9.6-distilled:bf16", 1216, 704, true);
        let error = validate_and_normalize_chain_family(&config, &mut legacy)
            .expect_err("legacy LTX-Video cannot honestly auto-chain");
        assert_eq!(
            error.error,
            mold_core::chain::text_only_auto_chain_refusal(
                Some("ltx-video"),
                "ltx-video-0.9.6-distilled:bf16",
                Some(mold_core::SourceImageCapability::Unsupported),
                259,
                97,
            )
            .expect("core refuses the repeated legacy chain"),
        );

        let mut authored_legacy = one_shot("ltx-video-0.9.6-distilled:bf16", 1216, 704, false);
        validate_and_normalize_chain_family(&config, &mut authored_legacy)
            .expect("an authored legacy sequence still joins independent clips");

        // The explicit-stages shape of the same one-shot is refused too — the
        // CLI's own auto-chain posts stages, not the auto-expand sugar.
        let mut staged = one_shot("wan21-t2v-1.3b:turbo", 832, 480, true);
        staged.prompt = None;
        staged.total_frames = None;
        staged.stages = vec![
            ChainStage {
                prompt: "a village at dusk".into(),
                frames: 121,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "a village at dusk".into(),
                frames: 121,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
        ];
        assert!(
            validate_and_normalize_chain_family(&config, &mut staged).is_err(),
            "an ephemeral multi-stage wan T2V job repeats its clip whatever shape it arrives in"
        );

        // An image-conditioned tier seeds each continuation from the previous
        // clip's final frame, so its one-shot still chains.
        let mut admitted = one_shot("wan22-ti2v-5b:turbo", 1280, 704, true);
        validate_and_normalize_chain_family(&config, &mut admitted)
            .expect("an image-conditioned tier carries its seam and still chains");

        // At or below the tier's own clip size there is no chain to refuse.
        let mut single = one_shot("wan21-t2v-1.3b:turbo", 832, 480, true);
        single.total_frames = Some(121);
        validate_and_normalize_chain_family(&config, &mut single)
            .expect("one clip of the tier's own routing size is one render");
    }

    /// `POST /api/generate/chain/validate` must never refuse what
    /// `POST /api/chain-jobs` admits.
    ///
    /// `validate_chain` calls `validate_and_normalize_chain_family` BEFORE and
    /// AFTER `normalise`, while `create_chain_job` calls it once; `normalise`
    /// clears `clip_frames` along with the rest of the auto-expand sugar. A
    /// derivation that read `req.clip_frames` therefore saw the caller's clip
    /// on the first pass and the tier's routing default on the second, so
    /// `{total_frames: 97, clip_frames: 97}` on a 73-frame-routing tier passed
    /// validate's first call, normalised to ONE 97-frame stage, and was refused
    /// by its second — validate saying no to a body chain-jobs says yes to,
    /// which is exactly the disagreement #783 closed.
    ///
    /// Deriving the clip from the stages when there are stages fixes it at the
    /// root: the largest stage IS the clip the caller rendered with, and it
    /// survives normalisation. The CLI depends on the same thing — it
    /// normalises client-side, so its ephemeral chain arrives with stages and
    /// no `clip_frames` at all.
    #[test]
    fn chain_preflight_agrees_with_itself_across_normalise() {
        let config = mold_core::Config::default();
        let one_shot_clip = || {
            let mut request = req(OutputFormat::Mp4);
            // Routing clip 73 (family floor 53 raised by the tier's own
            // recorded default), and the caller asked for one 97-frame clip.
            request.model = "wan22-t2v-a14b:q8".into();
            request.width = 832;
            request.height = 480;
            request.motion_tail_frames = 0;
            request.stages = Vec::new();
            request.prompt = Some("a village at dusk".into());
            request.total_frames = Some(97);
            request.clip_frames = Some(97);
            request.ephemeral = true;
            request
        };

        // The validate endpoint's own sequence: preflight, normalise,
        // preflight again. Both preflights must reach the same verdict.
        let mut request = one_shot_clip();
        validate_and_normalize_chain_family(&config, &mut request)
            .expect("one 97-frame clip is one generation, not a repeated chain");
        let mut normalized = request
            .normalise_with_family(Some("wan"))
            .expect("the auto-expand form normalises");
        assert_eq!(
            normalized.stages.len(),
            1,
            "97 frames at a 97-frame clip is a single stage"
        );
        validate_and_normalize_chain_family(&config, &mut normalized).expect(
            "the post-normalise preflight must agree with the pre-normalise one, or              validate refuses a body chain-jobs admits",
        );

        // And the refusal still stands on the shape that survives normalise:
        // an ephemeral multi-stage chain whose stitched total runs past the
        // clip each stage rendered.
        let mut staged = one_shot_clip();
        staged.prompt = None;
        staged.total_frames = None;
        staged.clip_frames = None;
        staged.stages = (0..3)
            .map(|idx| ChainStage {
                prompt: "a village at dusk".into(),
                frames: if idx == 2 { 17 } else { 73 },
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            })
            .collect();
        let error = validate_and_normalize_chain_family(&config, &mut staged)
            .expect_err("three clips of one repeated scene is not a longer video");
        assert_eq!(
            error.error,
            mold_core::chain::text_only_auto_chain_refusal(
                Some("wan"),
                "wan22-t2v-a14b:q8",
                Some(mold_core::SourceImageCapability::Unsupported),
                73 + 73 + 17,
                73,
            )
            .expect("core refuses this one"),
        );
    }
}
