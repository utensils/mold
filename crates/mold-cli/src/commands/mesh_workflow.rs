//! `mold mesh-workflow` — durable multi-stage 3-D jobs.
//!
//! A workflow is not a one-shot render. Every stage — the picture a
//! text-to-3-D run starts from, its matted and delighted copies, the shape,
//! the paint — is admitted as its own generation and keeps its own retained
//! artifact, the job survives a server restart, and a failure resumes from
//! the first unfinished stage instead of rerunning the whole thing.
//!
//! Consequently every verb here is remote. The job's manifest, its media and
//! its queue rows live in ONE host's data root, so there is no local form to
//! fall back to and `--local` is refused by name rather than quietly running
//! something else.
//!
//! The request is built exactly as the web and desktop 3-D Studio builds it
//! (`studio/lib/meshWorkflowAuthoring.ts`): a GLB output pinned at 0x0, one
//! `kind: "mesh"` reference carrying its format, byte length, coordinates and
//! provenance, and `batch_size = 1` on every stage. The CLI is the first
//! surface to expose the whole `mesh` block on a durable workflow — the
//! Studio authors only texture, texture resolution and delight — so the
//! octree ladder, the iso-level and the decimation target are flags here.

use std::path::Path;

use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::generation_profile::MeshWorkflowMode;
use mold_core::mesh_workflow::{
    validate_create_mesh_workflow, CreateMeshWorkflowRequest, MeshWorkflowEvent,
    MeshWorkflowJobDetail, MeshWorkflowJobState, MeshWorkflowJobSummary, MeshWorkflowStageRecord,
    MeshWorkflowStageState,
};
use mold_core::{
    GenerateRequest, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance, MeshReferenceCoordinates, MeshReferenceFormat,
    MeshRequestOptions, MeshUpAxis, ModelDefaults, MoldClient, OutputFormat,
};

use crate::{theme, MeshMattingArg, MeshWorkflowAction, MeshWorkflowModeArg};

/// `mold mesh-workflow create`, resolved from clap.
pub struct CreateArgs {
    pub prompt: Option<String>,
    pub mode: Option<MeshWorkflowModeArg>,
    pub model: Option<String>,
    pub image_model: Option<String>,
    pub image: Option<std::path::PathBuf>,
    pub mesh: Option<std::path::PathBuf>,
    pub up_axis: Option<MeshUpAxis>,
    pub meters_per_unit: Option<f64>,
    pub texture: bool,
    pub no_texture: bool,
    pub texture_resolution: Option<u32>,
    pub matting: Option<MeshMattingArg>,
    pub delight: bool,
    pub octree: Option<u32>,
    pub threshold: Option<f32>,
    pub target_faces: Option<u32>,
    pub seed: Option<u64>,
    pub follow: bool,
    pub json: bool,
    pub local: bool,
}

pub async fn run(host: Option<&str>, action: MeshWorkflowAction) -> Result<()> {
    let client = crate::control::client_for_host(host);
    match action {
        MeshWorkflowAction::Create {
            prompt,
            mode,
            model,
            image_model,
            image,
            mesh,
            up_axis,
            meters_per_unit,
            texture,
            no_texture,
            texture_resolution,
            matting,
            delight,
            octree,
            threshold,
            target_faces,
            seed,
            follow,
            json,
            local,
        } => {
            create(
                &client,
                CreateArgs {
                    prompt,
                    mode,
                    model,
                    image_model,
                    image,
                    mesh,
                    up_axis,
                    meters_per_unit,
                    texture,
                    no_texture,
                    texture_resolution,
                    matting,
                    delight,
                    octree,
                    threshold,
                    target_faces,
                    seed,
                    follow,
                    json,
                    local,
                },
            )
            .await
        }
        MeshWorkflowAction::List { json } => list(&client, json).await,
        MeshWorkflowAction::Show { id, json } => {
            show(&client, require_workflow_id(&id)?, json).await
        }
        MeshWorkflowAction::Events { id } => {
            follow_workflow(&client, require_workflow_id(&id)?).await?;
            Ok(())
        }
        MeshWorkflowAction::Resume { id } => {
            let id = require_workflow_id(&id)?;
            client.resume_mesh_workflow(id).await?;
            println!("{} resumed {id}", theme::icon_ok());
            Ok(())
        }
        MeshWorkflowAction::Cancel { id } => {
            let id = require_workflow_id(&id)?;
            client.cancel_mesh_workflow(id).await?;
            println!("{} cancelled {id}", theme::icon_ok());
            Ok(())
        }
        MeshWorkflowAction::Delete { id } => {
            let id = require_workflow_id(&id)?;
            client.delete_mesh_workflow(id).await?;
            println!(
                "{} deleted {id} and the artifacts it retained",
                theme::icon_ok()
            );
            Ok(())
        }
    }
}

/// The refusal a `--local` workflow gets.
///
/// Named rather than ignored: a user who reaches for it is asking for
/// something this command cannot mean, and the answer — a one-shot render —
/// is a different command, not a different flag.
pub const LOCAL_REFUSAL: &str =
    "a 3-D workflow is durable on one machine, so there is no local form of it: \
its manifest, its stage artifacts and its queue rows live in that server's data root. \
Render locally with `mold run <model> --format glb`, or point MOLD_HOST at a server.";

/// Which workflow the flags describe.
///
/// Inference is by what was SUPPLIED, because each mode needs a different
/// input and no two of them need the same one: a prompt is a picture to
/// render, a mesh with an appearance image is a mesh to paint, and a mesh on
/// its own is a mesh to rebuild. `--mode` names it outright when a script
/// would rather not depend on that.
pub fn resolve_mode(args: &CreateArgs) -> Result<MeshWorkflowModeArg> {
    if let Some(mode) = args.mode {
        return Ok(mode);
    }
    match (
        args.prompt.is_some(),
        args.mesh.is_some(),
        args.image.is_some(),
    ) {
        (true, false, _) => Ok(MeshWorkflowModeArg::TextToMesh),
        (false, true, true) => Ok(MeshWorkflowModeArg::MeshTexture),
        (false, true, false) => Ok(MeshWorkflowModeArg::MeshRoundtrip),
        (true, true, _) => bail!(
            "--prompt starts a text-to-3-D run and --mesh supplies one to work from; \
             they name different workflows. Pick one, or say which with --mode."
        ),
        (false, false, _) => bail!(
            "nothing to work from: give --prompt to render a picture and reconstruct it, \
             or --mesh to texture or rebuild a mesh you already have."
        ),
    }
}

/// The 3-D model this mode runs on when `--model` names none.
///
/// The modes do not share a checkpoint, so neither can their default:
/// `mold_core::generation_profile::default_model_for_mesh_workflow_mode` is
/// the one authority, and it sits beside the function that decides which
/// modes a recipe advertises so the two cannot disagree.
pub fn default_model_for(mode: MeshWorkflowModeArg) -> &'static str {
    mold_core::generation_profile::default_model_for_mesh_workflow_mode(wire_mode(mode))
}

/// The wire mode this flag names.
fn wire_mode(mode: MeshWorkflowModeArg) -> MeshWorkflowMode {
    match mode {
        MeshWorkflowModeArg::TextToMesh => MeshWorkflowMode::TextToMesh,
        MeshWorkflowModeArg::MeshTexture => MeshWorkflowMode::MeshTexture,
        MeshWorkflowModeArg::MeshRoundtrip => MeshWorkflowMode::MeshRoundtrip,
    }
}

/// Refuse a checkpoint the chosen mode cannot run, naming the flag that fixes
/// it.
///
/// `validate_create_mesh_workflow` refuses the same request, but its sentence
/// is written for the HTTP door and names no flag — "mesh-roundtrip requires
/// a Hunyuan3D 2.1 shape checkpoint" leaves a terminal user to guess both
/// which model that is and how to select it. Checked before the request is
/// built so nothing is read from disk first.
fn refuse_a_model_the_mode_cannot_run(mode: MeshWorkflowModeArg, model: &str) -> Result<()> {
    if mode == MeshWorkflowModeArg::MeshRoundtrip
        && !mold_core::manifest::hunyuan3d_shape21_model(model)
    {
        bail!(
            "a roundtrip runs through the 2.1 shape VAE, and '{model}' has none; \
             pass --model {}",
            default_model_for(MeshWorkflowModeArg::MeshRoundtrip)
        );
    }
    Ok(())
}

/// Refuse an empty workflow id by name.
///
/// An empty path segment resolves to the LIST route, so the id every
/// lifecycle verb takes would otherwise be answered by a listing the client
/// tries to read as one job: `error decoding response body: expected value at
/// line 1 column 1`, which says nothing about the argument that was blank.
pub fn require_workflow_id(id: &str) -> Result<&str> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        bail!("a workflow id is required; `mold mesh-workflow list` shows the ids on this machine");
    }
    Ok(trimmed)
}

async fn create(client: &MoldClient, args: CreateArgs) -> Result<()> {
    if args.local {
        bail!("{LOCAL_REFUSAL}");
    }
    let mode = resolve_mode(&args)?;
    refuse_flags_the_mode_cannot_use(mode, &args)?;

    let mesh_model = args
        .model
        .clone()
        .unwrap_or_else(|| default_model_for(mode).to_string());
    let mesh_model = mold_core::manifest::resolve_model_name(&mesh_model);
    refuse_a_model_the_mode_cannot_run(mode, &mesh_model)?;

    let models = client
        .list_models_extended()
        .await
        .with_context(|| format!("could not read the models on {}", client.host()))?;
    let mesh_defaults = defaults_for(&models, &mesh_model)?;

    let request = match mode {
        MeshWorkflowModeArg::TextToMesh => {
            let image_model = args
                .image_model
                .clone()
                .unwrap_or_else(|| mold_core::Config::load_or_default().resolved_default_model());
            let image_model = mold_core::manifest::resolve_model_name(&image_model);
            let image_defaults = defaults_for(&models, &image_model)?;
            let mut image_request =
                stage_request(&image_model, &image_defaults, OutputFormat::Png, args.seed);
            image_request.prompt = args
                .prompt
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .to_string();
            let mut mesh_request =
                stage_request(&mesh_model, &mesh_defaults, OutputFormat::Glb, args.seed);
            mesh_request.mesh = Some(mesh_options(&args, texture_for(mode, &args)));
            CreateMeshWorkflowRequest::TextToMesh {
                image_request: Box::new(image_request),
                mesh_request: Box::new(mesh_request),
            }
        }
        MeshWorkflowModeArg::MeshTexture => {
            let mut request =
                stage_request(&mesh_model, &mesh_defaults, OutputFormat::Glb, args.seed);
            let appearance = args.image.as_deref().expect("checked by the mode rules");
            request.source_image = Some(read_media(appearance)?);
            request.source_image_name = file_name(appearance);
            request.mesh = Some(mesh_options(&args, true));
            request.references = Some(vec![mesh_reference(&args)?]);
            CreateMeshWorkflowRequest::MeshTexture {
                texture_request: Box::new(request),
            }
        }
        MeshWorkflowModeArg::MeshRoundtrip => {
            let mut request =
                stage_request(&mesh_model, &mesh_defaults, OutputFormat::Glb, args.seed);
            request.mesh = Some(mesh_options(&args, false));
            request.references = Some(vec![mesh_reference(&args)?]);
            CreateMeshWorkflowRequest::MeshRoundtrip {
                roundtrip_request: Box::new(request),
            }
        }
    };

    // Validate here so an obvious mistake reads as a sentence about the flags
    // rather than as an HTTP status. The server runs the same function plus
    // per-recipe validation it alone can do.
    validate_create_mesh_workflow(&request).map_err(|error| anyhow::anyhow!(error))?;

    let (request, lease) = lease_mesh_upload(client, request, args.mesh.as_deref()).await?;
    let created = match client.create_mesh_workflow(&request).await {
        Ok(created) => created,
        Err(error) => {
            if let Some(handle) = lease {
                let _ = client.cancel_reference_upload_session(&handle).await;
            }
            return Err(error);
        }
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&created)?);
    } else {
        println!("{} {}", "workflow".bold(), created.job_id);
        for warning in &created.request_warnings {
            println!("{} {warning}", theme::icon_warn());
        }
        let stages = request.planned_stage_kinds();
        println!(
            "  {} {}",
            "stages".dimmed(),
            stages
                .iter()
                .map(|stage| stage.as_str())
                .collect::<Vec<_>>()
                .join(" → ")
        );
        if !args.follow {
            println!(
                "  {} mold mesh-workflow show {}",
                "next".dimmed(),
                created.job_id
            );
        }
    }
    if args.follow {
        follow_workflow(client, &created.job_id).await?;
    }
    Ok(())
}

/// Refuse a flag the chosen mode cannot honour, by name.
///
/// Each of these would otherwise be dropped silently or refused by the
/// server in wording about a request field the user never typed.
fn refuse_flags_the_mode_cannot_use(mode: MeshWorkflowModeArg, args: &CreateArgs) -> Result<()> {
    match mode {
        MeshWorkflowModeArg::TextToMesh => {
            if args
                .prompt
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .is_empty()
            {
                bail!("a text-to-3-D workflow needs --prompt");
            }
            if args.mesh.is_some() {
                bail!("a text-to-3-D workflow renders its own picture; --mesh belongs to --mode mesh_texture or mesh_roundtrip");
            }
            if args.image.is_some() {
                bail!("a text-to-3-D workflow renders its own picture; --image belongs to --mode mesh_texture");
            }
        }
        MeshWorkflowModeArg::MeshTexture => {
            if args.mesh.is_none() {
                bail!("texturing a mesh needs --mesh");
            }
            if args.image.is_none() {
                bail!("texturing a mesh needs --image: the appearance the paint stage reads");
            }
            if args.prompt.is_some() {
                bail!("the Hunyuan3D family reads no prompt, so a texture run takes none");
            }
            if args.no_texture {
                bail!("--no-texture leaves a texture-only workflow with nothing to do; use --mode mesh_roundtrip to rebuild geometry");
            }
        }
        MeshWorkflowModeArg::MeshRoundtrip => {
            if args.mesh.is_none() {
                bail!("rebuilding a mesh needs --mesh");
            }
            if args.image.is_some() {
                bail!("a roundtrip rebuilds geometry only; --image belongs to --mode mesh_texture");
            }
            if args.prompt.is_some() {
                bail!("the Hunyuan3D family reads no prompt, so a roundtrip takes none");
            }
            if args.texture {
                bail!("a roundtrip reconstructs geometry and cannot paint it; texture the result with --mode mesh_texture");
            }
            // Both stages prepare a conditioning IMAGE, and a roundtrip has
            // none: its stage graph is shape then finalize. Sending either
            // would be a control the run never reaches.
            if args.matting.is_some() {
                bail!("a roundtrip conditions on the supplied mesh, not a picture, so --matting has nothing to remove");
            }
            if args.delight {
                bail!("a roundtrip conditions on the supplied mesh, not a picture, so --delight has nothing to relight");
            }
        }
    }
    if args.texture_resolution.is_some() && !texture_for(mode, args) {
        bail!("--texture-resolution needs --texture; it has no effect on a geometry-only run");
    }
    if args.image_model.is_some() && mode != MeshWorkflowModeArg::TextToMesh {
        bail!("--image-model names the model that renders a text-to-3-D run's picture; this workflow renders none");
    }
    if (args.up_axis.is_some() || args.meters_per_unit.is_some()) && args.mesh.is_none() {
        bail!("--up-axis and --meters-per-unit describe a supplied mesh, and this workflow builds its own");
    }
    Ok(())
}

/// Whether this run paints.
///
/// `mesh_texture` is texturing by definition, a roundtrip never paints, and
/// a text-to-3-D run is geometry-only unless asked — the same opt-in
/// `mold run --texture` uses, because the paint bundle is a separate
/// download and a request that assumes it is refused rather than answered
/// with bare geometry.
fn texture_for(mode: MeshWorkflowModeArg, args: &CreateArgs) -> bool {
    match mode {
        MeshWorkflowModeArg::MeshTexture => true,
        MeshWorkflowModeArg::MeshRoundtrip => false,
        MeshWorkflowModeArg::TextToMesh => args.texture && !args.no_texture,
    }
}

/// The `mesh` block, built from the flags that name its controls.
///
/// Absent stays absent: the engine's own defaults answer for an omitted
/// octree resolution, iso-level or decimation target, and `mold` records the
/// resolved values on the print rather than inventing them here.
pub fn mesh_options(args: &CreateArgs, texture: bool) -> MeshRequestOptions {
    MeshRequestOptions {
        octree_resolution: args.octree,
        threshold: args.threshold,
        target_faces: args.target_faces,
        texture: Some(texture),
        texture_resolution: args.texture_resolution.filter(|_| texture),
        matting: args.matting.map(MeshMattingArg::mode),
        delight: args.delight.then_some(true),
    }
}

/// One stage's request: the Studio's `requestFor`, in Rust.
///
/// A GLB stage is pinned to 0x0 because the mesh recipes are canvasless —
/// there is no picture to size — and every stage carries `batch_size = 1`,
/// which the workflow contract requires.
fn stage_request(
    model: &str,
    defaults: &ModelDefaults,
    output: OutputFormat,
    seed: Option<u64>,
) -> GenerateRequest {
    let glb = output == OutputFormat::Glb;
    GenerateRequest {
        mesh_workflow: None,
        offload: None,
        mesh: None,
        video_only: None,
        title: None,
        tags: None,
        collection: None,
        source_fit: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: String::new(),
        negative_prompt: None,
        model: model.to_string(),
        width: if glb { 0 } else { defaults.default_width },
        height: if glb { 0 } else { defaults.default_height },
        steps: defaults.default_steps,
        guidance: defaults.default_guidance,
        seed,
        batch_size: 1,
        output_format: Some(output),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        source_image_name: None,
        edit_images: None,
        reference_weight: None,
        references: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        save_to_gallery: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: None,
        pipeline: None,
        ic_lora_control: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
        id_image: None,
        id_image_name: None,
        id_weight: None,
        id_start_step: None,
        id_images: None,
        id_image_names: None,
        true_cfg: None,
        cfg_start_step: None,
    }
}

fn defaults_for(models: &[mold_core::ModelInfoExtended], name: &str) -> Result<ModelDefaults> {
    models
        .iter()
        .find(|model| model.info.name == name)
        .map(|model| model.defaults.clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model named '{name}' on this server. Run `mold list` to see what is there, \
                 or `mold pull {name}` to install it."
            )
        })
}

/// The container a supplied mesh is in, from its extension.
pub fn mesh_format_of(path: &Path) -> Result<MeshReferenceFormat> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("glb") | Some("gltf") => Ok(MeshReferenceFormat::Glb),
        Some("obj") => Ok(MeshReferenceFormat::Obj),
        _ => bail!(
            "--mesh takes a .glb or .obj file; {} is neither",
            path.display()
        ),
    }
}

/// The MIME type that container travels as.
pub fn mesh_mime_type(format: MeshReferenceFormat) -> &'static str {
    match format {
        MeshReferenceFormat::Glb => "model/gltf-binary",
        MeshReferenceFormat::Obj => "model/obj",
    }
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
}

fn read_media(path: &Path) -> Result<Vec<u8>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
    anyhow::ensure!(!bytes.is_empty(), "{} is empty", path.display());
    Ok(bytes)
}

/// The one mesh reference a supplied-mesh workflow carries.
///
/// Inline to begin with: the lease path, when the host offers one, rewrites
/// the authority after the bytes have streamed. The digest is computed either
/// way, because it is what binds the upload session to this exact file.
fn mesh_reference(args: &CreateArgs) -> Result<GenerationReference> {
    let path = args.mesh.as_deref().expect("checked by the mode rules");
    let format = mesh_format_of(path)?;
    let bytes = read_media(path)?;
    let byte_length = bytes.len() as u64;
    anyhow::ensure!(
        byte_length <= mold_core::validation::MESH_REFERENCE_MAX_BYTES,
        "{} is {} bytes; a mesh reference may be at most {} bytes",
        path.display(),
        byte_length,
        mold_core::validation::MESH_REFERENCE_MAX_BYTES
    );
    let sha256 = sha256_hex(&bytes);
    Ok(GenerationReference::Mesh {
        media: GenerationReferenceAuthority::Inline { data: bytes },
        provenance: GenerationReferenceProvenance {
            name: file_name(path),
            sha256: Some(sha256),
            crop: None,
        },
        mime_type: mesh_mime_type(format).to_string(),
        format,
        byte_length,
        coordinates: MeshReferenceCoordinates {
            up_axis: args.up_axis.unwrap_or(MeshUpAxis::Y),
            meters_per_unit: args.meters_per_unit.unwrap_or(1.0),
        },
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Stream a supplied mesh through a request-bound upload lease when the host
/// offers one, instead of carrying it as base64 in the request body.
///
/// The Studio's rule, verbatim: lease when the host advertises
/// `reference_uploads.available` AND this client is authenticated. Without a
/// key there is no identity to bind a session to, and a host that advertises
/// nothing is an older one that takes validated inline references. Returns
/// the session handle so a failed create can release the lease rather than
/// leaving it to expire.
/// Refuse a mesh the host's upload limits cannot take, by name and before a
/// session is opened.
///
/// Without this the run gets partway through `PUT /api/generate/reference-upload`
/// and dies on whatever the server says about a body it refused, with a
/// half-open lease behind it. `mold_core::reference_upload` checks the same
/// two numbers for the H3 path and the Studio's `validateCapabilities` checks
/// them in the browser; this is the third caller of the same rule.
///
/// One mesh is the whole session here, so the per-file and per-session limits
/// are both a bound on the same number — named separately because a host may
/// set them independently and the user needs to know which one they hit.
fn refuse_a_mesh_this_host_will_not_accept(
    capabilities: &mold_core::ReferenceUploadCapabilities,
    byte_length: u64,
) -> Result<()> {
    anyhow::ensure!(
        byte_length <= capabilities.max_file_bytes,
        "the mesh is {byte_length} bytes and this machine accepts at most {} per uploaded file",
        capabilities.max_file_bytes
    );
    anyhow::ensure!(
        byte_length <= capabilities.max_session_bytes,
        "the mesh is {byte_length} bytes and this machine accepts at most {} per upload session",
        capabilities.max_session_bytes
    );
    Ok(())
}

/// The supplied mesh's size, read from the filesystem rather than from the
/// bytes already in memory, so the check happens before anything is loaded.
fn mesh_byte_length(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)
        .with_context(|| format!("could not read {}", path.display()))?
        .len())
}

async fn lease_mesh_upload(
    client: &MoldClient,
    request: CreateMeshWorkflowRequest,
    mesh_path: Option<&Path>,
) -> Result<(CreateMeshWorkflowRequest, Option<String>)> {
    let Some(path) = mesh_path else {
        return Ok((request, None));
    };
    let Ok(capabilities) = client.capabilities().await else {
        return Ok((request, None));
    };
    if !capabilities.reference_uploads.available || !client.has_api_key() {
        return Ok((request, None));
    }
    refuse_a_mesh_this_host_will_not_accept(
        &capabilities.reference_uploads,
        mesh_byte_length(path)?,
    )?;

    let (mut inner, rebuild): (
        GenerateRequest,
        fn(GenerateRequest) -> CreateMeshWorkflowRequest,
    ) = match request {
        CreateMeshWorkflowRequest::MeshTexture { texture_request } => {
            (*texture_request, |request| {
                CreateMeshWorkflowRequest::MeshTexture {
                    texture_request: Box::new(request),
                }
            })
        }
        CreateMeshWorkflowRequest::MeshRoundtrip { roundtrip_request } => {
            (*roundtrip_request, |request| {
                CreateMeshWorkflowRequest::MeshRoundtrip {
                    roundtrip_request: Box::new(request),
                }
            })
        }
        // A text-to-3-D run supplies no media at all.
        other => return Ok((other, None)),
    };

    let format = mesh_format_of(path)?;
    // The session is bound to a payload-free request: every authority is
    // `descriptor` while the scope hash is computed, and the bytes arrive
    // afterwards under the slot handle the host hands back.
    let descriptor_references = inner
        .references
        .as_ref()
        .map(|references| {
            references
                .iter()
                .map(|reference| match reference.clone() {
                    GenerationReference::Mesh {
                        provenance,
                        mime_type,
                        format,
                        byte_length,
                        coordinates,
                        ..
                    } => GenerationReference::Mesh {
                        media: GenerationReferenceAuthority::Descriptor,
                        provenance,
                        mime_type,
                        format,
                        byte_length,
                        coordinates,
                    },
                    other => other,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut scoped = inner.clone();
    scoped.references = Some(descriptor_references.clone());

    let session = client
        .create_reference_upload_session(&mold_core::ReferenceUploadSessionRequest {
            request: scoped,
            upload_references: vec![1],
        })
        .await
        .context("could not open a reference-upload session for the mesh")?;
    let handle = session.session_handle.clone();
    let slot = session
        .uploads
        .iter()
        .find(|slot| slot.reference == 1)
        .map(|slot| slot.handle.clone())
        .ok_or_else(|| anyhow::anyhow!("the host opened a session without a slot for the mesh"))?;

    if let Err(error) = client
        .upload_reference_file(&slot, path, mesh_mime_type(format))
        .await
    {
        let _ = client.cancel_reference_upload_session(&handle).await;
        return Err(error.context("uploading the mesh failed"));
    }

    inner.references = Some(
        descriptor_references
            .into_iter()
            .map(|reference| match reference {
                GenerationReference::Mesh {
                    provenance,
                    mime_type,
                    format,
                    byte_length,
                    coordinates,
                    ..
                } => GenerationReference::Mesh {
                    media: GenerationReferenceAuthority::Upload {
                        handle: slot.clone(),
                    },
                    provenance,
                    mime_type,
                    format,
                    byte_length,
                    coordinates,
                },
                other => other,
            })
            .collect(),
    );
    Ok((rebuild(inner), Some(handle)))
}

async fn list(client: &MoldClient, json: bool) -> Result<()> {
    let listing = client.list_mesh_workflows().await?;
    // Teach the shell the ids this listing just showed: every other verb here
    // takes one, and a completer cannot ask a server (see
    // `crate::completion_cache`). The machine is recorded with them, because
    // a workflow lives on ONE host and the request just succeeded.
    crate::completion_cache::record_reached_host(client.host(), |cache| {
        cache.record_workflow_ids(listing.jobs.iter().map(|job| job.id.clone()));
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    if listing.jobs.is_empty() {
        println!(
            "{} No 3-D workflows on {}.",
            theme::icon_neutral(),
            client.host()
        );
        return Ok(());
    }
    // CHARACTERS, not bytes, because `{:<N}` pads by `char` count and sizing
    // a column from `str::len()` over-pads any row with a multi-byte
    // character in it.
    let id_width =
        crate::ui::col_width(listing.jobs.iter().map(|job| job.id.chars().count()), 2, 2);
    let mode_width = crate::ui::col_width(
        listing
            .jobs
            .iter()
            .map(|job| mode_label(job).chars().count()),
        4,
        2,
    );
    println!(
        "{:<id_width$} {:<mode_width$} {:<10} {:<9} {}",
        "ID".bold(),
        "MODE".bold(),
        "STATE".bold(),
        "STAGE".bold(),
        "OUTPUT".bold(),
    );
    println!("{}", "─".repeat(id_width + mode_width + 34).dimmed());
    for job in &listing.jobs {
        // Pad the plain text first: ANSI codes break `{:<N}`.
        println!(
            "{:<id_width$} {:<mode_width$} {} {:<9} {}",
            job.id,
            mode_label(job),
            colored_state(job.state, 10),
            format!("{}/{}", job.current_stage, job.stage_count),
            job.output_filename.as_deref().unwrap_or("—"),
        );
    }
    Ok(())
}

fn mode_label(job: &MeshWorkflowJobSummary) -> String {
    serde_json::to_value(job.mode)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn colored_state(state: MeshWorkflowJobState, width: usize) -> String {
    let padded = format!("{:<width$}", state.as_str());
    match state {
        MeshWorkflowJobState::Completed => padded.green().to_string(),
        MeshWorkflowJobState::Failed => padded.red().to_string(),
        MeshWorkflowJobState::Cancelled | MeshWorkflowJobState::Paused => {
            padded.yellow().to_string()
        }
        _ => padded,
    }
}

async fn show(client: &MoldClient, id: &str, json: bool) -> Result<()> {
    let detail = client.get_mesh_workflow(id).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&detail)?);
        return Ok(());
    }
    print_detail(&detail);
    Ok(())
}

fn print_detail(detail: &MeshWorkflowJobDetail) {
    let summary = &detail.summary;
    println!("{} {}", "workflow".bold(), summary.id);
    println!("  {:<10} {}", "mode".dimmed(), mode_label(summary));
    println!(
        "  {:<10} {}",
        "state".dimmed(),
        colored_state(summary.state, 0)
    );
    println!(
        "  {:<10} {} of {}",
        "stage".dimmed(),
        summary.current_stage,
        summary.stage_count
    );
    if let Some(output) = &summary.output_filename {
        println!("  {:<10} {output}", "output".dimmed());
    }
    if let Some(error) = &summary.error {
        println!("  {:<10} {}", "error".dimmed(), error.red());
    }
    if detail.stages.is_empty() {
        return;
    }
    println!();
    for stage in &detail.stages {
        print_stage(stage);
    }
}

fn print_stage(stage: &MeshWorkflowStageRecord) {
    println!(
        "  {} {:<9} {}",
        format!("{}.", stage.index).dimmed(),
        stage.kind.as_str(),
        stage.state.as_str()
    );
    if let Some(error) = &stage.error {
        println!("      {}", error.red());
    }
    for artifact in &stage.artifacts {
        println!(
            "      {} {} ({})",
            artifact.role.dimmed(),
            artifact.relative_path,
            crate::ui::format_disk_size(artifact.byte_length)
        );
    }
}

/// Follow one workflow to settlement, printing a line each time a stage
/// changes state, and exit non-zero when it settles as failed or cancelled.
///
/// The lines are DIFFED FROM SNAPSHOTS, not read from stage events, because
/// `GET /api/mesh-workflows/:id/events` emits nothing else: the server polls
/// its own record every 500 ms and yields a whole `Snapshot` whenever
/// `(updated_at_ms, state)` moves (`routes_mesh_workflows.rs`). A snapshot
/// carries the full `stages` list, so comparing each one against the last is
/// how a real per-stage line gets rendered; matching on the four
/// stage-shaped variants instead printed a repeated `running stage 1 of 3`
/// and nothing else.
async fn follow_workflow(client: &MoldClient, id: &str) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let printer = tokio::spawn(async move {
        let mut seen: Vec<MeshWorkflowStageState> = Vec::new();
        let mut state: Option<MeshWorkflowJobState> = None;
        while let Some(event) = rx.recv().await {
            match event {
                MeshWorkflowEvent::Snapshot { job } => {
                    for line in stage_transitions(&mut seen, &job.stages) {
                        println!("{line}");
                    }
                    if state != Some(job.summary.state) {
                        state = Some(job.summary.state);
                        if let Some(line) = state_line(job.summary.state, &job.summary.error) {
                            println!("{line}");
                        }
                    }
                }
                // The four variants below are forward-compatible with a
                // push-based event stream. No mold server constructs one
                // today — every frame is a `Snapshot` — so these are
                // unreachable against a real host and exist so that a server
                // which starts emitting them needs no client change.
                MeshWorkflowEvent::StageStarted { stage_index, kind } => {
                    println!("{} {stage_index} {}", theme::icon_neutral(), kind.as_str());
                }
                MeshWorkflowEvent::StageProgress {
                    stage_index,
                    kind,
                    current,
                    total,
                } => {
                    println!(
                        "  {stage_index} {} {current}/{total}",
                        kind.as_str().dimmed()
                    );
                }
                MeshWorkflowEvent::StageCompleted {
                    stage_index, kind, ..
                } => {
                    println!("{} {stage_index} {} done", theme::icon_ok(), kind.as_str());
                }
                MeshWorkflowEvent::StateChanged {
                    state: changed,
                    error,
                } => {
                    state = Some(changed);
                    if let Some(line) = state_line(changed, &error) {
                        println!("{line}");
                    }
                }
            }
        }
    });
    let outcome = client.stream_mesh_workflow_events(id, tx).await?;
    let _ = printer.await;
    match outcome.state {
        MeshWorkflowJobState::Completed => {
            println!(
                "{} {}",
                theme::icon_ok(),
                outcome.output_filename.as_deref().unwrap_or("completed")
            );
            Ok(())
        }
        MeshWorkflowJobState::Failed => bail!(
            "workflow {id} failed: {}",
            outcome.error.as_deref().unwrap_or("no reason recorded")
        ),
        MeshWorkflowJobState::Cancelled => bail!("workflow {id} was cancelled"),
        other => {
            println!("{} {} — still {}", theme::icon_warn(), id, other.as_str());
            Ok(())
        }
    }
}

/// The lines one snapshot owes, given the stage states the last one showed.
///
/// `seen` is the running record and is updated in place, so a stage that has
/// not moved prints nothing however many snapshots repeat it — which is the
/// whole point, since the server re-sends the entire job every time anything
/// about it changes. A stage the previous snapshot did not have at all
/// (the record grows as the runner admits stages) prints its current state,
/// so attaching late still shows what has happened.
pub fn stage_transitions(
    seen: &mut Vec<MeshWorkflowStageState>,
    stages: &[MeshWorkflowStageRecord],
) -> Vec<String> {
    let mut lines = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        if seen.get(index) == Some(&stage.state) {
            continue;
        }
        if index < seen.len() {
            seen[index] = stage.state;
        } else {
            seen.resize(index + 1, stage.state);
        }
        lines.push(stage_line(stage));
    }
    lines
}

fn stage_line(stage: &MeshWorkflowStageRecord) -> String {
    let icon = match stage.state {
        MeshWorkflowStageState::Completed => theme::icon_ok(),
        MeshWorkflowStageState::Failed => theme::icon_fail(),
        _ => theme::icon_neutral(),
    };
    let mut line = format!(
        "{icon} {} {} {}",
        stage.index,
        stage.kind.as_str(),
        stage.state.as_str()
    );
    if let Some(error) = &stage.error {
        line.push_str(&format!(" — {}", error.red()));
    }
    line
}

/// The line a job-level state change owes, or `None` while it is simply
/// running — that is the state every stage line already implies.
fn state_line(state: MeshWorkflowJobState, error: &Option<String>) -> Option<String> {
    match (state, error) {
        (MeshWorkflowJobState::Running, None) => None,
        (state, Some(error)) => Some(format!(
            "{} {} — {error}",
            theme::icon_fail(),
            state.as_str()
        )),
        (state, None) => Some(format!("{} {}", theme::icon_neutral(), state.as_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> CreateArgs {
        CreateArgs {
            prompt: None,
            mode: None,
            model: None,
            image_model: None,
            image: None,
            mesh: None,
            up_axis: None,
            meters_per_unit: None,
            texture: false,
            no_texture: false,
            texture_resolution: None,
            matting: None,
            delight: false,
            octree: None,
            threshold: None,
            target_faces: None,
            seed: None,
            follow: false,
            json: false,
            local: false,
        }
    }

    /// Each mode needs a different input, so what was supplied names it.
    #[test]
    fn the_supplied_inputs_name_the_workflow() {
        let text = CreateArgs {
            prompt: Some("a ceramic fox".into()),
            ..args()
        };
        assert_eq!(
            resolve_mode(&text).unwrap(),
            MeshWorkflowModeArg::TextToMesh
        );

        let texture = CreateArgs {
            mesh: Some("chair.glb".into()),
            image: Some("albedo.png".into()),
            ..args()
        };
        assert_eq!(
            resolve_mode(&texture).unwrap(),
            MeshWorkflowModeArg::MeshTexture
        );

        let roundtrip = CreateArgs {
            mesh: Some("chair.glb".into()),
            ..args()
        };
        assert_eq!(
            resolve_mode(&roundtrip).unwrap(),
            MeshWorkflowModeArg::MeshRoundtrip
        );

        // An explicit --mode wins over whatever was supplied.
        let named = CreateArgs {
            mesh: Some("chair.glb".into()),
            mode: Some(MeshWorkflowModeArg::MeshTexture),
            ..args()
        };
        assert_eq!(
            resolve_mode(&named).unwrap(),
            MeshWorkflowModeArg::MeshTexture
        );
    }

    /// Two inputs that name two different workflows are a question, not a
    /// default — and neither input at all says what to supply.
    #[test]
    fn contradictory_or_absent_inputs_are_refused_by_name() {
        let both = CreateArgs {
            prompt: Some("a fox".into()),
            mesh: Some("chair.glb".into()),
            ..args()
        };
        let message = resolve_mode(&both).unwrap_err().to_string();
        assert!(message.contains("--prompt"), "{message}");
        assert!(message.contains("--mesh"), "{message}");

        let neither = resolve_mode(&args()).unwrap_err().to_string();
        assert!(neither.contains("--prompt"), "{neither}");
        assert!(neither.contains("--mesh"), "{neither}");
    }

    /// A flag the mode cannot honour is named rather than dropped.
    #[test]
    fn a_mode_refuses_the_flags_it_cannot_honour() {
        let roundtrip_painting = CreateArgs {
            mesh: Some("chair.glb".into()),
            texture: true,
            ..args()
        };
        let message = refuse_flags_the_mode_cannot_use(
            MeshWorkflowModeArg::MeshRoundtrip,
            &roundtrip_painting,
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("mesh_texture"), "{message}");

        let texture_without_appearance = CreateArgs {
            mesh: Some("chair.glb".into()),
            ..args()
        };
        let message = refuse_flags_the_mode_cannot_use(
            MeshWorkflowModeArg::MeshTexture,
            &texture_without_appearance,
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("--image"), "{message}");

        let resolution_without_texture = CreateArgs {
            prompt: Some("a fox".into()),
            texture_resolution: Some(2048),
            ..args()
        };
        let message = refuse_flags_the_mode_cannot_use(
            MeshWorkflowModeArg::TextToMesh,
            &resolution_without_texture,
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("--texture-resolution"), "{message}");
    }

    /// A control whose stage the run never reaches is refused rather than
    /// sent and ignored.
    #[test]
    fn a_control_with_no_stage_to_act_on_is_refused() {
        // A roundtrip's graph is shape then finalize: no picture, so nothing
        // for matting or delight to do.
        for (args, flag) in [
            (
                CreateArgs {
                    mesh: Some("chair.glb".into()),
                    matting: Some(MeshMattingArg::On),
                    ..args()
                },
                "--matting",
            ),
            (
                CreateArgs {
                    mesh: Some("chair.glb".into()),
                    delight: true,
                    ..args()
                },
                "--delight",
            ),
            (
                CreateArgs {
                    mesh: Some("chair.glb".into()),
                    image_model: Some("flux-schnell:q8".into()),
                    ..args()
                },
                "--image-model",
            ),
        ] {
            let message =
                refuse_flags_the_mode_cannot_use(MeshWorkflowModeArg::MeshRoundtrip, &args)
                    .unwrap_err()
                    .to_string();
            assert!(message.contains(flag), "{flag}: {message}");
        }

        // The mesh coordinates describe a mesh you supplied, and a
        // text-to-3-D run builds its own.
        let message = refuse_flags_the_mode_cannot_use(
            MeshWorkflowModeArg::TextToMesh,
            &CreateArgs {
                prompt: Some("a fox".into()),
                up_axis: Some(MeshUpAxis::Z),
                ..args()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("--up-axis"), "{message}");
    }

    /// A texture-only run paints by definition; a roundtrip never does; a
    /// text-to-3-D run is geometry unless asked.
    #[test]
    fn texturing_follows_the_mode_and_the_opt_in() {
        assert!(texture_for(MeshWorkflowModeArg::MeshTexture, &args()));
        assert!(!texture_for(MeshWorkflowModeArg::MeshRoundtrip, &args()));
        assert!(!texture_for(MeshWorkflowModeArg::TextToMesh, &args()));
        assert!(texture_for(
            MeshWorkflowModeArg::TextToMesh,
            &CreateArgs {
                texture: true,
                ..args()
            }
        ));
    }

    /// The mesh block carries only what was named; the engine answers for
    /// the rest.
    #[test]
    fn the_mesh_block_carries_only_the_named_controls() {
        let bare = mesh_options(&args(), false);
        assert_eq!(bare.octree_resolution, None);
        assert_eq!(bare.threshold, None);
        assert_eq!(bare.target_faces, None);
        assert_eq!(bare.texture, Some(false));
        assert_eq!(bare.delight, None);

        let full = mesh_options(
            &CreateArgs {
                octree: Some(320),
                threshold: Some(0.55),
                target_faces: Some(40_000),
                texture_resolution: Some(4096),
                matting: Some(MeshMattingArg::On),
                delight: true,
                ..args()
            },
            true,
        );
        assert_eq!(full.octree_resolution, Some(320));
        assert_eq!(full.target_faces, Some(40_000));
        assert_eq!(full.texture_resolution, Some(4096));
        assert_eq!(full.matting, Some(mold_core::MeshMattingMode::On));
        assert_eq!(full.delight, Some(true));
    }

    /// A GLB stage is canvasless: 0x0 and one output, which is what the
    /// workflow contract validates.
    #[test]
    fn a_glb_stage_is_pinned_to_a_single_canvasless_output() {
        let defaults = ModelDefaults {
            default_steps: 30,
            default_guidance: 5.0,
            default_width: 1024,
            default_height: 1024,
            description: String::new(),
            ..Default::default()
        };
        let mesh = stage_request("hunyuan3d-2.1:fp16", &defaults, OutputFormat::Glb, Some(7));
        assert_eq!((mesh.width, mesh.height), (0, 0));
        assert_eq!(mesh.output_format, Some(OutputFormat::Glb));
        assert_eq!(mesh.batch_size, 1);
        assert_eq!(mesh.seed, Some(7));
        assert!(mesh.prompt.is_empty());

        let image = stage_request("flux-schnell:q8", &defaults, OutputFormat::Png, Some(7));
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.steps, 30);
    }

    /// The container comes from the file, and both are declared on the wire.
    #[test]
    fn a_supplied_mesh_declares_its_container_and_its_mime_type() {
        assert_eq!(
            mesh_format_of(Path::new("chair.GLB")).unwrap(),
            MeshReferenceFormat::Glb
        );
        assert_eq!(
            mesh_format_of(Path::new("chair.obj")).unwrap(),
            MeshReferenceFormat::Obj
        );
        assert!(mesh_format_of(Path::new("chair.stl")).is_err());
        assert_eq!(
            mesh_mime_type(MeshReferenceFormat::Glb),
            "model/gltf-binary"
        );
        assert_eq!(mesh_mime_type(MeshReferenceFormat::Obj), "model/obj");
    }

    /// A roundtrip request built from the flags passes the same validation
    /// the server runs, so a mistake reads as a sentence about the flags.
    #[test]
    fn a_built_roundtrip_request_satisfies_the_workflow_contract() {
        let dir = tempfile::tempdir().unwrap();
        let mesh_path = dir.path().join("chair.glb");
        std::fs::write(&mesh_path, b"glTF binary").unwrap();
        let built = CreateArgs {
            mesh: Some(mesh_path),
            ..args()
        };
        let defaults = ModelDefaults {
            default_steps: 30,
            default_guidance: 5.0,
            default_width: 1024,
            default_height: 1024,
            description: String::new(),
            ..Default::default()
        };
        let mut request =
            stage_request("hunyuan3d-2.1:fp16", &defaults, OutputFormat::Glb, Some(42));
        request.mesh = Some(mesh_options(&built, false));
        request.references = Some(vec![mesh_reference(&built).unwrap()]);
        let workflow = CreateMeshWorkflowRequest::MeshRoundtrip {
            roundtrip_request: Box::new(request),
        };
        assert_eq!(validate_create_mesh_workflow(&workflow), Ok(()));
        assert_eq!(
            workflow
                .planned_stage_kinds()
                .iter()
                .map(|stage| stage.as_str())
                .collect::<Vec<_>>(),
            ["shape", "finalize"]
        );
        assert_eq!(workflow.mode_str(), "mesh_roundtrip");
    }

    /// A mode's default model is one that can actually run it.
    ///
    /// A roundtrip needs the 2.1 shape VAE and nothing else has one, so a
    /// single default meant `mold mesh-workflow create --mesh chair.glb` was
    /// refused by the door it had just been sent to.
    #[test]
    fn each_mode_defaults_to_a_model_that_can_run_it() {
        assert_eq!(
            default_model_for(MeshWorkflowModeArg::MeshRoundtrip),
            mold_core::manifest::HUNYUAN3D_21_MODEL
        );
        assert_eq!(
            default_model_for(MeshWorkflowModeArg::TextToMesh),
            mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL
        );
        assert_eq!(
            default_model_for(MeshWorkflowModeArg::MeshTexture),
            mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL
        );
        // The roundtrip default passes the validator's own test, so the
        // default can never be the thing admission refuses.
        assert!(mold_core::manifest::hunyuan3d_shape21_model(
            default_model_for(MeshWorkflowModeArg::MeshRoundtrip)
        ));
    }

    /// A model the mode cannot run is refused by naming the flag that fixes
    /// it, not by repeating the wire validator's flagless sentence.
    #[test]
    fn a_model_the_mode_cannot_run_names_the_flag_that_fixes_it() {
        let message = refuse_a_model_the_mode_cannot_run(
            MeshWorkflowModeArg::MeshRoundtrip,
            mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL,
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("--model hunyuan3d-2.1:fp16"), "{message}");
        assert!(
            message.contains(mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL),
            "the refusal names the model that cannot do it: {message}"
        );

        // The 2.1 tier and its derived quantizations are accepted, and the
        // other modes take any tier.
        for model in ["hunyuan3d-2.1:fp16", "hunyuan3d-2.1:q4"] {
            assert!(
                refuse_a_model_the_mode_cannot_run(MeshWorkflowModeArg::MeshRoundtrip, model)
                    .is_ok(),
                "{model} runs a roundtrip"
            );
        }
        assert!(refuse_a_model_the_mode_cannot_run(
            MeshWorkflowModeArg::TextToMesh,
            mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL
        )
        .is_ok());
    }

    /// An empty id is refused by name rather than resolving to the LIST route.
    ///
    /// `/api/mesh-workflows/` with a blank segment answers with the listing,
    /// which the client then fails to read as one job — "expected value at
    /// line 1 column 1", a sentence about JSON that says nothing about the
    /// blank argument.
    #[test]
    fn an_empty_workflow_id_is_refused_rather_than_listing_every_job() {
        for blank in ["", "   ", "\t"] {
            let message = require_workflow_id(blank).unwrap_err().to_string();
            assert!(message.contains("workflow id is required"), "{message}");
            assert!(message.contains("mesh-workflow list"), "{message}");
        }
        // A real id survives, trimmed of whatever a shell handed us.
        assert_eq!(require_workflow_id("  mw-1  ").unwrap(), "mw-1");
    }

    fn stage(index: u32, kind: &str, state: &str) -> MeshWorkflowStageRecord {
        MeshWorkflowStageRecord {
            index,
            kind: kind.parse().unwrap(),
            state: state.parse().unwrap(),
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
        }
    }

    /// `--follow` renders stage lines by DIFFING snapshots, because a
    /// snapshot is the only frame a real server sends — and it re-sends the
    /// whole job every time anything about it moves.
    #[test]
    fn following_prints_a_line_only_when_a_stage_actually_moves() {
        let mut seen = Vec::new();

        // Attaching shows where the job already is.
        let first = stage_transitions(
            &mut seen,
            &[
                stage(0, "shape", "running"),
                stage(1, "finalize", "pending"),
            ],
        );
        assert_eq!(first.len(), 2);
        assert!(first[0].contains("shape"), "{first:?}");
        assert!(first[0].contains("running"), "{first:?}");

        // An identical snapshot says nothing, however many times it arrives.
        for _ in 0..3 {
            assert!(stage_transitions(
                &mut seen,
                &[
                    stage(0, "shape", "running"),
                    stage(1, "finalize", "pending")
                ],
            )
            .is_empty());
        }

        // Only the stage that moved prints.
        let moved = stage_transitions(
            &mut seen,
            &[
                stage(0, "shape", "completed"),
                stage(1, "finalize", "pending"),
            ],
        );
        assert_eq!(moved.len(), 1);
        assert!(moved[0].contains("completed"), "{moved:?}");

        // A stage the record did not have yet prints its current state, so
        // a job whose runner admits stages as it goes still reads correctly.
        let grown = stage_transitions(
            &mut seen,
            &[
                stage(0, "shape", "completed"),
                stage(1, "finalize", "pending"),
                stage(2, "paint", "running"),
            ],
        );
        assert_eq!(grown.len(), 1);
        assert!(grown[0].contains("paint"), "{grown:?}");
    }

    /// A failed stage carries its reason onto the line.
    #[test]
    fn a_failed_stage_prints_the_reason_the_server_gave() {
        let mut failed = stage(0, "shape", "failed");
        failed.error = Some("ran out of memory".into());
        let lines = stage_transitions(&mut Vec::new(), &[failed]);
        assert!(lines[0].contains("ran out of memory"), "{lines:?}");
    }

    /// Plain `running` says nothing: every stage line already implies it.
    /// Settlement and failure both speak.
    #[test]
    fn the_job_state_line_speaks_only_when_it_adds_something() {
        assert_eq!(state_line(MeshWorkflowJobState::Running, &None), None);
        assert!(state_line(MeshWorkflowJobState::Completed, &None)
            .is_some_and(|line| line.contains("completed")));
        assert!(
            state_line(MeshWorkflowJobState::Failed, &Some("no runner".into()))
                .is_some_and(|line| line.contains("no runner"))
        );
    }

    /// A mesh past either advertised limit is refused by name before a
    /// session is opened, the way the H3 upload path and the Studio both do.
    #[test]
    fn a_mesh_past_the_hosts_upload_limits_is_refused_by_name() {
        let capabilities =
            |max_file_bytes: u64, max_session_bytes: u64| mold_core::ReferenceUploadCapabilities {
                available: true,
                protocol_version: 1,
                requires_api_key: true,
                session_path: String::new(),
                upload_path: String::new(),
                session_handle_header: String::new(),
                upload_handle_header: String::new(),
                max_file_bytes,
                max_session_bytes,
                max_active_sessions: 4,
                session_ttl_ms: 0,
            };
        assert!(refuse_a_mesh_this_host_will_not_accept(&capabilities(100, 100), 100).is_ok());

        let per_file = refuse_a_mesh_this_host_will_not_accept(&capabilities(50, 1000), 100)
            .unwrap_err()
            .to_string();
        assert!(per_file.contains("per uploaded file"), "{per_file}");
        assert!(per_file.contains("50"), "{per_file}");

        // Named separately, because a host may set the two independently and
        // the user needs to know which one they hit.
        let per_session = refuse_a_mesh_this_host_will_not_accept(&capabilities(1000, 50), 100)
            .unwrap_err()
            .to_string();
        assert!(per_session.contains("per upload session"), "{per_session}");
    }

    /// `--local` is refused by name: there is no local form of a durable
    /// workflow, and the honest alternative is a different command.
    #[test]
    fn a_local_workflow_is_refused_by_name() {
        assert!(LOCAL_REFUSAL.contains("durable on one machine"));
        assert!(LOCAL_REFUSAL.contains("mold run"));
    }
}
