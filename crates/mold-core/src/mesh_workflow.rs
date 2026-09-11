//! Typed contract for durable multi-stage mesh workflows.
//!
//! A mesh workflow is separate from a video [`crate::ChainRequest`]. Text to
//! mesh composes one ordinary image request with one Hunyuan3D request, while
//! supplied-mesh texturing executes only the latter. Keeping the two requests
//! intact lets each stage use the same validation, scheduling, provenance, and
//! publication contracts as a direct generation.

use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::{GenerateRequest, GenerationReference, MoldError, MoldResult, OutputFormat};

pub const MESH_WORKFLOW_CONTRACT_VERSION: u32 = 1;
pub const MESH_WORKFLOW_MANIFEST_SCHEMA: &str = "mold.mesh-workflow.v1";
pub const MESH_WORKFLOW_MANIFEST_FILE: &str = "manifest.toml";

/// User-authored work accepted by `POST /api/mesh-workflows`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CreateMeshWorkflowRequest {
    /// Generate and retain an image, then feed that exact image to the mesh
    /// recipe. A resumed shape or paint failure never reruns this stage.
    TextToMesh {
        image_request: Box<GenerateRequest>,
        mesh_request: Box<GenerateRequest>,
    },
    /// Texture a supplied GLB/OBJ using the appearance image carried by the
    /// mesh request. Shape inference is absent from this workflow.
    MeshTexture {
        texture_request: Box<GenerateRequest>,
    },
    /// Reconstruct a supplied mesh through the 2.1 shape-VAE latent space.
    MeshRoundtrip {
        roundtrip_request: Box<GenerateRequest>,
    },
}

impl CreateMeshWorkflowRequest {
    /// Stable stage graph advertised before execution. Matting remains a
    /// stage even when `auto` later preserves an existing useful alpha mask;
    /// that decision and its retained input/output are part of provenance.
    pub fn planned_stage_kinds(&self) -> Vec<MeshWorkflowStageKind> {
        match self {
            Self::TextToMesh { mesh_request, .. } => {
                let mut stages = vec![MeshWorkflowStageKind::Image];
                if cfg!(feature = "mesh-matting")
                    && mesh_request
                        .mesh
                        .as_ref()
                        .and_then(|mesh| mesh.matting)
                        .unwrap_or_default()
                        != crate::MeshMattingMode::Off
                {
                    stages.push(MeshWorkflowStageKind::Matting);
                }
                if mesh_request
                    .mesh
                    .as_ref()
                    .and_then(|mesh| mesh.delight)
                    .unwrap_or(false)
                {
                    stages.push(MeshWorkflowStageKind::Delight);
                }
                stages.push(MeshWorkflowStageKind::Shape);
                if mesh_request
                    .mesh
                    .as_ref()
                    .and_then(|mesh| mesh.texture)
                    .unwrap_or(false)
                {
                    stages.push(MeshWorkflowStageKind::Paint);
                }
                stages.push(MeshWorkflowStageKind::Finalize);
                stages
            }
            Self::MeshTexture { texture_request } => {
                let mut stages = Vec::new();
                if cfg!(feature = "mesh-matting")
                    && texture_request
                        .mesh
                        .as_ref()
                        .and_then(|mesh| mesh.matting)
                        .unwrap_or_default()
                        != crate::MeshMattingMode::Off
                {
                    stages.push(MeshWorkflowStageKind::Matting);
                }
                if texture_request
                    .mesh
                    .as_ref()
                    .and_then(|mesh| mesh.delight)
                    .unwrap_or(false)
                {
                    stages.push(MeshWorkflowStageKind::Delight);
                }
                stages.extend([
                    MeshWorkflowStageKind::Paint,
                    MeshWorkflowStageKind::Finalize,
                ]);
                stages
            }
            Self::MeshRoundtrip { .. } => vec![
                MeshWorkflowStageKind::Shape,
                MeshWorkflowStageKind::Finalize,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshWorkflowJobState {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl MeshWorkflowJobState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_settled(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl std::str::FromStr for MeshWorkflowJobState {
    type Err = MoldError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "paused" => Ok(Self::Paused),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(MoldError::Validation(format!(
                "unknown mesh workflow job state '{value}'"
            ))),
        }
    }
}

/// Which durable 3-D workflow produced a piece of work, and the part it plays
/// in it.
///
/// Every stage of a workflow is admitted as an ORDINARY generation, so without
/// this the queue row and the finished print are indistinguishable from a
/// hand-authored render: a queue row could only route back to New image, and a
/// text-to-3-D run left its source picture, its matted and delighted copies and
/// its mesh in the gallery as four unrelated prints. It rides
/// [`crate::GenerateRequest`] and is copied onto [`crate::OutputMetadata`], so
/// the live queue entry (whose `metadata` IS the request) and the published
/// print carry the same answer.
///
/// Server-minted in the workflow runner and REFUSED on a public request — a
/// client that could stamp it would forge a print into someone's workflow.
/// Additive: absent on every print made outside the 3-D Studio, on every print
/// made before this field, and on every older host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowProvenance {
    /// The durable job under `/api/mesh-workflows`.
    pub job_id: String,
    /// The authored workflow shape: `text_to_mesh`, `mesh_roundtrip`, or
    /// `mesh_texture`. A value this build does not know is still displayable.
    pub mode: String,
    /// The artifact role this output fills, the same vocabulary
    /// [`MeshWorkflowArtifact::role`] uses: `generated_image`, `matted_image`,
    /// `delighted_image`, `final_glb`. `final_glb` is the run's LEAD — the one
    /// a client shows when it collapses the run into a single print.
    pub role: String,
    /// Zero-based stage that produced it, so a client can order the members
    /// without knowing the stage graph.
    pub stage_index: u32,
}

/// The role whose print represents the whole run.
pub const MESH_WORKFLOW_LEAD_ROLE: &str = "final_glb";

impl CreateMeshWorkflowRequest {
    /// The workflow's wire tag — the same word `mode` carries on the request.
    pub fn mode_str(&self) -> &'static str {
        match self {
            Self::TextToMesh { .. } => "text_to_mesh",
            Self::MeshTexture { .. } => "mesh_texture",
            Self::MeshRoundtrip { .. } => "mesh_roundtrip",
        }
    }
}

/// Refuse a client-supplied [`MeshWorkflowProvenance`] on a public request.
///
/// The field is provenance the SERVER mints in the workflow runner. A request
/// that arrived over the public generate doors carrying one would be claiming
/// membership of a durable workflow it is not a stage of — which would route
/// another person's queue row into the 3-D Studio, and file a stranger's print
/// inside their run's stack in the Library. Refused by name rather than
/// silently stripped, so a client that sends it learns why.
pub fn client_minted_mesh_workflow_refusal(req: &GenerateRequest) -> Option<&'static str> {
    req.mesh_workflow.is_some().then_some(
        "mesh_workflow is server-minted provenance and cannot be supplied by a client; \
         create a 3-D workflow with POST /api/mesh-workflows instead",
    )
}

impl MeshWorkflowProvenance {
    /// Whether this output is the one a collapsed view shows.
    pub fn is_lead(&self) -> bool {
        self.role == MESH_WORKFLOW_LEAD_ROLE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshWorkflowStageKind {
    Image,
    Matting,
    Delight,
    Shape,
    Paint,
    Finalize,
}

impl MeshWorkflowStageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Matting => "matting",
            Self::Delight => "delight",
            Self::Shape => "shape",
            Self::Paint => "paint",
            Self::Finalize => "finalize",
        }
    }
}

impl std::str::FromStr for MeshWorkflowStageKind {
    type Err = MoldError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "image" => Ok(Self::Image),
            "matting" => Ok(Self::Matting),
            "delight" => Ok(Self::Delight),
            "shape" => Ok(Self::Shape),
            "paint" => Ok(Self::Paint),
            "finalize" => Ok(Self::Finalize),
            _ => Err(MoldError::Validation(format!(
                "unknown mesh workflow stage kind '{value}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshWorkflowStageState {
    Pending,
    Running,
    Completed,
    Failed,
}

impl MeshWorkflowStageState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl std::str::FromStr for MeshWorkflowStageState {
    type Err = MoldError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            _ => Err(MoldError::Validation(format!(
                "unknown mesh workflow stage state '{value}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowArtifact {
    /// Stable semantic role such as `generated_image`, `processed_image`,
    /// `normalized_mesh`, or `final_glb`.
    pub role: String,
    /// Portable path relative to the workflow directory.
    pub relative_path: String,
    pub media_type: String,
    pub sha256: String,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowStageRecord {
    pub index: u32,
    pub kind: MeshWorkflowStageKind,
    pub state: MeshWorkflowStageState,
    /// Opaque durable generation batch driving this stage. Persisting it lets
    /// restart reconciliation reattach instead of submitting duplicate work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<MeshWorkflowArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowJobSummary {
    pub contract_version: u32,
    pub id: String,
    pub state: MeshWorkflowJobState,
    pub mode: crate::generation_profile::MeshWorkflowMode,
    pub stage_count: u32,
    pub current_stage: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_stage_kind: Option<MeshWorkflowStageKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowJobDetail {
    #[serde(flatten)]
    pub summary: MeshWorkflowJobSummary,
    pub request: CreateMeshWorkflowRequest,
    pub stages: Vec<MeshWorkflowStageRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MeshWorkflowJobListing {
    pub jobs: Vec<MeshWorkflowJobSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CreateMeshWorkflowResponse {
    pub job_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MeshWorkflowEvent {
    Snapshot {
        job: MeshWorkflowJobDetail,
    },
    StageStarted {
        stage_index: u32,
        kind: MeshWorkflowStageKind,
    },
    StageProgress {
        stage_index: u32,
        kind: MeshWorkflowStageKind,
        current: u32,
        total: u32,
    },
    StageCompleted {
        stage_index: u32,
        kind: MeshWorkflowStageKind,
        artifacts: Vec<MeshWorkflowArtifact>,
    },
    StateChanged {
        state: MeshWorkflowJobState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// How following a durable mesh workflow ended.
///
/// Not a wire type: the event stream carries no terminal summary, so a
/// client that followed a workflow to settlement assembles this from the
/// snapshot it opened with, the state changes it saw, and — when the run
/// settled mid-stream without republishing its filename — one final read of
/// the job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshWorkflowOutcome {
    pub state: MeshWorkflowJobState,
    pub error: Option<String>,
    /// The stitched print the run published, present only once it completed.
    pub output_filename: Option<String>,
}

/// Portable workflow record. During local execution SQLite is the
/// transactional authority and refreshes this manifest after every committed
/// stage transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshWorkflowManifest {
    pub schema: String,
    pub contract_version: u32,
    pub job_id: String,
    pub created_at_ms: i64,
    /// Canonical JSON of [`CreateMeshWorkflowRequest`] after request media has
    /// been sealed separately. No upload handles, paths, or inline bytes are
    /// allowed in this value.
    pub request_json: String,
    pub stages: Vec<MeshWorkflowStageRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_filename: Option<String>,
}

impl MeshWorkflowManifest {
    pub fn new(
        job_id: impl Into<String>,
        created_at_ms: i64,
        persisted_request: &CreateMeshWorkflowRequest,
        stages: Vec<MeshWorkflowStageRecord>,
    ) -> MoldResult<Self> {
        ensure_request_media_is_sealed(persisted_request)?;
        let request_json = serde_json::to_string(persisted_request).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "mesh workflow request JSON serialise failed: {error}"
            ))
        })?;
        Ok(Self {
            schema: MESH_WORKFLOW_MANIFEST_SCHEMA.into(),
            contract_version: MESH_WORKFLOW_CONTRACT_VERSION,
            job_id: job_id.into(),
            created_at_ms,
            request_json,
            stages,
            output_filename: None,
        })
    }

    pub fn request(&self) -> MoldResult<CreateMeshWorkflowRequest> {
        serde_json::from_str(&self.request_json).map_err(|error| {
            MoldError::Validation(format!("mesh workflow request JSON parse failed: {error}"))
        })
    }

    pub fn to_toml(&self) -> MoldResult<String> {
        self.validate()?;
        toml::to_string(self).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "mesh workflow manifest TOML serialise failed: {error}"
            ))
        })
    }

    pub fn from_toml(body: &str) -> MoldResult<Self> {
        #[derive(Deserialize)]
        struct Header {
            schema: Option<String>,
            contract_version: Option<u32>,
        }
        let header: Header = toml::from_str(body).map_err(|error| {
            MoldError::Validation(format!("mesh workflow manifest TOML parse failed: {error}"))
        })?;
        if header.schema.as_deref() != Some(MESH_WORKFLOW_MANIFEST_SCHEMA)
            || header.contract_version != Some(MESH_WORKFLOW_CONTRACT_VERSION)
        {
            return Err(MoldError::Validation(format!(
                "unsupported mesh workflow manifest; expected schema '{MESH_WORKFLOW_MANIFEST_SCHEMA}' version {MESH_WORKFLOW_CONTRACT_VERSION}"
            )));
        }
        let manifest: Self = toml::from_str(body).map_err(|error| {
            MoldError::Validation(format!("mesh workflow manifest TOML parse failed: {error}"))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn write_atomic(&self, workflow_dir: &Path) -> MoldResult<()> {
        fs::create_dir_all(workflow_dir).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "creating mesh workflow directory '{}': {error}",
                workflow_dir.display()
            ))
        })?;
        let destination = workflow_dir.join(MESH_WORKFLOW_MANIFEST_FILE);
        let temporary = workflow_dir.join(format!("{MESH_WORKFLOW_MANIFEST_FILE}.tmp"));
        fs::write(&temporary, self.to_toml()?).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "writing mesh workflow manifest '{}': {error}",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, &destination).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "committing mesh workflow manifest '{}': {error}",
                destination.display()
            ))
        })?;
        Ok(())
    }

    pub fn read_from_dir(workflow_dir: &Path) -> MoldResult<Self> {
        let path = workflow_dir.join(MESH_WORKFLOW_MANIFEST_FILE);
        let body = fs::read_to_string(&path).map_err(|error| {
            MoldError::Other(anyhow::anyhow!(
                "reading mesh workflow manifest '{}': {error}",
                path.display()
            ))
        })?;
        Self::from_toml(&body)
    }

    fn validate(&self) -> MoldResult<()> {
        if self.job_id.trim().is_empty() {
            return Err(MoldError::Validation(
                "mesh workflow manifest job_id must not be empty".into(),
            ));
        }
        let request = self.request()?;
        ensure_request_media_is_sealed(&request)?;
        for (position, stage) in self.stages.iter().enumerate() {
            if stage.index as usize != position {
                return Err(MoldError::Validation(
                    "mesh workflow stage indexes must be contiguous from zero".into(),
                ));
            }
            for artifact in &stage.artifacts {
                validate_relative_artifact_path(&artifact.relative_path)?;
                if artifact.role.trim().is_empty()
                    || artifact.media_type.trim().is_empty()
                    || artifact.sha256.len() != 64
                    || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err(MoldError::Validation(
                        "mesh workflow artifact metadata is incomplete".into(),
                    ));
                }
            }
        }
        if self.output_filename.as_deref().is_some_and(|filename| {
            filename.trim().is_empty()
                || filename.contains('/')
                || filename.contains('\\')
                || matches!(filename, "." | "..")
        }) {
            return Err(MoldError::Validation(
                "mesh workflow output_filename must be one gallery filename".into(),
            ));
        }
        Ok(())
    }
}

fn ensure_request_media_is_sealed(request: &CreateMeshWorkflowRequest) -> MoldResult<()> {
    let requests: Vec<&GenerateRequest> = match request {
        CreateMeshWorkflowRequest::TextToMesh {
            image_request,
            mesh_request,
        } => vec![image_request, mesh_request],
        CreateMeshWorkflowRequest::MeshTexture { texture_request } => vec![texture_request],
        CreateMeshWorkflowRequest::MeshRoundtrip { roundtrip_request } => vec![roundtrip_request],
    };
    for request in requests {
        if request.source_image.is_some()
            || request.id_image.is_some()
            || request.id_images.is_some()
            || request.edit_images.is_some()
            || request.mask_image.is_some()
            || request.control_image.is_some()
            || request.audio_file.is_some()
            || request.source_video.is_some()
            || request.extend_video.is_some()
            || request.references.as_deref().is_some_and(|references| {
                references.iter().any(|reference| {
                    !matches!(
                        reference.media(),
                        crate::GenerationReferenceAuthority::Descriptor
                    )
                })
            })
        {
            return Err(MoldError::Validation(
                "mesh workflow manifest request media must be sealed outside request_json".into(),
            ));
        }
    }
    Ok(())
}

fn validate_relative_artifact_path(value: &str) -> MoldResult<()> {
    let path = Path::new(value);
    if value.trim().is_empty()
        || value.contains('\\')
        || value.starts_with('/')
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(MoldError::Validation(format!(
            "mesh workflow artifact path '{value}' must be a portable relative path"
        )));
    }
    Ok(())
}

/// Validate invariants that do not require model discovery. The server runs
/// normal per-recipe validation after resolving both selected models.
pub fn validate_create_mesh_workflow(request: &CreateMeshWorkflowRequest) -> Result<(), String> {
    match request {
        CreateMeshWorkflowRequest::TextToMesh {
            image_request,
            mesh_request,
        } => {
            if image_request.prompt.trim().is_empty() {
                return Err("text-to-mesh image_request.prompt must not be empty".into());
            }
            if image_request.model.trim().is_empty() || mesh_request.model.trim().is_empty() {
                return Err("text-to-mesh requires explicit image and mesh models".into());
            }
            if image_request.source_image.is_some()
                || image_request.references.is_some()
                || image_request
                    .output_format
                    .is_some_and(|format| format.is_mesh())
            {
                return Err("text-to-mesh image_request must be a text-to-image request".into());
            }
            if !mesh_request.prompt.trim().is_empty() {
                return Err(
                    "text-to-mesh routes the prompt only to image_request; mesh_request.prompt must be empty"
                        .into(),
                );
            }
            if mesh_request.source_image.is_some() || mesh_request.references.is_some() {
                return Err(
                    "text-to-mesh mesh_request input is produced by the workflow and must not be supplied"
                        .into(),
                );
            }
            if mesh_request.batch_size != 1 || image_request.batch_size != 1 {
                return Err("mesh workflows require batch_size 1 for every stage".into());
            }
            require_glb(mesh_request)
        }
        CreateMeshWorkflowRequest::MeshTexture { texture_request } => {
            if texture_request.model.trim().is_empty() {
                return Err("mesh-texture requires an explicit mesh model".into());
            }
            if !texture_request.prompt.trim().is_empty() {
                return Err("mesh-texture does not accept a prompt".into());
            }
            if texture_request.source_image.is_none() {
                return Err("mesh-texture requires an appearance source_image".into());
            }
            let references = texture_request.references.as_deref().unwrap_or_default();
            if references.len() != 1 || !matches!(references[0], GenerationReference::Mesh { .. }) {
                return Err("mesh-texture requires exactly one GLB or OBJ mesh reference".into());
            }
            if texture_request.batch_size != 1 {
                return Err("mesh workflows require batch_size 1 for every stage".into());
            }
            if !texture_request
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.texture)
                .unwrap_or(false)
            {
                return Err("mesh-texture requires mesh.texture=true".into());
            }
            require_glb(texture_request)
        }
        CreateMeshWorkflowRequest::MeshRoundtrip { roundtrip_request } => {
            if roundtrip_request.model.trim().is_empty() {
                return Err("mesh-roundtrip requires an explicit Hunyuan3D 2.1 model".into());
            }
            if !crate::manifest::hunyuan3d_shape21_model(&roundtrip_request.model) {
                return Err("mesh-roundtrip requires a Hunyuan3D 2.1 shape checkpoint".into());
            }
            if !roundtrip_request.prompt.trim().is_empty() {
                return Err("mesh-roundtrip does not accept a prompt".into());
            }
            if roundtrip_request.source_image.is_some() {
                return Err("mesh-roundtrip does not accept an appearance image".into());
            }
            let references = roundtrip_request.references.as_deref().unwrap_or_default();
            if references.len() != 1 || !matches!(references[0], GenerationReference::Mesh { .. }) {
                return Err("mesh-roundtrip requires exactly one GLB or OBJ mesh reference".into());
            }
            if roundtrip_request.batch_size != 1 {
                return Err("mesh workflows require batch_size 1 for every stage".into());
            }
            if roundtrip_request
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.texture)
                .unwrap_or(false)
            {
                return Err("mesh-roundtrip cannot request texture generation".into());
            }
            require_glb(roundtrip_request)
        }
    }
}

fn require_glb(request: &GenerateRequest) -> Result<(), String> {
    if request.output_format != Some(OutputFormat::Glb) {
        return Err("mesh workflow output_format must be glb".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GenerationReferenceAuthority, GenerationReferenceProvenance, MeshReferenceCoordinates,
        MeshReferenceFormat, MeshRequestOptions, MeshUpAxis,
    };

    fn image_request() -> GenerateRequest {
        let mut request = crate::test_support::minimal_generate_request("flux-schnell:q8");
        request.prompt = "a small ceramic fox on a plain studio background".into();
        request.output_format = Some(OutputFormat::Png);
        request
    }

    fn mesh_request() -> GenerateRequest {
        let mut request = crate::test_support::minimal_generate_request("hunyuan3d-2.1:fp16");
        request.prompt.clear();
        request.width = 0;
        request.height = 0;
        request.output_format = Some(OutputFormat::Glb);
        request.mesh = Some(MeshRequestOptions {
            texture: Some(true),
            ..Default::default()
        });
        request
    }

    fn mesh_reference() -> GenerationReference {
        GenerationReference::Mesh {
            media: GenerationReferenceAuthority::Inline {
                data: b"glTF".to_vec(),
            },
            provenance: GenerationReferenceProvenance::default(),
            mime_type: "model/gltf-binary".into(),
            format: MeshReferenceFormat::Glb,
            byte_length: 4,
            coordinates: MeshReferenceCoordinates {
                up_axis: MeshUpAxis::Y,
                meters_per_unit: 1.0,
            },
        }
    }

    #[test]
    fn text_to_mesh_keeps_prompt_and_inputs_at_their_own_stage() {
        let request = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(mesh_request()),
        };
        assert_eq!(validate_create_mesh_workflow(&request), Ok(()));

        let CreateMeshWorkflowRequest::TextToMesh {
            mut image_request,
            mesh_request,
        } = request
        else {
            unreachable!()
        };
        image_request.source_image = Some(vec![1]);
        assert_eq!(
            validate_create_mesh_workflow(&CreateMeshWorkflowRequest::TextToMesh {
                image_request,
                mesh_request,
            }),
            Err("text-to-mesh image_request must be a text-to-image request".into())
        );
    }

    #[test]
    fn text_to_mesh_refuses_a_second_prompt_or_prebound_mesh_input() {
        let mut mesh = mesh_request();
        mesh.prompt = "silently ignored".into();
        let request = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(mesh),
        };
        assert_eq!(
            validate_create_mesh_workflow(&request),
            Err("text-to-mesh routes the prompt only to image_request; mesh_request.prompt must be empty".into())
        );
    }

    #[test]
    fn supplied_mesh_texture_requires_distinct_mesh_and_appearance_inputs() {
        let mut texture = mesh_request();
        texture.source_image = Some(vec![137, 80, 78, 71]);
        texture.references = Some(vec![mesh_reference()]);
        let request = CreateMeshWorkflowRequest::MeshTexture {
            texture_request: Box::new(texture.clone()),
        };
        assert_eq!(validate_create_mesh_workflow(&request), Ok(()));

        texture.source_image = None;
        assert_eq!(
            validate_create_mesh_workflow(&CreateMeshWorkflowRequest::MeshTexture {
                texture_request: Box::new(texture),
            }),
            Err("mesh-texture requires an appearance source_image".into())
        );
    }

    #[test]
    fn supplied_mesh_roundtrip_has_shape_and_finalize_stages() {
        let mut roundtrip = mesh_request();
        roundtrip.model = crate::manifest::HUNYUAN3D_21_MODEL.into();
        roundtrip.source_image = None;
        roundtrip.references = Some(vec![mesh_reference()]);
        roundtrip.mesh.as_mut().unwrap().texture = Some(false);
        let request = CreateMeshWorkflowRequest::MeshRoundtrip {
            roundtrip_request: Box::new(roundtrip),
        };
        assert_eq!(validate_create_mesh_workflow(&request), Ok(()));
        assert_eq!(
            request.planned_stage_kinds(),
            vec![
                MeshWorkflowStageKind::Shape,
                MeshWorkflowStageKind::Finalize
            ]
        );
    }

    #[test]
    fn mesh_workflow_state_names_are_stable() {
        assert_eq!(MeshWorkflowJobState::Queued.as_str(), "queued");
        assert!(MeshWorkflowJobState::Completed.is_settled());
        assert!(MeshWorkflowJobState::Failed.is_settled());
        assert!(!MeshWorkflowJobState::Paused.is_settled());
    }

    #[test]
    fn stage_graph_distinguishes_shape_and_supplied_mesh_paths() {
        let mut geometry_request = mesh_request();
        geometry_request.mesh.as_mut().unwrap().texture = Some(false);
        let geometry = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(geometry_request),
        };
        let mut expected_geometry = vec![MeshWorkflowStageKind::Image];
        if cfg!(feature = "mesh-matting") {
            expected_geometry.push(MeshWorkflowStageKind::Matting);
        }
        expected_geometry.extend([
            MeshWorkflowStageKind::Shape,
            MeshWorkflowStageKind::Finalize,
        ]);
        assert_eq!(geometry.planned_stage_kinds(), expected_geometry);
        let mut textured = mesh_request();
        textured.mesh.as_mut().unwrap().texture = Some(true);
        let text_to_textured = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(textured),
        };
        let mut expected_textured = vec![MeshWorkflowStageKind::Image];
        if cfg!(feature = "mesh-matting") {
            expected_textured.push(MeshWorkflowStageKind::Matting);
        }
        expected_textured.extend([
            MeshWorkflowStageKind::Shape,
            MeshWorkflowStageKind::Paint,
            MeshWorkflowStageKind::Finalize,
        ]);
        assert_eq!(text_to_textured.planned_stage_kinds(), expected_textured);

        let mut delighted = mesh_request();
        delighted.mesh.as_mut().unwrap().delight = Some(true);
        let delighted = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(delighted),
        };
        let mut expected_delighted = vec![MeshWorkflowStageKind::Image];
        if cfg!(feature = "mesh-matting") {
            expected_delighted.push(MeshWorkflowStageKind::Matting);
        }
        expected_delighted.extend([
            MeshWorkflowStageKind::Delight,
            MeshWorkflowStageKind::Shape,
            MeshWorkflowStageKind::Paint,
            MeshWorkflowStageKind::Finalize,
        ]);
        assert_eq!(delighted.planned_stage_kinds(), expected_delighted);

        let mut no_matting = mesh_request();
        no_matting.mesh.as_mut().unwrap().matting = Some(crate::MeshMattingMode::Off);
        let no_matting = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(no_matting),
        };
        assert!(!no_matting
            .planned_stage_kinds()
            .contains(&MeshWorkflowStageKind::Matting));
    }

    #[test]
    fn manifest_round_trips_only_sealed_media_and_relative_artifacts() {
        let mut image = image_request();
        image.source_image = None;
        let request = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image),
            mesh_request: Box::new(mesh_request()),
        };
        let stage = MeshWorkflowStageRecord {
            index: 0,
            kind: MeshWorkflowStageKind::Image,
            state: MeshWorkflowStageState::Completed,
            execution_batch_id: Some("batch-image".into()),
            artifacts: vec![MeshWorkflowArtifact {
                role: "generated_image".into(),
                relative_path: "stages/000/generated.png".into(),
                media_type: "image/png".into(),
                sha256: "a".repeat(64),
                byte_length: 123,
            }],
            error: None,
        };
        let manifest = MeshWorkflowManifest::new("01WORKFLOW", 42, &request, vec![stage]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        manifest.write_atomic(dir.path()).unwrap();
        let restored = MeshWorkflowManifest::read_from_dir(dir.path()).unwrap();
        assert_eq!(restored.job_id, "01WORKFLOW");
        assert_eq!(restored.stages[0].artifacts[0].byte_length, 123);
        assert!(matches!(
            restored.request().unwrap(),
            CreateMeshWorkflowRequest::TextToMesh { .. }
        ));
    }

    #[test]
    fn manifest_rejects_inline_media_and_path_escape() {
        let mut texture = mesh_request();
        texture.source_image = Some(vec![1, 2, 3]);
        texture.references = Some(vec![mesh_reference()]);
        let request = CreateMeshWorkflowRequest::MeshTexture {
            texture_request: Box::new(texture),
        };
        assert!(matches!(
            MeshWorkflowManifest::new("job", 1, &request, vec![]),
            Err(MoldError::Validation(message)) if message.contains("must be sealed")
        ));

        let request = CreateMeshWorkflowRequest::TextToMesh {
            image_request: Box::new(image_request()),
            mesh_request: Box::new(mesh_request()),
        };
        let stage = MeshWorkflowStageRecord {
            index: 0,
            kind: MeshWorkflowStageKind::Image,
            state: MeshWorkflowStageState::Completed,
            execution_batch_id: None,
            artifacts: vec![MeshWorkflowArtifact {
                role: "generated_image".into(),
                relative_path: "../stolen.png".into(),
                media_type: "image/png".into(),
                sha256: "b".repeat(64),
                byte_length: 1,
            }],
            error: None,
        };
        let manifest = MeshWorkflowManifest::new("job", 1, &request, vec![stage]).unwrap();
        assert!(matches!(
            manifest.to_toml(),
            Err(MoldError::Validation(message)) if message.contains("portable relative path")
        ));
    }

    /*
     * The field is the one thread tying a queue row and a finished print back
     * to the workflow that made them, so a client able to mint it could route
     * another person's row into the 3-D Studio and file a stranger's print
     * inside their run. It is refused by name rather than silently stripped.
     */
    #[test]
    fn a_client_cannot_mint_workflow_provenance() {
        let mut request = crate::test_support::minimal_generate_request("flux-dev:q8");
        assert!(client_minted_mesh_workflow_refusal(&request).is_none());
        request.mesh_workflow = Some(MeshWorkflowProvenance {
            job_id: "someone-elses-workflow".into(),
            mode: "text_to_mesh".into(),
            role: MESH_WORKFLOW_LEAD_ROLE.into(),
            stage_index: 4,
        });
        let refusal = client_minted_mesh_workflow_refusal(&request)
            .expect("a supplied mesh_workflow is refused");
        assert!(refusal.contains("server-minted"));
        assert!(refusal.contains("/api/mesh-workflows"));
    }

    #[test]
    fn only_the_mesh_is_the_lead_of_its_run() {
        let member = |role: &str| MeshWorkflowProvenance {
            job_id: "workflow-1".into(),
            mode: "text_to_mesh".into(),
            role: role.into(),
            stage_index: 0,
        };
        assert!(member(MESH_WORKFLOW_LEAD_ROLE).is_lead());
        for role in ["generated_image", "matted_image", "delighted_image"] {
            assert!(!member(role).is_lead(), "{role} is not the run's lead");
        }
    }

    /* The wire tag is the same word the create request carries. */
    #[test]
    fn the_mode_tag_matches_the_request_it_came_from() {
        let request = |json: serde_json::Value| {
            serde_json::from_value::<CreateMeshWorkflowRequest>(json).expect("mode parses")
        };
        for (json, expected) in [
            (
                serde_json::json!({
                    "mode": "text_to_mesh",
                    "image_request": crate::test_support::minimal_generate_request("flux-dev:q8"),
                    "mesh_request": crate::test_support::minimal_generate_request("flux-dev:q8"),
                }),
                "text_to_mesh",
            ),
            (
                serde_json::json!({
                    "mode": "mesh_texture",
                    "texture_request": crate::test_support::minimal_generate_request("flux-dev:q8"),
                }),
                "mesh_texture",
            ),
            (
                serde_json::json!({
                    "mode": "mesh_roundtrip",
                    "roundtrip_request": crate::test_support::minimal_generate_request("flux-dev:q8"),
                }),
                "mesh_roundtrip",
            ),
        ] {
            assert_eq!(request(json).mode_str(), expected);
        }
    }
}
