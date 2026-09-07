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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<MeshWorkflowArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Portable authority for restart and cross-host workflow recovery. SQLite is
/// only a queryable index; this manifest wins during reconciliation.
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
    fn mesh_workflow_state_names_are_stable() {
        assert_eq!(MeshWorkflowJobState::Queued.as_str(), "queued");
        assert!(MeshWorkflowJobState::Completed.is_settled());
        assert!(MeshWorkflowJobState::Failed.is_settled());
        assert!(!MeshWorkflowJobState::Paused.is_settled());
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
}
