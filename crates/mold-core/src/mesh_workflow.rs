//! Typed contract for durable multi-stage mesh workflows.
//!
//! A mesh workflow is separate from a video [`crate::ChainRequest`]. Text to
//! mesh composes one ordinary image request with one Hunyuan3D request, while
//! supplied-mesh texturing executes only the latter. Keeping the two requests
//! intact lets each stage use the same validation, scheduling, provenance, and
//! publication contracts as a direct generation.

use serde::{Deserialize, Serialize};

use crate::{GenerateRequest, GenerationReference, OutputFormat};

pub const MESH_WORKFLOW_CONTRACT_VERSION: u32 = 1;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshWorkflowStageState {
    Pending,
    Running,
    Completed,
    Failed,
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
}
