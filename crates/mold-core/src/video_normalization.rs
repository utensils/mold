//! Negotiated fresh H3 video ingress. Candidate facts never become generation
//! authority until the complete receipt set is explicitly finalized.
use crate::{
    GenerateRequest, GenerationReference, GenerationReferenceKind, GenerationReferenceMetadata,
    ReferenceUploadSlot,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const VERSION: u32 = 1;
pub const CANDIDATE_PATH: &str = "/api/generate/reference-normalization-sessions";
pub const FINALIZE_PATH: &str = "/api/generate/reference-normalization-finalize";
pub const CANDIDATE_DIGEST_HEADER: &str = "x-mold-reference-candidate-sha256";

/// SHA-256 over the domain-separated, exact frozen POST body.
pub fn candidate_scope_sha256(body: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold-reference-normalization-candidate-v1\0");
    hash.update(body);
    format!("{:x}", hash.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoNormalizationCapabilities {
    pub available: bool,
    pub protocol_version: u32,
    pub input_roles: Vec<String>,
    pub candidate_session_path: String,
    pub finalize_path: String,
    pub candidate_digest_header: String,
    pub max_original_file_bytes: u64,
    pub max_canonical_file_bytes: u64,
    pub max_session_bytes: u64,
    pub accepted_mime_types: Vec<String>,
    pub accepted_extensions: Vec<String>,
    pub accepted_video_codecs: Vec<String>,
    pub canonical_mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct OriginalVideo {
    pub mime_type: String,
    pub name: Option<String>,
    pub byte_length: u64,
    pub sha256: String,
}

/// A separate transport union; fresh videos deliberately have no canonical
/// dimensions, duration, audio facts or media authority.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReferenceCandidate {
    Canonical {
        index: u32,
        upload: bool,
        reference: GenerationReference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        byte_length: Option<u64>,
    },
    FreshVideo {
        index: u32,
        kind: GenerationReferenceKind,
        original: OriginalVideo,
    },
}
impl ReferenceCandidate {
    pub fn index(&self) -> u32 {
        match self {
            Self::Canonical { index, .. } | Self::FreshVideo { index, .. } => *index,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CandidateSessionRequest {
    pub protocol_version: u32,
    pub request: GenerateRequest,
    pub reference_candidates: Vec<ReferenceCandidate>,
}

// No Debug: bearer secrets must never enter diagnostic output.
#[derive(Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CandidateSessionResponse {
    pub protocol_version: u32,
    pub instance_id: String,
    pub session_id: String,
    pub candidate_scope_sha256: String,
    pub expires_at_ms: u64,
    pub session_handle: String,
    pub session_complete: bool,
    pub uploads: Vec<ReferenceUploadSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NormalizationReceipt {
    pub version: u32,
    pub kind: GenerationReferenceKind,
    pub original_sha256: String,
    pub original_byte_length: u64,
    pub original_mime_type: String,
    pub canonical_sha256: String,
    pub canonical_byte_length: u64,
    pub canonical_mime_type: String,
    pub converted: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct NormalizationUploadResponse {
    pub protocol_version: u32,
    pub instance_id: String,
    pub session_id: String,
    pub candidate_scope_sha256: String,
    pub reference: u32,
    pub session_complete: bool,
    pub receipt: NormalizationReceipt,
    pub metadata: GenerationReferenceMetadata,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct FinalizeReceipt {
    pub reference: u32,
    pub kind: GenerationReferenceKind,
    pub original_sha256: String,
    pub canonical_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NormalizationFinalizeRequest {
    pub protocol_version: u32,
    pub instance_id: String,
    pub session_id: String,
    pub candidate_scope_sha256: String,
    pub receipts: Vec<FinalizeReceipt>,
}
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct NormalizationFinalizeResponse {
    pub protocol_version: u32,
    pub instance_id: String,
    pub session_id: String,
    pub candidate_scope_sha256: String,
    pub request_scope_sha256: String,
    pub session_complete: bool,
    pub references: Vec<GenerationReference>,
    pub receipts: Vec<FinalizeReceipt>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fingerprint_binds_exact_wire_bytes_and_domain() {
        assert_ne!(
            candidate_scope_sha256(b"{}"),
            candidate_scope_sha256(b"{ }")
        );
        assert_eq!(
            candidate_scope_sha256(b"{}"),
            format!(
                "{:x}",
                Sha256::digest(b"mold-reference-normalization-candidate-v1\0{}")
            )
        );
    }
    #[test]
    fn fresh_video_requires_original_facts_but_no_guessed_canonical_facts() {
        let json = serde_json::json!({"index":1,"type":"fresh_video","kind":"video","original":{"name":"a.mov","mime_type":"video/quicktime","sha256":"a".repeat(64),"byte_length":123}});
        let candidate: ReferenceCandidate = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(candidate, ReferenceCandidate::FreshVideo { .. }));
        let mut guessed = json;
        guessed["width"] = 10.into();
        assert!(serde_json::from_value::<ReferenceCandidate>(guessed).is_err());
    }
}
