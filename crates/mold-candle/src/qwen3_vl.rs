//! Public Qwen3-VL vision facade.
//!
//! The tower, its processor and its multimodal position rules live with H3
//! (`minimax_h3`), which was their first consumer; Qwen Image 2.1 conditions
//! on the same architecture. This module only re-exports them so a second
//! family does not reach into H3's namespace. It moves no H3 arithmetic.

pub use crate::minimax_h3::{
    create_mm_token_type_ids, pack_qwen_vision_u8, pack_qwen_vision_u8_torchvision,
    qwen_mrope_positions, ConditionerCheckpoint, GridThw, PackedVisionPatches, ProcessorError,
    Qwen3VlVisionDimensions, Qwen3VlVisionModel, QwenMmTokenType,
};
