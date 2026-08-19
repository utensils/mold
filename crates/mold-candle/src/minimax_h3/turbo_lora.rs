//! Source-pinned contract for the published MiniMax H3 Turbo LoRA adapters.
//!
//! The Turbo distillation ships as ComfyUI-layout LoRA adapters rather than
//! alternate base checkpoints. Every adapter overlays the exact 208 linear
//! modules of the pruned checkpoint `comfy_dit` already authenticates — the 200
//! INT8 ConvRot block linears plus the 8 BF16 token-refiner linears — so this
//! module derives its expected module set, in/out features, and forbidden key
//! space from [`expected_h3_weight_specs`], never from a second hand-written
//! table.
//!
//! Constants are pinned to the Comfy-Org repository revision below and were
//! read from the published safetensors headers over HTTP range requests. The
//! effective LoRA scale differs 16x between tiers, so it is always **read from
//! the file's own `alpha` tensors** and merely cross-checked against the pinned
//! tier value; nothing here hands a hardcoded scale to a forward pass.
//!
//! This module parses, authenticates, and validates. It performs no tensor
//! loading and no forward computation, and it registers no engine, capability,
//! catalog entry, or download.

use std::error::Error as StdError;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::dit::{H3TransformerConfig, H3TransformerTask};

/// Comfy-Org repository revision that re-hosts the Turbo adapters under
/// `loras/`. It postdates [`super::comfy_dit::H3_COMFY_ORG_SOURCE_REVISION`],
/// whose tree carries no `loras/` directory at all.
pub const H3_TURBO_LORA_SOURCE_REVISION: &str = "dc559027db79c174125df4d827db55cd11178860";
/// Repository that publishes both the pruned base checkpoints and the adapters.
pub const H3_TURBO_LORA_REPOSITORY: &str = "Comfy-Org/MiniMax-H3";
/// Repository-relative directory holding every published adapter.
pub const H3_TURBO_LORA_DIRECTORY: &str = "loras";
/// Every ComfyUI-layout adapter key is namespaced under this prefix; the
/// remainder is a base-checkpoint tensor name.
pub const H3_TURBO_LORA_KEY_PREFIX: &str = "diffusion_model.";
/// `208 modules x {lora_A, lora_B, alpha}`. There is no `__metadata__` entry to
/// subtract from this count.
pub const H3_TURBO_LORA_TENSOR_COUNT: usize = 624;
/// `52 blocks x 4 linear modules` — 50 main blocks plus 2 token-refiner blocks.
pub const H3_TURBO_LORA_MODULE_COUNT: usize = 208;
/// Tensor payload bytes shared by all three published adapters; only the JSON
/// header length differs between them.
pub const H3_TURBO_LORA_PAYLOAD_BYTES: u64 = 1_956_119_360;
/// Rank of every non-fused module, and `training_rank` in the published
/// `__metadata__`.
pub const H3_TURBO_LORA_TRAINING_RANK: usize = 128;
/// The Diffusers-to-ComfyUI conversion concatenates Q/K/V on the `lora_A` axis
/// and block-diagonalizes `lora_B`, multiplying both the fused rank and the
/// fused alpha by three so `alpha / rank` is unchanged.
pub const H3_TURBO_LORA_FUSED_QKV_MULTIPLE: usize = 3;
/// Published `__metadata__.target_format`.
pub const H3_TURBO_LORA_TARGET_FORMAT: &str = "ComfyUI generic LoRA";
/// Published `__metadata__.source_format`.
pub const H3_TURBO_LORA_SOURCE_FORMAT: &str = "Diffusers PEFT LoRA";
/// Published `lora_A` / `lora_B` storage dtype.
pub const H3_TURBO_LORA_WEIGHT_DTYPE: &str = "BF16";
/// Published `alpha` storage dtype; every entry is a rank-0 scalar.
pub const H3_TURBO_LORA_ALPHA_DTYPE: &str = "F32";

const MODULE_SUFFIXES: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

/// One of the three reviewed published Turbo adapters. Detection never uses a
/// filename: the independently parsed header must agree with this authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum H3TurboLoraTier {
    /// FL2V Turbo, 8 transformer evaluations, v1.0, trained at 544p.
    Fl2v8StepV10,
    /// FL2V Turbo, 4 transformer evaluations, v1.0, trained at 768p.
    Fl2v768p4StepV10,
    /// Ref2V Turbo, 4 transformer evaluations, v0.1.
    Ref2v4StepV10,
}

impl H3TurboLoraTier {
    /// Every reviewed tier, in a stable order.
    pub const ALL: [Self; 3] = [
        Self::Fl2v8StepV10,
        Self::Fl2v768p4StepV10,
        Self::Ref2v4StepV10,
    ];

    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.v1",
            Self::Fl2v768p4StepV10 => "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.v1",
            Self::Ref2v4StepV10 => "minimax-h3.turbo-lora.ref2v-4step-v0.1.comfyui-bf16.v1",
        }
    }

    pub const fn file_name(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            Self::Fl2v768p4StepV10 => {
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
            }
            Self::Ref2v4StepV10 => "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        }
    }

    /// Repository-relative path at [`H3_TURBO_LORA_SOURCE_REVISION`].
    pub fn repository_path(self) -> String {
        format!("{H3_TURBO_LORA_DIRECTORY}/{}", self.file_name())
    }

    pub const fn file_bytes(self) -> u64 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 => 1_956_193_000,
            Self::Fl2v768p4StepV10 => 1_956_192_992,
        }
    }

    pub const fn content_sha256(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => {
                "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"
            }
            Self::Fl2v768p4StepV10 => {
                "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc"
            }
            Self::Ref2v4StepV10 => {
                "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
            }
        }
    }

    /// JSON header length, excluding the eight-byte safetensors length prefix.
    pub const fn header_len(self) -> u64 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 => 73_632,
            Self::Fl2v768p4StepV10 => 73_624,
        }
    }

    /// SHA-256 of the eight-byte little-endian length prefix followed by the
    /// exact published JSON header bytes.
    pub const fn header_identity_sha256(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => {
                "eadcdb12138db967789252da26d2abe41905b2579e1cf07b866a573e88d298fd"
            }
            Self::Fl2v768p4StepV10 => {
                "3db9fe99ff46229525c43cbe6ba5bafc8d96bdeb22ee69949ef61d4d58d561d8"
            }
            Self::Ref2v4StepV10 => {
                "53370bff715f074018793b9ebc71fa0ecd8bdfd8c5554a716ccf7bf5e6a6f745"
            }
        }
    }

    pub const fn task(self) -> H3TransformerTask {
        match self {
            Self::Fl2v8StepV10 | Self::Fl2v768p4StepV10 => H3TransformerTask::T2VaFl2Va,
            Self::Ref2v4StepV10 => H3TransformerTask::Ref2Va,
        }
    }

    /// `__metadata__.training_alpha`, and the exact `alpha` scalar carried by
    /// every non-fused module of this tier.
    pub const fn training_alpha(self) -> f32 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 => 8.0,
            Self::Fl2v768p4StepV10 => 128.0,
        }
    }

    /// `alpha / rank`. Pinned for cross-checking only — validation reads the
    /// file's own alphas and never substitutes this value.
    pub const fn training_scale(self) -> f32 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 => 0.0625,
            Self::Fl2v768p4StepV10 => 1.0,
        }
    }

    /// The exact contract an adapter file of this tier must satisfy.
    pub fn expectation(self) -> H3TurboLoraExpectation {
        H3TurboLoraExpectation {
            config: H3TransformerConfig::default(),
            task: self.task(),
            training_rank: H3_TURBO_LORA_TRAINING_RANK,
            training_alpha: self.training_alpha(),
            file_bytes: Some(self.file_bytes()),
            content_sha256: Some(self.content_sha256().to_owned()),
            header_len: Some(self.header_len()),
            header_identity_sha256: Some(self.header_identity_sha256().to_owned()),
        }
    }
}

/// The contract one adapter file is validated against. Tests construct reduced
/// expectations over a small [`H3TransformerConfig`]; the published tiers build
/// theirs from [`H3TurboLoraTier::expectation`].
#[derive(Clone, Debug, PartialEq)]
pub struct H3TurboLoraExpectation {
    pub config: H3TransformerConfig,
    pub task: H3TransformerTask,
    pub training_rank: usize,
    pub training_alpha: f32,
    pub file_bytes: Option<u64>,
    pub content_sha256: Option<String>,
    pub header_len: Option<u64>,
    pub header_identity_sha256: Option<String>,
}

impl H3TurboLoraExpectation {
    /// `alpha / rank`, identical for fused and non-fused modules because the
    /// ComfyUI conversion scales both by three.
    pub fn scale(&self) -> f32 {
        self.training_alpha / self.training_rank as f32
    }

    pub fn module_count(&self) -> usize {
        (self.config.num_layers + self.config.token_refiner_num_layers) * MODULE_SUFFIXES.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.module_count() * 3
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3TurboLoraErrorCode {
    Io,
    Cancelled,
    InvalidHeader,
    InvalidMetadata,
    ConfigMismatch,
    TaskAuthorityMismatch,
    SourceSizeMismatch,
    HeaderIdentityMismatch,
    ContentDigestMismatch,
    FileIdentityChanged,
    TensorCountMismatch,
    UnknownModule,
    MissingModule,
    ShapeMismatch,
    DTypeMismatch,
    AlphaMismatch,
    ScaleMismatch,
    BaseContractConflict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3TurboLoraError {
    pub code: H3TurboLoraErrorCode,
    pub message: String,
}

impl fmt::Display for H3TurboLoraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for H3TurboLoraError {}

pub type H3TurboLoraResult<T> = Result<T, H3TurboLoraError>;

/// Which of the four linear modules inside one block a delta targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum H3TurboLoraModuleKind {
    AttnQkvProj,
    AttnOutProj,
    MlpFc1,
    MlpFc2,
}

impl H3TurboLoraModuleKind {
    pub const ALL: [Self; 4] = [
        Self::AttnQkvProj,
        Self::AttnOutProj,
        Self::MlpFc1,
        Self::MlpFc2,
    ];

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::AttnQkvProj => MODULE_SUFFIXES[0],
            Self::AttnOutProj => MODULE_SUFFIXES[1],
            Self::MlpFc1 => MODULE_SUFFIXES[2],
            Self::MlpFc2 => MODULE_SUFFIXES[3],
        }
    }

    /// The fused Q/K/V module carries a three-times-wider rank and a
    /// three-times-larger alpha, leaving `alpha / rank` unchanged.
    pub const fn fuses_qkv(self) -> bool {
        matches!(self, Self::AttnQkvProj)
    }

    /// Resolve a module suffix such as `mlp.fc1`; unknown suffixes are rejected
    /// rather than ignored.
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.suffix() == suffix)
    }

    /// Rank of this module's `lora_A` / `lora_B` pair.
    pub const fn rank(self, training_rank: usize) -> usize {
        if self.fuses_qkv() {
            training_rank * H3_TURBO_LORA_FUSED_QKV_MULTIPLE
        } else {
            training_rank
        }
    }

    /// Exact `alpha` scalar this module must carry.
    pub fn alpha(self, training_alpha: f32) -> f32 {
        if self.fuses_qkv() {
            training_alpha * H3_TURBO_LORA_FUSED_QKV_MULTIPLE as f32
        } else {
            training_alpha
        }
    }
}

/// Which block of the transformer a module belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum H3TurboLoraModuleScope {
    /// One of the streamed INT8 ConvRot main blocks.
    MainBlock(usize),
    /// One of the resident BF16 token-refiner blocks.
    TokenRefinerBlock(usize),
}

impl H3TurboLoraModuleScope {
    /// Base-checkpoint prefix, i.e. the adapter key without
    /// [`H3_TURBO_LORA_KEY_PREFIX`] and without the module suffix.
    pub fn base_prefix(self) -> String {
        match self {
            Self::MainBlock(index) => format!("blocks.{index}"),
            Self::TokenRefinerBlock(index) => format!("token_refiner.blocks.{index}"),
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::MainBlock(index) | Self::TokenRefinerBlock(index) => index,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn published_tier_constants_stay_pinned_to_the_reviewed_revision() {
        assert_eq!(
            H3_TURBO_LORA_SOURCE_REVISION,
            "dc559027db79c174125df4d827db55cd11178860"
        );
        assert_eq!(H3_TURBO_LORA_REPOSITORY, "Comfy-Org/MiniMax-H3");
        assert_eq!(H3_TURBO_LORA_TENSOR_COUNT, H3_TURBO_LORA_MODULE_COUNT * 3);
        assert_eq!(H3_TURBO_LORA_MODULE_COUNT, 208);

        let pinned = [
            (
                H3TurboLoraTier::Fl2v8StepV10,
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                1_956_193_000_u64,
                "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
                73_632_u64,
                "eadcdb12138db967789252da26d2abe41905b2579e1cf07b866a573e88d298fd",
                8.0_f32,
                0.0625_f32,
                H3TransformerTask::T2VaFl2Va,
            ),
            (
                H3TurboLoraTier::Fl2v768p4StepV10,
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                1_956_192_992,
                "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",
                73_624,
                "3db9fe99ff46229525c43cbe6ba5bafc8d96bdeb22ee69949ef61d4d58d561d8",
                128.0,
                1.0,
                H3TransformerTask::T2VaFl2Va,
            ),
            (
                H3TurboLoraTier::Ref2v4StepV10,
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                1_956_193_000,
                "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
                73_632,
                "53370bff715f074018793b9ebc71fa0ecd8bdfd8c5554a716ccf7bf5e6a6f745",
                8.0,
                0.0625,
                H3TransformerTask::Ref2Va,
            ),
        ];

        assert_eq!(pinned.len(), H3TurboLoraTier::ALL.len());
        for (tier, file_name, bytes, sha, header_len, header_sha, alpha, scale, task) in pinned {
            assert_eq!(tier.file_name(), file_name);
            assert_eq!(tier.repository_path(), format!("loras/{file_name}"));
            assert_eq!(tier.file_bytes(), bytes);
            assert_eq!(tier.content_sha256(), sha);
            assert_eq!(tier.header_len(), header_len);
            assert_eq!(tier.header_identity_sha256(), header_sha);
            assert_eq!(tier.training_alpha(), alpha);
            assert_eq!(tier.training_scale(), scale);
            assert_eq!(tier.task(), task);
            // The published payload is byte-identical across tiers; only the
            // JSON header length moves the total.
            assert_eq!(
                tier.file_bytes(),
                8 + tier.header_len() + H3_TURBO_LORA_PAYLOAD_BYTES
            );
            assert_eq!(tier.training_scale(), tier.expectation().scale());
        }
    }

    #[test]
    fn tier_identities_and_digests_are_distinct() {
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut headers = BTreeSet::new();
        for tier in H3TurboLoraTier::ALL {
            assert!(ids.insert(tier.stable_id()));
            assert!(digests.insert(tier.content_sha256()));
            assert!(headers.insert(tier.header_identity_sha256()));
            assert_eq!(tier.content_sha256().len(), 64);
            assert_eq!(tier.header_identity_sha256().len(), 64);
        }
    }

    #[test]
    fn published_expectation_matches_the_shipped_checkpoint_geometry() {
        for tier in H3TurboLoraTier::ALL {
            let expectation = tier.expectation();
            assert_eq!(expectation.module_count(), H3_TURBO_LORA_MODULE_COUNT);
            assert_eq!(expectation.tensor_count(), H3_TURBO_LORA_TENSOR_COUNT);
            assert_eq!(expectation.training_rank, H3_TURBO_LORA_TRAINING_RANK);
            assert_eq!(expectation.task, tier.task());
        }
    }

    #[test]
    fn fused_qkv_triples_rank_and_alpha_so_the_scale_is_uniform() {
        for kind in H3TurboLoraModuleKind::ALL {
            let rank = kind.rank(H3_TURBO_LORA_TRAINING_RANK);
            let alpha = kind.alpha(8.0);
            assert_eq!(alpha / rank as f32, 0.0625);
            if kind.fuses_qkv() {
                assert_eq!(rank, 384);
                assert_eq!(alpha, 24.0);
            } else {
                assert_eq!(rank, 128);
                assert_eq!(alpha, 8.0);
            }
            assert_eq!(
                H3TurboLoraModuleKind::from_suffix(kind.suffix()),
                Some(kind)
            );
        }
        assert_eq!(H3TurboLoraModuleKind::from_suffix("attn.q_norm"), None);
    }

    #[test]
    fn module_scopes_name_base_checkpoint_prefixes() {
        assert_eq!(
            H3TurboLoraModuleScope::MainBlock(49).base_prefix(),
            "blocks.49"
        );
        assert_eq!(
            H3TurboLoraModuleScope::TokenRefinerBlock(1).base_prefix(),
            "token_refiner.blocks.1"
        );
        assert_eq!(H3TurboLoraModuleScope::MainBlock(7).index(), 7);
    }
}
