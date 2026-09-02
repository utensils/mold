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
//! Constants are pinned per tier to the repository and revision that tier's
//! adapter is published at, and were read from the published safetensors
//! headers over HTTP range requests.
//!
//! Two SHAPES share this one contract ([`H3TurboLoraShape`]). A **uniform**
//! adapter is a published PEFT export: one rank for every module (tripled on
//! the fused Q/K/V module) and one `alpha` scalar per module. Its effective
//! scale differs 16x between tiers, so it is always **read from the file's own
//! `alpha` tensors** and merely cross-checked against the pinned tier value;
//! nothing here hands a hardcoded scale to a forward pass. A **dynamic**
//! adapter is an SVD-resized derivative of one uniform tier: the rank is a
//! per-module fact read from the file's own header and bounded by the source
//! rank, there are no `alpha` tensors at all, and the source `alpha / rank`
//! was multiplied into `lora_B` when it was exported — so its runtime scale is
//! exactly `1.0` and the declared `baked_scale` is a fact the file is CHECKED
//! against, never a factor applied at forward time. What a dynamic file
//! relaxes is exactly two things, the per-module rank literal and the alpha
//! tensor; the 208-module set, the key space, the dtype, the in/out features,
//! `A.dim(0) == B.dim(1)`, the tensor count, the header identity, the file
//! size, and the full content digest all stay exact.
//!
//! This module parses, authenticates, and validates. It performs no tensor
//! loading and no forward computation, and it registers no engine, capability,
//! catalog entry, or download.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::comfy_dit::{
    h3_block_linear_targets, safetensors_dtype_size, sha256_hex, strict_json_value,
    H3BlockLinearScope, H3ComfyInt8Cancellation, H3OpenedFileIdentity, FILE_READ_CHUNK_BYTES,
    H3_BLOCK_LINEAR_WEIGHT_SUFFIXES, MAX_HEADER_BYTES, MAX_TENSORS, MAX_TENSOR_KEY_BYTES,
    MAX_TENSOR_RANK,
};
use super::dit::{
    expected_h3_weight_specs, H3AdaLnMode, H3PrecisionProfile, H3TransformerConfig,
    H3TransformerTask,
};

/// Comfy-Org repository revision that re-hosts the Turbo adapters under
/// `loras/`. It postdates [`super::comfy_dit::H3_COMFY_ORG_SOURCE_REVISION`],
/// whose tree carries no `loras/` directory at all.
pub const H3_TURBO_LORA_SOURCE_REVISION: &str = "dc559027db79c174125df4d827db55cd11178860";
/// Comfy-Org repository that publishes both the pruned base checkpoints and
/// the adapters it re-hosts.
pub const H3_TURBO_LORA_REPOSITORY: &str = "Comfy-Org/MiniMax-H3";
/// Repository-relative directory holding every adapter Comfy-Org publishes.
/// The lightx2v repository has no such directory — its files sit at the root —
/// which is why the path is a per-tier fact.
pub const H3_TURBO_LORA_DIRECTORY: &str = "loras";
/// ModelTC/lightx2v repository publishing the 768p Turbo adapters Comfy-Org
/// never re-hosted. Only ADAPTERS come from it; the base checkpoint a tier
/// overlays stays [`H3_TURBO_LORA_REPOSITORY`]'s. Its copies of the v1.0
/// adapters are byte-identical to Comfy-Org's, which is the provenance
/// corroboration for pinning it at all.
pub const H3_TURBO_LORA_LIGHTX2V_REPOSITORY: &str = "lightx2v/Minimax-h3-Turbo";
/// Pinned lightx2v revision. Files there sit at the repository ROOT.
pub const H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION: &str = "05ef678438e84933c406131b59abbf86919b3aac";
/// Repository publishing the SVD-resized dynamic-rank Turbo adapters.
///
/// These are DERIVATIVES of the rank-128 adapters above: an exact compact SVD
/// per module, `sqrt(S)`-balanced back into `lora_A`/`lora_B`, the alpha
/// tensors dropped and the source `alpha / rank` multiplied into `lora_B`.
/// Each file therefore keeps `training_rank "128"` — the rank it was resized
/// FROM — carries no `alpha` tensor at all, and declares a numeric
/// `baked_scale`. Only ADAPTERS come from here; the base checkpoint a tier
/// overlays stays [`H3_TURBO_LORA_REPOSITORY`]'s. Files sit at the repository
/// ROOT, like lightx2v's.
pub const H3_TURBO_LORA_DRBAPH_REPOSITORY: &str = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI";
/// Pinned drbaph revision. Files there sit at the repository ROOT.
pub const H3_TURBO_LORA_DRBAPH_SOURCE_REVISION: &str = "be8eb3ea3466cbb7def202ffec0d2fdc054256ac";
/// Every ComfyUI-layout adapter key is namespaced under this prefix; the
/// remainder is a base-checkpoint tensor name.
pub const H3_TURBO_LORA_KEY_PREFIX: &str = "diffusion_model.";
/// `208 modules x {lora_A, lora_B, alpha}`. There is no `__metadata__` entry to
/// subtract from this count.
pub const H3_TURBO_LORA_TENSOR_COUNT: usize = 624;
/// `208 modules x {lora_A, lora_B}`. An SVD-resized adapter bakes the source
/// `alpha / rank` into `lora_B` and ships no `alpha` tensor, so its tensor
/// count is two thirds of [`H3_TURBO_LORA_TENSOR_COUNT`]. The count is checked
/// FIRST, before any shape, so a rank-128 file offered for a resized tier is
/// named as a count mismatch rather than as 208 shape mismatches.
pub const H3_TURBO_LORA_DYNAMIC_TENSOR_COUNT: usize = 416;
/// `52 blocks x 4 linear modules` — 50 main blocks plus 2 token-refiner blocks.
pub const H3_TURBO_LORA_MODULE_COUNT: usize = 208;
/// Tensor payload bytes shared by all five published RANK-UNIFORM adapters;
/// only the JSON header length differs between them. A rank-dynamic tier's
/// payload is a per-file fact derived from its own golden header, never this
/// constant.
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

/// Mirrors [`H3_BLOCK_LINEAR_WEIGHT_SUFFIXES`] with `.weight` trimmed, purely
/// so [`H3TurboLoraModuleKind`] can be a `const` enum. A contract test pins
/// the two together, and module derivation resolves every kind through the
/// shared authority so an unmapped suffix fails closed instead of vanishing.
const MODULE_SUFFIXES: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];
const LORA_A_SUFFIX: &str = ".lora_A.weight";
const LORA_B_SUFFIX: &str = ".lora_B.weight";
const ALPHA_SUFFIX: &str = ".alpha";
/// Base-checkpoint sidecars that must never appear in an adapter. Their
/// presence means a merged or quantized checkpoint was supplied in place of one.
const BASE_SIDECAR_SUFFIXES: [&str; 2] = [".weight_scale", ".comfy_quant"];
const ALPHA_BYTES: u64 = 4;
/// Upper bound on `num_layers + token_refiner_num_layers` any expectation may
/// describe. The published geometry is 52; the bound only has to keep
/// caller-supplied arithmetic away from overflow.
const MAX_TURBO_BLOCKS: usize = 4_096;
/// Upper bound on `training_rank`. The published tiers train at 128.
const MAX_TURBO_TRAINING_RANK: usize = 65_536;
const STRUCTURE_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.turbo-lora-structure.v1\0";
/// Separate domain for the rank-dynamic structure digest. A resized adapter
/// hashes per-module ranks and its baked scale where a uniform one hashes one
/// alpha per module, so the two must never share a domain — and every
/// `adapter_identity_sha256` already recorded for a uniform tier stays byte
/// identical because that path still hashes under
/// [`STRUCTURE_IDENTITY_DOMAIN`].
const STRUCTURE_IDENTITY_DOMAIN_DYNAMIC: &[u8] =
    b"mold.minimax-h3.turbo-lora-structure.dynamic.v1\0";
/// `__metadata__` key carrying the scale a resized adapter already multiplied
/// into `lora_B`. It is REQUIRED on a rank-dynamic file and must parse as a
/// float: drbaph's own v1.1 exports declare an English sentence here, which is
/// exactly the file this contract has to refuse.
const METADATA_BAKED_SCALE_KEY: &str = "baked_scale";
/// `__metadata__` key naming the published file a resized adapter was derived
/// from. When present it must be the source tier's own file name.
const METADATA_RESIZED_FROM_KEY: &str = "resized_from";
/// Header + structure only. Deliberately NOT an artifact identity.
const CONTRACT_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.turbo-lora-contract.v1\0";
/// Binds the verified content digest; this is the artifact identity.
const ADAPTER_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.turbo-lora-adapter.v2\0";

/// How one reviewed adapter lays its low-rank factors out.
///
/// The two shapes share EVERYTHING except rank, alpha, and scale: the same 208
/// modules derived from [`h3_block_linear_targets`], the same key space, the
/// same BF16 dtype, the same in/out features, the same base-conflict
/// detection, the same header parsing, and the same runtime loader.
///
/// - [`Self::Uniform`] is a published PEFT export: every module's rank is
///   `training_rank` (tripled on the fused Q/K/V module), and each module
///   carries its own `alpha` scalar, so the runtime scale is `alpha / rank`.
/// - [`Self::Dynamic`] is an SVD-resized derivative of one Uniform tier: the
///   rank is a PER-MODULE fact read from the file's own header, there are no
///   `alpha` tensors, and the source `alpha / rank` was already multiplied
///   into `lora_B`. The runtime scale is therefore exactly `1.0` — applying
///   the pinned `baked_scale` again would apply it twice.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum H3TurboLoraShape {
    /// Published PEFT export: every module rank is `training_rank` (x3 fused),
    /// with one `alpha` scalar per module.
    Uniform {
        training_rank: usize,
        training_alpha: f32,
    },
    /// SVD-resized export: per-module rank read from the header, no alpha
    /// tensors, `alpha / rank` already multiplied into `lora_B`. `source` is
    /// the Uniform tier the file was resized from, and `baked_scale` is the
    /// scale that was multiplied in — a declared fact to check the file
    /// against, never a factor applied at forward time.
    Dynamic {
        source: H3TurboLoraTier,
        baked_scale: f32,
    },
}

impl H3TurboLoraShape {
    /// The rank the adapter was TRAINED at, which for a resized file is its
    /// source's rank rather than any rank in its own header.
    ///
    /// A resized file keeps `__metadata__.training_rank "128"`; comparing that
    /// against a per-module rank would refuse every one of them.
    pub fn source_training_rank(&self) -> usize {
        match self {
            Self::Uniform { training_rank, .. } => *training_rank,
            // Deliberately one level deep: `validate` refuses a Dynamic tier
            // whose source is itself Dynamic, so this arm is unreachable for
            // any reviewed tier and must not recurse.
            Self::Dynamic { source, .. } => match source.shape() {
                Self::Uniform { training_rank, .. } => training_rank,
                Self::Dynamic { .. } => H3_TURBO_LORA_TRAINING_RANK,
            },
        }
    }

    /// The per-module `alpha` a uniform export carries. A resized export has
    /// none — the alphas were dropped when the scale was baked in.
    pub fn training_alpha(&self) -> Option<f32> {
        match self {
            Self::Uniform { training_alpha, .. } => Some(*training_alpha),
            Self::Dynamic { .. } => None,
        }
    }

    /// The factor a forward pass multiplies the low-rank product by.
    pub fn effective_scale(&self) -> f32 {
        match self {
            Self::Uniform {
                training_rank,
                training_alpha,
            } => training_alpha / *training_rank as f32,
            // Already inside `lora_B`; multiplying again would square it.
            Self::Dynamic { .. } => 1.0,
        }
    }

    /// Tensors per module: `lora_A`, `lora_B`, and — uniform only — `alpha`.
    pub const fn tensors_per_module(&self) -> usize {
        match self {
            Self::Uniform { .. } => 3,
            Self::Dynamic { .. } => 2,
        }
    }

    /// The declared scale a resized file must carry in its `__metadata__`.
    pub const fn baked_scale(&self) -> Option<f32> {
        match self {
            Self::Uniform { .. } => None,
            Self::Dynamic { baked_scale, .. } => Some(*baked_scale),
        }
    }

    /// The Uniform tier a resized file was derived from.
    pub const fn source_tier(&self) -> Option<H3TurboLoraTier> {
        match self {
            Self::Uniform { .. } => None,
            Self::Dynamic { source, .. } => Some(*source),
        }
    }

    pub const fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic { .. })
    }
}

/// One of the eight reviewed published Turbo adapters. Detection never uses a
/// filename: the independently parsed header must agree with this authority,
/// and each tier names the repository and revision it was published at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum H3TurboLoraTier {
    /// FL2V Turbo, 8 transformer evaluations, v1.0, trained at 544p.
    Fl2v8StepV10,
    /// FL2V Turbo, 4 transformer evaluations, v1.0, trained at 768p.
    Fl2v768p4StepV10,
    /// Ref2V Turbo, 4 transformer evaluations, v0.1.
    Ref2v4StepV10,
    /// FL2V Turbo, 4 transformer evaluations, v1.1, trained at 768p. Published
    /// by lightx2v only.
    Fl2v768p4StepV11,
    /// FL2V Turbo, 8 transformer evaluations, v1.0, trained at 768p. Published
    /// by lightx2v only.
    Fl2v768p8StepV10,
    /// SVD-resized derivative of [`Self::Fl2v768p4StepV10`], average rank 21.
    Fl2v768p4StepV10Rank21,
    /// SVD-resized derivative of [`Self::Fl2v8StepV10`], average rank 21.
    Fl2v8StepV10Rank21,
    /// SVD-resized derivative of [`Self::Ref2v4StepV10`], average rank 21.
    Ref2v4StepV10Rank21,
}

impl H3TurboLoraTier {
    /// Every reviewed tier, in a stable order.
    pub const ALL: [Self; 8] = [
        Self::Fl2v8StepV10,
        Self::Fl2v768p4StepV10,
        Self::Ref2v4StepV10,
        Self::Fl2v768p4StepV11,
        Self::Fl2v768p8StepV10,
        Self::Fl2v768p4StepV10Rank21,
        Self::Fl2v8StepV10Rank21,
        Self::Ref2v4StepV10Rank21,
    ];

    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.v1",
            Self::Fl2v768p4StepV10 => "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.v1",
            Self::Ref2v4StepV10 => "minimax-h3.turbo-lora.ref2v-4step-v0.1.comfyui-bf16.v1",
            Self::Fl2v768p4StepV11 => "minimax-h3.turbo-lora.fl2v-4step-768p-v1.1.comfyui-bf16.v1",
            Self::Fl2v768p8StepV10 => "minimax-h3.turbo-lora.fl2v-8step-768p-v1.0.comfyui-bf16.v1",
            Self::Fl2v768p4StepV10Rank21 => {
                "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.resized-avg-rank-21.v1"
            }
            Self::Fl2v8StepV10Rank21 => {
                "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.resized-avg-rank-21.v1"
            }
            Self::Ref2v4StepV10Rank21 => {
                "minimax-h3.turbo-lora.ref2v-4step-v0.1.comfyui-bf16.resized-avg-rank-21.v1"
            }
        }
    }

    pub const fn file_name(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 => "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            Self::Fl2v768p4StepV10 => {
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
            }
            Self::Ref2v4StepV10 => "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            Self::Fl2v768p4StepV11 => {
                "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
            }
            Self::Fl2v768p8StepV10 => {
                "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
            }
            Self::Fl2v768p4StepV10Rank21 => {
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors"
            }
            Self::Fl2v8StepV10Rank21 => {
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors"
            }
            Self::Ref2v4StepV10Rank21 => {
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors"
            }
        }
    }

    /// The repository this tier's adapter is published in.
    pub const fn source_repository(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 | Self::Fl2v768p4StepV10 | Self::Ref2v4StepV10 => {
                H3_TURBO_LORA_REPOSITORY
            }
            Self::Fl2v768p4StepV11 | Self::Fl2v768p8StepV10 => H3_TURBO_LORA_LIGHTX2V_REPOSITORY,
            Self::Fl2v768p4StepV10Rank21 | Self::Fl2v8StepV10Rank21 | Self::Ref2v4StepV10Rank21 => {
                H3_TURBO_LORA_DRBAPH_REPOSITORY
            }
        }
    }

    /// The pinned revision of [`Self::source_repository`] this tier's adapter
    /// is published at.
    pub const fn source_revision(self) -> &'static str {
        match self {
            Self::Fl2v8StepV10 | Self::Fl2v768p4StepV10 | Self::Ref2v4StepV10 => {
                H3_TURBO_LORA_SOURCE_REVISION
            }
            Self::Fl2v768p4StepV11 | Self::Fl2v768p8StepV10 => {
                H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION
            }
            Self::Fl2v768p4StepV10Rank21 | Self::Fl2v8StepV10Rank21 | Self::Ref2v4StepV10Rank21 => {
                H3_TURBO_LORA_DRBAPH_SOURCE_REVISION
            }
        }
    }

    /// Repository-relative path at [`Self::source_revision`]. Comfy-Org
    /// publishes every adapter under [`H3_TURBO_LORA_DIRECTORY`]; lightx2v
    /// publishes at the repository root, so the path is per-tier rather than
    /// a shared prefix.
    ///
    /// Deliberately an exhaustive match rather than "lightx2v or else
    /// `loras/`": a third publishing source must state its own layout here
    /// and cannot inherit a prefix by being unrecognized.
    pub fn repository_path(self) -> String {
        match self {
            Self::Fl2v8StepV10 | Self::Fl2v768p4StepV10 | Self::Ref2v4StepV10 => {
                format!("{H3_TURBO_LORA_DIRECTORY}/{}", self.file_name())
            }
            Self::Fl2v768p4StepV11
            | Self::Fl2v768p8StepV10
            | Self::Fl2v768p4StepV10Rank21
            | Self::Fl2v8StepV10Rank21
            | Self::Ref2v4StepV10Rank21 => self.file_name().to_owned(),
        }
    }

    pub const fn file_bytes(self) -> u64 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 | Self::Fl2v768p8StepV10 => 1_956_193_000,
            Self::Fl2v768p4StepV10 | Self::Fl2v768p4StepV11 => 1_956_192_992,
            Self::Fl2v768p4StepV10Rank21 => 298_177_224,
            Self::Fl2v8StepV10Rank21 => 327_035_608,
            Self::Ref2v4StepV10Rank21 => 326_935_264,
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
            Self::Fl2v768p4StepV11 => {
                "449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558"
            }
            Self::Fl2v768p8StepV10 => {
                "08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980"
            }
            Self::Fl2v768p4StepV10Rank21 => {
                "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115"
            }
            Self::Fl2v8StepV10Rank21 => {
                "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840"
            }
            Self::Ref2v4StepV10Rank21 => {
                "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0"
            }
        }
    }

    /// JSON header length, excluding the eight-byte safetensors length prefix.
    pub const fn header_len(self) -> u64 {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 | Self::Fl2v768p8StepV10 => 73_632,
            Self::Fl2v768p4StepV10 | Self::Fl2v768p4StepV11 => 73_624,
            Self::Fl2v768p4StepV10Rank21 => 52_928,
            Self::Fl2v8StepV10Rank21 => 52_944,
            Self::Ref2v4StepV10Rank21 => 52_952,
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
            Self::Fl2v768p4StepV11 => {
                "e7a5b995877b2997c0055cad77d1a1ef48a28bc8fd388f8b19be601249e7d27c"
            }
            Self::Fl2v768p8StepV10 => {
                "0541a8b7d525096f45df5f6e8d076f49173cb2d3d58ad233e37e04a63677d78d"
            }
            Self::Fl2v768p4StepV10Rank21 => {
                "e9a8cf11d436ab25df9667896a02c9768aca800ed5b8e5d794e80b7cb866f539"
            }
            Self::Fl2v8StepV10Rank21 => {
                "f1bbb213d10d64aaf63d4e973d72887e43d356a3352ba73534e04aa317795f2a"
            }
            Self::Ref2v4StepV10Rank21 => {
                "3c1db66284973ee4eec4e9700e12b1fc587b1aa7f85af6a811daac0d15b4db6f"
            }
        }
    }

    pub const fn task(self) -> H3TransformerTask {
        match self {
            Self::Fl2v8StepV10
            | Self::Fl2v768p4StepV10
            | Self::Fl2v768p4StepV11
            | Self::Fl2v768p8StepV10
            | Self::Fl2v768p4StepV10Rank21
            | Self::Fl2v8StepV10Rank21 => H3TransformerTask::T2VaFl2Va,
            Self::Ref2v4StepV10 | Self::Ref2v4StepV10Rank21 => H3TransformerTask::Ref2Va,
        }
    }

    /// How this tier lays its factors out, and therefore what a file of it
    /// must carry: rank per module, alpha tensors or none, and the runtime
    /// scale.
    ///
    /// A resized tier names the Uniform tier it derives from, so the source
    /// rank a metadata check compares against and the source file name
    /// `resized_from` must equal are both derived here rather than restated.
    pub const fn shape(self) -> H3TurboLoraShape {
        match self {
            Self::Fl2v8StepV10 | Self::Ref2v4StepV10 | Self::Fl2v768p8StepV10 => {
                H3TurboLoraShape::Uniform {
                    training_rank: H3_TURBO_LORA_TRAINING_RANK,
                    training_alpha: 8.0,
                }
            }
            Self::Fl2v768p4StepV10 | Self::Fl2v768p4StepV11 => H3TurboLoraShape::Uniform {
                training_rank: H3_TURBO_LORA_TRAINING_RANK,
                training_alpha: 128.0,
            },
            // The baked scale is the SOURCE tier's own `alpha / rank`, which
            // is why it differs 16x between the 4-step 768p tier and the other
            // two exactly as the sources do.
            Self::Fl2v768p4StepV10Rank21 => H3TurboLoraShape::Dynamic {
                source: Self::Fl2v768p4StepV10,
                baked_scale: 1.0,
            },
            Self::Fl2v8StepV10Rank21 => H3TurboLoraShape::Dynamic {
                source: Self::Fl2v8StepV10,
                baked_scale: 0.0625,
            },
            Self::Ref2v4StepV10Rank21 => H3TurboLoraShape::Dynamic {
                source: Self::Ref2v4StepV10,
                baked_scale: 0.0625,
            },
        }
    }

    /// `__metadata__.training_alpha`, and the exact `alpha` scalar carried by
    /// every non-fused module of a rank-uniform tier. `None` for a resized
    /// tier, which ships no alpha tensors at all.
    pub fn training_alpha(self) -> Option<f32> {
        self.shape().training_alpha()
    }

    /// The factor the forward pass multiplies this tier's low-rank product by:
    /// `alpha / rank` for a uniform tier, and exactly `1.0` for a resized one
    /// whose scale is already inside `lora_B`.
    pub fn training_scale(self) -> f32 {
        self.shape().effective_scale()
    }

    /// The declared `__metadata__.baked_scale` a resized tier's file must
    /// carry, or `None` for a uniform tier.
    pub const fn baked_scale(self) -> Option<f32> {
        self.shape().baked_scale()
    }

    /// The uniform tier a resized tier was derived from.
    pub const fn source_tier(self) -> Option<Self> {
        self.shape().source_tier()
    }

    /// The exact contract an adapter file of this tier must satisfy.
    pub fn expectation(self) -> H3TurboLoraExpectation {
        H3TurboLoraExpectation {
            config: H3TransformerConfig::default(),
            task: self.task(),
            shape: self.shape(),
            file_bytes: Some(self.file_bytes()),
            content_sha256: Some(self.content_sha256().to_owned()),
            header_len: Some(self.header_len()),
            header_identity_sha256: Some(self.header_identity_sha256().to_owned()),
            source_repository: self.source_repository().to_owned(),
            source_revision: self.source_revision().to_owned(),
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
    /// Rank/alpha/scale layout. Every other field of this contract is shared
    /// by both shapes, which is why this is one enum rather than a second
    /// expectation type.
    pub shape: H3TurboLoraShape,
    pub file_bytes: Option<u64>,
    pub content_sha256: Option<String>,
    pub header_len: Option<u64>,
    pub header_identity_sha256: Option<String>,
    /// The repository this adapter is published in. Reported verbatim on the
    /// inspection so a two-source tier table cannot be flattened onto one
    /// global provenance constant.
    pub source_repository: String,
    /// The pinned revision of [`Self::source_repository`].
    pub source_revision: String,
}

impl H3TurboLoraExpectation {
    /// The rank the adapter was TRAINED at — the source rank for a resized
    /// file, which is also the upper bound every per-module rank must sit
    /// inside.
    pub fn training_rank(&self) -> usize {
        self.shape.source_training_rank()
    }

    /// The per-module `alpha` a uniform export carries, `None` for a resized
    /// one.
    pub fn training_alpha(&self) -> Option<f32> {
        self.shape.training_alpha()
    }

    /// The factor a forward pass multiplies by: `alpha / rank` for a uniform
    /// export (identical for fused and non-fused modules because the ComfyUI
    /// conversion scales both by three), and `1.0` for a resized one whose
    /// scale is already baked into `lora_B`.
    pub fn scale(&self) -> f32 {
        self.shape.effective_scale()
    }

    /// Refuse a geometry whose arithmetic could overflow or whose training
    /// constants are not usable, before any file is touched. Callers may
    /// supply an expectation, so nothing downstream may assume the published
    /// numbers.
    pub fn validate(&self) -> H3TurboLoraResult<()> {
        self.config.validate().map_err(|error| {
            failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                format!("H3 Turbo expectation carries an invalid transformer config: {error}"),
            )
        })?;
        let blocks = self
            .config
            .num_layers
            .checked_add(self.config.token_refiner_num_layers)
            .ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    "H3 Turbo expectation block count overflows",
                )
            })?;
        if blocks == 0 || blocks > MAX_TURBO_BLOCKS {
            return Err(failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                format!(
                    "H3 Turbo expectation describes {blocks} blocks, bound is {MAX_TURBO_BLOCKS}"
                ),
            ));
        }
        let training_rank = self.training_rank();
        if training_rank == 0 || training_rank > MAX_TURBO_TRAINING_RANK {
            return Err(failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                format!("H3 Turbo training rank {training_rank} is outside 1..={MAX_TURBO_TRAINING_RANK}"),
            ));
        }
        if let Some(training_alpha) = self.training_alpha() {
            if !training_alpha.is_finite() || training_alpha <= 0.0 {
                return Err(failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!("H3 Turbo training alpha {training_alpha} must be finite and positive"),
                ));
            }
        }
        if !self.scale().is_finite() || self.scale() <= 0.0 {
            return Err(failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                "H3 Turbo training alpha/rank must resolve to a finite positive scale",
            ));
        }
        if let H3TurboLoraShape::Dynamic {
            source,
            baked_scale,
        } = self.shape
        {
            if !baked_scale.is_finite() || baked_scale <= 0.0 {
                return Err(failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!("H3 Turbo baked scale {baked_scale} must be finite and positive"),
                ));
            }
            // A resized adapter is a derivative of ONE published rank-uniform
            // adapter of its OWN task. Chaining resizes would leave no pinned
            // source rank to bound the per-module ranks against, and crossing
            // tasks would let a ref2v resize claim an FL2VA source.
            if source.shape().is_dynamic() {
                return Err(failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!(
                        "H3 Turbo resized adapter names {source:?} as its source, which is itself resized"
                    ),
                ));
            }
            if source.task() != self.task {
                return Err(failure(
                    H3TurboLoraErrorCode::TaskAuthorityMismatch,
                    format!(
                        "H3 Turbo resized adapter for task {:?} names source {source:?} of task {:?}",
                        self.task,
                        source.task()
                    ),
                ));
            }
        }
        // Both counts and the widest rank must be representable.
        self.tensor_count()?;
        for kind in H3TurboLoraModuleKind::ALL {
            kind.checked_rank(training_rank)?;
        }
        Ok(())
    }

    pub fn module_count(&self) -> H3TurboLoraResult<usize> {
        self.config
            .num_layers
            .checked_add(self.config.token_refiner_num_layers)
            .and_then(|blocks| blocks.checked_mul(H3_BLOCK_LINEAR_WEIGHT_SUFFIXES.len()))
            .ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    "H3 Turbo expectation module count overflows",
                )
            })
    }

    pub fn tensor_count(&self) -> H3TurboLoraResult<usize> {
        self.module_count()?
            .checked_mul(self.shape.tensors_per_module())
            .ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    "H3 Turbo expectation tensor count overflows",
                )
            })
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
    /// `authenticate_*` was asked to run without a source-pinned digest.
    MissingContentPin,
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

fn failure(code: H3TurboLoraErrorCode, message: impl Into<String>) -> H3TurboLoraError {
    H3TurboLoraError {
        code,
        message: message.into(),
    }
}

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

    /// Rank of this module's `lora_A` / `lora_B` pair, refusing an overflowing
    /// caller-supplied training rank.
    pub fn checked_rank(self, training_rank: usize) -> H3TurboLoraResult<usize> {
        if !self.fuses_qkv() {
            return Ok(training_rank);
        }
        training_rank
            .checked_mul(H3_TURBO_LORA_FUSED_QKV_MULTIPLE)
            .ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!("H3 Turbo fused rank overflows at training rank {training_rank}"),
                )
            })
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

    const fn discriminant(self) -> u8 {
        match self {
            Self::MainBlock(_) => 0,
            Self::TokenRefinerBlock(_) => 1,
        }
    }
}

/// Where one validated adapter tensor lives inside the file. Byte ranges are
/// retained so a later streaming loader never has to re-parse the header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3TurboLoraTensorRef {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

/// One validated low-rank module of the adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct H3TurboLoraModule {
    /// Full adapter key prefix, `diffusion_model.` included.
    pub name: String,
    /// The base-checkpoint weight this module overlays.
    pub base_weight_name: String,
    pub scope: H3TurboLoraModuleScope,
    pub kind: H3TurboLoraModuleKind,
    /// This module's OWN rank: `training_rank` (x3 fused) for a uniform
    /// adapter, and the rank read from `lora_A.shape[0]` for a resized one.
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    /// Read from the file's own `alpha` tensor, never assumed. `None` for a
    /// resized adapter, which carries no alpha tensors.
    pub alpha: Option<f32>,
    /// The factor the forward pass applies: `alpha / rank` for a uniform
    /// adapter, `1.0` for a resized one whose scale is already in `lora_B`.
    pub scale: f32,
    pub lora_a: H3TurboLoraTensorRef,
    pub lora_b: H3TurboLoraTensorRef,
    /// `None` for a resized adapter.
    pub alpha_tensor: Option<H3TurboLoraTensorRef>,
}

/// The result of a header-only inspection.
///
/// This is **not** an authority. Its payload bytes were never read, so
/// arbitrary `lora_A` / `lora_B` contents that happen to preserve the header,
/// the size, and the alpha scalars produce a perfectly valid inspection. It
/// deliberately carries no tier and no content digest, and no loader may
/// accept it in place of an [`H3TurboLoraContract`].
#[derive(Clone, Debug, PartialEq)]
pub struct H3TurboLoraInspection {
    pub task: H3TransformerTask,
    /// The expectation's rank/alpha layout, so a consumer can tell a resized
    /// adapter from a uniform one without re-deriving it from the modules.
    pub shape: H3TurboLoraShape,
    /// The repository the expectation named, never a global constant.
    pub source_repository: String,
    pub source_repository_revision: String,
    pub file_bytes: u64,
    pub header_len: u64,
    pub header_identity_sha256: String,
    pub tensor_count: usize,
    /// The rank the adapter was TRAINED at — the SOURCE rank for a resized
    /// adapter, whose own per-module ranks live on each module.
    pub training_rank: usize,
    /// `None` for a resized adapter, which carries no alpha tensors.
    pub training_alpha: Option<f32>,
    /// The effective runtime scale: the one `alpha / rank` every module of a
    /// uniform adapter agreed on, or `1.0` for a resized one.
    pub scale: f32,
    pub payload_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub modules: BTreeMap<String, H3TurboLoraModule>,
    /// SHA-256 over the sorted validated structure: module names, kinds,
    /// ranks, in/out features, and alpha bits. Structural only — it says
    /// nothing about the payload bytes.
    pub structure_identity_sha256: String,
    /// SHA-256 over the header identity and the structure identity.
    ///
    /// This identifies the *shape of the contract*, never the artifact. It
    /// must never be used as an artifact, cache, or provenance identity; use
    /// [`H3TurboLoraContract::adapter_identity_sha256`], which binds the
    /// verified content digest.
    pub contract_identity_sha256: String,
}

impl H3TurboLoraInspection {
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Modules attached to one streamed main block, in a stable order.
    pub fn main_block_modules(&self, index: usize) -> Vec<&H3TurboLoraModule> {
        self.modules_for_scope(H3TurboLoraModuleScope::MainBlock(index))
    }

    /// Modules attached to one resident token-refiner block.
    pub fn token_refiner_modules(&self, index: usize) -> Vec<&H3TurboLoraModule> {
        self.modules_for_scope(H3TurboLoraModuleScope::TokenRefinerBlock(index))
    }

    fn modules_for_scope(&self, scope: H3TurboLoraModuleScope) -> Vec<&H3TurboLoraModule> {
        self.modules
            .values()
            .filter(|module| module.scope == scope)
            .collect()
    }

    /// Encoded `lora_A` + `lora_B` bytes of one scope, for the host and device
    /// budgets a later loader has to charge.
    pub fn scope_delta_bytes(&self, scope: H3TurboLoraModuleScope) -> u64 {
        self.modules
            .values()
            .filter(|module| module.scope == scope)
            .map(|module| {
                (module.lora_a.data_offsets[1] - module.lora_a.data_offsets[0])
                    + (module.lora_b.data_offsets[1] - module.lora_b.data_offsets[0])
            })
            .sum()
    }
}

/// An authenticated published Turbo adapter.
///
/// Every field is private and there is no public constructor, so the only way
/// to obtain one is [`authenticate_h3_turbo_lora_adapter`] — which reads the
/// complete file behind a descriptor-identity fence and verifies the
/// source-pinned content digest. An inspection can therefore never be
/// substituted for this type.
#[derive(Clone, Debug, PartialEq)]
pub struct H3TurboLoraContract {
    tier: H3TurboLoraTier,
    content_sha256: String,
    adapter_identity_sha256: String,
    inspection: H3TurboLoraInspection,
}

impl H3TurboLoraContract {
    /// The reviewed tier whose pinned size and digest this file matched.
    pub fn tier(&self) -> H3TurboLoraTier {
        self.tier
    }

    /// SHA-256 of the complete verified file.
    pub fn content_sha256(&self) -> &str {
        &self.content_sha256
    }

    /// The artifact identity: tier, task, header identity, validated
    /// structure, **and** the verified content digest. Safe as a cache,
    /// provenance, or frozen-plan identity.
    pub fn adapter_identity_sha256(&self) -> &str {
        &self.adapter_identity_sha256
    }

    /// The validated structure this authority was minted from.
    pub fn inspection(&self) -> &H3TurboLoraInspection {
        &self.inspection
    }

    pub fn task(&self) -> H3TransformerTask {
        self.inspection.task
    }

    pub fn scale(&self) -> f32 {
        self.inspection.scale
    }

    pub fn modules(&self) -> &BTreeMap<String, H3TurboLoraModule> {
        &self.inspection.modules
    }

    pub fn module_count(&self) -> usize {
        self.inspection.module_count()
    }

    pub fn main_block_modules(&self, index: usize) -> Vec<&H3TurboLoraModule> {
        self.inspection.main_block_modules(index)
    }

    pub fn token_refiner_modules(&self, index: usize) -> Vec<&H3TurboLoraModule> {
        self.inspection.token_refiner_modules(index)
    }

    pub fn scope_delta_bytes(&self, scope: H3TurboLoraModuleScope) -> u64 {
        self.inspection.scope_delta_bytes(scope)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TurboHeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
struct TurboParsedHeader {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TurboHeaderTensor>,
    header_len: u64,
    file_len: u64,
    header_identity_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTurboTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Parse and validate one adapter header against the published tier contract.
///
/// The result is deliberately an [`H3TurboLoraInspection`] and **not** an
/// authority: the payload bytes are never read, so this cannot distinguish the
/// reviewed weights from arbitrary contents that preserve the header, the
/// size, and the alpha scalars. Use [`authenticate_h3_turbo_lora_adapter`] for
/// anything that acts on the weights.
pub fn inspect_h3_turbo_lora_adapter(
    path: &Path,
    tier: H3TurboLoraTier,
) -> H3TurboLoraResult<H3TurboLoraInspection> {
    inspect_h3_turbo_lora_adapter_against(path, &tier.expectation())
}

/// Parse and validate one adapter header against an explicit expectation.
///
/// Crate-internal: a caller-supplied expectation can relax every published pin,
/// so it is not part of the public surface.
pub(crate) fn inspect_h3_turbo_lora_adapter_against(
    path: &Path,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<H3TurboLoraInspection> {
    expectation.validate()?;
    let mut file = File::open(path).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to open H3 Turbo adapter: {error}"),
        )
    })?;
    let parsed = read_turbo_header(&mut file)?;
    build_inspection(&mut file, parsed, expectation)
}

/// Open and fully authenticate one published Turbo adapter.
///
/// Unlike [`inspect_h3_turbo_lora_adapter`], this refuses anything but a
/// regular non-symlink file, fences the retained descriptor's identity before
/// and after the read, and verifies the complete source-pinned content digest.
/// It still grants no execution authority: nothing here reaches a runtime
/// factory, capability, catalog entry, or download.
pub fn authenticate_h3_turbo_lora_adapter(
    path: &Path,
    tier: H3TurboLoraTier,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> H3TurboLoraResult<H3TurboLoraContract> {
    authenticate_h3_turbo_lora_adapter_against(path, &tier.expectation(), tier, cancellation)
}

/// Open and fully authenticate one adapter against an explicit expectation.
///
/// Crate-internal for the same reason as
/// [`inspect_h3_turbo_lora_adapter_against`], and it still refuses to run
/// without a source-pinned content digest — "authenticate" always means the
/// bytes were checked against a pin.
pub(crate) fn authenticate_h3_turbo_lora_adapter_against(
    path: &Path,
    expectation: &H3TurboLoraExpectation,
    tier: H3TurboLoraTier,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> H3TurboLoraResult<H3TurboLoraContract> {
    authenticate_h3_turbo_lora_adapter_retaining(path, expectation, tier, cancellation)
        .map(|(contract, _)| contract)
}

/// Authenticate and hand back the retained descriptor that was hashed.
///
/// A runtime loader must read the adapter tensors from the *same* open file
/// the content digest was computed over; re-opening by path would reintroduce
/// the swap the descriptor fence exists to prevent, and re-hashing two
/// gigabytes to close that window again would double the load cost.
pub(crate) fn authenticate_h3_turbo_lora_adapter_retaining(
    path: &Path,
    expectation: &H3TurboLoraExpectation,
    tier: H3TurboLoraTier,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> H3TurboLoraResult<(H3TurboLoraContract, File)> {
    expectation.validate()?;
    if tier.task() != expectation.task {
        return Err(failure(
            H3TurboLoraErrorCode::TaskAuthorityMismatch,
            format!(
                "H3 Turbo tier {:?} expects task {:?}, not {:?}",
                tier,
                tier.task(),
                expectation.task
            ),
        ));
    }
    let Some(expected_digest) = expectation.content_sha256.clone() else {
        return Err(failure(
            H3TurboLoraErrorCode::MissingContentPin,
            "H3 Turbo authentication requires a source-pinned content digest",
        ));
    };
    cancellation_boundary(cancellation)?;
    let symlink_metadata = std::fs::symlink_metadata(path).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to inspect H3 Turbo adapter: {error}"),
        )
    })?;
    if symlink_metadata.file_type().is_symlink() || !symlink_metadata.is_file() {
        return Err(failure(
            H3TurboLoraErrorCode::FileIdentityChanged,
            "H3 Turbo adapter authority must be opened from a regular non-symlink file",
        ));
    }
    let canonical_path = std::fs::canonicalize(path).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to canonicalize H3 Turbo adapter: {error}"),
        )
    })?;
    let mut file = File::open(&canonical_path).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to open H3 Turbo adapter: {error}"),
        )
    })?;
    let requested_identity = H3OpenedFileIdentity::from_metadata(&symlink_metadata);
    let opened_metadata = file
        .metadata()
        .map_err(|error| failure(H3TurboLoraErrorCode::Io, error.to_string()))?;
    let identity = H3OpenedFileIdentity::from_metadata(&opened_metadata);
    if identity != requested_identity {
        return Err(failure(
            H3TurboLoraErrorCode::FileIdentityChanged,
            "H3 Turbo adapter changed while its descriptor was opened",
        ));
    }
    let parsed = read_turbo_header(&mut file)?;
    // Refuse a wrongly sized file before paying to hash it.
    if let Some(expected) = expectation.file_bytes {
        if parsed.file_len != expected {
            return Err(failure(
                H3TurboLoraErrorCode::SourceSizeMismatch,
                format!(
                    "H3 Turbo adapter size {} does not match source-pinned {expected}",
                    parsed.file_len
                ),
            ));
        }
    }
    let content_sha256 = hash_open_adapter(&mut file, cancellation)?;
    if content_sha256 != expected_digest {
        return Err(failure(
            H3TurboLoraErrorCode::ContentDigestMismatch,
            format!(
                "H3 Turbo adapter content digest {content_sha256} does not match source-pinned {expected_digest}"
            ),
        ));
    }
    let inspection = build_inspection(&mut file, parsed, expectation)?;
    let final_identity = H3OpenedFileIdentity::from_metadata(
        &file
            .metadata()
            .map_err(|error| failure(H3TurboLoraErrorCode::Io, error.to_string()))?,
    );
    if final_identity != identity {
        return Err(failure(
            H3TurboLoraErrorCode::FileIdentityChanged,
            "H3 Turbo adapter changed during content verification",
        ));
    }
    let adapter_identity_sha256 = turbo_adapter_identity(
        tier,
        inspection.task,
        &inspection.header_identity_sha256,
        &inspection.structure_identity_sha256,
        &content_sha256,
    );
    Ok((
        H3TurboLoraContract {
            tier,
            content_sha256,
            adapter_identity_sha256,
            inspection,
        },
        file,
    ))
}

fn cancellation_boundary(cancellation: &dyn H3ComfyInt8Cancellation) -> H3TurboLoraResult<()> {
    if cancellation.is_cancelled() {
        return Err(failure(
            H3TurboLoraErrorCode::Cancelled,
            "MiniMax H3 Turbo adapter read was cancelled",
        ));
    }
    Ok(())
}

fn hash_open_adapter(
    file: &mut File,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> H3TurboLoraResult<String> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to seek H3 Turbo adapter for hashing: {error}"),
        )
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; FILE_READ_CHUNK_BYTES];
    loop {
        cancellation_boundary(cancellation)?;
        let read = file.read(&mut buffer).map_err(|error| {
            failure(
                H3TurboLoraErrorCode::Io,
                format!("failed to read H3 Turbo adapter for hashing: {error}"),
            )
        })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(sha256_hex(digest.finalize()))
}

fn build_inspection(
    file: &mut File,
    parsed: TurboParsedHeader,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<H3TurboLoraInspection> {
    if let Some(expected) = expectation.file_bytes {
        if parsed.file_len != expected {
            return Err(failure(
                H3TurboLoraErrorCode::SourceSizeMismatch,
                format!(
                    "H3 Turbo adapter size {} does not match source-pinned {expected}",
                    parsed.file_len
                ),
            ));
        }
    }
    if let Some(expected) = expectation.header_len {
        if parsed.header_len != expected {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!(
                    "H3 Turbo adapter header length {} does not match source-pinned {expected}",
                    parsed.header_len
                ),
            ));
        }
    }
    if let Some(expected) = expectation.header_identity_sha256.as_deref() {
        if parsed.header_identity_sha256 != expected {
            return Err(failure(
                H3TurboLoraErrorCode::HeaderIdentityMismatch,
                format!(
                    "H3 Turbo adapter header identity {} does not match source-pinned {expected}",
                    parsed.header_identity_sha256
                ),
            ));
        }
    }
    validate_turbo_metadata(&parsed.metadata, expectation)?;
    // Classification runs before any payload read so a structurally wrong file
    // is named by its structure rather than by whatever its alpha bytes hold.
    let (expected, seen) = classify_turbo_tensors(&parsed, expectation)?;
    let alphas = read_turbo_alphas(file, &parsed, expectation)?;
    let modules = assemble_turbo_modules(&expected, seen, &alphas, expectation)?;
    let structure_identity_sha256 = turbo_structure_identity(&modules, &expectation.shape);
    let contract_identity_sha256 = turbo_contract_identity(
        expectation.task,
        &parsed.header_identity_sha256,
        &structure_identity_sha256,
    );
    let payload_bytes = parsed.file_len - parsed.header_len - 8;
    Ok(H3TurboLoraInspection {
        task: expectation.task,
        shape: expectation.shape,
        source_repository: expectation.source_repository.clone(),
        source_repository_revision: expectation.source_revision.clone(),
        file_bytes: parsed.file_len,
        header_len: parsed.header_len,
        header_identity_sha256: parsed.header_identity_sha256,
        tensor_count: parsed.tensors.len(),
        training_rank: expectation.training_rank(),
        training_alpha: expectation.training_alpha(),
        scale: expectation.scale(),
        payload_bytes,
        metadata: parsed.metadata,
        modules,
        structure_identity_sha256,
        contract_identity_sha256,
    })
}

fn read_turbo_header(file: &mut File) -> H3TurboLoraResult<TurboParsedHeader> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to seek H3 Turbo adapter: {error}"),
        )
    })?;
    let file_len = file
        .metadata()
        .map_err(|error| failure(H3TurboLoraErrorCode::Io, error.to_string()))?
        .len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::InvalidHeader,
            format!("failed to read H3 Turbo safetensors header length: {error}"),
        )
    })?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(failure(
            H3TurboLoraErrorCode::InvalidHeader,
            format!("invalid H3 Turbo safetensors header length {header_len}"),
        ));
    }
    let mut bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut bytes).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::InvalidHeader,
            format!("failed to read H3 Turbo safetensors header: {error}"),
        )
    })?;
    parse_turbo_header(length, &bytes, file_len)
}

/// Parse an already-read safetensors header. Split out from
/// [`read_turbo_header`] so the checked-in published headers can be validated
/// against their declared file length without their two-gigabyte payload.
fn parse_turbo_header(
    length: [u8; 8],
    bytes: &[u8],
    file_len: u64,
) -> H3TurboLoraResult<TurboParsedHeader> {
    let header_len = u64::from_le_bytes(length);
    let root = strict_json_value(bytes, "H3 Turbo safetensors header").map_err(|message| {
        failure(
            H3TurboLoraErrorCode::InvalidHeader,
            format!("invalid H3 Turbo safetensors header: {message}"),
        )
    })?;
    let object = root.as_object().ok_or_else(|| {
        failure(
            H3TurboLoraErrorCode::InvalidHeader,
            "H3 Turbo safetensors header must be a JSON object",
        )
    })?;
    let tensor_count = object.len() - usize::from(object.contains_key("__metadata__"));
    if tensor_count > MAX_TENSORS {
        return Err(failure(
            H3TurboLoraErrorCode::InvalidHeader,
            "H3 Turbo safetensors tensor count exceeds the header bound",
        ));
    }
    let metadata = match object.get("__metadata__") {
        Some(value) => turbo_metadata_strings(value)?,
        None => BTreeMap::new(),
    };
    let data_len = file_len - header_len - 8;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() || name.len() > MAX_TENSOR_KEY_BYTES {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                "H3 Turbo safetensors contains an empty or oversized tensor key",
            ));
        }
        let raw: RawTurboTensor = serde_json::from_value(value.clone()).map_err(|error| {
            failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("invalid H3 Turbo tensor header {name:?}: {error}"),
            )
        })?;
        if raw.shape.len() > MAX_TENSOR_RANK {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo tensor {name:?} exceeds the rank bound"),
            ));
        }
        if raw.data_offsets[0] > raw.data_offsets[1] || raw.data_offsets[1] > data_len {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo tensor {name:?} has invalid data offsets"),
            ));
        }
        let elements = raw.shape.iter().try_fold(1u64, |total, dimension| {
            total.checked_mul(*dimension as u64).ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::InvalidHeader,
                    format!("H3 Turbo tensor {name:?} shape overflows"),
                )
            })
        })?;
        let width = safetensors_dtype_size(&raw.dtype).ok_or_else(|| {
            failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("unsupported H3 Turbo safetensors dtype {:?}", raw.dtype),
            )
        })?;
        let expected_bytes = elements.checked_mul(width).ok_or_else(|| {
            failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo tensor {name:?} byte size overflows"),
            )
        })?;
        if raw.data_offsets[1] - raw.data_offsets[0] != expected_bytes {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo tensor {name:?} dtype/shape does not match its byte range"),
            ));
        }
        tensors.insert(
            name.clone(),
            TurboHeaderTensor {
                dtype: raw.dtype,
                shape: raw.shape,
                data_offsets: raw.data_offsets,
            },
        );
    }
    if tensors.is_empty() {
        return Err(failure(
            H3TurboLoraErrorCode::InvalidHeader,
            "H3 Turbo safetensors header contains no tensors",
        ));
    }
    let mut cursor = 0u64;
    let mut ranges = tensors
        .iter()
        .map(|(name, tensor)| (tensor.data_offsets, name.as_str()))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(offsets, _)| offsets[0]);
    for (offsets, name) in ranges {
        if offsets[0] != cursor {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo tensor data is non-contiguous before {name:?}"),
            ));
        }
        cursor = offsets[1];
    }
    if cursor != data_len {
        return Err(failure(
            H3TurboLoraErrorCode::InvalidHeader,
            "H3 Turbo safetensors has unclaimed trailing tensor data",
        ));
    }
    let mut identity = Sha256::new();
    identity.update(length);
    identity.update(bytes);
    Ok(TurboParsedHeader {
        metadata,
        tensors,
        header_len,
        file_len,
        header_identity_sha256: sha256_hex(identity.finalize()),
    })
}

fn turbo_metadata_strings(value: &Value) -> H3TurboLoraResult<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        failure(
            H3TurboLoraErrorCode::InvalidMetadata,
            "H3 Turbo __metadata__ must be a string map",
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_owned()))
                .ok_or_else(|| {
                    failure(
                        H3TurboLoraErrorCode::InvalidMetadata,
                        format!("H3 Turbo metadata {key:?} must be a string"),
                    )
                })
        })
        .collect()
}

/// The published `__metadata__` is advisory — the `alpha` tensors are the
/// authority — but a declared value that disagrees with the pinned tier means
/// the wrong file was supplied, so it is rejected rather than ignored.
fn validate_turbo_metadata(
    metadata: &BTreeMap<String, String>,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<()> {
    let mismatch = |key: &str, declared: &str, expected: String| {
        failure(
            H3TurboLoraErrorCode::InvalidMetadata,
            format!("H3 Turbo metadata {key:?} declares {declared:?}, expected {expected:?}"),
        )
    };
    // A resized file keeps the rank it was resized FROM, so this compares
    // against the SOURCE rank; comparing it to a per-module rank would refuse
    // every resized adapter published.
    let training_rank = expectation.training_rank();
    if let Some(declared) = metadata.get("training_rank") {
        let parsed = declared.trim().parse::<usize>().map_err(|error| {
            failure(
                H3TurboLoraErrorCode::InvalidMetadata,
                format!("H3 Turbo metadata \"training_rank\" is not an integer: {error}"),
            )
        })?;
        if parsed != training_rank {
            return Err(mismatch(
                "training_rank",
                declared,
                training_rank.to_string(),
            ));
        }
    }
    // `training_alpha`/`training_scale` describe the TRAINING of the adapter a
    // file carries, so for a resized derivative they describe its source.
    let (declared_alpha, declared_scale) = match expectation.shape {
        H3TurboLoraShape::Uniform { training_alpha, .. } => (training_alpha, expectation.scale()),
        H3TurboLoraShape::Dynamic { source, .. } => {
            let source_shape = source.shape();
            (
                source_shape.training_alpha().unwrap_or(f32::NAN),
                source_shape.effective_scale(),
            )
        }
    };
    for (key, expected) in [
        ("training_alpha", declared_alpha),
        ("training_scale", declared_scale),
    ] {
        if let Some(declared) = metadata.get(key) {
            let parsed = declared.trim().parse::<f32>().map_err(|error| {
                failure(
                    H3TurboLoraErrorCode::InvalidMetadata,
                    format!("H3 Turbo metadata {key:?} is not a float: {error}"),
                )
            })?;
            if parsed != expected {
                return Err(mismatch(key, declared, expected.to_string()));
            }
        }
    }
    if let H3TurboLoraShape::Dynamic {
        source,
        baked_scale,
    } = expectation.shape
    {
        // REQUIRED, not advisory: the baked scale is the one fact separating a
        // file whose factors already carry `alpha / rank` from one whose
        // factors do not, and applying the wrong answer silently scales every
        // delta by 16x or 1/16.
        let Some(declared) = metadata.get(METADATA_BAKED_SCALE_KEY) else {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidMetadata,
                format!(
                    "H3 Turbo resized adapter must declare a numeric {METADATA_BAKED_SCALE_KEY:?} of {baked_scale}"
                ),
            ));
        };
        // Some published resized exports write an English sentence here; a
        // sentence is not a scale, and it is refused by name rather than
        // parsed loosely or ignored.
        let parsed = declared.trim().parse::<f32>().map_err(|error| {
            failure(
                H3TurboLoraErrorCode::InvalidMetadata,
                format!(
                    "H3 Turbo metadata {METADATA_BAKED_SCALE_KEY:?} declares {declared:?}, which is not a float: {error}"
                ),
            )
        })?;
        if parsed.to_bits() != baked_scale.to_bits() {
            return Err(mismatch(
                METADATA_BAKED_SCALE_KEY,
                declared,
                baked_scale.to_string(),
            ));
        }
        if let Some(declared) = metadata.get(METADATA_RESIZED_FROM_KEY) {
            if declared != source.file_name() {
                return Err(mismatch(
                    METADATA_RESIZED_FROM_KEY,
                    declared,
                    source.file_name().to_owned(),
                ));
            }
        }
    }
    for (key, expected) in [
        ("target_format", H3_TURBO_LORA_TARGET_FORMAT),
        ("source_format", H3_TURBO_LORA_SOURCE_FORMAT),
    ] {
        if let Some(declared) = metadata.get(key) {
            if declared != expected {
                return Err(mismatch(key, declared, expected.to_owned()));
            }
        }
    }
    Ok(())
}

fn read_turbo_alphas(
    file: &mut File,
    parsed: &TurboParsedHeader,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<BTreeMap<String, f32>> {
    // A resized adapter has no alpha tensors, and classification already
    // refused any key that looked like one, so there is nothing to read.
    if expectation.shape.is_dynamic() {
        return Ok(BTreeMap::new());
    }
    let data_start = 8 + parsed.header_len;
    let mut alphas = BTreeMap::new();
    for (name, tensor) in &parsed.tensors {
        if !name.ends_with(ALPHA_SUFFIX) {
            continue;
        }
        if tensor.dtype != H3_TURBO_LORA_ALPHA_DTYPE || !tensor.shape.is_empty() {
            return Err(failure(
                H3TurboLoraErrorCode::DTypeMismatch,
                format!(
                    "H3 Turbo alpha {name:?} must be a rank-0 {H3_TURBO_LORA_ALPHA_DTYPE} scalar, found {} {:?}",
                    tensor.dtype, tensor.shape
                ),
            ));
        }
        debug_assert_eq!(
            tensor.data_offsets[1] - tensor.data_offsets[0],
            ALPHA_BYTES,
            "header parsing already bound a rank-0 F32 to four bytes"
        );
        file.seek(SeekFrom::Start(data_start + tensor.data_offsets[0]))
            .map_err(|error| {
                failure(
                    H3TurboLoraErrorCode::Io,
                    format!("failed to seek H3 Turbo alpha {name:?}: {error}"),
                )
            })?;
        let mut raw = [0u8; ALPHA_BYTES as usize];
        file.read_exact(&mut raw).map_err(|error| {
            failure(
                H3TurboLoraErrorCode::Io,
                format!("failed to read H3 Turbo alpha {name:?}: {error}"),
            )
        })?;
        let alpha = f32::from_le_bytes(raw);
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(failure(
                H3TurboLoraErrorCode::AlphaMismatch,
                format!("H3 Turbo alpha {name:?} must be finite and positive, found {alpha}"),
            ));
        }
        alphas.insert(name.clone(), alpha);
    }
    Ok(alphas)
}

/// What a module's rank axis is allowed to be.
///
/// A uniform adapter pins it exactly; a resized adapter chose a rank per
/// module, so the contract bounds it by the SOURCE rank instead — 128 for a
/// plain linear and 384 for the fused Q/K/V module, whose `lora_A` is the
/// concatenation of three per-projection factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpectedRank {
    Exact(usize),
    AtMost(usize),
}

impl ExpectedRank {
    /// The widest rank this module may carry.
    pub(super) const fn bound(self) -> usize {
        match self {
            Self::Exact(rank) | Self::AtMost(rank) => rank,
        }
    }

    pub(super) const fn accepts(self, rank: usize) -> bool {
        match self {
            Self::Exact(expected) => rank == expected,
            Self::AtMost(bound) => rank >= 1 && rank <= bound,
        }
    }

    fn describe(self) -> String {
        match self {
            Self::Exact(rank) => rank.to_string(),
            Self::AtMost(bound) => format!("1..={bound}"),
        }
    }
}

#[derive(Debug)]
pub(super) struct ExpectedModule {
    pub(super) scope: H3TurboLoraModuleScope,
    pub(super) kind: H3TurboLoraModuleKind,
    pub(super) base_weight_name: String,
    pub(super) rank: ExpectedRank,
    pub(super) in_features: usize,
    pub(super) out_features: usize,
    /// The exact `alpha` scalar a uniform module must carry; `None` for a
    /// resized adapter, whose alphas were baked into `lora_B`.
    pub(super) alpha: Option<f32>,
}

/// Derive the exact expected module set from the shared block-linear target
/// authority in `comfy_dit`.
///
/// The block/suffix product is NOT rebuilt here: `h3_block_linear_targets` is
/// the same derivation the published INT8 quantization policy consumes, so a
/// linear newly added to `H3_BLOCK_LINEAR_WEIGHT_SUFFIXES` reaches this
/// contract too. If such a target has no [`H3TurboLoraModuleKind`], the
/// contract fails closed with `ConfigMismatch` rather than silently omitting
/// an overlay the base checkpoint expects.
pub(super) fn expected_turbo_modules(
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<(BTreeMap<String, ExpectedModule>, BTreeSet<String>)> {
    let specs = expected_h3_weight_specs(
        &expectation.config,
        H3AdaLnMode::Full,
        H3PrecisionProfile::OfficialMixedBf16F32,
    )
    .map_err(|error| {
        failure(
            H3TurboLoraErrorCode::ConfigMismatch,
            format!("H3 Turbo adapter contract could not derive base weight specs: {error}"),
        )
    })?;
    let base_names = specs.keys().cloned().collect::<BTreeSet<_>>();
    let targets = h3_block_linear_targets(
        &expectation.config,
        &specs,
        &[
            H3BlockLinearScope::MainBlock,
            H3BlockLinearScope::TokenRefinerBlock,
        ],
    )
    .map_err(|error| {
        failure(
            H3TurboLoraErrorCode::ConfigMismatch,
            format!("H3 Turbo adapter contract could not derive block linear targets: {error}"),
        )
    })?;
    let mut expected = BTreeMap::new();
    for target in targets {
        let kind = H3TurboLoraModuleKind::from_suffix(target.suffix).ok_or_else(|| {
            failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                format!(
                    "H3 block linear {:?} has no Turbo adapter module kind; the shared target \
                     authority and H3TurboLoraModuleKind have diverged",
                    target.suffix
                ),
            )
        })?;
        let scope = match target.scope {
            H3BlockLinearScope::MainBlock => H3TurboLoraModuleScope::MainBlock(target.index),
            H3BlockLinearScope::TokenRefinerBlock => {
                H3TurboLoraModuleScope::TokenRefinerBlock(target.index)
            }
        };
        // Both shapes derive the rank axis from the SAME source rank; only
        // whether it is a pin or a ceiling differs.
        let source_rank = kind.checked_rank(expectation.training_rank())?;
        let previous = expected.insert(
            format!("{H3_TURBO_LORA_KEY_PREFIX}{}", target.module),
            ExpectedModule {
                scope,
                kind,
                base_weight_name: target.weight_name,
                rank: match expectation.shape {
                    H3TurboLoraShape::Uniform { .. } => ExpectedRank::Exact(source_rank),
                    H3TurboLoraShape::Dynamic { .. } => ExpectedRank::AtMost(source_rank),
                },
                in_features: target.in_features,
                out_features: target.out_features,
                alpha: expectation
                    .training_alpha()
                    .map(|training_alpha| kind.alpha(training_alpha)),
            },
        );
        if previous.is_some() {
            return Err(failure(
                H3TurboLoraErrorCode::ConfigMismatch,
                format!(
                    "H3 block linear authority yielded duplicate target {:?}",
                    target.module
                ),
            ));
        }
    }
    Ok((expected, base_names))
}

/// Bind every header tensor to an expected module slot, rejecting unknown
/// keys, base-checkpoint conflicts, wrong dtypes, and wrong shapes.
#[allow(clippy::type_complexity)]
fn classify_turbo_tensors(
    parsed: &TurboParsedHeader,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<(
    BTreeMap<String, ExpectedModule>,
    BTreeMap<String, ModuleSlots>,
)> {
    let expected_tensors = expectation.tensor_count()?;
    if parsed.tensors.len() != expected_tensors {
        return Err(failure(
            H3TurboLoraErrorCode::TensorCountMismatch,
            format!(
                "H3 Turbo adapter carries {} tensors, expected {expected_tensors}",
                parsed.tensors.len(),
            ),
        ));
    }
    let (expected, base_names) = expected_turbo_modules(expectation)?;
    let mut seen = BTreeMap::<String, ModuleSlots>::new();
    for (name, tensor) in &parsed.tensors {
        let reference = H3TurboLoraTensorRef {
            name: name.clone(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            data_offsets: tensor.data_offsets,
        };
        let Some(remainder) = name.strip_prefix(H3_TURBO_LORA_KEY_PREFIX) else {
            return Err(base_conflict_or_unknown(
                name,
                remainder_of(name),
                &base_names,
            ));
        };
        let (module_key, slot) = if let Some(module) = name.strip_suffix(LORA_A_SUFFIX) {
            (module.to_owned(), 0usize)
        } else if let Some(module) = name.strip_suffix(LORA_B_SUFFIX) {
            (module.to_owned(), 1)
        } else if let Some(module) = name.strip_suffix(ALPHA_SUFFIX) {
            (module.to_owned(), 2)
        } else {
            return Err(base_conflict_or_unknown(name, remainder, &base_names));
        };
        let Some(module_expected) = expected.get(&module_key) else {
            return Err(base_conflict_or_unknown(name, remainder, &base_names));
        };
        // A resized adapter's scale is already inside `lora_B`; an alpha
        // tensor beside it would be a second, contradictory scale authority.
        if slot == 2 && module_expected.alpha.is_none() {
            return Err(failure(
                H3TurboLoraErrorCode::AlphaMismatch,
                format!(
                    "H3 Turbo tensor {name:?} is an alpha scalar, but a resized adapter carries no alpha tensors"
                ),
            ));
        }
        let expected_dtype = if slot == 2 {
            H3_TURBO_LORA_ALPHA_DTYPE
        } else {
            H3_TURBO_LORA_WEIGHT_DTYPE
        };
        if tensor.dtype != expected_dtype {
            return Err(failure(
                H3TurboLoraErrorCode::DTypeMismatch,
                format!(
                    "H3 Turbo tensor {name:?} is {} , expected {expected_dtype}",
                    tensor.dtype
                ),
            ));
        }
        // The rank axis is the only dimension a resized adapter may move, and
        // it may only move DOWN from the source rank. Both other axes stay
        // exact, so a wrong in/out feature is still named as a shape mismatch.
        let shape_ok = match slot {
            0 => {
                tensor.shape.len() == 2
                    && tensor.shape[1] == module_expected.in_features
                    && module_expected.rank.accepts(tensor.shape[0])
            }
            1 => {
                tensor.shape.len() == 2
                    && tensor.shape[0] == module_expected.out_features
                    && module_expected.rank.accepts(tensor.shape[1])
            }
            _ => tensor.shape.is_empty(),
        };
        if !shape_ok {
            let expected_shape = match slot {
                0 => format!(
                    "[{}, {}]",
                    module_expected.rank.describe(),
                    module_expected.in_features
                ),
                1 => format!(
                    "[{}, {}]",
                    module_expected.out_features,
                    module_expected.rank.describe()
                ),
                _ => "[]".to_owned(),
            };
            return Err(failure(
                H3TurboLoraErrorCode::ShapeMismatch,
                format!(
                    "H3 Turbo tensor {name:?} has shape {:?}, expected {expected_shape}",
                    tensor.shape
                ),
            ));
        }
        let slots = seen.entry(module_key).or_default();
        if slots[slot].is_some() {
            return Err(failure(
                H3TurboLoraErrorCode::InvalidHeader,
                format!("H3 Turbo adapter repeats tensor {name:?}"),
            ));
        }
        slots[slot] = Some(reference);
    }
    Ok((expected, seen))
}

type ModuleSlots = [Option<H3TurboLoraTensorRef>; 3];

/// Complete every expected module from the classified tensors, binding each
/// one's alpha and resolved scale. The published contract's tensor-count bound
/// makes a gap unreachable in practice; it is still refused rather than
/// silently producing a partial overlay.
fn assemble_turbo_modules(
    expected: &BTreeMap<String, ExpectedModule>,
    mut seen: BTreeMap<String, ModuleSlots>,
    alphas: &BTreeMap<String, f32>,
    expectation: &H3TurboLoraExpectation,
) -> H3TurboLoraResult<BTreeMap<String, H3TurboLoraModule>> {
    let mut modules = BTreeMap::new();
    for (module_key, module_expected) in expected {
        let slots = seen.remove(module_key).ok_or_else(|| {
            failure(
                H3TurboLoraErrorCode::MissingModule,
                format!("H3 Turbo adapter is missing module {module_key:?}"),
            )
        })?;
        let [Some(lora_a), Some(lora_b), alpha_slot] = slots else {
            return Err(failure(
                H3TurboLoraErrorCode::MissingModule,
                format!("H3 Turbo module {module_key:?} is missing a lora_A or lora_B tensor"),
            ));
        };
        let (rank, alpha, alpha_tensor, scale) = match module_expected.alpha {
            Some(expected_alpha) => {
                let Some(alpha_tensor) = alpha_slot else {
                    return Err(failure(
                        H3TurboLoraErrorCode::MissingModule,
                        format!("H3 Turbo module {module_key:?} is missing its alpha tensor"),
                    ));
                };
                let alpha = *alphas.get(&alpha_tensor.name).ok_or_else(|| {
                    failure(
                        H3TurboLoraErrorCode::AlphaMismatch,
                        format!("H3 Turbo module {module_key:?} has no readable alpha scalar"),
                    )
                })?;
                if alpha != expected_alpha {
                    return Err(failure(
                        H3TurboLoraErrorCode::AlphaMismatch,
                        format!(
                            "H3 Turbo module {module_key:?} declares alpha {alpha}, expected {expected_alpha}"
                        ),
                    ));
                }
                let rank = module_expected.rank.bound();
                let scale = alpha / rank as f32;
                if scale != expectation.scale() {
                    return Err(failure(
                        H3TurboLoraErrorCode::ScaleMismatch,
                        format!(
                            "H3 Turbo module {module_key:?} resolves scale {scale}, expected {}",
                            expectation.scale()
                        ),
                    ));
                }
                (rank, Some(alpha), Some(alpha_tensor), scale)
            }
            None => {
                debug_assert!(
                    alpha_slot.is_none(),
                    "classification refuses an alpha tensor under a resized shape"
                );
                // The module's rank is the file's own, read off `lora_A` and
                // cross-checked against `lora_B`. Classification bounded each
                // axis independently, so a file whose two factors disagree —
                // `[7, in]` against `[out, 9]` — would otherwise assemble into
                // a delta that cannot multiply.
                let rank = lora_a.shape[0];
                let paired = lora_b.shape[1];
                if rank != paired {
                    return Err(failure(
                        H3TurboLoraErrorCode::ShapeMismatch,
                        format!(
                            "H3 Turbo module {module_key:?} has lora_A rank {rank} ({:?} shape {:?}) but lora_B rank {paired} ({:?} shape {:?})",
                            lora_a.name, lora_a.shape, lora_b.name, lora_b.shape
                        ),
                    ));
                }
                if !module_expected.rank.accepts(rank) {
                    return Err(failure(
                        H3TurboLoraErrorCode::ShapeMismatch,
                        format!(
                            "H3 Turbo module {module_key:?} has rank {rank}, expected {}",
                            module_expected.rank.describe()
                        ),
                    ));
                }
                (rank, None, None, expectation.scale())
            }
        };
        modules.insert(
            module_key.clone(),
            H3TurboLoraModule {
                name: module_key.clone(),
                base_weight_name: module_expected.base_weight_name.clone(),
                scope: module_expected.scope,
                kind: module_expected.kind,
                rank,
                in_features: module_expected.in_features,
                out_features: module_expected.out_features,
                alpha,
                scale,
                lora_a,
                lora_b,
                alpha_tensor,
            },
        );
    }
    debug_assert!(seen.is_empty(), "the tensor count bound already balanced");
    Ok(modules)
}

fn remainder_of(name: &str) -> &str {
    name.strip_prefix(H3_TURBO_LORA_KEY_PREFIX).unwrap_or(name)
}

/// A key that names a base-checkpoint tensor is a conflict, not merely an
/// unknown module: it means a merged or quantized checkpoint was supplied
/// where an adapter was required.
fn base_conflict_or_unknown(
    name: &str,
    remainder: &str,
    base_names: &BTreeSet<String>,
) -> H3TurboLoraError {
    if base_names.contains(remainder)
        || BASE_SIDECAR_SUFFIXES
            .iter()
            .any(|suffix| remainder.ends_with(suffix))
    {
        return failure(
            H3TurboLoraErrorCode::BaseContractConflict,
            format!("H3 Turbo adapter carries base checkpoint tensor {name:?}"),
        );
    }
    failure(
        H3TurboLoraErrorCode::UnknownModule,
        format!("H3 Turbo adapter carries unknown tensor {name:?}"),
    )
}

/// SHA-256 over the validated structure, under the domain its shape owns.
///
/// The uniform arm emits exactly the bytes it emitted before rank-dynamic
/// adapters existed, which is what keeps every recorded
/// `adapter_identity_sha256` for the five published rank-128 tiers valid; a
/// literal test pins those digests. The dynamic arm has its own domain and
/// hashes the baked scale plus each module's OWN rank, so two different
/// resizes of the same source can never collide.
fn turbo_structure_identity(
    modules: &BTreeMap<String, H3TurboLoraModule>,
    shape: &H3TurboLoraShape,
) -> String {
    let mut digest = Sha256::new();
    match shape {
        H3TurboLoraShape::Uniform { .. } => digest.update(STRUCTURE_IDENTITY_DOMAIN),
        H3TurboLoraShape::Dynamic { .. } => digest.update(STRUCTURE_IDENTITY_DOMAIN_DYNAMIC),
    }
    digest.update((modules.len() as u64).to_le_bytes());
    if let H3TurboLoraShape::Dynamic { baked_scale, .. } = shape {
        digest.update(baked_scale.to_bits().to_le_bytes());
    }
    for (name, module) in modules {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(module.base_weight_name.as_bytes());
        digest.update([0]);
        digest.update([module.scope.discriminant(), module.kind as u8]);
        digest.update((module.scope.index() as u64).to_le_bytes());
        digest.update((module.rank as u64).to_le_bytes());
        digest.update((module.in_features as u64).to_le_bytes());
        digest.update((module.out_features as u64).to_le_bytes());
        if let Some(alpha) = module.alpha {
            digest.update(alpha.to_bits().to_le_bytes());
        }
    }
    sha256_hex(digest.finalize())
}

/// Identity of the contract *shape*: header plus validated structure. It binds
/// no payload bytes and must never be used as an artifact or cache identity.
fn turbo_contract_identity(
    task: H3TransformerTask,
    header_identity_sha256: &str,
    structure_identity_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(CONTRACT_IDENTITY_DOMAIN);
    digest.update([0, task as u8]);
    digest.update(header_identity_sha256.as_bytes());
    digest.update(structure_identity_sha256.as_bytes());
    sha256_hex(digest.finalize())
}

/// Identity of the authenticated artifact. The verified full-content digest is
/// folded in, so two files sharing a header and a structure but differing in
/// one `lora_B` byte never collide.
fn turbo_adapter_identity(
    tier: H3TurboLoraTier,
    task: H3TransformerTask,
    header_identity_sha256: &str,
    structure_identity_sha256: &str,
    content_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(ADAPTER_IDENTITY_DOMAIN);
    digest.update(tier.stable_id().as_bytes());
    digest.update([0, task as u8]);
    digest.update(header_identity_sha256.as_bytes());
    digest.update(structure_identity_sha256.as_bytes());
    digest.update(content_sha256.as_bytes());
    sha256_hex(digest.finalize())
}

/// Synthetic adapter builders shared by the parser tests and the runtime
/// tests, so both exercise the same file shape.
#[cfg(test)]
pub(super) mod fixtures {
    use std::io::Write;

    use serde_json::{Map, Value};
    use sha2::{Digest, Sha256};

    use super::*;

    /// A valid but tiny transformer geometry, mirroring `comfy_dit`'s runtime
    /// fixture so the synthetic adapter overlays a real base weight set.
    pub(in crate::minimax_h3) fn config() -> H3TransformerConfig {
        H3TransformerConfig {
            hidden_size: 256,
            num_layers: 2,
            token_refiner_num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 128,
            ffn_hidden_size: 256,
            video_latent_channels: 64,
            audio_latent_channels: 256,
            patch_size: [1, 2, 2],
            text_dim: 256,
            timestep_input_dim: 64,
            time_embed_hidden_size: 256,
            time_embed_dim: 128,
            rope_inv_freq_len: 16,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    pub(in crate::minimax_h3) fn expectation() -> H3TurboLoraExpectation {
        H3TurboLoraExpectation {
            config: config(),
            task: H3TransformerTask::T2VaFl2Va,
            shape: H3TurboLoraShape::Uniform {
                training_rank: 4,
                training_alpha: 0.25,
            },
            file_bytes: None,
            content_sha256: None,
            header_len: None,
            header_identity_sha256: None,
            source_repository: H3_TURBO_LORA_REPOSITORY.to_owned(),
            source_revision: H3_TURBO_LORA_SOURCE_REVISION.to_owned(),
        }
    }

    /// A rank-dynamic expectation over the same tiny geometry.
    ///
    /// Its source is a real published Uniform tier of the same task, so the
    /// rank bound (128 plain / 384 fused) and the `resized_from` file name are
    /// the shipped ones rather than invented, and `baked_scale` is that
    /// source's own `alpha / rank`.
    pub(in crate::minimax_h3) fn dynamic_expectation() -> H3TurboLoraExpectation {
        H3TurboLoraExpectation {
            shape: H3TurboLoraShape::Dynamic {
                source: H3TurboLoraTier::Fl2v8StepV10,
                baked_scale: H3TurboLoraTier::Fl2v8StepV10.shape().effective_scale(),
            },
            source_repository: H3_TURBO_LORA_DRBAPH_REPOSITORY.to_owned(),
            source_revision: H3_TURBO_LORA_DRBAPH_SOURCE_REVISION.to_owned(),
            ..expectation()
        }
    }

    /// The per-module ranks a synthetic rank-dynamic adapter carries.
    ///
    /// Deliberately VARIED, and deliberately reproducible without reading the
    /// file: the first module sits exactly on its bound and the second at rank
    /// 1, so both edges of `ExpectedRank::AtMost` are exercised by the ordinary
    /// happy path rather than only by refusal tests, and the rest are spread
    /// across the range by a name hash. A uniform-rank fixture would have let
    /// the whole per-module rank contract pass while reading one number.
    pub(in crate::minimax_h3) fn dynamic_module_ranks(
        expectation: &H3TurboLoraExpectation,
    ) -> BTreeMap<String, usize> {
        let (expected, _) = expected_turbo_modules(expectation).unwrap();
        let source_rank = expectation.training_rank();
        expected
            .iter()
            .enumerate()
            .map(|(index, (module_key, module))| {
                let bound = module.kind.checked_rank(source_rank).unwrap();
                let rank = match index {
                    0 => bound,
                    1 => 1,
                    _ => {
                        let seed = module_key.bytes().fold(0usize, |sum, byte| {
                            sum.wrapping_mul(31).wrapping_add(usize::from(byte))
                        });
                        1 + seed % bound
                    }
                };
                (module_key.clone(), rank)
            })
            .collect()
    }

    /// Deterministic weight element, keyed by tensor name and index so a
    /// consumer can rebuild the exact same matrix without reading the file.
    pub(in crate::minimax_h3) fn weight_value(name: &str, index: usize) -> f32 {
        let seed = name.bytes().fold(0u32, |sum, byte| {
            sum.wrapping_mul(31).wrapping_add(u32::from(byte))
        });
        let raw = seed.wrapping_add((index as u32).wrapping_mul(2_654_435_761)) % 257;
        (raw as f32 / 128.0 - 1.0) * 0.25
    }

    /// Build a complete synthetic adapter for an expectation: the exact module
    /// set, shapes, dtypes, deterministic BF16 weights, and alpha scalars.
    pub(in crate::minimax_h3) fn adapter(expectation: &H3TurboLoraExpectation) -> (Value, Vec<u8>) {
        let (expected, _) = expected_turbo_modules(expectation).unwrap();
        let dynamic_ranks = expectation
            .shape
            .is_dynamic()
            .then(|| dynamic_module_ranks(expectation));
        let mut entries = BTreeMap::<String, (&'static str, Vec<usize>, Option<f32>)>::new();
        for (module_key, module) in &expected {
            let rank = match &dynamic_ranks {
                Some(ranks) => ranks[module_key],
                None => module.rank.bound(),
            };
            entries.insert(
                format!("{module_key}{LORA_A_SUFFIX}"),
                (
                    H3_TURBO_LORA_WEIGHT_DTYPE,
                    vec![rank, module.in_features],
                    None,
                ),
            );
            entries.insert(
                format!("{module_key}{LORA_B_SUFFIX}"),
                (
                    H3_TURBO_LORA_WEIGHT_DTYPE,
                    vec![module.out_features, rank],
                    None,
                ),
            );
            // A resized adapter ships no alpha tensors at all.
            if let Some(alpha) = module.alpha {
                entries.insert(
                    format!("{module_key}{ALPHA_SUFFIX}"),
                    (H3_TURBO_LORA_ALPHA_DTYPE, Vec::new(), Some(alpha)),
                );
            }
        }
        let mut header = Map::new();
        let mut data = Vec::new();
        for (name, (dtype, shape, alpha)) in entries {
            let elements = shape.iter().product::<usize>();
            let start = data.len() as u64;
            match alpha {
                Some(value) => data.extend_from_slice(&value.to_le_bytes()),
                None => {
                    for index in 0..elements {
                        let value = half::bf16::from_f32(weight_value(&name, index));
                        data.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
            }
            header.insert(
                name,
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len() as u64],
                }),
            );
        }
        let mut metadata = Map::new();
        metadata.insert(
            "training_rank".into(),
            Value::String(expectation.training_rank().to_string()),
        );
        match expectation.shape {
            H3TurboLoraShape::Uniform { training_alpha, .. } => {
                metadata.insert(
                    "training_alpha".into(),
                    Value::String(training_alpha.to_string()),
                );
                metadata.insert(
                    "training_scale".into(),
                    Value::String(expectation.scale().to_string()),
                );
            }
            H3TurboLoraShape::Dynamic {
                source,
                baked_scale,
            } => {
                metadata.insert(
                    METADATA_BAKED_SCALE_KEY.into(),
                    Value::String(baked_scale.to_string()),
                );
                metadata.insert(
                    METADATA_RESIZED_FROM_KEY.into(),
                    Value::String(source.file_name().to_owned()),
                );
            }
        }
        metadata.insert(
            "target_format".into(),
            Value::String(H3_TURBO_LORA_TARGET_FORMAT.to_owned()),
        );
        metadata.insert(
            "source_format".into(),
            Value::String(H3_TURBO_LORA_SOURCE_FORMAT.to_owned()),
        );
        header.insert("__metadata__".into(), Value::Object(metadata));
        (Value::Object(header), data)
    }

    pub(in crate::minimax_h3) fn write(
        header: &Value,
        data: &[u8],
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("turbo.safetensors");
        let encoded = serde_json::to_vec(header).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(data).unwrap();
        (directory, path)
    }

    /// A written fixture plus an expectation carrying its real content digest,
    /// which is what `authenticate` requires.
    pub(in crate::minimax_h3) fn pinned() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        H3TurboLoraExpectation,
    ) {
        pinned_for(expectation())
    }

    /// The same, over the rank-dynamic shape.
    pub(in crate::minimax_h3) fn pinned_dynamic() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        H3TurboLoraExpectation,
    ) {
        pinned_for(dynamic_expectation())
    }

    fn pinned_for(
        expectation: H3TurboLoraExpectation,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        H3TurboLoraExpectation,
    ) {
        let (header, data) = adapter(&expectation);
        let (directory, path) = write(&header, &data);
        let mut pinned = expectation;
        pinned.content_sha256 = Some(sha256_hex(Sha256::digest(std::fs::read(&path).unwrap())));
        (directory, path, pinned)
    }
}

/// The validated modules of one published RANK-DYNAMIC tier, read from its
/// checked-in golden header alone.
///
/// Only the header is available in-tree, so this is deliberately restricted to
/// the resized tiers: a rank-uniform tier's modules cannot be assembled
/// without reading the `alpha` scalars out of the two-gigabyte payload.
#[cfg(test)]
pub(in crate::minimax_h3) fn published_dynamic_header_modules(
    tier: H3TurboLoraTier,
) -> BTreeMap<String, H3TurboLoraModule> {
    let expectation = tier.expectation();
    assert!(
        expectation.shape.is_dynamic(),
        "{tier:?} is rank-uniform; its alphas live in the payload, not the header"
    );
    let blob = std::fs::read(tier.header_fixture_path()).unwrap();
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&blob[..8]);
    let parsed = parse_turbo_header(prefix, &blob[8..], tier.file_bytes()).unwrap();
    let (expected, seen) = classify_turbo_tensors(&parsed, &expectation).unwrap();
    assemble_turbo_modules(&expected, seen, &BTreeMap::new(), &expectation).unwrap()
}

#[cfg(test)]
impl H3TurboLoraTier {
    /// Repository-relative path of the checked-in golden: the exact published
    /// eight-byte length prefix followed by the exact published JSON header.
    fn header_fixture_path(self) -> std::path::PathBuf {
        let name = match self {
            Self::Fl2v8StepV10 => "fl2v-8step-v1.0.header",
            Self::Fl2v768p4StepV10 => "fl2v-4step-768p-v1.0.header",
            Self::Ref2v4StepV10 => "ref2v-4step-v0.1.header",
            Self::Fl2v768p4StepV11 => "fl2v-4step-768p-v1.1.header",
            Self::Fl2v768p8StepV10 => "fl2v-8step-768p-v1.0.header",
            Self::Fl2v768p4StepV10Rank21 => "fl2v-4step-768p-v1.0-r21.header",
            Self::Fl2v8StepV10Rank21 => "fl2v-8step-v1.0-r21.header",
            Self::Ref2v4StepV10Rank21 => "ref2v-4step-v0.1-r21.header",
        };
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/minimax_h3/turbo")
            .join(name)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::comfy_dit::H3ComfyNeverCancel;
    use super::*;

    use super::fixtures;

    fn fixture_config() -> H3TransformerConfig {
        fixtures::config()
    }

    fn fixture_expectation() -> H3TurboLoraExpectation {
        fixtures::expectation()
    }

    fn fixture_adapter(expectation: &H3TurboLoraExpectation) -> (Value, Vec<u8>) {
        fixtures::adapter(expectation)
    }

    fn write_adapter(header: &Value, data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        fixtures::write(header, data)
    }

    /// Re-lay the tensor data so offsets stay contiguous after a mutation.
    fn relayout(header: &mut Value) -> Vec<u8> {
        let object = header.as_object_mut().unwrap();
        let names = object
            .keys()
            .filter(|name| *name != "__metadata__")
            .cloned()
            .collect::<Vec<_>>();
        let mut data = Vec::new();
        for name in names {
            let entry = object.get_mut(&name).unwrap();
            let dtype = entry["dtype"].as_str().unwrap().to_owned();
            let shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let elements = shape.iter().product::<usize>();
            let bytes = elements * safetensors_dtype_size(&dtype).unwrap() as usize;
            let start = data.len() as u64;
            data.resize(data.len() + bytes, 0);
            entry["data_offsets"] = serde_json::json!([start, data.len() as u64]);
        }
        data
    }

    fn rename_key(header: &mut Value, from: &str, to: &str) {
        let object = header.as_object_mut().unwrap();
        let value = object.remove(from).unwrap_or_else(|| panic!("{from:?}"));
        assert!(object.insert(to.to_owned(), value).is_none());
    }

    fn inspect_fixture(header: &Value, data: &[u8]) -> H3TurboLoraResult<H3TurboLoraInspection> {
        let (_directory, path) = write_adapter(header, data);
        inspect_h3_turbo_lora_adapter_against(&path, &fixture_expectation())
    }

    fn expect_code(result: H3TurboLoraResult<H3TurboLoraInspection>) -> H3TurboLoraErrorCode {
        result.expect_err("adapter must be rejected").code
    }

    fn read_header_fixture(tier: H3TurboLoraTier) -> (Vec<u8>, [u8; 8], Vec<u8>) {
        let blob = std::fs::read(tier.header_fixture_path()).unwrap();
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&blob[..8]);
        let json = blob[8..].to_vec();
        (blob, prefix, json)
    }

    // ----------------------------------------------------------------- tiers

    /// Every checked-in golden header is claimed by a shipped tier.
    ///
    /// `published_tier_pins_are_recomputed_from_the_checked_in_headers` walks
    /// `H3TurboLoraTier::ALL`, so a `.header` no tier names is bytes that
    /// nothing validates: it can be captured from the wrong revision, or rot
    /// against its published source, and every gate stays green. Asserting the
    /// two sets are EQUAL closes it in both directions — an orphan fixture
    /// fails here, and so does a tier whose fixture was never committed.
    #[test]
    fn every_checked_in_header_fixture_is_claimed_by_a_tier() {
        let claimed = H3TurboLoraTier::ALL
            .into_iter()
            .map(|tier| tier.header_fixture_path())
            .collect::<std::collections::BTreeSet<_>>();
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/minimax_h3/turbo");
        let mut present = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("header") {
                present.insert(path);
            }
        }
        assert_eq!(
            present,
            claimed,
            "every .header under {} must be claimed by exactly one tier",
            directory.display()
        );
    }

    #[test]
    fn published_tier_pins_are_recomputed_from_the_checked_in_headers() {
        for tier in H3TurboLoraTier::ALL {
            let (blob, prefix, json) = read_header_fixture(tier);

            // Header length and identity are DERIVED from the golden bytes,
            // not restated: the fixture is exactly `length prefix || JSON`.
            let derived_len = u64::from_le_bytes(prefix);
            assert_eq!(derived_len, tier.header_len(), "{tier:?}");
            assert_eq!(json.len() as u64, derived_len, "{tier:?}");
            assert_eq!(
                sha256_hex(Sha256::digest(&blob)),
                tier.header_identity_sha256(),
                "{tier:?}"
            );

            // The payload is derived from the header's own offsets, and the
            // pinned file size is the sum of its three parts.
            let parsed = parse_turbo_header(prefix, &json, tier.file_bytes()).unwrap();
            let payload = parsed
                .tensors
                .values()
                .map(|tensor| tensor.data_offsets[1])
                .max()
                .unwrap();
            assert_eq!(tier.file_bytes(), 8 + derived_len + payload, "{tier:?}");

            // The alpha/scale/rank pins are read out of the file's own
            // declared training metadata rather than restated. Both shapes
            // keep `training_rank` at the rank the adapter was TRAINED at, so
            // a resized file declares its SOURCE's 128 rather than any rank in
            // its own header.
            let metadata_number =
                |key: &str| -> f32 { parsed.metadata.get(key).unwrap().parse::<f32>().unwrap() };
            assert_eq!(
                parsed
                    .metadata
                    .get("training_rank")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap(),
                H3_TURBO_LORA_TRAINING_RANK,
                "{tier:?}"
            );
            assert_eq!(
                parsed.metadata.get("target_format").unwrap(),
                H3_TURBO_LORA_TARGET_FORMAT,
                "{tier:?}"
            );
            assert_eq!(
                parsed.metadata.get("source_format").unwrap(),
                H3_TURBO_LORA_SOURCE_FORMAT,
                "{tier:?}"
            );
            match tier.shape() {
                H3TurboLoraShape::Uniform {
                    training_rank,
                    training_alpha,
                } => {
                    assert_eq!(training_rank, H3_TURBO_LORA_TRAINING_RANK, "{tier:?}");
                    assert_eq!(payload, H3_TURBO_LORA_PAYLOAD_BYTES, "{tier:?}");
                    assert_eq!(parsed.tensors.len(), H3_TURBO_LORA_TENSOR_COUNT, "{tier:?}");
                    assert_eq!(
                        metadata_number("training_alpha"),
                        training_alpha,
                        "{tier:?}"
                    );
                    assert_eq!(
                        metadata_number("training_scale"),
                        tier.training_scale(),
                        "{tier:?}"
                    );
                    assert_eq!(tier.training_alpha(), Some(training_alpha), "{tier:?}");
                    assert_eq!(tier.baked_scale(), None, "{tier:?}");
                    assert_eq!(tier.source_tier(), None, "{tier:?}");
                }
                H3TurboLoraShape::Dynamic {
                    source,
                    baked_scale,
                } => {
                    // Two thirds the tensors, because the alphas are gone.
                    assert_eq!(
                        parsed.tensors.len(),
                        H3_TURBO_LORA_DYNAMIC_TENSOR_COUNT,
                        "{tier:?}"
                    );
                    assert!(
                        parsed
                            .tensors
                            .keys()
                            .all(|name| !name.ends_with(ALPHA_SUFFIX)),
                        "{tier:?} carries an alpha tensor"
                    );
                    assert_ne!(payload, H3_TURBO_LORA_PAYLOAD_BYTES, "{tier:?}");
                    // The declared baked scale IS the pin, bit for bit, and
                    // the file names the exact published adapter it was
                    // resized from.
                    assert_eq!(
                        metadata_number(METADATA_BAKED_SCALE_KEY).to_bits(),
                        baked_scale.to_bits(),
                        "{tier:?}"
                    );
                    assert_eq!(
                        parsed.metadata.get(METADATA_RESIZED_FROM_KEY).unwrap(),
                        source.file_name(),
                        "{tier:?}"
                    );
                    // A resize changes weights, never tasks, and never chains.
                    assert_eq!(source.task(), tier.task(), "{tier:?}");
                    assert!(!source.shape().is_dynamic(), "{tier:?}");
                    // The scale is inside `lora_B`; the runtime multiplies by
                    // one, and there is no alpha tensor left to read.
                    assert_eq!(tier.training_scale(), 1.0, "{tier:?}");
                    assert_eq!(tier.training_alpha(), None, "{tier:?}");
                    assert!(
                        !parsed.metadata.contains_key("training_alpha"),
                        "{tier:?} still declares training_alpha"
                    );
                }
            }
            assert_eq!(
                tier.training_scale(),
                tier.expectation().scale(),
                "{tier:?}"
            );
            // Comfy-Org re-hosts every adapter under `loras/`; lightx2v and
            // drbaph publish their own at the repository root. Match the
            // source EXHAUSTIVELY rather than defaulting the unknown case, so
            // a fourth repository has to declare its layout in both places
            // instead of silently agreeing with whichever branch is the
            // fallback.
            let expected_path = match tier.source_repository() {
                H3_TURBO_LORA_REPOSITORY => {
                    format!("{H3_TURBO_LORA_DIRECTORY}/{}", tier.file_name())
                }
                H3_TURBO_LORA_LIGHTX2V_REPOSITORY | H3_TURBO_LORA_DRBAPH_REPOSITORY => {
                    tier.file_name().to_owned()
                }
                other => panic!("{tier:?} publishes from unregistered repository {other}"),
            };
            assert_eq!(tier.repository_path(), expected_path, "{tier:?}");
            assert_eq!(
                tier.expectation().source_repository,
                tier.source_repository(),
                "{tier:?}"
            );
            assert_eq!(
                tier.expectation().source_revision,
                tier.source_revision(),
                "{tier:?}"
            );
        }

        assert_eq!(
            H3_TURBO_LORA_SOURCE_REVISION,
            "dc559027db79c174125df4d827db55cd11178860"
        );
        assert_eq!(H3_TURBO_LORA_REPOSITORY, "Comfy-Org/MiniMax-H3");
        assert_eq!(
            H3_TURBO_LORA_LIGHTX2V_REPOSITORY,
            "lightx2v/Minimax-h3-Turbo"
        );
        assert_eq!(
            H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION,
            "05ef678438e84933c406131b59abbf86919b3aac"
        );
        assert_eq!(
            H3_TURBO_LORA_DRBAPH_REPOSITORY,
            "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
        );
        assert_eq!(
            H3_TURBO_LORA_DRBAPH_SOURCE_REVISION,
            "be8eb3ea3466cbb7def202ffec0d2fdc054256ac"
        );
        assert_eq!(H3_TURBO_LORA_TENSOR_COUNT, H3_TURBO_LORA_MODULE_COUNT * 3);
        assert_eq!(
            H3_TURBO_LORA_DYNAMIC_TENSOR_COUNT,
            H3_TURBO_LORA_MODULE_COUNT * 2
        );
    }

    /// The content digest is the one fact no local artifact can derive — it is
    /// the published blob's own SHA-256 and stays a literal pin.
    #[test]
    fn tier_identities_and_digests_are_distinct() {
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut headers = BTreeSet::new();
        let mut files = BTreeSet::new();
        let mut sources = BTreeSet::new();
        let mut resized_sources = BTreeSet::new();
        for tier in H3TurboLoraTier::ALL {
            assert!(ids.insert(tier.stable_id()));
            assert!(digests.insert(tier.content_sha256()));
            assert!(headers.insert(tier.header_identity_sha256()));
            // A resized tier is a DIFFERENT artifact from its source, and two
            // resized tiers never share one source: sharing one would mean two
            // tags claiming the same approximation of the same adapter.
            if let Some(source) = tier.source_tier() {
                assert_ne!(tier, source);
                assert!(!source.shape().is_dynamic(), "{tier:?}");
                assert!(resized_sources.insert(source), "{tier:?} repeats a source");
            }
            // The basename is the on-disk key mold-core's storage rule
            // flattens every adapter to, so two tiers may never share one.
            assert!(files.insert(tier.file_name()));
            assert!(sources.insert((tier.source_repository(), tier.repository_path())));
            assert_eq!(tier.content_sha256().len(), 64);
            assert!(tier
                .content_sha256()
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
        }
    }

    /// An inspection reports the source the EXPECTATION named, never a global
    /// constant: with two publishing repositories a global would label every
    /// lightx2v adapter as Comfy-Org's.
    #[test]
    fn inspection_provenance_is_the_tiers_own_source() {
        for tier in H3TurboLoraTier::ALL {
            let mut expectation = fixture_expectation();
            expectation.task = tier.task();
            // A synthetic file over the tiny fixture geometry, but a resized
            // tier keeps its OWN source and baked scale so it is exercised
            // through the shape it actually ships. A uniform tier keeps the
            // fixture's small rank, exactly as before.
            if let shape @ H3TurboLoraShape::Dynamic { .. } = tier.shape() {
                expectation.shape = shape;
            }
            expectation.source_repository = tier.source_repository().to_owned();
            expectation.source_revision = tier.source_revision().to_owned();
            let (header, data) = fixture_adapter(&expectation);
            let (_directory, path) = write_adapter(&header, &data);
            let inspection = inspect_h3_turbo_lora_adapter_against(&path, &expectation).unwrap();
            assert_eq!(
                inspection.source_repository,
                tier.source_repository(),
                "{tier:?}"
            );
            assert_eq!(
                inspection.source_repository_revision,
                tier.source_revision(),
                "{tier:?}"
            );
        }
        // Exactly three reviewed sources, and every tier names one of them.
        for tier in H3TurboLoraTier::ALL {
            match tier.source_repository() {
                H3_TURBO_LORA_REPOSITORY => {
                    assert_eq!(tier.source_revision(), H3_TURBO_LORA_SOURCE_REVISION);
                }
                H3_TURBO_LORA_LIGHTX2V_REPOSITORY => {
                    assert_eq!(
                        tier.source_revision(),
                        H3_TURBO_LORA_LIGHTX2V_SOURCE_REVISION
                    );
                }
                H3_TURBO_LORA_DRBAPH_REPOSITORY => {
                    assert_eq!(tier.source_revision(), H3_TURBO_LORA_DRBAPH_SOURCE_REVISION);
                    // Only a resized derivative comes from this repository.
                    assert!(tier.shape().is_dynamic(), "{tier:?}");
                }
                other => panic!("{tier:?} names unreviewed repository {other}"),
            }
        }
    }

    #[test]
    fn published_expectation_matches_the_shipped_checkpoint_geometry() {
        for tier in H3TurboLoraTier::ALL {
            let expectation = tier.expectation();
            expectation.validate().unwrap();
            assert_eq!(
                expectation.module_count().unwrap(),
                H3_TURBO_LORA_MODULE_COUNT
            );
            assert_eq!(
                expectation.tensor_count().unwrap(),
                match tier.shape() {
                    H3TurboLoraShape::Uniform { .. } => H3_TURBO_LORA_TENSOR_COUNT,
                    H3TurboLoraShape::Dynamic { .. } => H3_TURBO_LORA_DYNAMIC_TENSOR_COUNT,
                },
                "{tier:?}"
            );
            assert_eq!(expectation.training_rank(), H3_TURBO_LORA_TRAINING_RANK);
            assert_eq!(expectation.task, tier.task());
            assert_eq!(
                expectation.content_sha256.as_deref(),
                Some(tier.content_sha256())
            );
        }
    }

    // ------------------------------------------------ shared target authority

    #[test]
    fn module_kinds_mirror_the_shared_block_linear_authority() {
        let shared = H3_BLOCK_LINEAR_WEIGHT_SUFFIXES
            .iter()
            .map(|suffix| suffix.strip_suffix(".weight").unwrap())
            .collect::<Vec<_>>();
        assert_eq!(shared, MODULE_SUFFIXES.to_vec());
        for suffix in shared {
            assert!(
                H3TurboLoraModuleKind::from_suffix(suffix).is_some(),
                "{suffix:?} has no Turbo module kind"
            );
        }
        assert_eq!(
            H3TurboLoraModuleKind::ALL.len(),
            H3_BLOCK_LINEAR_WEIGHT_SUFFIXES.len()
        );
    }

    #[test]
    fn the_turbo_module_set_is_exactly_the_shared_target_authority() {
        for config in [H3TransformerConfig::default(), fixture_config()] {
            let expectation = H3TurboLoraExpectation {
                config: config.clone(),
                ..fixture_expectation()
            };
            let specs = expected_h3_weight_specs(
                &config,
                H3AdaLnMode::Full,
                H3PrecisionProfile::OfficialMixedBf16F32,
            )
            .unwrap();
            let targets = h3_block_linear_targets(
                &config,
                &specs,
                &[
                    H3BlockLinearScope::MainBlock,
                    H3BlockLinearScope::TokenRefinerBlock,
                ],
            )
            .unwrap();

            let (derived, _) = expected_turbo_modules(&expectation).unwrap();
            // Full set equality, not a sample: every target becomes exactly one
            // module and no module exists without a target.
            let from_authority = targets
                .iter()
                .map(|target| format!("{H3_TURBO_LORA_KEY_PREFIX}{}", target.module))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                derived.keys().cloned().collect::<BTreeSet<_>>(),
                from_authority
            );
            assert_eq!(derived.len(), targets.len());

            for target in &targets {
                let module = derived
                    .get(&format!("{H3_TURBO_LORA_KEY_PREFIX}{}", target.module))
                    .unwrap();
                assert_eq!(module.base_weight_name, target.weight_name);
                assert_eq!(module.in_features, target.in_features);
                assert_eq!(module.out_features, target.out_features);
                assert_eq!(module.kind.suffix(), target.suffix);
                assert_eq!(module.scope.index(), target.index);
            }
        }
    }

    #[test]
    fn the_published_geometry_matches_every_shape_read_from_the_real_headers() {
        let expectation = H3TurboLoraTier::Fl2v8StepV10.expectation();
        let (derived, base_names) = expected_turbo_modules(&expectation).unwrap();
        assert_eq!(derived.len(), H3_TURBO_LORA_MODULE_COUNT);

        // Exactly the four shape signatures read from the published headers,
        // asserted against ALL 208 modules rather than a sample.
        let published: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::from([
            ("attn.qkv_proj", (384_usize, 5_376_usize, 21_504_usize)),
            ("attn.out_proj", (128, 7_168, 5_376)),
            ("mlp.fc1", (128, 5_376, 28_672)),
            ("mlp.fc2", (128, 14_336, 5_376)),
        ]);
        let mut main_blocks = BTreeSet::new();
        let mut refiner_blocks = BTreeSet::new();
        for (key, module) in &derived {
            let (rank, in_features, out_features) = published[module.kind.suffix()];
            assert_eq!(module.rank, ExpectedRank::Exact(rank), "{key}");
            assert_eq!(module.in_features, in_features, "{key}");
            assert_eq!(module.out_features, out_features, "{key}");
            assert_eq!(module.alpha, Some(module.kind.alpha(8.0)), "{key}");
            assert!(base_names.contains(&module.base_weight_name), "{key}");
            match module.scope {
                H3TurboLoraModuleScope::MainBlock(index) => {
                    main_blocks.insert(index);
                }
                H3TurboLoraModuleScope::TokenRefinerBlock(index) => {
                    refiner_blocks.insert(index);
                }
            }
        }
        assert_eq!(main_blocks, (0..50).collect::<BTreeSet<_>>());
        assert_eq!(refiner_blocks, (0..2).collect::<BTreeSet<_>>());
    }

    #[test]
    fn every_published_header_satisfies_the_derived_contract() {
        for tier in H3TurboLoraTier::ALL {
            let (_, prefix, json) = read_header_fixture(tier);
            let parsed = parse_turbo_header(prefix, &json, tier.file_bytes()).unwrap();
            let expectation = tier.expectation();
            expectation.validate().unwrap();
            validate_turbo_metadata(&parsed.metadata, &expectation).unwrap();

            // Every one of the real tensors binds to a derived module slot
            // with the derived dtype and shape — 624 for a uniform tier, 416
            // for a resized one whose alphas were baked away.
            let (expected, seen) = classify_turbo_tensors(&parsed, &expectation).unwrap();
            assert_eq!(expected.len(), H3_TURBO_LORA_MODULE_COUNT, "{tier:?}");
            assert_eq!(seen.len(), H3_TURBO_LORA_MODULE_COUNT, "{tier:?}");
            let slots_per_module = expectation.shape.tensors_per_module();
            for (key, slots) in &seen {
                assert_eq!(
                    slots.iter().filter(|slot| slot.is_some()).count(),
                    slots_per_module,
                    "{tier:?} {key} is incomplete"
                );
                assert!(slots[0].is_some() && slots[1].is_some(), "{tier:?} {key}");
            }
            assert_eq!(
                seen.keys().cloned().collect::<BTreeSet<_>>(),
                expected.keys().cloned().collect::<BTreeSet<_>>(),
                "{tier:?}"
            );

            // Every module of a resized tier carries a rank of its own, inside
            // the source bound and equal on both factors. Asserting the RANGE
            // of the observed per-kind ranks is what would catch a file that
            // secretly kept one uniform rank.
            if expectation.shape.is_dynamic() {
                let modules =
                    assemble_turbo_modules(&expected, seen, &BTreeMap::new(), &expectation)
                        .unwrap();
                let mut per_kind: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
                for module in modules.values() {
                    assert_eq!(module.alpha, None, "{tier:?} {}", module.name);
                    assert_eq!(module.alpha_tensor, None, "{tier:?} {}", module.name);
                    assert_eq!(module.scale, 1.0, "{tier:?} {}", module.name);
                    assert_eq!(module.rank, module.lora_a.shape[0], "{tier:?}");
                    assert_eq!(module.rank, module.lora_b.shape[1], "{tier:?}");
                    assert_eq!(module.lora_a.shape[1], module.in_features, "{tier:?}");
                    assert_eq!(module.lora_b.shape[0], module.out_features, "{tier:?}");
                    let bound = module
                        .kind
                        .checked_rank(H3_TURBO_LORA_TRAINING_RANK)
                        .unwrap();
                    assert!(
                        module.rank >= 1 && module.rank <= bound,
                        "{tier:?} {} rank {} outside 1..={bound}",
                        module.name,
                        module.rank
                    );
                    let entry = per_kind
                        .entry(module.kind.suffix())
                        .or_insert((usize::MAX, 0));
                    entry.0 = entry.0.min(module.rank);
                    entry.1 = entry.1.max(module.rank);
                }
                assert_eq!(per_kind.len(), H3TurboLoraModuleKind::ALL.len(), "{tier:?}");
                for (suffix, (low, high)) in per_kind {
                    assert!(low < high, "{tier:?} {suffix} carries one uniform rank");
                }
            }
        }
    }

    #[test]
    fn fused_qkv_triples_rank_and_alpha_so_the_scale_is_uniform() {
        for kind in H3TurboLoraModuleKind::ALL {
            let rank = kind.checked_rank(H3_TURBO_LORA_TRAINING_RANK).unwrap();
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
        assert!(H3TurboLoraModuleKind::AttnQkvProj
            .checked_rank(usize::MAX)
            .is_err());
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

    // ---------------------------------------------------- expectation bounds

    #[test]
    fn an_out_of_bounds_expectation_is_refused_before_any_file_is_touched() {
        let mut huge = fixture_expectation();
        huge.config.num_layers = MAX_TURBO_BLOCKS;
        huge.config.token_refiner_num_layers = 1;
        assert_eq!(
            huge.validate().unwrap_err().code,
            H3TurboLoraErrorCode::ConfigMismatch
        );

        let mut ranked = fixture_expectation();
        ranked.shape = H3TurboLoraShape::Uniform {
            training_rank: MAX_TURBO_TRAINING_RANK + 1,
            training_alpha: 0.25,
        };
        assert_eq!(
            ranked.validate().unwrap_err().code,
            H3TurboLoraErrorCode::ConfigMismatch
        );

        let mut zero = fixture_expectation();
        zero.shape = H3TurboLoraShape::Uniform {
            training_rank: 0,
            training_alpha: 0.25,
        };
        assert_eq!(
            zero.validate().unwrap_err().code,
            H3TurboLoraErrorCode::ConfigMismatch
        );

        for alpha in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut bad = fixture_expectation();
            bad.shape = H3TurboLoraShape::Uniform {
                training_rank: 4,
                training_alpha: alpha,
            };
            assert_eq!(
                bad.validate().unwrap_err().code,
                H3TurboLoraErrorCode::ConfigMismatch,
                "{alpha}"
            );
        }

        // The bound is enforced at the entry point, not only in validate().
        let (header, data) = fixture_adapter(&fixture_expectation());
        let (_directory, path) = write_adapter(&header, &data);
        let mut unranked = fixture_expectation();
        unranked.shape = H3TurboLoraShape::Uniform {
            training_rank: 0,
            training_alpha: 0.25,
        };
        assert_eq!(
            inspect_h3_turbo_lora_adapter_against(&path, &unranked)
                .unwrap_err()
                .code,
            H3TurboLoraErrorCode::ConfigMismatch
        );
    }

    // -------------------------------------------------------- header parsing

    #[test]
    fn synthetic_adapter_validates_every_module_and_reads_its_alphas() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let inspection = inspect_fixture(&header, &data).unwrap();

        assert_eq!(
            inspection.module_count(),
            expectation.module_count().unwrap()
        );
        assert_eq!(inspection.tensor_count, expectation.tensor_count().unwrap());
        assert_eq!(inspection.scale, 0.0625);
        assert_eq!(inspection.payload_bytes, data.len() as u64);
        assert_eq!(inspection.header_identity_sha256.len(), 64);
        assert_eq!(inspection.structure_identity_sha256.len(), 64);
        assert_eq!(inspection.contract_identity_sha256.len(), 64);

        let qkv = inspection
            .modules
            .get("diffusion_model.blocks.0.attn.qkv_proj")
            .unwrap();
        assert_eq!(qkv.kind, H3TurboLoraModuleKind::AttnQkvProj);
        assert_eq!(qkv.scope, H3TurboLoraModuleScope::MainBlock(0));
        assert_eq!(qkv.base_weight_name, "blocks.0.attn.qkv_proj.weight");
        // Fused Q/K/V: rank and alpha are both tripled, so the scale holds.
        assert_eq!(qkv.rank, 12);
        assert_eq!(qkv.alpha, Some(0.75));
        assert_eq!(qkv.scale, inspection.scale);
        assert_eq!(qkv.lora_a.shape, vec![12, 256]);
        assert_eq!(qkv.lora_b.shape, vec![768, 12]);

        let fc1 = inspection
            .modules
            .get("diffusion_model.token_refiner.blocks.0.mlp.fc1")
            .unwrap();
        assert_eq!(fc1.scope, H3TurboLoraModuleScope::TokenRefinerBlock(0));
        assert_eq!(fc1.rank, 4);
        assert_eq!(fc1.alpha, Some(0.25));
        assert_eq!(fc1.lora_a.shape, vec![4, 256]);
        assert_eq!(fc1.lora_b.shape, vec![512, 4]);

        assert_eq!(inspection.main_block_modules(0).len(), 4);
        assert_eq!(inspection.token_refiner_modules(0).len(), 4);
        let block_bytes = inspection.scope_delta_bytes(H3TurboLoraModuleScope::MainBlock(0));
        assert!(block_bytes > 0);
        assert_eq!(
            block_bytes,
            inspection.scope_delta_bytes(H3TurboLoraModuleScope::MainBlock(1))
        );
    }

    // --------------------------------------------------- rank-dynamic shape

    fn dynamic_expectation() -> H3TurboLoraExpectation {
        fixtures::dynamic_expectation()
    }

    fn inspect_against(
        header: &Value,
        data: &[u8],
        expectation: &H3TurboLoraExpectation,
    ) -> H3TurboLoraResult<H3TurboLoraInspection> {
        let (_directory, path) = write_adapter(header, data);
        inspect_h3_turbo_lora_adapter_against(&path, expectation)
    }

    /// The published `baked_scale` of drbaph's v1.1 exports, verbatim. It is a
    /// SENTENCE, not a number, which is the whole reason a resized adapter's
    /// scale has to be parsed and compared rather than trusted as present.
    const NON_NUMERIC_BAKED_SCALE: &str =
        "source alpha/rank scale baked into lora_B (effective scale 1.0)";

    #[test]
    fn a_dynamic_synthetic_adapter_validates_per_module_ranks_without_alphas() {
        let expectation = dynamic_expectation();
        let ranks = fixtures::dynamic_module_ranks(&expectation);
        let (header, data) = fixture_adapter(&expectation);
        let inspection = inspect_against(&header, &data, &expectation).unwrap();

        // Two tensors per module, not three.
        assert_eq!(
            inspection.tensor_count,
            inspection.module_count() * 2,
            "a resized adapter ships no alpha tensors"
        );
        assert_eq!(inspection.tensor_count, expectation.tensor_count().unwrap());
        assert_eq!(inspection.shape, expectation.shape);
        // `training_rank` is the SOURCE's, and the runtime scale is one.
        assert_eq!(inspection.training_rank, H3_TURBO_LORA_TRAINING_RANK);
        assert_eq!(inspection.training_alpha, None);
        assert_eq!(inspection.scale, 1.0);
        assert_eq!(inspection.payload_bytes, data.len() as u64);

        let mut observed = BTreeSet::new();
        for (key, module) in &inspection.modules {
            let bound = module
                .kind
                .checked_rank(H3_TURBO_LORA_TRAINING_RANK)
                .unwrap();
            assert_eq!(module.rank, ranks[key], "{key}");
            assert_eq!(module.rank, module.lora_a.shape[0], "{key}");
            assert_eq!(module.rank, module.lora_b.shape[1], "{key}");
            assert_eq!(module.lora_a.shape[1], module.in_features, "{key}");
            assert_eq!(module.lora_b.shape[0], module.out_features, "{key}");
            assert!(module.rank >= 1 && module.rank <= bound, "{key}");
            assert_eq!(module.alpha, None, "{key}");
            assert_eq!(module.alpha_tensor, None, "{key}");
            assert_eq!(module.scale, 1.0, "{key}");
            observed.insert((module.rank, bound));
        }
        // Both edges of the AtMost bound are exercised by the happy path: one
        // module sits exactly on its bound and one at rank 1.
        assert!(
            observed.iter().any(|(rank, bound)| rank == bound),
            "no module sits on its rank bound"
        );
        assert!(
            observed.iter().any(|(rank, _)| *rank == 1),
            "no module sits at rank 1"
        );
        assert!(observed.len() > 2, "the fixture ranks are not varied");
    }

    #[test]
    fn a_dynamic_module_whose_a_and_b_ranks_disagree_is_a_shape_mismatch() {
        let expectation = dynamic_expectation();
        let (mut header, _) = fixture_adapter(&expectation);
        // Both values are INSIDE the bound, so each tensor passes its own
        // shape check; only the pairing is wrong.
        header["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]["shape"] =
            serde_json::json!([7, 256]);
        header["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]["shape"] =
            serde_json::json!([512, 9]);
        let data = relayout(&mut header);
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::ShapeMismatch);
        assert!(error.message.contains("lora_A rank 7"), "{}", error.message);
        assert!(error.message.contains("lora_B rank 9"), "{}", error.message);
    }

    #[test]
    fn a_dynamic_module_above_the_source_rank_bound_is_refused() {
        // 128 for a plain linear, 384 for the fused Q/K/V module: a resized
        // adapter may only go DOWN from the rank it was resized from.
        for (module, over_bound, in_features, out_features) in [
            ("blocks.0.mlp.fc1", 129usize, 256usize, 512usize),
            ("blocks.0.attn.qkv_proj", 385, 256, 768),
        ] {
            let expectation = dynamic_expectation();
            let (mut header, _) = fixture_adapter(&expectation);
            header[&format!("diffusion_model.{module}.lora_A.weight")]["shape"] =
                serde_json::json!([over_bound, in_features]);
            header[&format!("diffusion_model.{module}.lora_B.weight")]["shape"] =
                serde_json::json!([out_features, over_bound]);
            let data = relayout(&mut header);
            let error = inspect_against(&header, &data, &expectation).unwrap_err();
            assert_eq!(error.code, H3TurboLoraErrorCode::ShapeMismatch, "{module}");
            assert!(
                error.message.contains(&format!("1..={}", over_bound - 1)),
                "{module}: {}",
                error.message
            );
        }
    }

    #[test]
    fn a_dynamic_adapter_carrying_an_alpha_tensor_is_refused() {
        let expectation = dynamic_expectation();
        let (mut header, _) = fixture_adapter(&expectation);
        // Renamed rather than added, so the tensor COUNT still matches and the
        // refusal is the alpha rule itself rather than the count check.
        rename_key(
            &mut header,
            "diffusion_model.blocks.0.mlp.fc2.lora_A.weight",
            "diffusion_model.blocks.0.mlp.fc2.alpha",
        );
        let data = relayout(&mut header);
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::AlphaMismatch);
        assert!(
            error.message.contains("no alpha tensors"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_non_numeric_baked_scale_is_invalid_metadata() {
        let expectation = dynamic_expectation();
        let (mut header, data) = fixture_adapter(&expectation);
        header["__metadata__"][METADATA_BAKED_SCALE_KEY] =
            serde_json::json!(NON_NUMERIC_BAKED_SCALE);
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidMetadata);
        assert!(
            error.message.contains(NON_NUMERIC_BAKED_SCALE),
            "{}",
            error.message
        );

        // And an absent one is refused too: the scale is required, not
        // advisory, because it is the only thing that says whether the factors
        // already carry `alpha / rank`.
        let (mut header, data) = fixture_adapter(&expectation);
        header["__metadata__"]
            .as_object_mut()
            .unwrap()
            .remove(METADATA_BAKED_SCALE_KEY)
            .unwrap();
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidMetadata);
        assert!(
            error.message.contains(METADATA_BAKED_SCALE_KEY),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_baked_scale_that_disagrees_with_the_tier_is_refused() {
        let expectation = dynamic_expectation();
        let (mut header, data) = fixture_adapter(&expectation);
        header["__metadata__"][METADATA_BAKED_SCALE_KEY] = serde_json::json!("0.125");
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidMetadata);

        // `resized_from` is welded to the source tier's own file name.
        let (mut header, data) = fixture_adapter(&expectation);
        header["__metadata__"][METADATA_RESIZED_FROM_KEY] =
            serde_json::json!(H3TurboLoraTier::Ref2v4StepV10.file_name());
        let error = inspect_against(&header, &data, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidMetadata);
        assert!(
            error.message.contains(METADATA_RESIZED_FROM_KEY),
            "{}",
            error.message
        );
    }

    /// A full-rank file offered for a resized tier is named by its COUNT.
    ///
    /// The tensor-count check runs before any shape check on purpose: 624
    /// tensors against 416 is one clear sentence, where shape-first would
    /// produce "module X has rank 128, expected 1..=128" — a refusal that
    /// reads like a rank problem and is not one.
    #[test]
    fn a_full_rank_file_offered_for_a_rank21_tier_is_a_tensor_count_mismatch() {
        let full_rank = H3TurboLoraTier::Fl2v768p4StepV10;
        let resized = H3TurboLoraTier::Fl2v768p4StepV10Rank21;
        assert_eq!(resized.source_tier(), Some(full_rank));
        let (_, prefix, json) = read_header_fixture(full_rank);
        let parsed = parse_turbo_header(prefix, &json, full_rank.file_bytes()).unwrap();
        let error = classify_turbo_tensors(&parsed, &resized.expectation()).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::TensorCountMismatch);
        assert!(
            error.message.contains("624") && error.message.contains("416"),
            "{}",
            error.message
        );
    }

    /// The runtime scale of a resized adapter is `1.0` BY CONSTRUCTION, not by
    /// arithmetic on the pin.
    ///
    /// `baked_scale` records what the exporter multiplied into `lora_B`; it is
    /// a fact the file is checked against and never a factor handed to a
    /// forward pass, so no value of it — however wrong, however large — can
    /// reach `H3TurboLoraDelta`. That is what makes "applied twice"
    /// unreachable rather than merely unlikely.
    #[test]
    fn a_dynamic_shape_always_resolves_to_a_unit_runtime_scale() {
        for baked_scale in [1.0_f32, 0.0625, 16.0, 1e-9] {
            let shape = H3TurboLoraShape::Dynamic {
                source: H3TurboLoraTier::Fl2v8StepV10,
                baked_scale,
            };
            assert_eq!(shape.effective_scale(), 1.0, "{baked_scale}");
            assert_eq!(shape.training_alpha(), None, "{baked_scale}");
            assert_eq!(shape.tensors_per_module(), 2, "{baked_scale}");
            assert_eq!(shape.source_training_rank(), H3_TURBO_LORA_TRAINING_RANK);
        }
        for tier in H3TurboLoraTier::ALL {
            if !tier.shape().is_dynamic() {
                continue;
            }
            assert_eq!(tier.expectation().scale(), 1.0, "{tier:?}");
            assert_eq!(tier.training_scale(), 1.0, "{tier:?}");
            // And the source's own scale is NOT what runs.
            let source = tier.source_tier().unwrap();
            assert_eq!(
                tier.baked_scale(),
                Some(source.training_scale()),
                "{tier:?}"
            );
        }
    }

    #[test]
    fn a_dynamic_tier_must_resize_from_a_uniform_tier_of_its_own_task() {
        // Every shipped resized tier already satisfies this.
        for tier in H3TurboLoraTier::ALL {
            tier.expectation().validate().unwrap();
        }

        // Across tasks: a ref2v source can never back an FL2VA resize.
        let mut crossed = dynamic_expectation();
        crossed.shape = H3TurboLoraShape::Dynamic {
            source: H3TurboLoraTier::Ref2v4StepV10,
            baked_scale: 0.0625,
        };
        assert_eq!(
            crossed.validate().unwrap_err().code,
            H3TurboLoraErrorCode::TaskAuthorityMismatch
        );

        // Chained: a resize of a resize has no pinned source rank at all.
        let mut chained = dynamic_expectation();
        chained.shape = H3TurboLoraShape::Dynamic {
            source: H3TurboLoraTier::Fl2v8StepV10Rank21,
            baked_scale: 0.0625,
        };
        assert_eq!(
            chained.validate().unwrap_err().code,
            H3TurboLoraErrorCode::ConfigMismatch
        );

        // And a non-positive baked scale is refused before any file is read.
        for baked_scale in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut bad = dynamic_expectation();
            bad.shape = H3TurboLoraShape::Dynamic {
                source: H3TurboLoraTier::Fl2v8StepV10,
                baked_scale,
            };
            assert_eq!(
                bad.validate().unwrap_err().code,
                H3TurboLoraErrorCode::ConfigMismatch,
                "{baked_scale}"
            );
        }
    }

    /// The dynamic structure digest lives on its own domain and binds the two
    /// facts a uniform digest cannot express: the baked scale and each
    /// module's own rank.
    #[test]
    fn the_dynamic_structure_identity_binds_the_ranks_and_the_baked_scale() {
        let expectation = dynamic_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let inspection = inspect_against(&header, &data, &expectation).unwrap();
        let modules = inspection.modules.clone();

        // Domain separation: the SAME modules under the uniform domain hash
        // differently, so a resized structure can never collide with a
        // rank-uniform one.
        assert_ne!(
            turbo_structure_identity(&modules, &expectation.shape),
            turbo_structure_identity(
                &modules,
                &H3TurboLoraShape::Uniform {
                    training_rank: 4,
                    training_alpha: 0.25,
                }
            )
        );
        assert_eq!(
            inspection.structure_identity_sha256,
            turbo_structure_identity(&modules, &expectation.shape)
        );

        // A different baked scale is a different overlay at identical shapes.
        let louder = H3TurboLoraShape::Dynamic {
            source: H3TurboLoraTier::Fl2v8StepV10,
            baked_scale: 0.125,
        };
        assert_ne!(
            turbo_structure_identity(&modules, &expectation.shape),
            turbo_structure_identity(&modules, &louder)
        );

        // And so is one module resized to a different rank.
        let mut narrower = modules.clone();
        let key = "diffusion_model.blocks.0.mlp.fc1".to_owned();
        narrower.get_mut(&key).unwrap().rank += 1;
        assert_ne!(
            turbo_structure_identity(&modules, &expectation.shape),
            turbo_structure_identity(&narrower, &expectation.shape)
        );
    }

    #[test]
    fn a_missing_tensor_is_a_tensor_count_mismatch() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        header
            .as_object_mut()
            .unwrap()
            .remove("diffusion_model.blocks.1.mlp.fc2.alpha")
            .unwrap();
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::TensorCountMismatch
        );
    }

    #[test]
    fn an_unknown_module_is_rejected() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        rename_key(
            &mut header,
            "diffusion_model.blocks.1.mlp.fc2.lora_A.weight",
            "diffusion_model.blocks.1.attn.q_norm.lora_A.weight",
        );
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::UnknownModule
        );
    }

    #[test]
    fn a_module_outside_the_block_range_is_rejected() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        for suffix in [LORA_A_SUFFIX, LORA_B_SUFFIX, ALPHA_SUFFIX] {
            rename_key(
                &mut header,
                &format!("diffusion_model.blocks.1.mlp.fc2{suffix}"),
                &format!("diffusion_model.blocks.9.mlp.fc2{suffix}"),
            );
        }
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::UnknownModule
        );
    }

    #[test]
    fn a_shape_mismatch_is_rejected() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        header["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]["shape"] =
            serde_json::json!([8, 256]);
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::ShapeMismatch
        );
    }

    #[test]
    fn a_dtype_mismatch_is_rejected() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        header["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]["dtype"] =
            serde_json::json!("F32");
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::DTypeMismatch
        );
    }

    #[test]
    fn an_alpha_that_disagrees_with_the_tier_is_rejected() {
        let expectation = fixture_expectation();
        let (header, mut data) = fixture_adapter(&expectation);
        let offsets = &header["diffusion_model.blocks.0.mlp.fc2.alpha"]["data_offsets"];
        let start = offsets[0].as_u64().unwrap() as usize;
        data[start..start + 4].copy_from_slice(&0.5_f32.to_le_bytes());
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::AlphaMismatch
        );
    }

    #[test]
    fn a_non_positive_alpha_is_rejected() {
        let expectation = fixture_expectation();
        let (header, mut data) = fixture_adapter(&expectation);
        let offsets = &header["diffusion_model.blocks.0.mlp.fc2.alpha"]["data_offsets"];
        let start = offsets[0].as_u64().unwrap() as usize;
        data[start..start + 4].copy_from_slice(&0.0_f32.to_le_bytes());
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::AlphaMismatch
        );
    }

    #[test]
    fn a_base_checkpoint_weight_is_a_contract_conflict() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        rename_key(
            &mut header,
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
            "diffusion_model.blocks.0.attn.qkv_proj.weight",
        );
        let data = relayout(&mut header);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::BaseContractConflict
        );
    }

    #[test]
    fn a_quantization_sidecar_is_a_contract_conflict() {
        for sidecar in BASE_SIDECAR_SUFFIXES {
            let (mut header, _) = fixture_adapter(&fixture_expectation());
            rename_key(
                &mut header,
                "diffusion_model.blocks.0.attn.out_proj.lora_A.weight",
                &format!("diffusion_model.blocks.0.attn.out_proj{sidecar}"),
            );
            let data = relayout(&mut header);
            assert_eq!(
                expect_code(inspect_fixture(&header, &data)),
                H3TurboLoraErrorCode::BaseContractConflict,
                "{sidecar}"
            );
        }
    }

    #[test]
    fn an_oversized_header_is_refused_before_it_is_read() {
        let (header, data) = fixture_adapter(&fixture_expectation());
        let (_directory, path) = write_adapter(&header, &data);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[..8].copy_from_slice(&(MAX_HEADER_BYTES + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let error =
            inspect_h3_turbo_lora_adapter_against(&path, &fixture_expectation()).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);
        assert!(error.message.contains("header length"), "{}", error.message);
    }

    #[test]
    fn a_gap_between_tensor_ranges_is_refused() {
        let (mut header, data) = fixture_adapter(&fixture_expectation());
        // Push everything after the first tensor forward by two bytes and grow
        // the payload to match, leaving an unclaimed two-byte hole.
        let names = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| *name != "__metadata__")
            .cloned()
            .collect::<Vec<_>>();
        let first_end = header[&names[0]]["data_offsets"][1].as_u64().unwrap();
        for name in &names {
            let start = header[name]["data_offsets"][0].as_u64().unwrap();
            let end = header[name]["data_offsets"][1].as_u64().unwrap();
            if start >= first_end {
                header[name]["data_offsets"] = serde_json::json!([start + 2, end + 2]);
            }
        }
        let mut holed = data.clone();
        holed.extend_from_slice(&[0, 0]);
        let error = inspect_fixture(&header, &holed).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);
        assert!(
            error.message.contains("non-contiguous"),
            "{}",
            error.message
        );
    }

    #[test]
    fn overlapping_tensor_ranges_are_refused() {
        let (mut header, data) = fixture_adapter(&fixture_expectation());
        let names = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| *name != "__metadata__")
            .cloned()
            .collect::<Vec<_>>();
        // Slide every range from one alpha scalar onward back by four bytes so
        // it overlaps its predecessor, shrinking the payload to match.
        let victim = names
            .iter()
            .find(|name| {
                name.ends_with(ALPHA_SUFFIX)
                    && header[*name]["data_offsets"][0].as_u64().unwrap() >= 4
            })
            .cloned()
            .unwrap();
        let victim_start = header[&victim]["data_offsets"][0].as_u64().unwrap();
        for name in &names {
            let start = header[name]["data_offsets"][0].as_u64().unwrap();
            let end = header[name]["data_offsets"][1].as_u64().unwrap();
            if start >= victim_start {
                header[name]["data_offsets"] = serde_json::json!([start - 4, end - 4]);
            }
        }
        let shortened = data[..data.len() - 4].to_vec();
        let error = inspect_fixture(&header, &shortened).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);
        assert!(
            error.message.contains("non-contiguous"),
            "{}",
            error.message
        );
    }

    #[test]
    fn trailing_tensor_data_is_refused() {
        let (header, mut data) = fixture_adapter(&fixture_expectation());
        data.push(0);
        let error = inspect_fixture(&header, &data).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);
        assert!(
            error.message.contains("unclaimed trailing"),
            "{}",
            error.message
        );
    }

    #[test]
    fn declared_metadata_that_contradicts_the_expectation_is_rejected() {
        let (mut header, data) = fixture_adapter(&fixture_expectation());
        header["__metadata__"]["training_scale"] = serde_json::json!("1.0");
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::InvalidMetadata
        );

        let (mut header, data) = fixture_adapter(&fixture_expectation());
        header["__metadata__"]["target_format"] = serde_json::json!("Diffusers PEFT LoRA");
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::InvalidMetadata
        );
    }

    #[test]
    fn published_pins_reject_a_file_that_is_not_the_reviewed_artifact() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let (_directory, path) = write_adapter(&header, &data);
        let error =
            inspect_h3_turbo_lora_adapter(&path, H3TurboLoraTier::Fl2v8StepV10).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::SourceSizeMismatch);

        let mut sized = expectation.clone();
        sized.header_len = Some(1);
        let error = inspect_h3_turbo_lora_adapter_against(&path, &sized).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);

        let mut identified = expectation.clone();
        identified.header_identity_sha256 = Some("0".repeat(64));
        let error = inspect_h3_turbo_lora_adapter_against(&path, &identified).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::HeaderIdentityMismatch);
    }

    #[test]
    fn an_incomplete_module_never_produces_a_partial_overlay() {
        let expectation = fixture_expectation();
        let (expected, _) = expected_turbo_modules(&expectation).unwrap();
        let (header, data) = fixture_adapter(&expectation);
        let (_directory, path) = write_adapter(&header, &data);
        let inspection = inspect_h3_turbo_lora_adapter_against(&path, &expectation).unwrap();

        let mut seen = BTreeMap::new();
        let mut alphas = BTreeMap::new();
        for (key, module) in &inspection.modules {
            let slots = if key.ends_with("blocks.1.mlp.fc2") {
                [
                    Some(module.lora_a.clone()),
                    Some(module.lora_b.clone()),
                    None,
                ]
            } else {
                [
                    Some(module.lora_a.clone()),
                    Some(module.lora_b.clone()),
                    module.alpha_tensor.clone(),
                ]
            };
            seen.insert(key.clone(), slots);
            let alpha_tensor = module.alpha_tensor.clone().unwrap();
            alphas.insert(alpha_tensor.name.clone(), module.alpha.unwrap());
        }
        let error = assemble_turbo_modules(&expected, seen, &alphas, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::MissingModule);
    }

    // ------------------------------------------------------------ identities

    /// The rank-uniform structure digest is FROZEN.
    ///
    /// Adding a rank-dynamic shape to this contract touched
    /// `turbo_structure_identity`, `assemble_turbo_modules`, and every field
    /// they read. `structure_identity_sha256` feeds `contract_identity_sha256`
    /// and, through it, `adapter_identity_sha256` — the artifact identity
    /// frozen execution plans and provenance records already carry for the
    /// five published rank-128 tiers. A refactor that reorders one hashed
    /// field, or hashes an `Option` differently, would silently reissue every
    /// one of those identities. These literals were computed BEFORE the shape
    /// refactor and are the proof it changed no Uniform byte; they are not
    /// recomputable from anything else in the tree, so they must never be
    /// "updated to match" — a diff here means the Uniform path moved.
    #[test]
    fn the_uniform_structure_identity_is_byte_identical_after_the_shape_refactor() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let inspection = inspect_fixture(&header, &data).unwrap();
        assert_eq!(
            inspection.header_identity_sha256,
            "8ad29d9c7c2c60c0cd7a30998d60418efb464fcd4c37794464f7ef5ee7f1c41a"
        );
        assert_eq!(
            inspection.structure_identity_sha256,
            "86970563baa338bab96d112566e2a951a9fe6350cf5f28930b904241a8567553"
        );
        assert_eq!(
            inspection.contract_identity_sha256,
            "858c8986dbc4d88a2da50da6a96f0f4236f7e0782daaff87b230f3ef34e0bb53"
        );

        // The authenticated artifact identity is the one frozen plans record,
        // and it folds the two digests above together with the tier and the
        // content digest.
        let (_directory, path, pinned) = fixtures::pinned();
        let contract = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        assert_eq!(
            contract.inspection().structure_identity_sha256,
            "86970563baa338bab96d112566e2a951a9fe6350cf5f28930b904241a8567553"
        );
        assert_eq!(
            contract.adapter_identity_sha256(),
            turbo_adapter_identity(
                H3TurboLoraTier::Fl2v8StepV10,
                H3TransformerTask::T2VaFl2Va,
                &contract.inspection().header_identity_sha256,
                "86970563baa338bab96d112566e2a951a9fe6350cf5f28930b904241a8567553",
                contract.content_sha256(),
            )
        );
    }

    #[test]
    fn the_contract_identity_is_structural_and_never_an_artifact_identity() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let first = inspect_fixture(&header, &data).unwrap();
        let second = inspect_fixture(&header, &data).unwrap();
        assert_eq!(
            first.structure_identity_sha256,
            second.structure_identity_sha256
        );
        assert_eq!(
            first.contract_identity_sha256,
            second.contract_identity_sha256
        );

        // Same header, same structure, DIFFERENT weight bytes: the contract
        // identity cannot tell them apart, which is exactly why it must never
        // stand in for an artifact identity.
        let mut tampered = data.clone();
        let weight = &header["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]["data_offsets"];
        let start = weight[0].as_u64().unwrap() as usize;
        tampered[start] ^= 0xff;
        let other = inspect_fixture(&header, &tampered).unwrap();
        assert_eq!(
            first.contract_identity_sha256,
            other.contract_identity_sha256
        );

        // The authenticated identity does tell them apart.
        let (_a, path_a) = write_adapter(&header, &data);
        let (_b, path_b) = write_adapter(&header, &tampered);
        let mut pinned_a = expectation.clone();
        pinned_a.content_sha256 = Some(sha256_hex(Sha256::digest(std::fs::read(&path_a).unwrap())));
        let authentic_a = authenticate_h3_turbo_lora_adapter_against(
            &path_a,
            &pinned_a,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        let mut pinned_b = expectation.clone();
        pinned_b.content_sha256 = Some(sha256_hex(Sha256::digest(std::fs::read(&path_b).unwrap())));
        let authentic_b = authenticate_h3_turbo_lora_adapter_against(
            &path_b,
            &pinned_b,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        assert_ne!(
            authentic_a.adapter_identity_sha256(),
            authentic_b.adapter_identity_sha256()
        );
        assert_eq!(
            authentic_a.inspection().contract_identity_sha256,
            authentic_b.inspection().contract_identity_sha256
        );
        // And it is not the contract identity under another name.
        assert_ne!(
            authentic_a.adapter_identity_sha256(),
            authentic_a.inspection().contract_identity_sha256
        );

        // A different alpha is a different overlay even at identical shapes.
        let mut louder = expectation.clone();
        louder.shape = H3TurboLoraShape::Uniform {
            training_rank: 4,
            training_alpha: 0.5,
        };
        let (other_header, other_data) = fixture_adapter(&louder);
        let (_directory, path) = write_adapter(&other_header, &other_data);
        let louder_inspection = inspect_h3_turbo_lora_adapter_against(&path, &louder).unwrap();
        assert_ne!(
            first.structure_identity_sha256,
            louder_inspection.structure_identity_sha256
        );
        assert_eq!(louder_inspection.scale, 0.125);
    }

    // -------------------------------------------------------- authentication

    #[derive(Default)]
    struct CancelAfter {
        remaining: std::sync::atomic::AtomicUsize,
    }

    impl H3ComfyInt8Cancellation for CancelAfter {
        fn is_cancelled(&self) -> bool {
            self.remaining
                .fetch_update(
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                    |value| Some(value.saturating_sub(1)),
                )
                .is_ok_and(|value| value == 0)
        }
    }

    fn pinned_fixture() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        H3TurboLoraExpectation,
    ) {
        fixtures::pinned()
    }

    #[test]
    fn authentication_verifies_the_content_digest_and_reports_it() {
        let (_directory, path, pinned) = pinned_fixture();
        let digest = pinned.content_sha256.clone().unwrap();
        let contract = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap();
        assert_eq!(contract.content_sha256(), digest);
        assert_eq!(contract.tier(), H3TurboLoraTier::Fl2v8StepV10);
        assert_eq!(contract.task(), H3TransformerTask::T2VaFl2Va);
        assert_eq!(contract.module_count(), 12);
        assert_eq!(contract.scale(), 0.0625);

        // Authentication agrees with inspection on the whole structure.
        let inspected = inspect_h3_turbo_lora_adapter_against(&path, &pinned).unwrap();
        assert_eq!(contract.inspection(), &inspected);
        assert_eq!(contract.modules(), &inspected.modules);

        let mut wrong = pinned.clone();
        wrong.content_sha256 = Some("0".repeat(64));
        let error = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &wrong,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::ContentDigestMismatch);
    }

    #[test]
    fn authentication_refuses_to_run_without_a_content_pin() {
        let (_directory, path, mut pinned) = pinned_fixture();
        pinned.content_sha256 = None;
        let error = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::MissingContentPin);
    }

    #[cfg(unix)]
    #[test]
    fn authentication_refuses_a_symlinked_adapter() {
        let (directory, path, pinned) = pinned_fixture();
        let link = directory.path().join("linked.safetensors");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        let error = authenticate_h3_turbo_lora_adapter_against(
            &link,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::FileIdentityChanged);
        // The same bytes through the real path still authenticate.
        assert!(authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &H3ComfyNeverCancel,
        )
        .is_ok());
    }

    #[test]
    fn authentication_stops_at_a_cancellation_boundary() {
        let (_directory, path, pinned) = pinned_fixture();
        let cancellation = CancelAfter::default();
        let error = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Fl2v8StepV10,
            &cancellation,
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::Cancelled);
    }

    #[test]
    fn authentication_rejects_a_file_that_is_not_the_pinned_tier() {
        let (_directory, path, _) = pinned_fixture();
        let error = authenticate_h3_turbo_lora_adapter(
            &path,
            H3TurboLoraTier::Fl2v768p4StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        // A wrongly sized file is refused before it is hashed, so a stand-in
        // never reaches module validation under a published pin.
        assert_eq!(error.code, H3TurboLoraErrorCode::SourceSizeMismatch);
    }

    #[test]
    fn authentication_refuses_a_tier_whose_task_disagrees_with_the_expectation() {
        let (_directory, path, pinned) = pinned_fixture();
        let error = authenticate_h3_turbo_lora_adapter_against(
            &path,
            &pinned,
            H3TurboLoraTier::Ref2v4StepV10,
            &H3ComfyNeverCancel,
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::TaskAuthorityMismatch);
    }
}
