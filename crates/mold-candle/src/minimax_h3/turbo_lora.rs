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
    safetensors_dtype_size, sha256_hex, strict_json_value, H3ComfyInt8Cancellation,
    H3OpenedFileIdentity, FILE_READ_CHUNK_BYTES, MAX_HEADER_BYTES, MAX_TENSORS,
    MAX_TENSOR_KEY_BYTES, MAX_TENSOR_RANK,
};
use super::dit::{
    expected_h3_weight_specs, H3AdaLnMode, H3PrecisionProfile, H3TransformerConfig,
    H3TransformerTask,
};

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
const LORA_A_SUFFIX: &str = ".lora_A.weight";
const LORA_B_SUFFIX: &str = ".lora_B.weight";
const ALPHA_SUFFIX: &str = ".alpha";
/// Base-checkpoint sidecars that must never appear in an adapter. Their
/// presence means a merged or quantized checkpoint was supplied in place of one.
const BASE_SIDECAR_SUFFIXES: [&str; 2] = [".weight_scale", ".comfy_quant"];
const ALPHA_BYTES: u64 = 4;
const STRUCTURE_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.turbo-lora-structure.v1\0";
const ADAPTER_IDENTITY_DOMAIN: &[u8] = b"mold.minimax-h3.turbo-lora-adapter.v1\0";

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
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    /// Read from the file's own `alpha` tensor, never assumed.
    pub alpha: f32,
    /// `alpha / rank`.
    pub scale: f32,
    pub lora_a: H3TurboLoraTensorRef,
    pub lora_b: H3TurboLoraTensorRef,
    pub alpha_tensor: H3TurboLoraTensorRef,
}

/// A fully validated adapter contract. Holding one proves the header, module
/// map, shapes, dtypes, and alphas agreed with the expectation; it grants no
/// execution authority and loads no tensor data.
#[derive(Clone, Debug, PartialEq)]
pub struct H3TurboLoraContract {
    /// `None` when validated against a bespoke expectation rather than one of
    /// the published tiers.
    pub tier: Option<H3TurboLoraTier>,
    pub task: H3TransformerTask,
    pub source_repository_revision: String,
    pub file_bytes: u64,
    pub header_len: u64,
    pub header_identity_sha256: String,
    /// Present only after full-content authentication.
    pub content_sha256: Option<String>,
    pub tensor_count: usize,
    pub training_rank: usize,
    pub training_alpha: f32,
    /// The one `alpha / rank` every module agreed on.
    pub scale: f32,
    pub payload_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub modules: BTreeMap<String, H3TurboLoraModule>,
    /// SHA-256 over the sorted validated structure: module names, kinds,
    /// ranks, in/out features, and alpha bits.
    pub structure_identity_sha256: String,
    /// SHA-256 binding the tier, task, header identity, and structure identity.
    pub adapter_identity_sha256: String,
}

impl H3TurboLoraContract {
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
/// This reads the JSON header plus the 4-byte `alpha` scalars. It does not
/// verify the full-content digest — see
/// [`authenticate_h3_turbo_lora_adapter`] for that.
pub fn inspect_h3_turbo_lora_adapter(
    path: &Path,
    tier: H3TurboLoraTier,
) -> H3TurboLoraResult<H3TurboLoraContract> {
    inspect_h3_turbo_lora_adapter_against(path, &tier.expectation(), Some(tier))
}

/// Parse and validate one adapter header against an explicit expectation.
pub fn inspect_h3_turbo_lora_adapter_against(
    path: &Path,
    expectation: &H3TurboLoraExpectation,
    tier: Option<H3TurboLoraTier>,
) -> H3TurboLoraResult<H3TurboLoraContract> {
    if let Some(tier) = tier {
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
    }
    let mut file = File::open(path).map_err(|error| {
        failure(
            H3TurboLoraErrorCode::Io,
            format!("failed to open H3 Turbo adapter: {error}"),
        )
    })?;
    let parsed = read_turbo_header(&mut file)?;
    build_contract(&mut file, parsed, expectation, tier, None)
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
    authenticate_h3_turbo_lora_adapter_against(path, &tier.expectation(), Some(tier), cancellation)
}

/// Open and fully authenticate one adapter against an explicit expectation.
pub fn authenticate_h3_turbo_lora_adapter_against(
    path: &Path,
    expectation: &H3TurboLoraExpectation,
    tier: Option<H3TurboLoraTier>,
    cancellation: &dyn H3ComfyInt8Cancellation,
) -> H3TurboLoraResult<H3TurboLoraContract> {
    if let Some(tier) = tier {
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
    }
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
    if let Some(expected) = expectation.content_sha256.as_deref() {
        if content_sha256 != expected {
            return Err(failure(
                H3TurboLoraErrorCode::ContentDigestMismatch,
                format!(
                    "H3 Turbo adapter content digest {content_sha256} does not match source-pinned {expected}"
                ),
            ));
        }
    }
    let contract = build_contract(
        &mut file,
        parsed,
        expectation,
        tier,
        Some(content_sha256.clone()),
    )?;
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
    Ok(contract)
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

fn build_contract(
    file: &mut File,
    parsed: TurboParsedHeader,
    expectation: &H3TurboLoraExpectation,
    tier: Option<H3TurboLoraTier>,
    content_sha256: Option<String>,
) -> H3TurboLoraResult<H3TurboLoraContract> {
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
    let alphas = read_turbo_alphas(file, &parsed)?;
    let modules = assemble_turbo_modules(&expected, seen, &alphas, expectation)?;
    let structure_identity_sha256 = turbo_structure_identity(&modules);
    let adapter_identity_sha256 = turbo_adapter_identity(
        tier,
        expectation.task,
        &parsed.header_identity_sha256,
        &structure_identity_sha256,
    );
    let payload_bytes = parsed.file_len - parsed.header_len - 8;
    Ok(H3TurboLoraContract {
        tier,
        task: expectation.task,
        source_repository_revision: H3_TURBO_LORA_SOURCE_REVISION.to_owned(),
        file_bytes: parsed.file_len,
        header_len: parsed.header_len,
        header_identity_sha256: parsed.header_identity_sha256,
        content_sha256,
        tensor_count: parsed.tensors.len(),
        training_rank: expectation.training_rank,
        training_alpha: expectation.training_alpha,
        scale: expectation.scale(),
        payload_bytes,
        metadata: parsed.metadata,
        modules,
        structure_identity_sha256,
        adapter_identity_sha256,
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
    let root = strict_json_value(&bytes, "H3 Turbo safetensors header").map_err(|message| {
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
    identity.update(&bytes);
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
    if let Some(declared) = metadata.get("training_rank") {
        let parsed = declared.trim().parse::<usize>().map_err(|error| {
            failure(
                H3TurboLoraErrorCode::InvalidMetadata,
                format!("H3 Turbo metadata \"training_rank\" is not an integer: {error}"),
            )
        })?;
        if parsed != expectation.training_rank {
            return Err(mismatch(
                "training_rank",
                declared,
                expectation.training_rank.to_string(),
            ));
        }
    }
    for (key, expected) in [
        ("training_alpha", expectation.training_alpha),
        ("training_scale", expectation.scale()),
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
) -> H3TurboLoraResult<BTreeMap<String, f32>> {
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

struct ExpectedModule {
    scope: H3TurboLoraModuleScope,
    kind: H3TurboLoraModuleKind,
    base_weight_name: String,
    rank: usize,
    in_features: usize,
    out_features: usize,
    alpha: f32,
}

/// Derive the exact expected module set from the base checkpoint's own weight
/// specs, so the adapter contract can never drift from the checkpoint contract.
fn expected_turbo_modules(
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
    let scopes = (0..expectation.config.num_layers)
        .map(H3TurboLoraModuleScope::MainBlock)
        .chain(
            (0..expectation.config.token_refiner_num_layers)
                .map(H3TurboLoraModuleScope::TokenRefinerBlock),
        );
    let mut expected = BTreeMap::new();
    for scope in scopes {
        for kind in H3TurboLoraModuleKind::ALL {
            let base_prefix = scope.base_prefix();
            let base_weight_name = format!("{base_prefix}.{}.weight", kind.suffix());
            let spec = specs.get(&base_weight_name).ok_or_else(|| {
                failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!("H3 base checkpoint has no weight {base_weight_name:?}"),
                )
            })?;
            let [out_features, in_features] = spec.shape[..] else {
                return Err(failure(
                    H3TurboLoraErrorCode::ConfigMismatch,
                    format!("H3 base weight {base_weight_name:?} is not a rank-2 linear"),
                ));
            };
            expected.insert(
                format!("{H3_TURBO_LORA_KEY_PREFIX}{base_prefix}.{}", kind.suffix()),
                ExpectedModule {
                    scope,
                    kind,
                    base_weight_name,
                    rank: kind.rank(expectation.training_rank),
                    in_features,
                    out_features,
                    alpha: kind.alpha(expectation.training_alpha),
                },
            );
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
    if parsed.tensors.len() != expectation.tensor_count() {
        return Err(failure(
            H3TurboLoraErrorCode::TensorCountMismatch,
            format!(
                "H3 Turbo adapter carries {} tensors, expected {}",
                parsed.tensors.len(),
                expectation.tensor_count()
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
        let expected_shape = match slot {
            0 => vec![module_expected.rank, module_expected.in_features],
            1 => vec![module_expected.out_features, module_expected.rank],
            _ => Vec::new(),
        };
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
        if tensor.shape != expected_shape {
            return Err(failure(
                H3TurboLoraErrorCode::ShapeMismatch,
                format!(
                    "H3 Turbo tensor {name:?} has shape {:?}, expected {expected_shape:?}",
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
        let [Some(lora_a), Some(lora_b), Some(alpha_tensor)] = slots else {
            return Err(failure(
                H3TurboLoraErrorCode::MissingModule,
                format!("H3 Turbo module {module_key:?} is missing a lora_A/lora_B/alpha tensor"),
            ));
        };
        let alpha = *alphas.get(&alpha_tensor.name).ok_or_else(|| {
            failure(
                H3TurboLoraErrorCode::AlphaMismatch,
                format!("H3 Turbo module {module_key:?} has no readable alpha scalar"),
            )
        })?;
        if alpha != module_expected.alpha {
            return Err(failure(
                H3TurboLoraErrorCode::AlphaMismatch,
                format!(
                    "H3 Turbo module {module_key:?} declares alpha {alpha}, expected {}",
                    module_expected.alpha
                ),
            ));
        }
        let scale = alpha / module_expected.rank as f32;
        if scale != expectation.scale() {
            return Err(failure(
                H3TurboLoraErrorCode::ScaleMismatch,
                format!(
                    "H3 Turbo module {module_key:?} resolves scale {scale}, expected {}",
                    expectation.scale()
                ),
            ));
        }
        modules.insert(
            module_key.clone(),
            H3TurboLoraModule {
                name: module_key.clone(),
                base_weight_name: module_expected.base_weight_name.clone(),
                scope: module_expected.scope,
                kind: module_expected.kind,
                rank: module_expected.rank,
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

fn turbo_structure_identity(modules: &BTreeMap<String, H3TurboLoraModule>) -> String {
    let mut digest = Sha256::new();
    digest.update(STRUCTURE_IDENTITY_DOMAIN);
    digest.update((modules.len() as u64).to_le_bytes());
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
        digest.update(module.alpha.to_bits().to_le_bytes());
    }
    sha256_hex(digest.finalize())
}

fn turbo_adapter_identity(
    tier: Option<H3TurboLoraTier>,
    task: H3TransformerTask,
    header_identity_sha256: &str,
    structure_identity_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(ADAPTER_IDENTITY_DOMAIN);
    digest.update(
        tier.map(H3TurboLoraTier::stable_id)
            .unwrap_or("minimax-h3.turbo-lora.unpinned")
            .as_bytes(),
    );
    digest.update([0, task as u8]);
    digest.update(header_identity_sha256.as_bytes());
    digest.update(structure_identity_sha256.as_bytes());
    sha256_hex(digest.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::Write;

    use serde_json::{Map, Value};

    use super::super::comfy_dit::H3ComfyNeverCancel;
    use super::*;

    /// A valid but tiny transformer geometry, mirroring `comfy_dit`'s runtime
    /// fixture so the synthetic adapter overlays a real base weight set.
    fn fixture_config() -> H3TransformerConfig {
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

    fn fixture_expectation() -> H3TurboLoraExpectation {
        H3TurboLoraExpectation {
            config: fixture_config(),
            task: H3TransformerTask::T2VaFl2Va,
            training_rank: 4,
            training_alpha: 0.25,
            file_bytes: None,
            content_sha256: None,
            header_len: None,
            header_identity_sha256: None,
        }
    }

    /// Build a complete synthetic adapter for an expectation: the exact module
    /// set, shapes, dtypes, and alpha scalars the contract requires.
    fn fixture_adapter(expectation: &H3TurboLoraExpectation) -> (Value, Vec<u8>) {
        let (expected, _) = expected_turbo_modules(expectation).unwrap();
        let mut entries = BTreeMap::<String, (&'static str, Vec<usize>, Option<f32>)>::new();
        for (module_key, module) in &expected {
            entries.insert(
                format!("{module_key}{LORA_A_SUFFIX}"),
                (
                    H3_TURBO_LORA_WEIGHT_DTYPE,
                    vec![module.rank, module.in_features],
                    None,
                ),
            );
            entries.insert(
                format!("{module_key}{LORA_B_SUFFIX}"),
                (
                    H3_TURBO_LORA_WEIGHT_DTYPE,
                    vec![module.out_features, module.rank],
                    None,
                ),
            );
            entries.insert(
                format!("{module_key}{ALPHA_SUFFIX}"),
                (H3_TURBO_LORA_ALPHA_DTYPE, Vec::new(), Some(module.alpha)),
            );
        }
        let mut header = Map::new();
        let mut data = Vec::new();
        for (name, (dtype, shape, alpha)) in entries {
            let elements = shape.iter().product::<usize>();
            let bytes = elements * safetensors_dtype_size(dtype).unwrap() as usize;
            let start = data.len() as u64;
            match alpha {
                Some(value) => data.extend_from_slice(&value.to_le_bytes()),
                None => data.resize(data.len() + bytes, 0),
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
            Value::String(expectation.training_rank.to_string()),
        );
        metadata.insert(
            "training_alpha".into(),
            Value::String(expectation.training_alpha.to_string()),
        );
        metadata.insert(
            "training_scale".into(),
            Value::String(expectation.scale().to_string()),
        );
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

    fn write_adapter(header: &Value, data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
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

    /// Re-lay the tensor data so offsets stay contiguous after a mutation.
    fn relayout(header: &mut Value, data_len_hint: usize) -> Vec<u8> {
        let object = header.as_object_mut().unwrap();
        let names = object
            .keys()
            .filter(|name| *name != "__metadata__")
            .cloned()
            .collect::<Vec<_>>();
        let mut data = Vec::with_capacity(data_len_hint);
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

    fn inspect_fixture(header: &Value, data: &[u8]) -> H3TurboLoraResult<H3TurboLoraContract> {
        let (_directory, path) = write_adapter(header, data);
        inspect_h3_turbo_lora_adapter_against(&path, &fixture_expectation(), None)
    }

    fn expect_code(result: H3TurboLoraResult<H3TurboLoraContract>) -> H3TurboLoraErrorCode {
        result.expect_err("adapter must be rejected").code
    }

    #[test]
    fn synthetic_adapter_validates_every_module_and_reads_its_alphas() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let contract = inspect_fixture(&header, &data).unwrap();

        assert_eq!(contract.module_count(), expectation.module_count());
        assert_eq!(contract.tensor_count, expectation.tensor_count());
        assert_eq!(contract.tier, None);
        assert_eq!(contract.scale, 0.0625);
        assert_eq!(contract.payload_bytes, data.len() as u64);
        assert_eq!(contract.header_identity_sha256.len(), 64);
        assert_eq!(contract.structure_identity_sha256.len(), 64);
        assert_eq!(contract.adapter_identity_sha256.len(), 64);
        assert_eq!(contract.content_sha256, None);

        let qkv = contract
            .modules
            .get("diffusion_model.blocks.0.attn.qkv_proj")
            .unwrap();
        assert_eq!(qkv.kind, H3TurboLoraModuleKind::AttnQkvProj);
        assert_eq!(qkv.scope, H3TurboLoraModuleScope::MainBlock(0));
        assert_eq!(qkv.base_weight_name, "blocks.0.attn.qkv_proj.weight");
        // Fused Q/K/V: rank and alpha are both tripled, so the scale holds.
        assert_eq!(qkv.rank, 12);
        assert_eq!(qkv.alpha, 0.75);
        assert_eq!(qkv.scale, contract.scale);
        assert_eq!(qkv.lora_a.shape, vec![12, 256]);
        assert_eq!(qkv.lora_b.shape, vec![768, 12]);

        let fc1 = contract
            .modules
            .get("diffusion_model.token_refiner.blocks.0.mlp.fc1")
            .unwrap();
        assert_eq!(fc1.scope, H3TurboLoraModuleScope::TokenRefinerBlock(0));
        assert_eq!(fc1.rank, 4);
        assert_eq!(fc1.alpha, 0.25);
        assert_eq!(fc1.lora_a.shape, vec![4, 256]);
        assert_eq!(fc1.lora_b.shape, vec![512, 4]);

        assert_eq!(contract.main_block_modules(0).len(), 4);
        assert_eq!(contract.token_refiner_modules(0).len(), 4);
        let block_bytes = contract.scope_delta_bytes(H3TurboLoraModuleScope::MainBlock(0));
        assert!(block_bytes > 0);
        assert_eq!(
            block_bytes,
            contract.scope_delta_bytes(H3TurboLoraModuleScope::MainBlock(1))
        );
    }

    #[test]
    fn identity_tracks_the_validated_structure_and_the_header() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let first = inspect_fixture(&header, &data).unwrap();
        let second = inspect_fixture(&header, &data).unwrap();
        assert_eq!(
            first.structure_identity_sha256,
            second.structure_identity_sha256
        );
        assert_eq!(
            first.adapter_identity_sha256,
            second.adapter_identity_sha256
        );

        // A different alpha is a different overlay even at identical shapes.
        let mut louder = expectation.clone();
        louder.training_alpha = 0.5;
        let (other_header, other_data) = fixture_adapter(&louder);
        let (_directory, path) = write_adapter(&other_header, &other_data);
        let other = inspect_h3_turbo_lora_adapter_against(&path, &louder, None).unwrap();
        assert_ne!(
            first.structure_identity_sha256,
            other.structure_identity_sha256
        );
        assert_ne!(first.adapter_identity_sha256, other.adapter_identity_sha256);
        assert_eq!(other.scale, 0.125);
    }

    #[test]
    fn a_missing_tensor_is_a_tensor_count_mismatch() {
        let (mut header, _) = fixture_adapter(&fixture_expectation());
        header
            .as_object_mut()
            .unwrap()
            .remove("diffusion_model.blocks.1.mlp.fc2.alpha")
            .unwrap();
        let data = relayout(&mut header, 0);
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
        let data = relayout(&mut header, 0);
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
        let data = relayout(&mut header, 0);
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
        let data = relayout(&mut header, 0);
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
        let data = relayout(&mut header, 0);
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
        let data = relayout(&mut header, 0);
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
            let data = relayout(&mut header, 0);
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
            inspect_h3_turbo_lora_adapter_against(&path, &fixture_expectation(), None).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);
        assert!(error.message.contains("header length"), "{}", error.message);
    }

    #[test]
    fn trailing_and_non_contiguous_tensor_data_is_refused() {
        let (header, mut data) = fixture_adapter(&fixture_expectation());
        data.push(0);
        assert_eq!(
            expect_code(inspect_fixture(&header, &data)),
            H3TurboLoraErrorCode::InvalidHeader
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
        sized.file_bytes = None;
        sized.header_len = Some(1);
        let error = inspect_h3_turbo_lora_adapter_against(&path, &sized, None).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::InvalidHeader);

        let mut identified = expectation.clone();
        identified.header_identity_sha256 = Some("0".repeat(64));
        let error = inspect_h3_turbo_lora_adapter_against(&path, &identified, None).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::HeaderIdentityMismatch);
    }

    #[test]
    fn a_tier_whose_task_disagrees_with_the_expectation_is_refused() {
        let (header, data) = fixture_adapter(&fixture_expectation());
        let (_directory, path) = write_adapter(&header, &data);
        let error = inspect_h3_turbo_lora_adapter_against(
            &path,
            &fixture_expectation(),
            Some(H3TurboLoraTier::Ref2v4StepV10),
        )
        .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::TaskAuthorityMismatch);
    }

    #[test]
    fn an_incomplete_module_never_produces_a_partial_overlay() {
        let expectation = fixture_expectation();
        let (expected, _) = expected_turbo_modules(&expectation).unwrap();
        let (header, data) = fixture_adapter(&expectation);
        let (_directory, path) = write_adapter(&header, &data);
        let contract = inspect_h3_turbo_lora_adapter_against(&path, &expectation, None).unwrap();

        let mut seen = BTreeMap::new();
        let mut alphas = BTreeMap::new();
        for (key, module) in &contract.modules {
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
                    Some(module.alpha_tensor.clone()),
                ]
            };
            seen.insert(key.clone(), slots);
            alphas.insert(module.alpha_tensor.name.clone(), module.alpha);
        }
        let error = assemble_turbo_modules(&expected, seen, &alphas, &expectation).unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::MissingModule);
    }

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

    #[test]
    fn authentication_verifies_the_content_digest_and_reports_it() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let (_directory, path) = write_adapter(&header, &data);
        let bytes = std::fs::read(&path).unwrap();
        let digest = sha256_hex(Sha256::digest(&bytes));

        let mut pinned = expectation.clone();
        pinned.content_sha256 = Some(digest.clone());
        let contract =
            authenticate_h3_turbo_lora_adapter_against(&path, &pinned, None, &H3ComfyNeverCancel)
                .unwrap();
        assert_eq!(contract.content_sha256.as_deref(), Some(digest.as_str()));
        // Authentication must agree with inspection on everything else.
        let inspected = inspect_h3_turbo_lora_adapter_against(&path, &pinned, None).unwrap();
        assert_eq!(
            contract.adapter_identity_sha256,
            inspected.adapter_identity_sha256
        );
        assert_eq!(contract.modules, inspected.modules);

        let mut wrong = expectation.clone();
        wrong.content_sha256 = Some("0".repeat(64));
        let error =
            authenticate_h3_turbo_lora_adapter_against(&path, &wrong, None, &H3ComfyNeverCancel)
                .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::ContentDigestMismatch);
    }

    #[test]
    fn authentication_refuses_a_symlinked_adapter() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let (directory, path) = write_adapter(&header, &data);
        let link = directory.path().join("linked.safetensors");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&path, &link).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&path, &link).unwrap();
        let result = authenticate_h3_turbo_lora_adapter_against(
            &link,
            &expectation,
            None,
            &H3ComfyNeverCancel,
        );
        #[cfg(unix)]
        assert_eq!(
            result.unwrap_err().code,
            H3TurboLoraErrorCode::FileIdentityChanged
        );
        #[cfg(not(unix))]
        assert!(result.is_ok());
    }

    #[test]
    fn authentication_stops_at_a_cancellation_boundary() {
        let expectation = fixture_expectation();
        let (header, data) = fixture_adapter(&expectation);
        let (_directory, path) = write_adapter(&header, &data);
        let cancellation = CancelAfter::default();
        let error =
            authenticate_h3_turbo_lora_adapter_against(&path, &expectation, None, &cancellation)
                .unwrap_err();
        assert_eq!(error.code, H3TurboLoraErrorCode::Cancelled);
    }

    #[test]
    fn authentication_rejects_a_file_that_is_not_the_pinned_tier() {
        let (header, data) = fixture_adapter(&fixture_expectation());
        let (_directory, path) = write_adapter(&header, &data);
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
    fn the_published_geometry_derives_from_the_shipped_checkpoint_specs() {
        let expectation = H3TurboLoraTier::Fl2v8StepV10.expectation();
        let (expected, base_names) = expected_turbo_modules(&expectation).unwrap();
        assert_eq!(expected.len(), H3_TURBO_LORA_MODULE_COUNT);

        // Exactly the shapes read from the published headers.
        let published = [
            ("attn.qkv_proj", 384_usize, 5_376_usize, 21_504_usize),
            ("attn.out_proj", 128, 7_168, 5_376),
            ("mlp.fc1", 128, 5_376, 28_672),
            ("mlp.fc2", 128, 14_336, 5_376),
        ];
        for (suffix, rank, in_features, out_features) in published {
            for prefix in ["blocks.0", "blocks.49", "token_refiner.blocks.1"] {
                let module = expected
                    .get(&format!("{H3_TURBO_LORA_KEY_PREFIX}{prefix}.{suffix}"))
                    .unwrap_or_else(|| panic!("{prefix}.{suffix}"));
                assert_eq!(module.rank, rank, "{prefix}.{suffix}");
                assert_eq!(module.in_features, in_features, "{prefix}.{suffix}");
                assert_eq!(module.out_features, out_features, "{prefix}.{suffix}");
                assert!(base_names.contains(&module.base_weight_name));
            }
        }
        assert!(!expected.contains_key("diffusion_model.blocks.50.mlp.fc1"));
        assert!(!expected.contains_key("diffusion_model.token_refiner.blocks.2.mlp.fc1"));
    }

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
