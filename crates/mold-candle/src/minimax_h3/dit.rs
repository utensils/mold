//! Weight-strict MiniMax H3 DiT primitives.
//!
//! This module intentionally stops at the reusable Candle model boundary. It
//! does not register an engine, advertise a capability, resolve artifacts, or
//! download weights. The two released task partitions use indistinguishable
//! tensor names and shapes, so callers must supply the component identity
//! (`transformer` versus `transformer_ref`) separately and the audit fails
//! closed when that identity does not match the requested task.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error as StdError;
use std::fmt;
use std::path::Path;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};

pub const H3_MODALITY_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum H3Modality {
    Video = 0,
    Text = 1,
    Audio = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3TransformerTask {
    /// The shared text-to-A/V and first/last-frame-to-A/V partition.
    T2VaFl2Va,
    /// The ordered multimodal-reference partition.
    Ref2Va,
}

impl H3TransformerTask {
    pub const fn checkpoint_component(self) -> H3CheckpointComponent {
        match self {
            Self::T2VaFl2Va => H3CheckpointComponent::Transformer,
            Self::Ref2Va => H3CheckpointComponent::TransformerRef,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H3CheckpointComponent {
    Transformer,
    TransformerRef,
}

impl fmt::Display for H3CheckpointComponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Transformer => "transformer",
            Self::TransformerRef => "transformer_ref",
        })
    }
}

/// The two checkpoint-level representations supported by this correctness
/// primitive. Quantized auxiliary tensors are deliberately not accepted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3AdaLnMode {
    /// Released full transformer: FP32 timestep MLP and BF16 AdaLN projections.
    Full,
    /// Comfy's pruned representation: a shared FP32 curve table replaces the
    /// timestep MLP and every AdaLN projection consumes the compact basis.
    Curve { grid: usize, basis_dim: usize },
}

impl H3AdaLnMode {
    pub const fn input_dim(self, config: &H3TransformerConfig) -> usize {
        match self {
            Self::Full => config.time_embed_dim,
            Self::Curve { basis_dim, .. } => basis_dim,
        }
    }

    pub const fn qkv_layout(self) -> H3QkvLayout {
        match self {
            // The official files group Q/K/V rows per attention head.
            Self::Full => H3QkvLayout::GroupedPerHead,
            // Comfy's pruned files are repacked for its direct Q/K/V split.
            Self::Curve { .. } => H3QkvLayout::QkvMajor,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3QkvLayout {
    GroupedPerHead,
    QkvMajor,
}

/// Production checkpoints are mixed BF16/FP32. The all-F32 profile exists only
/// for weight-free synthetic conformance and must be selected explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3PrecisionProfile {
    OfficialMixedBf16F32,
    SyntheticF32,
}

impl H3PrecisionProfile {
    pub const fn compute_dtype(self) -> DType {
        match self {
            Self::OfficialMixedBf16F32 => DType::BF16,
            Self::SyntheticF32 => DType::F32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl H3TensorSpec {
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }
}

/// Complete, header-derived tensor inventory for one component. Supplying the
/// component separately is load-bearing: official FL2VA and Ref2VA indexes
/// have the same hash, names, and shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3CheckpointInventory {
    pub component: H3CheckpointComponent,
    pub tensors: BTreeMap<String, H3TensorSpec>,
}

/// One concrete checkpoint source whose header inventory and tensor backend
/// cannot be separated after construction. The component role is still an
/// authority supplied by the artifact resolver because the released
/// `transformer` and `transformer_ref` headers are indistinguishable.
pub struct H3CheckpointSource<'a> {
    inventory: H3CheckpointInventory,
    vb: VarBuilder<'a>,
}

impl fmt::Debug for H3CheckpointSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3CheckpointSource")
            .field("component", &self.inventory.component)
            .field("tensor_count", &self.inventory.tensors.len())
            .finish_non_exhaustive()
    }
}

impl H3CheckpointSource<'static> {
    /// Build an owned source from the exact tensors that its loader will use.
    /// This is useful for synthetic conformance and buffered integrations; the
    /// inventory is derived rather than accepted separately.
    pub fn from_tensors(
        component: H3CheckpointComponent,
        tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let inventory = H3CheckpointInventory {
            component,
            tensors: tensors
                .iter()
                .map(|(name, tensor)| {
                    (
                        name.clone(),
                        H3TensorSpec::new(tensor.dims().to_vec(), tensor.dtype()),
                    )
                })
                .collect(),
        };
        Ok(Self {
            inventory,
            // Every H3 load site requests its checkpoint dtype explicitly.
            vb: VarBuilder::from_tensors(tensors, DType::F32, device),
        })
    }

    /// Memory-map safetensors shards, derive their inventory from those exact
    /// mappings, and retain the same mappings as the only tensor backend.
    /// Duplicate tensor names across shards are rejected rather than inheriting
    /// Candle's last-shard-wins behavior.
    ///
    /// # Safety
    ///
    /// The mapped files must not be modified or truncated until this source has
    /// been consumed by [`H3Transformer::load`]. This is the same requirement as
    /// [`candle::safetensors::MmapedSafetensors::multi`].
    pub unsafe fn from_mmaped_safetensors<P: AsRef<Path>>(
        component: H3CheckpointComponent,
        paths: &[P],
        device: &Device,
    ) -> Result<Self> {
        // SAFETY: inherited by this function's documented caller contract.
        let mapped = unsafe { candle::safetensors::MmapedSafetensors::multi(paths)? };
        let mut tensors = BTreeMap::new();
        for (name, view) in mapped.tensors() {
            let spec = H3TensorSpec::new(view.shape().to_vec(), DType::try_from(view.dtype())?);
            if tensors.insert(name.clone(), spec).is_some() {
                candle::bail!("duplicate MiniMax H3 tensor {name:?} across checkpoint shards")
            }
        }
        Ok(Self {
            inventory: H3CheckpointInventory { component, tensors },
            // Every H3 load site requests its checkpoint dtype explicitly.
            vb: VarBuilder::from_backend(Box::new(mapped), DType::F32, device.clone()),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum H3WeightAuditIssue {
    InvalidConfig(String),
    WrongComponent {
        expected: H3CheckpointComponent,
        actual: H3CheckpointComponent,
    },
    AmbiguousAdaLnRepresentation,
    MissingAdaLnRepresentation,
    InvalidCurveTable {
        shape: Vec<usize>,
    },
    MissingKey(String),
    UnexpectedKey(String),
    Shape {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DType {
        key: String,
        expected: DType,
        actual: DType,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3WeightAuditError {
    pub issues: Vec<H3WeightAuditIssue>,
}

impl fmt::Display for H3WeightAuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MiniMax H3 checkpoint audit failed with {} issue(s)",
            self.issues.len()
        )?;
        for issue in self.issues.iter().take(8) {
            write!(formatter, "; {issue:?}")?;
        }
        if self.issues.len() > 8 {
            write!(formatter, "; ... {} more", self.issues.len() - 8)?;
        }
        Ok(())
    }
}

impl StdError for H3WeightAuditError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3LoadPlan {
    task: H3TransformerTask,
    component: H3CheckpointComponent,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
    expected: BTreeMap<String, H3TensorSpec>,
}

/// A single-use audit result that owns the validated architecture and the exact
/// checkpoint backend. It deliberately does not implement `Clone`: a plan
/// cannot be replayed with another component, config, or `VarBuilder`.
pub struct H3AuditedCheckpoint<'a> {
    config: H3TransformerConfig,
    plan: H3LoadPlan,
    source: H3CheckpointSource<'a>,
}

impl fmt::Debug for H3AuditedCheckpoint<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H3AuditedCheckpoint")
            .field("config", &self.config)
            .field("plan", &self.plan)
            .field("source", &self.source)
            .finish()
    }
}

impl H3AuditedCheckpoint<'_> {
    pub fn config(&self) -> &H3TransformerConfig {
        &self.config
    }

    pub fn plan(&self) -> &H3LoadPlan {
        &self.plan
    }
}

impl H3LoadPlan {
    pub const fn task(&self) -> H3TransformerTask {
        self.task
    }

    pub const fn component(&self) -> H3CheckpointComponent {
        self.component
    }

    pub const fn adaln_mode(&self) -> H3AdaLnMode {
        self.mode
    }

    pub const fn qkv_layout(&self) -> H3QkvLayout {
        self.mode.qkv_layout()
    }

    pub const fn precision(&self) -> H3PrecisionProfile {
        self.precision
    }

    pub fn expected_tensors(&self) -> &BTreeMap<String, H3TensorSpec> {
        &self.expected
    }

    /// The correctness loader has no best-effort or ignored tensor list.
    pub fn ignored_keys(&self) -> BTreeSet<&str> {
        BTreeSet::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct H3TransformerConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub token_refiner_num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub ffn_hidden_size: usize,
    pub video_latent_channels: usize,
    pub audio_latent_channels: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub timestep_input_dim: usize,
    pub time_embed_hidden_size: usize,
    pub time_embed_dim: usize,
    pub rope_inv_freq_len: usize,
    pub norm_eps: f64,
    pub qk_norm_eps: f64,
    pub final_norm_eps: f64,
}

impl Default for H3TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 5_376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14_336,
            video_latent_channels: 24,
            audio_latent_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5_120,
            timestep_input_dim: 256,
            time_embed_hidden_size: 5_376,
            time_embed_dim: 2_688,
            rope_inv_freq_len: 16,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }
}

impl H3TransformerConfig {
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("token_refiner_num_layers", self.token_refiner_num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("attention_head_dim", self.attention_head_dim),
            ("ffn_hidden_size", self.ffn_hidden_size),
            ("video_latent_channels", self.video_latent_channels),
            ("audio_latent_channels", self.audio_latent_channels),
            ("text_dim", self.text_dim),
            ("timestep_input_dim", self.timestep_input_dim),
            ("time_embed_hidden_size", self.time_embed_hidden_size),
            ("time_embed_dim", self.time_embed_dim),
            ("rope_inv_freq_len", self.rope_inv_freq_len),
        ];
        for (name, value) in positive {
            if value == 0 {
                candle::bail!("MiniMax H3 {name} must be positive")
            }
        }
        if self.patch_size.contains(&0) {
            candle::bail!("MiniMax H3 patch dimensions must be positive")
        }
        if !self.timestep_input_dim.is_multiple_of(2) {
            candle::bail!("MiniMax H3 timestep_input_dim must be even")
        }
        let checked_product = |name: &str, values: &[usize]| -> Result<usize> {
            values.iter().try_fold(1usize, |product, value| {
                product.checked_mul(*value).ok_or_else(|| {
                    candle::Error::Msg(format!("MiniMax H3 {name} dimension arithmetic overflows"))
                })
            })
        };
        checked_product(
            "attention inner",
            &[self.num_attention_heads, self.attention_head_dim],
        )?;
        checked_product(
            "QKV",
            &[3, self.num_attention_heads, self.attention_head_dim],
        )?;
        checked_product("SwiGLU", &[2, self.ffn_hidden_size])?;
        checked_product(
            "video patch",
            &[
                self.video_latent_channels,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            ],
        )?;
        checked_product("AdaLN", &[6, H3_MODALITY_COUNT, self.hidden_size])?;
        checked_product("final AdaLN", &[2, self.hidden_size])?;
        let rotary_dim = checked_product("rotary", &[6, self.rope_inv_freq_len])?;
        if rotary_dim > self.attention_head_dim {
            candle::bail!(
                "MiniMax H3 rotary width {rotary_dim} exceeds head width {}",
                self.attention_head_dim
            )
        }
        for (name, value) in [
            ("norm_eps", self.norm_eps),
            ("qk_norm_eps", self.qk_norm_eps),
            ("final_norm_eps", self.final_norm_eps),
        ] {
            if !value.is_finite() || value <= 0.0 {
                candle::bail!("MiniMax H3 {name} must be finite and positive")
            }
        }
        Ok(())
    }

    pub const fn attention_inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub const fn video_patch_dim(&self) -> usize {
        self.video_latent_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
    }

    #[cfg(test)]
    fn tiny() -> Self {
        Self {
            hidden_size: 8,
            num_layers: 2,
            token_refiner_num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 8,
            ffn_hidden_size: 12,
            video_latent_channels: 2,
            audio_latent_channels: 3,
            patch_size: [1, 2, 2],
            text_dim: 6,
            timestep_input_dim: 4,
            time_embed_hidden_size: 8,
            time_embed_dim: 4,
            rope_inv_freq_len: 1,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }
}

fn insert_spec(
    specs: &mut BTreeMap<String, H3TensorSpec>,
    name: impl Into<String>,
    shape: impl Into<Vec<usize>>,
    dtype: DType,
) {
    let previous = specs.insert(name.into(), H3TensorSpec::new(shape, dtype));
    debug_assert!(previous.is_none());
}

fn dtype_for(name: &str, mode: H3AdaLnMode, precision: H3PrecisionProfile) -> DType {
    if precision == H3PrecisionProfile::SyntheticF32 {
        return DType::F32;
    }
    if name.starts_with("video_patch_proj.")
        || name.starts_with("audio_patch_proj.")
        || name.starts_with("time_embedder.")
        || name == "rope.inv_freq"
        || name.starts_with("final_layer.video_out.")
        || name.starts_with("final_layer.audio_out.")
        || (matches!(mode, H3AdaLnMode::Curve { .. })
            && (name == "adaln_t_table" || name.contains(".adaln_proj.")))
    {
        DType::F32
    } else {
        DType::BF16
    }
}

fn attention_specs(
    specs: &mut BTreeMap<String, H3TensorSpec>,
    prefix: &str,
    config: &H3TransformerConfig,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
) {
    let inner = config.attention_inner_dim();
    for (suffix, shape) in [
        ("qkv_proj.weight", vec![3 * inner, config.hidden_size]),
        ("q_norm.weight", vec![config.attention_head_dim]),
        ("k_norm.weight", vec![config.attention_head_dim]),
        ("out_proj.weight", vec![config.hidden_size, inner]),
    ] {
        let name = format!("{prefix}.{suffix}");
        insert_spec(specs, &name, shape, dtype_for(&name, mode, precision));
    }
}

fn transformer_block_specs(
    specs: &mut BTreeMap<String, H3TensorSpec>,
    prefix: &str,
    config: &H3TransformerConfig,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
    with_adaln: bool,
) {
    for suffix in ["norm1.weight", "norm2.weight"] {
        let name = format!("{prefix}.{suffix}");
        insert_spec(
            specs,
            &name,
            [config.hidden_size],
            dtype_for(&name, mode, precision),
        );
    }
    attention_specs(specs, &format!("{prefix}.attn"), config, mode, precision);
    for (suffix, shape) in [
        (
            "mlp.fc1.weight",
            vec![2 * config.ffn_hidden_size, config.hidden_size],
        ),
        (
            "mlp.fc2.weight",
            vec![config.hidden_size, config.ffn_hidden_size],
        ),
    ] {
        let name = format!("{prefix}.{suffix}");
        insert_spec(specs, &name, shape, dtype_for(&name, mode, precision));
    }
    if with_adaln {
        let width = 6 * H3_MODALITY_COUNT * config.hidden_size;
        for (suffix, shape) in [
            (
                "adaln_proj.linear.weight",
                vec![width, mode.input_dim(config)],
            ),
            ("adaln_proj.linear.bias", vec![width]),
        ] {
            let name = format!("{prefix}.{suffix}");
            insert_spec(specs, &name, shape, dtype_for(&name, mode, precision));
        }
    }
}

pub fn expected_h3_weight_specs(
    config: &H3TransformerConfig,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
) -> Result<BTreeMap<String, H3TensorSpec>> {
    config.validate()?;
    if let H3AdaLnMode::Curve { grid, basis_dim } = mode {
        if grid < 2 || basis_dim == 0 {
            candle::bail!("MiniMax H3 curve mode requires grid >= 2 and a positive basis width")
        }
    }
    let mut specs = BTreeMap::new();
    for (name, shape) in [
        (
            "video_patch_proj.weight",
            vec![config.hidden_size, config.video_patch_dim()],
        ),
        ("video_patch_proj.bias", vec![config.hidden_size]),
        (
            "audio_patch_proj.weight",
            vec![config.hidden_size, config.audio_latent_channels],
        ),
        ("audio_patch_proj.bias", vec![config.hidden_size]),
        (
            "condition_proj.weight",
            vec![config.hidden_size, config.text_dim],
        ),
        ("condition_proj.bias", vec![config.hidden_size]),
    ] {
        insert_spec(&mut specs, name, shape, dtype_for(name, mode, precision));
    }

    match mode {
        H3AdaLnMode::Full => {
            for (name, shape) in [
                (
                    "time_embedder.proj_in.weight",
                    vec![config.time_embed_hidden_size, config.timestep_input_dim],
                ),
                (
                    "time_embedder.proj_in.bias",
                    vec![config.time_embed_hidden_size],
                ),
                (
                    "time_embedder.proj_out.weight",
                    vec![config.time_embed_dim, config.time_embed_hidden_size],
                ),
                ("time_embedder.proj_out.bias", vec![config.time_embed_dim]),
            ] {
                insert_spec(&mut specs, name, shape, dtype_for(name, mode, precision));
            }
        }
        H3AdaLnMode::Curve { grid, basis_dim } => insert_spec(
            &mut specs,
            "adaln_t_table",
            [grid, basis_dim],
            dtype_for("adaln_t_table", mode, precision),
        ),
    }
    insert_spec(
        &mut specs,
        "rope.inv_freq",
        [config.rope_inv_freq_len],
        dtype_for("rope.inv_freq", mode, precision),
    );

    for index in 0..config.token_refiner_num_layers {
        transformer_block_specs(
            &mut specs,
            &format!("token_refiner.blocks.{index}"),
            config,
            mode,
            precision,
            false,
        );
    }
    insert_spec(
        &mut specs,
        "token_refiner.final_norm.weight",
        [config.hidden_size],
        dtype_for("token_refiner.final_norm.weight", mode, precision),
    );

    for index in 0..config.num_layers {
        transformer_block_specs(
            &mut specs,
            &format!("blocks.{index}"),
            config,
            mode,
            precision,
            true,
        );
    }

    insert_spec(
        &mut specs,
        "final_layer.norm.weight",
        [config.hidden_size],
        dtype_for("final_layer.norm.weight", mode, precision),
    );
    let final_width = 2 * config.hidden_size;
    for (name, shape) in [
        (
            "final_layer.adaln_proj.linear.weight",
            vec![final_width, mode.input_dim(config)],
        ),
        ("final_layer.adaln_proj.linear.bias", vec![final_width]),
        (
            "final_layer.video_out.weight",
            vec![config.video_patch_dim(), config.hidden_size],
        ),
        ("final_layer.video_out.bias", vec![config.video_patch_dim()]),
        (
            "final_layer.audio_out.weight",
            vec![config.audio_latent_channels, config.hidden_size],
        ),
        (
            "final_layer.audio_out.bias",
            vec![config.audio_latent_channels],
        ),
    ] {
        insert_spec(&mut specs, name, shape, dtype_for(name, mode, precision));
    }
    Ok(specs)
}

pub fn audit_h3_checkpoint<'a>(
    config: H3TransformerConfig,
    task: H3TransformerTask,
    precision: H3PrecisionProfile,
    source: H3CheckpointSource<'a>,
) -> std::result::Result<H3AuditedCheckpoint<'a>, H3WeightAuditError> {
    let mut issues = Vec::new();
    if let Err(error) = config.validate() {
        return Err(H3WeightAuditError {
            issues: vec![H3WeightAuditIssue::InvalidConfig(error.to_string())],
        });
    }
    let inventory = &source.inventory;
    let expected_component = task.checkpoint_component();
    if inventory.component != expected_component {
        issues.push(H3WeightAuditIssue::WrongComponent {
            expected: expected_component,
            actual: inventory.component,
        });
    }

    let has_curve = inventory.tensors.contains_key("adaln_t_table");
    let full_keys = [
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
    ];
    let full_count = full_keys
        .iter()
        .filter(|key| inventory.tensors.contains_key(**key))
        .count();
    let mode = if has_curve && full_count != 0 {
        issues.push(H3WeightAuditIssue::AmbiguousAdaLnRepresentation);
        H3AdaLnMode::Full
    } else if has_curve {
        let shape = &inventory.tensors["adaln_t_table"].shape;
        if shape.len() == 2 && shape[0] >= 2 && shape[1] > 0 {
            H3AdaLnMode::Curve {
                grid: shape[0],
                basis_dim: shape[1],
            }
        } else {
            issues.push(H3WeightAuditIssue::InvalidCurveTable {
                shape: shape.clone(),
            });
            H3AdaLnMode::Curve {
                grid: shape.first().copied().unwrap_or(0).max(2),
                basis_dim: shape.get(1).copied().unwrap_or(0).max(1),
            }
        }
    } else if full_count == full_keys.len() {
        H3AdaLnMode::Full
    } else {
        issues.push(H3WeightAuditIssue::MissingAdaLnRepresentation);
        H3AdaLnMode::Full
    };

    let expected = match expected_h3_weight_specs(&config, mode, precision) {
        Ok(expected) => expected,
        Err(error) => {
            issues.push(H3WeightAuditIssue::InvalidConfig(error.to_string()));
            return Err(H3WeightAuditError { issues });
        }
    };
    for (key, expected_spec) in &expected {
        match inventory.tensors.get(key) {
            None => issues.push(H3WeightAuditIssue::MissingKey(key.clone())),
            Some(actual) => {
                if actual.shape != expected_spec.shape {
                    issues.push(H3WeightAuditIssue::Shape {
                        key: key.clone(),
                        expected: expected_spec.shape.clone(),
                        actual: actual.shape.clone(),
                    });
                }
                if actual.dtype != expected_spec.dtype {
                    issues.push(H3WeightAuditIssue::DType {
                        key: key.clone(),
                        expected: expected_spec.dtype,
                        actual: actual.dtype,
                    });
                }
            }
        }
    }
    for key in inventory.tensors.keys() {
        if !expected.contains_key(key) {
            issues.push(H3WeightAuditIssue::UnexpectedKey(key.clone()));
        }
    }

    if issues.is_empty() {
        Ok(H3AuditedCheckpoint {
            config,
            plan: H3LoadPlan {
                task,
                component: inventory.component,
                mode,
                precision,
                expected,
            },
            source,
        })
    } else {
        Err(H3WeightAuditError { issues })
    }
}

#[derive(Clone, Debug)]
pub struct H3PackedLayout {
    position_ids: Vec<[f32; 3]>,
    timestep_indices: Vec<u32>,
    token_tags: Vec<u32>,
    video_indices: Vec<u32>,
    audio_indices: Vec<u32>,
    text_indices: Vec<u32>,
    max_timestep_index: usize,
}

impl H3PackedLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        position_ids: Vec<[f32; 3]>,
        timestep_indices: Vec<u32>,
        token_tags: Vec<u32>,
        video_indices: Vec<u32>,
        audio_indices: Vec<u32>,
        text_indices: Vec<u32>,
    ) -> Result<Self> {
        let seq_len = position_ids.len();
        if seq_len == 0 {
            candle::bail!("MiniMax H3 packed layout cannot be empty")
        }
        if timestep_indices.len() != seq_len || token_tags.len() != seq_len {
            candle::bail!(
                "MiniMax H3 positions, timestep indices, and token tags must have one entry per row"
            )
        }
        if video_indices.is_empty() || audio_indices.is_empty() || text_indices.is_empty() {
            candle::bail!("MiniMax H3 packed layout requires video, audio, and text rows")
        }
        if position_ids
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            candle::bail!("MiniMax H3 packed positions must be finite")
        }
        if token_tags
            .iter()
            .any(|tag| *tag >= H3_MODALITY_COUNT as u32)
        {
            candle::bail!("MiniMax H3 token tags must be video=0, text=1, or audio=2")
        }
        if timestep_indices
            .iter()
            .any(|index| *index > u32::MAX / H3_MODALITY_COUNT as u32)
        {
            candle::bail!("MiniMax H3 timestep index overflows the AdaLN row address")
        }

        let mut owner = vec![None; seq_len];
        let mut claim = |indices: &[u32], name: &'static str| -> Result<()> {
            for &index in indices {
                let index = index as usize;
                if index >= seq_len {
                    candle::bail!(
                        "MiniMax H3 {name} row index {index} is outside seq_len {seq_len}"
                    )
                }
                if let Some(previous) = owner[index] {
                    candle::bail!(
                        "MiniMax H3 packed row {index} is claimed by both {previous} and {name}"
                    )
                }
                owner[index] = Some(name);
            }
            Ok(())
        };
        claim(&video_indices, "video")?;
        claim(&audio_indices, "audio")?;
        claim(&text_indices, "text")?;
        if let Some(index) = owner.iter().position(Option::is_none) {
            candle::bail!("MiniMax H3 packed row {index} has no projection source")
        }

        for &index in &video_indices {
            if token_tags[index as usize] != H3Modality::Video as u32 {
                candle::bail!("MiniMax H3 projected video rows must carry modality tag 0")
            }
        }
        for &index in &audio_indices {
            if token_tags[index as usize] != H3Modality::Audio as u32 {
                candle::bail!("MiniMax H3 projected audio rows must carry modality tag 2")
            }
        }
        // Qwen vision-pad tokens are projected by the text path but deliberately
        // carry the video AdaLN tag. Audio is never valid in the text span.
        for &index in &text_indices {
            if token_tags[index as usize] == H3Modality::Audio as u32 {
                candle::bail!("MiniMax H3 projected text rows cannot carry the audio tag")
            }
        }
        let max_timestep_index = timestep_indices.iter().copied().max().unwrap_or(0) as usize;
        Ok(Self {
            position_ids,
            timestep_indices,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            max_timestep_index,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.position_ids.len()
    }

    /// Number of dense attention-score elements before dtype byte width.
    pub fn attention_score_elements(&self, batch_size: usize, heads: usize) -> Result<usize> {
        batch_size
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(self.seq_len()))
            .and_then(|value| value.checked_mul(self.seq_len()))
            .ok_or_else(|| candle::Error::Msg("MiniMax H3 attention estimate overflow".into()))
    }

    pub fn freeze(&self, device: &Device) -> Result<H3FrozenPackedLayout> {
        let position_ids = Tensor::from_slice(
            &self
                .position_ids
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (self.seq_len(), 3),
            device,
        )?;
        let timestep_indices =
            Tensor::from_slice(&self.timestep_indices, self.timestep_indices.len(), device)?;
        let adaln_indices = self
            .timestep_indices
            .iter()
            .zip(&self.token_tags)
            .map(|(&timestep, &tag)| timestep * H3_MODALITY_COUNT as u32 + tag)
            .collect::<Vec<_>>();
        Ok(H3FrozenPackedLayout {
            seq_len: self.seq_len(),
            position_ids,
            timestep_indices,
            adaln_indices: Tensor::from_slice(&adaln_indices, adaln_indices.len(), device)?,
            video_indices: Tensor::from_slice(
                &self.video_indices,
                self.video_indices.len(),
                device,
            )?,
            audio_indices: Tensor::from_slice(
                &self.audio_indices,
                self.audio_indices.len(),
                device,
            )?,
            text_indices: Tensor::from_slice(&self.text_indices, self.text_indices.len(), device)?,
            video_rows: self.video_indices.len(),
            audio_rows: self.audio_indices.len(),
            text_rows: self.text_indices.len(),
            max_timestep_index: self.max_timestep_index,
        })
    }
}

#[derive(Clone, Debug)]
pub struct H3FrozenPackedLayout {
    seq_len: usize,
    position_ids: Tensor,
    timestep_indices: Tensor,
    adaln_indices: Tensor,
    video_indices: Tensor,
    audio_indices: Tensor,
    text_indices: Tensor,
    video_rows: usize,
    audio_rows: usize,
    text_rows: usize,
    max_timestep_index: usize,
}

impl H3FrozenPackedLayout {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn position_ids(&self) -> &Tensor {
        &self.position_ids
    }
}

/// `[B,C,T,H,W] -> [B,T*H/PH*W/PW,C*PT*PH*PW]` in temporal,
/// height, width, channel, and patch-offset order.
pub fn patchify_h3_video(latent: &Tensor, patch_size: [usize; 3]) -> Result<Tensor> {
    let [b, c, t_full, h_full, w_full]: [usize; 5] = latent
        .dims()
        .try_into()
        .map_err(|_| candle::Error::Msg("MiniMax H3 video latent must be rank 5".into()))?;
    let [pt, ph, pw] = patch_size;
    if pt == 0 || ph == 0 || pw == 0 {
        candle::bail!("MiniMax H3 patch dimensions must be positive")
    }
    if t_full % pt != 0 || h_full % ph != 0 || w_full % pw != 0 {
        candle::bail!(
            "MiniMax H3 video shape [{t_full},{h_full},{w_full}] is not divisible by patch [{pt},{ph},{pw}]"
        )
    }
    let (t, h, w) = (t_full / pt, h_full / ph, w_full / pw);
    latent
        .reshape(&[b, c, t, pt, h, ph, w, pw])?
        .permute([0, 2, 4, 6, 1, 3, 5, 7])?
        .contiguous()?
        .reshape((b, t * h * w, c * pt * ph * pw))
}

pub fn unpatchify_h3_video(
    rows: &Tensor,
    latent_grid: [usize; 3],
    channels: usize,
    patch_size: [usize; 3],
) -> Result<Tensor> {
    let (b, row_count, row_width) = rows.dims3()?;
    let [t, h, w] = latent_grid;
    let [pt, ph, pw] = patch_size;
    if row_count != t * h * w || row_width != channels * pt * ph * pw {
        candle::bail!("MiniMax H3 video rows do not match the requested latent grid")
    }
    rows.reshape(&[b, t, h, w, channels, pt, ph, pw])?
        .permute([0, 4, 1, 5, 2, 6, 3, 7])?
        .contiguous()?
        .reshape((b, channels, t * pt, h * ph, w * pw))
}

/// `[B,C,channels,T] -> [B,channels*T,C]`, preserving channel-major rows.
pub fn pack_h3_audio(latent: &Tensor) -> Result<Tensor> {
    let (b, c, channels, t) = latent.dims4()?;
    latent
        .permute((0, 2, 3, 1))?
        .contiguous()?
        .reshape((b, channels * t, c))
}

pub fn unpack_h3_audio(rows: &Tensor, channels: usize) -> Result<Tensor> {
    let (b, row_count, c) = rows.dims3()?;
    if channels == 0 || row_count % channels != 0 {
        candle::bail!("MiniMax H3 audio row count must be divisible by the channel count")
    }
    let t = row_count / channels;
    rows.reshape((b, channels, t, c))?
        .permute((0, 3, 1, 2))?
        .contiguous()
}

fn silu(tensor: &Tensor) -> Result<Tensor> {
    tensor.broadcast_mul(&nn::ops::sigmoid(tensor)?)
}

fn linear_with_dtype(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    dtype: DType,
    vb: VarBuilder,
) -> Result<nn::Linear> {
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        nn::init::DEFAULT_KAIMING_NORMAL,
        dtype,
    )?;
    let bias = if bias {
        Some(vb.get_with_hints_dtype(out_dim, "bias", nn::Init::Const(0.0), dtype)?)
    } else {
        None
    };
    Ok(nn::Linear::new(weight, bias))
}

fn rms_norm_with_dtype(size: usize, eps: f64, dtype: DType, vb: VarBuilder) -> Result<nn::RmsNorm> {
    let weight = vb.get_with_hints_dtype(size, "weight", nn::Init::Const(1.0), dtype)?;
    Ok(nn::RmsNorm::new(weight, eps))
}

/// Convert the official per-head `[Q,K,V]` row grouping into the Q-major,
/// K-major, V-major representation consumed by the attention implementation.
pub fn reorder_h3_grouped_qkv(weight: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    if weight.rank() < 1 || weight.dim(0)? != 3 * heads * head_dim {
        candle::bail!("MiniMax H3 grouped QKV weight has an incompatible output dimension")
    }
    let mut grouped_shape = vec![heads, 3, head_dim];
    grouped_shape.extend_from_slice(&weight.dims()[1..]);
    let grouped = weight.reshape(grouped_shape)?;
    let mut part_shape = vec![heads * head_dim];
    part_shape.extend_from_slice(&weight.dims()[1..]);
    let q = grouped.narrow(1, 0, 1)?.squeeze(1)?.reshape(&*part_shape)?;
    let k = grouped.narrow(1, 1, 1)?.squeeze(1)?.reshape(&*part_shape)?;
    let v = grouped.narrow(1, 2, 1)?.squeeze(1)?.reshape(&*part_shape)?;
    Tensor::cat(&[q, k, v], 0)
}

fn qkv_linear(
    config: &H3TransformerConfig,
    qkv_layout: H3QkvLayout,
    dtype: DType,
    vb: VarBuilder,
) -> Result<nn::Linear> {
    let inner = config.attention_inner_dim();
    let mut weight = vb.get_with_hints_dtype(
        (3 * inner, config.hidden_size),
        "weight",
        nn::init::DEFAULT_KAIMING_NORMAL,
        dtype,
    )?;
    if qkv_layout == H3QkvLayout::GroupedPerHead {
        weight = reorder_h3_grouped_qkv(
            &weight,
            config.num_attention_heads,
            config.attention_head_dim,
        )?;
    }
    Ok(nn::Linear::new(weight, None))
}

fn apply_h3_rotary(
    hidden: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<Tensor> {
    let (b, seq_len, heads, head_dim) = hidden.dims4()?;
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        candle::bail!("MiniMax H3 rotary width is incompatible with the attention head")
    }
    let dtype = hidden.dtype();
    let rotary = hidden.narrow(3, 0, rotary_dim)?;
    let half = rotary_dim / 2;
    let first = rotary.narrow(3, 0, half)?;
    let second = rotary.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[second.neg()?, first], 3)?;
    let cos = cos.to_dtype(dtype)?.reshape((1, seq_len, 1, rotary_dim))?;
    let sin = sin.to_dtype(dtype)?.reshape((1, seq_len, 1, rotary_dim))?;
    let rotary = rotary
        .broadcast_mul(&cos)?
        .broadcast_add(&rotated.broadcast_mul(&sin)?)?;
    if rotary_dim == head_dim {
        rotary.contiguous()
    } else {
        let tail = hidden.narrow(3, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[rotary, tail], 3)?
            .reshape((b, seq_len, heads, head_dim))?
            .contiguous()
    }
}

#[derive(Clone, Debug)]
struct H3Rope {
    inv_freq: Tensor,
}

impl H3Rope {
    fn load(config: &H3TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inv_freq: vb.get_with_hints_dtype(
                config.rope_inv_freq_len,
                "inv_freq",
                nn::Init::Const(1.0),
                DType::F32,
            )?,
        })
    }

    fn forward(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let (seq_len, axes) = positions.dims2()?;
        if axes != 3 {
            candle::bail!("MiniMax H3 position_ids must have shape [seq_len,3]")
        }
        let per_axis = positions
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_mul(&self.inv_freq.reshape((1, 1, ()))?)?;
        let temporal = per_axis.i((.., 0, ..))?;
        let height = per_axis.i((.., 1, ..))?;
        let width = per_axis.i((.., 2, ..))?;
        let half = Tensor::cat(&[temporal, height, width], 1)?;
        let frequencies = Tensor::cat(&[half.clone(), half], 1)?;
        debug_assert_eq!(frequencies.dim(0)?, seq_len);
        Ok((frequencies.cos()?, frequencies.sin()?))
    }
}

#[derive(Clone, Debug)]
struct H3Attention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    qkv: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    out: nn::Linear,
}

impl H3Attention {
    fn load(
        config: &H3TransformerConfig,
        qkv_layout: H3QkvLayout,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = config.attention_inner_dim();
        Ok(Self {
            heads: config.num_attention_heads,
            head_dim: config.attention_head_dim,
            inner_dim,
            qkv: qkv_linear(config, qkv_layout, dtype, vb.pp("qkv_proj"))?,
            q_norm: rms_norm_with_dtype(
                config.attention_head_dim,
                config.qk_norm_eps,
                dtype,
                vb.pp("q_norm"),
            )?,
            k_norm: rms_norm_with_dtype(
                config.attention_head_dim,
                config.qk_norm_eps,
                dtype,
                vb.pp("k_norm"),
            )?,
            out: linear_with_dtype(
                inner_dim,
                config.hidden_size,
                false,
                dtype,
                vb.pp("out_proj"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rotary: Option<(&Tensor, &Tensor, usize)>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden.dims3()?;
        let qkv = self.qkv.forward(hidden)?;
        let q = qkv.narrow(2, 0, self.inner_dim)?;
        let k = qkv.narrow(2, self.inner_dim, self.inner_dim)?;
        let v = qkv.narrow(2, 2 * self.inner_dim, self.inner_dim)?;

        let mut q = q
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .contiguous()?;
        let mut k = k
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .contiguous()?;
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        if let Some((cos, sin, rotary_dim)) = rotary {
            q = apply_h3_rotary(&q, cos, sin, rotary_dim)?;
            k = apply_h3_rotary(&k, cos, sin, rotary_dim)?;
        }

        // Full, non-causal attention over one packed document. The F32 score
        // path is the correctness backend; fused/chunked policies live above
        // this primitive and must be selected explicitly.
        let q = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let probabilities = nn::ops::softmax(&scores, D::Minus1)?;
        let output = probabilities
            .matmul(&v)?
            .to_dtype(hidden.dtype())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.inner_dim))?;
        self.out.forward(&output)
    }
}

#[derive(Clone, Debug)]
struct H3Mlp {
    width: usize,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl H3Mlp {
    fn load(config: &H3TransformerConfig, dtype: DType, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            width: config.ffn_hidden_size,
            fc1: linear_with_dtype(
                config.hidden_size,
                2 * config.ffn_hidden_size,
                false,
                dtype,
                vb.pp("fc1"),
            )?,
            fc2: linear_with_dtype(
                config.ffn_hidden_size,
                config.hidden_size,
                false,
                dtype,
                vb.pp("fc2"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let projected = self.fc1.forward(hidden)?;
        let gate = projected.narrow(D::Minus1, 0, self.width)?;
        let up = projected.narrow(D::Minus1, self.width, self.width)?;
        self.fc2.forward(&silu(&gate)?.broadcast_mul(&up)?)
    }
}

#[derive(Clone, Debug)]
struct H3TokenRefinerBlock {
    norm1: nn::RmsNorm,
    attention: H3Attention,
    norm2: nn::RmsNorm,
    mlp: H3Mlp,
}

impl H3TokenRefinerBlock {
    fn load(
        config: &H3TransformerConfig,
        qkv_layout: H3QkvLayout,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm_with_dtype(config.hidden_size, config.norm_eps, dtype, vb.pp("norm1"))?,
            attention: H3Attention::load(config, qkv_layout, dtype, vb.pp("attn"))?,
            norm2: rms_norm_with_dtype(config.hidden_size, config.norm_eps, dtype, vb.pp("norm2"))?,
            mlp: H3Mlp::load(config, dtype, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.norm1.forward(hidden)?, None)?;
        let hidden = hidden.broadcast_add(&attention)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&hidden)?)?;
        hidden.broadcast_add(&mlp)
    }
}

#[derive(Clone, Debug)]
struct H3AdaLnProjection {
    hidden_size: usize,
    chunks: usize,
    modalities: usize,
    linear: nn::Linear,
}

impl H3AdaLnProjection {
    #[allow(clippy::too_many_arguments)]
    fn load(
        input_dim: usize,
        hidden_size: usize,
        chunks: usize,
        modalities: usize,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            hidden_size,
            chunks,
            modalities,
            linear: linear_with_dtype(
                input_dim,
                chunks * modalities * hidden_size,
                true,
                dtype,
                vb.pp("linear"),
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let count = input.dim(0)?;
        self.linear
            .forward(&input.to_dtype(self.linear.weight().dtype())?)?
            .reshape((count * self.modalities, self.chunks * self.hidden_size))?
            .chunk(self.chunks, 1)
    }
}

fn select_modulation(table: &Tensor, indices: &Tensor, dtype: DType) -> Result<Tensor> {
    table
        .contiguous()?
        .index_select(indices, 0)?
        .to_dtype(dtype)?
        .unsqueeze(0)
}

fn modulate(hidden: &Tensor, shift: &Tensor, scale: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let shift = select_modulation(shift, indices, hidden.dtype())?;
    let scale = select_modulation(scale, indices, hidden.dtype())?.affine(1.0, 1.0)?;
    hidden.broadcast_mul(&scale)?.broadcast_add(&shift)
}

fn gated_residual(
    residual: &Tensor,
    update: &Tensor,
    gate: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    let gate = select_modulation(gate, indices, residual.dtype())?;
    residual.broadcast_add(&update.broadcast_mul(&gate)?)
}

#[derive(Clone, Debug)]
struct H3TransformerBlock {
    norm1: nn::RmsNorm,
    attention: H3Attention,
    norm2: nn::RmsNorm,
    mlp: H3Mlp,
    adaln: H3AdaLnProjection,
}

impl H3TransformerBlock {
    fn load(
        config: &H3TransformerConfig,
        mode: H3AdaLnMode,
        precision: H3PrecisionProfile,
        vb: VarBuilder,
    ) -> Result<Self> {
        let compute_dtype = precision.compute_dtype();
        let adaln_dtype = if matches!(mode, H3AdaLnMode::Curve { .. }) {
            DType::F32
        } else {
            compute_dtype
        };
        Ok(Self {
            norm1: rms_norm_with_dtype(
                config.hidden_size,
                config.norm_eps,
                compute_dtype,
                vb.pp("norm1"),
            )?,
            attention: H3Attention::load(config, mode.qkv_layout(), compute_dtype, vb.pp("attn"))?,
            norm2: rms_norm_with_dtype(
                config.hidden_size,
                config.norm_eps,
                compute_dtype,
                vb.pp("norm2"),
            )?,
            mlp: H3Mlp::load(config, compute_dtype, vb.pp("mlp"))?,
            adaln: H3AdaLnProjection::load(
                mode.input_dim(config),
                config.hidden_size,
                6,
                H3_MODALITY_COUNT,
                adaln_dtype,
                vb.pp("adaln_proj"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        adaln_input: &Tensor,
        adaln_indices: &Tensor,
        rotary: (&Tensor, &Tensor, usize),
    ) -> Result<Tensor> {
        let parameters = self.adaln.forward(adaln_input)?;
        let [shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp]:
            [Tensor; 6] = parameters.try_into().map_err(|_| {
                candle::Error::Msg("MiniMax H3 block AdaLN must produce six tensors".into())
            })?;
        let normalized = modulate(
            &self.norm1.forward(hidden)?,
            &shift_attention,
            &scale_attention,
            adaln_indices,
        )?;
        let update = self.attention.forward(&normalized, Some(rotary))?;
        let hidden = gated_residual(hidden, &update, &gate_attention, adaln_indices)?;
        let normalized = modulate(
            &self.norm2.forward(&hidden)?,
            &shift_mlp,
            &scale_mlp,
            adaln_indices,
        )?;
        let update = self.mlp.forward(&normalized)?;
        gated_residual(&hidden, &update, &gate_mlp, adaln_indices)
    }
}

#[derive(Clone, Debug)]
enum H3TimeConditioner {
    Full {
        frequency_dim: usize,
        proj_in: nn::Linear,
        proj_out: nn::Linear,
    },
    Curve {
        table: Tensor,
    },
}

impl H3TimeConditioner {
    fn load(config: &H3TransformerConfig, mode: H3AdaLnMode, vb: VarBuilder) -> Result<Self> {
        match mode {
            H3AdaLnMode::Full => Ok(Self::Full {
                frequency_dim: config.timestep_input_dim,
                proj_in: linear_with_dtype(
                    config.timestep_input_dim,
                    config.time_embed_hidden_size,
                    true,
                    DType::F32,
                    vb.pp("time_embedder.proj_in"),
                )?,
                proj_out: linear_with_dtype(
                    config.time_embed_hidden_size,
                    config.time_embed_dim,
                    true,
                    DType::F32,
                    vb.pp("time_embedder.proj_out"),
                )?,
            }),
            H3AdaLnMode::Curve { grid, basis_dim } => Ok(Self::Curve {
                table: vb.get_with_hints_dtype(
                    (grid, basis_dim),
                    "adaln_t_table",
                    nn::Init::Const(0.0),
                    DType::F32,
                )?,
            }),
        }
    }

    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let count = timesteps.dims1()?;
        if count == 0 {
            candle::bail!("MiniMax H3 requires at least one distinct timestep")
        }
        let timesteps = timesteps.to_dtype(DType::F32)?;
        match self {
            Self::Full {
                frequency_dim,
                proj_in,
                proj_out,
            } => {
                let half = frequency_dim / 2;
                let indices =
                    Tensor::arange(0u32, half as u32, timesteps.device())?.to_dtype(DType::F32)?;
                let frequencies = indices.affine(-10_000f64.ln() / half as f64, 0.0)?.exp()?;
                let arguments = timesteps
                    .unsqueeze(1)?
                    .broadcast_mul(&frequencies.unsqueeze(0)?)?;
                let embedding = Tensor::cat(&[arguments.cos()?, arguments.sin()?], 1)?;
                proj_out.forward(&silu(&proj_in.forward(&embedding)?)?)
            }
            Self::Curve { table } => {
                let grid = table.dim(0)?;
                let position = timesteps.clamp(0.0, 1.0)?.affine(grid as f64 - 1.0, 0.0)?;
                let lower_f32 = position.floor()?.clamp(0.0, grid as f64 - 2.0)?;
                let upper_f32 = lower_f32.affine(1.0, 1.0)?;
                let lower = lower_f32.to_dtype(DType::U32)?;
                let upper = upper_f32.to_dtype(DType::U32)?;
                let fraction = position.broadcast_sub(&lower_f32)?.unsqueeze(1)?;
                let first = table.index_select(&lower, 0)?;
                let second = table.index_select(&upper, 0)?;
                first
                    .broadcast_mul(&fraction.affine(-1.0, 1.0)?)?
                    .broadcast_add(&second.broadcast_mul(&fraction)?)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct H3FinalLayer {
    norm: nn::RmsNorm,
    adaln: H3AdaLnProjection,
    video_out: nn::Linear,
    audio_out: nn::Linear,
}

impl H3FinalLayer {
    fn load(
        config: &H3TransformerConfig,
        mode: H3AdaLnMode,
        precision: H3PrecisionProfile,
        vb: VarBuilder,
    ) -> Result<Self> {
        let compute_dtype = precision.compute_dtype();
        let adaln_dtype = if matches!(mode, H3AdaLnMode::Curve { .. }) {
            DType::F32
        } else {
            compute_dtype
        };
        Ok(Self {
            norm: rms_norm_with_dtype(
                config.hidden_size,
                config.final_norm_eps,
                compute_dtype,
                vb.pp("norm"),
            )?,
            adaln: H3AdaLnProjection::load(
                mode.input_dim(config),
                config.hidden_size,
                2,
                1,
                adaln_dtype,
                vb.pp("adaln_proj"),
            )?,
            video_out: linear_with_dtype(
                config.hidden_size,
                config.video_patch_dim(),
                true,
                DType::F32,
                vb.pp("video_out"),
            )?,
            audio_out: linear_with_dtype(
                config.hidden_size,
                config.audio_latent_channels,
                true,
                DType::F32,
                vb.pp("audio_out"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        adaln_input: &Tensor,
        timestep_indices: &Tensor,
        video_indices: &Tensor,
        audio_indices: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let parameters = self.adaln.forward(adaln_input)?;
        let [shift, scale]: [Tensor; 2] = parameters.try_into().map_err(|_| {
            candle::Error::Msg("MiniMax H3 final AdaLN must produce two tensors".into())
        })?;
        let hidden = modulate(
            &self.norm.forward(hidden)?,
            &shift,
            &scale,
            timestep_indices,
        )?
        .to_dtype(DType::F32)?;
        // Run both heads over every packed row before selection. This preserves
        // the released GEMM and rounding contract, including dead text rows.
        let video = self
            .video_out
            .forward(&hidden)?
            .index_select(video_indices, 1)?;
        let audio = self
            .audio_out
            .forward(&hidden)?
            .index_select(audio_indices, 1)?;
        Ok((video, audio))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3BlockStack {
    TokenRefiner,
    Transformer,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H3BlockProgress {
    pub stack: H3BlockStack,
    pub completed: usize,
    pub total: usize,
}

pub struct H3ForwardInput<'a> {
    pub video_rows: &'a Tensor,
    pub audio_rows: &'a Tensor,
    pub text_states: &'a Tensor,
    pub timesteps: &'a Tensor,
}

#[derive(Clone, Debug)]
pub struct H3TransformerOutput {
    pub video: Tensor,
    pub audio: Tensor,
}

#[derive(Clone, Debug)]
pub struct H3Transformer {
    config: H3TransformerConfig,
    task: H3TransformerTask,
    mode: H3AdaLnMode,
    precision: H3PrecisionProfile,
    video_projection: nn::Linear,
    audio_projection: nn::Linear,
    text_projection: nn::Linear,
    time: H3TimeConditioner,
    rope: H3Rope,
    token_refiner: Vec<H3TokenRefinerBlock>,
    token_refiner_norm: nn::RmsNorm,
    blocks: Vec<H3TransformerBlock>,
    final_layer: H3FinalLayer,
}

impl H3Transformer {
    pub fn load(checkpoint: H3AuditedCheckpoint<'_>) -> Result<Self> {
        let H3AuditedCheckpoint {
            config,
            plan,
            source,
        } = checkpoint;
        debug_assert_eq!(source.inventory.component, plan.component);
        let vb = source.vb;
        for key in plan.expected.keys() {
            if !vb.contains_tensor(key) {
                candle::bail!("MiniMax H3 audited tensor {key:?} is absent from the loader")
            }
        }
        let compute_dtype = plan.precision.compute_dtype();
        let mut token_refiner = Vec::with_capacity(config.token_refiner_num_layers);
        for index in 0..config.token_refiner_num_layers {
            token_refiner.push(H3TokenRefinerBlock::load(
                &config,
                plan.qkv_layout(),
                compute_dtype,
                vb.pp(format!("token_refiner.blocks.{index}")),
            )?);
        }
        let mut blocks = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            blocks.push(H3TransformerBlock::load(
                &config,
                plan.mode,
                plan.precision,
                vb.pp(format!("blocks.{index}")),
            )?);
        }
        Ok(Self {
            task: plan.task,
            mode: plan.mode,
            precision: plan.precision,
            video_projection: linear_with_dtype(
                config.video_patch_dim(),
                config.hidden_size,
                true,
                DType::F32,
                vb.pp("video_patch_proj"),
            )?,
            audio_projection: linear_with_dtype(
                config.audio_latent_channels,
                config.hidden_size,
                true,
                DType::F32,
                vb.pp("audio_patch_proj"),
            )?,
            text_projection: linear_with_dtype(
                config.text_dim,
                config.hidden_size,
                true,
                compute_dtype,
                vb.pp("condition_proj"),
            )?,
            time: H3TimeConditioner::load(&config, plan.mode, vb.clone())?,
            rope: H3Rope::load(&config, vb.pp("rope"))?,
            token_refiner,
            token_refiner_norm: rms_norm_with_dtype(
                config.hidden_size,
                config.final_norm_eps,
                compute_dtype,
                vb.pp("token_refiner.final_norm"),
            )?,
            blocks,
            final_layer: H3FinalLayer::load(
                &config,
                plan.mode,
                plan.precision,
                vb.pp("final_layer"),
            )?,
            config,
        })
    }

    pub const fn task(&self) -> H3TransformerTask {
        self.task
    }

    pub const fn adaln_mode(&self) -> H3AdaLnMode {
        self.mode
    }

    pub const fn precision(&self) -> H3PrecisionProfile {
        self.precision
    }

    pub fn forward(
        &self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
    ) -> Result<H3TransformerOutput> {
        self.forward_with_observer(input, layout, |_| Ok(()))
    }

    pub fn forward_with_observer<F>(
        &self,
        input: H3ForwardInput<'_>,
        layout: &H3FrozenPackedLayout,
        mut observer: F,
    ) -> Result<H3TransformerOutput>
    where
        F: FnMut(H3BlockProgress) -> Result<()>,
    {
        let (batch, video_rows, video_width) = input.video_rows.dims3()?;
        let (audio_batch, audio_rows, audio_width) = input.audio_rows.dims3()?;
        let (text_batch, text_rows, text_width) = input.text_states.dims3()?;
        if batch == 0 || audio_batch != batch || text_batch != batch {
            candle::bail!("MiniMax H3 modalities must share a non-empty batch axis")
        }
        if video_rows != layout.video_rows
            || audio_rows != layout.audio_rows
            || text_rows != layout.text_rows
            || video_width != self.config.video_patch_dim()
            || audio_width != self.config.audio_latent_channels
            || text_width != self.config.text_dim
        {
            candle::bail!("MiniMax H3 modality rows do not match the frozen packed layout/config")
        }
        if layout.max_timestep_index >= input.timesteps.dims1()? {
            candle::bail!("MiniMax H3 packed layout addresses a missing distinct timestep")
        }
        let device = input.video_rows.device();
        if !device.same_device(input.audio_rows.device())
            || !device.same_device(input.text_states.device())
            || !device.same_device(input.timesteps.device())
            || !device.same_device(layout.position_ids.device())
        {
            candle::bail!("MiniMax H3 inputs and frozen layout must share one device")
        }

        let compute_dtype = self.precision.compute_dtype();
        let video = self
            .video_projection
            .forward(&input.video_rows.to_dtype(DType::F32)?)?
            .to_dtype(compute_dtype)?;
        let audio = self
            .audio_projection
            .forward(&input.audio_rows.to_dtype(DType::F32)?)?
            .to_dtype(compute_dtype)?;
        let mut text = self
            .text_projection
            .forward(&input.text_states.to_dtype(compute_dtype)?)?;

        observer(H3BlockProgress {
            stack: H3BlockStack::TokenRefiner,
            completed: 0,
            total: self.token_refiner.len(),
        })?;
        for (index, block) in self.token_refiner.iter().enumerate() {
            text = block.forward(&text)?;
            observer(H3BlockProgress {
                stack: H3BlockStack::TokenRefiner,
                completed: index + 1,
                total: self.token_refiner.len(),
            })?;
        }
        text = self.token_refiner_norm.forward(&text)?;

        let mut hidden = Tensor::zeros(
            (batch, layout.seq_len, self.config.hidden_size),
            compute_dtype,
            device,
        )?;
        hidden = hidden.index_add(&layout.text_indices, &text, 1)?;
        hidden = hidden.index_add(&layout.video_indices, &video, 1)?;
        hidden = hidden.index_add(&layout.audio_indices, &audio, 1)?;

        let time_embedding = self.time.forward(input.timesteps)?;
        let adaln_input = match self.mode {
            H3AdaLnMode::Full => silu(&time_embedding)?.to_dtype(compute_dtype)?,
            H3AdaLnMode::Curve { .. } => time_embedding,
        };
        let (cos, sin) = self.rope.forward(&layout.position_ids)?;
        let rotary_dim = 6 * self.config.rope_inv_freq_len;
        observer(H3BlockProgress {
            stack: H3BlockStack::Transformer,
            completed: 0,
            total: self.blocks.len(),
        })?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(
                &hidden,
                &adaln_input,
                &layout.adaln_indices,
                (&cos, &sin, rotary_dim),
            )?;
            observer(H3BlockProgress {
                stack: H3BlockStack::Transformer,
                completed: index + 1,
                total: self.blocks.len(),
            })?;
        }
        observer(H3BlockProgress {
            stack: H3BlockStack::Output,
            completed: 0,
            total: 1,
        })?;
        let (video, audio) = self.final_layer.forward(
            &hidden,
            &adaln_input,
            &layout.timestep_indices,
            &layout.video_indices,
            &layout.audio_indices,
        )?;
        observer(H3BlockProgress {
            stack: H3BlockStack::Output,
            completed: 1,
            total: 1,
        })?;
        Ok(H3TransformerOutput { video, audio })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;

    fn inventory(
        config: &H3TransformerConfig,
        mode: H3AdaLnMode,
        precision: H3PrecisionProfile,
        component: H3CheckpointComponent,
    ) -> H3CheckpointInventory {
        H3CheckpointInventory {
            component,
            tensors: expected_h3_weight_specs(config, mode, precision).unwrap(),
        }
    }

    fn deterministic_values(name: &str, count: usize) -> Vec<f32> {
        if name == "rope.inv_freq" {
            return (0..count)
                .map(|index| 10_000f32.powf(-(index as f32) / count as f32))
                .collect();
        }
        let name_seed = name.bytes().fold(0u32, |sum, byte| {
            sum.wrapping_mul(33).wrapping_add(byte as u32)
        });
        let norm = name.ends_with("norm.weight");
        (0..count)
            .map(|index| {
                let raw =
                    name_seed.wrapping_add((index as u32).wrapping_mul(2_654_435_761)) % 2_003;
                let centered = raw as f32 / 1_001.0 - 1.0;
                if norm {
                    1.0 + centered * 0.015
                } else if name.ends_with(".bias") {
                    centered * 0.01
                } else {
                    centered * 0.035
                }
            })
            .collect()
    }

    fn tensor_map(
        specs: &BTreeMap<String, H3TensorSpec>,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        specs
            .iter()
            .map(|(name, spec)| {
                let count = spec.shape.iter().product();
                let tensor = Tensor::from_vec(
                    deterministic_values(name, count),
                    spec.shape.as_slice(),
                    device,
                )?
                .to_dtype(spec.dtype)?;
                Ok((name.clone(), tensor))
            })
            .collect()
    }

    fn source_from_inventory(
        inventory: H3CheckpointInventory,
        device: &Device,
    ) -> Result<H3CheckpointSource<'static>> {
        let tensors = tensor_map(&inventory.tensors, device)?;
        H3CheckpointSource::from_tensors(inventory.component, tensors, device)
    }

    fn load_tiny(
        device: &Device,
        mode: H3AdaLnMode,
        precision: H3PrecisionProfile,
        task: H3TransformerTask,
    ) -> Result<H3Transformer> {
        let config = H3TransformerConfig::tiny();
        let inventory = inventory(&config, mode, precision, task.checkpoint_component());
        let source = source_from_inventory(inventory, device)?;
        let checkpoint = audit_h3_checkpoint(config, task, precision, source)
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        H3Transformer::load(checkpoint)
    }

    fn tiny_layout() -> H3PackedLayout {
        H3PackedLayout::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.5],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, -0.5],
            ],
            vec![0, 0, 1, 0, 1, 0],
            vec![1, 0, 2, 0, 2, 0],
            vec![1, 5],
            vec![2, 4],
            vec![0, 3],
        )
        .unwrap()
    }

    struct OwnedInput {
        video: Tensor,
        audio: Tensor,
        text: Tensor,
        timesteps: Tensor,
    }

    impl OwnedInput {
        fn new(device: &Device) -> Result<Self> {
            Ok(Self {
                video: Tensor::from_vec(
                    (0..16).map(|value| value as f32 / 17.0 - 0.4).collect(),
                    (1, 2, 8),
                    device,
                )?,
                audio: Tensor::from_vec(
                    (0..6).map(|value| value as f32 / 11.0 - 0.2).collect(),
                    (1, 2, 3),
                    device,
                )?,
                text: Tensor::from_vec(
                    (0..12).map(|value| value as f32 / 13.0 - 0.3).collect(),
                    (1, 2, 6),
                    device,
                )?,
                timesteps: Tensor::from_vec(vec![0.2f32, 0.7], 2, device)?,
            })
        }

        fn borrowed(&self) -> H3ForwardInput<'_> {
            H3ForwardInput {
                video_rows: &self.video,
                audio_rows: &self.audio,
                text_states: &self.text,
                timesteps: &self.timesteps,
            }
        }
    }

    fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) -> Result<()> {
        let left = left.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let right = right
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(left.len(), right.len());
        let max = left
            .iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= tolerance,
            "maximum absolute error {max} > {tolerance}"
        );
        Ok(())
    }

    #[test]
    fn released_config_and_complete_key_counts_are_frozen() {
        let config = H3TransformerConfig::default();
        config.validate().unwrap();
        assert_eq!(config.hidden_size, 5_376);
        assert_eq!(config.attention_inner_dim(), 7_168);
        assert_eq!(config.video_patch_dim(), 96);
        assert_eq!(6 * config.rope_inv_freq_len, 96);
        assert_eq!(
            expected_h3_weight_specs(
                &config,
                H3AdaLnMode::Full,
                H3PrecisionProfile::OfficialMixedBf16F32,
            )
            .unwrap()
            .len(),
            535
        );
        assert_eq!(
            expected_h3_weight_specs(
                &config,
                H3AdaLnMode::Curve {
                    grid: 129,
                    basis_dim: 64,
                },
                H3PrecisionProfile::OfficialMixedBf16F32,
            )
            .unwrap()
            .len(),
            532
        );
    }

    #[test]
    fn pruned_mixed_precision_schema_uses_fp32_curve_and_adaln_only() {
        let config = H3TransformerConfig::tiny();
        let mode = H3AdaLnMode::Curve {
            grid: 9,
            basis_dim: 3,
        };
        let specs =
            expected_h3_weight_specs(&config, mode, H3PrecisionProfile::OfficialMixedBf16F32)
                .unwrap();

        assert_eq!(specs["adaln_t_table"].dtype, DType::F32);
        assert!(!specs.keys().any(|name| name.starts_with("time_embedder.")));
        let adaln_specs = specs
            .iter()
            .filter(|(name, _)| name.contains(".adaln_proj."))
            .collect::<Vec<_>>();
        assert_eq!(adaln_specs.len(), 2 * (config.num_layers + 1));
        assert!(adaln_specs.iter().all(|(_, spec)| spec.dtype == DType::F32));
        assert_eq!(specs["blocks.0.norm1.weight"].dtype, DType::BF16);
        assert_eq!(specs["condition_proj.weight"].dtype, DType::BF16);
        assert_eq!(specs["video_patch_proj.weight"].dtype, DType::F32);
    }

    #[test]
    fn full_and_pruned_layouts_detect_with_explicit_qkv_physics() {
        let config = H3TransformerConfig::tiny();
        let full_inventory = inventory(
            &config,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3CheckpointComponent::Transformer,
        );
        let full_source = source_from_inventory(full_inventory, &Device::Cpu).unwrap();
        let full = audit_h3_checkpoint(
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            H3PrecisionProfile::SyntheticF32,
            full_source,
        )
        .unwrap();
        assert_eq!(full.plan().adaln_mode(), H3AdaLnMode::Full);
        assert_eq!(full.plan().qkv_layout(), H3QkvLayout::GroupedPerHead);
        assert!(full.plan().ignored_keys().is_empty());

        let curve_mode = H3AdaLnMode::Curve {
            grid: 9,
            basis_dim: 3,
        };
        let pruned_inventory = inventory(
            &config,
            curve_mode,
            H3PrecisionProfile::SyntheticF32,
            H3CheckpointComponent::TransformerRef,
        );
        let pruned_source = source_from_inventory(pruned_inventory, &Device::Cpu).unwrap();
        let pruned = audit_h3_checkpoint(
            config,
            H3TransformerTask::Ref2Va,
            H3PrecisionProfile::SyntheticF32,
            pruned_source,
        )
        .unwrap();
        assert_eq!(pruned.plan().adaln_mode(), curve_mode);
        assert_eq!(pruned.plan().qkv_layout(), H3QkvLayout::QkvMajor);
    }

    #[test]
    fn official_fl_and_ref_sources_cannot_be_interchanged() {
        let config = H3TransformerConfig::tiny();
        let ref_inventory = inventory(
            &config,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3CheckpointComponent::TransformerRef,
        );
        let ref_source = source_from_inventory(ref_inventory, &Device::Cpu).unwrap();
        let error = audit_h3_checkpoint(
            config.clone(),
            H3TransformerTask::T2VaFl2Va,
            H3PrecisionProfile::SyntheticF32,
            ref_source,
        )
        .unwrap_err();
        assert_eq!(
            error.issues,
            vec![H3WeightAuditIssue::WrongComponent {
                expected: H3CheckpointComponent::Transformer,
                actual: H3CheckpointComponent::TransformerRef,
            }]
        );

        let fl_inventory = inventory(
            &config,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3CheckpointComponent::Transformer,
        );
        let fl_source = source_from_inventory(fl_inventory, &Device::Cpu).unwrap();
        let error = audit_h3_checkpoint(
            config,
            H3TransformerTask::Ref2Va,
            H3PrecisionProfile::SyntheticF32,
            fl_source,
        )
        .unwrap_err();
        assert_eq!(
            error.issues,
            vec![H3WeightAuditIssue::WrongComponent {
                expected: H3CheckpointComponent::TransformerRef,
                actual: H3CheckpointComponent::Transformer,
            }]
        );
    }

    #[test]
    fn checkpoint_source_cannot_be_audited_with_a_different_config() {
        let source_config = H3TransformerConfig::tiny();
        let source = source_from_inventory(
            inventory(
                &source_config,
                H3AdaLnMode::Full,
                H3PrecisionProfile::SyntheticF32,
                H3CheckpointComponent::Transformer,
            ),
            &Device::Cpu,
        )
        .unwrap();
        let mut requested_config = source_config;
        requested_config.num_layers = 1;

        let error = audit_h3_checkpoint(
            requested_config,
            H3TransformerTask::T2VaFl2Va,
            H3PrecisionProfile::SyntheticF32,
            source,
        )
        .unwrap_err();
        assert!(error.issues.iter().any(|issue| matches!(
            issue,
            H3WeightAuditIssue::UnexpectedKey(key) if key.starts_with("blocks.1.")
        )));
    }

    #[test]
    fn mmap_audit_and_load_share_one_checkpoint_backend() -> Result<()> {
        let device = Device::Cpu;
        let config = H3TransformerConfig::tiny();
        let tensors = tensor_map(
            &expected_h3_weight_specs(
                &config,
                H3AdaLnMode::Full,
                H3PrecisionProfile::SyntheticF32,
            )?,
            &device,
        )?;
        let directory = std::env::temp_dir().join(format!(
            "mold-h3-dit-source-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| candle::Error::Msg(error.to_string()))?
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("transformer.safetensors");
        candle::safetensors::save(&tensors, &path)?;

        // SAFETY: the file stays immutable until `load` consumes the source.
        let source = unsafe {
            H3CheckpointSource::from_mmaped_safetensors(
                H3CheckpointComponent::Transformer,
                std::slice::from_ref(&path),
                &device,
            )?
        };
        let checkpoint = audit_h3_checkpoint(
            config,
            H3TransformerTask::T2VaFl2Va,
            H3PrecisionProfile::SyntheticF32,
            source,
        )
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let model = H3Transformer::load(checkpoint)?;
        assert_eq!(model.task(), H3TransformerTask::T2VaFl2Va);
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn audit_rejects_hybrid_missing_extra_bad_shape_and_bad_dtype() {
        let config = H3TransformerConfig::tiny();
        let mut inventory = inventory(
            &config,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3CheckpointComponent::Transformer,
        );
        inventory.tensors.insert(
            "adaln_t_table".into(),
            H3TensorSpec::new([5, 3], DType::F32),
        );
        inventory.tensors.remove("blocks.1.norm2.weight");
        inventory.tensors.insert(
            "blocks.0.norm1.weight".into(),
            H3TensorSpec::new([9], DType::BF16),
        );
        inventory.tensors.insert(
            "quantized.aux_scale".into(),
            H3TensorSpec::new([1], DType::F32),
        );
        let source = source_from_inventory(inventory, &Device::Cpu).unwrap();
        let error = audit_h3_checkpoint(
            config,
            H3TransformerTask::T2VaFl2Va,
            H3PrecisionProfile::SyntheticF32,
            source,
        )
        .unwrap_err();
        assert!(error
            .issues
            .contains(&H3WeightAuditIssue::AmbiguousAdaLnRepresentation));
        assert!(error.issues.contains(&H3WeightAuditIssue::MissingKey(
            "blocks.1.norm2.weight".into()
        )));
        assert!(error.issues.iter().any(|issue| matches!(
            issue,
            H3WeightAuditIssue::Shape { key, .. } if key == "blocks.0.norm1.weight"
        )));
        assert!(error.issues.iter().any(|issue| matches!(
            issue,
            H3WeightAuditIssue::DType { key, .. } if key == "blocks.0.norm1.weight"
        )));
        assert!(error.issues.contains(&H3WeightAuditIssue::UnexpectedKey(
            "quantized.aux_scale".into()
        )));
    }

    #[test]
    fn official_grouped_qkv_is_reordered_per_head() -> Result<()> {
        let source = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((12, 1))?;
        let reordered = reorder_h3_grouped_qkv(&source, 2, 2)?.to_vec2::<f32>()?;
        assert_eq!(
            reordered.into_iter().flatten().collect::<Vec<_>>(),
            vec![0., 1., 6., 7., 2., 3., 8., 9., 4., 5., 10., 11.]
        );
        Ok(())
    }

    #[test]
    fn swiglu_uses_gate_then_up_halves() -> Result<()> {
        let device = Device::Cpu;
        let mlp = H3Mlp {
            width: 2,
            fc1: nn::Linear::new(
                Tensor::from_vec(vec![1f32, 2., 3., 4.], (4, 1), &device)?,
                None,
            ),
            fc2: nn::Linear::new(Tensor::from_vec(vec![5f32, 6.], (1, 2), &device)?, None),
        };
        let output = mlp
            .forward(&Tensor::from_vec(vec![1f32], (1, 1, 1), &device)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = 5.0 * (1.0 / (1.0 + (-1.0f32).exp())) * 3.0
            + 6.0 * (2.0 / (1.0 + (-2.0f32).exp())) * 4.0;
        assert!((output[0] - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn video_and_stereo_audio_packing_round_trip_exactly() -> Result<()> {
        let video = Tensor::arange(0f32, 32f32, &Device::Cpu)?.reshape((1, 2, 2, 2, 4))?;
        let rows = patchify_h3_video(&video, [1, 2, 2])?;
        assert_eq!(rows.dims3()?, (1, 4, 8));
        let restored = unpatchify_h3_video(&rows, [2, 1, 2], 2, [1, 2, 2])?;
        assert_close(&video, &restored, 0.0)?;

        let audio = Tensor::arange(0f32, 18f32, &Device::Cpu)?.reshape((1, 3, 2, 3))?;
        let rows = pack_h3_audio(&audio)?;
        assert_eq!(rows.dims3()?, (1, 6, 3));
        assert_eq!(
            rows.i((0, .., 0))?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );
        assert_close(&audio, &unpack_h3_audio(&rows, 2)?, 0.0)
    }

    #[test]
    fn packed_layout_is_exhaustive_and_estimates_attention_from_rows() {
        let layout = tiny_layout();
        assert_eq!(layout.seq_len(), 6);
        assert_eq!(layout.attention_score_elements(2, 4).unwrap(), 288);

        let error = H3PackedLayout::new(
            vec![[0.0; 3]; 3],
            vec![0; 3],
            vec![1, 0, 2],
            vec![1],
            vec![2],
            vec![1],
        )
        .unwrap_err();
        assert!(error.to_string().contains("claimed by both"));

        let error = H3PackedLayout::new(
            vec![[0.0; 3]; 3],
            vec![u32::MAX, 0, 0],
            vec![0, 2, 1],
            vec![0],
            vec![1],
            vec![2],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("overflows the AdaLN row address"));
    }

    #[test]
    fn multimodal_rope_rotates_leading_axes_and_preserves_tail() -> Result<()> {
        let rope = H3Rope {
            inv_freq: Tensor::new(&[1f32], &Device::Cpu)?,
        };
        let positions = Tensor::new(&[[1f32, 2., 3.]], &Device::Cpu)?;
        let (cos, sin) = rope.forward(&positions)?;
        let expected_angles = [1f32, 2., 3., 1., 2., 3.];
        assert_close(
            &cos,
            &Tensor::from_vec(
                expected_angles.iter().map(|value| value.cos()).collect(),
                (1, 6),
                &Device::Cpu,
            )?,
            1e-6,
        )?;
        let hidden = Tensor::from_vec(
            (1..=8).map(|value| value as f32).collect(),
            (1, 1, 1, 8),
            &Device::Cpu,
        )?;
        let rotated = apply_h3_rotary(&hidden, &cos, &sin, 6)?;
        assert_eq!(rotated.i((0, 0, 0, 6..))?.to_vec1::<f32>()?, vec![7., 8.]);
        Ok(())
    }

    #[test]
    fn full_tiny_stack_runs_both_heads_deterministically() -> Result<()> {
        let device = Device::Cpu;
        let model = load_tiny(
            &device,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        let layout = tiny_layout().freeze(&device)?;
        let input = OwnedInput::new(&device)?;
        let first = model.forward(input.borrowed(), &layout)?;
        let second = model.forward(input.borrowed(), &layout)?;
        assert_eq!(first.video.dims3()?, (1, 2, 8));
        assert_eq!(first.audio.dims3()?, (1, 2, 3));
        assert_eq!(first.video.dtype(), DType::F32);
        assert_eq!(first.audio.dtype(), DType::F32);
        assert_close(
            &first.video,
            &Tensor::from_vec(
                vec![
                    0.09703499,
                    -0.096913226,
                    -0.11364641,
                    -0.0983998,
                    -0.10740345,
                    -0.074998744,
                    0.05799303,
                    0.053229645,
                    0.065745726,
                    -0.15126213,
                    -0.11983699,
                    -0.08727337,
                    -0.06793387,
                    -0.06282898,
                    -0.008962024,
                    0.003591614,
                ],
                (1, 2, 8),
                &device,
            )?,
            1e-6,
        )?;
        assert_close(
            &first.audio,
            &Tensor::from_vec(
                vec![
                    0.031730644,
                    0.027603628,
                    0.038541544,
                    0.047282588,
                    0.05488261,
                    0.07754755,
                ],
                (1, 2, 3),
                &device,
            )?,
            1e-6,
        )?;
        assert_close(&first.video, &second.video, 0.0)?;
        assert_close(&first.audio, &second.audio, 0.0)?;
        assert!(first
            .video
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn pruned_curve_stack_interpolates_dual_timesteps() -> Result<()> {
        let device = Device::Cpu;
        let mode = H3AdaLnMode::Curve {
            grid: 7,
            basis_dim: 3,
        };
        let model = load_tiny(
            &device,
            mode,
            H3PrecisionProfile::SyntheticF32,
            H3TransformerTask::Ref2Va,
        )?;
        assert_eq!(model.adaln_mode(), mode);
        assert_eq!(model.task(), H3TransformerTask::Ref2Va);
        let layout = tiny_layout().freeze(&device)?;
        let input = OwnedInput::new(&device)?;
        let original = model.forward(input.borrowed(), &layout)?;
        let changed_timesteps = Tensor::from_vec(vec![0.2f32, 0.3], 2, &device)?;
        let changed = model.forward(
            H3ForwardInput {
                timesteps: &changed_timesteps,
                ..input.borrowed()
            },
            &layout,
        )?;
        let audio_original = original.audio.flatten_all()?.to_vec1::<f32>()?;
        let audio_changed = changed.audio.flatten_all()?.to_vec1::<f32>()?;
        assert!(audio_original
            .iter()
            .zip(audio_changed)
            .any(|(left, right)| (left - right).abs() > 1e-7));
        Ok(())
    }

    #[test]
    fn observer_is_a_block_boundary_cancellation_contract() -> Result<()> {
        let device = Device::Cpu;
        let model = load_tiny(
            &device,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        let layout = tiny_layout().freeze(&device)?;
        let input = OwnedInput::new(&device)?;
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let error = model
            .forward_with_observer(input.borrowed(), &layout, move |event| {
                captured.lock().unwrap().push(event);
                if event.stack == H3BlockStack::Transformer && event.completed == 1 {
                    candle::bail!("synthetic cancellation")
                }
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("synthetic cancellation"));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                H3BlockProgress {
                    stack: H3BlockStack::TokenRefiner,
                    completed: 0,
                    total: 1,
                },
                H3BlockProgress {
                    stack: H3BlockStack::TokenRefiner,
                    completed: 1,
                    total: 1,
                },
                H3BlockProgress {
                    stack: H3BlockStack::Transformer,
                    completed: 0,
                    total: 2,
                },
                H3BlockProgress {
                    stack: H3BlockStack::Transformer,
                    completed: 1,
                    total: 2,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn official_mixed_precision_keeps_fp32_islands_and_output_heads() -> Result<()> {
        let device = Device::Cpu;
        let model = load_tiny(
            &device,
            H3AdaLnMode::Full,
            H3PrecisionProfile::OfficialMixedBf16F32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        assert_eq!(model.video_projection.weight().dtype(), DType::F32);
        assert_eq!(model.text_projection.weight().dtype(), DType::BF16);
        assert_eq!(model.blocks[0].attention.qkv.weight().dtype(), DType::BF16);
        assert_eq!(model.blocks[0].adaln.linear.weight().dtype(), DType::BF16);
        assert_eq!(model.final_layer.video_out.weight().dtype(), DType::F32);
        assert_eq!(model.final_layer.audio_out.weight().dtype(), DType::F32);
        match &model.time {
            H3TimeConditioner::Full {
                proj_in, proj_out, ..
            } => {
                assert_eq!(proj_in.weight().dtype(), DType::F32);
                assert_eq!(proj_out.weight().dtype(), DType::F32);
            }
            H3TimeConditioner::Curve { .. } => panic!("expected full timestep MLP"),
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_full_stack_cpu_cuda_parity() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cpu_model = load_tiny(
            &Device::Cpu,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        let cuda_model = load_tiny(
            &cuda,
            H3AdaLnMode::Full,
            H3PrecisionProfile::SyntheticF32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        let cpu_input = OwnedInput::new(&Device::Cpu)?;
        let cuda_input = OwnedInput::new(&cuda)?;
        let cpu = cpu_model.forward(cpu_input.borrowed(), &tiny_layout().freeze(&Device::Cpu)?)?;
        let gpu = cuda_model.forward(cuda_input.borrowed(), &tiny_layout().freeze(&cuda)?)?;
        assert_close(&cpu.video, &gpu.video, 2e-5)?;
        assert_close(&cpu.audio, &gpu.audio, 2e-5)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_official_mixed_precision_stack_runs_on_cuda() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let model = load_tiny(
            &cuda,
            H3AdaLnMode::Full,
            H3PrecisionProfile::OfficialMixedBf16F32,
            H3TransformerTask::T2VaFl2Va,
        )?;
        let input = OwnedInput::new(&cuda)?;
        let output = model.forward(input.borrowed(), &tiny_layout().freeze(&cuda)?)?;
        assert_eq!(output.video.dtype(), DType::F32);
        assert_eq!(output.audio.dtype(), DType::F32);
        assert!(output
            .video
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        assert!(output
            .audio
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_pruned_mixed_precision_stack_runs_on_cuda() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let model = load_tiny(
            &cuda,
            H3AdaLnMode::Curve {
                grid: 9,
                basis_dim: 3,
            },
            H3PrecisionProfile::OfficialMixedBf16F32,
            H3TransformerTask::Ref2Va,
        )?;
        let input = OwnedInput::new(&cuda)?;
        let output = model.forward(input.borrowed(), &tiny_layout().freeze(&cuda)?)?;
        assert_eq!(output.video.dtype(), DType::F32);
        assert_eq!(output.audio.dtype(), DType::F32);
        assert!(output
            .video
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        assert!(output
            .audio
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }
}
