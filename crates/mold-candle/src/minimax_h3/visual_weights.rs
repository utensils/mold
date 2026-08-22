use super::visual_vae::{MiniMaxH3VisualVaeConfig, WeightLayout};
use candle::{bail, DType, Device, Result};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualVaeComponentRole {
    F16T4D24,
}

impl VisualVaeComponentRole {
    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::F16T4D24 => "minimax-h3.visual-vae.f16t4d24",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiffusersWeightIndex {
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsTensorHeader {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsHeader {
    pub tensors: BTreeMap<String, SafetensorsTensorHeader>,
    pub header_len: u64,
    pub file_len: u64,
}

struct UniqueSafetensorsHeader(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueSafetensorsHeader {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UniqueHeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for UniqueHeaderVisitor {
            type Value = UniqueSafetensorsHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header with unique tensor names")
            }

            fn visit_map<A>(
                self,
                mut map: A,
            ) -> std::result::Result<UniqueSafetensorsHeader, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut entries = BTreeMap::new();
                while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if entries.insert(name.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate safetensors tensor {name}"
                        )));
                    }
                }
                Ok(UniqueSafetensorsHeader(entries))
            }
        }

        deserializer.deserialize_map(UniqueHeaderVisitor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisualVaeWeightInspection {
    pub role: &'static str,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub total_size: u64,
    /// Header/index identity only. Use [`component_fingerprint`] when a full
    /// content fingerprint is required after authorization.
    pub header_identity_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VisualWeightFileIdentity {
    canonical_path: PathBuf,
    logical_name: String,
    descriptor_bound: bool,
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

/// Opaque proof that one exact visual-VAE artifact set passed its complete
/// layout, dtype, key, shape, and immutable-file inspection. Creating this
/// token opens model artifacts, so callers must cross Mold's authorization
/// gate before invoking either validator that returns it.
#[derive(Clone, Debug)]
pub struct ValidatedVisualVaeWeights {
    layout: WeightLayout,
    inspection: VisualVaeWeightInspection,
    files: Vec<VisualWeightFileIdentity>,
}

impl ValidatedVisualVaeWeights {
    pub fn layout(&self) -> WeightLayout {
        self.layout
    }

    pub fn artifact_dtype(&self) -> DType {
        self.layout.artifact_dtype()
    }

    pub fn inspection(&self) -> &VisualVaeWeightInspection {
        &self.inspection
    }

    pub fn component_fingerprint(
        &self,
        observer: &mut dyn VisualWeightReadObserver,
    ) -> Result<String> {
        self.revalidate_files()?;
        fingerprint_identities(&self.files, observer)
    }

    pub(crate) fn revalidate_files(&self) -> Result<()> {
        for expected in &self.files {
            let current = current_visual_weight_file_identity(expected)?;
            if !same_visual_weight_file_identity(&current, expected) {
                bail!(
                    "validated H3 visual VAE artifact changed after inspection: {}",
                    expected.canonical_path.display()
                )
            }
        }
        Ok(())
    }

    pub(crate) fn mmap_var_builder<'a>(&self, device: &Device) -> Result<VarBuilder<'a>> {
        self.revalidate_files()?;
        let paths = self
            .files
            .iter()
            .map(|identity| identity.canonical_path.as_path())
            .collect::<Vec<_>>();
        // SAFETY: every canonical regular file was fully schema-validated and
        // its immutable file identity was rechecked immediately before mmap.
        // The returned backend owns its mappings; a second identity check
        // below rejects replacement or mutation racing the open operation.
        let builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&paths, self.artifact_dtype(), device)? };
        self.revalidate_files()?;
        Ok(builder)
    }
}

pub trait VisualWeightReadObserver {
    /// Returning an error cancels hashing at the next 1 MiB read boundary.
    fn checkpoint(&mut self, bytes_read: u64, total_bytes: u64) -> Result<()>;
}

#[derive(Default)]
pub struct NoopVisualWeightReadObserver;

impl VisualWeightReadObserver for NoopVisualWeightReadObserver {
    fn checkpoint(&mut self, _bytes_read: u64, _total_bytes: u64) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComfyTensorTransform {
    Direct {
        source: String,
        target: String,
    },
    /// Reorder the source checkpoint's per-head `[q,k,v]` rows into
    /// contiguous `[q_all,k_all,v_all]`, then split the first axis.
    ReorderAndSplitQkv {
        source: String,
        query: String,
        key: String,
        value: String,
        num_attention_heads: usize,
        attention_head_dim: usize,
    },
    /// Swap the source `[gate,up]` halves into Diffusers' `[up,gate]`.
    SwapFeedForwardHalves {
        source: String,
        target: String,
        split_axis: usize,
    },
    /// Checkpoint-only training state intentionally omitted at inference.
    Drop {
        source: String,
    },
}

/// Build the complete Diffusers checkpoint key/shape contract without opening
/// model data. Tiny configs use the same naming arithmetic in unit tests.
pub fn expected_diffusers_weight_shapes(
    config: &MiniMaxH3VisualVaeConfig,
) -> Result<BTreeMap<String, Vec<usize>>> {
    config.validate()?;
    let mut shapes = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>| -> Result<()> {
        if shapes.insert(name.clone(), shape).is_some() {
            bail!("duplicate H3 visual VAE weight key {name}")
        }
        Ok(())
    };
    let add_conv = |add: &mut dyn FnMut(String, Vec<usize>) -> Result<()>,
                    prefix: String,
                    out_channels: usize,
                    in_channels: usize,
                    kernel: usize| {
        add(
            format!("{prefix}.weight"),
            vec![out_channels, in_channels, kernel, kernel, kernel],
        )?;
        add(format!("{prefix}.bias"), vec![out_channels])
    };
    let add_norm =
        |add: &mut dyn FnMut(String, Vec<usize>) -> Result<()>, prefix: String, channels: usize| {
            add(format!("{prefix}.weight"), vec![channels])?;
            add(format!("{prefix}.bias"), vec![channels])
        };

    add_conv(
        &mut add,
        "encoder.conv_in".into(),
        config.block_out_channels[0],
        config.in_channels,
        3,
    )?;
    for level in 0..config.block_out_channels.len() {
        let output_channels = config.block_out_channels[level];
        let input_channels = if level == 0 {
            output_channels
        } else {
            config.block_out_channels[level - 1]
        };
        for block in 0..config.layers_per_block {
            let block_input = if block == 0 {
                input_channels
            } else {
                output_channels
            };
            let prefix = format!("encoder.down_blocks.{level}.resnets.{block}");
            add_norm(&mut add, format!("{prefix}.norm1"), block_input)?;
            add_conv(
                &mut add,
                format!("{prefix}.conv1"),
                output_channels,
                block_input,
                3,
            )?;
            add_norm(&mut add, format!("{prefix}.norm2"), output_channels)?;
            add_conv(
                &mut add,
                format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
            )?;
            if block_input != output_channels {
                add_conv(
                    &mut add,
                    format!("{prefix}.conv_shortcut"),
                    output_channels,
                    block_input,
                    1,
                )?;
            }
        }
        if config.spatial_downsample_factors[level] * config.temporal_downsample_factors[level] > 1
        {
            add_conv(
                &mut add,
                format!("encoder.down_blocks.{level}.downsamplers.0.conv"),
                output_channels,
                output_channels,
                3,
            )?;
        }
    }
    let final_channels = config
        .block_out_channels
        .last()
        .copied()
        .ok_or_else(|| candle::Error::Msg("H3 visual VAE channel schedule is empty".into()))?;
    add_norm(&mut add, "encoder.norm_out".into(), final_channels)?;
    add_conv(
        &mut add,
        "encoder.conv_out".into(),
        config.latent_channels * 2,
        final_channels,
        3,
    )?;
    add_conv(
        &mut add,
        "quant_conv".into(),
        config.latent_channels * 2,
        config.latent_channels * 2,
        1,
    )?;
    add_conv(
        &mut add,
        "post_quant_conv".into(),
        config.latent_channels,
        config.latent_channels,
        1,
    )?;

    let dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
    let inner = dim * config.decoder_ffn_mult;
    add(
        "decoder.proj_in.weight".into(),
        vec![dim, config.latent_channels],
    )?;
    add("decoder.proj_in.bias".into(), vec![dim])?;
    add(
        "decoder.register_tokens".into(),
        vec![1, config.decoder_num_register_tokens, dim],
    )?;
    for block in 0..config.decoder_num_layers {
        let prefix = format!("decoder.transformer_blocks.{block}");
        add(format!("{prefix}.norm1.weight"), vec![dim])?;
        for projection in ["to_q", "to_k", "to_v"] {
            add(format!("{prefix}.attn.{projection}.weight"), vec![dim, dim])?;
            add(format!("{prefix}.attn.{projection}.bias"), vec![dim])?;
        }
        add(format!("{prefix}.attn.to_out.0.weight"), vec![dim, dim])?;
        add(format!("{prefix}.attn.to_out.0.bias"), vec![dim])?;
        add(format!("{prefix}.scale1"), vec![dim])?;
        add(format!("{prefix}.norm2.weight"), vec![dim])?;
        add(
            format!("{prefix}.ff.net.0.proj.weight"),
            vec![inner * 2, dim],
        )?;
        add(format!("{prefix}.ff.net.0.proj.bias"), vec![inner * 2])?;
        add(format!("{prefix}.ff.net.2.weight"), vec![dim, inner])?;
        add(format!("{prefix}.ff.net.2.bias"), vec![dim])?;
        add(format!("{prefix}.scale2"), vec![dim])?;
    }
    add("decoder.norm_out.weight".into(), vec![dim])?;
    add("decoder.norm_out.bias".into(), vec![dim])?;
    let patch_volume = config.out_channels
        * config.temporal_compression_ratio()
        * config.spatial_compression_ratio()
        * config.spatial_compression_ratio();
    add("decoder.proj_out.weight".into(), vec![patch_volume, dim])?;
    add("decoder.proj_out.bias".into(), vec![patch_volume])?;
    Ok(shapes)
}

/// Map old bundled/Comfy source names to the converted Diffusers layout. The
/// fused QKV transform is returned separately because it is a split, not a
/// rename. `decoder.mask_token` is checkpoint-only training state. The two
/// latent-stat buffers duplicate the authenticated config authority used by
/// encode/decode. All three remain validated source tensors but are not model
/// parameters.
pub fn comfy_tensor_transforms(
    config: &MiniMaxH3VisualVaeConfig,
) -> Result<Vec<ComfyTensorTransform>> {
    let expected = expected_diffusers_weight_shapes(config)?;
    let mut transforms = Vec::new();
    for target in expected.keys() {
        if target.contains(".attn.to_q.")
            || target.contains(".attn.to_k.")
            || target.contains(".attn.to_v.")
            || target.contains(".ff.net.0.proj.")
        {
            continue;
        }
        let source = diffusers_to_comfy_direct_key(target);
        transforms.push(ComfyTensorTransform::Direct {
            source,
            target: target.clone(),
        });
    }
    for block in 0..config.decoder_num_layers {
        for suffix in ["weight", "bias"] {
            let prefix = format!("decoder.transformer_blocks.{block}.attn");
            transforms.push(ComfyTensorTransform::ReorderAndSplitQkv {
                source: format!("{prefix}.to_qkv.{suffix}"),
                query: format!("{prefix}.to_q.{suffix}"),
                key: format!("{prefix}.to_k.{suffix}"),
                value: format!("{prefix}.to_v.{suffix}"),
                num_attention_heads: config.decoder_num_attention_heads,
                attention_head_dim: config.decoder_attention_head_dim,
            });
            transforms.push(ComfyTensorTransform::SwapFeedForwardHalves {
                source: format!("decoder.transformer_blocks.{block}.ff.w1.{suffix}"),
                target: format!("decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}"),
                split_axis: 0,
            });
        }
    }
    transforms.push(ComfyTensorTransform::Drop {
        source: "decoder.mask_token".into(),
    });
    for source in ["latents_mean", "latents_std"] {
        transforms.push(ComfyTensorTransform::Drop {
            source: source.into(),
        });
    }
    Ok(transforms)
}

/// Exact tensor contract of Mold's pinned single-file Comfy checkpoint. The
/// source keeps attention QKV fused per head, stores the feed-forward gate
/// half first, and includes three source-only tensors: the decoder mask token
/// plus config-backed latent mean and standard deviation buffers.
pub fn expected_comfy_weight_shapes(
    config: &MiniMaxH3VisualVaeConfig,
) -> Result<BTreeMap<String, Vec<usize>>> {
    let diffusers = expected_diffusers_weight_shapes(config)?;
    let mut source = BTreeMap::new();
    let mut insert = |name: String, shape: Vec<usize>| -> Result<()> {
        if source.insert(name.clone(), shape).is_some() {
            bail!("duplicate H3 Comfy source weight key {name}")
        }
        Ok(())
    };
    for transform in comfy_tensor_transforms(config)? {
        match transform {
            ComfyTensorTransform::Direct {
                source: source_name,
                target,
            }
            | ComfyTensorTransform::SwapFeedForwardHalves {
                source: source_name,
                target,
                ..
            } => insert(source_name, diffusers[&target].clone())?,
            ComfyTensorTransform::ReorderAndSplitQkv {
                source: source_name,
                query,
                key,
                value,
                ..
            } => {
                let query_shape = &diffusers[&query];
                if diffusers[&key] != *query_shape || diffusers[&value] != *query_shape {
                    bail!("H3 Diffusers QKV target shapes disagree")
                }
                let mut fused_shape = query_shape.clone();
                fused_shape[0] = fused_shape[0]
                    .checked_mul(3)
                    .ok_or_else(|| candle::Error::Msg("H3 fused QKV shape overflow".into()))?;
                insert(source_name, fused_shape)?;
            }
            ComfyTensorTransform::Drop {
                source: source_name,
            } => {
                let shape = match source_name.as_str() {
                    "decoder.mask_token" => {
                        let dim = config
                            .decoder_num_attention_heads
                            .checked_mul(config.decoder_attention_head_dim)
                            .ok_or_else(|| {
                                candle::Error::Msg(
                                    "H3 decoder width overflow for mask token".into(),
                                )
                            })?;
                        vec![1, 1, dim]
                    }
                    "latents_mean" | "latents_std" => vec![config.latent_channels],
                    other => bail!("unknown dropped H3 Comfy visual VAE tensor {other}"),
                };
                insert(source_name, shape)?;
            }
        }
    }
    Ok(source)
}

/// Dense model-parameter bytes after source-only tensors are discarded and
/// fused source tensors are mapped into their execution layout.
pub fn expected_visual_vae_parameter_bytes(
    config: &MiniMaxH3VisualVaeConfig,
    dtype: DType,
) -> Result<u64> {
    if !matches!(dtype, DType::F16 | DType::F32) {
        bail!("H3 visual VAE parameter accounting does not support {dtype:?}")
    }
    let expected = expected_diffusers_weight_shapes(config)?;
    expected_data_size(&expected, dtype.size_in_bytes() as u64)
}

fn diffusers_to_comfy_direct_key(target: &str) -> String {
    let mut source = target
        .replace("encoder.down_blocks.", "encoder.down.")
        .replace(".resnets.", ".block.")
        .replace(".conv_shortcut.", ".nin_shortcut.")
        .replace(".downsamplers.0.conv.", ".downsample.conv.")
        .replace("decoder.proj_in.", "decoder.x_embedder.")
        .replace(".attn.to_out.0.", ".attn.to_out.")
        .replace(".ff.net.0.proj.", ".ff.w1.")
        .replace(".ff.net.2.", ".ff.w2.");
    // The source Downsample3D owns `downsample.conv`; some repacks flatten it
    // to `downsample`. The loader accepts both, but the canonical source path
    // is the nested form above.
    if source == "decoder.proj_in" {
        source = "decoder.x_embedder".into();
    }
    source
}

pub fn inspect_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    // A retained staging descriptor must stay descriptor-bound. Canonicalizing
    // `/dev/fd/N` on macOS either fails or resolves it back through the
    // replaceable staging pathname, defeating both portability and the inode
    // fence the caller deliberately retained (#1296).
    let descriptor_bound = path.parent().is_some_and(|parent| {
        parent == Path::new("/dev/fd") || parent == Path::new("/proc/self/fd")
    });
    let identity = if descriptor_bound {
        visual_weight_file_identity_from_descriptor(path)?
    } else {
        visual_weight_file_identity(path)?
    };
    inspect_safetensors_header_for_identity(&identity)
}

fn inspect_safetensors_header_for_identity(
    identity: &VisualWeightFileIdentity,
) -> Result<SafetensorsHeader> {
    const MAX_HEADER: u64 = 100_000_000;
    let mut file = open_expected_file(identity, "before header inspection")?;
    let file_len = identity.len;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > MAX_HEADER || header_len + 8 > file_len {
        bail!(
            "invalid safetensors header length {header_len} for {}",
            identity.canonical_path.display()
        )
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let UniqueSafetensorsHeader(raw) =
        serde_json::from_slice(&header_bytes).map_err(candle::Error::wrap)?;
    let data_len = file_len - header_len - 8;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        #[derive(Deserialize)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [u64; 2],
        }
        let entry: Entry = serde_json::from_value(value).map_err(candle::Error::wrap)?;
        if entry.data_offsets[0] > entry.data_offsets[1] || entry.data_offsets[1] > data_len {
            bail!(
                "invalid safetensors offsets for {name} in {}",
                identity.canonical_path.display()
            )
        }
        let elements = entry
            .shape
            .iter()
            .try_fold(1u64, |total, dim| total.checked_mul(*dim as u64))
            .ok_or_else(|| candle::Error::Msg(format!("shape overflow for {name}")))?;
        let element_size = dtype_size(&entry.dtype)?;
        let expected_bytes = elements
            .checked_mul(element_size)
            .ok_or_else(|| candle::Error::Msg(format!("byte-size overflow for {name}")))?;
        if entry.data_offsets[1] - entry.data_offsets[0] != expected_bytes {
            bail!("safetensors byte range does not match dtype/shape for {name}")
        }
        if tensors
            .insert(
                name.clone(),
                SafetensorsTensorHeader {
                    dtype: entry.dtype,
                    shape: entry.shape,
                    data_offsets: entry.data_offsets,
                },
            )
            .is_some()
        {
            bail!("duplicate safetensors tensor {name}")
        }
    }
    let mut cursor = 0u64;
    let mut ranges = tensors
        .iter()
        .map(|(name, tensor)| (tensor.data_offsets, name.as_str()))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(offsets, _)| offsets[0]);
    for (offsets, name) in ranges {
        if offsets[0] != cursor {
            bail!(
                "non-contiguous or overlapping safetensors data before {name} in {}",
                identity.canonical_path.display()
            )
        }
        cursor = offsets[1];
    }
    if cursor != data_len {
        bail!(
            "safetensors data section has {} unclaimed bytes in {}",
            data_len - cursor,
            identity.canonical_path.display()
        )
    }
    ensure_open_file_unchanged(&file, identity, "during header inspection")?;
    ensure_identity_unchanged(identity, "during header inspection")?;
    Ok(SafetensorsHeader {
        tensors,
        header_len,
        file_len,
    })
}

/// Read and validate Diffusers component metadata without opening any weight
/// shard. Unknown framework bookkeeping fields are intentionally ignored;
/// every inference-relevant field must equal the released f16t4d24 contract.
pub fn inspect_visual_vae_config(path: &Path) -> Result<MiniMaxH3VisualVaeConfig> {
    let bytes = std::fs::read(path)?;
    inspect_visual_vae_config_bytes(&bytes)
}

/// Validate an already-authenticated visual-VAE config without reopening its
/// source path. Production loaders use this after binding the config bytes to
/// one opened descriptor and the pinned manifest digest.
pub fn inspect_visual_vae_config_bytes(bytes: &[u8]) -> Result<MiniMaxH3VisualVaeConfig> {
    let raw: serde_json::Value = serde_json::from_slice(bytes).map_err(candle::Error::wrap)?;
    let object = raw
        .as_object()
        .ok_or_else(|| candle::Error::Msg("H3 visual VAE config must be a JSON object".into()))?;
    const REQUIRED: &[&str] = &[
        "in_channels",
        "out_channels",
        "latent_channels",
        "block_out_channels",
        "layers_per_block",
        "spatial_downsample_factors",
        "temporal_downsample_factors",
        "norm_num_groups",
        "norm_eps",
        "spatial_padding_mode",
        "decoder_num_layers",
        "decoder_num_attention_heads",
        "decoder_attention_head_dim",
        "decoder_num_register_tokens",
        "decoder_ffn_mult",
        "decoder_rope_theta",
        "decoder_rope_dim_ratio",
        "decoder_norm_eps",
        "clip_length",
        "token_drop",
        "latents_mean",
        "latents_std",
    ];
    let missing = REQUIRED
        .iter()
        .filter(|key| !object.contains_key(**key))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!("H3 visual VAE config is missing required fields {missing:?}")
    }
    let config: MiniMaxH3VisualVaeConfig =
        serde_json::from_value(raw).map_err(candle::Error::wrap)?;
    config.validate_production_contract()?;
    Ok(config)
}

pub fn validate_diffusers_weight_index(
    config: &MiniMaxH3VisualVaeConfig,
    index_path: &Path,
    component_dir: &Path,
) -> Result<ValidatedVisualVaeWeights> {
    config.validate_production_contract()?;
    let index_bytes = std::fs::read(index_path)?;
    let index: DiffusersWeightIndex =
        serde_json::from_slice(&index_bytes).map_err(candle::Error::wrap)?;
    let expected = expected_diffusers_weight_shapes(config)?;
    validate_weight_map_keys(expected.keys(), index.weight_map.keys())?;
    let expected_data_size = expected_visual_vae_parameter_bytes(config, DType::F32)?;
    let indexed_data_size = index
        .metadata
        .get("total_size")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            candle::Error::Msg(
                "H3 visual VAE weight index is missing numeric metadata.total_size".into(),
            )
        })?;
    if indexed_data_size != expected_data_size {
        bail!(
            "H3 visual VAE index total_size {indexed_data_size} does not match expected {expected_data_size}"
        )
    }

    let actual_keys = index.weight_map.keys().cloned().collect::<BTreeSet<_>>();
    let expected_keys = expected.keys().cloned().collect::<BTreeSet<_>>();
    debug_assert_eq!(actual_keys, expected_keys);

    let mut by_shard: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (tensor, shard) in &index.weight_map {
        validate_shard_basename(shard)?;
        by_shard
            .entry(shard.clone())
            .or_default()
            .insert(tensor.clone());
    }
    let indexed_shards = by_shard.keys().cloned().collect::<BTreeSet<_>>();
    validate_canonical_shard_names(&indexed_shards)?;
    let present_shards = safetensors_files(component_dir)?;
    if present_shards != indexed_shards {
        let missing = indexed_shards
            .difference(&present_shards)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = present_shards
            .difference(&indexed_shards)
            .cloned()
            .collect::<Vec<_>>();
        bail!("H3 visual VAE shard set mismatch: missing={missing:?} unexpected={unexpected:?}")
    }
    let mut total_size = 0u64;
    let mut identity = Sha256::new();
    let mut files = Vec::with_capacity(by_shard.len());
    identity.update(VisualVaeComponentRole::F16T4D24.stable_id().as_bytes());
    identity.update(b"diffusers-f32");
    identity.update(&index_bytes);
    for (shard, assigned) in &by_shard {
        let path = component_dir.join(shard);
        let file_identity = visual_weight_file_identity(&path)?;
        let header = inspect_safetensors_header_for_identity(&file_identity)?;
        let present = header.tensors.keys().cloned().collect::<BTreeSet<_>>();
        if &present != assigned {
            bail!("H3 visual VAE shard {shard} contents do not match the weight index")
        }
        for name in assigned {
            let tensor = &header.tensors[name];
            if tensor.dtype != "F32" {
                bail!(
                    "H3 released visual VAE tensor {name} must be stored F32, got {}",
                    tensor.dtype
                )
            }
            if tensor.shape != expected[name] {
                bail!(
                    "H3 visual VAE tensor {name} shape mismatch: {:?} != {:?}",
                    tensor.shape,
                    expected[name]
                )
            }
        }
        total_size = total_size
            .checked_add(header.file_len)
            .ok_or_else(|| candle::Error::Msg("H3 shard size overflow".into()))?;
        identity.update(shard.as_bytes());
        identity.update(header.header_len.to_le_bytes());
        identity.update(header.file_len.to_le_bytes());
        let mut file = open_expected_file(&file_identity, "before header identity read")?;
        let mut bytes = vec![0u8; (header.header_len + 8) as usize];
        file.read_exact(&mut bytes)?;
        identity.update(bytes);
        ensure_open_file_unchanged(&file, &file_identity, "during header identity read")?;
        ensure_identity_unchanged(&file_identity, "during validation")?;
        files.push(file_identity);
    }
    Ok(ValidatedVisualVaeWeights {
        layout: WeightLayout::Diffusers,
        inspection: VisualVaeWeightInspection {
            role: VisualVaeComponentRole::F16T4D24.stable_id(),
            shard_count: by_shard.len(),
            tensor_count: expected.len(),
            total_size,
            header_identity_sha256: hex_digest(identity.finalize()),
        },
        files,
    })
}

pub const COMFY_VISUAL_VAE_FILENAME: &str = "minimax_h3_video_vae_fp16.safetensors";

/// Validate the exact pinned Comfy single-file source layout. Unlike the
/// Diffusers conversion, this artifact is genuinely FP16 and must remain FP16
/// in resident storage.
pub fn validate_comfy_weight_file(
    config: &MiniMaxH3VisualVaeConfig,
    weight_path: &Path,
) -> Result<ValidatedVisualVaeWeights> {
    validate_comfy_weight_file_inner(config, weight_path, true)
}

/// Validate the Comfy visual VAE through a retained process descriptor.
/// Descriptor paths intentionally have no model filename; the already-opened
/// inode remains authoritative even if a writable staging parent is replaced.
pub fn validate_comfy_weight_file_from_opened_descriptor(
    config: &MiniMaxH3VisualVaeConfig,
    weight_path: &Path,
) -> Result<ValidatedVisualVaeWeights> {
    let parent = weight_path.parent();
    let is_descriptor =
        parent == Some(Path::new("/dev/fd")) || parent == Some(Path::new("/proc/self/fd"));
    let has_numeric_descriptor = weight_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_digit()));
    if !is_descriptor || !has_numeric_descriptor {
        bail!("H3 Comfy visual VAE opened authority must use a process descriptor path")
    }
    validate_comfy_weight_file_inner(config, weight_path, false)
}

fn validate_comfy_weight_file_inner(
    config: &MiniMaxH3VisualVaeConfig,
    weight_path: &Path,
    require_canonical_filename: bool,
) -> Result<ValidatedVisualVaeWeights> {
    config.validate_production_contract()?;
    if require_canonical_filename {
        let basename = weight_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| candle::Error::Msg("H3 Comfy VAE path has no UTF-8 basename".into()))?;
        if basename != COMFY_VISUAL_VAE_FILENAME {
            bail!(
                "H3 Comfy visual VAE must use canonical filename {COMFY_VISUAL_VAE_FILENAME}, got {basename}"
            )
        }
    }
    let file_identity = if require_canonical_filename {
        visual_weight_file_identity(weight_path)?
    } else {
        visual_weight_file_identity_from_descriptor(weight_path)?
    };
    if require_canonical_filename {
        let canonical_basename = file_identity
            .canonical_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                candle::Error::Msg("H3 Comfy VAE canonical path has no UTF-8 basename".into())
            })?;
        if canonical_basename != COMFY_VISUAL_VAE_FILENAME {
            bail!(
                "H3 Comfy visual VAE canonical file must be named {COMFY_VISUAL_VAE_FILENAME}, got {canonical_basename}"
            )
        }
    }
    let expected = expected_comfy_weight_shapes(config)?;
    let header = inspect_safetensors_header_for_identity(&file_identity)?;
    validate_tensor_headers(&header, &expected, "F16", "Comfy")?;

    let mut identity = Sha256::new();
    identity.update(VisualVaeComponentRole::F16T4D24.stable_id().as_bytes());
    identity.update(b"comfy-source-f16");
    identity.update(COMFY_VISUAL_VAE_FILENAME.as_bytes());
    identity.update(header.header_len.to_le_bytes());
    identity.update(header.file_len.to_le_bytes());
    let mut file = open_expected_file(&file_identity, "before header identity read")?;
    let mut bytes = vec![0u8; (header.header_len + 8) as usize];
    file.read_exact(&mut bytes)?;
    identity.update(bytes);
    ensure_open_file_unchanged(&file, &file_identity, "during header identity read")?;
    ensure_identity_unchanged(&file_identity, "during validation")?;

    Ok(ValidatedVisualVaeWeights {
        layout: WeightLayout::OfficialComfySource,
        inspection: VisualVaeWeightInspection {
            role: VisualVaeComponentRole::F16T4D24.stable_id(),
            shard_count: 1,
            tensor_count: expected.len(),
            total_size: header.file_len,
            header_identity_sha256: hex_digest(identity.finalize()),
        },
        files: vec![file_identity],
    })
}

fn validate_tensor_headers(
    header: &SafetensorsHeader,
    expected: &BTreeMap<String, Vec<usize>>,
    expected_dtype: &str,
    label: &str,
) -> Result<()> {
    validate_weight_map_keys(expected.keys(), header.tensors.keys())?;
    for (name, shape) in expected {
        let tensor = &header.tensors[name];
        if tensor.dtype != expected_dtype {
            bail!(
                "H3 {label} visual VAE tensor {name} must be stored {expected_dtype}, got {}",
                tensor.dtype
            )
        }
        if tensor.shape != *shape {
            bail!(
                "H3 {label} visual VAE tensor {name} shape mismatch: {:?} != {:?}",
                tensor.shape,
                shape
            )
        }
    }
    Ok(())
}

fn validate_weight_map_keys<'a>(
    expected: impl Iterator<Item = &'a String>,
    actual: impl Iterator<Item = &'a String>,
) -> Result<()> {
    let actual_keys = actual.cloned().collect::<BTreeSet<_>>();
    let expected_keys = expected.cloned().collect::<BTreeSet<_>>();
    if actual_keys != expected_keys {
        let missing = expected_keys
            .difference(&actual_keys)
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = actual_keys
            .difference(&expected_keys)
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        bail!("H3 visual VAE weight index mismatch: missing={missing:?} unexpected={unexpected:?}")
    }
    Ok(())
}

/// Full content fingerprint. This intentionally reads every byte on the first
/// immutable file identity and must only be called after Mold's H3
/// authorization gate permits artifact access.
pub fn component_fingerprint(paths: &[PathBuf]) -> Result<String> {
    component_fingerprint_with_observer(paths, &mut NoopVisualWeightReadObserver)
}

pub fn component_fingerprint_with_observer(
    paths: &[PathBuf],
    observer: &mut dyn VisualWeightReadObserver,
) -> Result<String> {
    if paths.is_empty() {
        bail!("cannot fingerprint an empty H3 visual VAE component")
    }
    let identities = paths
        .iter()
        .map(|path| visual_weight_file_identity(path))
        .collect::<Result<Vec<_>>>()?;
    fingerprint_identities(&identities, observer)
}

fn fingerprint_identities(
    identities: &[VisualWeightFileIdentity],
    observer: &mut dyn VisualWeightReadObserver,
) -> Result<String> {
    let mut sorted = identities.to_vec();
    sorted.sort_by(|left, right| left.canonical_path.cmp(&right.canonical_path));
    if sorted
        .windows(2)
        .any(|pair| pair[0].canonical_path == pair[1].canonical_path)
    {
        bail!("H3 fingerprint paths contain duplicate canonical files")
    }
    let total_bytes = sorted.iter().try_fold(0u64, |total, identity| {
        total
            .checked_add(identity.len)
            .ok_or_else(|| candle::Error::Msg("H3 fingerprint byte count overflow".into()))
    })?;
    observer.checkpoint(0, total_bytes)?;
    let mut digest = Sha256::new();
    digest.update(VisualVaeComponentRole::F16T4D24.stable_id().as_bytes());
    let mut completed = 0u64;
    for identity in sorted {
        let name = identity.logical_name.as_str();
        validate_shard_basename(name)?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(identity.len.to_le_bytes());
        let file_digest = fingerprint_file(&identity, completed, total_bytes, observer)?;
        digest.update(file_digest);
        completed = completed
            .checked_add(identity.len)
            .ok_or_else(|| candle::Error::Msg("H3 fingerprint progress overflow".into()))?;
        observer.checkpoint(completed, total_bytes)?;
    }
    Ok(hex_digest(digest.finalize()))
}

fn fingerprint_file(
    identity: &VisualWeightFileIdentity,
    completed_before: u64,
    total_bytes: u64,
    observer: &mut dyn VisualWeightReadObserver,
) -> Result<[u8; 32]> {
    static CACHE: OnceLock<Mutex<BTreeMap<VisualWeightFileIdentity, [u8; 32]>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(digest) = cache
        .lock()
        .map_err(|_| candle::Error::Msg("H3 fingerprint cache mutex is poisoned".into()))?
        .get(identity)
        .copied()
    {
        observer.checkpoint(completed_before, total_bytes)?;
        return Ok(digest);
    }

    let mut file = File::open(&identity.canonical_path)?;
    let opened = visual_weight_file_identity_from_metadata(
        identity.canonical_path.clone(),
        &file.metadata()?,
    )?;
    if !same_visual_weight_file_identity(&opened, identity) {
        bail!(
            "H3 visual VAE artifact changed before fingerprinting: {}",
            identity.canonical_path.display()
        )
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut read_total = 0u64;
    loop {
        observer.checkpoint(completed_before + read_total, total_bytes)?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        read_total = read_total
            .checked_add(read as u64)
            .ok_or_else(|| candle::Error::Msg("H3 fingerprint progress overflow".into()))?;
        digest.update(&buffer[..read]);
    }
    if read_total != identity.len {
        bail!(
            "H3 visual VAE artifact length changed while fingerprinting: {}",
            identity.canonical_path.display()
        )
    }
    let closed_identity = visual_weight_file_identity_from_metadata(
        identity.canonical_path.clone(),
        &file.metadata()?,
    )?;
    if !same_visual_weight_file_identity(&closed_identity, identity)
        || !same_visual_weight_file_identity(
            &current_visual_weight_file_identity(identity)?,
            identity,
        )
    {
        bail!(
            "H3 visual VAE artifact changed while fingerprinting: {}",
            identity.canonical_path.display()
        )
    }
    let digest: [u8; 32] = digest.finalize().into();
    cache
        .lock()
        .map_err(|_| candle::Error::Msg("H3 fingerprint cache mutex is poisoned".into()))?
        .insert(identity.clone(), digest);
    Ok(digest)
}

fn visual_weight_file_identity(path: &Path) -> Result<VisualWeightFileIdentity> {
    let canonical_path = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical_path)?;
    visual_weight_file_identity_from_metadata(canonical_path, &metadata)
}

fn visual_weight_file_identity_from_descriptor(path: &Path) -> Result<VisualWeightFileIdentity> {
    let metadata = std::fs::metadata(path)?;
    visual_weight_file_identity_from_metadata(path.to_path_buf(), &metadata)
}

fn current_visual_weight_file_identity(
    expected: &VisualWeightFileIdentity,
) -> Result<VisualWeightFileIdentity> {
    if expected.descriptor_bound {
        visual_weight_file_identity_from_descriptor(&expected.canonical_path)
    } else {
        visual_weight_file_identity(&expected.canonical_path)
    }
}

fn ensure_identity_unchanged(expected: &VisualWeightFileIdentity, operation: &str) -> Result<()> {
    if !same_visual_weight_file_identity(&current_visual_weight_file_identity(expected)?, expected)
    {
        bail!(
            "H3 visual VAE artifact changed {operation}: {}",
            expected.canonical_path.display()
        )
    }
    Ok(())
}

fn open_expected_file(expected: &VisualWeightFileIdentity, operation: &str) -> Result<File> {
    let mut file = File::open(&expected.canonical_path)?;
    file.seek(SeekFrom::Start(0))?;
    ensure_open_file_unchanged(&file, expected, operation)?;
    Ok(file)
}

fn ensure_open_file_unchanged(
    file: &File,
    expected: &VisualWeightFileIdentity,
    operation: &str,
) -> Result<()> {
    let opened = visual_weight_file_identity_from_metadata(
        expected.canonical_path.clone(),
        &file.metadata()?,
    )?;
    if !same_visual_weight_file_identity(&opened, expected) {
        bail!(
            "H3 visual VAE artifact changed {operation}: {}",
            expected.canonical_path.display()
        )
    }
    Ok(())
}

fn same_visual_weight_file_identity(
    opened: &VisualWeightFileIdentity,
    expected: &VisualWeightFileIdentity,
) -> bool {
    #[cfg(all(unix, not(target_os = "linux")))]
    if expected.descriptor_bound {
        // macOS reports a synthetic device id when statting `/dev/fd/N`, but
        // the descriptor's own metadata retains the underlying filesystem id.
        return opened.canonical_path == expected.canonical_path
            && opened.descriptor_bound == expected.descriptor_bound
            && opened.len == expected.len
            && opened.inode == expected.inode
            && opened.modified_seconds == expected.modified_seconds
            && opened.modified_nanoseconds == expected.modified_nanoseconds
            && opened.changed_seconds == expected.changed_seconds
            && opened.changed_nanoseconds == expected.changed_nanoseconds;
    }
    opened == expected
}

fn visual_weight_file_identity_from_metadata(
    canonical_path: PathBuf,
    metadata: &std::fs::Metadata,
) -> Result<VisualWeightFileIdentity> {
    if !metadata.is_file() {
        bail!(
            "H3 visual VAE artifact must be a regular file: {}",
            canonical_path.display()
        )
    }
    let descriptor_bound = canonical_path.parent().is_some_and(|parent| {
        parent == Path::new("/dev/fd") || parent == Path::new("/proc/self/fd")
    });
    let logical_name = if descriptor_bound {
        COMFY_VISUAL_VAE_FILENAME.to_owned()
    } else {
        canonical_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| candle::Error::Msg("H3 visual VAE path has no UTF-8 basename".into()))?
            .to_owned()
    };
    Ok(VisualWeightFileIdentity {
        canonical_path,
        logical_name,
        descriptor_bound,
        len: metadata.len(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        modified_seconds: metadata.mtime(),
        #[cfg(unix)]
        modified_nanoseconds: metadata.mtime_nsec(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
        #[cfg(not(unix))]
        modified: metadata.modified().ok(),
    })
}

fn expected_data_size(
    expected: &BTreeMap<String, Vec<usize>>,
    bytes_per_element: u64,
) -> Result<u64> {
    expected.values().try_fold(0u64, |total, shape| {
        let elements = shape.iter().try_fold(1u64, |elements, dimension| {
            elements
                .checked_mul(*dimension as u64)
                .ok_or_else(|| candle::Error::Msg("H3 visual VAE expected shape overflow".into()))
        })?;
        let bytes = elements.checked_mul(bytes_per_element).ok_or_else(|| {
            candle::Error::Msg("H3 visual VAE expected byte-size overflow".into())
        })?;
        total
            .checked_add(bytes)
            .ok_or_else(|| candle::Error::Msg("H3 visual VAE total byte-size overflow".into()))
    })
}

fn validate_canonical_shard_names(shards: &BTreeSet<String>) -> Result<()> {
    if shards.is_empty() {
        bail!("H3 visual VAE weight index does not name any shards")
    }
    let count = shards.len();
    let expected = (1..=count)
        .map(|index| format!("diffusion_pytorch_model-{index:05}-of-{count:05}.safetensors"))
        .collect::<BTreeSet<_>>();
    if shards != &expected {
        bail!("H3 visual VAE index uses a non-canonical shard sequence")
    }
    Ok(())
}

fn safetensors_files(component_dir: &Path) -> Result<BTreeSet<String>> {
    let mut files = BTreeSet::new();
    for entry in std::fs::read_dir(component_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        if !entry.file_type()?.is_file() {
            bail!(
                "H3 visual VAE shard {} must be a regular file",
                path.display()
            )
        }
        let name = entry.file_name().into_string().map_err(|_| {
            candle::Error::Msg("H3 visual VAE shard name is not valid UTF-8".into())
        })?;
        if !files.insert(name.clone()) {
            bail!("duplicate H3 visual VAE shard name {name}")
        }
    }
    Ok(files)
}

fn validate_shard_basename(shard: &str) -> Result<()> {
    let path = Path::new(shard);
    if path.components().count() != 1
        || path.extension().and_then(|extension| extension.to_str()) != Some("safetensors")
    {
        bail!("unsafe or invalid H3 visual VAE shard name {shard:?}")
    }
    Ok(())
}

fn dtype_size(dtype: &str) -> Result<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Ok(1),
        "I16" | "U16" | "F16" | "BF16" => Ok(2),
        "I32" | "U32" | "F32" => Ok(4),
        "I64" | "U64" | "F64" => Ok(8),
        other => bail!("unsupported safetensors dtype {other}"),
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn write_sparse_production_comfy_weights(path: &Path) -> File {
        use std::fs::OpenOptions;
        use std::io::Write;

        let expected =
            expected_comfy_weight_shapes(&MiniMaxH3VisualVaeConfig::production()).unwrap();
        let mut offset = 0u64;
        let mut header = serde_json::Map::new();
        for (name, shape) in expected {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                name,
                serde_json::json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file
    }

    #[test]
    fn production_key_contract_has_the_pinned_703_tensors() {
        let shapes =
            expected_diffusers_weight_shapes(&MiniMaxH3VisualVaeConfig::production()).unwrap();
        assert_eq!(shapes.len(), 703);
        assert_eq!(shapes["encoder.conv_in.weight"], vec![128, 3, 3, 3, 3]);
        assert_eq!(
            shapes["decoder.transformer_blocks.35.ff.net.0.proj.weight"],
            vec![16384, 2048]
        );
        assert_eq!(shapes["decoder.proj_out.weight"], vec![3072, 2048]);
    }

    #[test]
    fn comfy_contract_has_exact_562_source_tensors_and_fp16_payload() {
        let shapes = expected_comfy_weight_shapes(&MiniMaxH3VisualVaeConfig::production()).unwrap();
        assert_eq!(shapes.len(), 562);
        assert_eq!(
            shapes["decoder.transformer_blocks.35.attn.to_qkv.weight"],
            vec![6144, 2048]
        );
        assert_eq!(
            shapes["decoder.transformer_blocks.35.ff.w1.weight"],
            vec![16384, 2048]
        );
        assert_eq!(shapes["decoder.mask_token"], vec![1, 1, 2048]);
        assert_eq!(shapes["latents_mean"], vec![24]);
        assert_eq!(shapes["latents_std"], vec![24]);
        let elements = shapes
            .values()
            .map(|shape| shape.iter().product::<usize>() as u64)
            .sum::<u64>();
        assert_eq!(elements * 2, 5_207_742_160);
        assert_eq!(
            expected_visual_vae_parameter_bytes(
                &MiniMaxH3VisualVaeConfig::production(),
                DType::F16
            )
            .unwrap(),
            5_207_737_968
        );
    }

    #[test]
    fn comfy_header_contract_rejects_key_dtype_and_shape_drift() {
        let expected =
            expected_comfy_weight_shapes(&MiniMaxH3VisualVaeConfig::tiny_for_tests()).unwrap();
        let mut offset = 0u64;
        let tensors = expected
            .iter()
            .map(|(name, shape)| {
                let bytes = shape.iter().product::<usize>() as u64 * 2;
                let tensor = SafetensorsTensorHeader {
                    dtype: "F16".into(),
                    shape: shape.clone(),
                    data_offsets: [offset, offset + bytes],
                };
                offset += bytes;
                (name.clone(), tensor)
            })
            .collect::<BTreeMap<_, _>>();
        let header = SafetensorsHeader {
            tensors,
            header_len: 1,
            file_len: offset + 9,
        };
        validate_tensor_headers(&header, &expected, "F16", "Comfy").unwrap();

        let mut bad_dtype = header.clone();
        bad_dtype
            .tensors
            .get_mut("decoder.mask_token")
            .unwrap()
            .dtype = "F32".into();
        assert!(validate_tensor_headers(&bad_dtype, &expected, "F16", "Comfy").is_err());

        let mut bad_shape = header.clone();
        bad_shape
            .tensors
            .get_mut("decoder.mask_token")
            .unwrap()
            .shape = vec![1, 2, 8];
        assert!(validate_tensor_headers(&bad_shape, &expected, "F16", "Comfy").is_err());

        let mut unexpected = header;
        unexpected.tensors.insert(
            "decoder.unplanned".into(),
            SafetensorsTensorHeader {
                dtype: "F16".into(),
                shape: vec![1],
                data_offsets: [offset, offset + 2],
            },
        );
        assert!(validate_tensor_headers(&unexpected, &expected, "F16", "Comfy").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn opened_comfy_validator_and_mmap_survive_storage_path_replacement() {
        use std::os::fd::AsRawFd;

        let directory = tempfile::tempdir().unwrap();
        let storage_path = directory.path().join(COMFY_VISUAL_VAE_FILENAME);
        let retained = write_sparse_production_comfy_weights(&storage_path);
        #[cfg(target_os = "linux")]
        let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", retained.as_raw_fd()));
        #[cfg(not(target_os = "linux"))]
        let descriptor_path = PathBuf::from(format!("/dev/fd/{}", retained.as_raw_fd()));

        let authenticated_path = directory.path().join("authenticated.safetensors");
        std::fs::rename(&storage_path, &authenticated_path).unwrap();
        std::fs::write(&storage_path, b"replacement").unwrap();

        let validated = validate_comfy_weight_file_from_opened_descriptor(
            &MiniMaxH3VisualVaeConfig::production(),
            &descriptor_path,
        )
        .unwrap();
        validated.revalidate_files().unwrap();
        validated.mmap_var_builder(&Device::Cpu).unwrap();
        assert_eq!(validated.files[0].canonical_path, descriptor_path);
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_bound_fingerprint_survives_storage_path_replacement() {
        use std::os::fd::AsRawFd;

        let directory = tempfile::tempdir().unwrap();
        let storage_path = directory.path().join("weights.safetensors");
        std::fs::write(&storage_path, b"authenticated bytes").unwrap();
        let retained = File::open(&storage_path).unwrap();
        #[cfg(target_os = "linux")]
        let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", retained.as_raw_fd()));
        #[cfg(not(target_os = "linux"))]
        let descriptor_path = PathBuf::from(format!("/dev/fd/{}", retained.as_raw_fd()));
        let authenticated_path = directory.path().join("authenticated.safetensors");
        std::fs::rename(&storage_path, authenticated_path).unwrap();
        std::fs::write(&storage_path, b"replacement").unwrap();

        let identity = visual_weight_file_identity_from_descriptor(&descriptor_path).unwrap();
        let token = ValidatedVisualVaeWeights {
            layout: WeightLayout::OfficialComfySource,
            inspection: VisualVaeWeightInspection {
                role: VisualVaeComponentRole::F16T4D24.stable_id(),
                shard_count: 1,
                tensor_count: 0,
                total_size: identity.len,
                header_identity_sha256: "test".into(),
            },
            files: vec![identity],
        };
        let digest = token
            .component_fingerprint(&mut NoopVisualWeightReadObserver)
            .unwrap();
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn config_inspection_accepts_only_the_released_contract() {
        let config = MiniMaxH3VisualVaeConfig::production();
        let bytes = serde_json::to_vec(&config).unwrap();
        assert_eq!(inspect_visual_vae_config_bytes(&bytes).unwrap(), config);
        assert!(inspect_visual_vae_config_bytes(b"{}").is_err());

        let mut value = serde_json::to_value(config).unwrap();
        value["latent_channels"] = serde_json::Value::from(16);
        assert!(inspect_visual_vae_config_bytes(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut value = serde_json::to_value(MiniMaxH3VisualVaeConfig::production()).unwrap();
        value["spatial_padding_mode"] = serde_json::Value::from("zeros");
        assert!(inspect_visual_vae_config_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn comfy_mapping_names_old_encoder_and_fused_qkv_without_guessing() {
        let config = MiniMaxH3VisualVaeConfig::tiny_for_tests();
        let transforms = comfy_tensor_transforms(&config).unwrap();
        assert!(transforms.contains(&ComfyTensorTransform::Direct {
            source: "encoder.down.1.block.0.nin_shortcut.weight".into(),
            target: "encoder.down_blocks.1.resnets.0.conv_shortcut.weight".into(),
        }));
        assert!(
            transforms.contains(&ComfyTensorTransform::ReorderAndSplitQkv {
                source: "decoder.transformer_blocks.0.attn.to_qkv.weight".into(),
                query: "decoder.transformer_blocks.0.attn.to_q.weight".into(),
                key: "decoder.transformer_blocks.0.attn.to_k.weight".into(),
                value: "decoder.transformer_blocks.0.attn.to_v.weight".into(),
                num_attention_heads: 1,
                attention_head_dim: 8,
            })
        );
        assert!(
            transforms.contains(&ComfyTensorTransform::SwapFeedForwardHalves {
                source: "decoder.transformer_blocks.0.ff.w1.weight".into(),
                target: "decoder.transformer_blocks.0.ff.net.0.proj.weight".into(),
                split_axis: 0,
            })
        );
        assert!(transforms.contains(&ComfyTensorTransform::Drop {
            source: "decoder.mask_token".into(),
        }));
        assert!(transforms.contains(&ComfyTensorTransform::Drop {
            source: "latents_mean".into(),
        }));
        assert!(transforms.contains(&ComfyTensorTransform::Drop {
            source: "latents_std".into(),
        }));
    }

    #[test]
    fn comfy_source_plan_covers_every_source_and_target_exactly_once() {
        let config = MiniMaxH3VisualVaeConfig::production();
        let transforms = comfy_tensor_transforms(&config).unwrap();
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for transform in transforms {
            let (source, outputs): (&str, Vec<&str>) = match &transform {
                ComfyTensorTransform::Direct { source, target }
                | ComfyTensorTransform::SwapFeedForwardHalves { source, target, .. } => {
                    (source, vec![target])
                }
                ComfyTensorTransform::ReorderAndSplitQkv {
                    source,
                    query,
                    key,
                    value,
                    ..
                } => (source, vec![query, key, value]),
                ComfyTensorTransform::Drop { source } => (source, vec![]),
            };
            assert!(
                sources.insert(source.to_owned()),
                "duplicate source {source}"
            );
            for target in outputs {
                assert!(
                    targets.insert(target.to_owned()),
                    "duplicate target {target}"
                );
            }
        }
        assert_eq!(sources.len(), 562);
        assert_eq!(
            targets,
            expected_diffusers_weight_shapes(&config)
                .unwrap()
                .into_keys()
                .collect()
        );
    }

    #[test]
    fn weight_map_drift_fails_closed_for_missing_and_unexpected_keys() {
        let expected =
            expected_diffusers_weight_shapes(&MiniMaxH3VisualVaeConfig::production()).unwrap();
        let mut actual = expected.keys().cloned().collect::<BTreeSet<_>>();
        actual.remove("encoder.conv_in.weight");
        actual.insert("encoder.unplanned.weight".into());
        let error = validate_weight_map_keys(expected.keys(), actual.iter())
            .unwrap_err()
            .to_string();
        assert!(error.contains("encoder.conv_in.weight"));
        assert!(error.contains("encoder.unplanned.weight"));
    }

    #[test]
    fn header_inspection_reads_metadata_and_rejects_bad_byte_spans() {
        let dir = std::env::temp_dir().join(format!(
            "mold-h3-header-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        let header = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, bytes).unwrap();
        let inspected = inspect_safetensors_header(&path).unwrap();
        assert_eq!(inspected.tensors["x"].shape, vec![2]);

        let bad = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (bad.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(bad);
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&path, bytes).unwrap();
        assert!(inspect_safetensors_header(&path).is_err());

        let duplicate = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (duplicate.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(duplicate);
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&path, bytes).unwrap();
        assert!(inspect_safetensors_header(&path)
            .unwrap_err()
            .to_string()
            .contains("duplicate safetensors tensor x"));

        let overlapping = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut bytes = (overlapping.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(overlapping);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, bytes).unwrap();
        assert!(inspect_safetensors_header(&path).is_err());

        let gap = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
        let mut bytes = (gap.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(gap);
        bytes.extend_from_slice(&[0u8; 12]);
        std::fs::write(&path, bytes).unwrap();
        assert!(inspect_safetensors_header(&path).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn shard_sequence_is_canonical_and_complete() {
        let canonical = [
            "diffusion_pytorch_model-00001-of-00003.safetensors".to_owned(),
            "diffusion_pytorch_model-00002-of-00003.safetensors".to_owned(),
            "diffusion_pytorch_model-00003-of-00003.safetensors".to_owned(),
        ]
        .into_iter()
        .collect();
        assert!(validate_canonical_shard_names(&canonical).is_ok());

        let incomplete = [
            "diffusion_pytorch_model-00001-of-00003.safetensors".to_owned(),
            "diffusion_pytorch_model-00003-of-00003.safetensors".to_owned(),
        ]
        .into_iter()
        .collect();
        assert!(validate_canonical_shard_names(&incomplete).is_err());
    }

    #[test]
    fn fingerprint_is_cancellable_cached_and_invalidates_on_file_change() {
        struct CancelAfter {
            limit: u64,
            calls: Vec<u64>,
        }
        impl VisualWeightReadObserver for CancelAfter {
            fn checkpoint(&mut self, bytes_read: u64, _total_bytes: u64) -> Result<()> {
                self.calls.push(bytes_read);
                if bytes_read >= self.limit {
                    bail!("synthetic fingerprint cancellation")
                }
                Ok(())
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "mold-h3-fingerprint-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.safetensors");
        std::fs::write(&path, vec![7u8; 2 * 1024 * 1024 + 17]).unwrap();

        let mut cancel = CancelAfter {
            limit: 1024 * 1024,
            calls: Vec::new(),
        };
        let error = component_fingerprint_with_observer(std::slice::from_ref(&path), &mut cancel)
            .unwrap_err()
            .to_string();
        assert!(error.contains("synthetic fingerprint cancellation"));
        assert!(cancel.calls.contains(&(1024 * 1024)));

        let fingerprint = component_fingerprint(std::slice::from_ref(&path)).unwrap();
        let mut cached = CancelAfter {
            limit: u64::MAX,
            calls: Vec::new(),
        };
        assert_eq!(
            component_fingerprint_with_observer(std::slice::from_ref(&path), &mut cached).unwrap(),
            fingerprint
        );
        assert!(
            cached.calls.len() <= 3,
            "cached fingerprint reread the file"
        );

        let token = ValidatedVisualVaeWeights {
            layout: WeightLayout::OfficialComfySource,
            inspection: VisualVaeWeightInspection {
                role: VisualVaeComponentRole::F16T4D24.stable_id(),
                shard_count: 1,
                tensor_count: 0,
                total_size: std::fs::metadata(&path).unwrap().len(),
                header_identity_sha256: "test".into(),
            },
            files: vec![visual_weight_file_identity(&path).unwrap()],
        };
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.push(9);
        std::fs::write(&path, bytes).unwrap();
        assert!(token.revalidate_files().is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
