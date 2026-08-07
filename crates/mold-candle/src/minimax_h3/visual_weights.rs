use super::visual_vae::MiniMaxH3VisualVaeConfig;
use candle::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
/// rename. `decoder.mask_token` is deliberately absent: it is checkpoint-only
/// training state and unused at inference.
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
    Ok(transforms)
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
    const MAX_HEADER: u64 = 100_000_000;
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > MAX_HEADER || header_len + 8 > file_len {
        bail!(
            "invalid safetensors header length {header_len} for {}",
            path.display()
        )
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let raw: BTreeMap<String, serde_json::Value> =
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
                path.display()
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
                path.display()
            )
        }
        cursor = offsets[1];
    }
    if cursor != data_len {
        bail!(
            "safetensors data section has {} unclaimed bytes in {}",
            data_len - cursor,
            path.display()
        )
    }
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
    parse_visual_vae_config(&bytes)
}

fn parse_visual_vae_config(bytes: &[u8]) -> Result<MiniMaxH3VisualVaeConfig> {
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
) -> Result<VisualVaeWeightInspection> {
    config.validate_production_contract()?;
    let index_bytes = std::fs::read(index_path)?;
    let index: DiffusersWeightIndex =
        serde_json::from_slice(&index_bytes).map_err(candle::Error::wrap)?;
    let expected = expected_diffusers_weight_shapes(config)?;
    validate_weight_map_keys(expected.keys(), index.weight_map.keys())?;
    let expected_data_size = expected_f32_data_size(&expected)?;
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
    identity.update(VisualVaeComponentRole::F16T4D24.stable_id().as_bytes());
    identity.update(&index_bytes);
    for (shard, assigned) in &by_shard {
        let path = component_dir.join(shard);
        let header = inspect_safetensors_header(&path)?;
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
        let mut file = File::open(&path)?;
        let mut bytes = vec![0u8; (header.header_len + 8) as usize];
        file.read_exact(&mut bytes)?;
        identity.update(bytes);
    }
    Ok(VisualVaeWeightInspection {
        role: VisualVaeComponentRole::F16T4D24.stable_id(),
        shard_count: by_shard.len(),
        tensor_count: expected.len(),
        total_size,
        header_identity_sha256: hex_digest(identity.finalize()),
    })
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

/// Full content fingerprint. This intentionally reads every byte and must only
/// be called after Mold's H3 authorization gate permits artifact access.
pub fn component_fingerprint(paths: &[PathBuf]) -> Result<String> {
    if paths.is_empty() {
        bail!("cannot fingerprint an empty H3 visual VAE component")
    }
    let mut sorted = paths
        .iter()
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    candle::Error::Msg("H3 fingerprint path has no UTF-8 basename".into())
                })?;
            validate_shard_basename(name)?;
            Ok((name.to_owned(), path.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    sorted.sort_by(|(left, _), (right, _)| left.cmp(right));
    if sorted
        .windows(2)
        .any(|pair| pair[0].0.as_str() == pair[1].0.as_str())
    {
        bail!("H3 fingerprint paths contain duplicate shard basenames")
    }
    let mut digest = Sha256::new();
    digest.update(VisualVaeComponentRole::F16T4D24.stable_id().as_bytes());
    let mut buffer = vec![0u8; 1024 * 1024];
    for (name, path) in sorted {
        let file_len = std::fs::metadata(&path)?.len();
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(file_len.to_le_bytes());
        let mut file = File::open(&path)?;
        file.seek(SeekFrom::Start(0))?;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(hex_digest(digest.finalize()))
}

fn expected_f32_data_size(expected: &BTreeMap<String, Vec<usize>>) -> Result<u64> {
    expected.values().try_fold(0u64, |total, shape| {
        let elements = shape.iter().try_fold(1u64, |elements, dimension| {
            elements
                .checked_mul(*dimension as u64)
                .ok_or_else(|| candle::Error::Msg("H3 visual VAE expected shape overflow".into()))
        })?;
        let bytes = elements.checked_mul(4).ok_or_else(|| {
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
    fn config_inspection_accepts_only_the_released_contract() {
        let config = MiniMaxH3VisualVaeConfig::production();
        let bytes = serde_json::to_vec(&config).unwrap();
        assert_eq!(parse_visual_vae_config(&bytes).unwrap(), config);
        assert!(parse_visual_vae_config(b"{}").is_err());

        let mut value = serde_json::to_value(config).unwrap();
        value["latent_channels"] = serde_json::Value::from(16);
        assert!(parse_visual_vae_config(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut value = serde_json::to_value(MiniMaxH3VisualVaeConfig::production()).unwrap();
        value["spatial_padding_mode"] = serde_json::Value::from("zeros");
        assert!(parse_visual_vae_config(&serde_json::to_vec(&value).unwrap()).is_err());
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
        assert_eq!(sources.len(), 560);
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
}
