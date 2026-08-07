//! Strict MiniMax H3 audio-VAE safetensors contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use candle::{bail, DType, Device, Result};
use candle_nn::VarBuilder;
use serde_json::Value;

use super::audio::AudioVae;
use super::audio_config::AudioVaeConfig;

const MAX_HEADER_BYTES: usize = 100 * 1024 * 1024;

/// Original/Diffusers H3 files retain PyTorch `weight_g`/`weight_v`.
/// Comfy's pruned file folds weight normalization into a plain `weight`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioTensorLayout {
    OfficialWeightNorm,
    ComfyFolded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTensorSpec {
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedAudioWeights {
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub layout: AudioTensorLayout,
}

pub fn audio_tensor_specs(
    config: &AudioVaeConfig,
    layout: AudioTensorLayout,
) -> Result<BTreeMap<String, AudioTensorSpec>> {
    config.validate()?;
    let mut specs = BTreeMap::new();

    add_weight_norm_conv(
        &mut specs,
        "encoder.block.0",
        config.encoder_dim,
        1,
        7,
        true,
        layout,
    );
    let mut dim = config.encoder_dim;
    for (stage, stride) in config.encoder_rates.iter().copied().enumerate() {
        dim *= 2;
        let prefix = format!("encoder.block.{}.block", stage + 1);
        let half = dim / 2;
        for (residual, dilation) in [1usize, 3, 9].iter().copied().enumerate() {
            let residual_prefix = format!("{prefix}.{residual}.block");
            add(
                &mut specs,
                format!("{residual_prefix}.0.alpha"),
                [1, half, 1],
            );
            add_weight_norm_conv(
                &mut specs,
                &format!("{residual_prefix}.1"),
                half,
                half,
                7,
                true,
                layout,
            );
            add(
                &mut specs,
                format!("{residual_prefix}.2.alpha"),
                [1, half, 1],
            );
            add_weight_norm_conv(
                &mut specs,
                &format!("{residual_prefix}.3"),
                half,
                half,
                1,
                true,
                layout,
            );
            let _ = dilation;
        }
        add(&mut specs, format!("{prefix}.3.alpha"), [1, half, 1]);
        add_weight_norm_conv(
            &mut specs,
            &format!("{prefix}.4"),
            dim,
            half,
            2 * stride,
            true,
            layout,
        );
    }
    let final_index = config.encoder_rates.len() + 1;
    add(
        &mut specs,
        format!("encoder.block.{final_index}.alpha"),
        [1, dim, 1],
    );
    add_weight_norm_conv(
        &mut specs,
        &format!("encoder.block.{}", final_index + 1),
        config.latent_dim,
        dim,
        3,
        true,
        layout,
    );

    add_layer_norm(&mut specs, "pre_block.norm1", config.latent_dim);
    add(
        &mut specs,
        "pre_block.attn.qkv.weight",
        [config.latent_dim * 3, config.latent_dim],
    );
    add(&mut specs, "pre_block.attn.q_bias", [config.latent_dim]);
    add(
        &mut specs,
        "pre_block.attn.zero_k_bias",
        [config.latent_dim],
    );
    add(&mut specs, "pre_block.attn.v_bias", [config.latent_dim]);
    add_linear(
        &mut specs,
        "pre_block.attn.proj",
        config.latent_channels,
        config.latent_channels,
        true,
    );
    add_linear(
        &mut specs,
        "pre_block.proj",
        config.latent_channels,
        config.latent_dim,
        true,
    );
    add_layer_norm(&mut specs, "pre_block.norm3", config.latent_dim);
    add_layer_norm(&mut specs, "pre_block.norm2", config.latent_channels);
    add_layer_norm(&mut specs, "pre_block.mlp.norm", config.latent_channels);
    add_linear(
        &mut specs,
        "pre_block.mlp.w0",
        config.latent_channels * 2,
        config.latent_channels,
        true,
    );
    add_linear(
        &mut specs,
        "pre_block.mlp.w1",
        config.latent_channels * 2,
        config.latent_channels,
        true,
    );
    add_linear(
        &mut specs,
        "pre_block.mlp.w2",
        config.latent_channels,
        config.latent_channels * 2,
        true,
    );

    for prefix in ["mean_proj", "logs_proj"] {
        add_plain_conv(
            &mut specs,
            prefix,
            config.latent_channels,
            config.latent_channels,
            1,
            true,
        );
    }
    add_plain_conv(
        &mut specs,
        "dec_in_proj",
        config.latent_dim,
        config.latent_channels,
        1,
        true,
    );

    add_weight_norm_conv(
        &mut specs,
        "decoder.conv_pre",
        config.decoder_dim,
        config.latent_dim,
        7,
        true,
        layout,
    );
    let num_kernels = config.resblock_kernel_sizes.len();
    for (stage, kernel) in config.decoder_kernel_sizes.iter().copied().enumerate() {
        let in_channels = config.decoder_dim / (1usize << stage);
        let out_channels = config.decoder_dim / (1usize << (stage + 1));
        add_weight_norm_conv_transpose(
            &mut specs,
            &format!("decoder.ups.{stage}.0"),
            in_channels,
            out_channels,
            kernel,
            true,
            layout,
        );
        for (kernel_index, (res_kernel, dilations)) in config
            .resblock_kernel_sizes
            .iter()
            .zip(&config.resblock_dilation_sizes)
            .enumerate()
        {
            let block = stage * num_kernels + kernel_index;
            let prefix = format!("decoder.resblocks.{block}");
            for index in 0..dilations.len() {
                add_weight_norm_conv(
                    &mut specs,
                    &format!("{prefix}.convs1.{index}"),
                    out_channels,
                    out_channels,
                    *res_kernel,
                    true,
                    layout,
                );
                add_weight_norm_conv(
                    &mut specs,
                    &format!("{prefix}.convs2.{index}"),
                    out_channels,
                    out_channels,
                    *res_kernel,
                    true,
                    layout,
                );
            }
            for index in 0..(dilations.len() * 2) {
                add_alias_free_activation(
                    &mut specs,
                    &format!("{prefix}.activations.{index}"),
                    out_channels,
                );
            }
        }
    }
    let final_channels = config.decoder_dim / (1usize << config.decoder_rates.len());
    add_alias_free_activation(&mut specs, "decoder.activation_post", final_channels);
    add_weight_norm_conv(
        &mut specs,
        "decoder.conv_post",
        1,
        final_channels,
        7,
        false,
        layout,
    );

    if layout == AudioTensorLayout::ComfyFolded {
        add(&mut specs, "latents_mean", [config.latent_channels]);
        add(&mut specs, "latents_std", [config.latent_channels]);
    }
    Ok(specs)
}

pub fn validate_audio_safetensors(
    path: impl AsRef<Path>,
    config: &AudioVaeConfig,
    layout: AudioTensorLayout,
) -> Result<ValidatedAudioWeights> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|error| {
        candle::Error::Msg(format!(
            "failed to open MiniMax H3 audio weights {}: {error}",
            path.display()
        ))
    })?;
    let file_len = file.metadata().map_err(io_error)?.len();
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(io_error)?;
    let header_len = usize::try_from(u64::from_le_bytes(len)).map_err(|_| {
        candle::Error::Msg("audio safetensors header length overflows usize".into())
    })?;
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!("MiniMax H3 audio safetensors header length {header_len} is invalid")
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(io_error)?;
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    validate_header_and_file_len(&header, file_len, config, layout)
}

pub fn load_validated_audio_vae(
    path: impl AsRef<Path>,
    config: AudioVaeConfig,
    layout: AudioTensorLayout,
    requested_dtype: DType,
    device: &Device,
) -> Result<AudioVae> {
    config.validate_execution_dtype(requested_dtype)?;
    let path = path.as_ref();
    validate_audio_safetensors(path, &config, layout)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    if layout == AudioTensorLayout::ComfyFolded {
        validate_comfy_normalization_values(&vb, &config)?;
    }
    AudioVae::load(config, layout, requested_dtype, vb)
}

fn validate_header_and_file_len(
    header: &[u8],
    file_len: u64,
    config: &AudioVaeConfig,
    layout: AudioTensorLayout,
) -> Result<ValidatedAudioWeights> {
    let root: Value = serde_json::from_slice(header).map_err(|error| {
        candle::Error::Msg(format!(
            "invalid MiniMax H3 audio safetensors JSON header: {error}"
        ))
    })?;
    let object = root
        .as_object()
        .ok_or_else(|| candle::Error::Msg("audio safetensors header must be an object".into()))?;
    if let Some(metadata) = object.get("__metadata__") {
        let metadata = metadata.as_object().ok_or_else(|| {
            candle::Error::Msg("audio safetensors __metadata__ must be an object".into())
        })?;
        if let Some(format) = metadata.get("format") {
            if format.as_str() != Some("pt") {
                bail!("MiniMax H3 audio safetensors metadata format must be 'pt'")
            }
        }
    }

    let specs = audio_tensor_specs(config, layout)?;
    let actual_names = object
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect::<BTreeSet<_>>();
    let expected_names = specs.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&actual_names)
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = actual_names
        .difference(&expected_names)
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        bail!(
            "MiniMax H3 audio tensor layout mismatch for {layout:?}: missing={missing:?}, unexpected={unexpected:?}"
        )
    }

    let mut ranges = Vec::with_capacity(specs.len());
    for (name, spec) in &specs {
        let entry = object[name].as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "MiniMax H3 audio tensor {name} header is not an object"
            ))
        })?;
        let fields = entry.keys().map(String::as_str).collect::<BTreeSet<_>>();
        if fields != BTreeSet::from(["data_offsets", "dtype", "shape"]) {
            bail!("MiniMax H3 audio tensor {name} has unexpected header fields")
        }
        if entry.get("dtype").and_then(Value::as_str) != Some("F32") {
            bail!("MiniMax H3 audio tensor {name} must be F32")
        }
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| candle::Error::Msg(format!("audio tensor {name} has no shape")))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        candle::Error::Msg(format!("audio tensor {name} has invalid shape"))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if shape != spec.shape {
            bail!(
                "MiniMax H3 audio tensor {name} shape mismatch: expected {:?}, got {shape:?}",
                spec.shape
            )
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                candle::Error::Msg(format!("audio tensor {name} has invalid data offsets"))
            })?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| candle::Error::Msg(format!("audio tensor {name} start is invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| candle::Error::Msg(format!("audio tensor {name} end is invalid")))?;
        let elements = shape.iter().try_fold(1u64, |product, dim| {
            product
                .checked_mul(*dim as u64)
                .ok_or_else(|| candle::Error::Msg(format!("audio tensor {name} size overflow")))
        })?;
        let expected_bytes = elements
            .checked_mul(4)
            .ok_or_else(|| candle::Error::Msg(format!("audio tensor {name} byte overflow")))?;
        if end.checked_sub(start) != Some(expected_bytes) {
            bail!("MiniMax H3 audio tensor {name} byte span does not match its F32 shape")
        }
        ranges.push((start, end, name));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0u64;
    for (start, end, name) in &ranges {
        if *start != cursor {
            bail!(
                "MiniMax H3 audio safetensors data is not contiguous at {name}: expected offset {cursor}, got {start}"
            )
        }
        cursor = *end;
    }
    let header_bytes = 8u64
        .checked_add(header.len() as u64)
        .ok_or_else(|| candle::Error::Msg("audio header size overflow".into()))?;
    if header_bytes.checked_add(cursor) != Some(file_len) {
        bail!(
            "MiniMax H3 audio safetensors file length mismatch: header {header_bytes} + tensors {cursor} != {file_len}"
        )
    }
    Ok(ValidatedAudioWeights {
        tensor_count: specs.len(),
        tensor_bytes: cursor,
        layout,
    })
}

fn validate_comfy_normalization_values(vb: &VarBuilder, config: &AudioVaeConfig) -> Result<()> {
    for (name, expected) in [
        ("latents_mean", &config.latents_mean),
        ("latents_std", &config.latents_std),
    ] {
        let actual = vb
            .get(config.latent_channels, name)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(expected.iter())
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            bail!("MiniMax H3 Comfy {name} does not match the pinned normalization config")
        }
    }
    Ok(())
}

fn io_error(error: std::io::Error) -> candle::Error {
    candle::Error::Msg(format!("MiniMax H3 audio safetensors I/O error: {error}"))
}

fn add<const N: usize>(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    name: impl Into<String>,
    shape: [usize; N],
) {
    let previous = specs.insert(
        name.into(),
        AudioTensorSpec {
            shape: shape.to_vec(),
        },
    );
    debug_assert!(previous.is_none());
}

fn add_layer_norm(specs: &mut BTreeMap<String, AudioTensorSpec>, prefix: &str, size: usize) {
    add(specs, format!("{prefix}.weight"), [size]);
    add(specs, format!("{prefix}.bias"), [size]);
}

fn add_linear(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    bias: bool,
) {
    add(
        specs,
        format!("{prefix}.weight"),
        [out_features, in_features],
    );
    if bias {
        add(specs, format!("{prefix}.bias"), [out_features]);
    }
}

fn add_plain_conv(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    bias: bool,
) {
    add(
        specs,
        format!("{prefix}.weight"),
        [out_channels, in_channels, kernel],
    );
    if bias {
        add(specs, format!("{prefix}.bias"), [out_channels]);
    }
}

fn add_weight_norm_conv(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    bias: bool,
    layout: AudioTensorLayout,
) {
    match layout {
        AudioTensorLayout::OfficialWeightNorm => {
            add(specs, format!("{prefix}.weight_g"), [out_channels, 1, 1]);
            add(
                specs,
                format!("{prefix}.weight_v"),
                [out_channels, in_channels, kernel],
            );
        }
        AudioTensorLayout::ComfyFolded => add(
            specs,
            format!("{prefix}.weight"),
            [out_channels, in_channels, kernel],
        ),
    }
    if bias {
        add(specs, format!("{prefix}.bias"), [out_channels]);
    }
}

fn add_weight_norm_conv_transpose(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    bias: bool,
    layout: AudioTensorLayout,
) {
    match layout {
        AudioTensorLayout::OfficialWeightNorm => {
            add(specs, format!("{prefix}.weight_g"), [in_channels, 1, 1]);
            add(
                specs,
                format!("{prefix}.weight_v"),
                [in_channels, out_channels, kernel],
            );
        }
        AudioTensorLayout::ComfyFolded => add(
            specs,
            format!("{prefix}.weight"),
            [in_channels, out_channels, kernel],
        ),
    }
    if bias {
        add(specs, format!("{prefix}.bias"), [out_channels]);
    }
}

fn add_alias_free_activation(
    specs: &mut BTreeMap<String, AudioTensorSpec>,
    prefix: &str,
    channels: usize,
) {
    add(specs, format!("{prefix}.act.alpha"), [channels]);
    add(specs, format!("{prefix}.act.beta"), [channels]);
    add(specs, format!("{prefix}.upsample.filter"), [1, 1, 12]);
    add(
        specs,
        format!("{prefix}.downsample.lowpass.filter"),
        [1, 1, 12],
    );
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn safetensors_bytes(
        config: &AudioVaeConfig,
        layout: AudioTensorLayout,
        mutation: impl FnOnce(&mut serde_json::Map<String, Value>),
    ) -> Result<Vec<u8>> {
        let specs = audio_tensor_specs(config, layout)?;
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut offset = 0usize;
        for (name, spec) in specs {
            let bytes = spec.shape.iter().product::<usize>() * 4;
            object.insert(
                name,
                serde_json::json!({
                    "dtype": "F32",
                    "shape": spec.shape,
                    "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
        }
        mutation(&mut object);
        let mut header = serde_json::to_vec(&object)
            .map_err(|error| candle::Error::Msg(format!("test header encode failed: {error}")))?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + offset, 0);
        Ok(bytes)
    }

    fn validate_bytes(
        bytes: &[u8],
        config: &AudioVaeConfig,
        layout: AudioTensorLayout,
    ) -> Result<ValidatedAudioWeights> {
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        validate_header_and_file_len(
            &bytes[8..8 + header_len],
            bytes.len() as u64,
            config,
            layout,
        )
    }

    #[test]
    fn both_exact_layouts_validate_and_are_distinct() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        for layout in [
            AudioTensorLayout::OfficialWeightNorm,
            AudioTensorLayout::ComfyFolded,
        ] {
            let bytes = safetensors_bytes(&config, layout, |_| {})?;
            let summary = validate_bytes(&bytes, &config, layout)?;
            assert_eq!(summary.layout, layout);
            assert!(summary.tensor_count > 100);
            assert!(summary.tensor_bytes > 0);
            let other = if layout == AudioTensorLayout::OfficialWeightNorm {
                AudioTensorLayout::ComfyFolded
            } else {
                AudioTensorLayout::OfficialWeightNorm
            };
            assert!(validate_bytes(&bytes, &config, other).is_err());
        }
        Ok(())
    }

    #[test]
    fn production_tensor_payload_is_revision_locked() -> Result<()> {
        let specs = audio_tensor_specs(
            &AudioVaeConfig::default(),
            AudioTensorLayout::OfficialWeightNorm,
        )?;
        let tensor_bytes = specs
            .values()
            .map(|spec| spec.shape.iter().product::<usize>() * 4)
            .sum::<usize>();
        // The pinned 605,429,340-byte official file has 605,306,340 bytes of
        // F32 tensor payload and a 123,000-byte safetensors prefix/header.
        assert_eq!(tensor_bytes, 605_306_340);
        Ok(())
    }

    #[test]
    fn header_rejects_missing_logs_wrong_dtype_shape_and_offsets() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        let layout = AudioTensorLayout::ComfyFolded;
        let missing = safetensors_bytes(&config, layout, |header| {
            header.remove("logs_proj.weight");
        })?;
        assert!(validate_bytes(&missing, &config, layout).is_err());

        let wrong_dtype = safetensors_bytes(&config, layout, |header| {
            header["mean_proj.weight"]["dtype"] = Value::String("BF16".into());
        })?;
        assert!(validate_bytes(&wrong_dtype, &config, layout).is_err());

        let wrong_shape = safetensors_bytes(&config, layout, |header| {
            header["mean_proj.weight"]["shape"] = serde_json::json!([999, 1, 1]);
        })?;
        assert!(validate_bytes(&wrong_shape, &config, layout).is_err());

        let wrong_offsets = safetensors_bytes(&config, layout, |header| {
            header["mean_proj.weight"]["data_offsets"][1] = Value::from(1u64);
        })?;
        assert!(validate_bytes(&wrong_offsets, &config, layout).is_err());
        Ok(())
    }

    #[test]
    fn file_validator_checks_payload_length() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        let bytes = safetensors_bytes(&config, AudioTensorLayout::ComfyFolded, |_| {})?;
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mold-h3-audio-header-{}-{unique}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).map_err(io_error)?;
        let result = validate_audio_safetensors(&path, &config, AudioTensorLayout::ComfyFolded);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn validated_comfy_loader_requires_exact_normalization_buffers() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        let specs = audio_tensor_specs(&config, AudioTensorLayout::ComfyFolded)?;
        let mut tensors = std::collections::HashMap::new();
        for (name, spec) in specs {
            tensors.insert(
                name,
                candle::Tensor::zeros(spec.shape, DType::F32, &Device::Cpu)?,
            );
        }
        tensors.insert(
            "latents_mean".into(),
            candle::Tensor::from_vec(
                config.latents_mean.clone(),
                config.latent_channels,
                &Device::Cpu,
            )?,
        );
        tensors.insert(
            "latents_std".into(),
            candle::Tensor::from_vec(
                config.latents_std.clone(),
                config.latent_channels,
                &Device::Cpu,
            )?,
        );
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mold-h3-audio-load-{}-{unique}.safetensors",
            std::process::id()
        ));
        candle::safetensors::save(&tensors, &path)?;
        let loaded = load_validated_audio_vae(
            &path,
            config,
            AudioTensorLayout::ComfyFolded,
            DType::F32,
            &Device::Cpu,
        );
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_ok());
        Ok(())
    }
}
