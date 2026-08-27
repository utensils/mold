//! Header-only qualification for LTX-2.5 split-pack components.

use std::path::Path;

use serde_json::Value;

use crate::gguf_probe::{read_gguf_header, GgufHeader, GgufMetadataValue, GgufTensorInfo};
use crate::safetensors_probe::{read_safetensors_header, SafetensorsHeader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ltx25TransformerProbe {
    pub model_version: String,
    pub gemma_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ltx25GemmaProbe {
    pub model_type: String,
    pub gemma_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx25VideoVaeKind {
    Convolutional,
    Diffusion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx25UpscalerKind {
    Spatial,
    Temporal,
}

fn invalid_data(path: &Path, message: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("{}: {message}", path.display()),
    )
}

fn metadata_string<'a>(header: &'a SafetensorsHeader, key: &str) -> Option<&'a str> {
    header.metadata.get(key).and_then(Value::as_str)
}

fn config_value<'a>(header: &'a SafetensorsHeader, path: &[&str]) -> Option<&'a Value> {
    let mut value = header.metadata.get("config")?;
    for key in path {
        value = value.get(*key)?;
    }
    Some(value)
}

fn gemma_config_value<'a>(header: &'a SafetensorsHeader, path: &[&str]) -> Option<&'a Value> {
    let mut value = header
        .metadata
        .get("gemma_config")
        .or_else(|| header.metadata.get("config"))?;
    for key in path {
        value = value.get(*key)?;
    }
    Some(value)
}

fn transformer_config_value<'a>(header: &'a SafetensorsHeader, key: &str) -> Option<&'a Value> {
    config_value(header, &["transformer", key]).or_else(|| config_value(header, &[key]))
}

fn is_ltx_2_5(version: &str) -> bool {
    let normalized = version.replace('-', ".");
    let mut parts = normalized.split('.');
    let major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let minor = parts.next().and_then(|part| part.parse::<u32>().ok());
    matches!((major, minor), (Some(2), Some(5)))
}

fn gguf_metadata_json(value: &GgufMetadataValue) -> Value {
    match value {
        GgufMetadataValue::String(value) => {
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.clone()))
        }
        GgufMetadataValue::Bool(value) => Value::Bool(*value),
        GgufMetadataValue::U64(value) => Value::from(*value),
        GgufMetadataValue::I64(value) => Value::from(*value),
        GgufMetadataValue::F64(value) => Value::from(*value),
        GgufMetadataValue::Array(values) => {
            Value::Array(values.iter().map(gguf_metadata_json).collect())
        }
    }
}

fn gguf_metadata_path(header: &GgufHeader, path: &[&str]) -> Option<Value> {
    let dotted = path.join(".");
    if let Some(value) = header.metadata.get(&dotted) {
        return Some(gguf_metadata_json(value));
    }
    let (first, rest) = path.split_first()?;
    let mut value = gguf_metadata_json(header.metadata.get(*first)?);
    for key in rest {
        value = value.get(*key)?.clone();
    }
    Some(value)
}

fn gguf_tensor<'a>(header: &'a GgufHeader, name: &str) -> Option<&'a GgufTensorInfo> {
    header
        .tensors
        .get(name)
        .or_else(|| header.tensors.get(&format!("model.diffusion_model.{name}")))
}

fn require_gguf_tensor<'a>(
    path: &Path,
    header: &'a GgufHeader,
    name: &str,
) -> std::io::Result<&'a GgufTensorInfo> {
    gguf_tensor(header, name)
        .ok_or_else(|| invalid_data(path, format!("missing required LTX-2.5 GGUF tensor {name}")))
}

/// Qualify an LTX-2.5 joint audio/video GGUF without reading tensor payloads.
pub fn probe_ltx25_gguf_transformer(path: &Path) -> std::io::Result<Ltx25TransformerProbe> {
    let header = read_gguf_header(path).map_err(|error| invalid_data(path, error))?;
    if header
        .metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_str)
        != Some("ltxv")
    {
        return Err(invalid_data(path, "expected general.architecture=ltxv"));
    }

    let model_version = gguf_metadata_path(&header, &["model_version"])
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|version| is_ltx_2_5(version))
        .ok_or_else(|| invalid_data(path, "expected an LTX-2.5 model_version"))?;
    let gemma_version = gguf_metadata_path(&header, &["gemma_source_checkpoint", "gemma_version"])
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|version| !version.is_empty())
        .ok_or_else(|| invalid_data(path, "missing gemma_source_checkpoint.gemma_version"))?;

    let ff_bias = gguf_metadata_path(&header, &["config", "transformer", "ff_bias"])
        .or_else(|| gguf_metadata_path(&header, &["config", "ff_bias"]))
        .and_then(|value| value.as_bool());
    let audio_ff_bias = gguf_metadata_path(&header, &["config", "transformer", "audio_ff_bias"])
        .or_else(|| gguf_metadata_path(&header, &["config", "audio_ff_bias"]))
        .and_then(|value| value.as_bool());
    if ff_bias != Some(false) || audio_ff_bias == Some(false) {
        return Err(invalid_data(
            path,
            "LTX-2.5 GGUF must set ff_bias=false and keep audio_ff_bias=true",
        ));
    }

    const SUPPORTED_TYPES: &[u32] = &[0, 1, 8, 11, 12, 13, 14, 30];
    if let Some((name, tensor)) = header
        .tensors
        .iter()
        .find(|(_, tensor)| !SUPPORTED_TYPES.contains(&tensor.ggml_type))
    {
        return Err(invalid_data(
            path,
            format!(
                "unsupported GGML dtype {} for tensor {name}",
                tensor.ggml_type
            ),
        ));
    }

    let required = [
        "patchify_proj.weight",
        "caption_projection.linear_2.weight",
        "proj_out.weight",
        "audio_patchify_proj.weight",
        "audio_caption_projection.linear_2.weight",
        "audio_proj_out.weight",
        "transformer_blocks.47.attn1.to_q.weight",
        "transformer_blocks.47.audio_attn1.to_q.weight",
        "transformer_blocks.47.audio_to_video_attn.to_q.weight",
        "transformer_blocks.47.video_to_audio_attn.to_q.weight",
    ];
    for name in required {
        require_gguf_tensor(path, &header, name)?;
    }

    let video = require_gguf_tensor(path, &header, "caption_projection.linear_2.weight")?;
    let audio = require_gguf_tensor(path, &header, "audio_caption_projection.linear_2.weight")?;
    if video.shape.len() != 2 || !video.shape.contains(&4096) {
        return Err(invalid_data(
            path,
            format!("expected video width 4096, got shape {:?}", video.shape),
        ));
    }
    if audio.shape.len() != 2 || !audio.shape.contains(&2048) {
        return Err(invalid_data(
            path,
            format!("expected audio width 2048, got shape {:?}", audio.shape),
        ));
    }

    Ok(Ltx25TransformerProbe {
        model_version,
        gemma_version,
    })
}

/// Qualify a split LTX-2.5 transformer from metadata only.
///
/// The generation stamp and Gemma source version are the upstream authority;
/// the bias flags are the architecture seam that prevents a 2.3 loader from
/// accepting 2.5 and constructing nonexistent FFN bias tensors.
pub fn probe_ltx25_transformer(path: &Path) -> std::io::Result<Ltx25TransformerProbe> {
    let header = read_safetensors_header(path)?;
    let model_version = metadata_string(&header, "model_version")
        .filter(|version| is_ltx_2_5(version))
        .ok_or_else(|| invalid_data(path, "expected an LTX-2.5 model_version"))?;
    let gemma_version = header
        .metadata
        .get("gemma_source_checkpoint")
        .and_then(|value| value.get("gemma_version"))
        .and_then(Value::as_str)
        .filter(|version| !version.is_empty())
        .ok_or_else(|| invalid_data(path, "missing gemma_source_checkpoint.gemma_version"))?;
    if transformer_config_value(&header, "ff_bias").and_then(Value::as_bool) != Some(false) {
        return Err(invalid_data(
            path,
            "LTX-2.5 transformer config must set ff_bias=false",
        ));
    }
    if transformer_config_value(&header, "audio_ff_bias").and_then(Value::as_bool) == Some(false) {
        return Err(invalid_data(
            path,
            "LTX-2.5 transformer config must keep audio_ff_bias=true",
        ));
    }
    Ok(Ltx25TransformerProbe {
        model_version: model_version.to_string(),
        gemma_version: gemma_version.to_string(),
    })
}

/// Qualify the fine-tuned Gemma 4 Unified encoder used for LTX-2.5.
pub fn probe_ltx25_gemma(path: &Path) -> std::io::Result<Ltx25GemmaProbe> {
    let header = read_safetensors_header(path)?;
    let has_12b_marker = header
        .tensor_names
        .iter()
        .any(|name| name.ends_with("model.layers.47.self_attn.q_norm.weight"));
    let has_ltx_projection = header
        .tensor_names
        .iter()
        .any(|name| name.contains("text_embedding_projection.video_aggregate_embed.weight"));
    let has_legacy_v_projection = header
        .tensor_names
        .iter()
        .any(|name| name.ends_with("model.layers.5.self_attn.v_proj.weight"));
    if !has_12b_marker || !has_ltx_projection || has_legacy_v_projection {
        return Err(invalid_data(
            path,
            "expected Gemma 4 12B Unified tensors and LTX-2.5 projection",
        ));
    }
    let model_type = gemma_config_value(&header, &["model_type"])
        .and_then(Value::as_str)
        .or_else(|| metadata_string(&header, "model_type"))
        .filter(|model_type| *model_type == "gemma4_unified")
        .ok_or_else(|| invalid_data(path, "expected config.model_type=gemma4_unified"))?;
    let gemma_version = gemma_config_value(&header, &["gemma_version"])
        .and_then(Value::as_str)
        .or_else(|| metadata_string(&header, "gemma_version"))
        .filter(|version| !version.is_empty())
        .ok_or_else(|| invalid_data(path, "missing Gemma config gemma_version"))?;
    Ok(Ltx25GemmaProbe {
        model_type: model_type.to_string(),
        gemma_version: gemma_version.to_string(),
    })
}

/// Fail closed when a 2.5 transformer and Gemma checkpoint were produced for
/// different Gemma generations.
pub fn validate_ltx25_transformer_gemma(
    transformer: &Path,
    gemma_path: &Path,
) -> std::io::Result<()> {
    let transformer = probe_ltx25_transformer(transformer)?;
    let gemma = probe_ltx25_gemma(gemma_path)?;
    if transformer.gemma_version != gemma.gemma_version {
        return Err(invalid_data(
            gemma_path,
            format!(
                "Gemma version mismatch: transformer expects {}, encoder declares {}",
                transformer.gemma_version, gemma.gemma_version
            ),
        ));
    }
    Ok(())
}

/// Qualify either the official safetensors transformer or a third-party GGUF
/// transformer against the same official Gemma 4 checkpoint.
pub fn validate_ltx25_transformer_gemma_any(
    transformer_path: &Path,
    gemma_path: &Path,
) -> std::io::Result<()> {
    let transformer = if transformer_path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        probe_ltx25_gguf_transformer(transformer_path)?
    } else {
        probe_ltx25_transformer(transformer_path)?
    };
    let gemma = probe_ltx25_gemma(gemma_path)?;
    if transformer.gemma_version != gemma.gemma_version {
        return Err(invalid_data(
            gemma_path,
            format!(
                "Gemma version mismatch: transformer expects {}, encoder declares {}",
                transformer.gemma_version, gemma.gemma_version
            ),
        ));
    }
    Ok(())
}

/// Distinguish the two LTX-2.5 video decoders that share one latent space.
pub fn probe_ltx25_video_vae(path: &Path) -> std::io::Result<Ltx25VideoVaeKind> {
    let header = read_safetensors_header(path)?;
    let class_name = config_value(&header, &["vae", "_class_name"]).and_then(Value::as_str);
    let has_diffusion_marker = header
        .tensor_names
        .iter()
        .any(|name| name.ends_with("decoder.conv_in_x_t.weight"));
    if has_diffusion_marker || class_name == Some("CausalDiffusionVAE") {
        return Ok(Ltx25VideoVaeKind::Diffusion);
    }
    let has_encoder = header
        .tensor_names
        .iter()
        .any(|name| name.starts_with("encoder.") || name.starts_with("vae.encoder."));
    let has_decoder = header
        .tensor_names
        .iter()
        .any(|name| name.starts_with("decoder.") || name.starts_with("vae.decoder."));
    if class_name == Some("CausalVideoAutoencoder") && has_encoder && has_decoder {
        return Ok(Ltx25VideoVaeKind::Convolutional);
    }
    Err(invalid_data(path, "unrecognized LTX-2.5 video VAE layout"))
}

/// Qualify the split checkpoint that owns both audio decode namespaces.
pub fn validate_ltx25_audio_components(path: &Path) -> std::io::Result<()> {
    let header = read_safetensors_header(path)?;
    let version = metadata_string(&header, "model_version")
        .filter(|version| is_ltx_2_5(version))
        .ok_or_else(|| invalid_data(path, "expected an LTX-2.5 model_version"))?;
    let has_audio_decoder = header
        .tensor_names
        .iter()
        .any(|name| name == "audio_vae.decoder.conv_in.conv.weight");
    let has_vocoder = header
        .tensor_names
        .iter()
        .any(|name| name == "vocoder.vocoder.conv_pre.weight");
    let has_bwe = header
        .tensor_names
        .iter()
        .any(|name| name == "vocoder.bwe_generator.conv_pre.weight");
    if !has_audio_decoder || !has_vocoder || !has_bwe {
        return Err(invalid_data(
            path,
            format!(
                "LTX-{version} audio checkpoint must contain audio_vae, vocoder, and BWE tensors"
            ),
        ));
    }
    Ok(())
}

/// Qualify the official 15-tensor automatic-duration head.
pub fn validate_ltx25_duration_head(path: &Path) -> std::io::Result<()> {
    let header = read_safetensors_header(path)?;
    if metadata_string(&header, "model_version")
        .filter(|version| is_ltx_2_5(version))
        .is_none()
    {
        return Err(invalid_data(path, "expected an LTX-2.5 model_version"));
    }
    let required = [
        "duration_head.video_input_proj.weight",
        "duration_head.audio_input_proj.weight",
        "duration_head.attention_pooler.query_tokens",
        "duration_head.attention_pooler.cross_attn.in_proj_weight",
        "duration_head.mlp_out.weight",
    ];
    if header.tensor_names.len() != 15
        || required
            .iter()
            .any(|required| !header.tensor_names.iter().any(|name| name == required))
    {
        return Err(invalid_data(
            path,
            "expected the official 15-tensor LTX-2.5 duration head",
        ));
    }
    Ok(())
}

/// Qualify one of the two 2.5 latent upscalers by its embedded config.
pub fn validate_ltx25_upscaler(path: &Path, kind: Ltx25UpscalerKind) -> std::io::Result<()> {
    let header = read_safetensors_header(path)?;
    let config = header
        .metadata
        .get("config")
        .ok_or_else(|| invalid_data(path, "missing latent upscaler config"))?;
    let expected = match kind {
        Ltx25UpscalerKind::Spatial => (true, false),
        Ltx25UpscalerKind::Temporal => (false, true),
    };
    let actual = (
        config.get("spatial_upsample").and_then(Value::as_bool),
        config.get("temporal_upsample").and_then(Value::as_bool),
    );
    let has_weights = header
        .tensor_names
        .iter()
        .any(|name| name == "initial_conv.conv.weight" || name == "initial_conv.weight")
        && header
            .tensor_names
            .iter()
            .any(|name| name == "final_conv.conv.weight" || name == "final_conv.weight");
    if actual != (Some(expected.0), Some(expected.1)) || !has_weights {
        return Err(invalid_data(
            path,
            format!("expected the LTX-2.5 {kind:?} latent upscaler layout"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_safetensors(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-ltx25-probe-{name}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ))
    }

    fn temp_gguf(name: &str) -> PathBuf {
        temp_safetensors(name).with_extension("gguf")
    }

    fn write_gguf_string(file: &mut File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn write_gguf_fixture(
        path: &Path,
        architecture: &str,
        model_version: &str,
        gemma_version: &str,
        ff_bias: bool,
        tensors: &[(&str, &[u64], u32)],
    ) {
        let mut file = File::create(path).expect("create GGUF fixture");
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(tensors.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        for (key, value) in [
            ("general.architecture", architecture.to_string()),
            ("model_version", model_version.to_string()),
            (
                "gemma_source_checkpoint",
                serde_json::json!({"gemma_version": gemma_version}).to_string(),
            ),
            (
                "config",
                serde_json::json!({
                    "transformer": {"ff_bias": ff_bias, "audio_ff_bias": true}
                })
                .to_string(),
            ),
        ] {
            write_gguf_string(&mut file, key);
            file.write_all(&8u32.to_le_bytes()).unwrap();
            write_gguf_string(&mut file, &value);
        }
        for (name, shape, ggml_type) in tensors {
            write_gguf_string(&mut file, name);
            file.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
            for dim in shape.iter().rev() {
                file.write_all(&dim.to_le_bytes()).unwrap();
            }
            file.write_all(&ggml_type.to_le_bytes()).unwrap();
            file.write_all(&0u64.to_le_bytes()).unwrap();
        }
    }

    fn ltx25_gguf_tensors() -> Vec<(&'static str, &'static [u64], u32)> {
        vec![
            ("patchify_proj.weight", &[4096, 128], 12),
            ("caption_projection.linear_2.weight", &[4096, 4096], 12),
            ("proj_out.weight", &[128, 4096], 12),
            ("audio_patchify_proj.weight", &[2048, 16], 12),
            (
                "audio_caption_projection.linear_2.weight",
                &[2048, 2048],
                12,
            ),
            ("audio_proj_out.weight", &[16, 2048], 12),
            ("transformer_blocks.47.attn1.to_q.weight", &[4096, 4096], 11),
            (
                "transformer_blocks.47.audio_attn1.to_q.weight",
                &[2048, 2048],
                11,
            ),
            (
                "transformer_blocks.47.audio_to_video_attn.to_q.weight",
                &[4096, 4096],
                13,
            ),
            (
                "transformer_blocks.47.video_to_audio_attn.to_q.weight",
                &[2048, 2048],
                13,
            ),
        ]
    }

    fn write_fixture(path: &Path, keys: &[&str], metadata: serde_json::Map<String, Value>) {
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let encoded_metadata = metadata
            .into_iter()
            .map(|(key, value)| {
                let encoded = match value {
                    Value::String(raw) => raw,
                    other => serde_json::to_string(&other).unwrap(),
                };
                (key, Value::String(encoded))
            })
            .collect();
        header.insert("__metadata__".to_string(), Value::Object(encoded_metadata));
        let header_json = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = File::create(path).expect("create fixture");
        file.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_json).unwrap();
        file.write_all(&[0u8; 4]).unwrap();
    }

    fn transformer_metadata(ff_bias: bool) -> serde_json::Map<String, Value> {
        serde_json::Map::from_iter([
            ("model_version".into(), Value::String("2.5.0".into())),
            (
                "gemma_source_checkpoint".into(),
                serde_json::json!({"gemma_version": "gemma4-ltx-2.5"}),
            ),
            (
                "config".into(),
                serde_json::json!({
                    "transformer": {"ff_bias": ff_bias, "audio_ff_bias": true}
                }),
            ),
        ])
    }

    fn gemma_metadata(model_type: &str, gemma_version: &str) -> serde_json::Map<String, Value> {
        serde_json::Map::from_iter([(
            "gemma_config".into(),
            serde_json::json!({
                "model_type": model_type,
                "gemma_version": gemma_version
            }),
        )])
    }

    #[test]
    fn transformer_and_gemma_metadata_must_match() {
        let transformer = temp_safetensors("transformer");
        let gemma = temp_safetensors("gemma");
        let mismatched_gemma = temp_safetensors("gemma-mismatch");
        write_fixture(
            &transformer,
            &["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"],
            transformer_metadata(false),
        );
        write_fixture(
            &gemma,
            &[
                "model.layers.47.self_attn.q_norm.weight",
                "text_embedding_projection.video_aggregate_embed.weight",
                "tokenizer_json",
            ],
            gemma_metadata("gemma4_unified", "gemma4-ltx-2.5"),
        );
        write_fixture(
            &mismatched_gemma,
            &[
                "model.layers.47.self_attn.q_norm.weight",
                "text_embedding_projection.video_aggregate_embed.weight",
            ],
            gemma_metadata("gemma4_unified", "gemma4-other"),
        );

        assert_eq!(
            probe_ltx25_transformer(&transformer).unwrap().model_version,
            "2.5.0"
        );
        assert_eq!(
            probe_ltx25_gemma(&gemma).unwrap().model_type,
            "gemma4_unified"
        );
        validate_ltx25_transformer_gemma(&transformer, &gemma).unwrap();
        assert_eq!(
            validate_ltx25_transformer_gemma(&transformer, &mismatched_gemma)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );

        let _ = std::fs::remove_file(transformer);
        let _ = std::fs::remove_file(gemma);
        let _ = std::fs::remove_file(mismatched_gemma);
    }

    #[test]
    fn gguf_transformer_qualification_is_header_only_and_fail_closed() {
        let valid = temp_gguf("valid");
        let wrong_arch = temp_gguf("wrong-arch");
        let wrong_version = temp_gguf("wrong-version");
        let unsupported_dtype = temp_gguf("unsupported-dtype");
        let missing_audio = temp_gguf("missing-audio");
        let truncated = temp_gguf("truncated");
        let tensors = ltx25_gguf_tensors();
        write_gguf_fixture(&valid, "ltxv", "2.5.0", "gemma4-ltx-2.5", false, &tensors);
        write_gguf_fixture(
            &wrong_arch,
            "flux",
            "2.5.0",
            "gemma4-ltx-2.5",
            false,
            &tensors,
        );
        write_gguf_fixture(
            &wrong_version,
            "ltxv",
            "2.6.0",
            "gemma4-ltx-2.5",
            false,
            &tensors,
        );
        let mut bad_dtype = tensors.clone();
        bad_dtype[0].2 = 99;
        write_gguf_fixture(
            &unsupported_dtype,
            "ltxv",
            "2.5.0",
            "gemma4-ltx-2.5",
            false,
            &bad_dtype,
        );
        let no_audio = tensors
            .iter()
            .copied()
            .filter(|(name, _, _)| !name.contains("audio"))
            .collect::<Vec<_>>();
        write_gguf_fixture(
            &missing_audio,
            "ltxv",
            "2.5.0",
            "gemma4-ltx-2.5",
            false,
            &no_audio,
        );
        std::fs::write(&truncated, b"GGUF\x03\0").unwrap();

        assert_eq!(
            probe_ltx25_gguf_transformer(&valid).unwrap(),
            Ltx25TransformerProbe {
                model_version: "2.5.0".into(),
                gemma_version: "gemma4-ltx-2.5".into(),
            }
        );
        for invalid in [
            &wrong_arch,
            &wrong_version,
            &unsupported_dtype,
            &missing_audio,
            &truncated,
        ] {
            assert_eq!(
                probe_ltx25_gguf_transformer(invalid).unwrap_err().kind(),
                std::io::ErrorKind::InvalidData,
                "{} should fail closed",
                invalid.display()
            );
        }

        for path in [
            valid,
            wrong_arch,
            wrong_version,
            unsupported_dtype,
            missing_audio,
            truncated,
        ] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn rejects_legacy_gemma_and_bias_contracts() {
        let transformer = temp_safetensors("transformer-bias");
        let future_transformer = temp_safetensors("transformer-future");
        let gemma = temp_safetensors("gemma3");
        write_fixture(
            &transformer,
            &["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"],
            transformer_metadata(true),
        );
        let mut future_metadata = transformer_metadata(false);
        future_metadata.insert("model_version".into(), Value::String("2.6.0".into()));
        write_fixture(
            &future_transformer,
            &["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"],
            future_metadata,
        );
        write_fixture(
            &gemma,
            &["model.layers.47.self_attn.q_norm.weight"],
            gemma_metadata("gemma3", "gemma3"),
        );

        assert_eq!(
            probe_ltx25_transformer(&transformer).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            probe_ltx25_gemma(&gemma).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            probe_ltx25_transformer(&future_transformer)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );

        let _ = std::fs::remove_file(transformer);
        let _ = std::fs::remove_file(future_transformer);
        let _ = std::fs::remove_file(gemma);
    }

    #[test]
    fn distinguishes_conv_and_diffusion_video_vaes() {
        let conv = temp_safetensors("conv-vae");
        let diffusion = temp_safetensors("diffusion-vae");
        let unknown = temp_safetensors("unknown-vae");
        write_fixture(
            &conv,
            &["encoder.conv_in.weight", "decoder.conv_in.weight"],
            serde_json::Map::from_iter([(
                "config".into(),
                serde_json::json!({"vae": {"_class_name": "CausalVideoAutoencoder"}}),
            )]),
        );
        write_fixture(
            &unknown,
            &["encoder.conv_in.weight", "decoder.conv_in.weight"],
            serde_json::Map::from_iter([(
                "config".into(),
                serde_json::json!({"vae": {"_class_name": "FutureVideoVAE"}}),
            )]),
        );
        write_fixture(
            &diffusion,
            &["encoder.conv_in.weight", "decoder.conv_in_x_t.weight"],
            serde_json::Map::from_iter([(
                "config".into(),
                serde_json::json!({"vae": {"_class_name": "CausalDiffusionVAE"}}),
            )]),
        );

        assert_eq!(
            probe_ltx25_video_vae(&conv).unwrap(),
            Ltx25VideoVaeKind::Convolutional
        );
        assert_eq!(
            probe_ltx25_video_vae(&diffusion).unwrap(),
            Ltx25VideoVaeKind::Diffusion
        );
        assert_eq!(
            probe_ltx25_video_vae(&unknown).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let _ = std::fs::remove_file(conv);
        let _ = std::fs::remove_file(diffusion);
        let _ = std::fs::remove_file(unknown);
    }

    #[test]
    fn split_auxiliary_components_fail_closed_on_wrong_roles() {
        let audio = temp_safetensors("audio");
        let duration = temp_safetensors("duration");
        let spatial = temp_safetensors("spatial-upscaler");
        write_fixture(
            &audio,
            &[
                "audio_vae.decoder.conv_in.conv.weight",
                "vocoder.vocoder.conv_pre.weight",
                "vocoder.bwe_generator.conv_pre.weight",
            ],
            serde_json::Map::from_iter([("model_version".into(), Value::String("2.5.0".into()))]),
        );
        let duration_keys = [
            "duration_head.video_input_proj.weight",
            "duration_head.audio_input_proj.weight",
            "duration_head.attention_pooler.query_tokens",
            "duration_head.attention_pooler.cross_attn.in_proj_weight",
            "duration_head.mlp_out.weight",
            "duration_head.0",
            "duration_head.1",
            "duration_head.2",
            "duration_head.3",
            "duration_head.4",
            "duration_head.5",
            "duration_head.6",
            "duration_head.7",
            "duration_head.8",
            "duration_head.9",
        ];
        write_fixture(
            &duration,
            &duration_keys,
            serde_json::Map::from_iter([("model_version".into(), Value::String("2.5.0".into()))]),
        );
        write_fixture(
            &spatial,
            &["initial_conv.weight", "final_conv.weight"],
            serde_json::Map::from_iter([(
                "config".into(),
                serde_json::json!({
                    "spatial_upsample": true,
                    "temporal_upsample": false
                }),
            )]),
        );

        validate_ltx25_audio_components(&audio).unwrap();
        validate_ltx25_duration_head(&duration).unwrap();
        validate_ltx25_upscaler(&spatial, Ltx25UpscalerKind::Spatial).unwrap();
        assert!(validate_ltx25_upscaler(&spatial, Ltx25UpscalerKind::Temporal).is_err());

        let _ = std::fs::remove_file(audio);
        let _ = std::fs::remove_file(duration);
        let _ = std::fs::remove_file(spatial);
    }
}
