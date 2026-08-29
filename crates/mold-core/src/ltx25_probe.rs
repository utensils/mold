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

/// GGUF string entries that hold a whole JSON document. Only a NESTED lookup
/// under one of these decodes the string; every other string value —
/// `model_version = "2.5.0"`, `license`, `general.architecture` — is
/// returned verbatim, so a converter that stamps a bare `"2.5"` can never
/// turn a version into a JSON number and fail the 2.5 check.
const GGUF_JSON_DOCUMENT_KEYS: &[&str] = &["config", "gemma_source_checkpoint"];

fn gguf_metadata_json(value: &GgufMetadataValue) -> Value {
    match value {
        GgufMetadataValue::String(value) => Value::String(value.clone()),
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
    let root = header.metadata.get(*first)?;
    if rest.is_empty() {
        return Some(gguf_metadata_json(root));
    }
    let mut value = match root {
        GgufMetadataValue::String(document) if GGUF_JSON_DOCUMENT_KEYS.contains(first) => {
            serde_json::from_str::<Value>(document).ok()?
        }
        GgufMetadataValue::String(_) => return None,
        other => gguf_metadata_json(other),
    };
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
    // Same contract as the safetensors probe: `ff_bias` must be stated false,
    // while `audio_ff_bias` defaults to true upstream and the published
    // export omits it — only an explicit `false` is the 2.3 layout.
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

    // The block count is the converter's own statement about the graph, so a
    // future distill with a different depth is checked against itself rather
    // than against a hard-coded 47. Absent means fail closed.
    let num_layers = gguf_metadata_path(&header, &["config", "transformer", "num_layers"])
        .or_else(|| gguf_metadata_path(&header, &["config", "num_layers"]))
        .and_then(|value| value.as_u64())
        .filter(|layers| *layers > 0)
        .ok_or_else(|| invalid_data(path, "missing config.transformer.num_layers"))?;
    let last_block = num_layers - 1;

    // The 2.5 export carries no `caption_projection` (its text projection
    // lives in the Gemma checkpoint); the modulation tables are what every
    // real header has, and their input width is the branch width.
    let required = [
        "patchify_proj.weight".to_string(),
        "adaln_single.linear.weight".to_string(),
        "proj_out.weight".to_string(),
        "audio_patchify_proj.weight".to_string(),
        "audio_adaln_single.linear.weight".to_string(),
        "audio_proj_out.weight".to_string(),
        format!("transformer_blocks.{last_block}.attn1.to_q.weight"),
        format!("transformer_blocks.{last_block}.audio_attn1.to_q.weight"),
        format!("transformer_blocks.{last_block}.audio_to_video_attn.to_q.weight"),
        format!("transformer_blocks.{last_block}.video_to_audio_attn.to_q.weight"),
    ];
    for name in &required {
        require_gguf_tensor(path, &header, name)?;
    }

    let video = require_gguf_tensor(path, &header, "adaln_single.linear.weight")?;
    let audio = require_gguf_tensor(path, &header, "audio_adaln_single.linear.weight")?;
    if video.shape.last() != Some(&4096) {
        return Err(invalid_data(
            path,
            format!("expected video width 4096, got shape {:?}", video.shape),
        ));
    }
    if audio.shape.last() != Some(&2048) {
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

    fn golden(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/ltx25")
            .join(name)
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
        write_gguf_fixture_with_layers(
            path,
            architecture,
            model_version,
            gemma_version,
            ff_bias,
            Some(48),
            tensors,
        );
    }

    fn write_gguf_fixture_with_layers(
        path: &Path,
        architecture: &str,
        model_version: &str,
        gemma_version: &str,
        ff_bias: bool,
        num_layers: Option<u64>,
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
            ("config", {
                let mut transformer = serde_json::json!({
                    "ff_bias": ff_bias,
                    "audio_ff_bias": true,
                });
                if let Some(num_layers) = num_layers {
                    transformer["num_layers"] = serde_json::json!(num_layers);
                }
                serde_json::json!({ "transformer": transformer }).to_string()
            }),
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
            ("patchify_proj.weight", &[4096, 128], 0),
            ("adaln_single.linear.weight", &[36864, 4096], 0),
            ("proj_out.weight", &[128, 4096], 0),
            ("audio_patchify_proj.weight", &[2048, 16], 0),
            ("audio_adaln_single.linear.weight", &[18432, 2048], 0),
            ("audio_proj_out.weight", &[16, 2048], 0),
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
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("transformer.safetensors");
        let gemma = dir.path().join("gemma.safetensors");
        let mismatched_gemma = dir.path().join("gemma-mismatch.safetensors");
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
    }

    #[test]
    fn gguf_transformer_qualification_is_header_only_and_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("valid.gguf");
        let wrong_arch = dir.path().join("wrong-arch.gguf");
        let wrong_version = dir.path().join("wrong-version.gguf");
        let unsupported_dtype = dir.path().join("unsupported-dtype.gguf");
        let missing_audio = dir.path().join("missing-audio.gguf");
        let truncated = dir.path().join("truncated.gguf");
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
        // A converter that stamps "2.5" hands the probe something
        // `serde_json` would happily read as the number 2.5; the version is
        // a plain string and must stay one.
        let bare_version = dir.path().join("bare-version.gguf");
        write_gguf_fixture(
            &bare_version,
            "ltxv",
            "2.5",
            "gemma4-ltx-2.5",
            false,
            &tensors,
        );
        assert_eq!(
            probe_ltx25_gguf_transformer(&bare_version)
                .unwrap()
                .model_version,
            "2.5"
        );
        // The block count comes from the converter's config, never a
        // hard-coded 47: a two-layer graph qualifies against its own depth,
        // and a header that omits the count fails closed.
        let two_layers = dir.path().join("two-layers.gguf");
        let shallow = tensors
            .iter()
            .map(|(name, shape, ggml_type)| {
                (
                    name.replace("transformer_blocks.47.", "transformer_blocks.1."),
                    *shape,
                    *ggml_type,
                )
            })
            .collect::<Vec<_>>();
        let shallow_refs = shallow
            .iter()
            .map(|(name, shape, ggml_type)| (name.as_str(), *shape, *ggml_type))
            .collect::<Vec<_>>();
        write_gguf_fixture_with_layers(
            &two_layers,
            "ltxv",
            "2.5.0",
            "gemma4-ltx-2.5",
            false,
            Some(2),
            &shallow_refs,
        );
        assert!(probe_ltx25_gguf_transformer(&two_layers).is_ok());
        let no_depth = dir.path().join("no-depth.gguf");
        write_gguf_fixture_with_layers(
            &no_depth,
            "ltxv",
            "2.5.0",
            "gemma4-ltx-2.5",
            false,
            None,
            &tensors,
        );
        assert!(probe_ltx25_gguf_transformer(&no_depth)
            .unwrap_err()
            .to_string()
            .contains("num_layers"));
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
    }

    #[test]
    fn rejects_legacy_gemma_and_bias_contracts() {
        let dir = tempfile::tempdir().unwrap();
        let transformer = dir.path().join("transformer-bias.safetensors");
        let future_transformer = dir.path().join("transformer-future.safetensors");
        let gemma = dir.path().join("gemma3.safetensors");
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
    }

    #[test]
    fn distinguishes_conv_and_diffusion_video_vaes() {
        let dir = tempfile::tempdir().unwrap();
        let conv = dir.path().join("conv-vae.safetensors");
        let diffusion = dir.path().join("diffusion-vae.safetensors");
        let unknown = dir.path().join("unknown-vae.safetensors");
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
    }

    #[test]
    fn split_auxiliary_components_fail_closed_on_wrong_roles() {
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("audio.safetensors");
        let duration = dir.path().join("duration.safetensors");
        let spatial = dir.path().join("spatial-upscaler.safetensors");
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
    }

    /// The real Abiray Q4_K_M header (bytes 0..402810 of the published
    /// file): bare ComfyUI tensor names, `model_version` as a plain string,
    /// `config` / `gemma_source_checkpoint` as JSON documents, no
    /// `caption_projection` anywhere. The branch's first required-tensor
    /// list failed every tier on exactly that.
    #[test]
    fn real_q4_k_m_header_qualifies() {
        assert_eq!(
            probe_ltx25_gguf_transformer(&golden("distilled-q4-k-m.header.gguf")).unwrap(),
            Ltx25TransformerProbe {
                model_version: "2.5.0".into(),
                gemma_version: "gemma4-12b-ltx-v1".into(),
            }
        );
    }

    /// The official int8-convrot transformer header (metadata verbatim,
    /// blocks 0 and 47 retained) qualifies through the safetensors probe.
    #[test]
    fn real_int8_convrot_header_qualifies() {
        assert_eq!(
            probe_ltx25_transformer(&golden("distilled-int8-convrot.header.safetensors")).unwrap(),
            Ltx25TransformerProbe {
                model_version: "2.5.0".into(),
                gemma_version: "gemma4-12b-ltx-v1".into(),
            }
        );
    }

    /// The 194-byte stub that sat at `shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors`
    /// on hal9000 under a valid `.sha256-verified` sidecar: two `F32 [1]`
    /// tensors and no `__metadata__`. Every 2.5 pack reported this exact
    /// sentence as its readiness error until the file was re-pulled.
    #[test]
    fn audio_components_reject_the_194_byte_stub() {
        let path = golden("audio-vae-stub-194.safetensors");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 194);
        let error = validate_ltx25_audio_components(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains("expected an LTX-2.5 model_version"),
            "{error}"
        );
    }
}
