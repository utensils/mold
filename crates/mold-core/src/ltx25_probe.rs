//! Header-only qualification for LTX-2.5 split-pack components.

use std::path::Path;

use serde_json::Value;

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
}
