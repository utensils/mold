//! Internal runtime for the authenticated H3 Qwen NVFP4-AWQ artifact.
//!
//! This module deliberately has no engine, factory, catalog, download, or
//! capability registration. The caller must cross the external authorization
//! boundary before opening the private object. One retained regular-file
//! descriptor supplies schema inspection, complete SHA-256 authentication,
//! and every tensor payload read.

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use super::comfy_quant::{H3ComfyInt8TensorwiseEmbedding, H3ComfyNvfp4AwqLinear};
use super::config::{H3ConditionerConfig, H3_SELECTED_LANGUAGE_LAYERS};
use super::model::H3QwenNvfp4Layer50Conditioner;
use super::qwen_nvfp4::{
    open_h3_qwen_nvfp4_awq_artifact, released_config, H3QwenNvfp4AwqError, H3QwenNvfp4AwqExecution,
    H3QwenNvfp4AwqInspection, H3SafetensorsTensorHeader, OpenedH3QwenNvfp4AwqArtifact,
    H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES,
};
use super::text::{Qwen3VlNvfp4LayerWeights, Qwen3VlNvfp4Weights};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3QwenNvfp4LoadEvent {
    Authenticating {
        completed_bytes: u64,
        total_bytes: u64,
    },
    LoadingTensor {
        name: String,
        tensor_index: usize,
        tensor_count: usize,
        completed_bytes: u64,
        total_bytes: u64,
    },
}

pub trait H3QwenNvfp4LoadObserver {
    /// Return true to cancel at the next bounded read checkpoint.
    fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool;
}

#[derive(Default)]
pub struct NoopH3QwenNvfp4LoadObserver;

impl H3QwenNvfp4LoadObserver for NoopH3QwenNvfp4LoadObserver {
    fn should_cancel(&mut self, _event: &H3QwenNvfp4LoadEvent) -> bool {
        false
    }
}

#[derive(Debug, Error)]
pub enum H3QwenNvfp4RuntimeError {
    #[error(transparent)]
    Artifact(#[from] H3QwenNvfp4AwqError),
    #[error("invalid H3 Qwen NVFP4 runtime contract: {0}")]
    Contract(String),
    #[error("failed to construct H3 Qwen NVFP4 runtime: {0}")]
    Candle(#[from] candle::Error),
}

pub struct LoadedH3QwenNvfp4Conditioner {
    model: H3QwenNvfp4Layer50Conditioner,
    artifact: OpenedH3QwenNvfp4AwqArtifact,
}

impl LoadedH3QwenNvfp4Conditioner {
    fn model(&self) -> &H3QwenNvfp4Layer50Conditioner {
        &self.model
    }

    pub fn encode(
        &self,
        input: &super::model::H3ConditionerInput,
        checkpoint: &mut dyn FnMut(super::model::ConditionerCheckpoint) -> candle::Result<()>,
    ) -> candle::Result<Tensor> {
        self.model.encode(input, checkpoint)
    }

    pub fn dtype_profile(&self) -> super::model::H3DTypeProfile {
        self.model.dtype_profile()
    }

    pub fn resident_language_layers(&self) -> usize {
        self.model.resident_language_layers()
    }

    pub(crate) fn inspection(&self) -> &H3QwenNvfp4AwqInspection {
        self.artifact.inspection()
    }

    pub(crate) fn artifact_path(&self) -> &Path {
        self.artifact.path()
    }

    pub(crate) fn revalidate_artifact(&self) -> Result<(), H3QwenNvfp4RuntimeError> {
        self.artifact
            .revalidate("while retaining loaded H3 Qwen runtime")?;
        Ok(())
    }
}

struct TensorLoadProgress<'a> {
    observer: &'a mut dyn H3QwenNvfp4LoadObserver,
    tensor_index: usize,
    tensor_count: usize,
    completed_before: u64,
    total_bytes: u64,
}

impl TensorLoadProgress<'_> {
    fn read(
        &mut self,
        artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
        name: &str,
    ) -> Result<Vec<u8>, H3QwenNvfp4RuntimeError> {
        let event_name = name.to_string();
        let observer = &mut self.observer;
        let tensor_index = self.tensor_index;
        let tensor_count = self.tensor_count;
        let completed_before = self.completed_before;
        let total_bytes = self.total_bytes;
        let bytes = artifact.read_tensor_bytes(name, &mut |completed, _| {
            let event = H3QwenNvfp4LoadEvent::LoadingTensor {
                name: event_name.clone(),
                tensor_index,
                tensor_count,
                completed_bytes: completed_before.saturating_add(completed),
                total_bytes,
            };
            if observer.should_cancel(&event) {
                return Err(H3QwenNvfp4AwqError::Io(
                    "H3 Qwen NVFP4 load cancelled".into(),
                ));
            }
            Ok(())
        })?;
        self.completed_before = self
            .completed_before
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| {
                H3QwenNvfp4RuntimeError::Contract("loaded byte count overflows".into())
            })?;
        self.tensor_index += 1;
        Ok(bytes)
    }
}

pub fn load_h3_qwen_nvfp4_conditioner(
    path: &Path,
    config: &H3ConditionerConfig,
    device: &Device,
    observer: &mut dyn H3QwenNvfp4LoadObserver,
) -> Result<LoadedH3QwenNvfp4Conditioner, H3QwenNvfp4RuntimeError> {
    config
        .validate()
        .map_err(|error| H3QwenNvfp4RuntimeError::Contract(error.to_string()))?;
    if config != &released_config()? {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "runtime config differs from the frozen published layer-50 Qwen contract".into(),
        ));
    }

    let mut artifact = open_h3_qwen_nvfp4_awq_artifact(path)?;
    artifact.authenticate_full_sha256(&mut |completed_bytes, total_bytes| {
        let event = H3QwenNvfp4LoadEvent::Authenticating {
            completed_bytes,
            total_bytes,
        };
        if observer.should_cancel(&event) {
            return Err(H3QwenNvfp4AwqError::Io(
                "H3 Qwen NVFP4 authentication cancelled".into(),
            ));
        }
        Ok(())
    })?;

    let loadable_names = artifact
        .tensors()
        .iter()
        .filter(|(name, _)| !name.ends_with(".comfy_quant"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let loadable_bytes = loadable_names.iter().try_fold(0_u64, |total, name| {
        let header = &artifact.tensors()[name];
        total.checked_add(header.data_offsets[1] - header.data_offsets[0])
    });
    let loadable_bytes = loadable_bytes.ok_or_else(|| {
        H3QwenNvfp4RuntimeError::Contract("loadable tensor bytes overflow".into())
    })?;
    if loadable_bytes > H3_QWEN_NVFP4_AWQ_PAYLOAD_BYTES {
        return Err(H3QwenNvfp4RuntimeError::Contract(
            "loadable tensor bytes exceed authenticated payload".into(),
        ));
    }
    let mut progress = TensorLoadProgress {
        observer,
        tensor_index: 1,
        tensor_count: loadable_names.len(),
        completed_before: 0,
        total_bytes: loadable_bytes,
    };

    let embed_weight = load_tensor(
        &mut artifact,
        "model.embed_tokens.weight",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_scale = load_tensor(
        &mut artifact,
        "model.embed_tokens.weight_scale",
        &Device::Cpu,
        &mut progress,
    )?;
    let embed_tokens = H3ComfyInt8TensorwiseEmbedding::new(embed_weight, embed_scale)?;

    let mut layers = Vec::with_capacity(H3_SELECTED_LANGUAGE_LAYERS);
    for layer in 0..H3_SELECTED_LANGUAGE_LAYERS {
        layers.push(Qwen3VlNvfp4LayerWeights {
            q_proj: load_linear(
                &mut artifact,
                config,
                layer,
                "self_attn.q_proj",
                &mut progress,
            )?,
            k_proj: load_linear(
                &mut artifact,
                config,
                layer,
                "self_attn.k_proj",
                &mut progress,
            )?,
            v_proj: load_linear(
                &mut artifact,
                config,
                layer,
                "self_attn.v_proj",
                &mut progress,
            )?,
            o_proj: load_linear(
                &mut artifact,
                config,
                layer,
                "self_attn.o_proj",
                &mut progress,
            )?,
            gate_proj: load_linear(&mut artifact, config, layer, "mlp.gate_proj", &mut progress)?,
            up_proj: load_linear(&mut artifact, config, layer, "mlp.up_proj", &mut progress)?,
            down_proj: load_linear(&mut artifact, config, layer, "mlp.down_proj", &mut progress)?,
        });
    }

    let dense_names = artifact
        .tensors()
        .iter()
        .filter(|(name, header)| header.dtype == "BF16" && !name.ends_with(".pre_quant_scale"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut dense = HashMap::with_capacity(dense_names.len());
    for name in dense_names {
        let tensor = load_tensor(&mut artifact, &name, device, &mut progress)?;
        if dense.insert(name.clone(), tensor).is_some() {
            return Err(H3QwenNvfp4RuntimeError::Contract(format!(
                "duplicate dense tensor {name:?}"
            )));
        }
    }
    if progress.tensor_index - 1 != progress.tensor_count
        || progress.completed_before != progress.total_bytes
    {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "runtime loaded {} of {} tensors and {} of {} bytes",
            progress.tensor_index - 1,
            progress.tensor_count,
            progress.completed_before,
            progress.total_bytes
        )));
    }
    artifact.revalidate("after constructing H3 Qwen tensor storage")?;
    let builder = VarBuilder::from_tensors(dense, DType::BF16, device);
    let model = H3QwenNvfp4Layer50Conditioner::new(
        config,
        builder,
        Qwen3VlNvfp4Weights {
            embed_tokens,
            layers,
        },
    )?;
    if model.resident_language_layers() != H3_SELECTED_LANGUAGE_LAYERS {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "runtime constructed {} language layers, expected exactly {H3_SELECTED_LANGUAGE_LAYERS}",
            model.resident_language_layers()
        )));
    }
    artifact.revalidate("after constructing H3 Qwen layer-50 model")?;
    Ok(LoadedH3QwenNvfp4Conditioner { model, artifact })
}

fn load_linear(
    artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
    config: &H3ConditionerConfig,
    layer: usize,
    suffix: &str,
    progress: &mut TensorLoadProgress<'_>,
) -> Result<H3ComfyNvfp4AwqLinear, H3QwenNvfp4RuntimeError> {
    let canonical = format!("model.language_model.layers.{layer}.{suffix}.weight");
    let execution = artifact.policy().execution_for_weight(&canonical)?;
    let H3QwenNvfp4AwqExecution::Nvfp4FullPrecision {
        packed_weight_tensor,
        block_scale_tensor,
        tensor_scale_tensor,
        pre_quant_scale_tensor,
    } = execution
    else {
        return Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "projection {canonical:?} did not resolve to NVFP4 execution"
        )));
    };
    let (out_features, in_features) = linear_dimensions(config, suffix)?;
    let packed_weight = load_tensor(artifact, &packed_weight_tensor, &Device::Cpu, progress)?;
    let block_scales = load_tensor(artifact, &block_scale_tensor, &Device::Cpu, progress)?;
    let tensor_scale = load_tensor(artifact, &tensor_scale_tensor, &Device::Cpu, progress)?;
    let pre_quant_scale = pre_quant_scale_tensor
        .as_deref()
        .map(|name| load_tensor(artifact, name, &Device::Cpu, progress))
        .transpose()?;
    H3ComfyNvfp4AwqLinear::new_with_optional_awq(
        packed_weight,
        block_scales,
        tensor_scale,
        pre_quant_scale,
        out_features,
        in_features,
    )
    .map_err(Into::into)
}

fn linear_dimensions(
    config: &H3ConditionerConfig,
    suffix: &str,
) -> Result<(usize, usize), H3QwenNvfp4RuntimeError> {
    let text = &config.text_config;
    let query_width = text
        .num_attention_heads
        .checked_mul(text.head_dim)
        .ok_or_else(|| H3QwenNvfp4RuntimeError::Contract("query width overflows".into()))?;
    let key_value_width = text
        .num_key_value_heads
        .checked_mul(text.head_dim)
        .ok_or_else(|| H3QwenNvfp4RuntimeError::Contract("key/value width overflows".into()))?;
    match suffix {
        "self_attn.q_proj" => Ok((query_width, text.hidden_size)),
        "self_attn.k_proj" | "self_attn.v_proj" => Ok((key_value_width, text.hidden_size)),
        "self_attn.o_proj" => Ok((text.hidden_size, query_width)),
        "mlp.gate_proj" | "mlp.up_proj" => Ok((text.intermediate_size, text.hidden_size)),
        "mlp.down_proj" => Ok((text.hidden_size, text.intermediate_size)),
        other => Err(H3QwenNvfp4RuntimeError::Contract(format!(
            "unsupported Qwen projection {other:?}"
        ))),
    }
}

fn load_tensor(
    artifact: &mut OpenedH3QwenNvfp4AwqArtifact,
    name: &str,
    device: &Device,
    progress: &mut TensorLoadProgress<'_>,
) -> Result<Tensor, H3QwenNvfp4RuntimeError> {
    let header = artifact.tensors().get(name).cloned().ok_or_else(|| {
        H3QwenNvfp4RuntimeError::Contract(format!("missing authenticated tensor {name:?}"))
    })?;
    let bytes = progress.read(artifact, name)?;
    tensor_from_bytes(&bytes, &header, device).map_err(Into::into)
}

fn tensor_from_bytes(
    bytes: &[u8],
    header: &H3SafetensorsTensorHeader,
    device: &Device,
) -> candle::Result<Tensor> {
    let dtype = match header.dtype.as_str() {
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F32" => DType::F32,
        // Candle has no signed I8 storage. Preserve exact two's-complement
        // bytes and let the embedding primitive widen them explicitly.
        "I8" | "U8" => DType::U8,
        other => candle::bail!("unsupported authenticated Qwen dtype {other:?}"),
    };
    Tensor::from_raw_buffer(bytes, dtype, &header.shape, device)
}

const _: () = assert!(H3_SELECTED_LANGUAGE_LAYERS == 50);

#[cfg(test)]
mod tests {
    use super::super::qwen_nvfp4::tests::sparse_published_fixture;
    use super::*;

    struct CancelAuthentication {
        events: Vec<H3QwenNvfp4LoadEvent>,
    }

    impl H3QwenNvfp4LoadObserver for CancelAuthentication {
        fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool {
            self.events.push(event.clone());
            matches!(
                event,
                H3QwenNvfp4LoadEvent::Authenticating {
                    completed_bytes,
                    ..
                } if *completed_bytes > 0
            )
        }
    }

    struct CancelTensorRead {
        events: Vec<H3QwenNvfp4LoadEvent>,
    }

    impl H3QwenNvfp4LoadObserver for CancelTensorRead {
        fn should_cancel(&mut self, event: &H3QwenNvfp4LoadEvent) -> bool {
            self.events.push(event.clone());
            matches!(
                event,
                H3QwenNvfp4LoadEvent::LoadingTensor {
                    completed_bytes,
                    ..
                } if *completed_bytes > 0
            )
        }
    }

    #[test]
    fn authentication_is_cancellable_at_a_bounded_read_checkpoint() {
        let path = sparse_published_fixture();
        let mut observer = CancelAuthentication { events: Vec::new() };
        let error = load_h3_qwen_nvfp4_conditioner(
            &path,
            &released_config().unwrap(),
            &Device::Cpu,
            &mut observer,
        )
        .err()
        .expect("the observer must cancel authentication");
        assert!(error.to_string().contains("authentication cancelled"));
        assert_eq!(observer.events.len(), 1);
        assert!(matches!(
            observer.events[0],
            H3QwenNvfp4LoadEvent::Authenticating {
                completed_bytes: 1_048_576,
                total_bytes: super::super::qwen_nvfp4::H3_QWEN_NVFP4_AWQ_FILE_BYTES,
            }
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn tensor_staging_is_cancellable_at_a_bounded_read_checkpoint() {
        let path = sparse_published_fixture();
        let mut artifact = open_h3_qwen_nvfp4_awq_artifact(&path).unwrap();
        let mut observer = CancelTensorRead { events: Vec::new() };
        let mut progress = TensorLoadProgress {
            observer: &mut observer,
            tensor_index: 1,
            tensor_count: 1,
            completed_before: 0,
            total_bytes: 777_912_320,
        };
        let error = progress
            .read(&mut artifact, "model.embed_tokens.weight")
            .expect_err("the observer must cancel tensor staging");
        assert!(error.to_string().contains("load cancelled"));
        assert_eq!(observer.events.len(), 1);
        assert!(matches!(
            observer.events[0],
            H3QwenNvfp4LoadEvent::LoadingTensor {
                ref name,
                tensor_index: 1,
                tensor_count: 1,
                completed_bytes: 1_048_576,
                total_bytes: 777_912_320,
            } if name == "model.embed_tokens.weight"
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn runtime_rejects_config_drift_before_opening_the_artifact() {
        let missing = std::env::temp_dir().join("mold-h3-qwen-deliberately-missing.safetensors");
        let mut config = released_config().unwrap();
        config.text_config.max_position_embeddings -= 1;
        let error = load_h3_qwen_nvfp4_conditioner(
            &missing,
            &config,
            &Device::Cpu,
            &mut NoopH3QwenNvfp4LoadObserver,
        )
        .err()
        .expect("frozen config drift must fail");
        assert!(matches!(error, H3QwenNvfp4RuntimeError::Contract(_)));
        assert!(!missing.exists());
    }

    #[test]
    #[ignore = "requires the authorized 15.7 GB private H3 Qwen artifact"]
    fn authorized_real_artifact_constructs_exact_layer_50_runtime() {
        let path = std::env::var_os("MOLD_H3_QWEN_NVFP4_PATH")
            .map(std::path::PathBuf::from)
            .expect("set MOLD_H3_QWEN_NVFP4_PATH to the authorized artifact");
        let loaded = load_h3_qwen_nvfp4_conditioner(
            &path,
            &released_config().unwrap(),
            &Device::Cpu,
            &mut NoopH3QwenNvfp4LoadObserver,
        )
        .unwrap();
        assert_eq!(
            loaded.model().resident_language_layers(),
            H3_SELECTED_LANGUAGE_LAYERS
        );
        assert_eq!(
            loaded.inspection().expected_artifact_sha256,
            super::super::qwen_nvfp4::H3_QWEN_NVFP4_AWQ_SHA256
        );
        assert_eq!(loaded.artifact_path(), path);
        loaded.revalidate_artifact().unwrap();

        let ids = Tensor::new(&[[0_u32, 1, 151_935]], &Device::Cpu).unwrap();
        let embeddings = loaded
            .model()
            .embed_tokens(&ids)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(embeddings.len(), 3 * 5_120);
        assert!(embeddings.iter().all(|value| value.is_finite()));
    }
}
