use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::artifacts::ConditionerWeightLayout;
use super::config::{H3ConditionerConfig, H3_SELECTED_LANGUAGE_LAYERS};
#[cfg(feature = "h3-private-uat")]
use super::text::Qwen3VlStreamingTextEncoder;
use super::text::{Qwen3VlNvfp4Weights, Qwen3VlTextDimensions, Qwen3VlTextEncoder};
use super::vision::{Qwen3VlVisionDimensions, Qwen3VlVisionModel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionerCheckpoint {
    TokenEmbedding,
    VisionPatchEmbedding,
    VisionLayer { completed: usize, total: usize },
    LanguageLayer { completed: usize, total: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3DTypeProfile {
    pub parameter_dtype: DType,
    pub vision_input_dtype: DType,
    pub rotary_compute_dtype: DType,
    pub vision_attention_dtype: DType,
    pub language_attention_dtype: DType,
    pub attention_softmax_dtype: DType,
    pub output_dtype: DType,
}

#[derive(Clone, Debug)]
pub struct H3VisionInput {
    /// Flattened Qwen processor patches, `(patches, C*T*P*P)`.
    pub pixel_values: Tensor,
    /// One `(temporal, height, width)` row per image or video.
    pub grid_thw: Tensor,
}

#[derive(Clone, Debug)]
pub struct H3ConditionerInput {
    pub input_ids: Tensor,
    /// Qwen three-axis positions, `(3, batch, sequence)`.
    pub position_ids: Tensor,
    pub image: Option<H3VisionInput>,
    pub video: Option<H3VisionInput>,
}

/// The exact Qwen3-VL feature extractor consumed by H3. Construction requests
/// only decoder layers 0-49 and never asks the VarBuilder for a final norm or
/// LM head.
pub struct H3Layer50Conditioner {
    text: Qwen3VlTextEncoder,
    vision: Qwen3VlVisionModel,
    dtype_profile: H3DTypeProfile,
}

trait H3TextBackend {
    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor>;
    fn forward_embeds(
        &self,
        hidden: Tensor,
        position_ids: &Tensor,
        visual_indices: &[usize],
        deepstack: Option<&[Tensor]>,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor>;
}

impl H3TextBackend for Qwen3VlTextEncoder {
    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens(input_ids)
    }

    fn forward_embeds(
        &self,
        hidden: Tensor,
        position_ids: &Tensor,
        visual_indices: &[usize],
        deepstack: Option<&[Tensor]>,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        self.forward_embeds(hidden, position_ids, visual_indices, deepstack, checkpoint)
    }
}

#[cfg(feature = "h3-private-uat")]
impl H3TextBackend for Qwen3VlStreamingTextEncoder {
    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens(input_ids)
    }

    fn forward_embeds(
        &self,
        hidden: Tensor,
        position_ids: &Tensor,
        visual_indices: &[usize],
        deepstack: Option<&[Tensor]>,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        self.forward_embeds(hidden, position_ids, visual_indices, deepstack, checkpoint)
    }
}

impl H3Layer50Conditioner {
    pub fn new(config: &H3ConditionerConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_weight_layout(config, vb, ConditionerWeightLayout::Official)
    }

    pub(super) fn new_with_weight_layout(
        config: &H3ConditionerConfig,
        vb: VarBuilder,
        weight_layout: ConditionerWeightLayout,
    ) -> Result<Self> {
        config
            .validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let vision_dimensions = Qwen3VlVisionDimensions::from_h3(config);
        let text_dimensions = Qwen3VlTextDimensions::from_h3(config);
        let (vision_vb, text_vb) = match weight_layout {
            ConditionerWeightLayout::Official => (
                vb.pp("model").pp("visual"),
                vb.pp("model").pp("language_model"),
            ),
            ConditionerWeightLayout::ComfyLayer50 => (vb.pp("visual"), vb.pp("model")),
        };
        let vision = Qwen3VlVisionModel::new(&vision_dimensions, vision_vb)?;
        let text = Qwen3VlTextEncoder::new(&text_dimensions, text_vb)?;
        let parameter_dtype = vb.dtype();
        Ok(Self {
            text,
            vision,
            dtype_profile: H3DTypeProfile {
                parameter_dtype,
                vision_input_dtype: DType::F32,
                rotary_compute_dtype: DType::F32,
                vision_attention_dtype: parameter_dtype,
                language_attention_dtype: parameter_dtype,
                attention_softmax_dtype: DType::F32,
                output_dtype: parameter_dtype,
            },
        })
    }

    pub fn dtype_profile(&self) -> H3DTypeProfile {
        self.dtype_profile
    }

    pub fn resident_language_layers(&self) -> usize {
        self.text.layer_count()
    }

    pub fn encode(
        &self,
        input: &H3ConditionerInput,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        encode_conditioner(&self.text, &self.vision, input, checkpoint)
    }
}

#[cfg(feature = "h3-private-uat")]
pub struct H3Layer50StreamingConditioner {
    text: Qwen3VlStreamingTextEncoder,
    vision: Qwen3VlVisionModel,
    dtype_profile: H3DTypeProfile,
}

#[cfg(feature = "h3-private-uat")]
impl H3Layer50StreamingConditioner {
    pub(super) fn new(
        config: &H3ConditionerConfig,
        vb: VarBuilder,
        checkpoint_paths: Vec<std::path::PathBuf>,
    ) -> Result<Self> {
        config
            .validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let vision_dimensions = Qwen3VlVisionDimensions::from_h3(config);
        let text_dimensions = Qwen3VlTextDimensions::from_h3(config);
        let vision = Qwen3VlVisionModel::new(&vision_dimensions, vb.pp("model").pp("visual"))?;
        let text = Qwen3VlStreamingTextEncoder::new(
            &text_dimensions,
            vb.pp("model").pp("language_model"),
            checkpoint_paths,
        )?;
        let parameter_dtype = vb.dtype();
        Ok(Self {
            text,
            vision,
            dtype_profile: H3DTypeProfile {
                parameter_dtype,
                vision_input_dtype: DType::F32,
                rotary_compute_dtype: DType::F32,
                vision_attention_dtype: parameter_dtype,
                language_attention_dtype: parameter_dtype,
                attention_softmax_dtype: DType::F32,
                output_dtype: parameter_dtype,
            },
        })
    }

    pub fn dtype_profile(&self) -> H3DTypeProfile {
        self.dtype_profile
    }

    pub fn language_layer_count(&self) -> usize {
        self.text.layer_count()
    }

    pub const fn peak_resident_language_layers(&self) -> usize {
        1
    }

    pub fn encode(
        &self,
        input: &H3ConditionerInput,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        encode_conditioner(&self.text, &self.vision, input, checkpoint)
    }
}

/// Internal mixed-storage conditioner for the authenticated Comfy Qwen
/// NVFP4-AWQ object. It is not registered with any Mold engine or capability.
#[allow(dead_code)]
pub(super) struct H3QwenNvfp4Layer50Conditioner {
    text: Qwen3VlTextEncoder,
    vision: Qwen3VlVisionModel,
    dtype_profile: H3DTypeProfile,
}

#[allow(dead_code)]
impl H3QwenNvfp4Layer50Conditioner {
    pub(super) fn new(
        config: &H3ConditionerConfig,
        dense_vb: VarBuilder,
        text_weights: Qwen3VlNvfp4Weights,
    ) -> Result<Self> {
        config
            .validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let vision_dimensions = Qwen3VlVisionDimensions::from_h3(config);
        let text_dimensions = Qwen3VlTextDimensions::from_h3(config);
        let parameter_dtype = dense_vb.dtype();
        let vision = Qwen3VlVisionModel::new(&vision_dimensions, dense_vb.pp("visual"))?;
        let text =
            Qwen3VlTextEncoder::new_nvfp4(&text_dimensions, dense_vb.pp("model"), text_weights)?;
        Ok(Self {
            text,
            vision,
            dtype_profile: H3DTypeProfile {
                parameter_dtype,
                vision_input_dtype: DType::F32,
                rotary_compute_dtype: DType::F32,
                vision_attention_dtype: parameter_dtype,
                language_attention_dtype: parameter_dtype,
                attention_softmax_dtype: DType::F32,
                output_dtype: parameter_dtype,
            },
        })
    }

    pub fn dtype_profile(&self) -> H3DTypeProfile {
        self.dtype_profile
    }

    pub fn resident_language_layers(&self) -> usize {
        self.text.layer_count()
    }

    pub fn encode(
        &self,
        input: &H3ConditionerInput,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        encode_conditioner(&self.text, &self.vision, input, checkpoint)
    }

    pub(crate) fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text.embed_tokens(input_ids)
    }
}

fn encode_conditioner(
    text: &impl H3TextBackend,
    vision: &Qwen3VlVisionModel,
    input: &H3ConditionerInput,
    checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
) -> Result<Tensor> {
    let (input_batch, input_sequence) = input.input_ids.dims2()?;
    if input_batch != 1 {
        candle::bail!("H3 conditioner accepts one packed presentation, got batch {input_batch}");
    }
    if input_sequence == 0 {
        candle::bail!(
            "the raw H3 presentation tokenized to an empty sequence; no special token is inserted"
        );
    }
    checkpoint(ConditionerCheckpoint::TokenEmbedding)?;
    let mut embeddings = text.embed_tokens(&input.input_ids)?;
    let (batch, sequence, hidden) = embeddings.dims3()?;
    if input.position_ids.dims() != [3, batch, sequence] {
        candle::bail!(
            "H3 Qwen position ids have shape {:?}, expected ({}, {}, {})",
            input.position_ids.dims(),
            3,
            batch,
            sequence
        );
    }

    let mut image_deepstack = None;
    let mut video_deepstack = None;
    let mut image_indices = Vec::new();
    let mut video_indices = Vec::new();

    if let Some(image) = &input.image {
        let (features, deepstack) =
            vision.forward(&image.pixel_values, &image.grid_thw, checkpoint)?;
        image_indices = matching_token_indices(&input.input_ids, 151_655)?;
        embeddings = replace_rows(
            &embeddings,
            &image_indices,
            &features
                .to_device(embeddings.device())?
                .to_dtype(embeddings.dtype())?,
            hidden,
        )?;
        image_deepstack = Some(deepstack);
    }
    if let Some(video) = &input.video {
        let (features, deepstack) =
            vision.forward(&video.pixel_values, &video.grid_thw, checkpoint)?;
        video_indices = matching_token_indices(&input.input_ids, 151_656)?;
        embeddings = replace_rows(
            &embeddings,
            &video_indices,
            &features
                .to_device(embeddings.device())?
                .to_dtype(embeddings.dtype())?,
            hidden,
        )?;
        video_deepstack = Some(deepstack);
    }

    if input.image.is_none() && !matching_token_indices(&input.input_ids, 151_655)?.is_empty() {
        candle::bail!("H3 presentation contains image pads without image processor tensors");
    }
    if input.video.is_none() && !matching_token_indices(&input.input_ids, 151_656)?.is_empty() {
        candle::bail!("H3 presentation contains video pads without video processor tensors");
    }

    let (visual_indices, deepstack) = merge_deepstack(
        embeddings.device(),
        embeddings.dtype(),
        image_indices,
        image_deepstack,
        video_indices,
        video_deepstack,
    )?;
    text.forward_embeds(
        embeddings,
        &input.position_ids,
        &visual_indices,
        deepstack.as_deref(),
        checkpoint,
    )
}

fn matching_token_indices(input_ids: &Tensor, token_id: u32) -> Result<Vec<usize>> {
    let (batch, sequence) = input_ids.dims2()?;
    if batch != 1 {
        candle::bail!("H3 conditioner accepts one packed presentation, got batch {batch}");
    }
    Ok(input_ids
        .to_device(&candle::Device::Cpu)?
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .take(sequence)
        .enumerate()
        .filter_map(|(index, id)| (id == token_id).then_some(index))
        .collect())
}

fn replace_rows(
    embeddings: &Tensor,
    indices: &[usize],
    features: &Tensor,
    hidden: usize,
) -> Result<Tensor> {
    if features.dims() != [indices.len(), hidden] {
        candle::bail!(
            "H3 vision features have shape {:?}, expected ({}, {}) for the presentation pads",
            features.dims(),
            indices.len(),
            hidden
        );
    }
    let mut result = embeddings.clone();
    for (feature_index, &sequence_index) in indices.iter().enumerate() {
        result = result.slice_assign(
            &[0..1, sequence_index..sequence_index + 1, 0..hidden],
            &features.narrow(0, feature_index, 1)?.unsqueeze(0)?,
        )?;
    }
    Ok(result)
}

#[allow(clippy::type_complexity)]
fn merge_deepstack(
    device: &candle::Device,
    dtype: DType,
    image_indices: Vec<usize>,
    image: Option<Vec<Tensor>>,
    video_indices: Vec<usize>,
    video: Option<Vec<Tensor>>,
) -> Result<(Vec<usize>, Option<Vec<Tensor>>)> {
    let mut indexed_modalities = image_indices
        .into_iter()
        .map(|index| (index, true))
        .chain(video_indices.into_iter().map(|index| (index, false)))
        .collect::<Vec<_>>();
    indexed_modalities.sort_unstable_by_key(|(index, _)| *index);
    let visual_indices = indexed_modalities
        .iter()
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    let layer_count = match (&image, &video) {
        (Some(image), Some(video)) if image.len() != video.len() => {
            candle::bail!(
                "H3 image/video DeepStack layer counts differ: {} and {}",
                image.len(),
                video.len()
            );
        }
        (Some(image), _) => image.len(),
        (_, Some(video)) => video.len(),
        _ => return Ok((visual_indices, None)),
    };
    let mut image_offset;
    let mut video_offset;
    let mut merged = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        image_offset = 0;
        video_offset = 0;
        let mut rows = Vec::with_capacity(indexed_modalities.len());
        for (_, is_image) in &indexed_modalities {
            let row = if *is_image {
                let tensor = image.as_ref().expect("an image index has image features");
                let row = tensor[layer].narrow(0, image_offset, 1)?;
                image_offset += 1;
                row
            } else {
                let tensor = video.as_ref().expect("a video index has video features");
                let row = tensor[layer].narrow(0, video_offset, 1)?;
                video_offset += 1;
                row
            };
            rows.push(row.to_device(device)?.to_dtype(dtype)?);
        }
        let refs = rows.iter().collect::<Vec<_>>();
        merged.push(Tensor::cat(&refs, 0)?);
    }
    Ok((visual_indices, Some(merged)))
}

#[cfg(test)]
impl H3Layer50Conditioner {
    pub(super) fn new_tiny(
        text_dimensions: &Qwen3VlTextDimensions,
        vision_dimensions: &Qwen3VlVisionDimensions,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_tiny_with_weight_layout(
            text_dimensions,
            vision_dimensions,
            vb,
            ConditionerWeightLayout::Official,
        )
    }

    pub(super) fn new_tiny_with_weight_layout(
        text_dimensions: &Qwen3VlTextDimensions,
        vision_dimensions: &Qwen3VlVisionDimensions,
        vb: VarBuilder,
        weight_layout: ConditionerWeightLayout,
    ) -> Result<Self> {
        let (vision_vb, text_vb) = match weight_layout {
            ConditionerWeightLayout::Official => (
                vb.pp("model").pp("visual"),
                vb.pp("model").pp("language_model"),
            ),
            ConditionerWeightLayout::ComfyLayer50 => (vb.pp("visual"), vb.pp("model")),
        };
        let text = Qwen3VlTextEncoder::new(text_dimensions, text_vb)?;
        let vision = Qwen3VlVisionModel::new(vision_dimensions, vision_vb)?;
        Ok(Self {
            text,
            vision,
            dtype_profile: H3DTypeProfile {
                parameter_dtype: vb.dtype(),
                vision_input_dtype: DType::F32,
                rotary_compute_dtype: DType::F32,
                vision_attention_dtype: vb.dtype(),
                language_attention_dtype: vb.dtype(),
                attention_softmax_dtype: DType::F32,
                output_dtype: vb.dtype(),
            },
        })
    }
}

const _: () = assert!(H3_SELECTED_LANGUAGE_LAYERS == 50);

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use candle_nn::VarMap;

    #[test]
    fn tiny_combined_conditioner_constructs_only_requested_layers() {
        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
        let conditioner = H3Layer50Conditioner::new_tiny(
            &Qwen3VlTextDimensions::tiny(),
            &Qwen3VlVisionDimensions::tiny(),
            builder,
        )
        .unwrap();
        assert_eq!(conditioner.resident_language_layers(), 3);
        let keys = variables.data().lock().unwrap();
        assert!(!keys.contains_key("model.language_model.norm.weight"));
        assert!(!keys.contains_key("lm_head.weight"));
    }

    #[test]
    fn tiny_comfy_layout_requests_only_pruned_namespace_keys() {
        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
        H3Layer50Conditioner::new_tiny_with_weight_layout(
            &Qwen3VlTextDimensions::tiny(),
            &Qwen3VlVisionDimensions::tiny(),
            builder,
            ConditionerWeightLayout::ComfyLayer50,
        )
        .unwrap();
        let keys = variables.data().lock().unwrap();
        assert!(keys.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(keys.contains_key("visual.deepstack_merger_list.0.norm.weight"));
        assert!(!keys.keys().any(
            |key| key.starts_with("model.language_model.") || key.starts_with("model.visual.")
        ));
    }
}
