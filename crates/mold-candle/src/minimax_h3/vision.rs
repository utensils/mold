// Tensor layout follows Qwen3-VL's released implementation. The learned
// position interpolation and post-shuffle DeepStack mergers are not shared by
// Mold's older Qwen2.5-VL paths and therefore live here.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm,
    LayerNormConfig, Linear, VarBuilder,
};

use super::config::H3ConditionerConfig;
use super::model::ConditionerCheckpoint;

#[derive(Clone, Debug)]
pub(super) struct Qwen3VlVisionDimensions {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    output_hidden_size: usize,
    num_position_embeddings: usize,
    deepstack_visual_indexes: Vec<usize>,
    activation: Activation,
}

impl Qwen3VlVisionDimensions {
    pub(super) fn from_h3(config: &H3ConditionerConfig) -> Self {
        let vision = &config.vision_config;
        Self {
            depth: vision.depth,
            hidden_size: vision.hidden_size,
            intermediate_size: vision.intermediate_size,
            num_heads: vision.num_heads,
            in_channels: vision.in_channels,
            patch_size: vision.patch_size,
            temporal_patch_size: vision.temporal_patch_size,
            spatial_merge_size: vision.spatial_merge_size,
            output_hidden_size: vision.out_hidden_size,
            num_position_embeddings: vision.num_position_embeddings,
            deepstack_visual_indexes: vision.deepstack_visual_indexes.clone(),
            activation: Activation::GeluPytorchTanh,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.depth == 0
            || self.hidden_size == 0
            || self.num_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_heads)
            || self.patch_size == 0
            || self.temporal_patch_size != 2
            || self.spatial_merge_size == 0
            || self
                .deepstack_visual_indexes
                .iter()
                .any(|index| *index >= self.depth)
        {
            candle::bail!("invalid H3 Qwen3-VL vision dimensions: {self:?}");
        }
        let side = (self.num_position_embeddings as f64).sqrt().round() as usize;
        if side * side != self.num_position_embeddings {
            candle::bail!(
                "Qwen vision position count {} is not a square",
                self.num_position_embeddings
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn tiny() -> Self {
        Self {
            depth: 3,
            hidden_size: 8,
            intermediate_size: 16,
            num_heads: 2,
            in_channels: 3,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            output_hidden_size: 12,
            num_position_embeddings: 4,
            deepstack_visual_indexes: vec![0, 1, 2],
            activation: Activation::GeluPytorchTanh,
        }
    }
}

struct TemporalTwoConv3d {
    first: Conv2d,
    second: Conv2d,
}

impl TemporalTwoConv3d {
    fn new(
        input_channels: usize,
        output_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weights = vb.get(
            (output_channels, input_channels, 2, patch_size, patch_size),
            "weight",
        )?;
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        Ok(Self {
            first: Conv2d::new(weights.i((.., .., 0, .., ..))?.contiguous()?, None, config),
            second: Conv2d::new(weights.i((.., .., 1, .., ..))?.contiguous()?, None, config),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let first = conv2d_with_cpu_bf16_fallback(&self.first, &input.i((.., .., 0, .., ..))?)?;
        let second = conv2d_with_cpu_bf16_fallback(&self.second, &input.i((.., .., 1, .., ..))?)?;
        (first + second)?.unsqueeze(2)
    }
}

fn conv2d_with_cpu_bf16_fallback(conv: &Conv2d, input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu()
        && input.dtype() == DType::BF16
        && conv.weight().dtype() == DType::BF16
    {
        let weight = conv.weight().to_dtype(DType::F32)?;
        let bias = conv
            .bias()
            .map(|bias| bias.to_dtype(DType::F32))
            .transpose()?;
        Conv2d::new(weight, bias, *conv.config())
            .forward(&input.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)
    } else {
        conv.forward(input)
    }
}

struct PatchEmbed {
    projection: TemporalTwoConv3d,
    bias: Tensor,
    input_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(config: &Qwen3VlVisionDimensions, vb: VarBuilder) -> Result<Self> {
        let projection_vb = vb.pp("proj");
        Ok(Self {
            projection: TemporalTwoConv3d::new(
                config.in_channels,
                config.hidden_size,
                config.patch_size,
                projection_vb.clone(),
            )?,
            bias: projection_vb.get(config.hidden_size, "bias")?,
            input_channels: config.in_channels,
            patch_size: config.patch_size,
            temporal_patch_size: config.temporal_patch_size,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let expected_width =
            self.input_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let (_, width) = input.dims2()?;
        if width != expected_width {
            candle::bail!(
                "Qwen vision patches have width {width}, expected {expected_width} (C*T*P*P)"
            );
        }
        let patches = input.reshape((
            (),
            self.input_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        let projected = self
            .projection
            .forward(&patches)?
            .reshape(((), self.hidden_size))?;
        projected.broadcast_add(&self.bias.unsqueeze(0)?)
    }
}

struct VisionMlp {
    first: Linear,
    second: Linear,
    activation: Activation,
}

impl VisionMlp {
    fn new(config: &Qwen3VlVisionDimensions, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            first: linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("linear_fc1"),
            )?,
            second: linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("linear_fc2"),
            )?,
            activation: config.activation,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        linear_with_cpu_bf16_fallback(
            &self.second,
            &linear_with_cpu_bf16_fallback(&self.first, input)?.apply(&self.activation)?,
        )
    }
}

fn linear_with_cpu_bf16_fallback(linear: &Linear, input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu()
        && input.dtype() == DType::BF16
        && linear.weight().dtype() == DType::BF16
    {
        let weight = linear.weight().to_dtype(DType::F32)?;
        let bias = linear
            .bias()
            .map(|bias| bias.to_dtype(DType::F32))
            .transpose()?;
        Linear::new(weight, bias)
            .forward(&input.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)
    } else {
        linear.forward(input)
    }
}

fn matmul_with_cpu_bf16_fallback(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    if left.device().is_cpu() && left.dtype() == DType::BF16 && right.dtype() == DType::BF16 {
        left.to_dtype(DType::F32)?
            .matmul(&right.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)
    } else {
        left.matmul(right)
    }
}

fn rotate_half(input: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    let first = input.narrow(D::Minus1, 0, width / 2)?;
    let second = input.narrow(D::Minus1, width / 2, width / 2)?;
    Tensor::cat(&[&second.neg()?, &first], D::Minus1)
}

struct VisionAttention {
    qkv: Linear,
    projection: Linear,
    num_heads: usize,
    head_dimension: usize,
}

impl VisionAttention {
    fn new(config: &Qwen3VlVisionDimensions, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(config.hidden_size, config.hidden_size * 3, vb.pp("qkv"))?,
            projection: linear(config.hidden_size, config.hidden_size, vb.pp("proj"))?,
            num_heads: config.num_heads,
            head_dimension: config.hidden_size / config.num_heads,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cumulative_sequence_lengths: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let sequence = input.dim(0)?;
        let qkv = linear_with_cpu_bf16_fallback(&self.qkv, input)?
            .reshape((sequence, 3, self.num_heads, self.head_dimension))?
            .permute((1, 0, 2, 3))?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let apply_rotary = |value: Tensor| -> Result<Tensor> {
            let dtype = value.dtype();
            let value = value.to_dtype(DType::F32)?;
            (value.broadcast_mul(&cos)? + rotate_half(&value)?.broadcast_mul(&sin)?)?
                .to_dtype(dtype)
        };
        let query = apply_rotary(qkv.i(0)?)?;
        let key = apply_rotary(qkv.i(1)?)?;
        let value = qkv.i(2)?;
        let mut outputs = Vec::new();
        for bounds in cumulative_sequence_lengths.windows(2) {
            let start = bounds[0];
            let length = bounds[1].saturating_sub(start);
            if length == 0 {
                continue;
            }
            let query = query
                .narrow(0, start, length)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let key = key
                .narrow(0, start, length)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let value = value
                .narrow(0, start, length)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let key_transposed = key.transpose(2, 3)?.contiguous()?;
            let query_chunk = vision_attention_query_chunk_rows(self.num_heads, length);
            let mut chunk_outputs = Vec::with_capacity(length.div_ceil(query_chunk));
            for chunk_start in (0..length).step_by(query_chunk) {
                let rows = query_chunk.min(length - chunk_start);
                let query = query.narrow(2, chunk_start, rows)?;
                let scores = (matmul_with_cpu_bf16_fallback(&query, &key_transposed)?
                    / (self.head_dimension as f64).sqrt())?;
                let probabilities =
                    candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                        .to_dtype(value.dtype())?;
                chunk_outputs.push(matmul_with_cpu_bf16_fallback(&probabilities, &value)?);
            }
            let attended = if chunk_outputs.len() == 1 {
                chunk_outputs.pop().expect("one chunk")
            } else {
                Tensor::cat(&chunk_outputs, 2)?
            };
            outputs.push(
                attended
                    .squeeze(0)?
                    .transpose(0, 1)?
                    .reshape((length, self.num_heads * self.head_dimension))?
                    .to_dtype(input.dtype())?,
            );
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        linear_with_cpu_bf16_fallback(&self.projection, &Tensor::cat(&refs, 0)?)
    }
}

/// Score elements one attention chunk may materialize: `heads x Q x N`.
///
/// The vision tower attends densely within each image span and the released
/// tower has no window: a 2048-square reference is N = 16,384 patches, so the
/// unchunked `[1, 16, N, N]` scores plus their f32 softmax copy were ~34 GB
/// per layer (#1423) — impossible on any device and the whole of the CPU
/// route's host peak. Softmax is per query row, so chunking the QUERY axis is
/// exact: every row's scores, softmax, and value product are the same numbers
/// in the same order, only fewer rows at a time. 2^28 elements bounds one
/// chunk's transient at ~2 GiB (bf16 scores + f32 softmax + bf16 probs), and
/// is wide enough that FL2VA's 4,032-patch endpoint (16 x 4,032^2 = 2^28.0)
/// and every video block stay single-chunk, so their arithmetic is untouched.
const VISION_ATTENTION_SCORE_ELEMENT_BUDGET: usize = 1 << 28;

/// Query rows per attention chunk for one span of `length` patches.
pub fn vision_attention_query_chunk_rows(num_heads: usize, length: usize) -> usize {
    vision_attention_query_chunk_rows_for_budget(num_heads, length, score_element_budget())
}

fn vision_attention_query_chunk_rows_for_budget(
    num_heads: usize,
    length: usize,
    budget: usize,
) -> usize {
    let per_row = num_heads.max(1) * length.max(1);
    (budget / per_row).clamp(1, length.max(1))
}

#[cfg(test)]
thread_local! {
    static SCORE_ELEMENT_BUDGET_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

fn score_element_budget() -> usize {
    #[cfg(test)]
    {
        if let Some(budget) = SCORE_ELEMENT_BUDGET_OVERRIDE.with(|cell| cell.get()) {
            return budget;
        }
    }
    VISION_ATTENTION_SCORE_ELEMENT_BUDGET
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(config: &Qwen3VlVisionDimensions, vb: VarBuilder) -> Result<Self> {
        let norm_config = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(config.hidden_size, norm_config, vb.pp("norm1"))?,
            norm2: layer_norm(config.hidden_size, norm_config, vb.pp("norm2"))?,
            attention: VisionAttention::new(config, vb.pp("attn"))?,
            mlp: VisionMlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cumulative_sequence_lengths: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let attention = self.attention.forward(
            &self.norm1.forward(input)?,
            cumulative_sequence_lengths,
            cos,
            sin,
        )?;
        let hidden = (input + attention)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&hidden)?)?;
        hidden + mlp
    }
}

struct PatchMerger {
    norm: LayerNorm,
    postshuffle_norm: bool,
    merge_unit: usize,
    merged_hidden_size: usize,
    first: Linear,
    second: Linear,
}

impl PatchMerger {
    fn new(
        config: &Qwen3VlVisionDimensions,
        postshuffle_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let merge_unit = config.spatial_merge_size.pow(2);
        let merged_hidden_size = config.hidden_size * merge_unit;
        let norm_width = if postshuffle_norm {
            merged_hidden_size
        } else {
            config.hidden_size
        };
        Ok(Self {
            norm: layer_norm(
                norm_width,
                LayerNormConfig {
                    eps: 1e-6,
                    ..Default::default()
                },
                vb.pp("norm"),
            )?,
            postshuffle_norm,
            merge_unit,
            merged_hidden_size,
            first: linear(merged_hidden_size, merged_hidden_size, vb.pp("linear_fc1"))?,
            second: linear(
                merged_hidden_size,
                config.output_hidden_size,
                vb.pp("linear_fc2"),
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let sequence = input.dim(0)?;
        if !sequence.is_multiple_of(self.merge_unit) {
            candle::bail!(
                "Qwen vision sequence {sequence} is not divisible by merge unit {}",
                self.merge_unit
            );
        }
        let groups = sequence / self.merge_unit;
        let norm_input = if self.postshuffle_norm {
            input.reshape((groups, self.merged_hidden_size))?
        } else {
            input.clone()
        };
        let normalized = self.norm.forward(&norm_input)?;
        let merged = if self.postshuffle_norm {
            normalized
        } else {
            normalized.reshape((groups, self.merged_hidden_size))?
        };
        linear_with_cpu_bf16_fallback(
            &self.second,
            &linear_with_cpu_bf16_fallback(&self.first, &merged)?.gelu()?,
        )
    }
}

struct VisionRotaryEmbedding {
    inverse_frequency: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(dimension: usize, device: &Device) -> Result<Self> {
        let inverse_frequency = (0..dimension)
            .step_by(2)
            .map(|index| 1.0_f32 / 10_000.0_f32.powf(index as f32 / dimension as f32))
            .collect::<Vec<_>>();
        let length = inverse_frequency.len();
        Ok(Self {
            inverse_frequency: Tensor::from_vec(inverse_frequency, (1, length), device)?,
        })
    }

    fn frequencies(&self, sequence: usize) -> Result<Tensor> {
        Tensor::arange(0.0_f32, sequence as f32, self.inverse_frequency.device())?
            .unsqueeze(D::Minus1)?
            .broadcast_matmul(&self.inverse_frequency)
    }
}

pub(super) struct Qwen3VlVisionModel {
    patch_embed: PatchEmbed,
    position_embedding: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    deepstack_lookup: Vec<Option<usize>>,
    rotary: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    position_grid_side: usize,
    hidden_size: usize,
    parameter_dtype: DType,
}

impl Qwen3VlVisionModel {
    pub(super) fn new(config: &Qwen3VlVisionDimensions, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let patch_embed = PatchEmbed::new(config, vb.pp("patch_embed"))?;
        let position_embedding = embedding(
            config.num_position_embeddings,
            config.hidden_size,
            vb.pp("pos_embed"),
        )?;
        let blocks = (0..config.depth)
            .map(|index| VisionBlock::new(config, vb.pp("blocks").pp(index)))
            .collect::<Result<Vec<_>>>()?;
        let merger = PatchMerger::new(config, false, vb.pp("merger"))?;
        let deepstack_mergers = config
            .deepstack_visual_indexes
            .iter()
            .enumerate()
            .map(|(index, _)| {
                PatchMerger::new(config, true, vb.pp("deepstack_merger_list").pp(index))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut deepstack_lookup = vec![None; config.depth];
        for (merger, &layer) in config.deepstack_visual_indexes.iter().enumerate() {
            deepstack_lookup[layer] = Some(merger);
        }
        Ok(Self {
            patch_embed,
            position_embedding,
            blocks,
            merger,
            deepstack_mergers,
            deepstack_lookup,
            rotary: VisionRotaryEmbedding::new(
                config.hidden_size / config.num_heads / 2,
                vb.device(),
            )?,
            spatial_merge_size: config.spatial_merge_size,
            position_grid_side: (config.num_position_embeddings as f64).sqrt().round() as usize,
            hidden_size: config.hidden_size,
            parameter_dtype: vb.dtype(),
        })
    }

    pub(super) fn forward(
        &self,
        pixels: &Tensor,
        grid_thw: &Tensor,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let grid = parse_grid(grid_thw, self.spatial_merge_size)?;
        let expected_patches = grid
            .iter()
            .map(|grid| grid[0] * grid[1] * grid[2])
            .sum::<usize>();
        if pixels.dim(0)? != expected_patches {
            candle::bail!(
                "Qwen vision processor emitted {} patches but grid requires {expected_patches}",
                pixels.dim(0)?
            );
        }
        let embedded = self
            .patch_embed
            .forward(&pixels.to_dtype(self.parameter_dtype)?)?;
        checkpoint(ConditionerCheckpoint::VisionPatchEmbedding)?;
        let positions = self.interpolate_positions(&grid)?;
        let mut hidden = embedded.add(&positions)?;
        let rotary = self.rotary_positions(&grid)?;
        let doubled = Tensor::cat(&[&rotary, &rotary], D::Minus1)?;
        let cos = doubled.cos()?.to_dtype(DType::F32)?;
        let sin = doubled.sin()?.to_dtype(DType::F32)?;
        let cumulative = cumulative_sequence_lengths(&grid);
        let mut deepstack = Vec::with_capacity(self.deepstack_mergers.len());
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &cumulative, &cos, &sin)?;
            if let Some(merger) = self.deepstack_lookup[index] {
                deepstack.push(self.deepstack_mergers[merger].forward(&hidden)?);
            }
            checkpoint(ConditionerCheckpoint::VisionLayer {
                completed: index + 1,
                total: self.blocks.len(),
            })?;
        }
        Ok((self.merger.forward(&hidden)?, deepstack))
    }

    fn interpolate_positions(&self, grid: &[[usize; 3]]) -> Result<Tensor> {
        let device = self.position_embedding.embeddings().device();
        let dtype = self.position_embedding.embeddings().dtype();
        let mut indices: [Vec<u32>; 4] = Default::default();
        let mut weights: [Vec<f32>; 4] = Default::default();
        let mut areas = Vec::with_capacity(grid.len());
        for &[_, height, width] in grid {
            areas.push(height * width);
            let height_points = linspace(height, self.position_grid_side);
            let width_points = linspace(width, self.position_grid_side);
            for &height_point in &height_points {
                let h0 = height_point.floor() as usize;
                let h1 = (height_point.ceil() as usize).min(self.position_grid_side - 1);
                let dh = height_point - h0 as f32;
                for &width_point in &width_points {
                    let w0 = width_point.floor() as usize;
                    let w1 = (width_point.ceil() as usize).min(self.position_grid_side - 1);
                    let dw = width_point - w0 as f32;
                    for (slot, (row, column, weight)) in [
                        (h0, w0, (1.0 - dh) * (1.0 - dw)),
                        (h0, w1, (1.0 - dh) * dw),
                        (h1, w0, dh * (1.0 - dw)),
                        (h1, w1, dh * dw),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        indices[slot].push((row * self.position_grid_side + column) as u32);
                        weights[slot].push(weight);
                    }
                }
            }
        }
        let index_tensors = indices
            .into_iter()
            .map(|values| Tensor::from_vec(values.clone(), (values.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let weight_tensors = weights
            .into_iter()
            .map(|values| Tensor::from_vec(values.clone(), (values.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let index_refs = index_tensors.iter().collect::<Vec<_>>();
        let weight_refs = weight_tensors.iter().collect::<Vec<_>>();
        let gathered = self
            .position_embedding
            .forward(&Tensor::stack(&index_refs, 0)?)?;
        let interpolated = weighted_position_sum(
            &gathered,
            &Tensor::stack(&weight_refs, 0)?.unsqueeze(D::Minus1)?,
            dtype,
        )?;
        let mut outputs = Vec::with_capacity(grid.len());
        let mut offset = 0;
        for (&[temporal, height, width], area) in grid.iter().zip(areas) {
            let positions = interpolated
                .narrow(0, offset, area)?
                .repeat((temporal, 1))?;
            offset += area;
            outputs.push(
                positions
                    .reshape((
                        temporal,
                        height / self.spatial_merge_size,
                        self.spatial_merge_size,
                        width / self.spatial_merge_size,
                        self.spatial_merge_size,
                        self.hidden_size,
                    ))?
                    .permute((0, 1, 3, 2, 4, 5))?
                    .reshape((temporal * height * width, self.hidden_size))?,
            );
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)
    }

    fn rotary_positions(&self, grid: &[[usize; 3]]) -> Result<Tensor> {
        let maximum = grid
            .iter()
            .flat_map(|grid| [grid[1], grid[2]])
            .max()
            .unwrap_or(0);
        let frequency_table = self.rotary.frequencies(maximum)?;
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        for &[temporal, height, width] in grid {
            let merged_height = height / self.spatial_merge_size;
            let merged_width = width / self.spatial_merge_size;
            let mut frame = Vec::with_capacity(height * width);
            for block_row in 0..merged_height {
                for block_column in 0..merged_width {
                    for inner_row in 0..self.spatial_merge_size {
                        for inner_column in 0..self.spatial_merge_size {
                            frame.push((
                                block_row * self.spatial_merge_size + inner_row,
                                block_column * self.spatial_merge_size + inner_column,
                            ));
                        }
                    }
                }
            }
            for _ in 0..temporal {
                for &(row, column) in &frame {
                    rows.push(row as u32);
                    columns.push(column as u32);
                }
            }
        }
        let row_ids = Tensor::from_vec(rows.clone(), (rows.len(),), frequency_table.device())?;
        let column_ids =
            Tensor::from_vec(columns.clone(), (columns.len(),), frequency_table.device())?;
        let row = frequency_table.index_select(&row_ids, 0)?;
        let column = frequency_table.index_select(&column_ids, 0)?;
        Tensor::stack(&[&row, &column], D::Minus2)?.reshape((rows.len(), ()))
    }
}

fn weighted_position_sum(gathered: &Tensor, weights: &Tensor, dtype: DType) -> Result<Tensor> {
    gathered
        .to_dtype(DType::F32)?
        .broadcast_mul(weights)?
        .sum(0)?
        .to_dtype(dtype)
}

fn parse_grid(grid: &Tensor, merge: usize) -> Result<Vec<[usize; 3]>> {
    let rows = grid
        .to_device(&Device::Cpu)?
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let mut parsed = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 3 {
            candle::bail!("Qwen vision grid row has {} values, expected 3", row.len());
        }
        let value = [row[0] as usize, row[1] as usize, row[2] as usize];
        if value[0] == 0
            || value[1] == 0
            || value[2] == 0
            || !value[1].is_multiple_of(merge)
            || !value[2].is_multiple_of(merge)
        {
            candle::bail!("unsupported Qwen vision grid {value:?} for merge size {merge}");
        }
        parsed.push(value);
    }
    if parsed.is_empty() {
        candle::bail!("Qwen vision grid cannot be empty");
    }
    Ok(parsed)
}

fn linspace(steps: usize, position_grid_side: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![0.0];
    }
    (0..steps)
        .map(|index| index as f32 * (position_grid_side - 1) as f32 / (steps - 1) as f32)
        .collect()
}

fn cumulative_sequence_lengths(grid: &[[usize; 3]]) -> Vec<usize> {
    let mut cumulative = vec![0];
    let mut total = 0;
    for &[temporal, height, width] in grid {
        for _ in 0..temporal {
            total += height * width;
            cumulative.push(total);
        }
    }
    cumulative
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn interpolation_points_match_the_released_fp32_operation_order() {
        let points = linspace(16, 48);
        assert_eq!(points[3].to_bits(), 0x4116_6666);
        assert_eq!(points[7].to_bits(), 0x41af_7777);
        assert_eq!(points[14].to_bits(), 0x422f_7777);
    }

    #[test]
    fn bf16_position_interpolation_accumulates_with_fp32_weights() {
        let gathered = Tensor::new(
            &[[[4.09375_f32]], [[3.9375]], [[-4.875]], [[1.171875]]],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let weights = Tensor::new(
            &[
                [[0.15096787_f32]],
                [[0.21469931]],
                [[0.31650212]],
                [[0.31783068]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let value = weighted_position_sum(&gathered, &weights, DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(value, vec![0.29296875]);
    }

    #[test]
    fn tiny_vision_tower_returns_main_and_three_deepstack_rows() {
        let config = Qwen3VlVisionDimensions::tiny();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        let model = Qwen3VlVisionModel::new(&config, vb).unwrap();
        let pixels = Tensor::zeros((4, 24), DType::F32, &Device::Cpu).unwrap();
        let grid = Tensor::new(&[[1_u32, 2, 2]], &Device::Cpu).unwrap();
        let mut checkpoints = Vec::new();
        let (main, deepstack) = model
            .forward(&pixels, &grid, &mut |checkpoint| {
                checkpoints.push(checkpoint);
                Ok(())
            })
            .unwrap();
        assert_eq!(main.dims(), [1, 12]);
        assert_eq!(deepstack.len(), 3);
        assert!(deepstack.iter().all(|tensor| tensor.dims() == [1, 12]));
        assert_eq!(
            checkpoints.last(),
            Some(&ConditionerCheckpoint::VisionLayer {
                completed: 3,
                total: 3
            })
        );
    }

    #[test]
    fn tiny_bf16_vision_tower_executes_on_cpu_with_rounded_matmul_boundaries() {
        let config = Qwen3VlVisionDimensions::tiny();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::BF16, &Device::Cpu);
        let model = Qwen3VlVisionModel::new(&config, vb).unwrap();
        let pixels = Tensor::zeros((4, 24), DType::BF16, &Device::Cpu).unwrap();
        let grid = Tensor::new(&[[1_u32, 2, 2]], &Device::Cpu).unwrap();
        let (main, deepstack) = model.forward(&pixels, &grid, &mut |_| Ok(())).unwrap();
        assert_eq!(main.dims(), [1, 12]);
        assert_eq!(main.dtype(), DType::BF16);
        assert_eq!(deepstack.len(), 3);
        assert!(deepstack
            .iter()
            .all(|tensor| tensor.dims() == [1, 12] && tensor.dtype() == DType::BF16));
    }

    /// Chunking the query axis is exact: every row's scores, softmax, and
    /// value product are the same numbers, only fewer rows at a time. A
    /// forced budget that splits a 64-patch span into uneven chunks must
    /// reproduce the single-chunk tower to f32 rounding.
    #[test]
    fn query_chunked_attention_matches_the_unchunked_tower() {
        let config = Qwen3VlVisionDimensions::tiny();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        let model = Qwen3VlVisionModel::new(&config, vb).unwrap();
        // Deterministic non-trivial weights and pixels.
        for var in vars.data().lock().unwrap().values() {
            let n = var.elem_count();
            let values = (0..n)
                .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 0.2)
                .collect::<Vec<_>>();
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu).unwrap())
                .unwrap();
        }
        let patches = 8 * 8;
        let pixels = Tensor::from_vec(
            (0..patches * 24)
                .map(|i| ((i * 31 % 97) as f32 / 97.0) - 0.5)
                .collect::<Vec<_>>(),
            (patches, 24),
            &Device::Cpu,
        )
        .unwrap();
        let grid = Tensor::new(&[[1_u32, 8, 8]], &Device::Cpu).unwrap();
        let (single, single_deepstack) = model.forward(&pixels, &grid, &mut |_| Ok(())).unwrap();
        // heads 2 x 64 patches = 128 elements per query row; a budget of 640
        // forces 5-row chunks over a 64-row span (12 full chunks + 4 rows).
        assert_eq!(vision_attention_query_chunk_rows_for_budget(2, 64, 640), 5);
        SCORE_ELEMENT_BUDGET_OVERRIDE.with(|cell| cell.set(Some(640)));
        let chunked = model.forward(&pixels, &grid, &mut |_| Ok(()));
        SCORE_ELEMENT_BUDGET_OVERRIDE.with(|cell| cell.set(None));
        let (chunked, chunked_deepstack) = chunked.unwrap();
        let max_delta = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        assert!(
            max_delta(&single, &chunked) < 1e-5,
            "{}",
            max_delta(&single, &chunked)
        );
        for (a, b) in single_deepstack.iter().zip(&chunked_deepstack) {
            assert!(max_delta(a, b) < 1e-5);
        }
        assert!(
            single
                .flatten_all()
                .unwrap()
                .abs()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.0
        );
    }

    /// The released tower (16 heads) keeps FL2VA's 4,032-patch endpoint and
    /// every video block single-chunk, and splits a 2048-square reference's
    /// 16,384 patches into 1,024-row chunks — the shapes #1423 measured.
    #[test]
    fn query_chunk_policy_pins_the_released_shapes() {
        assert_eq!(vision_attention_query_chunk_rows(16, 4_032), 4_032);
        assert_eq!(vision_attention_query_chunk_rows(16, 2_304), 2_304);
        assert_eq!(vision_attention_query_chunk_rows(16, 16_384), 1_024);
        assert_eq!(vision_attention_query_chunk_rows(16, 7_296), 2_299);
        assert_eq!(vision_attention_query_chunk_rows(16, 0), 1);
        let budget = score_element_budget();
        assert!(16 * 4_032 * 4_032 <= budget);
    }

    #[test]
    fn mismatched_patch_tensor_is_rejected() {
        let config = Qwen3VlVisionDimensions::tiny();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3VlVisionModel::new(&config, vb).unwrap();
        let pixels = Tensor::zeros((3, 24), DType::F32, &Device::Cpu).unwrap();
        let grid = Tensor::new(&[[1_u32, 2, 2]], &Device::Cpu).unwrap();
        let error = model.forward(&pixels, &grid, &mut |_| Ok(())).unwrap_err();
        assert!(error.to_string().contains("emitted 3 patches"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_bf16_vision_tower_executes_on_cuda_with_fp32_rotary_and_softmax() {
        let device = Device::new_cuda(0).unwrap();
        let config = Qwen3VlVisionDimensions::tiny();
        let model =
            Qwen3VlVisionModel::new(&config, VarBuilder::zeros(DType::BF16, &device)).unwrap();
        let pixels = Tensor::zeros((4, 24), DType::F32, &device).unwrap();
        let grid = Tensor::new(&[[1_u32, 2, 2]], &Device::Cpu).unwrap();
        let (main, deepstack) = model.forward(&pixels, &grid, &mut |_| Ok(())).unwrap();
        assert_eq!(main.dims(), [1, 12]);
        assert_eq!(main.dtype(), DType::BF16);
        assert_eq!(deepstack.len(), 3);
        assert!(main
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}
