//! `glintr100` (iResNet100) as a resident candle module (#1227).
//!
//! The recognizer is where the re-materialization tax bites hardest: 261 MB of
//! initializers copied onto `Device::Cpu` on **every** `simple_eval` call
//! (`candle-onnx/src/eval.rs:191-232`), for a 112x112 forward pass.
//! `docs/architecture/pulid-perf.md` §1 chose to read them once and keep them.
//!
//! ## The topology
//!
//! Ported from `insightface/recognition/arcface_torch/backbones/iresnet.py`
//! (`IBasicBlock.forward`, lines 30-50; `IResNet.forward`, lines 145-160;
//! `iresnet100` = `IResNet(IBasicBlock, [3, 13, 30, 3])`, line 190), with the
//! exported graph as the authority for what survived the export. `IBasicBlock`
//! is pre-activation:
//!
//! ```text
//!   out = bn1(x)
//!   out = prelu(bn2(conv1(out)))
//!   out = bn3(conv2(out))
//!   out += downsample(x) if downsample else x
//! ```
//!
//! and the export folds `bn2`, `bn3`, and the downsample's batch-norm into
//! their convolutions' biases — which is why the graph carries 103 `Conv` nodes
//! but only 51 `BatchNormalization`s: one `bn1` per block (49), the trunk's
//! closing `bn2`, and the `features` batch-norm after the fully-connected
//! layer. `iresnet.py:124` gives every layer `stride=2`, including the first,
//! so 112 -> 56 -> 28 -> 14 -> 7 and the flatten is `512 * 7 * 7 = 25088`.
//!
//! ## The output is RAW
//!
//! `ArcFaceONNX.get` stores the network output unnormalized
//! (`arcface_onnx.py:63-66`) and PuLID conditions on exactly that
//! (`pipeline_flux.py:130`, `:156-158`). Nothing here normalizes; see
//! [`super::arcface`]'s module doc for the full argument.
//!
//! ONNX op semantics used here, opset 11
//! (<https://github.com/onnx/onnx/blob/main/docs/Operators.md>): `Conv`,
//! `BatchNormalization`, `PRelu`, `Add`, `Flatten`, `Gemm`.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv2d, PReLU};
use candle_onnx::onnx::ModelProto;

use super::onnx_weights::{FoldedBatchNorm, WeightTape};

/// `iresnet100`'s block ladder, `iresnet.py:190`.
const LAYER_BLOCKS: [usize; 4] = [3, 13, 30, 3];
/// Channel width per layer, `iresnet.py:120-133`.
const LAYER_CHANNELS: [usize; 4] = [64, 128, 256, 512];
/// The trunk's stem width, `iresnet.py:113`.
const STEM_CHANNELS: usize = 64;
/// Spatial extent reaching the flatten: 112 halved four times.
const FINAL_EXTENT: usize = 7;
/// Embedding width, `iresnet.py:139`.
pub const EMBEDDING_DIM: usize = 512;

/// One pre-activation IR block.
struct IrBlock {
    bn1: FoldedBatchNorm,
    conv1: Conv2d,
    prelu: PReLU,
    conv2: Conv2d,
    /// The 1x1 projection on a stage's first block. Unlike SCRFD's, it is
    /// **strided** rather than pooled, and it is applied to the block's input
    /// *before* `bn1` (`iresnet.py:41-42`: `identity = self.downsample(x)`).
    downsample: Option<Conv2d>,
}

impl IrBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let out = self.bn1.forward(xs)?;
        let out = self.conv1.forward(&out)?;
        let out = self.prelu.forward(&out)?;
        let out = self.conv2.forward(&out)?;
        Ok((out + identity)?)
    }
}

/// `glintr100`'s iResNet100 as resident tensors.
pub struct IResNet100 {
    stem_conv: Conv2d,
    stem_prelu: PReLU,
    blocks: Vec<IrBlock>,
    bn2: FoldedBatchNorm,
    /// `[512, 25088]`, applied as `x @ w^T` because the graph's `Gemm` sets
    /// `transB = 1`.
    fc_weight: Tensor,
    fc_bias: Tensor,
    features: FoldedBatchNorm,
    device: Device,
}

impl IResNet100 {
    /// Read every parameter out of a decoded graph, in graph order.
    ///
    /// `device` is a parameter so a future GPU path needs no change here;
    /// milestone 1 always passes `Device::Cpu`.
    pub fn new(model: &ModelProto, device: &Device) -> Result<Self> {
        let mut tape = WeightTape::new(model, device)?;
        let stem_conv = tape
            .next_conv(STEM_CHANNELS, 3, 3, 1)
            .context("the iResNet stem convolution")?;
        let stem_prelu = tape
            .next_prelu(STEM_CHANNELS)
            .context("the iResNet stem PReLU")?;

        let mut blocks = Vec::with_capacity(LAYER_BLOCKS.iter().sum());
        let mut in_channels = STEM_CHANNELS;
        for (layer, (&channels, &count)) in
            LAYER_CHANNELS.iter().zip(LAYER_BLOCKS.iter()).enumerate()
        {
            for block in 0..count {
                // `iresnet.py:124`: every layer is built with `stride=2`, so
                // each layer's FIRST block strides and projects its shortcut.
                let strided = block == 0;
                let stride = if strided { 2 } else { 1 };
                let block_in = if strided { in_channels } else { channels };
                let bn1 = tape
                    .next_batch_norm(block_in)
                    .with_context(|| format!("layer {layer} block {block} bn1"))?;
                let conv1 = tape
                    .next_conv(channels, block_in, 3, 1)
                    .with_context(|| format!("layer {layer} block {block} conv 1"))?;
                let prelu = tape
                    .next_prelu(channels)
                    .with_context(|| format!("layer {layer} block {block} prelu"))?;
                let conv2 = tape
                    .next_conv(channels, channels, 3, stride)
                    .with_context(|| format!("layer {layer} block {block} conv 2"))?;
                let downsample = if strided {
                    Some(
                        tape.next_conv(channels, block_in, 1, 2)
                            .with_context(|| format!("layer {layer} block {block} shortcut"))?,
                    )
                } else {
                    None
                };
                blocks.push(IrBlock {
                    bn1,
                    conv1,
                    prelu,
                    conv2,
                    downsample,
                });
            }
            in_channels = channels;
        }

        let bn2 = tape
            .next_batch_norm(EMBEDDING_DIM)
            .context("the iResNet closing batch-norm")?;
        let (fc_weight, fc_bias) = tape
            .next_gemm(EMBEDDING_DIM, EMBEDDING_DIM * FINAL_EXTENT * FINAL_EXTENT)
            .context("the iResNet fully-connected layer")?;
        let features = tape
            .next_batch_norm(EMBEDDING_DIM)
            .context("the iResNet feature batch-norm")?;
        tape.finish()
            .context("the ArcFace graph carries parameters this port does not run")?;

        Ok(Self {
            stem_conv,
            stem_prelu,
            blocks,
            bn2,
            fc_weight,
            fc_bias,
            features,
            device: device.clone(),
        })
    }

    /// The device every parameter is resident on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Embed one `[1, 3, 112, 112]` blob, returning the RAW 512-d output.
    pub fn forward(&self, blob: &Tensor) -> Result<Vec<f32>> {
        let (batch, _, _, _) = blob.dims4()?;
        if batch != 1 {
            bail!("the ArcFace port runs one crop at a time, got a batch of {batch}");
        }
        let mut xs = self.stem_conv.forward(blob)?;
        xs = self.stem_prelu.forward(&xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        xs = self.bn2.forward(&xs)?;
        // ONNX `Flatten(axis = 1)`: `[1, C, H, W] -> [1, C * H * W]`, C-major,
        // which is what the fully-connected layer's 25088 columns are ordered
        // by.
        let xs = xs.flatten_from(1)?;
        // `Gemm(alpha = 1, beta = 1, transB = 1)`.
        let xs = xs.matmul(&self.fc_weight.t()?.contiguous()?)?;
        let xs = xs.broadcast_add(&self.fc_bias.reshape((1, EMBEDDING_DIM))?)?;
        let xs = self.features.forward(&xs)?;
        Ok(xs.flatten_all()?.to_vec1::<f32>()?)
    }
}

/// Evaluate the graph through `candle-onnx`, the path this module replaced.
///
/// Retained as the **parity oracle** for `tests/pulid_handport_parity.rs`, and
/// for nothing else — it is exactly the per-call re-materialization
/// `pulid-perf.md` §1 set out to remove.
#[doc(hidden)]
pub fn reference_forward(model: &ModelProto, blob: &Tensor) -> Result<Vec<f32>> {
    let graph = model
        .graph
        .as_ref()
        .context("the ArcFace model carries no graph")?;
    let input = graph
        .input
        .first()
        .context("the ArcFace graph declares no input")?
        .name
        .clone();
    let output = graph
        .output
        .first()
        .context("the ArcFace graph declares no output")?
        .name
        .clone();
    let outputs = candle_onnx::simple_eval(
        model,
        std::collections::HashMap::from([(input, blob.clone())]),
    )
    .context("ArcFace graph evaluation failed")?;
    Ok(outputs
        .get(&output)
        .with_context(|| format!("ArcFace produced no `{output}` output"))?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ladder must account for exactly the op counts the pinned graph
    /// declares, or the tape runs out of nodes on the real weights.
    #[test]
    fn the_block_ladder_matches_the_pinned_graphs_op_counts() {
        let blocks: usize = LAYER_BLOCKS.iter().sum();
        assert_eq!(blocks, 49);
        // 1 stem + 2 per block + one 1x1 shortcut per layer.
        let convs = 1 + blocks * 2 + LAYER_BLOCKS.len();
        // The inventory records 4 + 95 + 4 = 103 convolutions.
        assert_eq!(convs, 103);
        // One `bn1` per block, plus `bn2` and `features`.
        assert_eq!(blocks + 2, 51);
        // One PReLU per block, plus the stem's.
        assert_eq!(blocks + 1, 50);
        // One residual add per block.
        assert_eq!(blocks, 49);
    }

    #[test]
    fn the_flatten_width_is_the_fully_connected_layers_column_count() {
        assert_eq!(EMBEDDING_DIM * FINAL_EXTENT * FINAL_EXTENT, 25_088);
        // 112 halved once per layer.
        assert_eq!(112 >> LAYER_BLOCKS.len(), FINAL_EXTENT);
    }
}
