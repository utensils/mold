//! `scrfd_10g_bnkps` as a resident candle module (#1227).
//!
//! The hand port `docs/architecture/pulid-perf.md` §1 chose: the pinned ONNX
//! file stays the weight container, [`super::onnx_weights::WeightTape`] reads
//! it once at load, and this module runs ordinary `candle-core`/`candle-nn`
//! forward passes thereafter. Everything downstream of the raw head outputs —
//! anchor decoding, thresholding, NMS — is unchanged in
//! [`super::scrfd`], because none of it was ever inside the graph
//! (`scrfd.py:158-225`).
//!
//! ## The topology, and why it is written out rather than inferred
//!
//! Upstream SCRFD is `mmdet`-configured
//! (`insightface/detection/scrfd/configs/scrfd/scrfd_10g_bnkps.py`: a
//! `ResNetV1e`-style backbone, `PAFPN` neck, `SCRFDHead` with `stacked_convs=3`
//! and per-stride `cls`/`reg`/`kps` outputs), but the exported graph is the
//! authority for what actually runs: batch-norm is already folded into every
//! backbone convolution (the inventory carries **no** `BatchNormalization` for
//! this graph), and the export pairs `neck.fpn_convs.1`'s weight with
//! `neck.downsample_convs.0`'s bias — an exporter quirk this port reproduces
//! exactly by consuming parameters in graph order rather than by module name.
//!
//! Traced from the pinned graph
//! (`sha256 5838f7fe…`, `crates/mold-inference/testdata/pulid/onnx-inventory.json`):
//!
//! ```text
//!  stem   conv3x3/2 3->28, relu; conv3x3 28->28, relu; conv3x3 28->56, relu; maxpool 2/2
//!  stage1 3 x basic(56)                                  ->  160x160
//!  stage2 down(56->88) + 3 x basic(88)                    ->   80x80   C2 (stride 8)
//!  stage3 down(88->88) + 1 x basic(88)                    ->   40x40   C3 (stride 16)
//!  stage4 down(88->224) + 2 x basic(224)                  ->   20x20   C4 (stride 32)
//!  neck   lateral 1x1 -> 56 on each of C2/C3/C4
//!         top-down    P3 = L3 + nearest2x(L4); P2 = L2 + nearest2x(P3)
//!         fpn convs   F2 = c(P2); F3 = c(P3); F4 = c(L4)
//!         bottom-up   B3 = F3 + down(F2); B4 = F4 + down(B3)
//!         pafpn       H2 = F2; H3 = pafpn0(B3); H4 = pafpn1(B4)
//!  heads  per stride: 3 x (conv3x3 ->80, relu), then cls->2 (sigmoid),
//!         reg->8 (x learned scale), kps->20
//! ```
//!
//! ONNX op semantics used here, opset 11
//! (<https://github.com/onnx/onnx/blob/main/docs/Operators.md>):
//! `Conv`, `Relu`, `Add`, `MaxPool`, `AveragePool`, `Resize` (nearest,
//! `asymmetric`, `nearest_mode=floor`), `Mul`, `Transpose`, `Reshape`,
//! `Sigmoid`. The graph's `Shape`/`Gather`/`Slice`/`Concat`/`Unsqueeze` nodes
//! exist only to compute each `Resize`'s target size from the lateral it is
//! being added to; at a fixed square input those sizes are exactly twice the
//! source, which is what `upsample_nearest2d` is given.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::Conv2d;
use candle_onnx::onnx::ModelProto;

use super::onnx_weights::{place_input, WeightTape};
use super::scrfd::{ANCHORS_PER_CELL, FEATURE_STRIDES};

/// Backbone channel widths, in stage order.
const STAGE_CHANNELS: [usize; 4] = [56, 88, 88, 224];
/// Basic blocks per stage, the first of each stage after the first being a
/// strided downsample block.
const STAGE_BLOCKS: [usize; 4] = [3, 4, 2, 3];
/// Every neck level is projected to this width.
const NECK_CHANNELS: usize = 56;
/// `stacked_convs`, and their width.
const HEAD_STACK: usize = 3;
const HEAD_CHANNELS: usize = 80;

/// The nine raw head tensors, flattened exactly as the graph's
/// `Transpose(2,3,0,1)` + `Reshape` produce them: one row per anchor, in
/// `(y, x, anchor)` order.
#[derive(Debug, Clone)]
pub struct ScrfdRawOutputs {
    /// Per-anchor sigmoid score, one value per row, per stride.
    pub scores: [Vec<f32>; 3],
    /// Per-anchor distance-encoded box, four values per row, per stride.
    pub bboxes: [Vec<f32>; 3],
    /// Per-anchor distance-encoded keypoints, ten values per row, per stride.
    pub keypoints: [Vec<f32>; 3],
}

/// One residual block. `downsample` is present exactly on a stage's first
/// block, where the shortcut is `AveragePool(2, ceil) -> Conv1x1`.
struct BasicBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    downsample: Option<Conv2d>,
}

impl BasicBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some(conv) => {
                let pooled = average_pool_2x2_ceil(xs)?;
                conv.forward(&pooled)?
            }
            None => xs.clone(),
        };
        let out = self.conv1.forward(xs)?.relu()?;
        let out = self.conv2.forward(&out)?;
        Ok((out + identity)?.relu()?)
    }
}

/// One stride's detection head.
struct Head {
    stack: Vec<Conv2d>,
    cls: Conv2d,
    reg: Conv2d,
    kps: Conv2d,
    /// `bbox_head.scales.N.scale`, applied to the regression output.
    scale: f32,
}

/// The whole detector as resident tensors.
pub struct ScrfdNet {
    stem: Vec<Conv2d>,
    stages: Vec<Vec<BasicBlock>>,
    laterals: Vec<Conv2d>,
    fpn_convs: Vec<Conv2d>,
    downsample_convs: Vec<Conv2d>,
    pafpn_convs: Vec<Conv2d>,
    heads: Vec<Head>,
    device: Device,
}

impl ScrfdNet {
    /// Read every parameter out of a decoded graph, in graph order.
    ///
    /// `device` is a parameter so a future GPU path (`pulid-perf.md` §3) needs
    /// no change here; milestone 1 always passes `Device::Cpu`, which
    /// [`super::IdentityExtractor::load`] still asserts.
    pub fn new(model: &ModelProto, device: &Device) -> Result<Self> {
        let mut tape = WeightTape::new(model, device)?;

        // Stem: 3 -> 28 -> 28 -> 56, the first strided.
        let stem = vec![
            tape.next_conv(28, 3, 3, 2).context("stem conv 1")?,
            tape.next_conv(28, 28, 3, 1).context("stem conv 2")?,
            tape.next_conv(56, 28, 3, 1).context("stem conv 3")?,
        ];

        let mut stages = Vec::with_capacity(STAGE_CHANNELS.len());
        let mut in_channels = 56usize;
        for (stage, (&channels, &blocks)) in
            STAGE_CHANNELS.iter().zip(STAGE_BLOCKS.iter()).enumerate()
        {
            let mut built = Vec::with_capacity(blocks);
            for block in 0..blocks {
                // Every stage after the first opens with a strided block whose
                // shortcut is pooled and projected. SCRFD strides the FIRST
                // convolution of that block and pools the shortcut — the
                // opposite of `glintr100`, which strides the second and strides
                // its 1x1 shortcut instead (see `arcface_net.rs`). Swapping
                // them still type-checks and still produces the right shapes.
                let strided = stage > 0 && block == 0;
                let stride = if strided { 2 } else { 1 };
                let block_in = if block == 0 { in_channels } else { channels };
                let conv1 = tape
                    .next_conv(channels, block_in, 3, stride)
                    .with_context(|| format!("stage {stage} block {block} conv 1"))?;
                let conv2 = tape
                    .next_conv(channels, channels, 3, 1)
                    .with_context(|| format!("stage {stage} block {block} conv 2"))?;
                let downsample = if strided {
                    Some(
                        tape.next_conv(channels, block_in, 1, 1)
                            .with_context(|| format!("stage {stage} block {block} shortcut"))?,
                    )
                } else {
                    None
                };
                built.push(BasicBlock {
                    conv1,
                    conv2,
                    downsample,
                });
            }
            in_channels = channels;
            stages.push(built);
        }

        // Neck. The laterals come first, then the three fpn convs, then the
        // two bottom-up downsamples, then the two pafpn convs — graph order,
        // which is also the order the forward pass below visits them.
        let laterals = vec![
            tape.next_conv(NECK_CHANNELS, STAGE_CHANNELS[1], 1, 1)
                .context("neck lateral 0")?,
            tape.next_conv(NECK_CHANNELS, STAGE_CHANNELS[2], 1, 1)
                .context("neck lateral 1")?,
            tape.next_conv(NECK_CHANNELS, STAGE_CHANNELS[3], 1, 1)
                .context("neck lateral 2")?,
        ];
        let mut fpn_convs = Vec::with_capacity(3);
        for level in 0..3 {
            fpn_convs.push(
                tape.next_conv(NECK_CHANNELS, NECK_CHANNELS, 3, 1)
                    .with_context(|| format!("neck fpn conv {level}"))?,
            );
        }
        // The two bottom-up convolutions are interleaved with their `Add`s in
        // the graph, so they are consumed one at a time.
        let mut downsample_convs = Vec::with_capacity(2);
        let mut pafpn_convs = Vec::with_capacity(2);
        downsample_convs.push(
            tape.next_conv(NECK_CHANNELS, NECK_CHANNELS, 3, 2)
                .context("neck downsample conv 0")?,
        );
        downsample_convs.push(
            tape.next_conv(NECK_CHANNELS, NECK_CHANNELS, 3, 2)
                .context("neck downsample conv 1")?,
        );
        pafpn_convs.push(
            tape.next_conv(NECK_CHANNELS, NECK_CHANNELS, 3, 1)
                .context("neck pafpn conv 0")?,
        );
        pafpn_convs.push(
            tape.next_conv(NECK_CHANNELS, NECK_CHANNELS, 3, 1)
                .context("neck pafpn conv 1")?,
        );

        let mut heads = Vec::with_capacity(FEATURE_STRIDES.len());
        for stride in FEATURE_STRIDES {
            let mut stack = Vec::with_capacity(HEAD_STACK);
            for i in 0..HEAD_STACK {
                let input = if i == 0 { NECK_CHANNELS } else { HEAD_CHANNELS };
                stack.push(
                    tape.next_conv(HEAD_CHANNELS, input, 3, 1)
                        .with_context(|| format!("stride {stride} head conv {i}"))?,
                );
            }
            let cls = tape
                .next_conv(ANCHORS_PER_CELL, HEAD_CHANNELS, 3, 1)
                .with_context(|| format!("stride {stride} cls head"))?;
            let reg = tape
                .next_conv(ANCHORS_PER_CELL * 4, HEAD_CHANNELS, 3, 1)
                .with_context(|| format!("stride {stride} reg head"))?;
            let scale = tape
                .next_scalar_mul()
                .with_context(|| format!("stride {stride} regression scale"))?;
            let kps = tape
                .next_conv(ANCHORS_PER_CELL * 10, HEAD_CHANNELS, 3, 1)
                .with_context(|| format!("stride {stride} kps head"))?;
            heads.push(Head {
                stack,
                cls,
                reg,
                kps,
                scale,
            });
        }

        tape.finish()
            .context("the SCRFD graph carries parameters this port does not run")?;

        Ok(Self {
            stem,
            stages,
            laterals,
            fpn_convs,
            downsample_convs,
            pafpn_convs,
            heads,
            device: device.clone(),
        })
    }

    /// The device every parameter is resident on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run the network over one `[1, 3, H, W]` blob.
    ///
    /// The blob arrives from [`super::scrfd::ScrfdDetector::blob`], which builds
    /// it on the CPU because that is where an `RgbImage` lives. It is moved onto
    /// the weights' device first — a clone on today's CPU path, and the
    /// difference between working and a cross-device `Conv` failure on any
    /// other.
    pub fn forward(&self, blob: &Tensor) -> Result<ScrfdRawOutputs> {
        let mut xs = place_input(blob, &self.device)?;
        for conv in &self.stem {
            xs = conv.forward(&xs)?.relu()?;
        }
        // `MaxPool` kernel 2, stride 2, `ceil_mode = 0`.
        xs = xs.max_pool2d(2)?;

        let mut features = Vec::with_capacity(3);
        for (stage, blocks) in self.stages.iter().enumerate() {
            for block in blocks {
                xs = block.forward(&xs)?;
            }
            if stage > 0 {
                features.push(xs.clone());
            }
        }
        if features.len() != 3 {
            bail!(
                "the SCRFD backbone produced {} levels, expected 3",
                features.len()
            );
        }

        // Lateral 1x1 projections, then the top-down path.
        let l2 = self.laterals[0].forward(&features[0])?;
        let l3 = self.laterals[1].forward(&features[1])?;
        let l4 = self.laterals[2].forward(&features[2])?;
        let p3 = (&l3 + upsample_to(&l4, &l3)?)?;
        let p2 = (&l2 + upsample_to(&p3, &l2)?)?;

        // `neck.fpn_convs.2` runs on the *lateral*, not on a top-down sum:
        // level 2 is the top of the pyramid and has nothing above it.
        let f2 = self.fpn_convs[0].forward(&p2)?;
        let f3 = self.fpn_convs[1].forward(&p3)?;
        let f4 = self.fpn_convs[2].forward(&l4)?;

        let b3 = (&f3 + self.downsample_convs[0].forward(&f2)?)?;
        let b4 = (&f4 + self.downsample_convs[1].forward(&b3)?)?;

        let inputs = [
            f2,
            self.pafpn_convs[0].forward(&b3)?,
            self.pafpn_convs[1].forward(&b4)?,
        ];

        let mut scores = Vec::with_capacity(3);
        let mut bboxes = Vec::with_capacity(3);
        let mut keypoints = Vec::with_capacity(3);
        for (head, input) in self.heads.iter().zip(inputs.iter()) {
            let mut xs = input.clone();
            for conv in &head.stack {
                xs = conv.forward(&xs)?.relu()?;
            }
            let cls = candle_nn::ops::sigmoid(&head.cls.forward(&xs)?)?;
            let reg = (head.reg.forward(&xs)? * head.scale as f64)?;
            let kps = head.kps.forward(&xs)?;
            scores.push(anchor_major(&cls)?);
            bboxes.push(anchor_major(&reg)?);
            keypoints.push(anchor_major(&kps)?);
        }

        let take = |mut v: Vec<Vec<f32>>| -> [Vec<f32>; 3] {
            let c = v.pop().expect("three levels");
            let b = v.pop().expect("three levels");
            let a = v.pop().expect("three levels");
            [a, b, c]
        };
        Ok(ScrfdRawOutputs {
            scores: take(scores),
            bboxes: take(bboxes),
            keypoints: take(keypoints),
        })
    }
}

/// `AveragePool` kernel 2, stride 2, `ceil_mode = 1`.
///
/// `ceil_mode` only changes the output when an extent is odd, and it is the one
/// attribute `candle-onnx` silently ignored (`onnx_inventory.rs`'s
/// `ignored_attributes`). Every extent this pool ever sees at the pinned 640 px
/// input is even (160, 80, 40), so ceil and floor agree — and rather than
/// inherit the old evaluator's silent assumption, an odd extent is refused
/// here, because a rounded-down pool would produce a plausible detection from
/// the wrong receptive field.
fn average_pool_2x2_ceil(xs: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    if h % 2 != 0 || w % 2 != 0 {
        bail!(
            "SCRFD's shortcut pool has ceil_mode=1 and this port only implements the even case; \
             got {h}x{w}"
        );
    }
    Ok(xs.avg_pool2d(2)?)
}

/// `Resize` with `mode = nearest`, `coordinate_transformation_mode =
/// asymmetric`, `nearest_mode = floor`, to the exact spatial size of `like`.
///
/// The graph computes that size at run time from the lateral being added to;
/// with a square input it is always exactly twice the source, where nearest
/// resampling is a plain 2x pixel replication under any rounding convention.
/// A non-integer ratio is refused rather than approximated.
fn upsample_to(xs: &Tensor, like: &Tensor) -> Result<Tensor> {
    let (_, _, sh, sw) = xs.dims4()?;
    let (_, _, th, tw) = like.dims4()?;
    if th != sh * 2 || tw != sw * 2 {
        bail!("SCRFD's top-down Resize expected an exact 2x, got {sh}x{sw} -> {th}x{tw}");
    }
    Ok(xs.upsample_nearest2d(th, tw)?)
}

/// `Transpose(perm = 2,3,0,1)` then `Reshape(-1, k)`, flattened.
///
/// The head emits `[1, C, H, W]`; the transpose reorders it to `[H, W, 1, C]`
/// and the reshape splits `C` into `anchors x k`, so every row is one anchor
/// and the rows run `(y, x, anchor)`. That ordering is exactly what
/// [`super::scrfd::anchor_centers`] enumerates, and getting it wrong would pair
/// each score with another cell's box. Flattening `[N, H, W, C]` in place gives
/// the same value sequence for `N = 1`, without materializing the transpose.
fn anchor_major(xs: &Tensor) -> Result<Vec<f32>> {
    let (n, _, _, _) = xs.dims4()?;
    if n != 1 {
        bail!("the SCRFD head port runs one image at a time, got a batch of {n}");
    }
    Ok(xs
        .permute((0, 2, 3, 1))?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Evaluate the graph through `candle-onnx`, the path this module replaced.
///
/// Retained as the **parity oracle** for `tests/pulid_handport_parity.rs`, and
/// for nothing else: it is exactly the per-call re-materialization
/// `pulid-perf.md` §1 set out to remove, so no production path may call it.
#[doc(hidden)]
pub fn reference_forward(model: &ModelProto, blob: &Tensor) -> Result<ScrfdRawOutputs> {
    let graph = model
        .graph
        .as_ref()
        .context("the SCRFD model carries no graph")?;
    let input = graph
        .input
        .first()
        .context("the SCRFD graph declares no input")?
        .name
        .clone();
    let names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let outputs = candle_onnx::simple_eval(
        model,
        std::collections::HashMap::from([(input, blob.clone())]),
    )
    .context("SCRFD graph evaluation failed")?;
    let fetch = |name: &String| -> Result<Vec<f32>> {
        Ok(outputs
            .get(name)
            .with_context(|| format!("SCRFD produced no `{name}` output"))?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let fmc = FEATURE_STRIDES.len();
    let mut scores = Vec::new();
    let mut bboxes = Vec::new();
    let mut keypoints = Vec::new();
    for idx in 0..fmc {
        scores.push(fetch(&names[idx])?);
        bboxes.push(fetch(&names[idx + fmc])?);
        keypoints.push(fetch(&names[idx + fmc * 2])?);
    }
    let take = |mut v: Vec<Vec<f32>>| -> [Vec<f32>; 3] {
        let c = v.pop().expect("three levels");
        let b = v.pop().expect("three levels");
        let a = v.pop().expect("three levels");
        [a, b, c]
    };
    Ok(ScrfdRawOutputs {
        scores: take(scores),
        bboxes: take(bboxes),
        keypoints: take(keypoints),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block ladder must sum to the convolution count the pinned graph
    /// declares, or the tape would run out of nodes at load — on the real
    /// weights, not in CI.
    #[test]
    fn the_backbone_ladder_accounts_for_every_convolution_in_the_pinned_graph() {
        // 3 stem + per stage (2 per block, plus one shortcut on each strided
        // opener) + 3 laterals + 3 fpn + 2 downsample + 2 pafpn
        // + 3 x (3 stacked + cls + reg + kps).
        let backbone: usize = STAGE_BLOCKS
            .iter()
            .enumerate()
            .map(|(stage, blocks)| blocks * 2 + usize::from(stage > 0))
            .sum();
        let neck = 3 + 3 + 2 + 2;
        let heads = FEATURE_STRIDES.len() * (HEAD_STACK + 3);
        // The inventory records 6 + 46 + 6 = 58 convolutions.
        assert_eq!(3 + backbone + neck + heads, 58);
    }

    #[test]
    fn an_odd_pooling_extent_is_refused_rather_than_rounded() {
        let xs = Tensor::zeros((1, 2, 5, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let err = average_pool_2x2_ceil(&xs).unwrap_err();
        assert!(format!("{err}").contains("ceil_mode"), "{err}");
    }

    #[test]
    fn a_non_doubling_resize_is_refused() {
        let small = Tensor::zeros((1, 2, 5, 5), candle_core::DType::F32, &Device::Cpu).unwrap();
        let odd = Tensor::zeros((1, 2, 11, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(upsample_to(&small, &odd).is_err());
        let doubled = Tensor::zeros((1, 2, 10, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(upsample_to(&small, &doubled).is_ok());
    }

    /// The anchor-major flattening is the one place a silent transposition
    /// would pair every score with another cell's box.
    #[test]
    fn anchor_major_flattening_runs_y_then_x_then_anchor() {
        // [1, 2, 1, 3]: channel c, column w -> value 10*c + w.
        let xs = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0],
            (1, 2, 1, 3),
            &Device::Cpu,
        )
        .unwrap();
        // Rows are (y, x); each row carries both channels consecutively, which
        // for k = 1 means two anchors per cell.
        assert_eq!(
            anchor_major(&xs).unwrap(),
            vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]
        );
    }
}
