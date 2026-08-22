//! facexlib's BiSeNet face parser, ported to candle (#1225).
//!
//! Upstream is `xinntao/facexlib` at `260620ae` —
//! `facexlib/parsing/bisenet.py` and `facexlib/parsing/resnet.py` — loaded by
//! `facexlib/parsing/__init__.py:9-11` from that project's own
//! `parsing_bisenet.pth` release. PuLID builds it at
//! `PuLID/pulid/pipeline_flux.py:53` and reads only its first output at
//! `:164`.
//!
//! ## Why a candle port and not `candle-onnx`
//!
//! The face stack's two other graphs are ONNX run through
//! `candle_onnx::simple_eval`, and the obvious thing was to export BiSeNet the
//! same way. #1222's Step-0 op gate — the same machine-derived gate, run over
//! a real `torch.onnx.export` at opset 11 through
//! `pulid_face_probe gate` — says no, on three counts that are missing
//! evaluator support rather than an exporter idiom mold could normalize away:
//!
//! ```text
//! - candle-onnx's `MaxPool` rejects pads=1,1,1,1  (eval.rs:472-476, :507-511)
//! - candle-onnx's `Resize` rejects mode=linear    (eval.rs:2325)
//! - candle-onnx's `Resize` rejects
//!   coordinate_transformation_mode=align_corners  (eval.rs:2333)
//! ```
//!
//! The `MaxPool` padding is ResNet18's stem (`resnet.py:54`) and the two
//! `Resize` restrictions are the final logit upsample (`bisenet.py:135-137`),
//! so all three are load-bearing. Closing them means three separate
//! candle-onnx changes, against one mold-side port of a network that is 191
//! tensors of `Conv -> BatchNorm -> ReLU`. The decision, with the gate output
//! verbatim, is recorded in `docs/architecture/pulid-face-extraction.md`.
//!
//! The weights therefore arrive the way the EVA02-CLIP tower's do: a pinned
//! torch pickle, converted ONCE to safetensors by
//! [`crate::encoders::pickle_convert`], and loaded through an ordinary
//! `VarBuilder`. Mold's runtime never reads a pickle.
//!
//! ## What is not built
//!
//! `conv_out16` / `conv_out32` are BiSeNet's auxiliary training heads.
//! Upstream returns them (`bisenet.py:132-134`) and PuLID discards them
//! (`pipeline_flux.py:164` takes `[0]`), so they are retained in the derived
//! safetensors — which is a faithful re-container of the release — and simply
//! never constructed here. Nothing else is skipped.

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ModuleT;
use candle_nn::{batch_norm, conv2d_no_bias, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, VarBuilder};

/// `BiSeNet(num_class=19)` (`pipeline_flux.py:53` via
/// `facexlib/parsing/__init__.py:9`).
pub const NUM_CLASSES: usize = 19;

/// `nn.BatchNorm2d`'s default `eps`.
const BN_EPS: f64 = 1e-5;

/// The parser's own input normalization: ImageNet statistics, applied by PuLID
/// at `pipeline_flux.py:163` and nowhere else in the pipeline — the vision
/// tower uses the OpenAI CLIP statistics on the SAME crop.
pub const PARSE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// See [`PARSE_MEAN`] (`pipeline_flux.py:163`).
pub const PARSE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `ConvBNReLU` (`bisenet.py:9-17`).
struct ConvBnRelu {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBnRelu {
    fn new(
        in_chan: usize,
        out_chan: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d_no_bias(in_chan, out_chan, kernel, cfg, vb.pp("conv"))?,
            bn: batch_norm(out_chan, bn_config(), vb.pp("bn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.bn.forward_t(&self.conv.forward(x)?, false)?.relu()?)
    }
}

fn bn_config() -> BatchNormConfig {
    BatchNormConfig {
        eps: BN_EPS,
        remove_mean: true,
        affine: true,
        momentum: 0.1,
    }
}

/// `BasicBlock` (`resnet.py:10-38`).
struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl BasicBlock {
    fn new(in_chan: usize, out_chan: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv3x3 = |stride| Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let downsample = if in_chan != out_chan || stride != 1 {
            let vb = vb.pp("downsample");
            let cfg = Conv2dConfig {
                stride,
                ..Default::default()
            };
            Some((
                conv2d_no_bias(in_chan, out_chan, 1, cfg, vb.pp("0"))?,
                batch_norm(out_chan, bn_config(), vb.pp("1"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: conv2d_no_bias(in_chan, out_chan, 3, conv3x3(stride), vb.pp("conv1"))?,
            bn1: batch_norm(out_chan, bn_config(), vb.pp("bn1"))?,
            conv2: conv2d_no_bias(out_chan, out_chan, 3, conv3x3(1), vb.pp("conv2"))?,
            bn2: batch_norm(out_chan, bn_config(), vb.pp("bn2"))?,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self
            .bn1
            .forward_t(&self.conv1.forward(x)?, false)?
            .relu()?;
        let residual = self.bn2.forward_t(&self.conv2.forward(&residual)?, false)?;
        let shortcut = match &self.downsample {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x)?, false)?,
            None => x.clone(),
        };
        Ok((shortcut + residual)?.relu()?)
    }
}

/// `create_layer_basic` (`resnet.py:41-45`).
fn basic_layer(
    in_chan: usize,
    out_chan: usize,
    blocks: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Vec<BasicBlock>> {
    let mut layer = vec![BasicBlock::new(in_chan, out_chan, stride, vb.pp("0"))?];
    for index in 1..blocks {
        layer.push(BasicBlock::new(
            out_chan,
            out_chan,
            1,
            vb.pp(index.to_string()),
        )?);
    }
    Ok(layer)
}

/// `ResNet18` (`resnet.py:48-70`).
struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
}

impl ResNet18 {
    fn new(vb: VarBuilder) -> Result<Self> {
        let stem = Conv2dConfig {
            padding: 3,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv2d_no_bias(3, 64, 7, stem, vb.pp("conv1"))?,
            bn1: batch_norm(64, bn_config(), vb.pp("bn1"))?,
            layer1: basic_layer(64, 64, 2, 1, vb.pp("layer1"))?,
            layer2: basic_layer(64, 128, 2, 2, vb.pp("layer2"))?,
            layer3: basic_layer(128, 256, 2, 2, vb.pp("layer3"))?,
            layer4: basic_layer(256, 512, 2, 2, vb.pp("layer4"))?,
        })
    }

    /// Returns `(feat8, feat16, feat32)` — the 1/8, 1/16 and 1/32 stages.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self
            .bn1
            .forward_t(&self.conv1.forward(x)?, false)?
            .relu()?;
        // `resnet.py:54`: `MaxPool2d(kernel_size=3, stride=2, padding=1)`.
        // candle's pooling has no padding argument, so the border is added
        // explicitly. Zero is the correct fill here and not merely a
        // convenient one: the input is the output of a ReLU, so every real
        // sample is >= 0 and a zero border can never win a max against one.
        // Were this pooling anywhere else in the network the fill would have
        // to be -inf.
        let x = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
        let mut x = x.max_pool2d_with_stride(3, 2)?;
        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        let mut feat8 = x;
        for block in &self.layer2 {
            feat8 = block.forward(&feat8)?;
        }
        let mut feat16 = feat8.clone();
        for block in &self.layer3 {
            feat16 = block.forward(&feat16)?;
        }
        let mut feat32 = feat16.clone();
        for block in &self.layer4 {
            feat32 = block.forward(&feat32)?;
        }
        Ok((feat8, feat16, feat32))
    }
}

/// `F.avg_pool2d(feat, feat.size()[2:])` — a global spatial mean, keeping the
/// `[N, C, 1, 1]` shape upstream relies on for broadcasting.
///
/// Reduced over a FLATTENED trailing axis rather than over dims 2 and 3 in
/// place. Same arithmetic, and it keeps the reduction on the final dimension,
/// which is the shape candle's backends all agree on (see `crate::metal_reduce`
/// for what non-final reductions cost elsewhere).
fn global_mean(x: &Tensor) -> Result<Tensor> {
    let (n, c, _, _) = x.dims4()?;
    let mean = x.flatten_from(2)?.mean_keepdim(2)?;
    Ok(mean.reshape((n, c, 1, 1))?)
}

/// `AttentionRefinementModule` (`bisenet.py:32-49`).
struct AttentionRefinement {
    conv: ConvBnRelu,
    conv_atten: Conv2d,
    bn_atten: BatchNorm,
}

impl AttentionRefinement {
    fn new(in_chan: usize, out_chan: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: ConvBnRelu::new(in_chan, out_chan, 3, 1, 1, vb.pp("conv"))?,
            conv_atten: conv2d_no_bias(
                out_chan,
                out_chan,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_atten"),
            )?,
            bn_atten: batch_norm(out_chan, bn_config(), vb.pp("bn_atten"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.conv.forward(x)?;
        let atten = global_mean(&feat)?;
        let atten = self
            .bn_atten
            .forward_t(&self.conv_atten.forward(&atten)?, false)?;
        let atten = candle_nn::ops::sigmoid(&atten)?;
        Ok(feat.broadcast_mul(&atten)?)
    }
}

/// `FeatureFusionModule` (`bisenet.py:87-107`).
struct FeatureFusion {
    convblk: ConvBnRelu,
    conv1: Conv2d,
    conv2: Conv2d,
}

impl FeatureFusion {
    fn new(in_chan: usize, out_chan: usize, vb: VarBuilder) -> Result<Self> {
        let point = Conv2dConfig::default();
        Ok(Self {
            convblk: ConvBnRelu::new(in_chan, out_chan, 1, 1, 0, vb.pp("convblk"))?,
            conv1: conv2d_no_bias(out_chan, out_chan / 4, 1, point, vb.pp("conv1"))?,
            conv2: conv2d_no_bias(out_chan / 4, out_chan, 1, point, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, fsp: &Tensor, fcp: &Tensor) -> Result<Tensor> {
        let feat = self.convblk.forward(&Tensor::cat(&[fsp, fcp], 1)?)?;
        let atten = global_mean(&feat)?;
        let atten = self.conv1.forward(&atten)?.relu()?;
        let atten = candle_nn::ops::sigmoid(&self.conv2.forward(&atten)?)?;
        Ok((feat.broadcast_mul(&atten)? + &feat)?)
    }
}

/// `ContextPath` (`bisenet.py:52-84`).
struct ContextPath {
    resnet: ResNet18,
    arm16: AttentionRefinement,
    arm32: AttentionRefinement,
    conv_head32: ConvBnRelu,
    conv_head16: ConvBnRelu,
    conv_avg: ConvBnRelu,
}

impl ContextPath {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            resnet: ResNet18::new(vb.pp("resnet"))?,
            arm16: AttentionRefinement::new(256, 128, vb.pp("arm16"))?,
            arm32: AttentionRefinement::new(512, 128, vb.pp("arm32"))?,
            conv_head32: ConvBnRelu::new(128, 128, 3, 1, 1, vb.pp("conv_head32"))?,
            conv_head16: ConvBnRelu::new(128, 128, 3, 1, 1, vb.pp("conv_head16"))?,
            conv_avg: ConvBnRelu::new(512, 128, 1, 1, 0, vb.pp("conv_avg"))?,
        })
    }

    /// Returns `(feat8, feat_cp8)`. Upstream also returns `feat_cp16`, which
    /// only the auxiliary heads consume.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (feat8, feat16, feat32) = self.resnet.forward(x)?;
        let (_, _, h8, w8) = feat8.dims4()?;
        let (_, _, h16, w16) = feat16.dims4()?;
        let (_, _, h32, w32) = feat32.dims4()?;

        // `bisenet.py:70-72`. The 1x1 average is broadcast back over the 1/32
        // grid; `mode='nearest'` from a 1x1 source is a plain broadcast, which
        // is what `upsample_nearest2d` does here.
        let avg = self.conv_avg.forward(&global_mean(&feat32)?)?;
        let avg_up = avg.upsample_nearest2d(h32, w32)?;

        let feat32_sum = (self.arm32.forward(&feat32)? + avg_up)?;
        let feat32_up = self
            .conv_head32
            .forward(&feat32_sum.upsample_nearest2d(h16, w16)?)?;

        let feat16_sum = (self.arm16.forward(&feat16)? + feat32_up)?;
        let feat_cp8 = self
            .conv_head16
            .forward(&feat16_sum.upsample_nearest2d(h8, w8)?)?;

        Ok((feat8, feat_cp8))
    }
}

/// `BiSeNetOutput` (`bisenet.py:20-29`), built for the main head only.
struct BiSeNetOutput {
    conv: ConvBnRelu,
    conv_out: Conv2d,
}

impl BiSeNetOutput {
    fn new(in_chan: usize, mid_chan: usize, classes: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: ConvBnRelu::new(in_chan, mid_chan, 3, 1, 1, vb.pp("conv"))?,
            conv_out: conv2d_no_bias(
                mid_chan,
                classes,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_out"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv_out.forward(&self.conv.forward(x)?)?)
    }
}

/// The loaded face parser.
pub struct BiSeNetParser {
    cp: ContextPath,
    ffm: FeatureFusion,
    conv_out: BiSeNetOutput,
    device: Device,
}

impl BiSeNetParser {
    /// Build from the DERIVED safetensors — never from the `.pth`.
    ///
    /// `device` is asserted rather than honoured, for the same reason
    /// [`super::IdentityExtractor::load`] asserts it: the whole extraction runs
    /// on the host at admission, before the scheduler has leased a device.
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        ensure!(
            device.is_cpu(),
            "PuLID face parsing runs on the CPU beside the rest of the extraction"
        );
        Ok(Self {
            cp: ContextPath::new(vb.pp("cp")).context("building the BiSeNet context path")?,
            ffm: FeatureFusion::new(256, 256, vb.pp("ffm"))
                .context("building the BiSeNet feature-fusion module")?,
            conv_out: BiSeNetOutput::new(256, 256, NUM_CLASSES, vb.pp("conv_out"))
                .context("building the BiSeNet output head")?,
        device: device.clone(),
        })
    }

    /// Build from a derived safetensors file on disk.
    ///
    /// The convenience `new` wants for callers that hold a path rather than a
    /// `VarBuilder` — the `dev-bins` probe, and any future consumer that is
    /// not already inside the extraction.
    ///
    /// # Safety contract
    ///
    /// The file is mmap'd, so it must not be mutated while the parser holds
    /// it. Production reaches this through
    /// `pickle_convert::ensure_bisenet_parser_safetensors`, which has just
    /// authenticated those bytes against a compiled-in pin.
    pub fn from_safetensors(path: &std::path::Path, device: &Device) -> Result<Self> {
        // SAFETY: see the contract above.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path.to_path_buf()), DType::F32, device)
                .with_context(|| format!("reading the face parser {}", path.display()))?
        };
        Self::new(vb, device)
    }

    /// `BiSeNet.forward`'s first output (`bisenet.py:126-135`), already
    /// upsampled to the input resolution and reduced to per-pixel labels —
    /// which is `parsing_out.argmax(dim=1)` at `pipeline_flux.py:165`.
    ///
    /// `planar_rgb` is the crop in `[0, 1]`, CHW, exactly the tensor
    /// `pipeline_flux.py:161` builds; the ImageNet normalization at `:163`
    /// happens here so no caller can forget it and no caller can apply it
    /// twice.
    pub fn labels(&self, planar_rgb: &[f32], height: usize, width: usize) -> Result<Vec<u8>> {
        const CHANNELS: usize = 3;
        ensure!(
            height > 0 && width > 0 && planar_rgb.len() == CHANNELS * height * width,
            "expected {CHANNELS} x {height} x {width} planar samples, got {}",
            planar_rgb.len()
        );
        let mut normalized = planar_rgb.to_vec();
        let plane = height * width;
        for channel in 0..CHANNELS {
            let (mean, std) = (PARSE_MEAN[channel], PARSE_STD[channel]);
            for value in &mut normalized[channel * plane..(channel + 1) * plane] {
                *value = (*value - mean) / std;
            }
        }
        let input = Tensor::from_vec(normalized, (1, CHANNELS, height, width), &self.device)?
            .to_dtype(DType::F32)?;

        let (feat_res8, feat_cp8) = self.cp.forward(&input)?;
        let logits = self.conv_out.forward(&self.ffm.forward(&feat_res8, &feat_cp8)?)?;

        let (_, classes, low_h, low_w) = logits.dims4()?;
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        Ok(bilinear_align_corners_argmax(
            &flat, classes, low_h, low_w, height, width,
        ))
    }
}

/// `F.interpolate(out, (h, w), mode='bilinear', align_corners=True)` followed
/// by `argmax(dim=1)` (`bisenet.py:135`, `pipeline_flux.py:165`).
///
/// Fused because the upsample exists only to be argmaxed: taking the argmax of
/// the low-resolution logits instead and upsampling the LABELS is a different
/// function — interpolation between two classes' logits can elect a third —
/// and it is exactly the shortcut this fusion makes impossible to take by
/// accident.
///
/// `align_corners=True` puts the outermost samples exactly on the outermost
/// source samples, i.e. PyTorch's `area_pixel_compute_scale`:
/// `scale = (in - 1) / (out - 1)` for `out > 1`, and `0` for a degenerate
/// single-sample axis (`aten/src/ATen/native/UpSample.h`).
fn bilinear_align_corners_argmax(
    logits: &[f32],
    classes: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<u8> {
    let plane = src_h * src_w;
    let taps = |src: usize, dst: usize, index: usize| -> (usize, usize, f32) {
        let scale = if dst > 1 {
            (src as f64 - 1.0) / (dst as f64 - 1.0)
        } else {
            0.0
        };
        let position = scale * index as f64;
        let low = position.floor() as usize;
        let high = (low + 1).min(src - 1);
        (low, high, (position - low as f64) as f32)
    };

    let mut labels = vec![0_u8; dst_h * dst_w];
    for y in 0..dst_h {
        let (y0, y1, fy) = taps(src_h, dst_h, y);
        for x in 0..dst_w {
            let (x0, x1, fx) = taps(src_w, dst_w, x);
            let (mut best, mut best_value) = (0_u8, f32::NEG_INFINITY);
            for class in 0..classes {
                let base = class * plane;
                let top = logits[base + y0 * src_w + x0] * (1.0 - fx)
                    + logits[base + y0 * src_w + x1] * fx;
                let bottom = logits[base + y1 * src_w + x0] * (1.0 - fx)
                    + logits[base + y1 * src_w + x1] * fx;
                let value = top * (1.0 - fy) + bottom * fy;
                // `>` and not `>=`: torch's argmax returns the FIRST maximum.
                if value > best_value {
                    best_value = value;
                    best = class as u8;
                }
            }
            labels[y * dst_w + x] = best;
        }
    }
    labels
}


// ---------------------------------------------------------------------------
// The mask itself (`PuLID/pulid/pipeline_flux.py:166-170`).
// ---------------------------------------------------------------------------

/// The labels PuLID replaces with white (`pipeline_flux.py:166`).
///
/// Transcribed in upstream's own order, which is not sorted, so a reviewer can
/// diff the two literals directly. In BiSeNet's 19-class face scheme these are
/// background (0), hat (16), hair (17 is KEPT — 18 is the hat's shadow class),
/// ear rings (9), necklace (15), neck (14), cloth (18), and the eyeglasses /
/// ear pair (7, 8). Mold does not re-derive that meaning; the list is the
/// contract.
pub const BACKGROUND_LABELS: [u8; 8] = [0, 16, 18, 7, 8, 9, 14, 15];

/// `to_gray` (`pipeline_flux.py:113-116`) — Rec. 601 luma, broadcast back over
/// all three channels.
const GRAY_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];

/// True when `label` is one PuLID paints white.
pub fn is_background(label: u8) -> bool {
    BACKGROUND_LABELS.contains(&label)
}

/// `torch.where(bg, white_image, self.to_gray(input))`
/// (`pipeline_flux.py:167-169`), in place on the planar `[0, 1]` crop.
///
/// Two things this is NOT, both of which look reasonable and are wrong:
///
/// * It is not a background *removal*. The face is converted to greyscale too,
///   so what reaches the vision tower carries no colour at all — the identity
///   PuLID conditions on is shape, not complexion.
/// * The white is exact `1.0`, not the crop's border grey. facexlib's
///   `borderValue=(135, 133, 132)` fills pixels the warp had no source for;
///   this fills pixels the parser identified as not-face. They coincide often
///   and mean different things.
pub fn apply_pulid_face_mask(planar_rgb: &mut [f32], labels: &[u8]) -> Result<()> {
    const CHANNELS: usize = 3;
    let plane = labels.len();
    ensure!(
        planar_rgb.len() == CHANNELS * plane,
        "expected {CHANNELS} x {plane} planar samples for {plane} labels, got {}",
        planar_rgb.len()
    );
    for (index, label) in labels.iter().enumerate() {
        if is_background(*label) {
            for channel in 0..CHANNELS {
                planar_rgb[channel * plane + index] = 1.0;
            }
        } else {
            let gray: f32 = (0..CHANNELS)
                .map(|channel| GRAY_WEIGHTS[channel] * planar_rgb[channel * plane + index])
                .sum();
            for channel in 0..CHANNELS {
                planar_rgb[channel * plane + index] = gray;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parser_normalization_is_imagenets_not_clips() {
        // `pipeline_flux.py:163` normalizes the parser's input with the
        // ImageNet statistics while `:174` normalizes the tower's with the
        // OpenAI CLIP ones, on the same crop. Swapping them is invisible
        // except as a slightly wrong mask.
        assert_eq!(PARSE_MEAN, [0.485, 0.456, 0.406]);
        assert_eq!(PARSE_STD, [0.229, 0.224, 0.225]);
        assert_ne!(
            PARSE_MEAN,
            crate::encoders::eva_clip_preprocess::CLIP_MEAN,
            "the parser and the tower must not share one normalization"
        );
    }

    #[test]
    fn align_corners_pins_the_outermost_samples() {
        // Two classes on a 2x2 grid; class 1 wins the bottom-right corner and
        // class 0 the rest. With align_corners the corners of the OUTPUT must
        // reproduce the corners of the INPUT exactly.
        let logits = vec![
            // class 0
            1.0, 1.0, 1.0, 0.0, //
            // class 1
            0.0, 0.0, 0.0, 1.0,
        ];
        let labels = bilinear_align_corners_argmax(&logits, 2, 2, 2, 5, 5);
        assert_eq!(labels[0], 0, "top-left corner");
        assert_eq!(labels[4], 0, "top-right corner");
        assert_eq!(labels[20], 0, "bottom-left corner");
        assert_eq!(labels[24], 1, "bottom-right corner");
    }

    #[test]
    fn interpolating_logits_can_elect_a_class_that_wins_nowhere_at_low_resolution() {
        // The reason the upsample and the argmax are fused in that order. A
        // third class that is second everywhere can still take the midpoint,
        // so upsampling the labels instead of the logits is a different
        // function, not an optimization.
        let logits = vec![
            // class 0: wins the left sample
            10.0, 0.0, //
            // class 1: wins the right sample
            0.0, 10.0, //
            // class 2: never wins, but never loses much
            6.0, 6.0,
        ];
        let labels = bilinear_align_corners_argmax(&logits, 3, 1, 2, 1, 3);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 2, "the midpoint belongs to the class that is second everywhere");
        assert_eq!(labels[2], 1);
    }

    #[test]
    fn a_degenerate_axis_does_not_divide_by_zero() {
        let logits = vec![1.0, 0.0];
        let labels = bilinear_align_corners_argmax(&logits, 2, 1, 1, 3, 3);
        assert_eq!(labels, vec![0; 9]);
    }

    #[test]
    fn a_tie_goes_to_the_lower_class_as_torch_argmax_does() {
        let logits = vec![1.0, 1.0];
        let labels = bilinear_align_corners_argmax(&logits, 2, 1, 1, 1, 1);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn the_background_label_set_is_upstreams_verbatim() {
        assert_eq!(BACKGROUND_LABELS, [0, 16, 18, 7, 8, 9, 14, 15]);
        // Hair (17) and every facial-feature class stay, which is the whole
        // point: a background list that swallowed 17 would white out the hair
        // PuLID conditions on.
        assert!(!is_background(17));
        for face_label in 1..=6_u8 {
            assert!(!is_background(face_label), "{face_label} is a face class");
        }
    }

    #[test]
    fn the_mask_whitens_background_and_greys_the_face() {
        // Two pixels: index 0 is background, index 1 is skin.
        let mut planar = vec![
            // R
            0.2, 0.4, // G
            0.6, 0.5, // B
            0.8, 0.9,
        ];
        apply_pulid_face_mask(&mut planar, &[0, 1]).unwrap();
        assert_eq!(planar[0], 1.0);
        assert_eq!(planar[2], 1.0);
        assert_eq!(planar[4], 1.0);
        let gray = 0.299 * 0.4 + 0.587 * 0.5 + 0.114 * 0.9;
        for channel in 0..3 {
            assert!((planar[channel * 2 + 1] - gray).abs() < 1e-6, "{planar:?}");
        }
    }

    #[test]
    fn the_masked_face_keeps_no_colour_at_all() {
        // The face branch is greyscale, not a passthrough. A port that left
        // the face in colour would still produce a plausible-looking image.
        let mut planar = vec![1.0, 0.0, 0.0];
        apply_pulid_face_mask(&mut planar, &[1]).unwrap();
        assert_eq!(planar[0], planar[1]);
        assert_eq!(planar[1], planar[2]);
        assert!(planar[0] < 1.0);
    }

    /// The one probe seed this issue's goldens are drawn from, mirrored from
    /// `capture_parse_goldens.py`. "PULIDPRS".
    const SEED_PARSE_PROBE: u64 = 0x50554C49_44505253;

    fn parse_golden(name: &str) -> candle_core::Tensor {
        let path = crate::pulid_fixtures::testdata_dir().join("parse_goldens.safetensors");
        let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        tensors
            .get(name)
            .unwrap_or_else(|| panic!("golden {name} is missing from {}", path.display()))
            .clone()
    }

    /// Every face `capture_parse_goldens.py` wrote, by stem.
    fn golden_faces() -> Vec<String> {
        let sources = crate::pulid_fixtures::testdata_dir().join("faces/sources.json");
        let body = std::fs::read_to_string(&sources)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", sources.display()));
        body.split("\"file\": \"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .map(|file| file.trim_end_matches(".jpg").to_string())
            .collect()
    }

    fn planar_from_png(path: &std::path::Path) -> (Vec<f32>, usize, usize) {
        let image = image::open(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        crate::encoders::eva_clip_preprocess::planar_rgb_from_image(&image)
    }

    fn parser() -> Option<BiSeNetParser> {
        if std::env::var_os("MOLD_TEST_PULID_ASSETS").is_none() {
            return None;
        }
        let source = crate::pulid_fixtures::pulid_asset("parsing_bisenet.pth");
        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let destination = dir
            .path()
            .join(crate::encoders::pickle_convert::BISENET_DERIVED_FILENAME);
        crate::encoders::pickle_convert::convert_bisenet_parser(&source, &destination).unwrap();
        // SAFETY: a file this process just wrote into its own temporary
        // directory and is about to read once.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[destination], DType::F32, &Device::Cpu).unwrap()
        };
        Some(BiSeNetParser::new(vb, &Device::Cpu).unwrap())
    }

    /// The parser's own output, against `facexlib`'s.
    ///
    /// Compared as LABELS rather than as logits, because labels are what the
    /// mask consumes and a logit tolerance would have to be invented. The
    /// budget is a fraction of disagreeing pixels: an argmax at a class
    /// boundary can legitimately flip on an f32 last digit, and on a 512x512
    /// crop there are tens of thousands of such pixels. What must NOT drift is
    /// the interior of a region, which the histogram catches.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_parser_matches_facexlibs_labels() {
        /// Each class's pixel count, as a fraction of the whole crop.
        /// Measured worst on these four faces: below 5e-7, i.e. the two
        /// implementations agree on every one of the 262 144 pixels. The
        /// budget is 1e-4 — about 26 pixels — so an f32 last-digit flip at a
        /// class boundary is not a flake while a real drift is a failure.
        const HISTOGRAM_BUDGET: f64 = 1e-4;

        let Some(parser) = parser() else {
            eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset");
            return;
        };
        let faces = crate::pulid_fixtures::testdata_dir().join("faces");
        for stem in golden_faces() {
            let (planar, height, width) = planar_from_png(&faces.join(format!("{stem}.eva512.png")));
            let labels = parser.labels(&planar, height, width).unwrap();
            assert_eq!(labels.len(), height * width);

            let probe_indices = crate::pulid_fixtures::DeterministicStream::new(SEED_PARSE_PROBE)
                .indices(crate::pulid_fixtures::PROBE_COUNT, labels.len());
            let expected = parse_golden(&format!("{stem}.labels.probe"))
                .to_dtype(DType::U8)
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            let probe_disagreements = probe_indices
                .iter()
                .zip(&expected)
                .filter(|(index, want)| labels[**index as usize] != **want)
                .count();
            // Measured: 0 of 512 on every face. One is allowed so a single
            // boundary pixel moving under a different BLAS is not a flake.
            assert!(
                probe_disagreements <= 1,
                "{stem}: {probe_disagreements} of {} probed labels differ",
                expected.len()
            );

            let golden_histogram = parse_golden(&format!("{stem}.labels.histogram"))
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut histogram = vec![0_f64; NUM_CLASSES];
            for label in &labels {
                histogram[*label as usize] += 1.0;
            }
            for class in 0..NUM_CLASSES {
                let delta = (histogram[class] - golden_histogram[class] as f64).abs()
                    / labels.len() as f64;
                assert!(
                    delta <= HISTOGRAM_BUDGET,
                    "{stem}: class {class} covers {delta} more of the crop than upstream"
                );
            }
            let disagreeing = histogram
                .iter()
                .zip(&golden_histogram)
                .map(|(a, b)| (a - *b as f64).abs())
                .sum::<f64>()
                / (2.0 * labels.len() as f64);
            eprintln!(
                "{stem}: {probe_disagreements} probed labels differ; \
                 histogram moves {disagreeing:.6} of the crop"
            );
        }
    }

    /// The whole masked crop, against the image upstream feeds its tower.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_masked_crop_matches_upstreams() {
        /// Mean absolute channel delta, out of 255. Measured worst 0.0001.
        const MEAN_ABS_BUDGET: f64 = 0.02;
        /// Fraction of channels allowed to differ at all — the class-boundary
        /// pixels above, seen from the mask's side. Measured worst 9.2e-5.
        const DIFFERING_FRACTION_BUDGET: f64 = 1e-3;

        let Some(parser) = parser() else {
            eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset");
            return;
        };
        let faces = crate::pulid_fixtures::testdata_dir().join("faces");
        for stem in golden_faces() {
            let (mut planar, height, width) =
                planar_from_png(&faces.join(format!("{stem}.eva512.png")));
            let labels = parser.labels(&planar, height, width).unwrap();
            apply_pulid_face_mask(&mut planar, &labels).unwrap();

            let golden = image::open(faces.join(format!("{stem}.parsed512.png")))
                .unwrap()
                .to_rgb8();
            let plane = height * width;
            let (mut sum, mut differing) = (0.0_f64, 0_usize);
            for (index, pixel) in golden.pixels().enumerate() {
                for channel in 0..3 {
                    let actual = (planar[channel * plane + index] * 255.0).round();
                    let delta = (actual as f64 - pixel.0[channel] as f64).abs();
                    sum += delta;
                    if delta > 0.0 {
                        differing += 1;
                    }
                }
            }
            let mean = sum / (3 * plane) as f64;
            let fraction = differing as f64 / (3 * plane) as f64;
            eprintln!("{stem}: masked mean abs {mean:.4}, differing fraction {fraction:.6}");
            assert!(mean <= MEAN_ABS_BUDGET, "{stem}: mean abs delta {mean}");
            assert!(
                fraction <= DIFFERING_FRACTION_BUDGET,
                "{stem}: {fraction} of channels differ at all"
            );
        }
    }

    /// What the tower actually receives: masked, resized to 336, and
    /// CLIP-normalized. This is the acceptance pin the issue names as the
    /// "preprocessing tensor error", and it is the composition of two ports
    /// (the parser and the bicubic resize) rather than either alone.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_masked_preprocessed_tensor_matches_upstreams() {
        /// Measured worst 4.3e-5. The bound is the one
        /// `eva_clip_preprocess`'s own golden uses, since the resize is the
        /// same code and the mask contributes nothing to the error.
        const ABS_BUDGET: f32 = 1e-3;

        let Some(parser) = parser() else {
            eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset");
            return;
        };
        let faces = crate::pulid_fixtures::testdata_dir().join("faces");
        for stem in golden_faces() {
            let (mut planar, height, width) =
                planar_from_png(&faces.join(format!("{stem}.eva512.png")));
            let labels = parser.labels(&planar, height, width).unwrap();
            apply_pulid_face_mask(&mut planar, &labels).unwrap();
            let pixels = crate::encoders::eva_clip_preprocess::preprocess_planar_rgb(
                &planar,
                height,
                width,
                &Device::Cpu,
            )
            .unwrap();

            let flat = pixels.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            // The masked and label probes are drawn from the same stream, in
            // the order the capture script draws them.
            let mut stream = crate::pulid_fixtures::DeterministicStream::new(SEED_PARSE_PROBE);
            let _labels = stream.indices(crate::pulid_fixtures::PROBE_COUNT, height * width);
            let _masked = stream.indices(crate::pulid_fixtures::PROBE_COUNT, 3 * height * width);
            let indices = stream.indices(crate::pulid_fixtures::PROBE_COUNT, flat.len());

            let expected = parse_golden(&format!("{stem}.preprocess.probe"))
                .to_vec1::<f32>()
                .unwrap();
            let actual: Vec<f32> = indices.iter().map(|i| flat[*i as usize]).collect();
            let (absolute, _) = crate::pulid_fixtures::max_errors(&actual, &expected);
            eprintln!("{stem}: preprocessed max abs {absolute:.6}");
            assert!(absolute <= ABS_BUDGET, "{stem}: max abs {absolute}");
        }
    }

    #[test]
    fn a_mask_that_does_not_match_the_crop_is_refused() {
        let mut planar = vec![0.0_f32; 3 * 4];
        assert!(apply_pulid_face_mask(&mut planar, &[0, 1, 2]).is_err());
    }
}
