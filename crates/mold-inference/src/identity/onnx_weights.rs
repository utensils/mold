//! Resident weights for the hand-ported face networks (#1227).
//!
//! `candle-onnx`'s `simple_eval` re-materializes **every** initializer on
//! `Device::Cpu` on **every** call (`candle-onnx/src/eval.rs:191-232`), so the
//! second call costs what the thousandth does — 261 MB of `glintr100` copied
//! per embedding. `docs/architecture/pulid-perf.md` §1 chose option A: keep the
//! ONNX file as the weight *container*, parse it once at load, and run ordinary
//! resident `candle-core`/`candle-nn` forward passes thereafter.
//!
//! This module is the load-time half. It exists so [`super::scrfd_net`] and
//! [`super::arcface_net`] never touch a `ModelProto` at inference time, and so
//! the `ModelProto` itself can be dropped the moment construction finishes.
//!
//! ## Why the ONNX file and not a converted safetensors sidecar
//!
//! `encoders::eva_clip_convert`'s secure-publish pattern exists because its
//! source is a **torch pickle**, which mold's runtime must never open more than
//! once, in a private staging copy. Nothing of the kind applies here: both
//! graphs are already read through
//! [`super::onnx_graph::load_onnx_model`], which performs one bounded read of a
//! retained descriptor and authenticates *those exact bytes* against the
//! manifest pin before anything is decoded. Converting would add a second
//! artifact, a second digest, a second staging-and-publish dance, and a cache
//! whose freshness is another thing to get wrong — to save a one-time proto
//! parse that already happens. The pinned file stays the single authority.
//!
//! ## Why the weights are consumed in graph order
//!
//! Both graphs name most of their parameters opaquely (`547`, `1335`), so a
//! lookup table of hard-coded strings would be an unreadable transcription with
//! nothing checking it. Instead [`WeightTape`] walks `graph.node` in
//! **topological order** and hands each parameterized op's tensors to the
//! hand-written module in the order that module visits them, asserting the
//! shape of every one. A reordered or substituted graph fails at load with the
//! op index and the shapes that disagreed, and [`WeightTape::finish`] refuses a
//! graph with parameters nobody consumed.
//!
//! ONNX op semantics are taken from the spec
//! (<https://github.com/onnx/onnx/blob/main/docs/Operators.md>), opset 11 for
//! both graphs; each reimplemented op cites its section where it is used.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, PReLU};
use candle_onnx::onnx::{tensor_proto, NodeProto, TensorProto};

/// ONNX ops that carry learned parameters in these two graphs.
///
/// The tape refuses to skip one of these. Everything else — `Relu`, `Add`,
/// `Shape`, `Gather`, `Slice`, `Concat`, `Unsqueeze`, `Reshape`, `Transpose`,
/// `Sigmoid`, `MaxPool`, `AveragePool`, `Resize`, `Flatten` — is structure the
/// hand-written module already encodes, and is stepped over.
const PARAMETERIZED: [&str; 5] = ["Conv", "BatchNormalization", "PRelu", "Gemm", "Mul"];

/// `BatchNormalization` in inference mode, pre-folded.
///
/// ONNX spec: `Y = (X - mean) / sqrt(var + epsilon) * scale + B`, normalizing
/// over the channel axis (dim 1). The division and the two affine terms are
/// constant per channel, so they collapse at load into one multiply and one
/// add:
///
/// ```text
///   w = scale / sqrt(var + epsilon)
///   b = B - mean * w
///   Y = X * w + b
/// ```
///
/// That is the same arithmetic, not an approximation of it, and the fold is
/// computed in `f64` so the reciprocal square root does not lose a bit that the
/// per-element form would have kept. `epsilon` comes from the node's own
/// attribute — never a constant retyped here, because a graph exported with a
/// different one would then be evaluated with mold's.
#[derive(Debug, Clone)]
pub struct FoldedBatchNorm {
    /// Per-channel multiplier, shaped for the rank it will meet.
    weight: Tensor,
    /// Per-channel offset, shaped for the rank it will meet.
    bias: Tensor,
    channels: usize,
}

impl FoldedBatchNorm {
    /// Apply to an `[N, C, ...]` tensor of any rank ≥ 2.
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let rank = xs.rank();
        let mut shape = vec![1usize; rank];
        shape[1] = self.channels;
        let w = self.weight.reshape(shape.clone())?;
        let b = self.bias.reshape(shape)?;
        xs.broadcast_mul(&w)?.broadcast_add(&b)
    }

    /// Channel count, so a builder can assert the shape it expected.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Every float initializer of one graph, plus a cursor over its nodes.
pub struct WeightTape<'a> {
    nodes: Vec<&'a NodeProto>,
    initializers: HashMap<&'a str, &'a TensorProto>,
    cursor: usize,
    device: Device,
}

impl<'a> WeightTape<'a> {
    /// Start a tape over a decoded graph.
    pub fn new(model: &'a candle_onnx::onnx::ModelProto, device: &Device) -> Result<Self> {
        let graph = model
            .graph
            .as_ref()
            .context("the ONNX model carries no graph")?;
        Ok(Self {
            nodes: graph.node.iter().collect(),
            initializers: graph
                .initializer
                .iter()
                .map(|t| (t.name.as_str(), t))
                .collect(),
            cursor: 0,
            device: device.clone(),
        })
    }

    /// Advance to the next node of `op_type`, refusing to step over any other
    /// parameterized op.
    fn next_node(&mut self, op_type: &str) -> Result<&'a NodeProto> {
        while self.cursor < self.nodes.len() {
            let node = self.nodes[self.cursor];
            self.cursor += 1;
            if node.op_type == op_type {
                return Ok(node);
            }
            if PARAMETERIZED.contains(&node.op_type.as_str()) {
                bail!(
                    "the graph's parameter order does not match the hand-ported module: \
                     expected `{op_type}` at node {}, found `{}`",
                    self.cursor - 1,
                    node.op_type
                );
            }
        }
        bail!("the graph ran out of nodes while looking for `{op_type}`")
    }

    /// Every parameterized op has been consumed.
    pub fn finish(mut self) -> Result<()> {
        while self.cursor < self.nodes.len() {
            let node = self.nodes[self.cursor];
            self.cursor += 1;
            if PARAMETERIZED.contains(&node.op_type.as_str()) {
                bail!(
                    "the graph carries a `{}` at node {} that the hand-ported module never \
                     consumed",
                    node.op_type,
                    self.cursor - 1
                );
            }
        }
        Ok(())
    }

    /// Materialize one float initializer by name.
    ///
    /// ONNX stores a `FLOAT` tensor either in `raw_data` (little-endian, which
    /// is the only byte order the format defines) or in the `float_data`
    /// repeated field. Both are accepted; anything else fails rather than being
    /// coerced, because a silently reinterpreted dtype is a wrong render, not
    /// an error.
    fn initializer(&self, name: &str) -> Result<Tensor> {
        let proto = self
            .initializers
            .get(name)
            .with_context(|| format!("the graph names `{name}` but carries no such initializer"))?;
        if proto.data_type != tensor_proto::DataType::Float as i32 {
            bail!(
                "initializer `{name}` is ONNX data type {}, expected FLOAT",
                proto.data_type
            );
        }
        let dims: Vec<usize> = proto
            .dims
            .iter()
            .map(|d| usize::try_from(*d).context("a negative initializer dimension"))
            .collect::<Result<_>>()?;
        let count: usize = dims.iter().product();
        let values: Vec<f32> = if !proto.raw_data.is_empty() {
            if proto.raw_data.len() != count * 4 {
                bail!(
                    "initializer `{name}` holds {} raw bytes for {count} floats",
                    proto.raw_data.len()
                );
            }
            proto
                .raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            proto.float_data.clone()
        };
        if values.len() != count {
            bail!(
                "initializer `{name}` holds {} values for dims {dims:?}",
                values.len()
            );
        }
        Tensor::from_vec(values, dims, &self.device)
            .with_context(|| format!("materializing initializer `{name}`"))
    }

    fn expect_dims(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
        if tensor.dims() != expected {
            bail!(
                "initializer `{name}` has shape {:?}, the module expected {expected:?}",
                tensor.dims()
            );
        }
        Ok(())
    }

    fn int_attr(node: &NodeProto, name: &str) -> Option<i64> {
        node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
    }

    fn ints_attr(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
        node.attribute
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.ints.clone())
    }

    fn float_attr(node: &NodeProto, name: &str) -> Option<f32> {
        node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
    }

    /// The next `Conv`, as a `candle_nn::Conv2d`.
    ///
    /// ONNX `Conv` (opset 11) with `auto_pad = NOTSET`: `pads` is
    /// `[x1_begin, x2_begin, x1_end, x2_end]`, and candle's `Conv2dConfig`
    /// carries one symmetric padding, so an asymmetric `pads` is refused rather
    /// than silently halved. Both shipped graphs use `[0,0,0,0]` or
    /// `[1,1,1,1]`, and `dilations` is always `[1,1]`, `group` always 1.
    pub fn next_conv(
        &mut self,
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Conv2d> {
        let node = self.next_node("Conv")?;
        let kernel_shape = Self::ints_attr(node, "kernel_shape").unwrap_or_default();
        if kernel_shape != vec![kernel as i64, kernel as i64] {
            bail!(
                "Conv at node {}: kernel_shape {kernel_shape:?}, expected [{kernel}, {kernel}]",
                self.cursor - 1
            );
        }
        let strides = Self::ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        if strides != vec![stride as i64, stride as i64] {
            bail!(
                "Conv at node {}: strides {strides:?}, expected [{stride}, {stride}]",
                self.cursor - 1
            );
        }
        let pads = Self::ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        if pads.len() != 4 || pads[0] != pads[1] || pads[0] != pads[2] || pads[0] != pads[3] {
            bail!(
                "Conv at node {}: candle takes one symmetric padding, the graph asks for {pads:?}",
                self.cursor - 1
            );
        }
        let padding = usize::try_from(pads[0]).context("a negative Conv padding")?;
        let dilations = Self::ints_attr(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        if dilations != vec![1, 1] {
            bail!(
                "Conv at node {}: dilations {dilations:?} are unsupported",
                self.cursor - 1
            );
        }
        if Self::int_attr(node, "group").unwrap_or(1) != 1 {
            bail!(
                "Conv at node {}: grouped convolution is unsupported",
                self.cursor - 1
            );
        }
        let weight_name = node
            .input
            .get(1)
            .context("a Conv with no weight input")?
            .clone();
        let weight = self.initializer(&weight_name)?;
        Self::expect_dims(
            &weight_name,
            &weight,
            &[out_channels, in_channels, kernel, kernel],
        )?;
        let bias = match node.input.get(2) {
            Some(name) if !name.is_empty() => {
                let bias = self.initializer(name)?;
                Self::expect_dims(name, &bias, &[out_channels])?;
                Some(bias)
            }
            _ => None,
        };
        Ok(Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                padding,
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        ))
    }

    /// The next `BatchNormalization`, folded to one multiply-add.
    pub fn next_batch_norm(&mut self, channels: usize) -> Result<FoldedBatchNorm> {
        let node = self.next_node("BatchNormalization")?;
        // ONNX defaults `epsilon` to 1e-5; both shipped graphs state it.
        let epsilon = Self::float_attr(node, "epsilon").unwrap_or(1e-5) as f64;
        let mut parts = Vec::with_capacity(4);
        for slot in 1..5 {
            let name = node
                .input
                .get(slot)
                .with_context(|| format!("BatchNormalization is missing input {slot}"))?;
            let tensor = self.initializer(name)?;
            Self::expect_dims(name, &tensor, &[channels])?;
            parts.push(tensor.to_vec1::<f32>()?);
        }
        let (scale, bias, mean, var) = (&parts[0], &parts[1], &parts[2], &parts[3]);
        let mut w = Vec::with_capacity(channels);
        let mut b = Vec::with_capacity(channels);
        for c in 0..channels {
            let inv = (var[c] as f64 + epsilon).sqrt();
            if inv <= 0.0 {
                bail!("BatchNormalization channel {c} has a non-positive variance");
            }
            let wc = scale[c] as f64 / inv;
            w.push(wc as f32);
            b.push((bias[c] as f64 - mean[c] as f64 * wc) as f32);
        }
        Ok(FoldedBatchNorm {
            weight: Tensor::from_vec(w, channels, &self.device)?,
            bias: Tensor::from_vec(b, channels, &self.device)?,
            channels,
        })
    }

    /// The next `PRelu`.
    ///
    /// ONNX `PRelu`: `f(x) = slope * x` for `x < 0`, `x` otherwise, with
    /// `slope` unidirectionally broadcast. `glintr100` stores it as `[C, 1, 1]`,
    /// which `candle_nn::PReLU` reshapes to `[1, C, 1, 1]` against a rank-4
    /// input.
    pub fn next_prelu(&mut self, channels: usize) -> Result<PReLU> {
        let node = self.next_node("PRelu")?;
        let name = node.input.get(1).context("a PRelu with no slope")?.clone();
        let slope = self.initializer(&name)?;
        if slope.elem_count() != channels {
            bail!(
                "PRelu slope `{name}` holds {} values, expected {channels}",
                slope.elem_count()
            );
        }
        Ok(PReLU::new(slope, false))
    }

    /// The next `Mul` by a scalar initializer.
    ///
    /// SCRFD's regression head is scaled by a learned per-stride scalar
    /// (`bbox_head.scales.N.scale`); ONNX `Mul` broadcasts it over the whole
    /// tensor.
    pub fn next_scalar_mul(&mut self) -> Result<f32> {
        let node = self.next_node("Mul")?;
        let name = node.input.get(1).context("a Mul with no operand")?.clone();
        let tensor = self.initializer(&name)?;
        if tensor.elem_count() != 1 {
            bail!(
                "Mul operand `{name}` holds {} values, expected a scalar",
                tensor.elem_count()
            );
        }
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?[0])
    }

    /// The next `Gemm`, as `(weight, bias)` in `[out, in]` layout.
    ///
    /// ONNX `Gemm`: `Y = alpha * A' * B' + beta * C`. Only the
    /// `alpha = beta = 1, transA = 0, transB = 1` form both graphs use is
    /// accepted; anything else is refused rather than approximated.
    pub fn next_gemm(
        &mut self,
        out_features: usize,
        in_features: usize,
    ) -> Result<(Tensor, Tensor)> {
        let node = self.next_node("Gemm")?;
        let alpha = Self::float_attr(node, "alpha").unwrap_or(1.0);
        let beta = Self::float_attr(node, "beta").unwrap_or(1.0);
        if alpha != 1.0 || beta != 1.0 {
            bail!("Gemm with alpha={alpha}, beta={beta} is unsupported");
        }
        if Self::int_attr(node, "transA").unwrap_or(0) != 0 {
            bail!("Gemm with transA=1 is unsupported");
        }
        if Self::int_attr(node, "transB").unwrap_or(0) != 1 {
            bail!("Gemm with transB=0 is unsupported");
        }
        let weight_name = node.input.get(1).context("a Gemm with no weight")?.clone();
        let weight = self.initializer(&weight_name)?;
        Self::expect_dims(&weight_name, &weight, &[out_features, in_features])?;
        let bias_name = node.input.get(2).context("a Gemm with no bias")?.clone();
        let bias = self.initializer(&bias_name)?;
        Self::expect_dims(&bias_name, &bias, &[out_features])?;
        Ok((weight, bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto};

    fn float_initializer(name: &str, dims: Vec<i64>, values: Vec<f32>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims,
            data_type: tensor_proto::DataType::Float as i32,
            float_data: values,
            ..Default::default()
        }
    }

    fn ints(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: candle_onnx::onnx::attribute_proto::AttributeType::Ints as i32,
            ints: values,
            ..Default::default()
        }
    }

    fn node(op: &str, inputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec!["out".to_string()],
            attribute: attrs,
            ..Default::default()
        }
    }

    fn model(nodes: Vec<NodeProto>, initializers: Vec<TensorProto>) -> ModelProto {
        ModelProto {
            graph: Some(GraphProto {
                node: nodes,
                initializer: initializers,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn a_conv_is_read_with_its_padding_and_stride() {
        let m = model(
            vec![node(
                "Conv",
                &["x", "w", "b"],
                vec![
                    ints("kernel_shape", vec![3, 3]),
                    ints("strides", vec![2, 2]),
                    ints("pads", vec![1, 1, 1, 1]),
                    ints("dilations", vec![1, 1]),
                ],
            )],
            vec![
                float_initializer("w", vec![2, 1, 3, 3], vec![0.5; 18]),
                float_initializer("b", vec![2], vec![1.0, -1.0]),
            ],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let conv = tape.next_conv(2, 1, 3, 2).unwrap();
        assert_eq!(conv.config().padding, 1);
        assert_eq!(conv.config().stride, 2);
        assert_eq!(conv.weight().dims(), &[2, 1, 3, 3]);
        tape.finish().unwrap();
    }

    #[test]
    fn a_shape_mismatch_names_the_initializer_rather_than_panicking_later() {
        let m = model(
            vec![node(
                "Conv",
                &["x", "w"],
                vec![ints("kernel_shape", vec![3, 3])],
            )],
            vec![float_initializer("w", vec![2, 1, 3, 3], vec![0.5; 18])],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let err = tape.next_conv(4, 1, 3, 1).unwrap_err();
        assert!(format!("{err}").contains("shape [2, 1, 3, 3]"), "{err}");
    }

    #[test]
    fn a_reordered_parameterized_op_is_refused_rather_than_consumed() {
        let m = model(
            vec![
                node("Relu", &["x"], vec![]),
                node("PRelu", &["x", "s"], vec![]),
                node("Conv", &["x", "w"], vec![ints("kernel_shape", vec![1, 1])]),
            ],
            vec![
                float_initializer("s", vec![2, 1, 1], vec![0.25, 0.25]),
                float_initializer("w", vec![2, 2, 1, 1], vec![1.0; 4]),
            ],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let err = tape.next_conv(2, 2, 1, 1).unwrap_err();
        assert!(format!("{err}").contains("found `PRelu`"), "{err}");
    }

    #[test]
    fn an_unconsumed_parameterized_op_is_refused_at_finish() {
        let m = model(
            vec![
                node("Conv", &["x", "w"], vec![ints("kernel_shape", vec![1, 1])]),
                node("PRelu", &["x", "s"], vec![]),
            ],
            vec![
                float_initializer("w", vec![2, 2, 1, 1], vec![1.0; 4]),
                float_initializer("s", vec![2, 1, 1], vec![0.25, 0.25]),
            ],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        tape.next_conv(2, 2, 1, 1).unwrap();
        let err = tape.finish().unwrap_err();
        assert!(format!("{err}").contains("never consumed"), "{err}");
    }

    /// The fold must be the ONNX formula, not an approximation of it.
    #[test]
    fn the_batch_norm_fold_reproduces_the_spec_formula() {
        let epsilon = 1e-5f32;
        let mut eps_attr = AttributeProto {
            name: "epsilon".to_string(),
            r#type: candle_onnx::onnx::attribute_proto::AttributeType::Float as i32,
            ..Default::default()
        };
        eps_attr.f = epsilon;
        let m = model(
            vec![node(
                "BatchNormalization",
                &["x", "scale", "b", "mean", "var"],
                vec![eps_attr],
            )],
            vec![
                float_initializer("scale", vec![2], vec![2.0, 0.5]),
                float_initializer("b", vec![2], vec![-1.0, 3.0]),
                float_initializer("mean", vec![2], vec![0.25, -2.0]),
                float_initializer("var", vec![2], vec![4.0, 9.0]),
            ],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let bn = tape.next_batch_norm(2).unwrap();
        assert_eq!(bn.channels(), 2);
        let xs = [1.0f32, -3.0];
        let got = bn
            .forward(&Tensor::from_vec(xs.to_vec(), (1, 2), &Device::Cpu).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let (scale, bias, mean, var) = (
            [2.0f32, 0.5],
            [-1.0f32, 3.0],
            [0.25f32, -2.0],
            [4.0f32, 9.0],
        );
        for c in 0..2 {
            let expected = (xs[c] - mean[c]) / (var[c] + epsilon).sqrt() * scale[c] + bias[c];
            assert!(
                (got[0][c] - expected).abs() < 1e-6,
                "channel {c}: {} vs {expected}",
                got[0][c]
            );
        }
    }

    #[test]
    fn a_non_float_initializer_is_refused_rather_than_reinterpreted() {
        let m = model(
            vec![node(
                "Conv",
                &["x", "w"],
                vec![ints("kernel_shape", vec![1, 1])],
            )],
            vec![TensorProto {
                name: "w".to_string(),
                dims: vec![1, 1, 1, 1],
                data_type: tensor_proto::DataType::Int64 as i32,
                int64_data: vec![7],
                ..Default::default()
            }],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let err = tape.next_conv(1, 1, 1, 1).unwrap_err();
        assert!(format!("{err}").contains("expected FLOAT"), "{err}");
    }

    #[test]
    fn raw_little_endian_data_and_float_data_agree() {
        let values = [1.5f32, -2.25, 0.0, 7.75];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = model(
            vec![node(
                "Conv",
                &["x", "w"],
                vec![ints("kernel_shape", vec![1, 1])],
            )],
            vec![TensorProto {
                name: "w".to_string(),
                dims: vec![4, 1, 1, 1],
                data_type: tensor_proto::DataType::Float as i32,
                raw_data: raw,
                ..Default::default()
            }],
        );
        let mut tape = WeightTape::new(&m, &Device::Cpu).unwrap();
        let conv = tape.next_conv(4, 1, 1, 1).unwrap();
        assert_eq!(
            conv.weight()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            values.to_vec()
        );
    }
}
