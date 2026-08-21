//! Step 0 (#1222): does `candle-onnx` actually implement the two InsightFace
//! graphs mold needs, attribute for attribute?
//!
//! This is the decision procedure, not a diagnostic. `candle_onnx::simple_eval`
//! is a `match` over op-type strings
//! (`candle-onnx/src/eval.rs:335-2560`); an op it does not name is a runtime
//! `bail!`, and — worse — several arms *silently ignore* attributes they do not
//! honour. Both failure modes have to be caught before mold builds an engine on
//! top of the evaluator, so the inventory is split into two verdicts:
//!
//! * [`Unsupported`] — the graph would fail, or would silently compute the
//!   wrong thing for some inputs. A single one of these fails the gate.
//! * [`IgnoredAttribute`] — the arm drops an attribute whose effect is
//!   provably nil at the input shape mold pins. Recorded, never assumed:
//!   [`AveragePool`'s `ceil_mode`](Ignored::AveragePoolCeilMode) is the real
//!   case, and the pinned 640x640 SCRFD input is what makes it harmless.
//!
//! The supported-op list below is transcribed from the fork's
//! `candle-onnx/src/eval.rs` at the `fix/mold-compat-0.11` merge base
//! (`candle-onnx` there is byte-identical to upstream 0.11.0 — the last commit
//! touching the crate is the 0.11.0 version bump). It is a hand-maintained
//! mirror on purpose: mold cannot introspect another crate's `match`, so the
//! *inventory* is derived from the pinned model bytes and only the *capability
//! set* is transcribed. Both halves are frozen in `crates/mold-inference/testdata/pulid/` so a
//! candle bump that changes either is a failing test rather than a surprise.

use std::collections::{BTreeMap, BTreeSet};

use candle_onnx::onnx::{AttributeProto, ModelProto};
use serde::{Deserialize, Serialize};

/// Every op `candle_onnx::simple_eval` has a `match` arm for.
///
/// Source: `candle-onnx/src/eval.rs`, the `node.op_type.as_str()` match
/// (`:335` through `:2560`). Kept sorted so a diff against a newer candle is
/// readable.
pub const CANDLE_ONNX_SUPPORTED_OPS: &[&str] = &[
    "Abs",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "Cos",
    "CumSum",
    "Div",
    "Dropout",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "Gather",
    "GatherElements",
    "Gelu",
    "Gemm",
    "Greater",
    "GreaterOrEqual",
    "HardSwish",
    "Identity",
    "If",
    "LSTM",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "MatMul",
    "MaxPool",
    "Min",
    "Mul",
    "Neg",
    "Not",
    "OneHot",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "RNN",
    "Range",
    "ReduceL2",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Resize",
    "ScatterND",
    "Selu",
    "Shape",
    "Sigmoid",
    "Sign",
    "Sin",
    "Size",
    "Slice",
    "Softmax",
    "Split",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Tanh",
    "Tile",
    "Transpose",
    "Trilu",
    "Unsqueeze",
    "Where",
    "Xor",
];

/// One attribute of one node, reduced to the shape the gate reasons about.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AttributeValue {
    /// Attribute name, e.g. `mode`.
    pub name: String,
    /// A canonical rendering of the value: an int, a float, a string, or a
    /// comma-joined list. Tensor/graph attributes render as their kind alone,
    /// since no op mold uses carries one.
    pub value: String,
}

/// A distinct `(op_type, attribute name+value set)` combination in a graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpSignature {
    /// ONNX op type, e.g. `Resize`.
    pub op_type: String,
    /// The node's attributes, sorted by name.
    pub attributes: Vec<AttributeValue>,
    /// How many nodes in the graph share this exact signature.
    pub count: usize,
}

/// The full op/attribute inventory of one ONNX graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphInventory {
    /// Opset versions declared by the model, `domain -> version`.
    pub opset: BTreeMap<String, i64>,
    /// Graph input names with their declared dimensions (`-1` for symbolic).
    pub inputs: Vec<TensorSignature>,
    /// Graph output names with their declared dimensions.
    pub outputs: Vec<TensorSignature>,
    /// Every distinct op signature, sorted by op type then attributes.
    pub signatures: Vec<OpSignature>,
}

/// A graph input or output as the model declares it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSignature {
    /// Value-info name.
    pub name: String,
    /// Static dimensions; `-1` stands in for any symbolic dimension.
    pub dims: Vec<i64>,
}

/// A reason the evaluator cannot run a graph correctly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unsupported {
    /// `simple_eval` has no arm for this op at all.
    MissingOp {
        /// The op type.
        op_type: String,
    },
    /// The arm exists but rejects (or would mis-evaluate) this attribute value.
    RestrictedAttribute {
        /// The op type.
        op_type: String,
        /// The offending attribute.
        attribute: String,
        /// The value the graph carries.
        value: String,
        /// What the evaluator accepts, and where that is enforced.
        accepted: String,
    },
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOp { op_type } => {
                write!(f, "candle-onnx has no evaluator arm for `{op_type}`")
            }
            Self::RestrictedAttribute {
                op_type,
                attribute,
                value,
                accepted,
            } => write!(
                f,
                "candle-onnx's `{op_type}` rejects {attribute}={value} ({accepted})"
            ),
        }
    }
}

/// An attribute the evaluator drops on the floor.
///
/// Never a pass by itself: each variant carries the exact precondition that
/// makes dropping it harmless, and [`ignored_attributes_are_harmless`] is what
/// checks the precondition holds for the shape mold pins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IgnoredAttribute {
    /// The op type whose arm ignores it.
    pub op_type: String,
    /// The attribute name.
    pub attribute: String,
    /// The value the graph declares.
    pub value: String,
    /// Why mold accepts the omission anyway.
    pub harmless_because: String,
}

fn render_attribute(attr: &AttributeProto) -> AttributeValue {
    // `AttributeProto.type` (field 20) is the discriminator. Values from
    // `candle-onnx/src/onnx.proto3:124-141`.
    let value = match attr.r#type {
        1 => format!("{}", attr.f),
        2 => format!("{}", attr.i),
        3 => String::from_utf8_lossy(&attr.s).to_string(),
        4 => "<tensor>".to_string(),
        5 => "<graph>".to_string(),
        6 => attr
            .floats
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        7 => attr
            .ints
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        8 => attr
            .strings
            .iter()
            .map(|v| String::from_utf8_lossy(v).to_string())
            .collect::<Vec<_>>()
            .join(","),
        other => format!("<attribute-type-{other}>"),
    };
    AttributeValue {
        name: attr.name.clone(),
        value,
    }
}

fn tensor_signatures(values: &[candle_onnx::onnx::ValueInfoProto]) -> Vec<TensorSignature> {
    values
        .iter()
        .map(|value| {
            let dims = value
                .r#type
                .as_ref()
                .and_then(|t| t.value.as_ref())
                .and_then(|v| match v {
                    candle_onnx::onnx::type_proto::Value::TensorType(tt) => tt.shape.as_ref(),
                    _ => None,
                })
                .map(|shape| {
                    shape
                        .dim
                        .iter()
                        .map(|d| match d.value.as_ref() {
                            Some(
                                candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                    v,
                                ),
                            ) => *v,
                            _ => -1,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            TensorSignature {
                name: value.name.clone(),
                dims,
            }
        })
        .collect()
}

/// Build the op/attribute inventory of a decoded model.
pub fn graph_inventory(model: &ModelProto) -> anyhow::Result<GraphInventory> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("the ONNX model carries no graph"))?;
    let mut grouped: BTreeMap<(String, Vec<AttributeValue>), usize> = BTreeMap::new();
    for node in &graph.node {
        let mut attributes: Vec<AttributeValue> =
            node.attribute.iter().map(render_attribute).collect();
        attributes.sort_by(|a, b| a.name.cmp(&b.name));
        *grouped
            .entry((node.op_type.clone(), attributes))
            .or_insert(0) += 1;
    }
    let signatures = grouped
        .into_iter()
        .map(|((op_type, attributes), count)| OpSignature {
            op_type,
            attributes,
            count,
        })
        .collect();
    let opset = model
        .opset_import
        .iter()
        .map(|o| (o.domain.clone(), o.version))
        .collect();
    Ok(GraphInventory {
        opset,
        inputs: tensor_signatures(&graph.input),
        outputs: tensor_signatures(&graph.output),
        signatures,
    })
}

fn attr<'a>(sig: &'a OpSignature, name: &str) -> Option<&'a str> {
    sig.attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.value.as_str())
}

fn restricted(sig: &OpSignature, attribute: &str, value: &str, accepted: &str) -> Unsupported {
    Unsupported::RestrictedAttribute {
        op_type: sig.op_type.clone(),
        attribute: attribute.to_string(),
        value: value.to_string(),
        accepted: accepted.to_string(),
    }
}

/// Everything about this inventory that `candle_onnx::simple_eval` cannot run.
///
/// Empty is the Step-0 op gate passing.
pub fn unsupported_by_candle_onnx(inventory: &GraphInventory) -> Vec<Unsupported> {
    let supported: BTreeSet<&str> = CANDLE_ONNX_SUPPORTED_OPS.iter().copied().collect();
    let mut out = Vec::new();
    for sig in &inventory.signatures {
        if !supported.contains(sig.op_type.as_str()) {
            out.push(Unsupported::MissingOp {
                op_type: sig.op_type.clone(),
            });
            continue;
        }
        // `auto_pad` is rejected by every pooling and conv arm
        // (`eval.rs:463-467`, `:498-502`, `:861-865`).
        if matches!(sig.op_type.as_str(), "Conv" | "MaxPool" | "AveragePool") {
            if let Some(pad) = attr(sig, "auto_pad") {
                if pad != "NOTSET" {
                    out.push(restricted(
                        sig,
                        "auto_pad",
                        pad,
                        "only NOTSET / absent (eval.rs:463-467, :498-502, :861-865)",
                    ));
                }
            }
        }
        match sig.op_type.as_str() {
            // `eval.rs:2325-2344`: nearest + floor + asymmetric, rank 4 only.
            "Resize" => {
                if let Some(mode) = attr(sig, "mode") {
                    if mode != "nearest" {
                        out.push(restricted(
                            sig,
                            "mode",
                            mode,
                            "only `nearest` (eval.rs:2325)",
                        ));
                    }
                }
                let nearest = attr(sig, "nearest_mode").unwrap_or("round_prefer_floor");
                if nearest != "floor" {
                    out.push(restricted(
                        sig,
                        "nearest_mode",
                        nearest,
                        "only `floor` (eval.rs:2329)",
                    ));
                }
                let coord = attr(sig, "coordinate_transformation_mode").unwrap_or("half_pixel");
                if coord != "asymmetric" {
                    out.push(restricted(
                        sig,
                        "coordinate_transformation_mode",
                        coord,
                        "only `asymmetric` (eval.rs:2333)",
                    ));
                }
            }
            // `eval.rs:472-476`, `:507-511`: any non-zero pad is a bail.
            "MaxPool" | "AveragePool" => {
                if let Some(pads) = attr(sig, "pads") {
                    if pads.split(',').any(|p| p.trim() != "0") {
                        out.push(restricted(
                            sig,
                            "pads",
                            pads,
                            "only all-zero pads (eval.rs:472-476, :507-511)",
                        ));
                    }
                }
                if let Some(d) = attr(sig, "dilations") {
                    if d.split(',').any(|v| v.trim() != "1") {
                        out.push(restricted(
                            sig,
                            "dilations",
                            d,
                            "only all-one dilations (eval.rs:468-471, :503-506)",
                        ));
                    }
                }
                if let Some(k) = attr(sig, "kernel_shape") {
                    if k.split(',').count() != 2 {
                        out.push(restricted(sig, "kernel_shape", k, "2-D pooling only"));
                    }
                }
            }
            // `eval.rs:920-949`: 2-D conv strides and dilations must be equal
            // on both axes.
            "Conv" => {
                for name in ["strides", "dilations"] {
                    if let Some(v) = attr(sig, name) {
                        let parts: Vec<&str> = v.split(',').map(str::trim).collect();
                        if parts.len() == 2 && parts[0] != parts[1] {
                            out.push(restricted(
                                sig,
                                name,
                                v,
                                "conv2d requires the same value on both axes (eval.rs:920-949)",
                            ));
                        }
                    }
                }
            }
            // `eval.rs:527-530`: training mode is a bail.
            "BatchNormalization" => {
                if let Some(mode) = attr(sig, "training_mode") {
                    if mode != "0" {
                        out.push(restricted(
                            sig,
                            "training_mode",
                            mode,
                            "inference only (eval.rs:527-530)",
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Attributes the evaluator drops, with the precondition that makes each drop
/// harmless.
///
/// `AveragePool` is the only real case in either antelopev2 graph:
/// `eval.rs:491-525` reads `kernel_shape`, `pads`, `strides`, `dilations`, and
/// `auto_pad`, and never looks at `ceil_mode`. `Tensor::avg_pool2d_with_stride`
/// floors. Flooring and ceiling agree exactly when every pooled extent is
/// divisible by the stride, which is what [`pinned_input_makes_pooling_exact`]
/// pins for mold's 640x640 SCRFD input.
pub fn ignored_attributes(inventory: &GraphInventory) -> Vec<IgnoredAttribute> {
    let mut out = Vec::new();
    for sig in &inventory.signatures {
        if sig.op_type == "AveragePool" {
            if let Some(mode) = attr(sig, "ceil_mode") {
                if mode != "0" {
                    out.push(IgnoredAttribute {
                        op_type: sig.op_type.clone(),
                        attribute: "ceil_mode".to_string(),
                        value: mode.to_string(),
                        harmless_because:
                            "candle-onnx's AveragePool floors (eval.rs:491-525); ceil and floor \
                             agree because every SCRFD feature map is a power-of-two divisor of \
                             the pinned 640x640 input, so no 2x2/stride-2 window is ever ragged"
                                .to_string(),
                    });
                }
            }
        }
        // MaxPool's arm ignores `ceil_mode` identically (`eval.rs:456-489`).
        if sig.op_type == "MaxPool" {
            if let Some(mode) = attr(sig, "ceil_mode") {
                if mode != "0" {
                    out.push(IgnoredAttribute {
                        op_type: sig.op_type.clone(),
                        attribute: "ceil_mode".to_string(),
                        value: mode.to_string(),
                        harmless_because:
                            "candle-onnx's MaxPool floors (eval.rs:456-489); see AveragePool"
                                .to_string(),
                    });
                }
            }
        }
    }
    out
}

/// The precondition behind every [`IgnoredAttribute`] this module tolerates:
/// at `input` pixels square, each 2x2/stride-2 pooling stage in SCRFD's
/// backbone consumes an even extent, so `ceil_mode` cannot change a shape.
///
/// SCRFD's stem halves twice (conv stride 2, then MaxPool stride 2) and the
/// FPN pools three more times, so the smallest pooled extent is `input / 32`.
pub fn pinned_input_makes_pooling_exact(input: usize) -> bool {
    input.is_multiple_of(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(op: &str, attrs: &[(&str, &str)]) -> OpSignature {
        OpSignature {
            op_type: op.to_string(),
            attributes: attrs
                .iter()
                .map(|(n, v)| AttributeValue {
                    name: n.to_string(),
                    value: v.to_string(),
                })
                .collect(),
            count: 1,
        }
    }

    fn inventory(signatures: Vec<OpSignature>) -> GraphInventory {
        GraphInventory {
            opset: BTreeMap::from([(String::new(), 11)]),
            inputs: vec![],
            outputs: vec![],
            signatures,
        }
    }

    #[test]
    fn an_op_candle_does_not_implement_fails_the_gate() {
        let found = unsupported_by_candle_onnx(&inventory(vec![sig("NonMaxSuppression", &[])]));
        assert_eq!(
            found,
            vec![Unsupported::MissingOp {
                op_type: "NonMaxSuppression".to_string()
            }]
        );
    }

    #[test]
    fn a_bilinear_resize_fails_the_gate() {
        let found = unsupported_by_candle_onnx(&inventory(vec![sig(
            "Resize",
            &[
                ("mode", "linear"),
                ("nearest_mode", "floor"),
                ("coordinate_transformation_mode", "asymmetric"),
            ],
        )]));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(format!("{}", found[0]).contains("mode=linear"));
    }

    #[test]
    fn a_half_pixel_resize_fails_the_gate() {
        let found = unsupported_by_candle_onnx(&inventory(vec![sig(
            "Resize",
            &[("mode", "nearest"), ("nearest_mode", "floor")],
        )]));
        // `coordinate_transformation_mode` defaults to `half_pixel`, which the
        // evaluator refuses — an absent attribute is not a pass.
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(format!("{}", found[0]).contains("coordinate_transformation_mode"));
    }

    #[test]
    fn a_padded_pool_fails_the_gate() {
        let found = unsupported_by_candle_onnx(&inventory(vec![sig(
            "MaxPool",
            &[("kernel_shape", "3,3"), ("pads", "1,1,1,1")],
        )]));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(format!("{}", found[0]).contains("pads"));
    }

    #[test]
    fn an_auto_pad_conv_fails_the_gate() {
        let found = unsupported_by_candle_onnx(&inventory(vec![sig(
            "Conv",
            &[("auto_pad", "SAME_UPPER")],
        )]));
        assert_eq!(found.len(), 1, "{found:?}");
    }

    #[test]
    fn asymmetric_conv_strides_fail_the_gate() {
        let found =
            unsupported_by_candle_onnx(&inventory(vec![sig("Conv", &[("strides", "2,1")])]));
        assert_eq!(found.len(), 1, "{found:?}");
    }

    #[test]
    fn ceil_mode_pooling_is_recorded_as_ignored_not_as_supported() {
        let inv = inventory(vec![sig(
            "AveragePool",
            &[
                ("ceil_mode", "1"),
                ("kernel_shape", "2,2"),
                ("pads", "0,0,0,0"),
                ("strides", "2,2"),
            ],
        )]);
        assert!(
            unsupported_by_candle_onnx(&inv).is_empty(),
            "ceil_mode is not a hard refusal"
        );
        let ignored = ignored_attributes(&inv);
        assert_eq!(ignored.len(), 1);
        assert_eq!(ignored[0].attribute, "ceil_mode");
        assert!(ignored[0].harmless_because.contains("floor"));
    }

    #[test]
    fn the_pooling_precondition_is_the_pinned_detector_input() {
        assert!(pinned_input_makes_pooling_exact(640));
        assert!(!pinned_input_makes_pooling_exact(650));
        // 128 is insightface's other default detector size and is also exact.
        assert!(pinned_input_makes_pooling_exact(128));
    }

    /// The transcribed capability set is load-bearing; a typo that dropped
    /// `Conv` would make the gate pass for the wrong reason.
    #[test]
    fn the_transcribed_op_set_is_sorted_and_unique() {
        let mut sorted = CANDLE_ONNX_SUPPORTED_OPS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), CANDLE_ONNX_SUPPORTED_OPS);
    }
}
