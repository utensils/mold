//! Descriptor-fenced loading of an ONNX `ModelProto`.
//!
//! `candle_onnx::read_file` takes a path and calls `std::fs::read`, which is
//! exactly the race mold's model-storage rules forbid: the path can be
//! replaced between the check and the read. This module opens the file with
//! [`mold_core::secure_file::open_regular_file_no_follow`] — no symlink in the
//! filename or any parent component, regular files only, canonical
//! traversal — and decodes the retained descriptor's bytes. The digest is
//! taken from that same descriptor, so the bytes hashed are the bytes parsed.
//!
//! Permissions are deliberately NOT checked: a model artifact on shared
//! storage with a collaborative umask is valid (see CLAUDE.md, "Model storage
//! permissions invariant"). Authenticity comes from the content digest.

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use candle_onnx::onnx::ModelProto;
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use prost::Message;

/// One ONNX graph plus the digest of the exact bytes it was decoded from.
#[derive(Debug, Clone)]
pub struct LoadedOnnxModel {
    /// The decoded graph.
    pub model: ModelProto,
    /// Lowercase hex SHA-256 of the file the graph came from.
    pub sha256: String,
    /// Encoded size in bytes.
    pub bytes: usize,
}

/// Rewrite `Resize` inputs that name a **zero-element** initializer into the
/// empty-string form that means "not provided".
///
/// This is a compatibility shim for a real `candle-onnx` defect, and it is
/// deliberately visible rather than buried.
///
/// ONNX's `Resize` (opset 11) declares `roi`, `scales`, and `sizes` optional,
/// and the spec's own note is that "one of `scales` and `sizes` MUST be
/// specified... if `sizes` is needed, set `scales` to an empty tensor". PyTorch
/// and MXNet exporters both take that route, and `scrfd_10g_bnkps.onnx` is one
/// of them: its two `Resize` nodes read `['382', '392', '392', '399']`, where
/// `392` is a single zero-length `float` initializer standing in for BOTH the
/// absent `roi` and the absent `scales`.
///
/// `candle-onnx` tests presence on the input **name**
/// (`candle-onnx/src/eval.rs:2291-2301`: `!node.input[2].is_empty()` is a
/// string check), never on the tensor, so it reads that empty tensor as a
/// supplied `scales`, finds `sizes` supplied as well, and bails with "Scales
/// and sizes cannot both be set for Resize operation" (`:2312-2314`). The
/// upstream fix is one condition — treat a zero-element tensor as absent — and
/// belongs in a candle PR, not smuggled into mold. Until that lands, mold
/// normalizes the graph it was handed.
///
/// Scope is narrow on purpose: only `Resize`, and only its optional
/// `roi`/`scales`/`sizes` slots, where the ONNX spec explicitly gives an empty
/// tensor the meaning "unspecified". A blanket rewrite of every zero-element
/// input would be wrong — plenty of ops take a legitimately empty tensor.
///
/// Returns the number of inputs rewritten.
pub fn normalize_empty_optional_resize_inputs(model: &mut ModelProto) -> usize {
    let Some(graph) = model.graph.as_mut() else {
        return 0;
    };
    let empty: HashSet<&str> = graph
        .initializer
        .iter()
        .filter(|t| t.dims.iter().product::<i64>() == 0 || t.dims == [0])
        .map(|t| t.name.as_str())
        .collect();
    let empty: HashSet<String> = empty.into_iter().map(str::to_string).collect();
    let mut rewritten = 0;
    for node in graph.node.iter_mut() {
        if node.op_type != "Resize" {
            continue;
        }
        // 0 is the required data input; 1..=3 are roi, scales, sizes.
        for slot in 1..node.input.len().min(4) {
            if empty.contains(&node.input[slot]) {
                node.input[slot].clear();
                rewritten += 1;
            }
        }
    }
    rewritten
}

/// Open, hash, and decode an ONNX model without ever re-opening its path.
///
/// The decoded graph is normalized by
/// [`normalize_empty_optional_resize_inputs`] before it is returned, so every
/// caller evaluates the same shape.
pub fn load_onnx_model(path: &Path) -> Result<LoadedOnnxModel> {
    let mut file = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open the ONNX model at {}", path.display()))?;
    let sha256 = sha256_open_file(&file)
        .with_context(|| format!("failed to hash the ONNX model at {}", path.display()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .with_context(|| format!("failed to read the ONNX model at {}", path.display()))?;
    let bytes = buf.len();
    let mut model = ModelProto::decode(buf.as_slice())
        .with_context(|| format!("failed to decode the ONNX model at {}", path.display()))?;
    normalize_empty_optional_resize_inputs(&mut model);
    Ok(LoadedOnnxModel {
        model,
        sha256,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_onnx::onnx::{GraphProto, NodeProto, TensorProto};

    fn tensor(name: &str, dims: Vec<i64>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims,
            data_type: 1,
            ..Default::default()
        }
    }

    fn resize(inputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: "Resize".to_string(),
            name: "Resize_0".to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec!["out".to_string()],
            ..Default::default()
        }
    }

    fn model_with(nodes: Vec<NodeProto>, initializers: Vec<TensorProto>) -> ModelProto {
        ModelProto {
            graph: Some(GraphProto {
                node: nodes,
                initializer: initializers,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// The exact shape `scrfd_10g_bnkps.onnx` ships: one zero-length
    /// initializer reused for both the absent `roi` and the absent `scales`.
    #[test]
    fn an_empty_scales_initializer_is_rewritten_to_absent() {
        let mut model = model_with(
            vec![resize(&["data", "392", "392", "sizes"])],
            vec![tensor("392", vec![0])],
        );
        assert_eq!(normalize_empty_optional_resize_inputs(&mut model), 2);
        let node = &model.graph.as_ref().unwrap().node[0];
        assert_eq!(node.input[0], "data");
        assert_eq!(node.input[1], "");
        assert_eq!(node.input[2], "");
        assert_eq!(node.input[3], "sizes");
    }

    #[test]
    fn a_populated_scales_input_is_left_alone() {
        let mut model = model_with(
            vec![resize(&["data", "", "scales"])],
            vec![tensor("scales", vec![4])],
        );
        assert_eq!(normalize_empty_optional_resize_inputs(&mut model), 0);
        assert_eq!(model.graph.as_ref().unwrap().node[0].input[2], "scales");
    }

    /// The data input is required; an empty tensor there is a broken graph and
    /// must not be silently turned into "absent".
    #[test]
    fn the_required_data_input_is_never_rewritten() {
        let mut model = model_with(
            vec![resize(&["392", "", "", "sizes"])],
            vec![tensor("392", vec![0])],
        );
        assert_eq!(normalize_empty_optional_resize_inputs(&mut model), 0);
        assert_eq!(model.graph.as_ref().unwrap().node[0].input[0], "392");
    }

    /// Narrow by design: an empty tensor is meaningful to other ops.
    #[test]
    fn non_resize_nodes_are_untouched() {
        let mut node = resize(&["data", "392"]);
        node.op_type = "Slice".to_string();
        let mut model = model_with(vec![node], vec![tensor("392", vec![0])]);
        assert_eq!(normalize_empty_optional_resize_inputs(&mut model), 0);
        assert_eq!(model.graph.as_ref().unwrap().node[0].input[1], "392");
    }

    #[test]
    fn a_multi_axis_zero_tensor_also_counts_as_empty() {
        let mut model = model_with(
            vec![resize(&["data", "", "roi0", "sizes"])],
            vec![tensor("roi0", vec![2, 0])],
        );
        assert_eq!(normalize_empty_optional_resize_inputs(&mut model), 1);
    }

    #[test]
    fn a_non_onnx_file_is_a_decode_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.onnx");
        std::fs::write(&path, b"this is not a protobuf at all, not even close").unwrap();
        let err = load_onnx_model(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("failed to decode"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn a_missing_model_names_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.onnx");
        let err = load_onnx_model(&path).unwrap_err();
        assert!(format!("{err:#}").contains("absent.onnx"), "{err:#}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_model_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.onnx");
        std::fs::write(&real, b"bytes").unwrap();
        let link = dir.path().join("link.onnx");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(
            load_onnx_model(&link).is_err(),
            "a symlinked model must not be opened"
        );
    }
}
