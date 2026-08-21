//! Descriptor-fenced loading of an ONNX `ModelProto`.
//!
//! `candle_onnx::read_file` takes a path and calls `std::fs::read`, which is
//! exactly the race mold's model-storage rules forbid: the path can be
//! replaced between the check and the read. This module opens the file with
//! [`mold_core::secure_file::open_regular_file_no_follow`] — no symlink in the
//! filename or any parent component, regular files only, canonical
//! traversal — and decodes the retained descriptor's bytes.
//!
//! Retaining the descriptor is necessary but not sufficient. The bytes must
//! also be read from it exactly **once**: hashing the descriptor and then
//! reading it is two passes, and on shared storage an in-place write between
//! them authenticates one set of bytes and executes another. [`AuthenticatedBytes`]
//! is the whole answer — it performs the single read, hashes what it read, and
//! offers no way to obtain a digest and a buffer that did not come from the
//! same call.
//!
//! Permissions are deliberately NOT checked: a model artifact on shared
//! storage with a collaborative umask is valid (see CLAUDE.md, "Model storage
//! permissions invariant"). Authenticity comes from the content digest.

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use candle_onnx::onnx::ModelProto;
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::pulid_manifest;
use mold_core::secure_file::open_regular_file_no_follow;
use prost::Message;
use sha2::{Digest, Sha256};

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

/// The bytes on disk are not the bytes the manifest pinned.
#[derive(Debug, thiserror::Error)]
#[error(
    "{path} does not match the pinned SHA-256 for this PuLID asset\n  expected {expected}\n  found    {found}\nre-pull the bundle: mold pull pulid-flux"
)]
pub struct DigestMismatch {
    /// The file that failed.
    pub path: String,
    /// The manifest's pin.
    pub expected: String,
    /// What the retained descriptor actually hashed to.
    pub found: String,
}

/// The manifest's SHA-256 pin for one PuLID component.
///
/// Read from [`mold_core::pulid_assets::pulid_manifest`] rather than copied,
/// so there is exactly one place a pin lives. `None` for a component the
/// manifest does not pin, which for this bundle cannot happen — a completeness
/// test in `manifest.rs` requires all four — but is not worth a panic.
pub fn pinned_sha256(component: ModelComponent) -> Option<&'static str> {
    pulid_manifest()
        .files
        .iter()
        .find(|file| file.component == component)
        .and_then(|file| file.sha256)
}

/// Open, hash, verify, and decode an ONNX model without ever re-opening its
/// path.
///
/// `expected_sha256` is checked **before** the proto is decoded, against the
/// digest of the same retained descriptor the bytes are read from — so a file
/// swapped between the check and the read cannot be the file that runs.
///
/// Verifying here rather than trusting the download is the point. The
/// downloader's `.sha256-verified` marker records that the bytes were correct
/// *when they landed*; it says nothing about the bytes now, and a marker file
/// sitting beside a since-modified model is exactly the state an attacker with
/// write access to the models directory would leave behind. These two graphs
/// are executed code in every sense that matters, so they are authenticated on
/// every load. Passing `None` skips the check and exists for tests and for the
/// probe binary, which inspect arbitrary graphs by design;
/// [`super::IdentityExtractor::load`] always supplies the manifest's pin.
///
/// This is deliberately NOT placement-time verification, and must not be
/// "simplified" into it. Placement-time pin checking (the download path, and
/// the per-file dependency verification landing in #1242) proves a file as it
/// is materialized and accepts an existing `.sha256-verified` marker without
/// rehashing — right for materialization, and precisely the assumption this
/// check exists to stop relying on. The two are complementary: one proves what
/// landed, the other proves what runs.
///
/// The decoded graph is normalized by
/// [`normalize_empty_optional_resize_inputs`] before it is returned, so every
/// caller evaluates the same shape.
pub fn load_onnx_model(path: &Path, expected_sha256: Option<&str>) -> Result<LoadedOnnxModel> {
    let mut file = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open the ONNX model at {}", path.display()))?;
    let authenticated = AuthenticatedBytes::read_once(&mut file, path)?;
    authenticated.verify(expected_sha256, path)?;
    let mut model = ModelProto::decode(authenticated.bytes())
        .with_context(|| format!("failed to decode the ONNX model at {}", path.display()))?;
    normalize_empty_optional_resize_inputs(&mut model);
    Ok(LoadedOnnxModel {
        bytes: authenticated.bytes().len(),
        sha256: authenticated.into_sha256(),
        model,
    })
}

/// One read of a retained descriptor, with the digest of exactly those bytes.
///
/// This type exists to make a specific bug unrepresentable. The obvious
/// loader — hash the descriptor, then read it — performs **two** reads, and on
/// shared storage an in-place write landing between them pairs an
/// authenticated digest with unauthenticated bytes: the check passes and
/// different bytes execute. Reading twice is not a slow version of reading
/// once; it is a different, wrong operation.
///
/// So there is no constructor that takes a digest, none that takes bytes, and
/// no accessor that yields one without the other having come from the same
/// [`read_once`](Self::read_once) call. A caller cannot hold a digest for one
/// read and a buffer from another, because it can never obtain them
/// separately. `mold_core::secure_file::sha256_open_file` is deliberately
/// unused here for the same reason — it takes a `&File` and hashes it, which
/// necessarily leaves the bytes to be fetched by a second read.
struct AuthenticatedBytes {
    bytes: Vec<u8>,
    sha256: String,
}

impl AuthenticatedBytes {
    /// Read the whole descriptor exactly once and hash what was read.
    ///
    /// The descriptor comes from `open_regular_file_no_follow`, so it is
    /// positioned at zero and no seek is needed — nor performed, which is what
    /// keeps this to a single pass over the bytes.
    fn read_once(file: &mut File, path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("failed to read the ONNX model at {}", path.display()))?;
        let digest = Sha256::digest(&bytes);
        Ok(Self {
            bytes,
            sha256: format!("{digest:x}"),
        })
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn into_sha256(self) -> String {
        self.sha256
    }

    /// Compare against a manifest pin, if one was supplied.
    fn verify(&self, expected: Option<&str>, path: &Path) -> Result<()> {
        let Some(expected) = expected else {
            return Ok(());
        };
        if self.sha256.eq_ignore_ascii_case(expected) {
            return Ok(());
        }
        Err(DigestMismatch {
            path: path.display().to_string(),
            expected: expected.to_string(),
            found: self.sha256.clone(),
        }
        .into())
    }
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

    /// A synthetic proto, written twice: once intact, once with one byte
    /// flipped. Only the intact one may be decoded.
    fn write_proto(
        dir: &std::path::Path,
        name: &str,
        flip_byte: bool,
    ) -> (std::path::PathBuf, String) {
        let model = model_with(
            vec![resize(&["data", "", "", "sizes"])],
            vec![tensor("sizes", vec![4])],
        );
        let mut bytes = model.encode_to_vec();
        if flip_byte {
            let last = bytes.len() - 1;
            bytes[last] ^= 0x01;
        }
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        (path, format!("{digest:x}"))
    }

    /// The digest describes the bytes that were read, and the read happened
    /// once. Cross-checked against `mold-core`'s own file hasher so this
    /// hand-rolled digest cannot quietly diverge from the rest of mold.
    #[test]
    fn the_digest_describes_the_bytes_that_were_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        let content: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &content).unwrap();

        let mut file = open_regular_file_no_follow(&path).unwrap();
        let authenticated = AuthenticatedBytes::read_once(&mut file, &path).unwrap();
        assert_eq!(authenticated.bytes(), content.as_slice());
        assert_eq!(
            authenticated.sha256,
            format!("{:x}", Sha256::digest(&content))
        );
        assert_eq!(
            authenticated.into_sha256(),
            mold_core::secure_file::sha256_open_file(&open_regular_file_no_follow(&path).unwrap())
                .unwrap(),
            "the single-read digest must agree with mold-core's file hasher"
        );
    }

    #[test]
    fn an_empty_file_still_hashes_what_was_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();
        let mut file = open_regular_file_no_follow(&path).unwrap();
        let authenticated = AuthenticatedBytes::read_once(&mut file, &path).unwrap();
        assert!(authenticated.bytes().is_empty());
        assert_eq!(
            authenticated.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// Structural guard for the invariant [`AuthenticatedBytes`] is shaped
    /// around: the module must contain exactly ONE read of the descriptor, and
    /// must not reach for a hasher that takes a `&File` — which would
    /// necessarily leave the bytes to a second read, reopening the window
    /// where an in-place write authenticates one set of bytes and executes
    /// another. A refactor that reintroduces the two-pass shape fails here.
    #[test]
    fn the_loader_reads_the_descriptor_exactly_once() {
        let source = include_str!("onnx_graph.rs");
        let code = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        // The needles are split so this test's own source does not count as a
        // match — otherwise every assertion below would find itself.
        let read_all = concat!("read_to", "_end");
        let file_hasher = concat!("sha256_open", "_file");
        let path_read = concat!("std::fs", "::read");
        assert_eq!(
            code.matches(read_all).count(),
            1,
            "the descriptor must be read exactly once"
        );
        assert_eq!(
            code.matches(file_hasher).count(),
            1,
            "only the cross-check test above may name mold-core's &File hasher"
        );
        assert!(
            !code.contains(path_read),
            "a path-based read would abandon the retained descriptor entirely"
        );
    }

    #[test]
    fn a_matching_digest_loads() {
        let dir = tempfile::tempdir().unwrap();
        let (path, digest) = write_proto(dir.path(), "model.onnx", false);
        let loaded = load_onnx_model(&path, Some(&digest)).expect("the pinned model loads");
        assert_eq!(loaded.sha256, digest);
    }

    /// The whole point of #1222's P1 fix: a modified model is refused even
    /// though the downloader's marker beside it still claims it was verified.
    #[test]
    fn a_byte_flipped_model_is_refused_before_it_is_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let (_, pinned) = write_proto(dir.path(), "pristine.onnx", false);
        let (tampered, actual) = write_proto(dir.path(), "tampered.onnx", true);
        assert_ne!(pinned, actual, "the flip must change the digest");

        let err = load_onnx_model(&tampered, Some(&pinned)).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("tampered.onnx"), "{message}");
        assert!(
            message.contains(&pinned),
            "expected digest missing: {message}"
        );
        assert!(message.contains(&actual), "found digest missing: {message}");
        assert!(
            err.downcast_ref::<DigestMismatch>().is_some(),
            "must be the typed error, got {message}"
        );
        // And it is still refused when the file happens to be a valid proto —
        // the check runs before the decode, so decodability is no defence.
        assert!(load_onnx_model(&tampered, None).is_ok());
    }

    /// A caller that passes the wrong component's pin must be refused too,
    /// which is what stops the two models being swapped for each other.
    #[test]
    fn the_recognizers_pin_does_not_admit_the_detector() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_proto(dir.path(), "model.onnx", false);
        assert!(load_onnx_model(&path, pinned_sha256(ModelComponent::FaceDetector)).is_err());
        assert!(load_onnx_model(&path, pinned_sha256(ModelComponent::FaceRecognizer)).is_err());
    }

    /// The pins come from the manifest, never from a second copy in this
    /// crate. If `manifest.rs` drops one, this fails rather than silently
    /// loading unverified.
    #[test]
    fn both_face_components_are_pinned_by_the_manifest() {
        let detector = pinned_sha256(ModelComponent::FaceDetector).expect("detector is pinned");
        let recognizer =
            pinned_sha256(ModelComponent::FaceRecognizer).expect("recognizer is pinned");
        assert_eq!(detector.len(), 64, "{detector}");
        assert_eq!(recognizer.len(), 64, "{recognizer}");
        assert_ne!(detector, recognizer);
        assert!(detector.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(recognizer.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn digest_comparison_is_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        let (path, digest) = write_proto(dir.path(), "model.onnx", false);
        assert!(load_onnx_model(&path, Some(&digest.to_uppercase())).is_ok());
    }

    #[test]
    fn a_non_onnx_file_is_a_decode_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.onnx");
        std::fs::write(&path, b"this is not a protobuf at all, not even close").unwrap();
        let err = load_onnx_model(&path, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("failed to decode"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn a_missing_model_names_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.onnx");
        let err = load_onnx_model(&path, None).unwrap_err();
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
            load_onnx_model(&link, None).is_err(),
            "a symlinked model must not be opened"
        );
    }
}
