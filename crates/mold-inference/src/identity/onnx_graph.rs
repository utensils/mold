//! Descriptor-fenced loading of an ONNX `ModelProto`.
//!
//! `candle_onnx::read_file` takes a path and calls `std::fs::read`, which is
//! exactly the race mold's model-storage rules forbid: the path can be
//! replaced between the check and the read. This module opens the file with
//! [`mold_core::secure_file::open_regular_file_no_follow`] — no symlink in the
//! filename or any parent component, regular files only, canonical
//! traversal — and decodes the retained descriptor's bytes.
//!
//! Installed loading reads the retained descriptor once into a bounded private
//! buffer, checks metadata stability, and decodes it without hashing. New
//! downloads are verified before publication. The explicit verification entry
//! point hashes that same buffer and compares the declared pin. Both paths
//! enforce exact manifest lengths (or [`UNPINNED_MAX_BYTES`]) and format checks.
//! Shared model permissions remain supported; runtime trusts complete installs.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use candle_onnx::onnx::ModelProto;
use mold_core::identity::IdentityFamily;
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::pulid_manifest_for;
use mold_core::secure_file::open_regular_file_no_follow;
use prost::Message;
use sha2::{Digest, Sha256};

/// One ONNX graph plus the digest of the exact bytes it was decoded from.
#[derive(Debug, Clone)]
pub struct LoadedOnnxModel {
    /// The decoded graph.
    pub model: ModelProto,
    /// Observed SHA-256 when available; empty for an unverified local installation.
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
    "{path} does not match the pinned SHA-256 for this PuLID asset\n  expected {expected}\n  found    {found}\nre-pull the bundle it came from: mold pull pulid-flux (or pulid-sdxl)"
)]
pub struct DigestMismatch {
    /// The file that failed.
    pub path: String,
    /// The manifest's pin.
    pub expected: String,
    /// What the retained descriptor actually hashed to.
    pub found: String,
}

/// The file on disk is not the size the manifest pinned, or is implausibly
/// large for a graph nobody pinned.
///
/// Separate from [`DigestMismatch`] because it is raised at a different
/// moment and for a different reason: the digest can only be computed from
/// bytes already in memory, so size is what decides whether they may be read
/// at all.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactSizeError {
    /// The manifest pins an exact byte count and the file is not it.
    #[error(
        "{path} is {found} bytes but the manifest pins {expected} for this PuLID asset\nre-pull the bundle it came from: mold pull pulid-flux (or pulid-sdxl)"
    )]
    Mismatch {
        /// The file that failed.
        path: String,
        /// The manifest's pinned length.
        expected: u64,
        /// What the retained descriptor reports.
        found: u64,
    },
    /// No pin was supplied and the file is past the unpinned ceiling.
    #[error(
        "{path} is {found} bytes, past the {cap}-byte ceiling for an ONNX graph loaded without a manifest pin"
    )]
    OverCap {
        /// The file that failed.
        path: String,
        /// What the retained descriptor reports.
        found: u64,
        /// The ceiling that was applied.
        cap: u64,
    },
}

/// Ceiling for a graph loaded without a manifest pin — the inventory tool and
/// the benchmark, which take arbitrary paths by design.
///
/// Generous on purpose: the largest graph mold itself loads is `glintr100` at
/// ~249 MiB, so this leaves four times that. It is not a correctness bound, it
/// is a refusal to allocate unbounded memory on behalf of a file nobody
/// vouched for.
pub const UNPINNED_MAX_BYTES: u64 = 1 << 30;

/// Everything the manifest pins about one artifact.
///
/// Digest and length travel together, from one manifest lookup, so a caller
/// cannot pair one component's digest with another's length. Both are needed
/// and they are only correct as a pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedArtifact {
    /// Lowercase hex SHA-256 the file must hash to.
    ///
    /// `Cow` so the manifest's `&'static str` is borrowed with no allocation
    /// on the real path, while a test can pin a digest it computed at runtime.
    pub sha256: Cow<'static, str>,
    /// Exact byte count the file must have.
    pub size_bytes: u64,
}

/// The manifest's pin for one SHARED PuLID component.
///
/// Read from the manifest rather than copied, so there is exactly one place a
/// pin lives. `None` for a component the manifest does not pin, which for this
/// bundle cannot happen — a completeness test in `manifest.rs` requires all
/// five — but is not worth a panic.
///
/// Deliberately takes no family: every component this answers for is one of the
/// four EXTRACTION artifacts, which both bundles carry identically at the same
/// on-disk path. `the_shared_extraction_pins_agree_across_families` pins that
/// claim. The family-specific `IdentityAdapter` is
/// `extraction::adapter_sha256`'s question, not this one.
pub fn pinned_artifact(component: ModelComponent) -> Option<PinnedArtifact> {
    debug_assert_ne!(component, ModelComponent::IdentityAdapter);
    let file = pulid_manifest_for(IdentityFamily::Flux)
        .files
        .iter()
        .find(|file| file.component == component)?;
    Some(PinnedArtifact {
        sha256: Cow::Borrowed(file.sha256?),
        size_bytes: file.size_bytes,
    })
}

/// Just the digest half of [`pinned_artifact`], for callers cross-checking a
/// pin they already hold.
pub fn pinned_sha256(component: ModelComponent) -> Option<&'static str> {
    pulid_manifest_for(IdentityFamily::Flux)
        .files
        .iter()
        .find(|file| file.component == component)
        .and_then(|file| file.sha256)
}

/// Open, hash, verify, and decode an ONNX model without ever re-opening its
/// path.
///
/// `pin` is checked **before** the proto is decoded, against the digest of the
/// same retained descriptor the bytes are read from — so a file swapped
/// between the check and the read cannot be the file that runs.
///
/// The pin's **length** is checked earlier still, and is what makes the read
/// safe to perform at all. A digest can only be computed from bytes already in
/// memory, so an unbounded `read_to_end` on a replacement file the size of the
/// disk would exhaust memory long before it could be reported as a mismatch.
/// The retained descriptor is `fstat`ed first, an unexpected length is refused
/// as [`ArtifactSizeError`], and the read is then bounded regardless — so a
/// file that grows between the stat and the read is refused too rather than
/// silently truncated. Without a pin the same bound applies at
/// [`UNPINNED_MAX_BYTES`].
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
pub fn load_onnx_model(path: &Path, pin: Option<PinnedArtifact>) -> Result<LoadedOnnxModel> {
    let mut file = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open the ONNX model at {}", path.display()))?;
    let authenticated =
        LoadedBytes::read_once(&mut file, path, pin.as_ref().map(|p| p.size_bytes))?;
    authenticated.verify(pin.as_ref().map(|p| p.sha256.as_ref()), path)?;
    let mut model = ModelProto::decode(authenticated.bytes())
        .with_context(|| format!("failed to decode the ONNX model at {}", path.display()))?;
    normalize_empty_optional_resize_inputs(&mut model);
    Ok(LoadedOnnxModel {
        bytes: authenticated.bytes().len(),
        sha256: authenticated.into_sha256(),
        model,
    })
}

/// Load installed graphs without checksum work. Retain read bounds, descriptor
/// identity and parser checks; explicit load_onnx_model remains a verifying probe.
pub(crate) fn load_installed_onnx_model(
    path: &Path,
    pin: Option<PinnedArtifact>,
) -> Result<LoadedOnnxModel> {
    let mut file = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open the ONNX model at {}", path.display()))?;
    let before = mold_core::download::installed_artifact_identity_from_file(path, &file)?;
    let loaded = LoadedBytes::read_bounded_with_policy(
        &mut file,
        path,
        pin.as_ref().map(|pin| pin.size_bytes),
        UNPINNED_MAX_BYTES,
        false,
    )?;
    anyhow::ensure!(
        mold_core::download::installed_artifact_identity_from_file(path, &file)? == before,
        "installed ONNX changed while loading"
    );
    let mut model = ModelProto::decode(loaded.bytes())
        .with_context(|| format!("failed to decode the ONNX model at {}", path.display()))?;
    normalize_empty_optional_resize_inputs(&mut model);
    Ok(LoadedOnnxModel {
        model,
        sha256: before.observed_sha256().unwrap_or_default().into(),
        bytes: loaded.bytes().len(),
    })
}

/// One bounded private read of a retained descriptor. Explicit verification
/// hashes this exact buffer; installed loading leaves the observed digest empty.
#[derive(Debug)]
struct LoadedBytes {
    bytes: Vec<u8>,
    sha256: String,
}

impl LoadedBytes {
    /// Read the whole descriptor exactly once, within a bound, and hash what
    /// was read.
    ///
    /// The descriptor comes from `open_regular_file_no_follow`, so it is
    /// positioned at zero and no seek is needed — nor performed, which is what
    /// keeps this to a single pass over the bytes.
    ///
    /// `expected_len` is the manifest's pinned byte count when there is one.
    /// The bound is not an optimization: an unbounded `read_to_end` on a
    /// replacement file the size of the disk exhausts memory before any digest
    /// can be computed, so the process dies instead of reporting the mismatch
    /// it was about to find.
    fn read_once(file: &mut File, path: &Path, expected_len: Option<u64>) -> Result<Self> {
        Self::read_bounded(file, path, expected_len, UNPINNED_MAX_BYTES)
    }

    /// [`read_once`](Self::read_once) with the unpinned ceiling supplied, so a
    /// test can exercise the ceiling without writing a gigabyte.
    fn read_bounded(
        file: &mut File,
        path: &Path,
        expected_len: Option<u64>,
        unpinned_cap: u64,
    ) -> Result<Self> {
        Self::read_bounded_with_policy(file, path, expected_len, unpinned_cap, true)
    }

    fn read_bounded_with_policy(
        file: &mut File,
        path: &Path,
        expected_len: Option<u64>,
        unpinned_cap: u64,
        verify: bool,
    ) -> Result<Self> {
        // `fstat` on the retained descriptor, never a `stat` on the path —
        // the point of holding the descriptor is that this describes the same
        // file the bytes come from.
        let reported = file
            .metadata()
            .with_context(|| format!("failed to stat the ONNX model at {}", path.display()))?
            .len();
        let limit = match expected_len {
            Some(expected) if reported != expected => {
                return Err(ArtifactSizeError::Mismatch {
                    path: path.display().to_string(),
                    expected,
                    found: reported,
                }
                .into())
            }
            Some(expected) => expected,
            None if reported > unpinned_cap => {
                return Err(ArtifactSizeError::OverCap {
                    path: path.display().to_string(),
                    found: reported,
                    cap: unpinned_cap,
                }
                .into())
            }
            None => reported,
        };
        // `limit + 1` so a file that GREW between the stat and the read
        // overshoots the bound and is refused below, rather than being
        // silently truncated to a prefix that happens to parse. The buffer
        // is reserved at that same `limit + 1` up front: `read_to_end` grows
        // a full vector geometrically, so reserving only `limit` would let
        // the one sentinel byte double a 260 MiB (or 1 GiB) allocation on
        // exactly the concurrent-growth race this bound exists to survive.
        let bounded = limit.saturating_add(1);
        let capacity = usize::try_from(bounded)
            .with_context(|| format!("{} does not fit in memory on this target", path.display()))?;
        let mut bytes = Vec::with_capacity(capacity);
        file.by_ref()
            .take(bounded)
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read the ONNX model at {}", path.display()))?;
        if bytes.len() as u64 != limit {
            return Err(ArtifactSizeError::Mismatch {
                path: path.display().to_string(),
                expected: limit,
                found: bytes.len() as u64,
            }
            .into());
        }

        let sha256 = if verify {
            let mut observation =
                mold_core::download::ArtifactHashObservation::new("explicit_onnx_verification");
            observation.read(bytes.len() as u64);
            format!("{:x}", Sha256::digest(&bytes))
        } else {
            String::new()
        };
        Ok(Self { bytes, sha256 })
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
    /// Build a pin for a digest computed at runtime.
    fn pin(sha256: &str, size_bytes: u64) -> PinnedArtifact {
        PinnedArtifact {
            sha256: Cow::Owned(sha256.to_string()),
            size_bytes,
        }
    }

    /// The on-disk length of a fixture, which every pin needs.
    fn len_of(path: &std::path::Path) -> u64 {
        std::fs::metadata(path).unwrap().len()
    }

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
        let authenticated = LoadedBytes::read_once(&mut file, &path, None).unwrap();
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
        let authenticated = LoadedBytes::read_once(&mut file, &path, None).unwrap();
        assert!(authenticated.bytes().is_empty());
        assert_eq!(
            authenticated.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// Structural guard for the invariant [`LoadedBytes`] is shaped
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

    /// The size check runs BEFORE the bytes are read, so a replacement file is
    /// refused without ever being pulled into memory. Proved by pinning the
    /// file's own true digest against a wrong length: were size checked after
    /// hashing, the digest would match and the load would succeed.
    #[test]
    fn a_right_digest_with_a_wrong_size_is_refused_before_hashing() {
        let dir = tempfile::tempdir().unwrap();
        let (path, digest) = write_proto(dir.path(), "model.onnx", false);
        let real = len_of(&path);

        let err = load_onnx_model(&path, Some(pin(&digest, real + 1))).unwrap_err();
        match err.downcast_ref::<ArtifactSizeError>() {
            Some(ArtifactSizeError::Mismatch {
                path: named,
                expected,
                found,
            }) => {
                assert!(named.contains("model.onnx"), "{named}");
                assert_eq!(*expected, real + 1);
                assert_eq!(*found, real);
            }
            _ => panic!("expected a size mismatch, got {err:#}"),
        }

        // The same digest with the true length is accepted, so the refusal
        // above was the length and nothing else.
        assert!(load_onnx_model(&path, Some(pin(&digest, real))).is_ok());
    }

    /// A pinned file whose real length is enormous is refused on length alone
    /// — the case the bound exists for, where an unbounded read would exhaust
    /// memory before the digest could report the mismatch.
    #[test]
    fn an_oversized_pinned_file_never_reaches_the_read() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_proto(dir.path(), "model.onnx", false);
        // The manifest's real 16 MB detector pin against a tiny file.
        let err =
            load_onnx_model(&path, pinned_artifact(ModelComponent::FaceDetector)).unwrap_err();
        assert!(
            err.downcast_ref::<ArtifactSizeError>().is_some(),
            "must fail on size, not digest: {err:#}"
        );
    }

    /// Without a pin the ceiling still applies. Exercised through
    /// `read_bounded` with a tiny ceiling, so proving a gigabyte is refused
    /// does not require writing one.
    #[test]
    fn an_unpinned_read_past_the_ceiling_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.bin");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();

        let mut file = open_regular_file_no_follow(&path).unwrap();
        let err = LoadedBytes::read_bounded(&mut file, &path, None, 1024).unwrap_err();
        match err.downcast_ref::<ArtifactSizeError>() {
            Some(ArtifactSizeError::OverCap { found, cap, .. }) => {
                assert_eq!(*found, 4096);
                assert_eq!(*cap, 1024);
            }
            _ => panic!("expected an over-cap refusal, got {err:#}"),
        }

        // At the ceiling the same file reads normally.
        let mut file = open_regular_file_no_follow(&path).unwrap();
        let ok = LoadedBytes::read_bounded(&mut file, &path, None, 4096).unwrap();
        assert_eq!(ok.bytes().len(), 4096);
    }

    /// The production ceiling is stated once and clears the largest graph mold
    /// loads, so a legitimate pull is never refused by it.
    #[test]
    fn the_unpinned_ceiling_clears_every_pinned_artifact() {
        assert_eq!(UNPINNED_MAX_BYTES, 1_073_741_824);
        for component in [ModelComponent::FaceDetector, ModelComponent::FaceRecognizer] {
            let pinned = pinned_artifact(component).expect("pinned");
            assert!(
                pinned.size_bytes < UNPINNED_MAX_BYTES,
                "{component:?} is {} bytes",
                pinned.size_bytes
            );
        }
    }

    /// Both halves of a pin come from one manifest lookup, so a caller cannot
    /// pair one component's digest with another's length.
    #[test]
    fn a_pin_carries_the_matching_digest_and_length() {
        let detector = pinned_artifact(ModelComponent::FaceDetector).expect("pinned");
        let recognizer = pinned_artifact(ModelComponent::FaceRecognizer).expect("pinned");
        assert_eq!(detector.size_bytes, 16_923_827);
        assert_eq!(recognizer.size_bytes, 260_665_334);
        assert_ne!(detector.sha256, recognizer.sha256);
        assert_eq!(
            pinned_sha256(ModelComponent::FaceDetector),
            Some(detector.sha256.as_ref())
        );
    }

    #[test]
    fn installed_onnx_uses_size_and_parser_without_pin_verification() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_proto(dir.path(), "installed.onnx", false);
        let wrong = "0".repeat(64);
        let loaded = load_installed_onnx_model(&path, Some(pin(&wrong, len_of(&path)))).unwrap();
        assert!(
            loaded.sha256.is_empty(),
            "must not invent an observed digest"
        );
        assert!(load_installed_onnx_model(&path, Some(pin(&wrong, len_of(&path) + 1))).is_err());
        assert!(load_onnx_model(&path, Some(pin(&wrong, len_of(&path)))).is_err());
    }

    #[test]
    fn a_matching_digest_loads() {
        let dir = tempfile::tempdir().unwrap();
        let (path, digest) = write_proto(dir.path(), "model.onnx", false);
        let loaded = load_onnx_model(&path, Some(pin(&digest, len_of(&path))))
            .expect("the pinned model loads");
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

        let err = load_onnx_model(&tampered, Some(pin(&pinned, len_of(&tampered)))).unwrap_err();
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
        assert!(load_onnx_model(&path, pinned_artifact(ModelComponent::FaceDetector)).is_err());
        assert!(load_onnx_model(&path, pinned_artifact(ModelComponent::FaceRecognizer)).is_err());
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
        assert!(load_onnx_model(&path, Some(pin(&digest.to_uppercase(), len_of(&path)))).is_ok());
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
