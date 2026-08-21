//! Convert the official EVA02-CLIP-L-14-336 release to vision-only
//! safetensors.
//!
//! BAAI publishes `EVA02_CLIP_L_336_psz14_s6B.pt` as a torch pickle, and
//! **mold's runtime never reads a pickle**. This module is the one place the
//! pickle is opened: it converts once, deterministically, from the SHA-verified
//! source, and everything downstream loads the derived safetensors through the
//! ordinary `VarBuilder`.
//!
//! ## Why this is safe to run on a pickle at all
//!
//! Per the Hugging Face scanner the checkpoint contains only `OrderedDict`,
//! `_rebuild_tensor_v2` and `HalfStorage`, all of which candle's reader
//! handles as data (`candle-core/src/pickle.rs:240-246`, `:636-645`). candle
//! never evaluates arbitrary opcodes.
//!
//! ## Why the fence looks the way it does
//!
//! candle's `PthTensors` re-opens the file **by pathname** for every tensor
//! ("We hope that the file has not changed since first reading it",
//! `pickle.rs:770-772`), so a retained descriptor alone proves nothing about
//! what the next `File::open` will return. Handing candle a `/dev/fd/N`
//! pathname derived from the retained descriptor does not help either: on
//! macOS opening `/dev/fd/N` is `dup(N)`, so candle's second open would
//! inherit an exhausted offset and read zero bytes.
//!
//! What does work is inode pinning. We open the source no-follow, keep that
//! descriptor open for the whole conversion, and verify `(device, inode)` on a
//! fresh no-follow open both before and after candle reads. An open descriptor
//! keeps the inode allocated, so it cannot be recycled — a swap-and-swap-back
//! would have to hand back an inode number that is provably still ours, and
//! cannot. Combined with the pinned source SHA-256 (hashed *through* the
//! retained descriptor, not through the pathname) this fences symlink
//! substitution, path swaps, and content substitution.

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use anyhow::{bail, ensure, Context, Result};
use candle_core::pickle::PthTensors;
use mold_core::pulid_assets::PulidPaths;
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The derived artifact's filename, written beside the `.pt` it came from.
pub(crate) const DERIVED_FILENAME: &str = "eva02_clip_l_336_vision.safetensors";
/// Records the derived SHA-256 so a later run can skip the conversion without
/// re-reading 856 MB of pickle.
pub(crate) const SIDECAR_FILENAME: &str = "eva02_clip_l_336_vision.json";
/// Staging name for the atomic sibling write.
const STAGING_SUFFIX: &str = ".staging";

/// Source pin, mirrored from `mold_core::manifest`'s `pulid-flux` entry. Kept
/// here as well so the conversion refuses to read anything else even if it is
/// handed a path directly.
pub(crate) const SOURCE_SHA256: &str =
    "84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05";

/// Tensors to keep: the vision tower, and nothing else.
const VISION_PREFIX: &str = "visual.";

/// The per-block RoPE buffers are byte-identical copies of the shared
/// `visual.rope.*` tables — upstream builds ONE `VisionRotaryEmbeddingFast`
/// and hands the same module to every block (`eva_vit_model.py:404,422`), so
/// `state_dict` serializes it 25 times. Mold derives the tables anyway; the
/// single top-level pair is retained so a test can check the derivation
/// against the checkpoint's own numbers, and the 48 per-block duplicates
/// (24 blocks x cos/sin) are dropped. 562 `visual.*` tensors become 514.
fn is_duplicate_rope_buffer(name: &str) -> bool {
    name.starts_with("visual.blocks.") && name.contains(".rope.")
}

/// A converted tensor on its way to disk.
struct RawTensor {
    name: String,
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl RawTensor {
    fn view(&self) -> Result<TensorView<'_>> {
        TensorView::new(self.dtype, self.shape.clone(), &self.data)
            .with_context(|| format!("failed to view {}", self.name))
    }
}

#[cfg(unix)]
fn file_identity(file: &File) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata().context("failed to stat the source")?;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(file: &File) -> Result<(u64, u64)> {
    let metadata = file.metadata().context("failed to stat the source")?;
    Ok((metadata.len(), metadata.len()))
}

/// Re-open `path` no-follow and confirm it is still the inode `retained` holds.
fn assert_still_the_same_file(path: &Path, retained: &File) -> Result<()> {
    let expected = file_identity(retained)?;
    let current = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to re-open {} no-follow", path.display()))?;
    ensure!(
        file_identity(&current)? == expected,
        "{} was replaced during conversion",
        path.display()
    );
    Ok(())
}

fn dtype_for(dtype: candle_core::DType) -> Result<SafeDtype> {
    Ok(match dtype {
        candle_core::DType::F16 => SafeDtype::F16,
        candle_core::DType::BF16 => SafeDtype::BF16,
        candle_core::DType::F32 => SafeDtype::F32,
        candle_core::DType::F64 => SafeDtype::F64,
        candle_core::DType::U8 => SafeDtype::U8,
        candle_core::DType::U32 => SafeDtype::U32,
        candle_core::DType::I64 => SafeDtype::I64,
        other => bail!("EVA02-CLIP conversion does not handle {other:?} weights"),
    })
}

/// Write `tensors` to `destination` atomically and deterministically.
///
/// Deterministic because `safetensors`' own `prepare` sorts by (descending
/// dtype alignment, name) before laying the buffer out, so the byte image does
/// not depend on the order tensors were read in. Atomic because the staging
/// file is a sibling — same directory, therefore same filesystem, therefore a
/// real `rename` — fsynced before it is published. A crash leaves either the
/// previous artifact or nothing, never a truncated one.
fn write_atomically(tensors: &[RawTensor], destination: &Path) -> Result<String> {
    let parent = destination
        .parent()
        .context("the conversion destination has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let staging = staging_path(destination);
    // A staging file left by an interrupted run is meaningless: it was never
    // published, so it is replaced outright rather than resumed.
    if staging.exists() {
        std::fs::remove_file(&staging)
            .with_context(|| format!("failed to clear {}", staging.display()))?;
    }

    let views = tensors
        .iter()
        .map(|tensor| Ok((tensor.name.clone(), tensor.view()?)))
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, &None, &staging)
        .with_context(|| format!("failed to write {}", staging.display()))?;

    let digest = {
        let file = File::open(&staging)
            .with_context(|| format!("failed to re-open {}", staging.display()))?;
        let sha = sha256_open_file(&file)?;
        file.sync_all()
            .with_context(|| format!("failed to fsync {}", staging.display()))?;
        sha
    };
    std::fs::rename(&staging, destination).with_context(|| {
        format!(
            "failed to publish {} as {}",
            staging.display(),
            destination.display()
        )
    })?;
    // Durability of the rename itself: fsync the directory entry.
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(digest)
}

fn staging_path(destination: &Path) -> PathBuf {
    let mut name = destination.as_os_str().to_os_string();
    name.push(STAGING_SUFFIX);
    PathBuf::from(name)
}

/// The sidecar sits beside its own artifact and is named after it, so two
/// destinations in one directory (which the tests do, and a future second
/// derived artifact might) cannot share one record.
/// [`SIDECAR_FILENAME`] is what this produces for [`DERIVED_FILENAME`].
fn sidecar_path(destination: &Path) -> PathBuf {
    destination.with_extension("json")
}

fn write_sidecar(destination: &Path, derived_sha256: &str) -> Result<()> {
    let body = format!(
        "{{\n  \"source_sha256\": \"{SOURCE_SHA256}\",\n  \
         \"derived_sha256\": \"{derived_sha256}\",\n  \
         \"derived_filename\": \"{DERIVED_FILENAME}\"\n}}\n"
    );
    let path = sidecar_path(destination);
    std::fs::write(&path, body).with_context(|| format!("failed to write {}", path.display()))
}

fn read_sidecar_sha(destination: &Path) -> Option<String> {
    let body = std::fs::read_to_string(sidecar_path(destination)).ok()?;
    let marker = "\"derived_sha256\"";
    let rest = body.split_once(marker)?.1;
    let rest = rest.split_once('"')?.1;
    Some(rest.split_once('"')?.0.to_string())
}

/// Convert `source` (the pinned `.pt`) into vision-only safetensors at
/// `destination`, returning the derived SHA-256.
pub(crate) fn convert_eva_clip_vision(source: &Path, destination: &Path) -> Result<String> {
    // 1. Open no-follow and RETAIN. Holding this descriptor for the whole
    //    conversion is what pins the inode.
    let retained = open_regular_file_no_follow(source)
        .with_context(|| format!("failed to open {} no-follow", source.display()))?;
    // 2. Hash through the descriptor, never through the pathname.
    let source_sha = sha256_open_file(&retained)?;
    ensure!(
        source_sha == SOURCE_SHA256,
        "{} is not the pinned EVA02-CLIP release (sha256 {source_sha})",
        source.display()
    );
    // 3. The pathname must still resolve to the descriptor we hashed before we
    //    hand it to candle...
    assert_still_the_same_file(source, &retained)?;

    let pth = PthTensors::new(source, None)
        .with_context(|| format!("failed to read {} as a torch pickle", source.display()))?;
    let mut names: Vec<String> = pth
        .tensor_infos()
        .keys()
        .filter(|name| name.starts_with(VISION_PREFIX) && !is_duplicate_rope_buffer(name))
        .cloned()
        .collect();
    names.sort();
    ensure!(
        !names.is_empty(),
        "{} contains no `{VISION_PREFIX}` tensors",
        source.display()
    );

    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        let tensor = pth
            .get(&name)?
            .with_context(|| format!("{name} vanished between listing and read"))?
            .contiguous()?;
        // Strip the `visual.` prefix: the derived file is a vision tower, and
        // `EvaClipVisionTower` should not have to know it once lived inside a
        // CLIP.
        let stripped = name
            .strip_prefix(VISION_PREFIX)
            .unwrap_or(&name)
            .to_string();
        tensors.push(RawTensor {
            name: stripped,
            dtype: dtype_for(tensor.dtype())?,
            shape: tensor.dims().to_vec(),
            data: tensor_bytes(&tensor)?,
        });
    }

    // 4. ...and still afterwards, closing the window candle's re-opens leave.
    assert_still_the_same_file(source, &retained)?;

    let derived = write_atomically(&tensors, destination)?;
    write_sidecar(destination, &derived)?;
    Ok(derived)
}

/// Raw little-endian bytes of a contiguous CPU tensor, dtype preserved.
///
/// The conversion is a re-container, not a cast: the f16 release stays f16 so
/// the derived artifact is ~609 MB rather than 1.2 GB, and the loading
/// `VarBuilder` picks the compute dtype.
fn tensor_bytes(tensor: &candle_core::Tensor) -> Result<Vec<u8>> {
    use candle_core::DType;
    let flat = tensor.flatten_all()?;
    Ok(match tensor.dtype() {
        DType::F16 => flat
            .to_vec1::<half::f16>()?
            .into_iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => flat
            .to_vec1::<half::bf16>()?
            .into_iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        DType::F32 => flat
            .to_vec1::<f32>()?
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F64 => flat
            .to_vec1::<f64>()?
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::U8 => flat.to_vec1::<u8>()?,
        DType::U32 => flat
            .to_vec1::<u32>()?
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::I64 => flat
            .to_vec1::<i64>()?
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        other => bail!("EVA02-CLIP conversion does not handle {other:?} weights"),
    })
}

/// Where the derived artifact lives for an installed bundle: beside the `.pt`,
/// in the shared PuLID root.
pub(crate) fn derived_vision_path(paths: &PulidPaths) -> PathBuf {
    paths.vision_encoder_source.with_file_name(DERIVED_FILENAME)
}

/// Materialize the vision tower's safetensors, converting on first use.
///
/// Idempotent: a derived file whose SHA-256 matches the recorded sidecar is
/// accepted as-is. Anything else — missing file, missing sidecar, mismatched
/// digest — reconverts, because a half-written or hand-edited artifact must
/// never be loaded as weights.
///
/// This is the entry point admission calls once it has resolved a complete
/// bundle through [`mold_core::pulid_assets::pulid_paths`]. It is deliberately
/// convert-on-first-use rather than a download post-hook: the bundle's install
/// flow is being built concurrently (#1220 / dependency planning), and hanging
/// an 856 MB pickle read off the download path would couple the two.
pub(crate) fn ensure_eva_clip_vision_safetensors(paths: &PulidPaths) -> Result<PathBuf> {
    let destination = derived_vision_path(paths);
    if destination.is_file() {
        if let Some(recorded) = read_sidecar_sha(&destination) {
            if let Ok(file) = open_regular_file_no_follow(&destination) {
                if sha256_open_file(&file).ok().as_deref() == Some(recorded.as_str()) {
                    return Ok(destination);
                }
            }
        }
    }
    convert_eva_clip_vision(&paths.vision_encoder_source, &destination)?;
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulid_fixtures::pulid_asset;
    use candle_core::{DType, Device};

    fn raw(name: &str, values: &[f32], shape: &[usize]) -> RawTensor {
        RawTensor {
            name: name.to_string(),
            dtype: SafeDtype::F32,
            shape: shape.to_vec(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// The write half is deterministic on its own: the same tensors in a
    /// different order must produce the same bytes, or the recorded derived
    /// SHA is meaningless.
    #[test]
    fn the_derived_bytes_do_not_depend_on_tensor_order() {
        let dir = tempfile::tempdir().unwrap();
        let forward = vec![
            raw("a.weight", &[1.0, 2.0], &[2]),
            raw("b.weight", &[3.0, 4.0, 5.0], &[3]),
            raw("c.bias", &[6.0], &[1]),
        ];
        let reversed: Vec<RawTensor> = vec![
            raw("c.bias", &[6.0], &[1]),
            raw("b.weight", &[3.0, 4.0, 5.0], &[3]),
            raw("a.weight", &[1.0, 2.0], &[2]),
        ];
        let first = write_atomically(&forward, &dir.path().join("one.safetensors")).unwrap();
        let second = write_atomically(&reversed, &dir.path().join("two.safetensors")).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            std::fs::read(dir.path().join("one.safetensors")).unwrap(),
            std::fs::read(dir.path().join("two.safetensors")).unwrap()
        );
    }

    /// An interrupted conversion leaves a `.staging` file. It was never
    /// published, so the next run must overwrite it and must not leave it
    /// behind.
    #[test]
    fn a_stale_staging_file_is_replaced_not_resumed() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("weights.safetensors");
        let staging = staging_path(&destination);
        std::fs::write(&staging, b"a truncated previous attempt").unwrap();
        assert!(staging.exists());

        let digest = write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination).unwrap();
        assert!(!staging.exists(), "the staging file survived");
        assert!(destination.is_file());

        let loaded = candle_core::safetensors::load(&destination, &Device::Cpu).unwrap();
        assert_eq!(
            loaded["a.weight"].to_vec1::<f32>().unwrap(),
            vec![1.0_f32, 2.0]
        );
        // Byte-identical to a run that never saw a staging file.
        let clean = dir.path().join("clean.safetensors");
        assert_eq!(
            write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &clean).unwrap(),
            digest
        );
    }

    /// A symlink at the source path is refused before anything is parsed, and
    /// the message says so rather than blaming the pickle.
    #[test]
    #[cfg(unix)]
    fn a_symlinked_source_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.pt");
        std::fs::write(&real, b"not a pickle").unwrap();
        let link = dir.path().join("link.pt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let error =
            convert_eva_clip_vision(&link, &dir.path().join("out.safetensors")).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("no-follow"),
            "expected a no-follow refusal, got: {message}"
        );
        assert!(!dir.path().join("out.safetensors").exists());
    }

    /// A file whose bytes are not the pinned release is refused on the hash,
    /// before the pickle reader ever sees it.
    #[test]
    fn an_unpinned_source_is_refused_on_its_digest() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("impostor.pt");
        std::fs::write(&source, b"definitely not EVA02-CLIP").unwrap();
        let error =
            convert_eva_clip_vision(&source, &dir.path().join("out.safetensors")).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("not the pinned EVA02-CLIP release"),
            "expected a digest refusal, got: {message}"
        );
    }

    /// A path swapped for a different file between the hash and the parse is
    /// caught by the inode fence.
    #[test]
    fn a_path_swap_is_caught_by_the_inode_fence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swapped.bin");
        std::fs::write(&path, b"original").unwrap();
        let retained = open_regular_file_no_follow(&path).unwrap();
        assert!(assert_still_the_same_file(&path, &retained).is_ok());

        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"replacement").unwrap();
        let error = assert_still_the_same_file(&path, &retained).unwrap_err();
        assert!(
            error.to_string().contains("was replaced during conversion"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_duplicate_rope_buffers_are_the_only_thing_dropped() {
        assert!(is_duplicate_rope_buffer(
            "visual.blocks.7.attn.rope.freqs_cos"
        ));
        assert!(!is_duplicate_rope_buffer("visual.rope.freqs_cos"));
        assert!(!is_duplicate_rope_buffer(
            "visual.blocks.7.attn.q_proj.weight"
        ));
    }

    #[test]
    fn the_sidecar_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join(DERIVED_FILENAME);
        assert_eq!(
            sidecar_path(&destination).file_name().unwrap(),
            SIDECAR_FILENAME
        );
        write_sidecar(&destination, "deadbeef").unwrap();
        assert_eq!(read_sidecar_sha(&destination).as_deref(), Some("deadbeef"));
        // A sibling artifact has its own record, so it is not accidentally
        // accepted on this one's digest.
        assert!(read_sidecar_sha(&dir.path().join("absent.safetensors")).is_none());
        assert!(read_sidecar_sha(&dir.path().join("nowhere/x.safetensors")).is_none());
    }

    /// Precondition test: the pinned checkpoint carries exactly the tensor
    /// names and shapes the tower is built against. This is what catches a
    /// re-uploaded checkpoint before a shape error in a block does.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_pinned_checkpoint_carries_the_expected_vision_tensors() {
        use super::super::eva_clip_vision::{
            DEPTH, EMBED_DIM, MLP_HIDDEN_DIM, PATCH_SIZE, PROJECTION_DIM, SEQUENCE_LEN,
        };
        let source = pulid_asset("EVA02_CLIP_L_336_psz14_s6B.pt");
        let pth = PthTensors::new(&source, None).unwrap();
        let infos = pth.tensor_infos();
        let shape_of = |name: &str| -> Vec<usize> {
            infos
                .get(name)
                .unwrap_or_else(|| panic!("{name} is missing"))
                .layout
                .shape()
                .dims()
                .to_vec()
        };
        assert_eq!(shape_of("visual.cls_token"), vec![1, 1, EMBED_DIM]);
        assert_eq!(
            shape_of("visual.pos_embed"),
            vec![1, SEQUENCE_LEN, EMBED_DIM]
        );
        assert_eq!(
            shape_of("visual.patch_embed.proj.weight"),
            vec![EMBED_DIM, 3, PATCH_SIZE, PATCH_SIZE]
        );
        assert_eq!(shape_of("visual.rope.freqs_cos"), vec![576, 64]);
        assert_eq!(
            shape_of("visual.head.weight"),
            vec![PROJECTION_DIM, EMBED_DIM]
        );
        for block in 0..DEPTH {
            let prefix = format!("visual.blocks.{block}");
            for name in ["q_proj", "k_proj", "v_proj", "proj"] {
                assert_eq!(
                    shape_of(&format!("{prefix}.attn.{name}.weight")),
                    vec![EMBED_DIM, EMBED_DIM],
                    "{prefix}.attn.{name}"
                );
            }
            // q and v carry an out-of-band bias; k deliberately has none.
            assert_eq!(shape_of(&format!("{prefix}.attn.q_bias")), vec![EMBED_DIM]);
            assert_eq!(shape_of(&format!("{prefix}.attn.v_bias")), vec![EMBED_DIM]);
            assert!(
                !infos.contains_key(&format!("{prefix}.attn.k_bias")),
                "the checkpoint grew a k bias"
            );
            assert_eq!(
                shape_of(&format!("{prefix}.mlp.w1.weight")),
                vec![MLP_HIDDEN_DIM, EMBED_DIM]
            );
            assert_eq!(
                shape_of(&format!("{prefix}.mlp.w3.weight")),
                vec![EMBED_DIM, MLP_HIDDEN_DIM]
            );
            // No layer scale for this config.
            assert!(!infos.contains_key(&format!("{prefix}.gamma_1")));
        }
    }

    /// The derived SHA-256 is stable across runs on the pinned source, and the
    /// derived file loads as the tower expects.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn conversion_is_deterministic_on_the_pinned_source() {
        let source = pulid_asset("EVA02_CLIP_L_336_psz14_s6B.pt");
        let dir = tempfile::tempdir().unwrap();
        let first = convert_eva_clip_vision(&source, &dir.path().join(DERIVED_FILENAME)).unwrap();
        let second =
            convert_eva_clip_vision(&source, &dir.path().join("again.safetensors")).unwrap();
        assert_eq!(first, second, "conversion is not deterministic");
        println!("derived sha256: {first}");

        let loaded =
            candle_core::safetensors::load(dir.path().join(DERIVED_FILENAME), &Device::Cpu)
                .unwrap();
        // `visual.` is stripped, the text tower is gone, and the duplicated
        // per-block RoPE buffers are gone.
        assert!(loaded.contains_key("cls_token"));
        assert!(loaded.contains_key("rope.freqs_cos"));
        assert!(!loaded.keys().any(|k| k.starts_with("text.")));
        assert!(!loaded.keys().any(|k| k.contains("blocks.0.attn.rope")));
        // The release is f16 and the conversion is a re-container, not a
        // cast: 609 MB out, not 1.2 GB.
        assert_eq!(loaded["cls_token"].dtype(), DType::F16);
        assert_eq!(loaded.len(), 514);
        // Pinned so a re-uploaded source or a changed retention rule is loud.
        assert_eq!(
            first,
            "2b0b0ab0baed6ee968c8a08a9dcba908fb602630303faa3515eeaf8e264f136b"
        );
    }
}
