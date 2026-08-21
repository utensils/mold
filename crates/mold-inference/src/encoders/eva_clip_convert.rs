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
//! ## The three things that authenticate a load
//!
//! 1. **The source is read through a private copy.** candle's `PthTensors`
//!    re-opens its file by pathname for every tensor, so hashing a descriptor
//!    and then handing candle a pathname authenticates nothing. The bytes are
//!    copied out of the retained descriptor into an exclusively created 0o700
//!    directory and hashed on that same stream, so the digest and the parse
//!    observe identical bytes by construction. See [`stage_private_copy`].
//! 2. **Every publish is a rename, never a write to the destination.**
//!    `rename` replaces a symlink instead of following it. See [`publish`].
//! 3. **Reuse is authenticated by a compiled-in pin, not by the sidecar.**
//!    See [`DERIVED_SHA256`].

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
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The derived artifact's filename, written beside the `.pt` it came from.
pub(crate) const DERIVED_FILENAME: &str = "eva02_clip_l_336_vision.safetensors";
/// Informational provenance beside the derived artifact.
pub(crate) const SIDECAR_FILENAME: &str = "eva02_clip_l_336_vision.json";

/// Source pin, mirrored from `mold_core::manifest`'s `pulid-flux` entry. Kept
/// here as well so the conversion refuses to read anything else even if it is
/// handed a path directly.
pub(crate) const SOURCE_SHA256: &str =
    "84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05";

/// Pin for the DERIVED artifact.
///
/// [`convert_eva_clip_vision`] is deterministic — it selects a fixed tensor set
/// from the pinned source and `safetensors`' own `prepare` sorts them before
/// laying out the buffer — so converting [`SOURCE_SHA256`] always produces
/// exactly these bytes. `conversion_is_deterministic_on_the_pinned_source`
/// re-derives this constant from the real checkpoint, so a `safetensors`
/// layout change or a re-uploaded source fails loudly here rather than
/// silently shipping different weights.
///
/// This constant, and never the sidecar, is what authenticates a derived file
/// that is being reused. The sidecar is written by mold, so anything able to
/// tamper with the weights can rewrite it to match; it is provenance for a
/// human, not an authenticator.
pub(crate) const DERIVED_SHA256: &str =
    "2b0b0ab0baed6ee968c8a08a9dcba908fb602630303faa3515eeaf8e264f136b";

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

/// An exclusively created, owner-only scratch directory beside a destination.
///
/// Everything this module writes before publishing goes in here, which is what
/// makes the staged bytes unforgeable: the directory is created with
/// `create_dir` — which fails rather than reusing an existing entry — at mode
/// `0o700`, so nothing can be waiting under any name we are about to use.
/// Contrast the model-storage rule in `CLAUDE.md`: shared, group-writable
/// *model* directories are legitimate and must keep working, which is exactly
/// why staging cannot happen directly in one.
///
/// Dropped on every path, success or error, so an interrupted conversion
/// leaves no partial 856 MB copy behind.
struct PrivateStagingDir {
    path: PathBuf,
}

impl PrivateStagingDir {
    /// Create the directory as a sibling of `destination` so a later `rename`
    /// into place stays within one filesystem.
    fn create_beside(destination: &Path) -> Result<Self> {
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .context("the conversion destination has no parent directory")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let mut last_error = None;
        for attempt in 0..16_u32 {
            let path = parent.join(format!(
                ".mold-eva-clip-convert.{}.{}.{attempt}",
                std::process::id(),
                nonce()
            ));
            match create_private_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no attempt was made")))
            .context("failed to create a private staging directory")
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for PrivateStagingDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A disambiguator for the staging directory's name. Not a security property —
/// exclusive creation is what makes the name safe to use — just enough that two
/// conversions in one process do not collide.
fn nonce() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed) as u128;
    let clock = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    clock ^ (count << 96)
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir(path).with_context(|| format!("failed to create {}", path.display()))
}

/// Create a file that must not already exist.
fn create_exclusive(path: &Path) -> Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

/// Publish `staging` at `destination`, durably.
///
/// `rename` REPLACES whatever sits at `destination` — including a symlink,
/// which it unlinks rather than follows. That is the whole reason every write
/// in this module goes through here instead of `std::fs::write`: a symlink
/// pre-planted at the destination would otherwise redirect our write into a
/// file of the attacker's choosing.
fn publish(staging: &Path, destination: &Path) -> Result<()> {
    std::fs::rename(staging, destination).with_context(|| {
        format!(
            "failed to publish {} as {}",
            staging.display(),
            destination.display()
        )
    })?;
    // Durability of the rename itself: fsync the directory entry.
    if let Some(parent) = destination.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Copy the bytes behind `retained` into a private file, hashing as we go, and
/// refuse anything that is not `expected_sha256`.
///
/// This is the answer to candle's `PthTensors`, which re-opens its file **by
/// pathname** for every tensor ("We hope that the file has not changed since
/// first reading it", `pickle.rs:770-772`). Hashing the original descriptor and
/// then handing candle the original pathname authenticates nothing: the name
/// can be renamed away and back between the two, and the parse would read bytes
/// the hash never saw. Re-checking `(device, inode)` around the parse does not
/// close it either — that only samples the pathname at two instants, and candle
/// re-opens between them.
///
/// A `/dev/fd/N` pathname derived from the retained descriptor is the obvious
/// alternative and does not work: on macOS opening `/dev/fd/N` is `dup(N)`, so
/// candle's second open would inherit an exhausted offset and read nothing.
///
/// So the bytes are copied out of the descriptor into a file only we can reach,
/// and the hash is computed on that same stream. The digest and the parse then
/// observe identical bytes by construction. The cost is one transient 856 MB
/// copy on an install-time path that is about to write 609 MB anyway.
fn stage_private_copy(
    retained: &File,
    staging: &PrivateStagingDir,
    name: &str,
    expected_sha256: &str,
) -> Result<PathBuf> {
    let path = staging.join(name);
    let mut source = retained
        .try_clone()
        .context("failed to clone the source descriptor")?;
    source
        .seek(SeekFrom::Start(0))
        .context("failed to rewind the source descriptor")?;

    let mut target = create_exclusive(&path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = source
            .read(&mut buffer)
            .context("failed to read the source descriptor")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        target
            .write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    target
        .sync_all()
        .with_context(|| format!("failed to fsync {}", path.display()))?;
    drop(target);

    let actual = format!("{:x}", digest.finalize());
    ensure!(
        actual == expected_sha256,
        "the source is not the pinned EVA02-CLIP release (sha256 {actual})"
    );
    Ok(path)
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
/// not depend on the order tensors were read in — which is what makes
/// [`DERIVED_SHA256`] a meaningful pin.
///
/// Atomic because the bytes are built inside a [`PrivateStagingDir`] beside the
/// destination and then renamed into place: same directory, therefore same
/// filesystem, therefore a real `rename`. A crash leaves either the previous
/// artifact or nothing — never a truncated one, and never a partially written
/// file under the destination's own name.
fn write_atomically(tensors: &[RawTensor], destination: &Path) -> Result<String> {
    let staging = PrivateStagingDir::create_beside(destination)?;
    let path = staging.join("weights.safetensors");

    let views = tensors
        .iter()
        .map(|tensor| Ok((tensor.name.clone(), tensor.view()?)))
        .collect::<Result<Vec<_>>>()?;
    // `serialize_to_file` opens the path itself, which is safe here and only
    // here: the containing directory was just created exclusively at 0o700, so
    // nothing can be waiting under this name.
    serialize_to_file(views, &None, &path)
        .with_context(|| format!("failed to write {}", path.display()))?;

    let digest = {
        let file =
            File::open(&path).with_context(|| format!("failed to re-open {}", path.display()))?;
        let sha = sha256_open_file(&file)?;
        file.sync_all()
            .with_context(|| format!("failed to fsync {}", path.display()))?;
        sha
    };
    publish(&path, destination)?;
    Ok(digest)
}

/// Write `bytes` at `destination` through the same staging-and-rename path.
fn publish_bytes(bytes: &[u8], destination: &Path) -> Result<()> {
    let staging = PrivateStagingDir::create_beside(destination)?;
    let path = staging.join("payload");
    let mut file = create_exclusive(&path)?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to fsync {}", path.display()))?;
    drop(file);
    publish(&path, destination)
}

/// The sidecar sits beside its own artifact and is named after it, so two
/// destinations in one directory (which the tests do, and a future second
/// derived artifact might) cannot share one record.
/// [`SIDECAR_FILENAME`] is what this produces for [`DERIVED_FILENAME`].
fn sidecar_path(destination: &Path) -> PathBuf {
    destination.with_extension("json")
}

/// Record what produced the derived artifact.
///
/// Informational only. Nothing reads this back to decide whether the weights
/// can be trusted — [`DERIVED_SHA256`] does that — because a file mold writes
/// is a file that anything able to reach the weights could rewrite to match.
fn write_sidecar(destination: &Path, derived_sha256: &str) -> Result<()> {
    let body = format!(
        "{{\n  \"source_sha256\": \"{SOURCE_SHA256}\",\n  \
         \"derived_sha256\": \"{derived_sha256}\",\n  \
         \"derived_filename\": \"{DERIVED_FILENAME}\",\n  \
         \"note\": \"Provenance only. Mold authenticates this artifact with a \
         compiled-in pin, never with this file.\"\n}}\n"
    );
    publish_bytes(body.as_bytes(), &sidecar_path(destination))
}

/// Read the sidecar's recorded digest. Diagnostics only — see
/// [`write_sidecar`].
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
    // 1. Open no-follow and RETAIN, so neither the filename nor any parent
    //    component can be a symlink and the bytes are now bound to a descriptor
    //    rather than to a name.
    let retained = open_regular_file_no_follow(source)
        .with_context(|| format!("failed to open {} no-follow", source.display()))?;

    // 2. Copy those bytes somewhere only we can reach, hashing the same stream,
    //    and require the manifest pin. From here on the pathname the caller
    //    gave us is irrelevant.
    let staging = PrivateStagingDir::create_beside(destination)?;
    let private_source = stage_private_copy(&retained, &staging, "source.pt", SOURCE_SHA256)
        .with_context(|| format!("{} failed its pin", source.display()))?;
    drop(retained);

    // 3. Parse the private copy. candle re-opens by pathname per tensor, which
    //    is now harmless: every one of those opens lands inside a 0o700
    //    directory created exclusively for this conversion.
    let pth = PthTensors::new(&private_source, None)
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
    // The pickle reader holds the private path; release it before the staging
    // directory is removed.
    drop(pth);
    drop(staging);

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

/// Is the file already at `destination` the artifact [`DERIVED_SHA256`] names?
///
/// Opened no-follow and hashed. The sidecar is deliberately not consulted: it
/// is mold's own writing, so anything that could tamper with the weights could
/// forge a matching record, and trusting it would turn "verified" into "the
/// attacker said so".
fn derived_artifact_is_authentic(destination: &Path) -> bool {
    let Ok(file) = open_regular_file_no_follow(destination) else {
        return false;
    };
    sha256_open_file(&file).ok().as_deref() == Some(DERIVED_SHA256)
}

/// Materialize the vision tower's safetensors, converting on first use.
///
/// Idempotent, and idempotent on the *bytes*: a derived file is reused only
/// when it hashes to [`DERIVED_SHA256`]. Anything else — missing, truncated,
/// tampered with, or carrying a forged sidecar — reconverts from the pinned
/// source, because a half-written or edited artifact must never be loaded as
/// weights. If the source itself fails its own pin the conversion errors rather
/// than falling back to whatever is on disk.
///
/// This is the entry point admission calls once it has resolved a complete
/// bundle through [`mold_core::pulid_assets::pulid_paths`]. It is deliberately
/// convert-on-first-use rather than a download post-hook: the bundle's install
/// flow is being built concurrently (#1220 / dependency planning), and hanging
/// an 856 MB pickle read off the download path would couple the two.
pub(crate) fn ensure_eva_clip_vision_safetensors(paths: &PulidPaths) -> Result<PathBuf> {
    let destination = derived_vision_path(paths);
    if derived_artifact_is_authentic(&destination) {
        return Ok(destination);
    }
    let derived = convert_eva_clip_vision(&paths.vision_encoder_source, &destination)?;
    // A fresh conversion that does not reproduce the pin means the pin and the
    // converter have diverged. Say so once, loudly, instead of silently
    // reconverting on every later call.
    ensure!(
        derived == DERIVED_SHA256,
        "converting {} produced sha256 {derived}, but this build pins \
         {DERIVED_SHA256}",
        paths.vision_encoder_source.display()
    );
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

    /// Nothing may be left in the destination directory except the artifacts
    /// themselves — no staging directory, no partial copy.
    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        names
    }

    /// The write half is deterministic on its own: the same tensors in a
    /// different order must produce the same bytes, or the pinned derived SHA
    /// is meaningless.
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

    /// Staging happens in a private directory that is removed on the way out,
    /// so an interrupted run leaves nothing to resume — or to trip over.
    #[test]
    fn staging_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("weights.safetensors");
        let digest = write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination).unwrap();
        write_sidecar(&destination, &digest).unwrap();
        assert_eq!(
            entries(dir.path()),
            vec![
                "weights.json".to_string(),
                "weights.safetensors".to_string()
            ],
        );

        let loaded = candle_core::safetensors::load(&destination, &Device::Cpu).unwrap();
        assert_eq!(
            loaded["a.weight"].to_vec1::<f32>().unwrap(),
            vec![1.0_f32, 2.0]
        );
    }

    /// A leftover staging directory from a killed run must not stop the next
    /// one: the name carries a nonce, so a new run never collides with it.
    #[test]
    fn a_stale_staging_directory_does_not_block_a_later_run() {
        let dir = tempfile::tempdir().unwrap();
        let stale = dir.path().join(".mold-eva-clip-convert.1.1.0");
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("source.pt"), b"a truncated previous attempt").unwrap();

        let destination = dir.path().join("weights.safetensors");
        let digest = write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination).unwrap();
        assert!(destination.is_file());
        // Byte-identical to a run that never saw one.
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
    /// before the pickle reader ever sees it — and the private copy of those
    /// unwanted bytes is cleaned up.
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
        assert_eq!(entries(dir.path()), vec!["impostor.pt".to_string()]);
    }

    /// The private copy is what gets parsed, and it holds exactly the bytes
    /// that were hashed. Renaming the source away mid-conversion — the race
    /// candle's re-open-by-pathname reader would otherwise expose — cannot
    /// reach it, because after `stage_private_copy` the original pathname is
    /// never used again.
    #[test]
    fn the_parsed_bytes_are_the_hashed_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.bin");
        let payload = b"the bytes that were hashed".repeat(1000);
        std::fs::write(&source, &payload).unwrap();
        let expected = format!("{:x}", Sha256::digest(&payload));

        let retained = open_regular_file_no_follow(&source).unwrap();
        let staging = PrivateStagingDir::create_beside(&dir.path().join("out.bin")).unwrap();

        // Swap the pathname for different content the moment the descriptor is
        // open. The copy must still be the original bytes.
        std::fs::remove_file(&source).unwrap();
        std::fs::write(&source, b"attacker payload").unwrap();

        let copy = stage_private_copy(&retained, &staging, "copy.bin", &expected).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), payload);
        // ...and re-reading the copy by pathname, exactly as candle does, sees
        // the same bytes.
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&copy).unwrap())),
            expected
        );

        let path = staging.path.clone();
        drop(staging);
        assert!(!path.exists(), "the staging directory survived");
    }

    #[test]
    fn a_copy_that_fails_its_pin_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.bin");
        std::fs::write(&source, b"content").unwrap();
        let retained = open_regular_file_no_follow(&source).unwrap();
        let staging = PrivateStagingDir::create_beside(&dir.path().join("out.bin")).unwrap();
        let error =
            stage_private_copy(&retained, &staging, "copy.bin", DERIVED_SHA256).unwrap_err();
        assert!(
            error.to_string().contains("not the pinned EVA02-CLIP"),
            "unexpected error: {error}"
        );
    }

    /// The staging directory is owner-only, which is what lets
    /// `serialize_to_file` and the pickle reader open paths inside it by name.
    #[test]
    #[cfg(unix)]
    fn the_staging_directory_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let staging = PrivateStagingDir::create_beside(&dir.path().join("out.bin")).unwrap();
        let mode = std::fs::metadata(&staging.path)
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "mode was {:o}", mode & 0o777);
        // Exclusive creation: a second attempt at the same name must fail.
        assert!(create_private_dir(&staging.path).is_err());
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

    /// A symlink pre-planted at the sidecar path must not redirect the write.
    /// `std::fs::write` follows it and truncates the victim; a staged rename
    /// replaces the link itself.
    #[test]
    #[cfg(unix)]
    fn the_sidecar_never_writes_through_a_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, b"do not touch").unwrap();

        let destination = dir.path().join(DERIVED_FILENAME);
        std::os::unix::fs::symlink(&victim, sidecar_path(&destination)).unwrap();

        write_sidecar(&destination, "deadbeef").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"do not touch");
        let metadata = std::fs::symlink_metadata(sidecar_path(&destination)).unwrap();
        assert!(
            metadata.file_type().is_file(),
            "the sidecar is still a symlink"
        );
        assert_eq!(read_sidecar_sha(&destination).as_deref(), Some("deadbeef"));
    }

    /// The same rule for the weights themselves.
    #[test]
    #[cfg(unix)]
    fn the_weights_never_write_through_a_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim.bin");
        std::fs::write(&victim, b"do not touch").unwrap();
        let destination = dir.path().join("weights.safetensors");
        std::os::unix::fs::symlink(&victim, &destination).unwrap();

        write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination).unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"do not touch");
        assert!(std::fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_file());
    }

    /// The reuse decision reads the bytes, never the sidecar. A forged record
    /// claiming the pinned digest must not launder tampered weights.
    #[test]
    fn a_forged_sidecar_cannot_authenticate_tampered_weights() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join(DERIVED_FILENAME);
        std::fs::write(&destination, b"tampered weights").unwrap();
        // Exactly what an attacker would write to make a sidecar-trusting
        // implementation accept the file above.
        write_sidecar(&destination, DERIVED_SHA256).unwrap();
        assert_eq!(
            read_sidecar_sha(&destination).as_deref(),
            Some(DERIVED_SHA256)
        );

        assert!(
            !derived_artifact_is_authentic(&destination),
            "a forged sidecar authenticated tampered weights"
        );
    }

    #[test]
    fn an_absent_or_symlinked_derived_artifact_is_not_authentic() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!derived_artifact_is_authentic(&dir.path().join("absent")));

        #[cfg(unix)]
        {
            let real = dir.path().join("real.safetensors");
            std::fs::write(&real, b"whatever").unwrap();
            let link = dir.path().join("link.safetensors");
            std::os::unix::fs::symlink(&real, &link).unwrap();
            assert!(!derived_artifact_is_authentic(&link));
        }
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

    /// The derived SHA-256 is stable across runs on the pinned source, matches
    /// the compiled-in pin, and the derived file loads as the tower expects.
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
        // This is where DERIVED_SHA256 comes from. A `safetensors` layout
        // change or a re-uploaded source fails here rather than silently
        // shipping different weights.
        assert_eq!(first, DERIVED_SHA256);

        let loaded =
            candle_core::safetensors::load(dir.path().join(DERIVED_FILENAME), &Device::Cpu)
                .unwrap();
        // `visual.` is stripped, the text tower is gone, and the duplicated
        // per-block RoPE buffers are gone.
        assert!(loaded.contains_key("cls_token"));
        assert!(loaded.contains_key("rope.freqs_cos"));
        assert!(!loaded.keys().any(|k| k.starts_with("text.")));
        assert!(!loaded.keys().any(|k| k.contains("blocks.0.attn.rope")));
        // The release is f16 and the conversion is a re-container, not a cast:
        // 609 MB out, not 1.2 GB.
        assert_eq!(loaded["cls_token"].dtype(), DType::F16);
        assert_eq!(loaded.len(), 514);
    }

    /// End-to-end reuse: a good artifact is accepted without reconverting, and
    /// a tampered one — even with a sidecar forged to match — is reconverted
    /// back to the pinned bytes.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn tampered_weights_are_reconverted_despite_a_matching_sidecar() {
        let source = pulid_asset("EVA02_CLIP_L_336_psz14_s6B.pt");
        let dir = tempfile::tempdir().unwrap();
        let staged_source = dir.path().join("EVA02_CLIP_L_336_psz14_s6B.pt");
        std::fs::copy(&source, &staged_source).unwrap();
        let paths = PulidPaths {
            adapter: dir.path().join("adapter.safetensors"),
            vision_encoder_source: staged_source,
            face_detector: dir.path().join("det.onnx"),
            face_recognizer: dir.path().join("rec.onnx"),
        };

        let destination = ensure_eva_clip_vision_safetensors(&paths).unwrap();
        assert!(derived_artifact_is_authentic(&destination));
        let good = std::fs::read(&destination).unwrap();

        // Reuse must not rewrite the file.
        let before = std::fs::metadata(&destination).unwrap().modified().unwrap();
        assert_eq!(
            ensure_eva_clip_vision_safetensors(&paths).unwrap(),
            destination
        );
        assert_eq!(
            std::fs::metadata(&destination).unwrap().modified().unwrap(),
            before,
            "an authentic artifact was reconverted"
        );

        // Tamper, and forge the sidecar exactly as an attacker would.
        let mut tampered = good.clone();
        *tampered.last_mut().unwrap() ^= 0xff;
        std::fs::write(&destination, &tampered).unwrap();
        write_sidecar(&destination, DERIVED_SHA256).unwrap();
        assert!(!derived_artifact_is_authentic(&destination));

        assert_eq!(
            ensure_eva_clip_vision_safetensors(&paths).unwrap(),
            destination
        );
        assert_eq!(
            std::fs::read(&destination).unwrap(),
            good,
            "the tampered artifact was not restored"
        );
    }
}
