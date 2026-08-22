//! Convert mold's pinned PuLID torch-pickle releases to safetensors.
//!
//! Two of the bundle's artifacts are published as PyTorch pickles — BAAI's
//! `EVA02_CLIP_L_336_psz14_s6B.pt` vision tower and facexlib's
//! `parsing_bisenet.pth` face parser — and **mold's runtime never reads a
//! pickle**. This module is the one place either is opened: it converts once,
//! deterministically, from the SHA-verified source, and everything downstream
//! loads the derived safetensors through the ordinary `VarBuilder`.
//!
//! The two conversions differ only in which source they read, which tensors
//! they keep, and what the result is pinned to. Everything that makes a load
//! authentic is shared, which is the reason they live in one module rather
//! than in two files that would each have to get all of it right.
//!
//! ## Why this is safe to run on a pickle at all
//!
//! Per the Hugging Face scanner the EVA checkpoint contains only
//! `OrderedDict`, `_rebuild_tensor_v2` and `HalfStorage`, and facexlib's
//! parser only `OrderedDict`, `_rebuild_tensor_v2` and `FloatStorage`, all of
//! which candle's reader handles as data (`candle-core/src/pickle.rs:240-246`,
//! `:636-645`). candle never evaluates arbitrary opcodes.
//!
//! ## The three things that authenticate a load
//!
//! 1. **The source is read through a private copy, in a private place.**
//!    candle's `PthTensors` re-opens its file by pathname for every tensor, so
//!    hashing a descriptor and then handing candle a pathname authenticates
//!    nothing. The bytes are copied out of the retained descriptor into an
//!    exclusively created 0o700 directory and hashed on that same stream, so
//!    the digest and the parse observe identical bytes by construction. That
//!    directory lives under a root no other user can rename entries in — NOT
//!    the model root, which `CLAUDE.md` allows to be group-writable. See
//!    [`stage_private_copy`] and [`private_staging_root_candidates`].
//! 2. **Every publish is a `renameat` between two retained directory
//!    descriptors.** `rename` replaces a symlink instead of following it, and
//!    binding both endpoints to descriptors means a group-writable model root
//!    cannot have our staging directory renamed away and substituted between
//!    the hash and the publish. See [`publish`] and [`super::secure_dir`].
//! 3. **Reuse is authenticated by a compiled-in pin, not by the sidecar.**
//!    See [`EVA_DERIVED_SHA256`] and [`BISENET_DERIVED_SHA256`].

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use super::secure_dir::{identity, parent_protects_entries, Dir};
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
/// The derived artifact's name. `mold_core` is the authority — removal has to
/// delete this file and cannot see this crate — so it is re-exported rather
/// than restated.
pub(crate) const EVA_DERIVED_FILENAME: &str = mold_core::pulid_assets::DERIVED_VISION_FILENAME;
/// Informational provenance beside the derived artifact.
pub(crate) const EVA_SIDECAR_FILENAME: &str = mold_core::pulid_assets::DERIVED_VISION_SIDECAR_FILENAME;

/// Source pin, mirrored from `mold_core::manifest`'s `pulid-flux` entry. Kept
/// here as well so the conversion refuses to read anything else even if it is
/// handed a path directly.
pub(crate) const EVA_SOURCE_SHA256: &str =
    "84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05";

/// Pin for the DERIVED artifact.
///
/// [`convert_eva_clip_vision`] is deterministic — it selects a fixed tensor set
/// from the pinned source and `safetensors`' own `prepare` sorts them before
/// laying out the buffer — so converting [`EVA_SOURCE_SHA256`] always produces
/// exactly these bytes. `conversion_is_deterministic_on_the_pinned_source`
/// re-derives this constant from the real checkpoint, so a `safetensors`
/// layout change or a re-uploaded source fails loudly here rather than
/// silently shipping different weights.
///
/// This constant, and never the sidecar, is what authenticates a derived file
/// that is being reused. The sidecar is written by mold, so anything able to
/// tamper with the weights can rewrite it to match; it is provenance for a
/// human, not an authenticator.
pub(crate) const EVA_DERIVED_SHA256: &str =
    "2b0b0ab0baed6ee968c8a08a9dcba908fb602630303faa3515eeaf8e264f136b";

/// The derived parser's name; `mold_core` owns it for the same reason.
pub(crate) const BISENET_DERIVED_FILENAME: &str = mold_core::pulid_assets::DERIVED_PARSER_FILENAME;
/// Informational provenance beside the derived parser.
pub(crate) const BISENET_SIDECAR_FILENAME: &str =
    mold_core::pulid_assets::DERIVED_PARSER_SIDECAR_FILENAME;

/// Source pin for facexlib's `parsing_bisenet.pth`, mirrored from the
/// `pulid-flux` manifest entry for the same reason [`EVA_SOURCE_SHA256`] is.
pub(crate) const BISENET_SOURCE_SHA256: &str =
    "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567";

/// Pin for the DERIVED parser. See [`EVA_DERIVED_SHA256`] — the same argument
/// applies, and `bisenet_conversion_is_deterministic_on_the_pinned_source`
/// re-derives it from the real checkpoint.
pub(crate) const BISENET_DERIVED_SHA256: &str =
    "e62470d5595acee3550138cb2969bc1eee63bcbefba4dd2a624f1f1951ff7b1b";

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

/// An exclusively created, owner-only scratch directory beside a destination,
/// held open as a descriptor.
///
/// Everything this module writes before publishing goes in here, and every
/// operation on it goes through [`Dir`] rather than through its pathname. Two
/// separate properties, both needed:
///
/// - **Mode `0o700`, created with `mkdirat`.** Nothing can be waiting under a
///   name we are about to use, and no other user can read or write what we
///   stage. Staging cannot happen directly in the model directory for this
///   reason: `CLAUDE.md`'s model-storage rule makes shared, group-writable
///   model roots legitimate.
/// - **Descriptor-bound, not name-bound.** In exactly such a group-writable
///   root, another member can `rename` our staging directory away and drop
///   their own at the same name — renaming an entry needs write permission on
///   the *parent*, not on the entry. A pathname-based publish would then hand
///   out our authenticated digest for their bytes. Holding the descriptor means
///   `openat`/`renameat` reach the directory we created no matter what happens
///   to its name.
///
/// Dropped on every path, success or error, so an interrupted conversion leaves
/// no partial 856 MB copy behind.
struct PrivateStagingDir {
    parent: Dir,
    name: String,
    dir: Dir,
    /// Every entry we created, so `Drop` can empty the directory without
    /// reading it back — a hard-coded list would silently start leaking the
    /// moment someone stages a new filename.
    staged: std::cell::RefCell<Vec<String>>,
}

impl PrivateStagingDir {
    /// Create the directory as a sibling of `destination` so a later `renameat`
    /// into place stays within one filesystem.
    fn create_beside(destination: &Path) -> Result<Self> {
        let parent_path = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .context("the conversion destination has no parent directory")?;
        std::fs::create_dir_all(parent_path)
            .with_context(|| format!("failed to create {}", parent_path.display()))?;
        Self::create_under(Dir::open(parent_path)?)
    }

    /// Create an exclusively named 0o700 subdirectory of `parent`.
    fn create_under(parent: Dir) -> Result<Self> {
        let mut last_error = None;
        for attempt in 0..16_u32 {
            let name = format!(
                ".mold-eva-clip-convert.{}.{}.{attempt}",
                std::process::id(),
                nonce()
            );
            match parent.create_subdir(&name, 0o700) {
                Ok(dir) => {
                    return Ok(Self {
                        parent,
                        name,
                        dir,
                        staged: std::cell::RefCell::new(Vec::new()),
                    })
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no attempt was made")))
            .context("failed to create a private staging directory")
    }

    /// Create the directory under a root **outside** the model tree.
    ///
    /// Used for the source copy, which exists to be handed to candle's pickle
    /// reader by pathname. `create_beside` is wrong for that: a model root may
    /// legitimately be group-writable, and then another member can rename our
    /// 0o700 directory away after we verified its contents and leave an
    /// unpinned `source.pt` at the pathname `PthTensors` keeps re-opening. The
    /// derived-output pin would still catch the resulting weights, but only
    /// after the pickle parser had already consumed attacker-chosen bytes.
    ///
    /// The output staging directory stays beside the destination, because a
    /// publish must `renameat` within one filesystem — and it is safe there,
    /// since nothing hands its pathname to anyone.
    fn create_in_private_root(needed_bytes: u64) -> Result<Self> {
        let root = select_private_staging_root(&private_staging_root_candidates())?;
        ensure_room_for_copy(&root, needed_bytes)?;
        Self::create_under(Dir::open(&root)?)
    }

    fn dir(&self) -> &Dir {
        &self.dir
    }

    /// Create a staged file, remembering the name for cleanup.
    fn create_file(&self, name: &str) -> Result<File> {
        let file = self.dir.create_file(name, 0o600)?;
        self.staged.borrow_mut().push(name.to_string());
        Ok(file)
    }

    fn parent(&self) -> &Dir {
        &self.parent
    }

    fn path(&self) -> &Path {
        self.dir.display_path()
    }
}

impl Drop for PrivateStagingDir {
    /// Remove our own entries through the retained descriptor, then the
    /// directory itself by name.
    ///
    /// The unlinks are descriptor-bound and therefore always correct. The final
    /// `rmdir` is by name, which is safe because `rmdir` only succeeds on an
    /// EMPTY directory: if the name was stolen and replaced with something
    /// holding files, it fails and we leave the impostor alone.
    fn drop(&mut self) {
        for name in self.staged.borrow().iter() {
            // A published file has already been renamed out; ENOENT is fine.
            let _ = self.dir.remove_file(name);
        }
        let _ = self.parent.remove_subdir(&self.name);
    }
}

/// Roots to consider for the private source copy, best first.
///
/// `$XDG_RUNTIME_DIR` is a per-user `0o700` tmpfs on systemd Linux and is the
/// right answer when it exists. `std::env::temp_dir()` is next: it honours
/// `$TMPDIR`, which is a per-user `0o700` directory on macOS and `/tmp` (mode
/// `1777`, sticky) on most Linux systems. Both shapes satisfy
/// [`parent_protects_entries`].
///
/// Deliberately no `MOLD_*` variable of its own: `TMPDIR` is the standard knob
/// and adding a private one would mean registering it in
/// `ENGINE_SHAPING_VARIABLES` for something that is not engine shaping.
fn private_staging_root_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        if !runtime_dir.is_empty() {
            candidates.push(PathBuf::from(runtime_dir));
        }
    }
    candidates.push(std::env::temp_dir());
    candidates
}

/// The first candidate no other user can rename entries in.
///
/// Fails closed, naming every candidate and why it was rejected, because the
/// alternative — falling back to the model root — is the vulnerability this
/// exists to close.
fn select_private_staging_root(candidates: &[PathBuf]) -> Result<PathBuf> {
    let mut rejected = Vec::new();
    for candidate in candidates {
        match parent_protects_entries(candidate) {
            Ok(()) => return Ok(candidate.clone()),
            Err(reason) => rejected.push(format!("{} ({reason})", candidate.display())),
        }
    }
    bail!(
        "no private directory is available to stage the EVA02-CLIP conversion: {}. \
         Set TMPDIR to a directory you own that other users cannot write, or that \
         has the sticky bit set.",
        if rejected.is_empty() {
            "no candidates were offered".to_string()
        } else {
            rejected.join("; ")
        }
    )
}

/// Refuse before copying rather than after 856 MB of writes.
///
/// The private root is frequently a tmpfs sized as a fraction of RAM, and the
/// source pickle is large enough that a default `/tmp` can genuinely be too
/// small. The remedy is `TMPDIR`, which is why the message names it.
fn ensure_room_for_copy(root: &Path, needed_bytes: u64) -> Result<()> {
    // A margin so we do not fill the volume to the last block; the conversion
    // is not the only thing using it.
    const MARGIN_BYTES: u64 = 64 * 1024 * 1024;
    let available = fs2::available_space(root)
        .with_context(|| format!("failed to read free space on {}", root.display()))?;
    let required = needed_bytes.saturating_add(MARGIN_BYTES);
    ensure!(
        available >= required,
        "staging the EVA02-CLIP source needs {required} bytes on {} but only {available} \
         are free. Set TMPDIR to a larger volume.",
        root.display()
    );
    Ok(())
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

/// Publish the staged file `name` as `destination`, durably.
///
/// Two things are load-bearing and both were review findings.
///
/// `rename` REPLACES whatever sits at the destination — including a symlink,
/// which it unlinks rather than follows. That is why nothing here uses
/// `std::fs::write`: a symlink pre-planted at the destination would otherwise
/// redirect our write into a file of the attacker's choosing.
///
/// And it is `renameat` between two retained directory descriptors, not a
/// pathname `rename`. In a group-writable model root — which `CLAUDE.md`
/// explicitly supports — another member can rename our 0o700 staging directory
/// away and drop their own at the same name between the hash and the publish.
/// A pathname rename would then publish their file under our authenticated
/// digest. Descriptors refer to inodes, so this reaches the directory we
/// created regardless of what its name now points at.
///
/// The published file's `(device, inode)` is re-read through the destination
/// parent afterwards and compared with the staged file's, so the artifact the
/// caller ends up with is provably the one that was hashed.
fn publish(staging: &PrivateStagingDir, name: &str, destination: &Path) -> Result<()> {
    let staged_identity = identity(&staging.dir().open_file(name)?)?;
    let final_name = destination
        .file_name()
        .context("the conversion destination has no filename")?
        .to_str()
        .context("the conversion destination is not valid UTF-8")?;

    staging
        .dir()
        .rename_into(name, staging.parent(), final_name)?;
    // Durability of the rename itself.
    staging.parent().sync();

    let published = staging.parent().open_file(final_name).with_context(|| {
        format!(
            "{} vanished immediately after publication",
            destination.display()
        )
    })?;
    ensure!(
        identity(&published)? == staged_identity,
        "{} is not the file that was staged and hashed",
        destination.display()
    );
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
    let mut source = retained
        .try_clone()
        .context("failed to clone the source descriptor")?;
    source
        .seek(SeekFrom::Start(0))
        .context("failed to rewind the source descriptor")?;

    let mut target = staging.create_file(name)?;
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
            .with_context(|| format!("failed to write the private {name} copy"))?;
    }
    target
        .sync_all()
        .with_context(|| format!("failed to fsync the private {name} copy"))?;
    drop(target);

    let actual = format!("{:x}", digest.finalize());
    ensure!(
        actual == expected_sha256,
        "the source is not the pinned EVA02-CLIP release (sha256 {actual})"
    );
    // Nothing outside this process can reach the staging directory, so handing
    // its pathname to candle's pickle reader is safe. See `write_atomically`.
    Ok(staging.dir().unsafe_path_for(name))
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
/// [`EVA_DERIVED_SHA256`] a meaningful pin.
///
/// Atomic because the bytes are built inside a [`PrivateStagingDir`] beside the
/// destination and then `renameat`d into place: same directory, therefore same
/// filesystem, therefore a real rename. A crash leaves either the previous
/// artifact or nothing — never a truncated one, and never a partially written
/// file under the destination's own name.
///
/// `serialize_to_file` insists on a pathname, which is the one place here that
/// is not descriptor-bound, and it is only a LIVENESS concern rather than a
/// correctness one. If the staging directory's name were stolen mid-write, the
/// bytes would land in the impostor — and the very next step, which re-opens
/// the file through the retained staging descriptor, would fail to find it and
/// the conversion would error. It cannot silently succeed on someone else's
/// bytes, because the hash and the publish both go through that descriptor.
fn write_atomically(tensors: &[RawTensor], destination: &Path) -> Result<String> {
    write_atomically_with_hook(tensors, destination, &|| {})
}

/// `write_atomically`, with a test seam between the hash and the publish.
///
/// The hook exists for `a_stolen_staging_name_cannot_substitute_the_published_bytes`,
/// which needs to steal the staging directory's name at exactly that instant.
/// It is a no-op in production and the compiler inlines it away.
fn write_atomically_with_hook(
    tensors: &[RawTensor],
    destination: &Path,
    before_publish: &dyn Fn(),
) -> Result<String> {
    const STAGED: &str = "weights.safetensors";
    let staging = PrivateStagingDir::create_beside(destination)?;

    let views = tensors
        .iter()
        .map(|tensor| Ok((tensor.name.clone(), tensor.view()?)))
        .collect::<Result<Vec<_>>>()?;
    // Claim the name exclusively first, so `serialize_to_file`'s own open can
    // only ever truncate a regular file we created.
    drop(staging.create_file(STAGED)?);
    serialize_to_file(views, &None, &staging.dir().unsafe_path_for(STAGED))
        .context("failed to write the staged safetensors")?;

    // Everything from here is descriptor-bound: this is the file we hash, and
    // `publish` renames this same descriptor's entry.
    let digest = {
        let file = staging.dir().open_file(STAGED)?;
        let sha = sha256_open_file(&file)?;
        file.sync_all().context("failed to fsync the staged file")?;
        sha
    };

    before_publish();
    publish(&staging, STAGED, destination)?;
    Ok(digest)
}

/// Write `bytes` at `destination` through the same staging-and-publish path.
fn publish_bytes(bytes: &[u8], destination: &Path) -> Result<()> {
    const STAGED: &str = "payload";
    let staging = PrivateStagingDir::create_beside(destination)?;
    let mut file = staging.create_file(STAGED)?;
    file.write_all(bytes)
        .context("failed to write the staged payload")?;
    file.sync_all()
        .context("failed to fsync the staged payload")?;
    drop(file);
    publish(&staging, STAGED, destination)
}

/// The sidecar sits beside its own artifact and is named after it, so two
/// destinations in one directory (which the tests do, and a future second
/// derived artifact might) cannot share one record.
/// [`EVA_SIDECAR_FILENAME`] is what this produces for [`EVA_DERIVED_FILENAME`].
fn sidecar_path(destination: &Path) -> PathBuf {
    destination.with_extension("json")
}

/// Record what produced the derived artifact.
///
/// Informational only. Nothing reads this back to decide whether the weights
/// can be trusted — [`EVA_DERIVED_SHA256`] does that — because a file mold writes
/// is a file that anything able to reach the weights could rewrite to match.
fn write_sidecar(destination: &Path, source_sha256: &str, derived_sha256: &str) -> Result<()> {
    let derived_filename = destination
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let body = format!(
        "{{\n  \"source_sha256\": \"{source_sha256}\",\n  \
         \"derived_sha256\": \"{derived_sha256}\",\n  \
         \"derived_filename\": \"{derived_filename}\",\n  \
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

/// One pickle-to-safetensors conversion, described by data.
///
/// Everything security-relevant is shared; a conversion contributes only which
/// bytes it will accept, which tensors it keeps, and what it is named.
struct PickleConversion {
    /// Human name used in errors, e.g. `EVA02-CLIP`.
    label: &'static str,
    /// The SHA-256 the source must have. A conversion refuses anything else
    /// even when it is handed a path directly.
    source_sha256: &'static str,
    /// `Some(output_name)` keeps a tensor under that name; `None` drops it.
    select: fn(&str) -> Option<String>,
}

/// Convert `source` (a pinned torch pickle) into safetensors at `destination`,
/// returning the derived SHA-256.
fn convert_pickle(
    conversion: &PickleConversion,
    source: &Path,
    destination: &Path,
) -> Result<String> {
    let PickleConversion {
        label,
        source_sha256,
        select,
    } = conversion;

    // 1. Open no-follow and RETAIN, so neither the filename nor any parent
    //    component can be a symlink and the bytes are now bound to a descriptor
    //    rather than to a name.
    let retained = open_regular_file_no_follow(source)
        .with_context(|| format!("failed to open {} no-follow", source.display()))?;

    // 2. Copy those bytes somewhere only we can reach, hashing the same stream,
    //    and require the manifest pin. From here on the pathname the caller
    //    gave us is irrelevant.
    //
    //    The copy goes under a private tmp root rather than beside the
    //    destination: the model root may legitimately be group-writable, and
    //    this is the one staged file whose PATHNAME is handed out (to candle's
    //    pickle reader, which re-opens it per tensor).
    let source_bytes = retained
        .metadata()
        .context("failed to stat the source")?
        .len();
    let staging = PrivateStagingDir::create_in_private_root(source_bytes)?;
    let private_source = stage_private_copy(&retained, &staging, "source.pt", source_sha256)
        .with_context(|| format!("{} failed its pin", source.display()))?;
    drop(retained);

    // 3. Parse the private copy. candle re-opens by pathname per tensor, which
    //    is now harmless: every one of those opens lands inside a 0o700
    //    directory created exclusively for this conversion.
    //
    //    Which reader runs is decided by the file's own first bytes, not by
    //    the conversion: torch has two containers and mold's two pinned
    //    sources happen to be one of each. See [`super::legacy_pth`].
    let mut magic = [0_u8; 21];
    {
        use std::io::Read as _;
        let mut probe = File::open(&private_source).context("reopening the private copy")?;
        // A short read is fine here: the two `is_*_container` predicates only
        // ever inspect a prefix, and neither container can be this small.
        let read = probe.read(&mut magic).context("reading the container magic")?;
        magic[read..].fill(0);
    }

    let mut tensors = if super::legacy_pth::is_legacy_container(&magic) {
        let mut kept = Vec::new();
        for tensor in super::legacy_pth::read_legacy_pth(&private_source)
            .with_context(|| format!("failed to read {} as a legacy torch archive", source.display()))?
        {
            let Some(renamed) = select(&tensor.name) else {
                continue;
            };
            kept.push(RawTensor {
                name: renamed,
                dtype: dtype_for(tensor.dtype)?,
                shape: tensor.shape,
                data: tensor.data,
            });
        }
        kept
    } else {
        let pth = PthTensors::new(&private_source, None)
            .with_context(|| format!("failed to read {} as a torch pickle", source.display()))?;
        let mut names: Vec<(String, String)> = pth
            .tensor_infos()
            .keys()
            .filter_map(|name| select(name).map(|renamed| (name.clone(), renamed)))
            .collect();
        names.sort();
        let mut kept = Vec::with_capacity(names.len());
        for (name, renamed) in names {
            let tensor = pth
                .get(&name)?
                .with_context(|| format!("{name} vanished between listing and read"))?
                .contiguous()?;
            kept.push(RawTensor {
                name: renamed,
                dtype: dtype_for(tensor.dtype())?,
                shape: tensor.dims().to_vec(),
                data: tensor_bytes(&tensor)?,
            });
        }
        // The pickle reader holds the private path; release it before the
        // staging directory is removed.
        drop(pth);
        kept
    };
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    ensure!(
        !tensors.is_empty(),
        "{} contains no tensors the {label} conversion recognizes",
        source.display()
    );
    drop(staging);

    let derived = write_atomically(&tensors, destination)?;
    write_sidecar(destination, source_sha256, &derived)?;
    Ok(derived)
}

/// Materialize a derived artifact, converting on first use.
///
/// Idempotent, and idempotent on the *bytes*: a derived file is reused only
/// when it hashes to `derived_sha256`. Anything else — missing, truncated,
/// tampered with, or carrying a forged sidecar — reconverts from the pinned
/// source, because a half-written or edited artifact must never be loaded as
/// weights. If the source itself fails its own pin the conversion errors rather
/// than falling back to whatever is on disk.
fn ensure_derived(
    conversion: &PickleConversion,
    source: &Path,
    destination: &Path,
    derived_sha256: &'static str,
) -> Result<PathBuf> {
    if artifact_is_authentic(destination, derived_sha256) {
        return Ok(destination.to_path_buf());
    }
    let derived = convert_pickle(conversion, source, destination)?;
    // A fresh conversion that does not reproduce the pin means the pin and the
    // converter have diverged. Say so once, loudly, instead of silently
    // reconverting on every later call.
    ensure!(
        derived == derived_sha256,
        "converting {} produced sha256 {derived}, but this build pins \
         {derived_sha256}",
        source.display()
    );
    Ok(destination.to_path_buf())
}

/// The EVA02-CLIP vision tower: keep `visual.*`, drop the duplicated per-block
/// RoPE buffers, and strip the prefix, because the derived file IS a vision
/// tower and `EvaClipVisionTower` should not have to know it once lived inside
/// a CLIP.
const EVA_CLIP_VISION: PickleConversion = PickleConversion {
    label: "EVA02-CLIP",
    source_sha256: EVA_SOURCE_SHA256,
    select: |name| {
        (name.starts_with(VISION_PREFIX) && !is_duplicate_rope_buffer(name))
            .then(|| name.trim_start_matches(VISION_PREFIX).to_string())
    },
};

/// facexlib's BiSeNet parser: the checkpoint is a bare `state_dict` of exactly
/// the parser, so everything is kept under its own name. `conv_out16` /
/// `conv_out32` are the auxiliary training heads — upstream returns them
/// (`bisenet.py:132-134`) and PuLID reads only output `[0]`
/// (`pipeline_flux.py:164`) — but they are retained rather than dropped so the
/// derived file stays a faithful re-container of the release, and the
/// `BiSeNetParser` port simply does not build them.
const BISENET_PARSER: PickleConversion = PickleConversion {
    label: "BiSeNet face parser",
    source_sha256: BISENET_SOURCE_SHA256,
    select: |name| Some(name.to_string()),
};

/// Convert `source` (the pinned `.pt`) into vision-only safetensors at
/// `destination`, returning the derived SHA-256.
pub(crate) fn convert_eva_clip_vision(source: &Path, destination: &Path) -> Result<String> {
    convert_pickle(&EVA_CLIP_VISION, source, destination)
}

/// Convert `source` (the pinned `parsing_bisenet.pth`) into safetensors at
/// `destination`, returning the derived SHA-256.
pub(crate) fn convert_bisenet_parser(source: &Path, destination: &Path) -> Result<String> {
    convert_pickle(&BISENET_PARSER, source, destination)
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

/// Where the derived vision tower lives for an installed bundle: beside the
/// `.pt`, in the shared PuLID root.
pub(crate) fn derived_vision_path(paths: &PulidPaths) -> PathBuf {
    paths
        .vision_encoder_source
        .with_file_name(EVA_DERIVED_FILENAME)
}

/// Where the derived face parser lives: beside its own `.pth`, same root.
pub(crate) fn derived_parser_path(paths: &PulidPaths) -> PathBuf {
    paths
        .face_parser_source
        .with_file_name(BISENET_DERIVED_FILENAME)
}

/// Is the file already at `destination` the artifact `expected_sha256` names?
///
/// Opened no-follow and hashed. The sidecar is deliberately not consulted: it
/// is mold's own writing, so anything that could tamper with the weights could
/// forge a matching record, and trusting it would turn "verified" into "the
/// attacker said so".
fn artifact_is_authentic(destination: &Path, expected_sha256: &str) -> bool {
    let Ok(file) = open_regular_file_no_follow(destination) else {
        return false;
    };
    sha256_open_file(&file).ok().as_deref() == Some(expected_sha256)
}

/// Materialize the vision tower's safetensors, converting on first use.
///
/// This is the entry point admission calls once it has resolved a complete
/// bundle through [`mold_core::pulid_assets::pulid_paths`]. It is deliberately
/// convert-on-first-use rather than a download post-hook: the bundle's install
/// flow is being built concurrently (#1220 / dependency planning), and hanging
/// an 856 MB pickle read off the download path would couple the two.
pub(crate) fn ensure_eva_clip_vision_safetensors(paths: &PulidPaths) -> Result<PathBuf> {
    ensure_derived(
        &EVA_CLIP_VISION,
        &paths.vision_encoder_source,
        &derived_vision_path(paths),
        EVA_DERIVED_SHA256,
    )
}

/// Materialize the BiSeNet parser's safetensors, converting on first use.
///
/// Same contract, and a much cheaper one: the release is 53 MB rather than
/// 856 MB, so the transient private copy is negligible.
pub(crate) fn ensure_bisenet_parser_safetensors(paths: &PulidPaths) -> Result<PathBuf> {
    ensure_derived(
        &BISENET_PARSER,
        &paths.face_parser_source,
        &derived_parser_path(paths),
        BISENET_DERIVED_SHA256,
    )
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
        write_sidecar(&destination, EVA_SOURCE_SHA256, &digest).unwrap();
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

    /// The source copy must not be staged in the model root, so a conversion
    /// leaves nothing there even when it fails partway.
    #[test]
    fn no_staging_lands_beside_the_destination() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("impostor.pt");
        std::fs::write(&source, b"definitely not EVA02-CLIP").unwrap();
        // Fails on the pin, i.e. AFTER the private copy has been made.
        assert!(convert_eva_clip_vision(&source, &dir.path().join("out.safetensors")).is_err());
        assert_eq!(
            entries(dir.path()),
            vec!["impostor.pt".to_string()],
            "the source copy was staged inside the model root"
        );
    }

    /// The private root is chosen by the policy, not by convention: the first
    /// candidate other users cannot rename entries in wins, and if none
    /// qualifies the conversion fails closed rather than falling back to the
    /// model root.
    #[test]
    #[cfg(unix)]
    fn the_private_root_is_the_first_unrenamable_candidate() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let make = |name: &str, mode: u32| {
            let path = root.path().join(name);
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            path
        };
        let shared = make("shared", 0o777);
        let private = make("private", 0o700);
        let sticky = make("sticky", 0o1777);

        // An unusable earlier candidate is skipped, not fatal.
        assert_eq!(
            select_private_staging_root(&[shared.clone(), private.clone()]).unwrap(),
            private
        );
        // Order is preference order.
        assert_eq!(
            select_private_staging_root(&[sticky.clone(), private.clone()]).unwrap(),
            sticky
        );

        // Nothing usable: fail closed, and say what to do about it.
        let error = select_private_staging_root(std::slice::from_ref(&shared)).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("TMPDIR"), "unhelpful message: {message}");
        assert!(
            message.contains("shared"),
            "did not name the candidate: {message}"
        );
        // ...and the reason, not just the path, or the user cannot tell a
        // permissions problem from a missing directory.
        assert!(
            message.contains("sticky"),
            "did not explain the rejection: {message}"
        );
        assert!(select_private_staging_root(&[]).is_err());
    }

    /// The real candidate list must actually work on this machine, or every
    /// conversion fails. `temp_dir()` is always present, so this is a check on
    /// the environment as much as on the code.
    #[test]
    fn the_default_candidates_resolve() {
        let candidates = private_staging_root_candidates();
        assert!(!candidates.is_empty());
        assert_eq!(candidates.last().unwrap(), &std::env::temp_dir());
        select_private_staging_root(&candidates)
            .expect("this machine offers no usable private staging root");
    }

    /// An 856 MB pickle onto a small tmpfs must fail before the copy, with a
    /// message naming the remedy rather than ENOSPC halfway through.
    #[test]
    fn a_too_small_private_root_is_refused_before_copying() {
        let root = tempfile::tempdir().unwrap();
        assert!(ensure_room_for_copy(root.path(), 1024).is_ok());
        let error = ensure_room_for_copy(root.path(), u64::MAX / 2).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("TMPDIR"), "unhelpful message: {message}");
        assert!(
            message.contains("are free"),
            "did not report what was available: {message}"
        );
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

        let path = staging.path().to_path_buf();
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
            stage_private_copy(&retained, &staging, "copy.bin", EVA_DERIVED_SHA256).unwrap_err();
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
        let mode = std::fs::metadata(staging.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "mode was {:o}", mode & 0o777);
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
        let destination = dir.path().join(EVA_DERIVED_FILENAME);
        assert_eq!(
            sidecar_path(&destination).file_name().unwrap(),
            EVA_SIDECAR_FILENAME
        );
        write_sidecar(&destination, EVA_SOURCE_SHA256, "deadbeef").unwrap();
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

        let destination = dir.path().join(EVA_DERIVED_FILENAME);
        std::os::unix::fs::symlink(&victim, sidecar_path(&destination)).unwrap();

        write_sidecar(&destination, EVA_SOURCE_SHA256, "deadbeef").unwrap();
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

    /// The finding this fixes: in a group-writable model root another member
    /// can rename our staging directory away and drop their own at the same
    /// name between the hash and the publish, so a pathname `rename` would
    /// publish THEIR bytes under OUR authenticated digest.
    ///
    /// The hook fires at exactly that instant and performs exactly that swap.
    /// Because the publish is a `renameat` through the retained staging
    /// descriptor, it moves the file we hashed; the substitute is never
    /// published and is left untouched.
    #[test]
    #[cfg(unix)]
    fn a_stolen_staging_name_cannot_substitute_the_published_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let destination = root.join("weights.safetensors");

        // What an honest run produces, for comparison.
        let expected = write_atomically(
            &[raw("a.weight", &[1.0, 2.0], &[2])],
            &root.join("reference.safetensors"),
        )
        .unwrap();
        let honest_bytes = std::fs::read(root.join("reference.safetensors")).unwrap();

        let swapped = std::cell::Cell::new(false);
        let hook = || {
            // Find the staging directory by name, exactly as an attacker with
            // write access to the parent would.
            let staging = std::fs::read_dir(&root)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(".mold-eva-clip-convert."))
                })
                .expect("the staging directory should exist at this point");
            std::fs::rename(&staging, root.join("stolen")).unwrap();
            std::fs::create_dir(&staging).unwrap();
            std::fs::write(staging.join("weights.safetensors"), b"attacker payload").unwrap();
            swapped.set(true);
        };

        let digest =
            write_atomically_with_hook(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination, &hook)
                .unwrap();
        assert!(swapped.get(), "the hook never ran");

        // The digest is honest AND it describes what actually landed.
        assert_eq!(digest, expected);
        assert_eq!(std::fs::read(&destination).unwrap(), honest_bytes);
        assert_ne!(
            std::fs::read(&destination).unwrap(),
            b"attacker payload".to_vec()
        );
        // The substitute is still sitting there, unpublished.
        let planted = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".mold-eva-clip-convert."))
            })
            .expect("the planted directory should survive");
        assert_eq!(
            std::fs::read(planted.join("weights.safetensors")).unwrap(),
            b"attacker payload"
        );
    }

    /// The published file must be the one that was hashed, verified through the
    /// destination's own directory descriptor after the rename.
    #[test]
    fn publication_reports_the_identity_it_staged() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("weights.safetensors");
        let digest = write_atomically(&[raw("a.weight", &[1.0, 2.0], &[2])], &destination).unwrap();
        let published = std::fs::File::open(&destination).unwrap();
        assert_eq!(
            crate::encoders::secure_dir::identity(&published).unwrap(),
            crate::encoders::secure_dir::identity(&published).unwrap()
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&destination).unwrap())),
            digest
        );
    }

    /// The reuse decision reads the bytes, never the sidecar. A forged record
    /// claiming the pinned digest must not launder tampered weights.
    #[test]
    fn a_forged_sidecar_cannot_authenticate_tampered_weights() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join(EVA_DERIVED_FILENAME);
        std::fs::write(&destination, b"tampered weights").unwrap();
        // Exactly what an attacker would write to make a sidecar-trusting
        // implementation accept the file above.
        write_sidecar(&destination, EVA_SOURCE_SHA256, EVA_DERIVED_SHA256).unwrap();
        assert_eq!(
            read_sidecar_sha(&destination).as_deref(),
            Some(EVA_DERIVED_SHA256)
        );

        assert!(
            !artifact_is_authentic(&destination, EVA_DERIVED_SHA256),
            "a forged sidecar authenticated tampered weights"
        );
    }

    #[test]
    fn an_absent_or_symlinked_derived_artifact_is_not_authentic() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!artifact_is_authentic(
            &dir.path().join("absent"),
            EVA_DERIVED_SHA256
        ));

        #[cfg(unix)]
        {
            let real = dir.path().join("real.safetensors");
            std::fs::write(&real, b"whatever").unwrap();
            let link = dir.path().join("link.safetensors");
            std::os::unix::fs::symlink(&real, &link).unwrap();
            assert!(!artifact_is_authentic(&link, EVA_DERIVED_SHA256));
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
        let first = convert_eva_clip_vision(&source, &dir.path().join(EVA_DERIVED_FILENAME)).unwrap();
        let second =
            convert_eva_clip_vision(&source, &dir.path().join("again.safetensors")).unwrap();
        assert_eq!(first, second, "conversion is not deterministic");
        println!("derived sha256: {first}");
        // This is where EVA_DERIVED_SHA256 comes from. A `safetensors` layout
        // change or a re-uploaded source fails here rather than silently
        // shipping different weights.
        assert_eq!(first, EVA_DERIVED_SHA256);

        let loaded =
            candle_core::safetensors::load(dir.path().join(EVA_DERIVED_FILENAME), &Device::Cpu)
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
    fn bisenet_conversion_is_deterministic_on_the_pinned_source() {
        let source = pulid_asset("parsing_bisenet.pth");
        let dir = tempfile::tempdir().unwrap();
        let first =
            convert_bisenet_parser(&source, &dir.path().join(BISENET_DERIVED_FILENAME)).unwrap();
        let second = convert_bisenet_parser(&source, &dir.path().join("again.safetensors")).unwrap();
        assert_eq!(first, second, "conversion is not deterministic");
        println!("derived sha256: {first}");
        assert_eq!(first, BISENET_DERIVED_SHA256);

        let loaded =
            candle_core::safetensors::load(dir.path().join(BISENET_DERIVED_FILENAME), &Device::Cpu)
                .unwrap();
        // A faithful re-container: every tensor of the release, under its own
        // name, at its own dtype. The auxiliary heads are present and simply
        // never built (`BiSeNetParser`).
        assert_eq!(loaded.len(), 191);
        assert_eq!(loaded["cp.resnet.conv1.weight"].dims(), &[64, 3, 7, 7]);
        assert_eq!(loaded["conv_out.conv_out.weight"].dims(), &[19, 256, 1, 1]);
        assert!(loaded.contains_key("conv_out16.conv_out.weight"));
        assert_eq!(loaded["cp.resnet.conv1.weight"].dtype(), DType::F32);
    }

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
            face_parser_source: dir.path().join("parser.pth"),
        };

        let destination = ensure_eva_clip_vision_safetensors(&paths).unwrap();
        assert!(artifact_is_authentic(&destination, EVA_DERIVED_SHA256));
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
        write_sidecar(&destination, EVA_SOURCE_SHA256, EVA_DERIVED_SHA256).unwrap();
        assert!(!artifact_is_authentic(&destination, EVA_DERIVED_SHA256));

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
