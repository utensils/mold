//! Read a **legacy** (pre-1.6) `torch.save` archive.
//!
//! `candle_core::pickle::PthTensors` reads only the modern container, which is
//! a zip file. facexlib's `parsing_bisenet.pth` — the face parser #1225 needs,
//! published in 2020 — is the older flat format, and there is no newer release
//! of those weights: the only authentic upstream artifact is this one. So mold
//! reads it here rather than pinning somebody's re-save, which would move the
//! provenance from "facexlib published these bytes" to "a stranger says these
//! are facexlib's weights".
//!
//! Only the *container* is new. Every pickle in it is parsed by candle's own
//! `Stack`, and the tensor metadata by candle's own
//! `Object::into_tensor_info`, so the opcode surface mold trusts is exactly
//! the one it already trusts for the EVA02-CLIP checkpoint — candle never
//! evaluates arbitrary opcodes, it interprets a fixed set as data
//! (`candle-core/src/pickle.rs:14-49`).
//!
//! ## The format
//!
//! `torch/serialization.py::_legacy_save` writes, in order:
//!
//! ```text
//! pickle  MAGIC_NUMBER          0x1950a86a20f9469cfc6c
//! pickle  PROTOCOL_VERSION      1001
//! pickle  sys_info              {'protocol_version', 'little_endian', 'type_sizes'}
//! pickle  the object            tensors as persistent ids into storage keys
//! pickle  sorted storage keys   ['94693637921008', ...]
//! raw     per key, in that order: i64 element count, then the elements
//! ```
//!
//! The first two pickles are a fixed 21-byte preamble at protocol 2 and are
//! compared byte for byte rather than parsed. That is not an optimization: the
//! magic is a **ten-byte** integer, and candle's `Long1` arm accumulates into
//! an `i64` with `<< (i * 8)`, which at `i = 8` is a shift past the width of
//! the type — a panic in a debug build. Comparing the bytes both avoids that
//! and is a stricter check than reading the value would be.
//!
//! ## What it refuses
//!
//! Everything it is not sure about, because it is only ever pointed at bytes
//! whose SHA-256 is already pinned: a wrong preamble, a big-endian archive, a
//! non-dict root, a storage no tensor claims (its element size would be a
//! guess), a declared element count that disagrees with the pickle, a
//! non-contiguous view, and a truncated tail.

#![allow(dead_code)]

use anyhow::{bail, ensure, Context, Result};
use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::DType;
use std::collections::BTreeMap;
use std::io::{BufReader, Read};
use std::path::Path;

/// `torch.save`'s protocol-2 pickles of `MAGIC_NUMBER` and
/// `PROTOCOL_VERSION` (`torch/serialization.py`), which every legacy archive
/// opens with verbatim.
const LEGACY_PREAMBLE: [u8; 21] = [
    // PROTO 2, LONG1 len 10, 0x1950a86a20f9469cfc6c little-endian, STOP
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, 0x2e,
    // PROTO 2, BININT2 1001, STOP
    0x80, 0x02, 0x4d, 0xe9, 0x03, 0x2e,
];

/// The zip local-file-header magic every MODERN `torch.save` archive opens
/// with. Used only to tell a caller which container it handed us.
const ZIP_MAGIC: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

/// One tensor lifted out of a legacy archive, as raw little-endian bytes.
///
/// Deliberately not a `candle_core::Tensor`: the only consumer is the
/// safetensors re-container, which wants the bytes it was given rather than a
/// round trip through a device buffer.
pub(crate) struct LegacyTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// Hand-written so a failing assertion prints a tensor's identity rather than
/// several megabytes of its bytes.
impl std::fmt::Debug for LegacyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LegacyTensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// True when `bytes` opens with the modern zip container.
pub(crate) fn is_zip_container(bytes: &[u8]) -> bool {
    bytes.starts_with(&ZIP_MAGIC)
}

/// True when `bytes` opens with the legacy preamble.
pub(crate) fn is_legacy_container(bytes: &[u8]) -> bool {
    bytes.starts_with(&LEGACY_PREAMBLE)
}

/// Read every tensor of a legacy archive, in the archive's own key order.
pub(crate) fn read_legacy_pth(path: &Path) -> Result<Vec<LegacyTensor>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut preamble = [0_u8; LEGACY_PREAMBLE.len()];
    reader
        .read_exact(&mut preamble)
        .context("the file is shorter than a legacy torch preamble")?;
    ensure!(
        preamble == LEGACY_PREAMBLE,
        "{} does not open with torch's legacy magic-number and protocol-version pickles",
        path.display()
    );

    let sys_info = read_pickle(&mut reader).context("reading the sys_info pickle")?;
    ensure_little_endian(&sys_info)?;

    let root = read_pickle(&mut reader).context("reading the object pickle")?;
    let entries = match root {
        Object::Dict(entries) => entries,
        other => bail!("a legacy checkpoint mold reads must be a state_dict, found {other:?}"),
    };

    // `into_tensor_info` renders the storage key as `{dir}/{key}`; the modern
    // reader passes the zip directory, and a legacy archive has none.
    let mut infos: Vec<TensorInfo> = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        if let Some(info) = value
            .into_tensor_info(name, Path::new(""))
            .context("reading a tensor's metadata")?
        {
            infos.push(info);
        }
    }
    ensure!(!infos.is_empty(), "{} carries no tensors", path.display());

    let keys = match read_pickle(&mut reader).context("reading the storage-key pickle")? {
        Object::List(keys) => keys
            .into_iter()
            .map(|key| {
                key.unicode()
                    .map_err(|other| anyhow::anyhow!("a storage key is not a string: {other:?}"))
            })
            .collect::<Result<Vec<String>>>()?,
        other => bail!("the storage-key list must be a list, found {other:?}"),
    };

    // Several tensors may share one storage (a view). Group first, so the
    // single pass over the data section can serve all of them.
    let mut by_storage: BTreeMap<String, Vec<TensorInfo>> = BTreeMap::new();
    for info in infos {
        by_storage
            .entry(storage_key(&info.path))
            .or_default()
            .push(info);
    }

    let mut out = Vec::new();
    for key in keys {
        let claimants = by_storage.remove(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "storage {key} is claimed by no tensor, so its element size would be a guess"
            )
        })?;
        let dtype = claimants[0].dtype;
        ensure!(
            claimants.iter().all(|info| info.dtype == dtype),
            "storage {key} is read as two different dtypes"
        );

        let mut count = [0_u8; 8];
        reader
            .read_exact(&mut count)
            .with_context(|| format!("storage {key} has no element count"))?;
        let count = i64::from_le_bytes(count);
        let count = usize::try_from(count)
            .map_err(|_| anyhow::anyhow!("storage {key} declares {count} elements"))?;
        ensure!(
            count == claimants[0].storage_size,
            "storage {key} declares {count} elements but the pickle says {}",
            claimants[0].storage_size
        );

        let mut storage = vec![0_u8; count * dtype.size_in_bytes()];
        reader
            .read_exact(&mut storage)
            .with_context(|| format!("storage {key} is truncated"))?;

        for info in claimants {
            ensure!(
                info.layout.is_contiguous(),
                "{} is a non-contiguous view, which mold does not re-container",
                info.name
            );
            // `rebuild_args` already multiplied the element offset by the
            // element size (`pickle.rs:637`), so this is a byte offset.
            let start = info.layout.start_offset();
            let len = info.layout.shape().elem_count() * dtype.size_in_bytes();
            let end = start
                .checked_add(len)
                .context("a tensor view overflows its storage")?;
            ensure!(
                end <= storage.len(),
                "{} reads past the end of storage {key}",
                info.name
            );
            out.push(LegacyTensor {
                name: info.name,
                dtype,
                shape: info.layout.shape().dims().to_vec(),
                data: storage[start..end].to_vec(),
            });
        }
    }
    ensure!(
        by_storage.is_empty(),
        "{} tensors reference storages the archive never wrote",
        by_storage.len()
    );

    let mut trailing = [0_u8; 1];
    ensure!(
        reader
            .read(&mut trailing)
            .context("reading past the last storage")?
            == 0,
        "{} carries bytes after its last storage",
        path.display()
    );
    Ok(out)
}

/// The storage key out of `into_tensor_info`'s `{dir}/{key}` rendering.
fn storage_key(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// Read exactly one pickle from `reader`, leaving it positioned on the next
/// byte. `Stack::read_loop` stops at `STOP`, and a `BufReader` carries its
/// position across calls, which is what makes five pickles in one stream
/// readable at all.
fn read_pickle<R: std::io::BufRead>(reader: &mut R) -> Result<Object> {
    let mut stack = Stack::empty();
    stack.read_loop(reader)?;
    Ok(stack.finalize()?)
}

/// A big-endian archive would need every storage byte-swapped. Refuse instead
/// of silently producing transposed nonsense.
fn ensure_little_endian(sys_info: &Object) -> Result<()> {
    let Object::Dict(entries) = sys_info else {
        bail!("sys_info is not a dict");
    };
    let little_endian = entries
        .iter()
        .find(|(key, _)| matches!(key, Object::Unicode(name) if name == "little_endian"))
        .map(|(_, value)| value);
    match little_endian {
        Some(Object::Bool(true)) => Ok(()),
        Some(Object::Bool(false)) => {
            bail!("this is a big-endian torch archive; mold reads little-endian storage only")
        }
        _ => bail!("sys_info does not record its endianness"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preamble_is_torchs_own_magic_number_and_protocol() {
        // Byte for byte: PROTO 2, a ten-byte LONG1 whose little-endian value
        // is torch's 0x1950a86a20f9469cfc6c, STOP; then PROTO 2, BININT2
        // 1001, STOP.
        assert_eq!(&LEGACY_PREAMBLE[0..4], &[0x80, 0x02, 0x8a, 0x0a]);
        let magic = &LEGACY_PREAMBLE[4..14];
        let mut value = 0_u128;
        for (index, byte) in magic.iter().enumerate() {
            value |= (*byte as u128) << (index * 8);
        }
        assert_eq!(value, 0x1950a86a20f9469cfc6c);
        assert_eq!(LEGACY_PREAMBLE[14], b'.');
        assert_eq!(
            &LEGACY_PREAMBLE[15..],
            &[0x80, 0x02, b'M', 0xe9, 0x03, b'.']
        );
        // 1001, the legacy PROTOCOL_VERSION.
        assert_eq!(u16::from_le_bytes([0xe9, 0x03]), 1001);
    }

    #[test]
    fn the_two_containers_are_told_apart_by_their_first_bytes() {
        assert!(is_zip_container(b"PK\x03\x04rest"));
        assert!(!is_legacy_container(b"PK\x03\x04rest"));
        assert!(is_legacy_container(&LEGACY_PREAMBLE));
        assert!(!is_zip_container(&LEGACY_PREAMBLE));
        assert!(!is_zip_container(b""));
        assert!(!is_legacy_container(b""));
    }

    #[test]
    fn a_big_endian_archive_is_refused_rather_than_byte_swapped() {
        let sys_info = Object::Dict(vec![(
            Object::Unicode("little_endian".to_string()),
            Object::Bool(false),
        )]);
        let error = ensure_little_endian(&sys_info).unwrap_err().to_string();
        assert!(error.contains("big-endian"), "{error}");
    }

    #[test]
    fn an_archive_that_does_not_say_is_refused_too() {
        let error = ensure_little_endian(&Object::Dict(vec![]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("endianness"), "{error}");
        assert!(ensure_little_endian(&Object::Bool(true)).is_err());
    }

    #[test]
    fn a_storage_key_survives_the_directory_prefix_into_tensor_info_adds() {
        assert_eq!(storage_key("/94693637921008"), "94693637921008");
        assert_eq!(storage_key("archive/data/12345"), "12345");
        assert_eq!(storage_key("12345"), "12345");
    }

    #[test]
    fn a_file_that_is_not_a_legacy_archive_is_refused_at_its_first_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-checkpoint.pth");
        std::fs::write(&path, b"PK\x03\x04 this is the modern container").unwrap();
        let error = format!("{:#}", read_legacy_pth(&path).unwrap_err());
        assert!(error.contains("legacy magic-number"), "{error}");
    }

    #[test]
    fn a_truncated_file_is_refused_rather_than_read_short() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.pth");
        std::fs::write(&path, &LEGACY_PREAMBLE[..8]).unwrap();
        let error = format!("{:#}", read_legacy_pth(&path).unwrap_err());
        assert!(error.contains("shorter than"), "{error}");
    }
}
