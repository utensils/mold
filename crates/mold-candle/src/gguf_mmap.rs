//! Memory-mapped GGUF tensor loading.
//!
//! Candle's own GGUF readers ([`candle::quantized::gguf_file::TensorInfo::read`])
//! allocate a fresh `Vec<u8>` per tensor, `read_exact` the payload into it, and
//! hand that buffer to `qtensor_from_ggml`, which then copies it once more onto
//! the device. On a warm page cache the file read is a `memcpy` out of the page
//! cache into a freshly faulted anonymous buffer, so every byte of the
//! checkpoint is copied twice on the host before it reaches the accelerator.
//!
//! Measured on plato (4x L40S, files already in page cache, 2026-09-11 audit):
//! `flux2-dev-Q8_0.gguf` (33 GB) took 34.4 s — 0.96 GB/s — against 3.99 s for
//! stable-diffusion.cpp reading the SAME file, and `flux1-dev-Q8_0.gguf`
//! (12.1 GB) took 8.4 s against 1.58 s. mold's own safetensors path, which
//! uploads straight out of a mapping, sustains 5.3 GB/s on the same machine.
//!
//! [`GgufMmap`] maps the file once and hands `qtensor_from_ggml` a slice of
//! that mapping, so the host-side copy disappears and the upload reads page
//! cache directly. Nothing else changes: the same candle constructor produces
//! the same `QTensor`, byte for byte, on every backend.
//!
//! Two things the reader path got for free have to be done explicitly here.
//! Bounds: a tensor's payload must lie inside the mapping, which the reader
//! checked against the file length. And ALIGNMENT: `qtensor_from_ggml` builds a
//! `&[Block]` from the raw pointer, so the slice's address must satisfy the
//! block type's alignment. A `Vec<u8>` is always suitably aligned by the
//! allocator; a mapping offset by the file's own `general.alignment` is not,
//! unless that alignment says so. [`GgufMmap::tensor_bytes`] checks both and
//! names the file and the tensor when either fails.

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// A GGUF checkpoint mapped into the address space, with its header parsed.
pub struct GgufMmap {
    map: memmap2::Mmap,
    content: gguf_file::Content,
    path: PathBuf,
}

impl std::fmt::Debug for GgufMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufMmap")
            .field("path", &self.path)
            .field("tensors", &self.content.tensor_infos.len())
            .finish()
    }
}

/// Alignment `qtensor_from_ggml`'s `slice::from_raw_parts` requires for a
/// tensor of this dtype.
///
/// Taken from the block types themselves rather than a table, so a new ggml
/// dtype in the fork cannot silently acquire a wrong constant here. Every
/// k-quant and legacy block is `#[repr(C)]` over `f16`/`i8`/`u8` arrays, so in
/// practice this is 4 for the dtypes carrying an `f32` field and 2 for the
/// rest — comfortably inside GGUF's own 32-byte default alignment.
fn required_alignment(dtype: GgmlDType) -> usize {
    use candle::quantized::k_quants;
    use std::mem::align_of;
    match dtype {
        GgmlDType::F32 => align_of::<f32>(),
        GgmlDType::F16 => align_of::<half::f16>(),
        GgmlDType::BF16 => align_of::<half::bf16>(),
        GgmlDType::Q4_0 => align_of::<k_quants::BlockQ4_0>(),
        GgmlDType::Q4_1 => align_of::<k_quants::BlockQ4_1>(),
        GgmlDType::Q5_0 => align_of::<k_quants::BlockQ5_0>(),
        GgmlDType::Q5_1 => align_of::<k_quants::BlockQ5_1>(),
        GgmlDType::Q8_0 => align_of::<k_quants::BlockQ8_0>(),
        GgmlDType::Q8_1 => align_of::<k_quants::BlockQ8_1>(),
        GgmlDType::Q2K => align_of::<k_quants::BlockQ2K>(),
        GgmlDType::Q3K => align_of::<k_quants::BlockQ3K>(),
        GgmlDType::Q4K => align_of::<k_quants::BlockQ4K>(),
        GgmlDType::Q5K => align_of::<k_quants::BlockQ5K>(),
        GgmlDType::Q6K => align_of::<k_quants::BlockQ6K>(),
        GgmlDType::Q8K => align_of::<k_quants::BlockQ8K>(),
    }
}

impl GgufMmap {
    /// Map `path` and parse its header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = std::fs::File::open(&path)?;
        let content = gguf_file::Content::read(&mut file)?;
        // SAFETY: the same contract every mmap'd checkpoint in mold takes —
        // the file must not be truncated or rewritten underneath us. Model
        // weights are verified at download and immutable thereafter.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { map, content, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    pub fn tensor_infos(&self) -> &HashMap<String, gguf_file::TensorInfo> {
        &self.content.tensor_infos
    }

    /// Total payload bytes across every tensor, for a progress denominator.
    ///
    /// Deliberately the tensors' own bytes rather than the file length: the
    /// header and any padding are not uploaded, and a counter that promised
    /// them would never reach its total.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.content
            .tensor_infos
            .values()
            .map(|info| tensor_payload_len(info).unwrap_or(0) as u64)
            .sum()
    }

    /// The raw payload of one tensor, bounds- and alignment-checked.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = match self.content.tensor_infos.get(name) {
            Some(info) => info,
            None => candle::bail!(
                "cannot find tensor info for {name} in {}",
                self.path.display()
            ),
        };
        let len = tensor_payload_len(info).ok_or_else(|| {
            candle::Error::Msg(format!(
                "tensor {name} in {} has {} elements, which is not a multiple of its {:?} block \
                 size {}",
                self.path.display(),
                info.shape.elem_count(),
                info.ggml_dtype,
                info.ggml_dtype.block_size(),
            ))
        })?;
        let start = usize::try_from(self.content.tensor_data_offset.saturating_add(info.offset))
            .map_err(|_| {
                candle::Error::Msg(format!(
                    "tensor {name} in {} starts past the addressable range",
                    self.path.display()
                ))
            })?;
        let end = start.checked_add(len).ok_or_else(|| {
            candle::Error::Msg(format!(
                "tensor {name} in {} overflows its own extent",
                self.path.display()
            ))
        })?;
        if end > self.map.len() {
            candle::bail!(
                "tensor {name} needs {len} bytes at offset {start} in {}, which holds only {} bytes",
                self.path.display(),
                self.map.len()
            )
        }
        let bytes = &self.map[start..end];
        let alignment = required_alignment(info.ggml_dtype);
        if !(bytes.as_ptr() as usize).is_multiple_of(alignment) {
            // A well-formed GGUF aligns its tensor data to `general.alignment`
            // (32 by default), and every mapping starts on a page boundary, so
            // this is a malformed or hand-edited file rather than a case the
            // loader should paper over with a copy.
            candle::bail!(
                "tensor {name} in {} starts at offset {start}, which is not {alignment}-byte \
                 aligned for {:?}; check the file's `general.alignment`",
                self.path.display(),
                info.ggml_dtype,
            )
        }
        Ok(bytes)
    }

    /// Build one tensor on `device` straight from the mapping.
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let info = match self.content.tensor_infos.get(name) {
            Some(info) => info,
            None => candle::bail!(
                "cannot find tensor info for {name} in {}",
                self.path.display()
            ),
        };
        let bytes = self.tensor_bytes(name)?;
        candle::quantized::ggml_file::qtensor_from_ggml(
            info.ggml_dtype,
            bytes,
            info.shape.dims().to_vec(),
            device,
        )
    }

    /// Load every tensor onto `device`, reporting `(bytes_done, bytes_total)`
    /// as it goes and logging the file's measured throughput once.
    pub fn load_all(
        &self,
        device: &Device,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<HashMap<String, Arc<QTensor>>> {
        let total = self.total_tensor_bytes();
        let started = Instant::now();
        let mut done = 0u64;
        progress(0, total);
        let mut tensors = HashMap::with_capacity(self.content.tensor_infos.len());
        // Ascending file order, so the kernel's readahead sees a sequential
        // pattern on a cold cache instead of the hash map's arbitrary one.
        let mut names: Vec<&String> = self.content.tensor_infos.keys().collect();
        names.sort_by_key(|name| self.content.tensor_infos[*name].offset);
        for name in names {
            let tensor = self.load_tensor(name, device)?;
            done = done.saturating_add(
                tensor_payload_len(&self.content.tensor_infos[name]).unwrap_or(0) as u64,
            );
            progress(done, total);
            tensors.insert(name.clone(), Arc::new(tensor));
        }
        let elapsed = started.elapsed();
        tracing::info!(
            path = %self.path.display(),
            bytes = total,
            elapsed_ms = elapsed.as_millis() as u64,
            gb_per_s = total as f64 / 1e9 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            "gguf load"
        );
        Ok(tensors)
    }
}

/// Payload length of one tensor, or `None` when its element count is not a
/// multiple of its block size.
fn tensor_payload_len(info: &gguf_file::TensorInfo) -> Option<usize> {
    let elems = info.shape.elem_count();
    let block = info.ggml_dtype.block_size();
    if !elems.is_multiple_of(block) {
        return None;
    }
    Some(elems / block * info.ggml_dtype.type_size())
}

/// A `candle-transformers` quantized `VarBuilder` built from a mapping.
///
/// Upstream's `quantized_var_builder::VarBuilder::from_gguf` runs the copying
/// reader loop; its public `from_qtensors` is the seam that lets mold hand it
/// the same map, loaded once, without forking the consumers that take that
/// type (`flux::quantized_model`, SD3, Z-Image, Qwen-Image).
pub fn transformers_var_builder_from_gguf_mmap<P: AsRef<Path>>(
    path: P,
    device: &Device,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<candle_transformers::quantized_var_builder::VarBuilder> {
    let map = GgufMmap::open(path)?;
    let tensors = map.load_all(device, progress)?;
    Ok(candle_transformers::quantized_var_builder::VarBuilder::from_qtensors(tensors, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::Value;
    use candle::Tensor;

    /// Alignment the fixture declares.
    ///
    /// It is 32 and cannot be anything else: `gguf_file::write` pads to a
    /// hard-coded 32 bytes (`candle-core/src/quantized/gguf_file.rs:625`,
    /// `:627`, `:641`) while `Content::read` derives `tensor_data_offset` from
    /// whatever `general.alignment` says (`:557-566`), so a fixture declaring
    /// 64 is a file candle's OWN reader cannot parse. Declaring the value the
    /// writer actually honours keeps this a test of the mapping rather than of
    /// that upstream inconsistency.
    const FIXTURE_ALIGNMENT: u32 = 32;

    /// Write a GGUF holding one tensor of each interesting storage class, with
    /// an explicit `general.alignment`.
    fn write_fixture(path: &Path, alignment: u32) -> Vec<(String, QTensor)> {
        let device = Device::Cpu;
        let dense = QTensor::quantize(
            &Tensor::arange(0f32, 512f32, &device)
                .unwrap()
                .reshape((8, 64))
                .unwrap(),
            GgmlDType::F32,
        )
        .unwrap();
        let q8 = QTensor::quantize(
            &Tensor::arange(0f32, 2048f32, &device)
                .unwrap()
                .reshape((8, 256))
                .unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        let q4k = QTensor::quantize(
            &Tensor::arange(0f32, 2048f32, &device)
                .unwrap()
                .reshape((8, 256))
                .unwrap(),
            GgmlDType::Q4K,
        )
        .unwrap();

        let mut file = std::fs::File::create(path).unwrap();
        let alignment = Value::U32(alignment);
        gguf_file::write(
            &mut file,
            &[("general.alignment", &alignment)],
            &[("stem.weight", &dense), ("blk.0.w", &q8), ("blk.1.w", &q4k)],
        )
        .unwrap();
        drop(file);

        vec![
            ("stem.weight".to_string(), dense),
            ("blk.0.w".to_string(), q8),
            ("blk.1.w".to_string(), q4k),
        ]
    }

    fn reader_tensors(path: &Path) -> HashMap<String, Arc<QTensor>> {
        let mut file = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let mut tensors = HashMap::new();
        for name in content.tensor_infos.keys() {
            tensors.insert(
                name.clone(),
                Arc::new(content.tensor(&mut file, name, &Device::Cpu).unwrap()),
            );
        }
        tensors
    }

    /// The mapping and the copying reader must produce the same checkpoint —
    /// same names, shapes, dtypes and payload bytes. This is the whole
    /// correctness claim: the loader is a transport change, not a numerical
    /// one.
    #[test]
    fn the_mapping_and_the_reader_agree_tensor_for_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        write_fixture(&path, FIXTURE_ALIGNMENT);

        let expected = reader_tensors(&path);
        let map = GgufMmap::open(&path).unwrap();
        let got = map.load_all(&Device::Cpu, &mut |_, _| {}).unwrap();

        assert_eq!(got.len(), expected.len());
        for (name, want) in &expected {
            let have = got.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(have.shape(), want.shape(), "{name}");
            assert_eq!(have.dtype(), want.dtype(), "{name}");
            assert_eq!(have.data().unwrap(), want.data().unwrap(), "{name}");
        }
        assert!(
            expected
                .values()
                .any(|tensor| tensor.dtype() == GgmlDType::Q4K),
            "the fixture must exercise a k-quant"
        );
    }

    /// The progress counter's total is the tensors' own bytes, and it arrives
    /// there exactly.
    #[test]
    fn progress_reaches_the_total_tensor_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let written = write_fixture(&path, FIXTURE_ALIGNMENT);

        let map = GgufMmap::open(&path).unwrap();
        let expected_total: u64 = written
            .iter()
            .map(|(_, tensor)| tensor.storage_size_in_bytes() as u64)
            .sum();
        assert_eq!(map.total_tensor_bytes(), expected_total);

        let mut samples = Vec::new();
        map.load_all(&Device::Cpu, &mut |done, total| samples.push((done, total)))
            .unwrap();
        assert_eq!(samples.first(), Some(&(0, expected_total)));
        assert_eq!(samples.last(), Some(&(expected_total, expected_total)));
        assert!(
            samples.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "the counter never goes backwards"
        );
    }

    /// A mapping offset the block type cannot be read at is a named error, not
    /// an unaligned `slice::from_raw_parts`.
    ///
    /// `qtensor_from_ggml` reinterprets the payload as `&[Block]` in place, so
    /// the reader path's fresh `Vec<u8>` was quietly doing the alignment work.
    /// A file whose `general.alignment` is 1 can place an F32 tensor on an odd
    /// byte, and that must be refused by name.
    #[test]
    fn an_alignment_the_block_type_cannot_take_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        write_fixture(&path, FIXTURE_ALIGNMENT);

        // Re-point the F32 stem one byte into the tensor data. Its payload
        // still lies inside the file, so only the alignment check can catch it.
        let mut map = GgufMmap::open(&path).unwrap();
        let info = map.content.tensor_infos.get_mut("stem.weight").unwrap();
        info.offset += 1;

        let error = map.tensor_bytes("stem.weight").unwrap_err().to_string();
        assert!(error.contains("aligned"), "{error}");
        assert!(error.contains("stem.weight"), "{error}");
        assert!(error.contains("general.alignment"), "{error}");
    }

    /// A payload that runs past the end of the mapping names the file and the
    /// tensor rather than reading whatever follows.
    #[test]
    fn a_tensor_past_the_end_of_the_mapping_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        write_fixture(&path, FIXTURE_ALIGNMENT);

        let mut map = GgufMmap::open(&path).unwrap();
        let info = map.content.tensor_infos.get_mut("blk.0.w").unwrap();
        info.offset += 1 << 20;

        let error = map.tensor_bytes("blk.0.w").unwrap_err().to_string();
        assert!(error.contains("blk.0.w"), "{error}");
        assert!(error.contains("holds only"), "{error}");
    }

    /// The `candle-transformers` builder gets the identical tensors.
    #[test]
    fn the_transformers_var_builder_sees_the_mapped_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let written = write_fixture(&path, FIXTURE_ALIGNMENT);

        let vb =
            transformers_var_builder_from_gguf_mmap(&path, &Device::Cpu, &mut |_, _| {}).unwrap();
        for (name, want) in &written {
            let have = vb.get(want.shape().dims(), name).unwrap();
            assert_eq!(have.data().unwrap(), want.data().unwrap(), "{name}");
        }
    }
}
