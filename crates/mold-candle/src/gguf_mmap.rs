//! GGUF tensor loading: a mapping for selective reads, a staged parallel read
//! for whole-checkpoint loads.
//!
//! Candle's own GGUF readers ([`candle::quantized::gguf_file::TensorInfo::read`])
//! allocate a fresh `Vec<u8>` per tensor, `read_exact` the payload into it, and
//! hand that buffer to `qtensor_from_ggml`, which then copies it once more onto
//! the device, so every byte of the checkpoint is copied twice on the host
//! before it reaches the accelerator.
//!
//! [`GgufMmap`] maps the file once, and [`GgufMmap::tensor_bytes`] /
//! [`GgufMmap::load_tensor`] hand `qtensor_from_ggml` a slice of that mapping.
//! That is the right shape for a caller reading a FEW tensors — a LoRA merge,
//! one encoder's weights — because an untouched page is never faulted at all.
//!
//! Two things the reader path got for free have to be done explicitly for a
//! mapped slice. Bounds: a tensor's payload must lie inside the mapping, which
//! the reader checked against the file length. And ALIGNMENT:
//! `qtensor_from_ggml` builds a `&[Block]` from the raw pointer, so the slice's
//! address must satisfy the block type's alignment. A `Vec<u8>` is always
//! suitably aligned by the allocator; a mapping offset by the file's own
//! `general.alignment` is not, unless that alignment says so.
//! [`GgufMmap::tensor_bytes`] checks both and names the file and the tensor
//! when either fails; the staged path repeats the same two checks against the
//! staging buffer, because a staged tensor is read as `&[Block]` in place too.
//!
//! # Why [`GgufMmap::load_all`] does NOT read from the mapping
//!
//! Reading a whole 22–35 GB checkpoint through the mapping is a PAGE FAULT per
//! 4 KiB, and on ZFS — which is what `$MOLD_HOME` is on every machine mold is
//! qualified on — that fault is not the cheap page-cache lookup it is on ext4.
//! ZFS keeps its own ARC and populates the page cache one faulting page at a
//! time, with no fault-around and no readahead: `madvise(MADV_WILLNEED)` over
//! the whole mapping returns in 0 ms and populates NOTHING.
//!
//! Measured on plato (4x L40S, ZFS, ARC warm, page cache dropped for the file,
//! `qwen-image-Q8_0.gguf`, 21.76 GB, 2026-09-11):
//!
//! | path                                   | elapsed  | GB/s | major faults |
//! | -------------------------------------- | -------- | ---- | ------------ |
//! | mapping, per tensor                    | 26.8 s   | 0.81 | 5,312,908    |
//! | mapping + `MADV_WILLNEED`              | 28.9 s   | 0.75 | 5,312,908    |
//! | mapping + `MAP_POPULATE`               | 27.5 s   | 0.79 | 0 (in-kernel)|
//! | staged parallel read (this loader)     | 3.4 s    | 6.3  | 0            |
//!
//! 5,312,908 is exactly one fault per 4 KiB page of the payload, at ~5 µs each.
//! `MAP_POPULATE` only moves the same per-page cost into the kernel.
//!
//! The full before/after, same machine, mapping against this loader, with the
//! file's page cache dropped ("cold") and then immediately repeated ("warm"):
//!
//! | checkpoint             | size     | cold                 | warm                 |
//! | ---------------------- | -------- | -------------------- | -------------------- |
//! | `flux1-dev-Q8_0.gguf`  | 12.71 GB | 15.5 s → 1.55 s      | 2.16 s → 0.80 s      |
//! | `qwen-image-Q8_0.gguf` | 21.76 GB | 26.5 s → 3.28 s      | 2.65 s → 1.81 s      |
//! | `flux2-dev-Q8_0.gguf`  | 35.00 GB | 41.7 s → 4.31 s      | 6.60 s → 2.68 s      |
//!
//! In GB/s: 0.82 → 8.2 cold and 5.9 → 15.9 warm on the 12.71 GB file, and
//! 0.84 → 8.1 cold and 5.3 → 13.1 warm on the 35 GB one. The staged cold
//! figure varies with how much of the file ZFS still holds in ARC (4.2–5.3 s
//! on the 35 GB file across runs); the mapping's does not move with it,
//! because its cost is the fault and not the read. Re-measure with `cargo run
//! --release -p mold-ai-candle --features dev-bins,cuda --bin gguf_load_bench
//! -- <file> mmap load_all` rather than re-deriving, and drop the file's page
//! cache first (`posix_fadvise` `DONTNEED`) or the "cold" column measures
//! nothing. `verify_cuda` on the same binary is the correctness half: it
//! compares what landed in DEVICE memory against the file, which is the only
//! check that can catch a staging buffer refilled while an upload out of it
//! was still in flight.
//!
//! The read path ZFS IS good at is `pread`, and it scales with threads: 2.4
//! GB/s on one thread, 6.5 on four, 7.8 on eight, flat thereafter. So
//! [`GgufMmap::load_all`] reads the payload in contiguous batches with a small
//! thread pool into a reused staging buffer and uploads each tensor from
//! there. On CUDA the staging buffer is PAGE-LOCKED, which both lets the
//! driver DMA straight out of it (14.2 GB/s measured, against 7.5 for a
//! pageable copy) and makes `memcpy_htod` genuinely asynchronous, so two
//! buffers alternate: one is being read into while the other is still being
//! uploaded. Host overhead is bounded by the pair, never by the checkpoint.

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Bytes in one staging buffer, and so the size of one read batch.
///
/// Two are held for the duration of a [`GgufMmap::load_all`], so the loader's
/// host overhead is twice this and does not grow with the checkpoint.
pub const STAGING_CHUNK_BYTES: usize = 256 << 20;

/// Ceiling on one staging buffer.
///
/// A tensor larger than the buffer cannot be staged (`qtensor_from_ggml` needs
/// its payload contiguous), so the buffer grows to the largest tensor in the
/// file — up to here, which bounds the PAIR at 1 GiB whatever the checkpoint
/// holds. Past it those individual tensors fall back to the mapping, which is
/// the right trade: a checkpoint with a >512 MiB tensor has few of them, and
/// the alternative is an unbounded page-locked allocation on a 64 GB desktop.
/// The largest tensor in any checkpoint mold ships is FLUX.2 [dev] Q8_0's 453
/// MB, so today every tensor of every shipped model stages.
const STAGING_MAX_BYTES: usize = 512 << 20;

/// Bytes one reader thread reads per `pread`.
const READ_BLOCK_BYTES: usize = 8 << 20;

/// Reader threads filling one staging buffer.
///
/// Eight is where ZFS's read path stops scaling on the qualification machine
/// (2.4 / 6.5 / 7.8 GB/s at 1 / 4 / 8 threads, and 7.9 at 16), and it is
/// clamped by the batch so a small checkpoint does not spawn eight threads to
/// read a few megabytes.
const READER_THREADS: usize = 8;

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
        Self::open_with(path, false)
    }

    /// Map `path` with `MAP_POPULATE`, pre-faulting every page table entry.
    ///
    /// Developer benchmark surface: the shipping loader deliberately does NOT
    /// take this route (see [`GgufMmap::load_all`] for what does instead).
    pub fn open_populated<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, true)
    }

    fn open_with<P: AsRef<Path>>(path: P, populate: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = std::fs::File::open(&path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut options = memmap2::MmapOptions::new();
        #[cfg(target_os = "linux")]
        if populate {
            options.populate();
        }
        #[cfg(not(target_os = "linux"))]
        let _ = populate;
        // SAFETY: the same contract every mmap'd checkpoint in mold takes —
        // the file must not be truncated or rewritten underneath us. Model
        // weights are verified at download and immutable thereafter.
        let map = unsafe { options.map(&file)? };
        Ok(Self { map, content, path })
    }

    /// `MADV_WILLNEED` over the whole mapping.
    ///
    /// Developer benchmark surface.
    pub fn advise_will_need(&self) -> Result<()> {
        #[cfg(unix)]
        self.map.advise(memmap2::Advice::WillNeed)?;
        Ok(())
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

    /// Tensor names in ascending file order.
    ///
    /// The staged reader needs this because a batch is a CONTIGUOUS file
    /// range; the mapped loop wanted it so a cold cache saw a sequential
    /// access pattern rather than the hash map's arbitrary one.
    fn names_in_file_order(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.content.tensor_infos.keys().collect();
        names.sort_by_key(|name| self.content.tensor_infos[*name].offset);
        names
    }

    /// Absolute file offset and payload length of one tensor.
    fn tensor_extent(&self, name: &str) -> Result<(u64, usize)> {
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
        Ok((
            self.content.tensor_data_offset.saturating_add(info.offset),
            len,
        ))
    }

    /// Load every tensor onto `device`, reporting `(bytes_done, bytes_total)`
    /// as it goes and logging the file's measured throughput once.
    ///
    /// See the module docs for why this reads the file rather than walking the
    /// mapping it already holds.
    pub fn load_all(
        &self,
        device: &Device,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<HashMap<String, Arc<QTensor>>> {
        let total = self.total_tensor_bytes();
        let started = Instant::now();
        // The staged read trades one host copy for the mapping's per-page
        // fault, and that trade only pays where the copy lands in page-locked
        // memory the driver DMAs out of. On CPU and Metal it would be a pure
        // extra copy plus a host buffer, so those keep the mapping exactly as
        // they had it.
        let tensors = if matches!(device, Device::Cuda(_)) {
            let staging = self
                .content
                .tensor_infos
                .values()
                .filter_map(tensor_payload_len)
                .max()
                .unwrap_or(0)
                .clamp(STAGING_CHUNK_BYTES, STAGING_MAX_BYTES);
            self.load_all_staged(device, progress, staging)?
        } else {
            self.load_all_from_mapping(device, progress)?
        };
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

    /// [`Self::load_all`]'s staged read, with an explicit staging-buffer size
    /// and on any device.
    ///
    /// Separated from the dispatch so a test can drive the batching and the
    /// oversize fallback on a fixture far smaller than a real staging buffer,
    /// and so the developer benchmark can run it against the mapping on the
    /// CPU, where the two are directly comparable byte for byte. The staging
    /// itself is device-agnostic; `load_all` picks it for CUDA alone.
    pub fn load_all_staged(
        &self,
        device: &Device,
        progress: &mut dyn FnMut(u64, u64),
        staging_bytes: usize,
    ) -> Result<HashMap<String, Arc<QTensor>>> {
        let total = self.total_tensor_bytes();
        let mut done = 0u64;
        progress(0, total);

        let names = self.names_in_file_order();
        let mut tensors = HashMap::with_capacity(names.len());
        let path = self.path.as_path();
        let mut buffers = StagingPair::new(device, staging_bytes.max(1))?;

        let mut index = 0usize;
        while index < names.len() {
            // A batch is every tensor that fits, whole, in one buffer, taken in
            // file order so its bytes are one contiguous read.
            let (batch_start, first_len) = self.tensor_extent(names[index])?;
            if first_len > staging_bytes {
                // Larger than the buffer: `qtensor_from_ggml` needs the payload
                // contiguous, so this one comes off the mapping.
                let tensor = self.load_tensor(names[index], device)?;
                tensors.insert(names[index].clone(), Arc::new(tensor));
                done = done.saturating_add(first_len as u64);
                progress(done, total);
                index += 1;
                continue;
            }
            let mut end = index;
            let mut batch_end = batch_start;
            while end < names.len() {
                let (start, len) = self.tensor_extent(names[end])?;
                let next_end = start.saturating_add(len as u64);
                if next_end.saturating_sub(batch_start) > staging_bytes as u64 {
                    break;
                }
                // `max` rather than assignment: nothing in the format forbids
                // one tensor's extent lying inside an earlier one's, and a
                // shrinking batch end would leave a tensor's bytes unread.
                batch_end = batch_end.max(next_end);
                end += 1;
            }

            let span = (batch_end - batch_start) as usize;
            let buffer = buffers.next_free()?;
            read_parallel(path, batch_start, &mut buffer.bytes_mut()[..span])?;

            for name in &names[index..end] {
                let info = &self.content.tensor_infos[*name];
                let (start, len) = self.tensor_extent(name)?;
                let at = (start - batch_start) as usize;
                let bytes = buffer.staged(name, &self.path, info.ggml_dtype, at, len)?;
                tensors.insert(
                    (*name).clone(),
                    Arc::new(candle::quantized::ggml_file::qtensor_from_ggml(
                        info.ggml_dtype,
                        bytes,
                        info.shape.dims().to_vec(),
                        device,
                    )?),
                );
                done = done.saturating_add(len as u64);
                progress(done, total);
            }
            // Every upload out of this buffer is now issued; mark it so the
            // next fill waits for them rather than overwriting bytes in flight.
            buffer.mark_uploads_issued()?;
            index = end;
        }
        buffers.finish()?;
        Ok(tensors)
    }

    /// The pre-staging loop: every tensor read straight from the mapping.
    ///
    /// Kept as the oracle [`Self::load_all`] is tested against, and as the
    /// developer benchmark's baseline. See the module docs for why it is not
    /// the shipping path.
    pub fn load_all_from_mapping(
        &self,
        device: &Device,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<HashMap<String, Arc<QTensor>>> {
        let total = self.total_tensor_bytes();
        let mut done = 0u64;
        progress(0, total);
        let mut tensors = HashMap::with_capacity(self.content.tensor_infos.len());
        for name in self.names_in_file_order() {
            let tensor = self.load_tensor(name, device)?;
            done = done.saturating_add(
                tensor_payload_len(&self.content.tensor_infos[name]).unwrap_or(0) as u64,
            );
            progress(done, total);
            tensors.insert(name.clone(), Arc::new(tensor));
        }
        Ok(tensors)
    }
}

/// Read `[at, at + buf.len())` of `file` into `buf`, split across threads.
///
/// One thread saturates neither ZFS's read path nor an NVMe array; see the
/// module docs for the measured scaling. `pread` takes `&File`, so the threads
/// share one descriptor and each owns a disjoint slice of the buffer.
fn read_parallel(path: &Path, at: u64, buf: &mut [u8]) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    let threads = READER_THREADS
        .min(buf.len().div_ceil(READ_BLOCK_BYTES))
        .max(1);
    if threads == 1 {
        return read_exact_at(&File::open(path)?, buf, at);
    }
    // Round the per-thread span up to a whole read block so no thread is handed
    // a sliver, and so the split lands on a boundary the filesystem likes.
    let per = buf
        .len()
        .div_ceil(threads)
        .next_multiple_of(READ_BLOCK_BYTES);
    let mut failure: Option<candle::Error> = None;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for (index, part) in buf.chunks_mut(per).enumerate() {
            let offset = at + (index * per) as u64;
            handles.push(scope.spawn(move || read_exact_at(&File::open(path)?, part, offset)));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                }
                Err(_) => {
                    failure.get_or_insert_with(|| {
                        candle::Error::Msg("gguf staging reader thread panicked".to_string())
                    });
                }
            }
        }
    });
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// `buf.len()` bytes from `at`, without disturbing any other reader.
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    Ok(file.read_exact_at(buf, at)?)
}

/// `buf.len()` bytes from `at`. `seek_read` moves this handle's file pointer
/// and returns short reads, so the caller owns the handle and the loop is ours
/// rather than the standard library's.
#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    let mut at = at;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], at) {
            Ok(0) => {
                return Err(candle::Error::from(std::io::Error::from(
                    std::io::ErrorKind::UnexpectedEof,
                )))
            }
            Ok(read) => {
                done += read;
                at += read as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

/// The staging buffers one [`GgufMmap::load_all`] alternates between.
///
/// Two on CUDA, where `memcpy_htod` out of page-locked memory is asynchronous
/// and one buffer can be read into while the other is still uploading. One
/// everywhere else, because `qtensor_from_ggml` copies synchronously there and
/// a second buffer would be idle memory.
struct StagingPair {
    buffers: Vec<StagingBuffer>,
    next: usize,
}

impl StagingPair {
    fn new(device: &Device, bytes: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(_) = device {
            if let Some(pair) = cuda_staging::pair(device, bytes)? {
                return Ok(Self {
                    buffers: pair,
                    next: 0,
                });
            }
        }
        let _ = device;
        Ok(Self {
            buffers: vec![StagingBuffer::heap(bytes)],
            next: 0,
        })
    }

    /// The next buffer in rotation, with any upload still reading it complete.
    fn next_free(&mut self) -> Result<&mut StagingBuffer> {
        let index = self.next;
        self.next = (self.next + 1) % self.buffers.len();
        let buffer = &mut self.buffers[index];
        buffer.wait_for_uploads()?;
        Ok(buffer)
    }

    /// Complete every outstanding upload, so the buffers can be released.
    ///
    /// Dropping page-locked memory the driver is still DMA-ing out of is a use
    /// after free, so this is not optional.
    fn finish(&mut self) -> Result<()> {
        for buffer in &mut self.buffers {
            buffer.wait_for_uploads()?;
        }
        Ok(())
    }
}

/// One staging buffer: host bytes a tensor is read into and uploaded from.
struct StagingBuffer {
    storage: StagingStorage,
}

enum StagingStorage {
    /// `u64` rather than `u8` so the allocation's base is 8-byte aligned; the
    /// block types need at most 4, and `Vec<u8>`'s alignment is 1.
    Heap(Vec<u64>, usize),
    #[cfg(feature = "cuda")]
    Pinned(cuda_staging::Pinned),
}

impl StagingBuffer {
    fn heap(bytes: usize) -> Self {
        Self {
            storage: StagingStorage::Heap(vec![0u64; bytes.div_ceil(8)], bytes),
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.storage {
            StagingStorage::Heap(words, bytes) => {
                // SAFETY: `words` holds at least `bytes` initialized bytes, and
                // every bit pattern is a valid `u8`.
                unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, *bytes) }
            }
            #[cfg(feature = "cuda")]
            StagingStorage::Pinned(pinned) => pinned.bytes_mut(),
        }
    }

    fn bytes(&self) -> &[u8] {
        match &self.storage {
            StagingStorage::Heap(words, bytes) => {
                // SAFETY: as above, and shared rather than exclusive.
                unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, *bytes) }
            }
            #[cfg(feature = "cuda")]
            StagingStorage::Pinned(pinned) => pinned.bytes(),
        }
    }

    /// The staged payload of one tensor, bounds- and alignment-checked.
    ///
    /// `qtensor_from_ggml` reinterprets this as `&[Block]` in place exactly as
    /// it does for a mapped slice, so the same two checks apply — and the
    /// errors name the file and the tensor identically, because from the
    /// caller's side a staged read and a mapped one are the same load.
    fn staged(
        &self,
        name: &str,
        path: &Path,
        dtype: GgmlDType,
        at: usize,
        len: usize,
    ) -> Result<&[u8]> {
        let bytes = self.bytes();
        let end = at.checked_add(len).ok_or_else(|| {
            candle::Error::Msg(format!(
                "tensor {name} in {} overflows its own extent",
                path.display()
            ))
        })?;
        if end > bytes.len() {
            candle::bail!(
                "tensor {name} needs {len} bytes at offset {at} of a {} byte staging buffer for {}",
                bytes.len(),
                path.display(),
            )
        }
        let bytes = &bytes[at..end];
        let alignment = required_alignment(dtype);
        if !(bytes.as_ptr() as usize).is_multiple_of(alignment) {
            candle::bail!(
                "tensor {name} in {} is staged at offset {at}, which is not {alignment}-byte \
                 aligned for {dtype:?}; check the file's `general.alignment`",
                path.display(),
            )
        }
        Ok(bytes)
    }

    /// Record that every upload reading this buffer has been issued.
    ///
    /// Takes `&mut self` because it arms the wait: with a shared receiver the
    /// flag could not be set, every wait would return immediately, and the
    /// next fill would overwrite bytes the driver was still DMA-ing.
    fn mark_uploads_issued(&mut self) -> Result<()> {
        match &mut self.storage {
            StagingStorage::Heap(..) => Ok(()),
            #[cfg(feature = "cuda")]
            StagingStorage::Pinned(pinned) => pinned.record(),
        }
    }

    fn wait_for_uploads(&mut self) -> Result<()> {
        match &mut self.storage {
            StagingStorage::Heap(..) => Ok(()),
            #[cfg(feature = "cuda")]
            StagingStorage::Pinned(pinned) => pinned.wait(),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_staging {
    use super::{StagingBuffer, StagingStorage};
    use candle::cuda::cudarc::driver::{sys, CudaEvent, CudaStream, PinnedHostSlice};
    use candle::{Device, Result};
    use std::sync::Arc;

    /// A page-locked staging buffer and the event that says when the uploads
    /// reading it have completed.
    pub(super) struct Pinned {
        slice: PinnedHostSlice<u8>,
        stream: Arc<CudaStream>,
        issued: CudaEvent,
        pending: bool,
    }

    impl Pinned {
        pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
            // The allocation is live and `u8` has no invalid bit patterns, so
            // the only failure this can report is a driver error on the
            // slice's own (never recorded) event; an empty slice would be
            // caught immediately by the bounds check in `staged`.
            self.slice.as_mut_slice().unwrap_or(&mut [])
        }

        pub(super) fn bytes(&self) -> &[u8] {
            self.slice.as_slice().unwrap_or(&[])
        }

        pub(super) fn record(&mut self) -> Result<()> {
            self.issued
                .record(&self.stream)
                .map_err(|e| candle::Error::Msg(format!("gguf staging event record: {e}")))?;
            self.pending = true;
            Ok(())
        }

        pub(super) fn wait(&mut self) -> Result<()> {
            if !self.pending {
                return Ok(());
            }
            self.pending = false;
            self.issued
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("gguf staging event wait: {e}")))
        }
    }

    /// Releasing page-locked memory the driver is still reading is a use after
    /// free, and a load that fails partway returns without reaching
    /// `StagingPair::finish`. So the wait is also enforced here, where no
    /// early return can skip it.
    impl Drop for Pinned {
        fn drop(&mut self) {
            if self.pending {
                if let Err(error) = self.issued.synchronize() {
                    tracing::warn!(%error, "gguf staging: event wait failed while releasing");
                    // The event is unusable; block on the stream instead rather
                    // than free memory a copy may still be reading.
                    let _ = self.stream.synchronize();
                }
            }
        }
    }

    /// Two page-locked buffers, or `None` when the driver will not pin them.
    ///
    /// A refusal is not an error: the heap path is correct, just slower, and a
    /// load must not fail because the host is out of page-locked memory.
    pub(super) fn pair(device: &Device, bytes: usize) -> Result<Option<Vec<StagingBuffer>>> {
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let context = stream.context().clone();
        let mut buffers = Vec::with_capacity(2);
        for _ in 0..2 {
            // SAFETY: the allocation is written in full before any byte of it
            // is read, by `read_parallel` into `bytes_mut`.
            let slice = match unsafe { context.alloc_pinned::<u8>(bytes) } {
                Ok(slice) => slice,
                Err(error) => {
                    tracing::debug!(
                        %error,
                        bytes,
                        "gguf staging: page-locked allocation refused, using heap buffers"
                    );
                    return Ok(None);
                }
            };
            // BLOCKING_SYNC so the wait yields the core instead of spinning on
            // it — the reader threads want that core.
            let issued = context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| candle::Error::Msg(format!("gguf staging event: {e}")))?;
            buffers.push(StagingBuffer {
                storage: StagingStorage::Pinned(Pinned {
                    slice,
                    stream: stream.clone(),
                    issued,
                    pending: false,
                }),
            });
        }
        Ok(Some(buffers))
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

    /// The staged read and the mapping produce the same checkpoint, at every
    /// staging size — including sizes that split the file into several batches
    /// and sizes no single tensor fits in.
    ///
    /// The staging logic is deliberately device-agnostic (the page-locked
    /// buffer is an allocation choice, not a different code path), so `Cpu`
    /// exercises the batching, the offsets, and the oversize fallback.
    #[test]
    fn the_staged_reader_and_the_mapping_agree_at_every_staging_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        write_fixture(&path, FIXTURE_ALIGNMENT);

        let map = GgufMmap::open(&path).unwrap();
        let expected = map
            .load_all_from_mapping(&Device::Cpu, &mut |_, _| {})
            .unwrap();
        let largest = map
            .tensor_infos()
            .values()
            .filter_map(tensor_payload_len)
            .max()
            .unwrap();

        // 1 byte forces every tensor onto the oversize fallback; `largest`
        // stages each one alone; the total stages them all in one batch.
        for staging in [1, 64, largest / 2, largest, largest * 4, 1 << 20] {
            let got = map
                .load_all_staged(&Device::Cpu, &mut |_, _| {}, staging)
                .unwrap();
            assert_eq!(got.len(), expected.len(), "staging={staging}");
            for (name, want) in &expected {
                let have = got
                    .get(name)
                    .unwrap_or_else(|| panic!("missing {name} at staging={staging}"));
                assert_eq!(have.shape(), want.shape(), "{name} staging={staging}");
                assert_eq!(have.dtype(), want.dtype(), "{name} staging={staging}");
                assert_eq!(
                    have.data().unwrap(),
                    want.data().unwrap(),
                    "{name} staging={staging}"
                );
            }
        }
    }

    /// A staging size below the largest tensor really does take the fallback,
    /// rather than the test above passing because every size took one path.
    #[test]
    fn a_tensor_larger_than_the_staging_buffer_is_read_from_the_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        write_fixture(&path, FIXTURE_ALIGNMENT);

        let map = GgufMmap::open(&path).unwrap();
        let sizes: Vec<usize> = map
            .tensor_infos()
            .values()
            .filter_map(tensor_payload_len)
            .collect();
        let largest = sizes.iter().copied().max().unwrap();
        let smallest = sizes.iter().copied().min().unwrap();
        assert!(
            smallest < largest,
            "the fixture must hold tensors of different sizes for this to mean anything"
        );

        // Between the two: the small tensors stage, the large one cannot.
        let staging = largest - 1;
        let got = map
            .load_all_staged(&Device::Cpu, &mut |_, _| {}, staging)
            .unwrap();
        let expected = map
            .load_all_from_mapping(&Device::Cpu, &mut |_, _| {})
            .unwrap();
        for (name, want) in &expected {
            assert_eq!(
                got[name].data().unwrap(),
                want.data().unwrap(),
                "{name} staging={staging}"
            );
        }
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
