//! Utility for loading safetensors model weights via lazy memory-mapped I/O.
//!
//! Wraps candle's `VarBuilder::from_mmaped_safetensors()` with progress events.
//! Only the safetensors header is parsed upfront — tensor data loads on demand
//! via OS page faults during model construction.

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use safetensors::tensor::TensorInfo;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};

/// Load a GGUF checkpoint into mold's own quantized `VarBuilder`, from a
/// memory mapping and with a byte counter.
///
/// The two GGUF loader entry points below exist so a call site names its stage
/// label once and gets both the mapping and the progress bar. See
/// [`mold_candle::gguf_mmap`] for why the mapping matters: the reader loop it
/// replaced copied every byte of the checkpoint into a fresh anonymous buffer
/// before uploading it, measured at 0.96 GB/s on a 33 GB file.
pub(crate) fn load_gguf_var_builder(
    path: &Path,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<mold_candle::quantized::VarBuilder> {
    mold_candle::quantized::VarBuilder::from_gguf_with_progress(path, device, &mut |done, total| {
        progress.weight_load(component, done, total)
    })
    .map_err(Into::into)
}

/// VarBuilder backend for FP8 safetensors that preserves native dtypes.
///
/// Loads tensors at their on-disk dtype: F8E4M3 weights stay F8E4M3 on GPU,
/// BF16 biases/norms stay BF16. The transformer's `QwenLinear::Fp8` handles
/// per-layer FP8→BF16 dequantization (with optional scale) during forward.
pub(crate) struct NativeFp8Backend {
    inner: candle_core::safetensors::MmapedSafetensors,
}

pub(crate) struct AliasSafetensorsBackend {
    inner: candle_core::safetensors::MmapedSafetensors,
    aliases: BTreeMap<String, String>,
}

impl NativeFp8Backend {
    /// Construct a backend directly from an already-mmapped tensor set.
    /// Used by family-specific LoRA paths (e.g. `qwen_image::lora`) to
    /// wrap the FP8 backend before the LoRA `SimpleBackend` layer.
    pub(crate) fn from_mmap(inner: candle_core::safetensors::MmapedSafetensors) -> Self {
        Self { inner }
    }
}

fn total_file_bytes(paths: &[impl AsRef<Path>]) -> u64 {
    paths
        .iter()
        .map(|p| std::fs::metadata(p.as_ref()).map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// The largest JSON header the `safetensors` crate itself accepts.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

pub(crate) fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, Value>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    // A downloaded file names this length, so it is checked before it sizes an
    // allocation: an absurd value would abort the process.
    anyhow::ensure!(
        header_len <= MAX_SAFETENSORS_HEADER_BYTES && header_len <= file_len.saturating_sub(8),
        "{} declares a {header_len}-byte safetensors header",
        path.display()
    );
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)?;
    Ok(serde_json::from_slice(&header_buf)?)
}

fn filtered_safetensors_tensor_bytes(
    paths: &[impl AsRef<Path>],
    include_tensor: impl Fn(&str) -> bool,
) -> Result<u64> {
    let mut total = 0u64;
    for path in paths {
        let header = read_safetensors_header(path.as_ref())?;
        for (name, value) in header {
            if name == "__metadata__" || !include_tensor(&name) {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value)?;
            total += info.data_offsets.1.saturating_sub(info.data_offsets.0) as u64;
        }
    }
    Ok(total)
}

impl candle_nn::var_builder::SimpleBackend for NativeFp8Backend {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: candle_nn::Init,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Load at native dtype — no casting
        let tensor = self.inner.load(path, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        path: &str,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.inner.load(path, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

impl SimpleBackend for AliasSafetensorsBackend {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let source = self.aliases.get(path).map(String::as_str).unwrap_or(path);
        let tensor = load_stored_tensor(&self.inner, source, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, path: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let source = self.aliases.get(path).map(String::as_str).unwrap_or(path);
        load_stored_tensor(&self.inner, source, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
            || self
                .aliases
                .get(name)
                .is_some_and(|source| self.inner.get(source).is_ok())
    }
}

/// Whether converting `from` to `to` only adds precision, so performing it on
/// the CPU rather than a device kernel cannot change a single rendered byte.
pub(crate) fn widening_is_lossless(from: DType, to: DType) -> bool {
    matches!(
        (from, to),
        (DType::BF16 | DType::F16, DType::F32 | DType::F64) | (DType::F32, DType::F64)
    )
}

/// Where a stored weight of dtype `stored` should be read before it is cast to
/// `dtype` and placed on `dev`.
///
/// Metal widens half-precision weights on the CPU. The device path uploads the
/// stored tensor, casts it into a new buffer the pool rounds up to the next power
/// of two, and leaves the upload in the pool until something next waits on the
/// queue. A CPU cast reaches the device through one exactly sized upload: Z-Image
/// holds its 22.9 GiB of F32 weights in 22.9 GiB rather than 65 GiB.
pub(crate) fn weight_staging_device(stored: DType, dtype: DType, dev: &Device) -> Device {
    if dev.is_metal() && widening_is_lossless(stored, dtype) {
        Device::Cpu
    } else {
        dev.clone()
    }
}

/// Load one stored tensor onto `dev` at `dtype`. See [`weight_staging_device`].
pub(crate) fn load_stored_tensor(
    st: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
    dtype: DType,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    // Candle's loader converts the few stored types `DType` cannot name (U16).
    let staging = DType::try_from(st.get(name)?.dtype()).map_or_else(
        |_| dev.clone(),
        |stored| weight_staging_device(stored, dtype, dev),
    );
    st.load(name, &staging)?.to_dtype(dtype)?.to_device(dev)
}

/// Bytes the safetensors tensors in `header` occupy once half-precision floats are
/// held as F32. Other dtypes, such as FP8 weights consumed at their stored width,
/// count at their stored size.
fn widened_tensor_bytes(header: &BTreeMap<String, Value>) -> Option<u64> {
    let mut total = 0u64;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value.clone()).ok()?;
        let stored = info.data_offsets.1.saturating_sub(info.data_offsets.0) as u64;
        let widens = matches!(
            info.dtype,
            safetensors::tensor::Dtype::F16 | safetensors::tensor::Dtype::BF16
        );
        total = total.saturating_add(if widens {
            stored.saturating_mul(2)
        } else {
            stored
        });
    }
    Some(total)
}

/// What `path` occupies on a device that computes in F32, or `None` when it is not
/// a readable safetensors file (a GGUF stays quantized, so its file size is right).
pub fn safetensors_bytes_widened_to_f32(path: &Path) -> Option<u64> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return None;
    }
    widened_tensor_bytes(&read_safetensors_header(path).ok()?)
}

/// The mapped-safetensors backend for Metal, where widening a stored weight goes
/// through [`load_stored_tensor`]. Otherwise answers exactly as candle's own does,
/// including its rule that a non-FP8 tensor in an FP8 load lands at BF16.
struct MetalSafetensorsBackend {
    inner: candle_core::safetensors::MmapedSafetensors,
}

impl MetalSafetensorsBackend {
    fn load(&self, path: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let dtype = if dtype == DType::F8E4M3
            && DType::try_from(self.inner.get(path)?.dtype()).ok() != Some(DType::F8E4M3)
        {
            DType::BF16
        } else {
            dtype
        };
        load_stored_tensor(&self.inner, path, dtype, dev)
    }
}

impl SimpleBackend for MetalSafetensorsBackend {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load(path, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, path: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(path, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

/// A lazy-mmap [`VarBuilder`] over `paths`.
///
/// # Safety
/// As [`VarBuilder::from_mmaped_safetensors`]: the files must not be modified
/// while the builder is alive.
unsafe fn mapped_var_builder<'a>(
    paths: &[&Path],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'a>> {
    if !device.is_metal() {
        return unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) };
    }
    let inner = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
    Ok(VarBuilder::from_backend(
        Box::new(MetalSafetensorsBackend { inner }),
        dtype,
        device.clone(),
    ))
}

/// Load FP8 safetensors preserving native dtypes on the target device.
///
/// F8E4M3 weights stay as F8E4M3 in VRAM (~19.5GB for full model).
/// BF16 biases/norms/scales stay as BF16. The transformer's `QwenLinear::Fp8`
/// handles per-layer dequantization during forward (ComfyUI "manual cast" style).
pub fn load_fp8_safetensors<'a>(
    paths: &[impl AsRef<Path>],
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    load_native_safetensors_with_progress(paths, DType::BF16, device, component, progress)
}

/// Load mixed native-dtype safetensors while retaining the caller's compute
/// dtype on the VarBuilder. Family loaders whose numerics require F16 use
/// this instead of the BF16-default FP8 helper.
pub(crate) fn load_native_safetensors_with_progress<'a>(
    paths: &[impl AsRef<Path>],
    compute_dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    let bytes_total = total_file_bytes(paths);

    progress.weight_load(component, 0, bytes_total);

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
    let backend = NativeFp8Backend { inner: tensors };
    let vb = VarBuilder::from_backend(Box::new(backend), compute_dtype, device.clone());

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

pub(crate) fn load_fp8_safetensors_with_callback<'a>(
    paths: &[impl AsRef<Path>],
    device: &Device,
    component: &str,
    progress: Option<&ProgressCallback>,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    let bytes_total = total_file_bytes(paths);

    emit_weight_load(progress, component, 0, bytes_total);

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
    let backend = NativeFp8Backend { inner: tensors };
    let vb = VarBuilder::from_backend(Box::new(backend), DType::BF16, device.clone());

    emit_weight_load(progress, component, bytes_total, bytes_total);

    Ok(vb)
}

fn load_safetensors_with_progress_total<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
    bytes_total: u64,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();

    progress.weight_load(component, 0, bytes_total);

    let vb = unsafe { mapped_var_builder(&path_refs, dtype, device)? };

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

fn emit_weight_load(
    progress: Option<&ProgressCallback>,
    component: &str,
    bytes_loaded: u64,
    bytes_total: u64,
) {
    if let Some(progress) = progress {
        progress(ProgressEvent::WeightLoad {
            bytes_loaded,
            bytes_total,
            component: component.to_string(),
        });
    }
}

/// Load safetensors via lazy mmap but report progress for only the tensors that
/// match `include_tensor`. This is useful when a shared shard set contains a
/// much larger model than the submodule we actually instantiate, such as the
/// Qwen2.5-VL vision tower embedded inside the shared text-encoder shards.
pub fn load_safetensors_with_filtered_progress<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
    include_tensor: impl Fn(&str) -> bool,
) -> Result<VarBuilder<'a>> {
    let bytes_total = filtered_safetensors_tensor_bytes(paths, include_tensor)
        .unwrap_or_else(|_| total_file_bytes(paths));
    load_safetensors_with_progress_total(paths, dtype, device, component, progress, bytes_total)
}

/// Load safetensors files via lazy mmap with progress events.
///
/// `component` is the human-readable label (e.g. "FLUX transformer", "VAE").
/// Emits start/complete `WeightLoad` events; actual tensor I/O is deferred
/// to model construction via OS page faults.
pub fn load_safetensors_with_progress<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    let bytes_total = total_file_bytes(paths);
    load_safetensors_with_progress_total(paths, dtype, device, component, progress, bytes_total)
}

pub(crate) fn load_safetensors_with_aliases<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
    aliases: BTreeMap<String, String>,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    let bytes_total = total_file_bytes(paths);

    progress.weight_load(component, 0, bytes_total);

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
    let backend = AliasSafetensorsBackend {
        inner: tensors,
        aliases,
    };
    let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

pub(crate) fn load_safetensors_with_progress_callback<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: Option<&ProgressCallback>,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    let bytes_total = total_file_bytes(paths);

    emit_weight_load(progress, component, 0, bytes_total);

    let vb = unsafe { mapped_var_builder(&path_refs, dtype, device)? };

    emit_weight_load(progress, component, bytes_total, bytes_total);

    Ok(vb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::ProgressEvent;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[test]
    fn widened_size_counts_only_half_precision_floats_twice() {
        let path = temp_file("widened-size");
        let half_precision = vec![0u8; 400];
        let single_precision = vec![0u8; 400];
        let fp8 = vec![0u8; 100];
        let mut tensors = HashMap::new();
        tensors.insert(
            "bf16".to_string(),
            TensorView::new(SafeDtype::BF16, vec![200], half_precision.as_slice()).unwrap(),
        );
        tensors.insert(
            "f16".to_string(),
            TensorView::new(SafeDtype::F16, vec![200], half_precision.as_slice()).unwrap(),
        );
        tensors.insert(
            "f32".to_string(),
            TensorView::new(SafeDtype::F32, vec![100], single_precision.as_slice()).unwrap(),
        );
        tensors.insert(
            "u8".to_string(),
            TensorView::new(SafeDtype::U8, vec![100], fp8.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();
        assert_eq!(
            safetensors_bytes_widened_to_f32(&path),
            Some(800 + 800 + 400 + 100)
        );
        std::fs::remove_file(&path).ok();
        assert_eq!(
            safetensors_bytes_widened_to_f32(Path::new("/nonexistent/x.safetensors")),
            None
        );
        assert_eq!(
            safetensors_bytes_widened_to_f32(Path::new("model.gguf")),
            None,
            "a GGUF stays quantized"
        );
    }

    #[test]
    fn stored_types_dtype_cannot_name_still_load() {
        let path = temp_file("stored-u16");
        let bytes = 7u16.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorView::new(SafeDtype::U16, vec![1], bytes.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let t = load_stored_tensor(&st, "w", DType::U32, &Device::Cpu).unwrap();
        assert_eq!(t.to_vec1::<u32>().unwrap(), vec![7]);
        std::fs::remove_file(path).ok();
    }

    /// The length is read from the file, so it must not size an allocation unchecked.
    #[test]
    fn an_impossible_header_length_is_an_error_rather_than_an_abort() {
        for declared in [u64::MAX, MAX_SAFETENSORS_HEADER_BYTES + 1, 4096] {
            let path = temp_file("huge-header");
            let mut contents = declared.to_le_bytes().to_vec();
            contents.extend_from_slice(b"{}");
            std::fs::write(&path, &contents).unwrap();
            assert!(read_safetensors_header(&path).is_err(), "{declared}");
            std::fs::remove_file(path).ok();
        }
    }

    /// Widening on the CPU is only safe because it cannot move a pixel. A
    /// narrowing cast keeps the device's own kernel, whose rounding may differ.
    #[test]
    fn only_lossless_widening_leaves_the_device_kernel() {
        for (from, to) in [
            (DType::BF16, DType::F32),
            (DType::F16, DType::F32),
            (DType::BF16, DType::F64),
            (DType::F16, DType::F64),
            (DType::F32, DType::F64),
        ] {
            assert!(widening_is_lossless(from, to), "{from:?} -> {to:?}");
        }
        for (from, to) in [
            (DType::F32, DType::BF16),
            (DType::F32, DType::F16),
            (DType::BF16, DType::F16),
            (DType::F16, DType::BF16),
            (DType::F32, DType::F32),
            (DType::U8, DType::F32),
            (DType::F8E4M3, DType::F32),
        ] {
            assert!(!widening_is_lossless(from, to), "{from:?} -> {to:?}");
        }
    }

    #[test]
    fn stored_tensor_loads_at_the_requested_dtype_off_metal() {
        let path = temp_file("cpu-widen");
        let bytes = half::bf16::from_f32(1.5).to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorView::new(SafeDtype::BF16, vec![1], bytes.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.unwrap();
        let t = load_stored_tensor(&st, "w", DType::F32, &Device::Cpu).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.5]);
        std::fs::remove_file(path).ok();
    }

    /// The pool rounds every kernel output to the next power of two and keeps
    /// the half-precision upload it was cast from: Z-Image's 22.9 GiB of F32
    /// weights occupied 65 GiB through the whole denoise.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_widened_weight_occupies_exactly_its_own_bytes() {
        let device = crate::device::metal_device(0).unwrap();
        let path = temp_file("metal-widen");
        // 3000 F32 elements are 12,000 bytes; the pool would round to 16,384.
        let bytes: Vec<u8> = (0..3000u16)
            .flat_map(|v| half::bf16::from_f32(f32::from(v)).to_le_bytes())
            .collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorView::new(SafeDtype::BF16, vec![3000], bytes.as_slice()).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let vb = load_safetensors_with_progress(
            std::slice::from_ref(&path),
            DType::F32,
            &device,
            "test",
            &ProgressReporter::default(),
        )
        .unwrap();
        let t = vb.get(3000, "w").unwrap();
        assert_eq!(t.dtype(), DType::F32);
        {
            let (storage, _) = t.storage_and_layout();
            let candle_core::Storage::Metal(metal) = &*storage else {
                panic!("expected Metal storage");
            };
            assert_eq!(metal.buffer().length(), 12_000);
        }
        let values = t.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        let expected: Vec<f32> = (0..3000u16)
            .map(|v| half::bf16::from_f32(f32::from(v)).to_f32())
            .collect();
        assert_eq!(values, expected);
        std::fs::remove_file(path).ok();
    }

    fn temp_file(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-weight-loader-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    #[test]
    fn filtered_safetensors_tensor_bytes_counts_matching_tensors_only() {
        let path = temp_file("visual-bytes");
        let visual_data = vec![0u8; 16];
        let text_data = vec![0u8; 64];
        let mut tensors = HashMap::new();
        tensors.insert(
            "visual.patch_embed.proj.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &visual_data).unwrap(),
        );
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![4, 4], &text_data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let total = filtered_safetensors_tensor_bytes(std::slice::from_ref(&path), |name| {
            name.starts_with("visual.")
        })
        .unwrap();
        assert_eq!(total, visual_data.len() as u64);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn alias_backend_maps_missing_rms_scale_to_weight_suffix() {
        let path = temp_file("alias-rms");
        let data = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let mut aliases = BTreeMap::new();
        aliases.insert(
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale".to_string(),
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight".to_string(),
        );
        let backend = AliasSafetensorsBackend {
            inner: unsafe { candle_core::safetensors::MmapedSafetensors::new(&path).unwrap() },
            aliases,
        };
        let dev = Device::Cpu;

        assert!(SimpleBackend::contains_tensor(
            &backend,
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale"
        ));
        let tensor = SimpleBackend::get_unchecked(
            &backend,
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale",
            DType::F32,
            &dev,
        )
        .unwrap();
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![1.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_safetensors_with_progress_callback_emits_weight_load_events() {
        let path = temp_file("callback-progress");
        let data = vec![0u8; 16];
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let callback: crate::progress::ProgressCallback = Box::new(move |event| {
            sink.lock().unwrap().push(event);
        });

        let _vb = load_safetensors_with_progress_callback(
            std::slice::from_ref(&path),
            DType::F32,
            &Device::Cpu,
            "test component",
            Some(&callback),
        )
        .unwrap();

        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [
                ProgressEvent::WeightLoad {
                    bytes_loaded: 0,
                    bytes_total,
                    component
                },
                ProgressEvent::WeightLoad {
                    bytes_loaded,
                    bytes_total: bytes_total_done,
                    component: component_done
                }
            ] if *bytes_total >= data.len() as u64
                && bytes_loaded == bytes_total_done
                && bytes_total == bytes_total_done
                && component == "test component"
                && component_done == "test component"
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_fp8_safetensors_with_callback_emits_weight_load_events() {
        let path = temp_file("fp8-callback-progress");
        let data = vec![0u8; 16];
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let callback: crate::progress::ProgressCallback = Box::new(move |event| {
            sink.lock().unwrap().push(event);
        });

        let _vb = load_fp8_safetensors_with_callback(
            std::slice::from_ref(&path),
            &Device::Cpu,
            "test fp8 component",
            Some(&callback),
        )
        .unwrap();

        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [
                ProgressEvent::WeightLoad {
                    bytes_loaded: 0,
                    bytes_total,
                    component
                },
                ProgressEvent::WeightLoad {
                    bytes_loaded,
                    bytes_total: bytes_total_done,
                    component: component_done
                }
            ] if *bytes_total >= data.len() as u64
                && bytes_loaded == bytes_total_done
                && bytes_total == bytes_total_done
                && component == "test fp8 component"
                && component_done == "test fp8 component"
        ));

        let _ = std::fs::remove_file(path);
    }
}
