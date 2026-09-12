//! Quantized helpers that compose Candle's public storage APIs.

use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{Device, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Re-quantize a CPU tensor and transfer its compact bytes with one target
/// device allocation.
///
/// Candle's generic `QTensor::quantize_onto` allocates the destination before
/// quantization. Mold's LoRA merge path already has a CPU F32 tensor, so doing
/// the quantization on CPU and transferring the finished bytes avoids a second
/// full-sized target allocation and allocator fragmentation.
pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, device: &Device) -> Result<QTensor> {
    if !src.device().is_cpu() {
        candle::bail!("quantize_onto expects a CPU source, got {:?}", src.device())
    }

    let shape = src.shape().clone();
    let quantized = QTensor::quantize(src, dtype)?;
    if device.is_cpu() {
        return Ok(quantized);
    }

    let storage = QStorage::from_data(quantized.data()?, device, dtype)?;
    QTensor::new(storage, shape)
}

/// Mold-owned in-memory quantized variable builder.
///
/// Upstream's builder only constructs from GGUF files. LoRA merging produces
/// an already-loaded tensor map, so Mold owns the small composition layer
/// instead of adding an application-specific constructor to Candle.
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    /// Load a GGUF checkpoint from a memory mapping.
    ///
    /// See [`crate::gguf_mmap`] for why: the reader loop this replaced copied
    /// every byte of the file into a fresh anonymous buffer before uploading
    /// it, which measured 0.96 GB/s on a 33 GB checkpoint against 8.3 GB/s for
    /// the same file read by stable-diffusion.cpp.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        Self::from_gguf_with_progress(path, device, &mut |_, _| {})
    }

    /// [`Self::from_gguf`] reporting `(bytes_done, bytes_total)` as it loads.
    pub fn from_gguf_with_progress<P: AsRef<std::path::Path>>(
        path: P,
        device: &Device,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Self> {
        let map = crate::gguf_mmap::GgufMmap::open(path)?;
        let data = map.load_all(device, progress)?;
        Ok(Self::from_qtensors(data, device))
    }

    /// The pre-mapping reader loop, kept for callers holding a reader rather
    /// than a path (and as the oracle the mapping is tested against).
    pub fn from_gguf_reader<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let content = candle::quantized::gguf_file::Content::read(reader)?;
        let mut data = HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(reader, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self::from_qtensors(data, device))
    }

    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        Self::from_gguf_reader(&mut std::io::Cursor::new(buffer), device)
    }

    pub fn from_qtensors(data: HashMap<String, Arc<QTensor>>, device: &Device) -> Self {
        Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        }
    }

    pub fn pp<S: ToString>(&self, segment: S) -> Self {
        let mut path = self.path.clone();
        path.push(segment.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), tensor_name)
        }
    }

    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        let tensor = self
            .data
            .get(&path)
            .ok_or_else(|| candle::Error::Msg(format!("cannot find tensor {path}")))?;
        let expected = shape.into();
        if tensor.shape() != &expected {
            candle::bail!(
                "shape mismatch for {path}, got {:?}, expected {expected:?}",
                tensor.shape()
            )
        }
        Ok(tensor.clone())
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        self.data
            .get(&path)
            .cloned()
            .ok_or_else(|| candle::Error::Msg(format!("cannot find tensor {path}")))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The full key `name` resolves to under the current path.
    ///
    /// Callers that key side-tables (LoRA patches, per-tensor overrides) on the
    /// checkpoint's own names need the same string `get` would look up, and
    /// rebuilding it from a borrowed prefix is how the two drift apart.
    pub fn key(&self, name: &str) -> String {
        self.path(name)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(&self.path(key))
    }

    /// Every tensor this builder holds, keyed by its full checkpoint name.
    ///
    /// Ignores the current `pp` path deliberately: the one caller is Wan's
    /// block offload, which selects a block's weights by name prefix and hands
    /// the result straight back to [`Self::from_qtensors`], so it needs the
    /// same absolute keys `get` resolves against rather than a view relative
    /// to wherever the builder happens to be positioned.
    pub fn tensors(&self) -> &HashMap<String, Arc<QTensor>> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_resolves_prefixed_tensors() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1f32; 32], (1, 32), &Device::Cpu)?;
        let tensor = Arc::new(QTensor::quantize(&tensor, GgmlDType::Q8_0)?);
        let mut tensors = HashMap::new();
        tensors.insert("block.weight".to_string(), tensor.clone());

        let builder = VarBuilder::from_qtensors(tensors, &Device::Cpu).pp("block");
        assert!(builder.contains_key("weight"));
        assert_eq!(builder.get((1, 32), "weight")?.shape(), tensor.shape());
        Ok(())
    }

    #[test]
    fn quantize_onto_cpu_preserves_shape_and_dtype() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.5f32; 32], (1, 32), &Device::Cpu)?;
        let quantized = quantize_onto(&tensor, GgmlDType::Q8_0, &Device::Cpu)?;
        assert_eq!(quantized.shape(), tensor.shape());
        assert_eq!(quantized.dtype(), GgmlDType::Q8_0);
        Ok(())
    }

    #[test]
    fn key_matches_what_get_looks_up() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1f32; 32], (1, 32), &Device::Cpu)?;
        let tensor = Arc::new(QTensor::quantize(&tensor, GgmlDType::Q8_0)?);
        let mut tensors = HashMap::new();
        tensors.insert("block.0.attn.weight".to_string(), tensor);

        let builder = VarBuilder::from_qtensors(tensors, &Device::Cpu);
        assert_eq!(builder.key("weight"), "weight");
        let nested = builder.pp("block").pp("0").pp("attn");
        assert_eq!(nested.key("weight"), "block.0.attn.weight");
        assert!(nested.contains_key("weight"));
        assert!(nested.get((1, 32), "weight").is_ok());
        Ok(())
    }
}
