//! Exact tensor consumption for published paint PyTorch checkpoints.
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, VarBuilder};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

struct State {
    tensors: candle_core::pickle::PthTensors,
    consumed: Mutex<BTreeSet<String>>,
}
struct Backend(Arc<State>);
impl SimpleBackend for Backend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hint: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let info = self
            .0
            .tensors
            .tensor_infos()
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing paint tensor {name}")))?;
        if info.layout.dims() != shape.dims() {
            candle_core::bail!(
                "paint tensor {name} has {:?}, expected {:?}",
                info.layout.dims(),
                shape.dims()
            );
        }
        self.get_unchecked(name, dtype, device)
    }
    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self
            .0
            .tensors
            .get(name)?
            .ok_or_else(|| candle_core::Error::Msg(format!("missing paint tensor {name}")))?;
        self.0
            .consumed
            .lock()
            .map_err(|_| candle_core::Error::Msg("paint tensor inventory lock poisoned".into()))?
            .insert(name.into());
        tensor.to_device(device)?.to_dtype(dtype)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.tensors.tensor_infos().contains_key(name)
    }
}

/// Reject incompatible storage before allocating model tensors, then require
/// the constructed network to consume the entire checkpoint's tensor inventory.
/// Candle interprets tensor serialization directly; no Python runtime executes.
pub(super) fn load_pth_exact<T>(
    path: &Path,
    dtype: DType,
    device: &Device,
    build: impl FnOnce(VarBuilder<'static>) -> candle_core::Result<T>,
) -> Result<T> {
    let tensors = candle_core::pickle::PthTensors::new(path, None)?;
    ensure!(
        !tensors.tensor_infos().is_empty() && tensors.tensor_infos().len() <= 20_000,
        "invalid paint tensor count"
    );
    for (name, info) in tensors.tensor_infos() {
        ensure!(
            matches!(info.dtype, DType::F32 | DType::F16 | DType::BF16),
            "paint tensor {name} has unsupported dtype {:?}",
            info.dtype
        );
        ensure!(
            info.layout.is_contiguous() && info.layout.start_offset() == 0,
            "paint tensor {name} has unsupported strided or offset storage"
        );
        let count = info
            .layout
            .dims()
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim));
        ensure!(
            count.is_some_and(|count| count > 0 && count <= 1_000_000_000),
            "paint tensor {name} has invalid dimensions"
        );
    }
    let state = Arc::new(State {
        tensors,
        consumed: Mutex::new(BTreeSet::new()),
    });
    let builder = VarBuilder::from_backend(Box::new(Backend(state.clone())), dtype, device.clone());
    let result = build(builder)?;
    let used = state
        .consumed
        .lock()
        .map_err(|_| anyhow::anyhow!("paint tensor inventory lock poisoned"))?;
    let mut unused: Vec<_> = state
        .tensors
        .tensor_infos()
        .keys()
        .filter(|name| !used.contains(*name))
        .collect();
    unused.sort();
    ensure!(
        unused.is_empty(),
        "paint checkpoint has {} unconsumed tensors: {:?}",
        unused.len(),
        &unused[..unused.len().min(5)]
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_real_pytorch_serialization_and_refuses_unused_or_wrong_tensors() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.bin");
        std::fs::write(
            &path,
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-vae-tiny.bin"),
        )
        .unwrap();
        let cfg = mold_candle::stable_diffusion::vae::AutoEncoderKLConfig {
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 4,
            use_quant_conv: true,
            use_post_quant_conv: true,
        };
        load_pth_exact(&path, DType::F32, &Device::Cpu, |vb| {
            mold_candle::stable_diffusion::vae::AutoEncoderKL::new(vb, 3, 3, cfg)
        })
        .unwrap();
        assert!(load_pth_exact(&path, DType::F32, &Device::Cpu, |vb| vb
            .get(8, "encoder.conv_in.bias"))
        .is_err());
        assert!(load_pth_exact(&path, DType::F32, &Device::Cpu, |vb| vb
            .get(9, "encoder.conv_in.bias"))
        .is_err());
    }
}
