//! Shared park-on-CPU plumbing for text encoders.
//!
//! When `MOLD_KEEP_TE_RAM=1`, encoders call `park_to_cpu()` after encoding
//! finishes instead of `drop_weights()`. Parking moves all encoder
//! parameters into a `HashMap<String, Tensor>` on host RAM, then drops the
//! GPU-resident model. Unparking takes the same HashMap, hands it to
//! `VarBuilder::from_tensors(map, dtype, target_device)`, and reconstructs
//! the model fresh on the target device — the H2D copy happens inside the
//! VarBuilder backend.
//!
//! Cost model:
//! - First park after load: a full disk read (safetensors → CPU tensors).
//!   Same wall-clock cost as the existing `reload()` would have paid on the
//!   *next* request.
//! - Subsequent unpark/park cycles: no disk I/O, just GPU↔CPU tensor copies.
//!   Saves ~2-4 s on FLUX (T5-XXL fp16, ~9 GB), ~1 s on SD3.
//!
//! The HashMap-of-CPU-tensors approach is intentionally kept narrow: it only
//! handles **safetensors-backed** encoders where the model is a pure function
//! of a `VarBuilder<'static>`. Quantized GGUF encoders fall through to the
//! existing `drop_weights`/`reload` path because their `QTensor` storage is
//! device-tied and not trivially walkable. This is deliberate — keeping
//! GGUF-with-park out of scope means callers get correct behavior
//! everywhere, even if the savings are smaller for the quantized variants
//! (which already reload faster than fp16).
//!
//! ComfyUI's equivalent: `model_management.py:1012`
//! (`text_encoder_offload_device()`).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Load every tensor in `paths` (safetensors files) onto `Device::Cpu`,
/// returning a `name → Tensor` map suitable for handing to
/// `VarBuilder::from_tensors`.
///
/// Reads each file via candle's standard `safetensors::load`, which mmaps
/// internally and copies tensor bytes into a fresh CPU buffer (no lifetime
/// tied to the mmap). The resulting tensors are owned and survive after
/// the file handle closes.
///
/// Use this when you don't already have the model's tensors in hand and want
/// to populate the parked state from disk.
pub(crate) fn load_tensors_to_cpu(paths: &[impl AsRef<Path>]) -> Result<HashMap<String, Tensor>> {
    let mut combined: HashMap<String, Tensor> = HashMap::new();
    for path in paths {
        let map = candle_core::safetensors::load(path.as_ref(), &Device::Cpu)
            .with_context(|| format!("failed to park-load {}", path.as_ref().display()))?;
        // Later shards win on collisions, matching candle's behavior — but
        // safetensors shards from a single model never collide on tensor names.
        combined.extend(map);
    }
    Ok(combined)
}

/// Convert a parked HashMap into a `VarBuilder` rooted on `target_device`
/// with the given compute dtype. The backend's `get()` does the H2D copy
/// transparently when the model is reconstructed.
///
/// We clone the map into the backend so the encoder can keep the original
/// CPU-resident copy for the next park/unpark cycle without re-reading
/// from disk.
pub(crate) fn varbuilder_from_parked<'a>(
    parked: &HashMap<String, Tensor>,
    dtype: DType,
    target_device: &Device,
) -> candle_nn::VarBuilder<'a> {
    candle_nn::VarBuilder::from_tensors(parked.clone(), dtype, target_device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Linear, Module, VarBuilder};
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap as StdHashMap;

    fn temp_safetensors(name: &str, kvs: &[(&str, Vec<f32>, Vec<usize>)]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-park-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut bufs: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (k, v, shape) in kvs {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for f in v {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            bufs.push(((*k).to_string(), bytes, shape.clone()));
        }
        let mut tensors: StdHashMap<String, TensorView> = StdHashMap::new();
        for (k, b, shape) in &bufs {
            tensors.insert(
                k.clone(),
                TensorView::new(SafeDtype::F32, shape.clone(), b).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, &path).unwrap();
        path
    }

    /// `load_tensors_to_cpu` returns CPU tensors with the right shapes, ready
    /// to feed back into a VarBuilder for model reconstruction.
    #[test]
    fn load_tensors_to_cpu_returns_owned_cpu_tensors() {
        let path = temp_safetensors(
            "load",
            &[
                ("weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ("bias", vec![5.0, 6.0], vec![2]),
            ],
        );
        let map = load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();
        assert_eq!(map.len(), 2);
        let w = map.get("weight").unwrap();
        assert_eq!(w.shape().dims(), &[2, 2]);
        assert!(w.device().is_cpu());
        assert_eq!(w.dtype(), DType::F32);

        let _ = std::fs::remove_file(&path);
    }

    /// Round-trip: park a Linear's tensors into CPU, rebuild VarBuilder
    /// from the parked map, and confirm the reconstructed Linear produces
    /// bit-identical output for the same input. This is the core invariant
    /// that all the encoder-specific park/unpark methods rely on.
    #[test]
    fn test_park_unpark_roundtrip_linear() {
        // Build a Linear from a safetensors file the same way encoders do
        let path = temp_safetensors(
            "linear",
            &[
                ("weight", vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
                ("bias", vec![0.5, -0.5], vec![2]),
            ],
        );

        // Original VB → original Linear
        let vb_orig = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu).unwrap()
        };
        let lin_orig = Linear::new(
            vb_orig.get((2, 2), "weight").unwrap(),
            Some(vb_orig.get(2, "bias").unwrap()),
        );

        // Park: load to CPU map (here Device::Cpu is the "GPU" target since
        // we're running on a host-only test environment)
        let parked = load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();

        // Unpark: rebuild VB from the parked map → reconstructed Linear
        let vb_unpark = varbuilder_from_parked(&parked, DType::F32, &Device::Cpu);
        let lin_new = Linear::new(
            vb_unpark.get((2, 2), "weight").unwrap(),
            Some(vb_unpark.get(2, "bias").unwrap()),
        );

        // Same input through both → identical outputs
        let x = Tensor::from_slice(&[1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let y_orig = lin_orig.forward(&x).unwrap();
        let y_new = lin_new.forward(&x).unwrap();

        let v_orig: Vec<f32> = y_orig.flatten_all().unwrap().to_vec1().unwrap();
        let v_new: Vec<f32> = y_new.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            v_orig, v_new,
            "park→unpark must be bit-identical (same dtype, same device, no lossy ops)"
        );

        // Park again from the same parked map should still work — `parked`
        // is .clone()'d into the VarBuilder, not consumed.
        let vb_again = varbuilder_from_parked(&parked, DType::F32, &Device::Cpu);
        let _ = vb_again.get((2, 2), "weight").unwrap();

        let _ = std::fs::remove_file(&path);
    }

    /// Missing tensor → clear error from the backend (not a panic).
    #[test]
    fn varbuilder_from_parked_errors_on_missing_tensor() {
        let parked: HashMap<String, Tensor> = HashMap::new();
        let vb = varbuilder_from_parked(&parked, DType::F32, &Device::Cpu);
        let err = vb.get((2, 2), "weight").unwrap_err();
        assert!(
            err.to_string().contains("weight") || err.to_string().contains("Cannot find"),
            "expected missing-tensor error, got: {err}"
        );
    }
}
