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
/// `VarBuilder::from_tensors`. The resulting tensors are owned and survive
/// after the mapping is released.
///
/// ONE copy per tensor, out of a mapping. The previous implementation called
/// `candle_core::safetensors::load`, whose doc comment here claimed it mmaps:
/// it does not. It is `std::fs::read` into a `Vec<u8>`
/// (`candle-core/src/safetensors.rs:408-411`) followed by `view.load(device)`
/// per tensor (`:413-419`), so parking a 9.79 GB `t5xxl_fp16` allocated a
/// 9.79 GB anonymous staging buffer and then copied every tensor out of it
/// again — a transient peak of twice the encoder for a feature whose whole
/// purpose is to fit the encoder in host RAM.
///
/// Use this when you don't already have the model's tensors in hand and want
/// to populate the parked state from disk.
pub(crate) fn load_tensors_to_cpu(paths: &[impl AsRef<Path>]) -> Result<HashMap<String, Tensor>> {
    load_tensors_to_cpu_filtered(paths, |_| true)
}

/// [`load_tensors_to_cpu`] over the subset of tensor names `include` accepts.
///
/// A filtered-out tensor is never materialized at all — the mapping's pages
/// for it are never touched. That matters where a checkpoint carries more than
/// the runtime uses: FLUX.2 [dev]'s single-file Mistral3 republication ships a
/// vision tower, a projector, and decoder layers 30-39 beside the prefix the
/// encoder streams, and parking the file whole would charge host RAM for every
/// byte of them.
pub(crate) fn load_tensors_to_cpu_filtered(
    paths: &[impl AsRef<Path>],
    include: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Tensor>> {
    let refs: Vec<&Path> = paths.iter().map(|path| path.as_ref()).collect();
    // SAFETY: the mapping is consumed inside this call and every tensor is
    // copied out of it before it is released. Model weights are verified at
    // download and immutable thereafter — the same contract every other
    // mmap'd checkpoint in mold takes.
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&refs) }
        .with_context(|| {
            format!(
                "failed to park-load {}",
                refs.first()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default()
            )
        })?;
    let mut combined: HashMap<String, Tensor> = HashMap::new();
    // Later shards win on collisions, matching candle's behavior — but
    // safetensors shards from a single model never collide on tensor names.
    for (name, _) in mapped.tensors() {
        if !include(&name) {
            continue;
        }
        let tensor = mapped
            .load(&name, &Device::Cpu)
            .with_context(|| format!("failed to park-load tensor {name}"))?;
        combined.insert(name, tensor);
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

    /// A filtered-out tensor is not loaded at all.
    ///
    /// The filter is the whole point of the mapping: a checkpoint may carry
    /// far more than the runtime uses, and parking it whole charges host RAM
    /// for every byte. A filter that merely dropped entries afterwards would
    /// still have paid for them.
    #[test]
    fn a_filtered_out_tensor_is_never_loaded() {
        let path = temp_safetensors(
            "filtered",
            &[
                ("prefix.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ("vision_tower.weight", vec![9.0, 9.0], vec![2]),
            ],
        );

        let visited = std::cell::RefCell::new(Vec::new());
        let map = load_tensors_to_cpu_filtered(std::slice::from_ref(&path), |name| {
            visited.borrow_mut().push(name.to_string());
            name.starts_with("prefix.")
        })
        .unwrap();

        assert_eq!(map.len(), 1);
        assert!(map.contains_key("prefix.weight"));
        assert!(
            !map.contains_key("vision_tower.weight"),
            "the filter must exclude, not merely reorder"
        );
        let visited = visited.into_inner();
        assert!(
            visited.contains(&"vision_tower.weight".to_string()),
            "the filter is asked about every tensor in the header"
        );

        let unfiltered = load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();
        assert_eq!(unfiltered.len(), 2, "the default admits everything");

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

    /// **A parked tensor is the checkpoint's bytes, exactly, after the mapping
    /// is gone.**
    ///
    /// The park is the only path that hands the encoder weights it did not
    /// read from a live mapping, so its fidelity is the one thing that cannot
    /// be inferred from the mapped path working. Every published dtype in a
    /// FLUX.2 [dev] Mistral3 shard is checked bit-for-bit — BF16 is the one
    /// that actually ships, and F16/F32 ride along because the loader is
    /// dtype-generic and a silent reinterpretation between two 16-bit dtypes
    /// would produce exactly the plausible-looking garbage a NaN hunt starts
    /// from.
    ///
    /// The mapping is dropped BEFORE anything is compared: `load_tensors_to_cpu`
    /// copies out of the mapping rather than borrowing it, and a regression to a
    /// borrow would leave these reads dangling instead of merely wrong.
    #[test]
    fn a_parked_tensor_round_trips_its_exact_bytes_dtype_and_shape() {
        use half::{bf16, f16};

        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-park-dtypes-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // Values chosen so a BF16/F16 mix-up cannot pass: each is exactly
        // representable in its own dtype and lands somewhere else in the other.
        let bf16_values: Vec<bf16> = [-2.5f32, 0.0, 1.5, 65536.0, -0.00390625, 3.25]
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
        let f16_values: Vec<f16> = [-2.5f32, 0.0, 1.5, 2048.0, 0.00048828125, 3.25]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let f32_values: Vec<f32> = vec![-2.5, 0.0, 1.5, 65536.0, -0.00390625, 3.25];

        let mut bf16_bytes = Vec::new();
        for v in &bf16_values {
            bf16_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let mut f16_bytes = Vec::new();
        for v in &f16_values {
            f16_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let mut f32_bytes = Vec::new();
        for v in &f32_values {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let mut tensors: StdHashMap<String, TensorView> = StdHashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorView::new(SafeDtype::BF16, vec![2, 3], &bf16_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorView::new(SafeDtype::F16, vec![6], &f16_bytes).unwrap(),
        );
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![3, 2], &f32_bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let parked = load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();
        // The file itself is gone before a single value is read back.
        std::fs::remove_file(&path).unwrap();

        assert_eq!(parked.len(), 3);

        let q = parked
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q.dtype(), DType::BF16, "the on-disk dtype is preserved");
        assert_eq!(q.shape().dims(), &[2, 3]);
        assert!(q.device().is_cpu());
        let got: Vec<bf16> = q.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bf16_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "BF16 weights must survive the park bit-for-bit"
        );

        let norm = parked.get("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(norm.dtype(), DType::F16);
        assert_eq!(norm.shape().dims(), &[6]);
        let got: Vec<f16> = norm.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            f16_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let embed = parked.get("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.dtype(), DType::F32);
        assert_eq!(embed.shape().dims(), &[3, 2]);
        let got: Vec<f32> = embed.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            f32_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // And the VarBuilder over the park answers with the same bytes at the
        // compute dtype the encoder asks for — the encoder addresses the park
        // and the mapping through identical keys, so a get that disagreed here
        // would disagree silently in a render.
        let vb = varbuilder_from_parked(&parked, DType::BF16, &Device::Cpu)
            .pp("model")
            .pp("layers")
            .pp(0);
        let weight = vb
            .pp("self_attn")
            .get((2, 3), "q_proj.weight")
            .expect("the park keeps the checkpoint's own key namespace");
        assert_eq!(weight.dtype(), DType::BF16);
        let got: Vec<bf16> = weight.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bf16_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Page-locking a parked tensor does not change a byte of it.
    ///
    /// `park_prefix(pinned = true)` registers every parked tensor with
    /// `cuMemHostRegister`. That is a DMA optimization and nothing else; it is
    /// asserted here because the FLUX.2 [dev] NaN was first suspected of being
    /// corruption in the pin. (It was not — it was the unsettled prefetch; see
    /// `encoders::mistral3::stream_layers`.) On a non-CUDA build the pin is a
    /// no-op, which is exactly what this then asserts.
    #[test]
    fn pinning_a_parked_tensor_leaves_its_values_alone() {
        let path = temp_safetensors(
            "pinned",
            &[("weight", vec![1.5, -2.25, 0.0, 4.75], vec![2, 2])],
        );
        let parked = load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();
        let before: Vec<f32> = parked
            .get("weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let tracker =
            crate::flux::pinned::PinnedMemoryTracker::new(crate::flux::pinned::pinned_cap_bytes());
        let mut regions = Vec::new();
        for tensor in parked.values() {
            if let Ok(Some(region)) = crate::flux::pinned::try_pin_to_host(tensor, &tracker) {
                regions.push(region);
            }
        }

        let after: Vec<f32> = parked
            .get("weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(before, after, "pinning must not touch the parked values");
        drop(regions);
        let unregistered: Vec<f32> = parked
            .get("weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(before, unregistered, "unpinning must not touch them either");

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
