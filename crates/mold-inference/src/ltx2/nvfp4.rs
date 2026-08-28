use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::collections::BTreeSet;
use std::path::Path;

const DIFFUSION_PREFIX: &str = "model.diffusion_model.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nvfp4Component {
    Packed,
    BlockScales,
    TensorScale,
}

pub(super) fn checkpoint_is_nvfp4(path: &Path) -> bool {
    let Ok(st) = (unsafe { MmapedSafetensors::new(path) }) else {
        return false;
    };
    st.tensors()
        .into_iter()
        .any(|(key, _)| key.ends_with(".weight_scale_2"))
}

/// Map a model-side logical name onto the single-file source key. The
/// segment table (`proj_in` → `patchify_proj`, …) is
/// `mold_core::ltx2_weight_index::canonical_tensor_name`, so the loaders and
/// the admission/runtime weight index can never disagree about a name.
pub(super) fn remap_ltx2_transformer_key(name: &str) -> String {
    format!(
        "{DIFFUSION_PREFIX}{}",
        mold_core::ltx2_weight_index::canonical_tensor_name(name)
    )
}

pub(super) struct Ltx2Nvfp4Backend {
    st: MmapedSafetensors,
    keys: BTreeSet<String>,
    nvfp4_bases: BTreeSet<String>,
}

impl Ltx2Nvfp4Backend {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }
            .with_context(|| format!("mmap LTX-2 NVFP4 checkpoint at {}", path.display()))?;
        let keys: BTreeSet<String> = st.tensors().into_iter().map(|(key, _)| key).collect();
        let nvfp4_bases = collect_nvfp4_bases(&keys);
        Ok(Self {
            st,
            keys,
            nvfp4_bases,
        })
    }

    fn source_key(&self, logical_name: &str) -> Option<String> {
        let prefixed = remap_ltx2_transformer_key(logical_name);
        if self.keys.contains(&prefixed) {
            return Some(prefixed);
        }
        let stripped = prefixed.strip_prefix(DIFFUSION_PREFIX)?;
        if self.keys.contains(stripped) {
            return Some(stripped.to_string());
        }
        None
    }

    fn source_key_or_default(&self, logical_name: &str) -> String {
        self.source_key(logical_name)
            .unwrap_or_else(|| remap_ltx2_transformer_key(logical_name))
    }

    fn is_nvfp4_weight_source(&self, source_key: &str) -> bool {
        source_key
            .strip_suffix(".weight")
            .is_some_and(|base| self.nvfp4_bases.contains(base))
    }

    fn nvfp4_component(name: &str) -> Option<(&str, Nvfp4Component)> {
        if let Some(weight_key) = name.strip_suffix(".nvfp4_packed") {
            Some((weight_key, Nvfp4Component::Packed))
        } else if let Some(weight_key) = name.strip_suffix(".nvfp4_block_scales") {
            Some((weight_key, Nvfp4Component::BlockScales))
        } else {
            name.strip_suffix(".nvfp4_tensor_scale")
                .map(|weight_key| (weight_key, Nvfp4Component::TensorScale))
        }
    }

    fn lookup_nvfp4_component(
        &self,
        logical_weight_key: &str,
        component: Nvfp4Component,
    ) -> candle_core::Result<Tensor> {
        let source_weight_key = self.source_key(logical_weight_key).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "LTX-2 NVFP4 backend: no source weight for logical key '{logical_weight_key}'",
            ))
        })?;
        let source_base = source_weight_key.strip_suffix(".weight").ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "LTX-2 NVFP4 backend: synthetic key '{logical_weight_key}' does not target a .weight tensor",
            ))
        })?;
        if !self.nvfp4_bases.contains(source_base) {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 NVFP4 backend: source '{source_base}' does not have NVFP4 sidecars",
            )));
        }

        let cpu = Device::Cpu;
        match component {
            Nvfp4Component::Packed => {
                let tensor = self.st.load(&source_weight_key, &cpu)?;
                if tensor.dtype() != DType::U8 {
                    return Err(candle_core::Error::Msg(format!(
                        "LTX-2 NVFP4: expected '{source_weight_key}' to be U8 packed FP4, got {:?}",
                        tensor.dtype()
                    )));
                }
                Ok(tensor)
            }
            Nvfp4Component::BlockScales => {
                let scale_key = format!("{source_base}.weight_scale");
                let tensor = self.st.load(&scale_key, &cpu)?;
                if tensor.dtype() != DType::F8E4M3 {
                    return Err(candle_core::Error::Msg(format!(
                        "LTX-2 NVFP4: expected '{scale_key}' to be F8E4M3 block scales, got {:?}",
                        tensor.dtype()
                    )));
                }
                Ok(tensor)
            }
            Nvfp4Component::TensorScale => {
                let scale_key = format!("{source_base}.weight_scale_2");
                self.st.load(&scale_key, &cpu)?.to_dtype(DType::F32)
            }
        }
    }

    fn lookup(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        if let Some((logical_weight_key, component)) = Self::nvfp4_component(name) {
            return self.lookup_nvfp4_component(logical_weight_key, component);
        }

        let source_key = self.source_key_or_default(name);
        if self.is_nvfp4_weight_source(&source_key) {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 NVFP4 backend: '{name}' is packed FP4; request weight.nvfp4_packed, weight.nvfp4_block_scales, and weight.nvfp4_tensor_scale instead",
            )));
        }
        self.st.load(&source_key, dev)
    }
}

fn collect_nvfp4_bases(keys: &BTreeSet<String>) -> BTreeSet<String> {
    let mut has_scale = BTreeSet::new();
    let mut has_scale_2 = BTreeSet::new();
    for key in keys {
        if let Some(base) = key.strip_suffix(".weight_scale") {
            has_scale.insert(base.to_string());
        } else if let Some(base) = key.strip_suffix(".weight_scale_2") {
            has_scale_2.insert(base.to_string());
        }
    }
    has_scale.intersection(&has_scale_2).cloned().collect()
}

impl SimpleBackend for Ltx2Nvfp4Backend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.lookup(name, dev)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("LTX-2 NVFP4 backend: shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        if tensor.dtype() == dtype {
            Ok(tensor)
        } else {
            tensor.to_dtype(dtype)
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.lookup(name, dev)?;
        if tensor.dtype() == dtype {
            Ok(tensor)
        } else {
            tensor.to_dtype(dtype)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if let Some((logical_weight_key, _component)) = Self::nvfp4_component(name) {
            return self
                .source_key(logical_weight_key)
                .as_deref()
                .and_then(|source_key| source_key.strip_suffix(".weight"))
                .is_some_and(|source_base| self.nvfp4_bases.contains(source_base));
        }

        let Some(source_key) = self.source_key(name) else {
            return false;
        };
        !self.is_nvfp4_weight_source(&source_key)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{checkpoint_is_nvfp4, Ltx2Nvfp4Backend};

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mold-ltx2-nvfp4-{}-{}-{}.safetensors",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ))
    }

    fn write_fixture(path: &std::path::Path) {
        let packed = vec![0x22u8; 16];
        let scales = vec![0x38u8; 512];
        let tensor_scale = 0.5f32.to_le_bytes().to_vec();
        let bias = [0.25f32.to_le_bytes(), (-0.5f32).to_le_bytes()].concat();

        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::U8, vec![2, 8], &packed).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale".to_string(),
            TensorView::new(SafeDtype::F8_E4M3, vec![128, 4], &scales).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale_2".to_string(),
            TensorView::new(SafeDtype::F32, vec![], &tensor_scale).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.bias".to_string(),
            TensorView::new(SafeDtype::F32, vec![2], &bias).unwrap(),
        );
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    fn write_convrot_marker_fixture(path: &std::path::Path) {
        let weight = vec![1u8; 16];
        let scale = 0.5f32.to_le_bytes();
        let marker = br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#;
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::I8, vec![2, 8], &weight).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &scale).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant".to_string(),
            TensorView::new(SafeDtype::U8, vec![marker.len()], marker).unwrap(),
        );
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn ltx2_nvfp4_backend_exposes_sidecar_subkeys_and_hides_packed_weight() {
        let path = temp_path("sidecars");
        write_fixture(&path);
        assert!(checkpoint_is_nvfp4(&path));

        let backend = Ltx2Nvfp4Backend::from_path(&path).unwrap();
        let device = Device::Cpu;
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone());
        let vb = vb.pp("transformer_blocks.0.attn1.to_q");

        assert!(vb.contains_tensor("weight.nvfp4_packed"));
        assert!(vb.contains_tensor("weight.nvfp4_block_scales"));
        assert!(vb.contains_tensor("weight.nvfp4_tensor_scale"));
        assert!(!vb.contains_tensor("weight"));

        let packed = vb
            .get_unchecked_dtype("weight.nvfp4_packed", DType::U8)
            .unwrap();
        let scales = vb
            .get_unchecked_dtype("weight.nvfp4_block_scales", DType::F8E4M3)
            .unwrap();
        let tensor_scale = vb
            .get_unchecked_dtype("weight.nvfp4_tensor_scale", DType::F32)
            .unwrap();
        let bias = vb.get(2, "bias").unwrap();

        assert_eq!(packed.dims(), &[2, 8]);
        assert_eq!(packed.dtype(), DType::U8);
        assert_eq!(scales.dims(), &[128, 4]);
        assert_eq!(scales.dtype(), DType::F8E4M3);
        assert_eq!(tensor_scale.to_scalar::<f32>().unwrap(), 0.5);
        assert_eq!(bias.dims(), &[2]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn convrot_comfy_markers_do_not_select_the_nvfp4_backend() {
        let path = temp_path("convrot-marker");
        write_convrot_marker_fixture(&path);
        assert!(!checkpoint_is_nvfp4(&path));
        let _ = std::fs::remove_file(path);
    }
}
