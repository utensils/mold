//! Exact tensor consumption for published paint PyTorch checkpoints.
use anyhow::{ensure, Context, Result};
use candle_core::pickle::Object;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, VarBuilder};
use std::collections::BTreeSet;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

struct State {
    tensors: candle_core::pickle::PthTensors,
    consumed: Mutex<BTreeSet<String>>,
}
struct Backend(Arc<State>);

// PthTensors intentionally skips entries it cannot interpret. Paint checkpoints
// are a flat state_dict, so validate the complete dictionary before that reader
// can discard entries or collapse duplicate names into its HashMap.
fn strict_inventory(path: &Path) -> Result<BTreeSet<String>> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?)?;
    ensure!(
        archive.len() <= 50_000,
        "paint checkpoint has too many archive entries"
    );
    let mut members = BTreeSet::new();
    let mut metadata = Vec::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        ensure!(
            members.insert(entry.name().to_string()),
            "duplicate paint archive member"
        );
        if entry.name().ends_with("data.pkl") {
            metadata.push(entry.name().to_string());
        }
    }
    ensure!(
        metadata.len() == 1,
        "paint checkpoint requires exactly one tensor dictionary"
    );
    let name = &metadata[0];
    let directory = Path::new(
        name.strip_suffix(".pkl")
            .context("invalid paint dictionary name")?,
    );
    let entry = archive.by_name(name)?;
    const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
    ensure!(
        entry.size() <= MAX_METADATA_BYTES,
        "paint tensor dictionary is too large"
    );
    let mut bytes = Vec::new();
    entry.take(MAX_METADATA_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_METADATA_BYTES,
        "paint tensor dictionary is too large"
    );
    let mut reader = Cursor::new(bytes);
    let mut stack = candle_core::pickle::Stack::empty();
    stack.read_loop(&mut reader)?;
    ensure!(
        reader.position() == reader.get_ref().len() as u64,
        "trailing paint dictionary bytes"
    );
    let Object::Dict(entries) = stack.finalize()? else {
        anyhow::bail!("paint checkpoint must contain a flat tensor dictionary")
    };
    ensure!(
        !entries.is_empty() && entries.len() <= 20_000,
        "invalid paint tensor count"
    );
    let mut names = BTreeSet::new();
    let mut metadata_seen = false;
    for (name, value) in entries {
        let Object::Unicode(ref key) = name else {
            anyhow::bail!("non-string paint tensor name")
        };
        // torch.nn.Module.state_dict attaches a module-version table to its
        // OrderedDict. Candle merges BUILD attributes into the dictionary.
        // Permit only that schema, never an arbitrary skipped object/tensor.
        if key == "_metadata" {
            ensure!(!metadata_seen, "duplicate paint module metadata");
            metadata_seen = true;
            validate_module_metadata(value)?;
            continue;
        }
        ensure!(
            !key.is_empty() && key.len() <= 1024 && names.insert(key.clone()),
            "invalid or duplicate paint tensor name"
        );
        // Guard the positional accesses in Candle's rebuild_args before calling
        // it; published state_dicts use _rebuild_tensor_v2 directly.
        let Object::Reduce {
            ref callable,
            ref args,
        } = value
        else {
            anyhow::bail!("non-tensor paint dictionary entry")
        };
        ensure!(
            matches!(callable.as_ref(), Object::Class { module_name, class_name } if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2"),
            "unsupported paint tensor reconstruction"
        );
        let Object::Tuple(args) = args.as_ref() else {
            anyhow::bail!("invalid paint tensor arguments")
        };
        ensure!(args.len() == 6, "invalid paint tensor argument count");
        ensure!(
            matches!(&args[1], Object::Int(0) | Object::Long(0)),
            "paint tensor storage offset must be zero"
        );
        let Object::Tuple(dimensions) = &args[2] else {
            anyhow::bail!("invalid paint tensor dimensions")
        };
        let Object::Tuple(strides) = &args[3] else {
            anyhow::bail!("invalid paint tensor strides")
        };
        ensure!(
            !dimensions.is_empty() && dimensions.len() <= 8 && dimensions.len() == strides.len(),
            "invalid paint tensor rank"
        );
        let mut count = 1usize;
        for dimension in dimensions {
            count = count
                .checked_mul(bounded_nonnegative_integer(dimension)?)
                .context("paint tensor dimensions overflow")?;
            ensure!(
                count > 0 && count <= 1_000_000_000,
                "invalid paint tensor dimensions"
            );
        }
        for stride in strides {
            bounded_nonnegative_integer(stride)?;
        }
        let Object::PersistentLoad(storage) = &args[0] else {
            anyhow::bail!("invalid paint tensor storage")
        };
        let Object::Tuple(storage) = storage.as_ref() else {
            anyhow::bail!("invalid paint tensor storage arguments")
        };
        ensure!(
            storage.len() == 5,
            "invalid paint tensor storage argument count"
        );
        ensure!(
            bounded_nonnegative_integer(&storage[4])? >= count,
            "paint tensor storage is smaller than its dimensions"
        );
        let info = value
            .into_tensor_info(name, directory)?
            .context("unsupported paint tensor entry")?;
        ensure!(members.contains(&info.path), "missing paint tensor storage");
    }
    Ok(names)
}

fn bounded_nonnegative_integer(value: &Object) -> Result<usize> {
    let number = match value {
        Object::Int(value) => i64::from(*value),
        Object::Long(value) => *value,
        _ => anyhow::bail!("invalid paint storage integer"),
    };
    ensure!(
        (0..=1_000_000_000).contains(&number),
        "paint storage integer is out of bounds"
    );
    Ok(number as usize)
}

fn validate_module_metadata(value: Object) -> Result<()> {
    let Object::Dict(modules) = value else {
        anyhow::bail!("invalid paint module metadata")
    };
    ensure!(modules.len() <= 20_000, "too many paint metadata modules");
    let mut seen = BTreeSet::new();
    for (name, value) in modules {
        let Object::Unicode(name) = name else {
            anyhow::bail!("invalid paint metadata module name")
        };
        ensure!(
            name.len() <= 1024 && seen.insert(name),
            "duplicate or invalid paint metadata module"
        );
        let Object::Dict(fields) = value else {
            anyhow::bail!("invalid paint module version")
        };
        ensure!(
            fields.len() == 1
                && matches!(&fields[0], (Object::Unicode(key), Object::Int(version)) if key == "version" && *version >= 0),
            "unsupported paint module metadata field"
        );
    }
    Ok(())
}
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
    let inventory = strict_inventory(path)?;
    let tensors = candle_core::pickle::PthTensors::new(path, None)?;
    ensure!(
        inventory.iter().eq(tensors
            .tensor_infos()
            .keys()
            .collect::<BTreeSet<_>>()
            .into_iter()),
        "paint tensor inventory changed while loading"
    );
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
    fn refuses_tensor_entries_that_candle_silently_skips() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.bin");
        std::fs::write(
            &path,
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-pth-unexpected-int32.bin"),
        )
        .unwrap();
        let result = load_pth_exact(&path, DType::F32, &Device::Cpu, |vb| vb.get(1, "expected"));
        assert!(
            result.is_err(),
            "unsupported checkpoint entries must not disappear from the inventory"
        );
    }
    #[test]
    fn refuses_negative_storage_offset_before_candle_arithmetic() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.bin");
        std::fs::write(
            &path,
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-pth-negative-offset.bin"),
        )
        .unwrap();
        let error = load_pth_exact(&path, DType::F32, &Device::Cpu, |vb| vb.get(1, "expected"))
            .unwrap_err();
        assert!(
            error.to_string().contains("offset must be zero"),
            "{error:#}"
        );
    }
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
