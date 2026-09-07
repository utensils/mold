use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mold_core::manifest::{find_manifest, resolve_model_name, HUNYUAN3D_FAMILY};
use mold_core::{Config, ModelConfig, ModelPaths};

pub fn run(
    model: &str,
    tier: mold_inference::hunyuan3d::quantization::ShapeQuantization,
    name: Option<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    let source_name = resolve_model_name(model);
    let mut config = Config::load_or_default();
    let manifest = find_manifest(&source_name);
    let configured = config.models.get(&source_name);
    let family = configured
        .and_then(|entry| entry.family.as_deref())
        .or_else(|| manifest.map(|entry| entry.family.as_str()));
    if family != Some(HUNYUAN3D_FAMILY) {
        bail!("model `{source_name}` is not a Hunyuan3D shape model");
    }
    let paths = ModelPaths::resolve(&source_name, &config)
        .with_context(|| format!("model `{source_name}` is not installed"))?;
    let derived_name = name.unwrap_or_else(|| {
        let family = source_name
            .split_once(':')
            .map_or(source_name.as_str(), |v| v.0);
        format!("{family}:{tier}")
    });
    if derived_name == source_name {
        bail!("derived model name must differ from its source model");
    }
    if config.models.contains_key(&derived_name) {
        bail!("model `{derived_name}` is already configured");
    }
    let output = output.unwrap_or_else(|| {
        config
            .resolved_models_dir()
            .join("derived")
            .join(derived_name.replace(':', "-"))
            .join(format!("shape-{tier}.gguf"))
    });

    eprintln!(
        "Quantizing {} to {} as {}...",
        paths.transformer.display(),
        tier,
        output.display()
    );
    let report = mold_inference::hunyuan3d::quantization::quantize_checkpoint(
        &paths.transformer,
        &output,
        &source_name,
        tier,
    )?;

    let mut derived = configured.cloned().unwrap_or_else(|| {
        let mut entry = ModelConfig::default();
        if let Some(manifest) = manifest {
            entry.default_steps = Some(manifest.defaults.steps);
            entry.default_guidance = Some(manifest.defaults.guidance);
            entry.default_width = Some(manifest.defaults.width);
            entry.default_height = Some(manifest.defaults.height);
            entry.is_schnell = Some(manifest.defaults.is_schnell);
            entry.scheduler = manifest.defaults.scheduler;
            entry.description = Some(manifest.description.clone());
        }
        entry
    });
    derived.transformer = Some(output.to_string_lossy().into_owned());
    derived.transformer_shards = None;
    derived.vae = Some(String::new());
    derived.family = Some(HUNYUAN3D_FAMILY.to_string());
    derived.description = Some(format!(
        "{} ({tier}, locally derived from {source_name})",
        derived
            .description
            .as_deref()
            .unwrap_or("Hunyuan3D shape model")
    ));
    config.models.insert(derived_name.clone(), derived);
    config
        .save()
        .context("register derived Hunyuan3D model; the completed GGUF was retained")?;

    eprintln!(
        "Created {derived_name}: {} bytes ({} of {} tensors quantized; source preserved at {})",
        report.output_bytes,
        report.quantized_tensor_count,
        report.tensor_count,
        report.source.display()
    );
    Ok(())
}
