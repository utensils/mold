use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mold_core::manifest::{find_manifest, resolve_model_name, HUNYUAN3D_FAMILY};
use mold_core::{Config, ModelPaths};

pub fn run(
    model: &str,
    tier: mold_inference::hunyuan3d::quantization::ShapeQuantization,
    name: Option<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    let source_name = resolve_model_name(model);
    let mut config = Config::load_or_default();
    let manifest = find_manifest(&source_name);
    let configured = config.models.get(&source_name).cloned();
    let family = configured
        .as_ref()
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

    // Quantization can take several minutes. Reload immediately before the
    // write so another completed conversion is not erased by this process's
    // stale pre-conversion snapshot.
    config = Config::load_or_default();
    if config.models.contains_key(&derived_name) {
        bail!(
            "model `{derived_name}` was registered while quantization was running; \
             the completed GGUF was retained at {}",
            output.display()
        );
    }

    let mut derived = configured.unwrap_or_default();
    if let Some(manifest) = manifest {
        derived.default_steps.get_or_insert(manifest.defaults.steps);
        derived
            .default_guidance
            .get_or_insert(manifest.defaults.guidance);
        derived.default_width.get_or_insert(manifest.defaults.width);
        derived
            .default_height
            .get_or_insert(manifest.defaults.height);
        derived
            .is_schnell
            .get_or_insert(manifest.defaults.is_schnell);
        if derived.scheduler.is_none() {
            derived.scheduler = manifest.defaults.scheduler;
        }
        if derived.description.is_none() {
            derived.description = Some(manifest.description.clone());
        }
    }
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
