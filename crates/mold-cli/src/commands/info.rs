use anyhow::Result;
use colored::Colorize;
use mold_core::manifest::{
    find_manifest, resolve_model_name, ModelComponent, SHARED_COMPONENTS_GB,
    SHARED_SDXL_COMPONENTS_GB, SHARED_ZIMAGE_COMPONENTS_GB,
};
use mold_core::Config;

fn format_family(family: &str) -> String {
    match family {
        "flux" => "FLUX.1".magenta().to_string(),
        "sdxl" => "SDXL".yellow().to_string(),
        "z-image" => "Z-Image".cyan().to_string(),
        other => other.to_uppercase(),
    }
}

fn component_label(component: &ModelComponent) -> &'static str {
    match component {
        ModelComponent::Transformer => "Transformer",
        ModelComponent::TransformerShard => "Transformer Shard",
        ModelComponent::Vae => "VAE",
        ModelComponent::T5Encoder => "T5 Encoder",
        ModelComponent::ClipEncoder => "CLIP-L Encoder",
        ModelComponent::T5Tokenizer => "T5 Tokenizer",
        ModelComponent::ClipTokenizer => "CLIP-L Tokenizer",
        ModelComponent::ClipEncoder2 => "CLIP-G Encoder",
        ModelComponent::ClipTokenizer2 => "CLIP-G Tokenizer",
        ModelComponent::TextEncoder => "Text Encoder",
        ModelComponent::TextTokenizer => "Text Tokenizer",
    }
}

pub fn run(name: &str) -> Result<()> {
    let canonical = resolve_model_name(name);
    let config = Config::load_or_default();

    let manifest = find_manifest(&canonical);
    let model_config = config
        .models
        .get(&canonical)
        .or_else(|| config.models.get(name));

    if manifest.is_none() && model_config.is_none() {
        anyhow::bail!(
            "unknown model '{}'. Use {} to see available models.",
            canonical,
            "mold list".bold()
        );
    }

    // Header
    println!("{}", canonical.bold());
    println!("{}", "─".repeat(60).dimmed());

    if let Some(ref m) = manifest {
        println!("  {:<16} {}", "Family:".dimmed(), format_family(&m.family));
        println!("  {:<16} {}", "Description:".dimmed(), m.description);

        // Size info
        let shared_gb = match m.family.as_str() {
            "sdxl" => SHARED_SDXL_COMPONENTS_GB,
            "z-image" => SHARED_ZIMAGE_COMPONENTS_GB,
            _ => SHARED_COMPONENTS_GB,
        };
        println!(
            "  {:<16} {:.1}GB (transformer) + {:.1}GB (shared) = {:.1}GB total",
            "Size:".dimmed(),
            m.size_gb,
            shared_gb,
            m.size_gb + shared_gb
        );

        // Generation defaults
        println!();
        println!("  {}", "Generation Defaults".bold());
        println!("  {:<16} {}", "Steps:".dimmed(), m.defaults.steps);
        println!("  {:<16} {:.1}", "Guidance:".dimmed(), m.defaults.guidance);
        println!(
            "  {:<16} {}x{}",
            "Dimensions:".dimmed(),
            m.defaults.width,
            m.defaults.height
        );
        if let Some(scheduler) = m.defaults.scheduler {
            println!("  {:<16} {}", "Scheduler:".dimmed(), scheduler);
        }

        // HuggingFace sources
        println!();
        println!("  {}", "HuggingFace Sources".bold());
        for file in &m.files {
            let url = format!(
                "https://huggingface.co/{}/blob/main/{}",
                file.hf_repo, file.hf_filename
            );
            let size_mb = file.size_bytes as f64 / (1024.0 * 1024.0);
            let size_str = if size_mb >= 1024.0 {
                format!("{:.1}GB", size_mb / 1024.0)
            } else {
                format!("{:.0}MB", size_mb)
            };
            println!(
                "  {:<16} {} ({})",
                format!("{}:", component_label(&file.component)).dimmed(),
                url,
                size_str,
            );
        }
    } else if let Some(mcfg) = model_config {
        // Custom model — show what we know from config
        let family = mcfg.family.as_deref().unwrap_or("unknown");
        println!("  {:<16} {}", "Family:".dimmed(), format_family(family));
        if let Some(ref desc) = mcfg.description {
            println!("  {:<16} {}", "Description:".dimmed(), desc);
        }
        println!();
        println!("  {}", "Generation Defaults".bold());
        println!(
            "  {:<16} {}",
            "Steps:".dimmed(),
            mcfg.effective_steps(&config)
        );
        println!(
            "  {:<16} {:.1}",
            "Guidance:".dimmed(),
            mcfg.effective_guidance()
        );
        println!(
            "  {:<16} {}x{}",
            "Dimensions:".dimmed(),
            mcfg.effective_width(&config),
            mcfg.effective_height(&config)
        );
    }

    // Installation status + local file paths
    println!();
    if let Some(mcfg) = model_config {
        println!(
            "  {:<16} {}",
            "Status:".dimmed(),
            "Installed".green().bold()
        );
        let config_path = Config::config_path();
        println!("  {:<16} {}", "Config:".dimmed(), config_path.display());

        // Show local file paths
        println!();
        println!("  {}", "Local Files".bold());
        let mut has_files = false;
        if let Some(ref p) = mcfg.transformer {
            println!("  {:<20} {}", "Transformer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref shards) = mcfg.transformer_shards {
            for (i, p) in shards.iter().enumerate() {
                let label = format!("Xformer Shard {}:", i + 1);
                println!("  {:<20} {}", label.dimmed(), p);
            }
            has_files = true;
        }
        if let Some(ref p) = mcfg.vae {
            println!("  {:<20} {}", "VAE:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.t5_encoder {
            println!("  {:<20} {}", "T5 Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_encoder {
            println!("  {:<20} {}", "CLIP-L Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.t5_tokenizer {
            println!("  {:<20} {}", "T5 Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_tokenizer {
            println!("  {:<20} {}", "CLIP-L Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_encoder_2 {
            println!("  {:<20} {}", "CLIP-G Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_tokenizer_2 {
            println!("  {:<20} {}", "CLIP-G Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref files) = mcfg.text_encoder_files {
            for (i, p) in files.iter().enumerate() {
                let label = format!("Text Enc Shard {}:", i + 1);
                println!("  {:<20} {}", label.dimmed(), p);
            }
            has_files = true;
        }
        if let Some(ref p) = mcfg.text_tokenizer {
            println!("  {:<20} {}", "Text Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if !has_files {
            println!("  {}", "(no paths configured)".dimmed());
        }
    } else {
        println!(
            "  {:<16} {} — run {} to download",
            "Status:".dimmed(),
            "Not installed".red(),
            format!("mold pull {}", canonical).bold()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_labels_are_all_nonempty() {
        let components = [
            ModelComponent::Transformer,
            ModelComponent::TransformerShard,
            ModelComponent::Vae,
            ModelComponent::T5Encoder,
            ModelComponent::ClipEncoder,
            ModelComponent::T5Tokenizer,
            ModelComponent::ClipTokenizer,
            ModelComponent::ClipEncoder2,
            ModelComponent::ClipTokenizer2,
            ModelComponent::TextEncoder,
            ModelComponent::TextTokenizer,
        ];
        for c in &components {
            assert!(!component_label(c).is_empty());
        }
    }

    #[test]
    fn format_family_flux() {
        let result = format_family("flux");
        assert!(result.contains("FLUX.1"));
    }

    #[test]
    fn format_family_sdxl() {
        let result = format_family("sdxl");
        assert!(result.contains("SDXL"));
    }

    #[test]
    fn format_family_unknown() {
        assert_eq!(format_family("other"), "OTHER");
    }

    #[test]
    fn unknown_model_returns_error() {
        let result = run("nonexistent-model-xyz");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("unknown model"));
    }

    #[test]
    fn known_model_succeeds() {
        // flux-schnell is always in the manifest registry
        let result = run("flux-schnell");
        assert!(result.is_ok());
    }

    #[test]
    fn sdxl_model_succeeds() {
        let result = run("sdxl-base");
        assert!(result.is_ok());
    }

    #[test]
    fn legacy_name_resolves() {
        // flux-dev-q4 should resolve to flux-dev:q4
        let result = run("flux-dev-q4");
        assert!(result.is_ok());
    }
}
