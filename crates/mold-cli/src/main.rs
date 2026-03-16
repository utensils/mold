mod commands;
mod output;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::engine::ArgValueCandidates;

#[derive(Parser)]
#[command(
    name = "mold",
    about = "Local AI image generation — FLUX & SDXL diffusion models on your GPU"
)]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate images from a text prompt
    ///
    /// First positional arg is treated as MODEL if it matches a known model name.
    /// Remaining args are the prompt.
    Run {
        /// Model name (e.g. flux-dev:q4, flux-schnell)
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model_or_prompt: Option<String>,

        /// Prompt text (remaining words after model)
        #[arg(trailing_var_arg = true)]
        prompt_rest: Vec<String>,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,

        /// Image width — defaults to model config value
        #[arg(long)]
        width: Option<u32>,

        /// Image height — defaults to model config value
        #[arg(long)]
        height: Option<u32>,

        /// Number of inference steps — defaults to model config value
        #[arg(long)]
        steps: Option<u32>,

        /// Guidance scale — defaults to model config value
        #[arg(long)]
        guidance: Option<f64>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value = "1")]
        batch: u32,

        /// Override MOLD_HOST
        #[arg(long)]
        host: Option<String>,

        /// Output format
        #[arg(long, default_value = "png")]
        format: String,

        /// Skip server and run inference locally (requires GPU features)
        #[arg(long)]
        local: bool,

        /// T5 encoder variant: auto (default), fp16, q8, q6, q5, q4, q3
        #[arg(long)]
        t5_variant: Option<String>,
    },

    /// Start the inference server
    Serve {
        /// Server port
        #[arg(long, default_value = "7680")]
        port: u16,

        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        bind: String,

        /// Models directory
        #[arg(long)]
        models_dir: Option<String>,
    },

    /// Download model weights from HuggingFace
    Pull {
        /// Model name to download
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: String,
    },

    /// List locally available models
    #[command(alias = "ls")]
    List,

    /// Show detailed model information
    Info {
        /// Model name (e.g. flux-dev:q4, sdxl-turbo:fp16)
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: String,
    },

    /// Show server status and loaded models
    Ps,

    /// Show version information
    Version,

    /// Generate shell completions (sources dynamic model-name completion)
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
        shell: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Reset SIGPIPE to default (terminate) so piping doesn't panic.
    // Rust ignores SIGPIPE by default, causing "broken pipe" panics when
    // stdout is a pipe and the reader closes (e.g. `mold run ... | head`).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    clap_complete::CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model_or_prompt,
            prompt_rest,
            output,
            width,
            height,
            steps,
            guidance,
            seed,
            batch,
            host,
            format,
            local,
            t5_variant,
        } => {
            commands::run::run(
                model_or_prompt,
                prompt_rest,
                output,
                width,
                height,
                steps,
                guidance,
                seed,
                batch,
                host,
                format,
                local,
                t5_variant,
            )
            .await?;
        }
        Commands::Serve {
            port,
            bind,
            models_dir,
        } => {
            commands::serve::run(port, &bind, models_dir).await?;
        }
        Commands::Pull { model } => {
            commands::pull::run(&model).await?;
        }
        Commands::List => {
            commands::list::run().await?;
        }
        Commands::Info { model } => {
            commands::info::run(&model)?;
        }
        Commands::Ps => {
            commands::ps::run().await?;
        }
        Commands::Version => {
            println!("mold {}", env!("CARGO_PKG_VERSION"));
        }
        Commands::Completions { shell } => {
            generate_completions(&shell)?;
        }
    }

    Ok(())
}

/// Generate shell completion script.
///
/// For zsh: custom script that separates flags from positional candidates so
/// `mold run <TAB>` shows only model names, while `mold run --<TAB>` shows flags.
/// For other shells: delegates to clap_complete's dynamic registration.
fn generate_completions(shell: &str) -> anyhow::Result<()> {
    if shell == "zsh" {
        let bin = std::env::args()
            .next()
            .unwrap_or_else(|| "mold".to_string());
        print!(
            r##"#compdef mold
function _clap_dynamic_completer_mold() {{
    local _CLAP_COMPLETE_INDEX=$(expr $CURRENT - 1)
    local _CLAP_IFS=$'\n'

    local completions=("${{(@f)$( \
        _CLAP_IFS="$_CLAP_IFS" \
        _CLAP_COMPLETE_INDEX="$_CLAP_COMPLETE_INDEX" \
        COMPLETE="zsh" \
        {bin} -- "${{words[@]}}" 2>/dev/null \
    )}}")

    if [[ -n $completions ]]; then
        local -a flags=()
        local -a values=()
        local completion
        for completion in $completions; do
            local value="${{completion%%:*}}"
            if [[ "$value" == -* ]]; then
                flags+=("$completion")
            elif [[ "$value" == */ ]]; then
                local dir_no_slash="${{value%/}}"
                if [[ "$completion" == *:* ]]; then
                    local desc="${{completion#*:}}"
                    values+=("$dir_no_slash:$desc")
                else
                    values+=("$dir_no_slash")
                fi
            else
                values+=("$completion")
            fi
        done

        if [[ "${{words[$CURRENT]}}" == -* ]]; then
            [[ -n $flags ]] && _describe 'options' flags
        else
            [[ -n $values ]] && _describe 'values' values
        fi
    fi
}}

compdef _clap_dynamic_completer_mold mold
"##,
            bin = bin,
        );
        return Ok(());
    }

    let shells = clap_complete::env::Shells::builtins();
    let completer = match shells.completer(shell) {
        Some(c) => c,
        None => {
            let names: Vec<_> = shells.names().collect();
            anyhow::bail!(
                "unknown shell '{}', expected one of: {}",
                shell,
                names.join(", ")
            );
        }
    };
    let bin = std::env::args()
        .next()
        .unwrap_or_else(|| "mold".to_string());
    completer.write_registration("COMPLETE", "mold", "mold", &bin, &mut std::io::stdout())?;
    Ok(())
}
