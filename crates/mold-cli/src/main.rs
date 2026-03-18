mod commands;
mod output;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::engine::ArgValueCandidates;

/// Sentinel error: the command already printed diagnostics to stderr.
/// The main handler should just exit(1) without printing anything extra.
#[derive(Debug)]
struct AlreadyReported;

impl std::fmt::Display for AlreadyReported {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::error::Error for AlreadyReported {}

#[derive(Parser)]
#[command(
    name = "mold",
    about = "Local AI image generation — FLUX, SD1.5, SDXL & Z-Image diffusion models on your GPU"
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

        /// Qwen3 text encoder variant (Z-Image): auto (default), bf16, q8, q6, iq4, q3
        #[arg(long)]
        qwen3_variant: Option<String>,

        /// Keep all model components loaded simultaneously (faster but uses more memory).
        /// By default, components are loaded and unloaded sequentially to reduce peak memory.
        #[arg(long)]
        eager: bool,
    },

    /// Start the inference server
    Serve {
        /// Server port (default: 7680, or MOLD_PORT env var)
        #[arg(long, default_value_t = default_port())]
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

    /// Remove downloaded model(s) and their unique files
    #[command(alias = "remove")]
    Rm {
        /// Model name(s) to remove
        #[arg(required = true, num_args = 1..)]
        #[arg(add = ArgValueCandidates::new(commands::rm::complete_installed_model_name))]
        models: Vec<String>,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// List locally available models
    #[command(alias = "ls")]
    List,

    /// Show detailed model information
    Info {
        /// Model name (e.g. flux-dev:q4, sdxl-turbo:fp16)
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: String,

        /// Verify file integrity via SHA-256 checksums
        #[arg(long)]
        verify: bool,
    },

    /// Unload the current model from the server to free GPU memory
    Unload,

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
async fn main() {
    // Reset SIGPIPE to default (terminate) so piping doesn't panic.
    // Rust ignores SIGPIPE by default, causing "broken pipe" panics when
    // stdout is a pipe and the reader closes (e.g. `mold run ... | head`).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    if let Err(e) = run().await {
        // If the command already printed its own diagnostics, just exit.
        if e.downcast_ref::<AlreadyReported>().is_some() {
            std::process::exit(1);
        }
        // Print the error chain cleanly without backtraces
        use colored::Colorize;
        eprintln!("{} {e}", "error:".red().bold());
        for cause in e.chain().skip(1) {
            eprintln!("  {} {cause}", "caused by:".dimmed());
        }
        std::process::exit(1);
    }
}

fn default_port() -> u16 {
    std::env::var("MOLD_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7680)
}

async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MOLD_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

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
            qwen3_variant,
            eager,
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
                qwen3_variant,
                eager,
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
        Commands::Rm { models, force } => {
            commands::rm::run(&models, force).await?;
        }
        Commands::List => {
            commands::list::run().await?;
        }
        Commands::Info { model, verify } => {
            commands::info::run(&model, verify)?;
        }
        Commands::Unload => {
            commands::unload::run().await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Parse CLI args from a vector (simulates command-line invocation).
    fn parse(args: &[&str]) -> Cli {
        Cli::parse_from(std::iter::once("mold").chain(args.iter().copied()))
    }

    #[test]
    fn run_parses_model_and_prompt() {
        let cli = parse(&["run", "flux-dev:q4", "a", "red", "apple"]);
        match cli.command {
            Commands::Run {
                model_or_prompt,
                prompt_rest,
                ..
            } => {
                assert_eq!(model_or_prompt.as_deref(), Some("flux-dev:q4"));
                assert_eq!(prompt_rest, vec!["a", "red", "apple"]);
            }
            _ => panic!("expected Run command"),
        }
    }

    #[test]
    fn run_seed_before_prompt() {
        let cli = parse(&["run", "model", "--seed", "42", "a", "cat"]);
        match cli.command {
            Commands::Run { seed, .. } => assert_eq!(seed, Some(42)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_seed_after_prompt() {
        let cli = parse(&["run", "model", "a", "cat", "--seed", "42"]);
        match cli.command {
            Commands::Run { seed, .. } => assert_eq!(seed, Some(42)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_steps_after_prompt() {
        let cli = parse(&["run", "model", "a", "cat", "--steps", "20"]);
        match cli.command {
            Commands::Run { steps, .. } => assert_eq!(steps, Some(20)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_width_height() {
        let cli = parse(&["run", "model", "--width", "512", "--height", "768", "test"]);
        match cli.command {
            Commands::Run { width, height, .. } => {
                assert_eq!(width, Some(512));
                assert_eq!(height, Some(768));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_guidance() {
        let cli = parse(&["run", "model", "test", "--guidance", "7.5"]);
        match cli.command {
            Commands::Run { guidance, .. } => assert_eq!(guidance, Some(7.5)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_format_jpeg() {
        let cli = parse(&["run", "model", "test", "--format", "jpeg"]);
        match cli.command {
            Commands::Run { format, .. } => assert_eq!(format, "jpeg"),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_batch() {
        let cli = parse(&["run", "model", "test", "--batch", "4"]);
        match cli.command {
            Commands::Run { batch, .. } => assert_eq!(batch, 4),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_output_flag() {
        let cli = parse(&["run", "model", "test", "-o", "/tmp/out.png"]);
        match cli.command {
            Commands::Run { output, .. } => assert_eq!(output.as_deref(), Some("/tmp/out.png")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_local_flag() {
        let cli = parse(&["run", "model", "test", "--local"]);
        match cli.command {
            Commands::Run { local, .. } => assert!(local),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_eager_flag() {
        let cli = parse(&["run", "model", "test", "--eager"]);
        match cli.command {
            Commands::Run { eager, .. } => assert!(eager),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_all_flags_combined() {
        let cli = parse(&[
            "run",
            "model",
            "a complex prompt with many words",
            "--seed",
            "99",
            "--steps",
            "10",
            "--width",
            "512",
            "--height",
            "768",
            "--guidance",
            "4.0",
            "--format",
            "jpeg",
            "--batch",
            "2",
            "-o",
            "/tmp/test.jpg",
            "--local",
            "--eager",
        ]);
        match cli.command {
            Commands::Run {
                seed,
                steps,
                width,
                height,
                guidance,
                format,
                batch,
                output,
                local,
                eager,
                ..
            } => {
                assert_eq!(seed, Some(99));
                assert_eq!(steps, Some(10));
                assert_eq!(width, Some(512));
                assert_eq!(height, Some(768));
                assert_eq!(guidance, Some(4.0));
                assert_eq!(format, "jpeg");
                assert_eq!(batch, 2);
                assert_eq!(output.as_deref(), Some("/tmp/test.jpg"));
                assert!(local);
                assert!(eager);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_defaults_when_no_flags() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run {
                seed,
                steps,
                width,
                height,
                guidance,
                format,
                batch,
                output,
                local,
                eager,
                ..
            } => {
                assert_eq!(seed, None);
                assert_eq!(steps, None);
                assert_eq!(width, None);
                assert_eq!(height, None);
                assert_eq!(guidance, None);
                assert_eq!(format, "png");
                assert_eq!(batch, 1);
                assert_eq!(output, None);
                assert!(!local);
                assert!(!eager);
            }
            _ => panic!("expected Run"),
        }
    }
}
