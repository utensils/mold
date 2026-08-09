//! Private MiniMax H3 artifact qualification under a reviewed campaign record.
//!
//! The binary is unreachable without `dev-bins,h3-private-uat`, refuses every
//! root outside the authorization record's owner-only campaign, never
//! constructs an engine, and embeds a marker that release verification rejects.

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mold_inference::private_qualification::{qualify_private_artifacts, H3ArtifactHashProgress};

fn usage() -> &'static str {
    "usage: h3_artifact_qualification --models-root \
     <absolute-campaign-root>/mold-home/models --model \
     <minimax-h3-fl2va:comfy-pruned-int8|minimax-h3-ref2va:comfy-pruned-int8> \
     --authorization-record <absolute-campaign-root>/compliance/<record>.json"
}

fn parse_args() -> Result<(PathBuf, String, PathBuf)> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--help" || argument == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        if !argument.starts_with("--") {
            bail!("unexpected positional argument {argument:?}; {}", usage())
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {argument}; {}", usage()))?;
        if value.starts_with("--") || values.insert(argument.clone(), value).is_some() {
            bail!("invalid or duplicate argument {argument}; {}", usage())
        }
    }
    if values.len() != 3 {
        bail!("all qualification arguments are required; {}", usage())
    }
    let models_root = values
        .remove("--models-root")
        .map(PathBuf::from)
        .with_context(usage)?;
    let model = values.remove("--model").with_context(usage)?;
    let authorization_record = values
        .remove("--authorization-record")
        .map(PathBuf::from)
        .with_context(usage)?;
    if !values.is_empty() {
        bail!("unknown qualification argument; {}", usage())
    }
    Ok((models_root, model, authorization_record))
}

fn main() -> Result<()> {
    let (models_root, model, authorization_record) = parse_args()?;
    let mut reported = BTreeMap::<String, u64>::new();
    let report = qualify_private_artifacts(
        &models_root,
        &model,
        &authorization_record,
        |progress: H3ArtifactHashProgress| {
            let previous = reported
                .get(&progress.relative_path)
                .copied()
                .unwrap_or_default();
            let complete = progress.artifact_bytes_verified == progress.artifact_bytes_total;
            if complete
                || progress.artifact_bytes_verified.saturating_sub(previous) >= 1024 * 1024 * 1024
            {
                eprintln!(
                    "authenticating {}: {}/{} bytes (total {}/{})",
                    progress.relative_path,
                    progress.artifact_bytes_verified,
                    progress.artifact_bytes_total,
                    progress.total_bytes_verified,
                    progress.total_bytes
                );
                reported.insert(
                    progress.relative_path.clone(),
                    progress.artifact_bytes_verified,
                );
            }
        },
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
