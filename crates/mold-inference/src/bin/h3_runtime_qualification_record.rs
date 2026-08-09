//! Produce a content-addressed private H3 runtime-record review candidate.

use std::collections::BTreeMap;
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mold_inference::{
    private_qualification::H3ArtifactHashProgress,
    produce_h3_private_runtime_qualification_candidate, H3_PRIVATE_RUNTIME_RECORD_PRODUCER_MARKER,
};

fn usage() -> String {
    format!(
        "usage: h3_runtime_qualification_record --models-root \
         <absolute-campaign-root>/mold-home/models --authorization-record \
         <absolute-campaign-root>/compliance/<record>.json --evidence-root \
         <absolute-campaign-root>/evidence/<campaign> --capture-manifest \
         <absolute-campaign-root>/evidence/<campaign>/<capture>.json \
         (development marker: {H3_PRIVATE_RUNTIME_RECORD_PRODUCER_MARKER})"
    )
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
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
    if values.len() != 4 {
        bail!("all qualification arguments are required; {}", usage())
    }
    let models_root = values
        .remove("--models-root")
        .map(PathBuf::from)
        .with_context(usage)?;
    let authorization_record = values
        .remove("--authorization-record")
        .map(PathBuf::from)
        .with_context(usage)?;
    let evidence_root = values
        .remove("--evidence-root")
        .map(PathBuf::from)
        .with_context(usage)?;
    let capture_manifest = values
        .remove("--capture-manifest")
        .map(PathBuf::from)
        .with_context(usage)?;
    if !values.is_empty() {
        bail!("unknown qualification argument; {}", usage())
    }
    Ok((
        models_root,
        authorization_record,
        evidence_root,
        capture_manifest,
    ))
}

fn main() -> Result<()> {
    let (models_root, authorization_record, evidence_root, capture_manifest) = parse_args()?;
    let mut reported = BTreeMap::<String, u64>::new();
    let candidate = produce_h3_private_runtime_qualification_candidate(
        &models_root,
        &authorization_record,
        &evidence_root,
        &capture_manifest,
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
    io::stdout().write_all(candidate.json_bytes())?;
    eprintln!(
        "candidate_record_sha256={} identity_sha256={} evidence_artifacts={} reviewed_allowlist_unchanged=true",
        candidate.record_file_sha256(),
        candidate.identity_sha256(),
        candidate.evidence_artifact_count(),
    );
    Ok(())
}
