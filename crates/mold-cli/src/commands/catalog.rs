//! `mold catalog` subcommand handlers.

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct ListArgs {
    #[arg(long)]
    pub family: Option<String>,
    #[arg(long)]
    pub modality: Option<String>,
    #[arg(long)]
    pub source: Option<String>,
    #[arg(long)]
    pub sub_family: Option<String>,
    #[arg(long)]
    pub q: Option<String>,
    /// Sort: downloads | rating | recent | name
    #[arg(long, default_value = "downloads")]
    pub sort: String,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub include_nsfw: bool,
}

#[derive(Args, Debug, Clone)]
pub struct RefreshArgs {
    #[arg(long)]
    pub family: Option<String>,
    #[arg(long, default_value_t = 100)]
    pub min_downloads: u64,
    #[arg(long)]
    pub no_nsfw: bool,
    #[arg(long)]
    pub dry_run: bool,
    /// Maintainer-only: write into the repo's `crates/mold-catalog/data/catalog/`
    /// instead of `$MOLD_HOME/catalog/`.
    #[arg(long)]
    pub commit_to_repo: bool,
}

pub async fn run_list(_args: ListArgs) -> Result<()> {
    Err(anyhow::anyhow!(
        "mold catalog list — implemented in Task 26"
    ))
}

pub async fn run_show(_id: String, _json: bool) -> Result<()> {
    Err(anyhow::anyhow!(
        "mold catalog show — implemented in Task 26"
    ))
}

pub async fn run_refresh(_args: RefreshArgs) -> Result<()> {
    Err(anyhow::anyhow!(
        "mold catalog refresh — implemented in Task 27"
    ))
}

pub async fn run_where(_id: String) -> Result<()> {
    Err(anyhow::anyhow!(
        "mold catalog where — implemented in Task 26"
    ))
}
