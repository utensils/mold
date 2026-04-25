//! `mold catalog` subcommand handlers.

use anyhow::Result;
use clap::Args;
use mold_db::catalog::{get_by_id, list, ListParams, SortBy};
use rusqlite::Connection;

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

fn open_conn() -> Result<Connection> {
    let path =
        mold_db::default_db_path().ok_or_else(|| anyhow::anyhow!("cannot resolve mold.db path"))?;
    let conn = Connection::open(path)?;
    Ok(conn)
}

pub async fn run_list(args: ListArgs) -> Result<()> {
    let conn = open_conn()?;
    let sort = match args.sort.as_str() {
        "rating" => SortBy::Rating,
        "recent" => SortBy::Recent,
        "name" => SortBy::Name,
        _ => SortBy::Downloads,
    };
    let params = ListParams {
        family: args.family,
        modality: args.modality,
        source: args.source,
        sub_family: args.sub_family,
        q: args.q,
        include_nsfw: args.include_nsfw,
        sort,
        limit: args.limit as i64,
        offset: 0,
        ..Default::default()
    };
    let rows = list(&conn, &params)?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &rows.iter().map(catalog_row_to_value).collect::<Vec<_>>()
            )?
        );
    } else {
        for r in rows {
            println!(
                "{:<48} {:<7} {:<8} {:<10} {:>7} ★{}",
                r.name,
                r.source,
                r.family,
                r.bundling,
                r.download_count,
                r.rating
                    .map(|v| format!("{:.1}", v))
                    .unwrap_or_else(|| "-".into()),
            );
        }
    }
    Ok(())
}

pub async fn run_show(id: String, json: bool) -> Result<()> {
    let conn = open_conn()?;
    let row =
        get_by_id(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("no catalog entry with id {id}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&catalog_row_to_value(&row))?
        );
    } else {
        println!("{}", row.name);
        println!("  id:           {}", row.id);
        println!("  source:       {}", row.source);
        println!("  family:       {} ({})", row.family, row.family_role);
        if let Some(sf) = &row.sub_family {
            println!("  sub-family:   {sf}");
        }
        println!("  bundling:     {}", row.bundling);
        println!("  engine_phase: {}", row.engine_phase);
        println!("  downloads:    {}", row.download_count);
        if let Some(rating) = row.rating {
            println!("  rating:       ★ {:.2}", rating);
        }
    }
    Ok(())
}

pub async fn run_refresh(_args: RefreshArgs) -> Result<()> {
    Err(anyhow::anyhow!(
        "mold catalog refresh — implemented in Task 27"
    ))
}

pub async fn run_where(id: String) -> Result<()> {
    let conn = open_conn()?;
    let row =
        get_by_id(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("no catalog entry with id {id}"))?;
    let cfg = mold_core::Config::load_or_default();
    let downloaded = cfg.manifest_model_is_downloaded(&row.source_id);
    if downloaded {
        println!("{}", cfg.resolved_models_dir().display());
    } else {
        println!("<not downloaded>");
    }
    Ok(())
}

fn catalog_row_to_value(r: &mold_db::catalog::CatalogRow) -> serde_json::Value {
    serde_json::json!({
        "id": r.id,
        "source": r.source,
        "source_id": r.source_id,
        "name": r.name,
        "family": r.family,
        "family_role": r.family_role,
        "sub_family": r.sub_family,
        "bundling": r.bundling,
        "engine_phase": r.engine_phase,
        "download_count": r.download_count,
        "rating": r.rating,
    })
}
