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
    /// Defensive bound on per-family wall-clock walk time, in seconds.
    /// When set, the HF stage of any single family bails gracefully
    /// (returning the partial entry bucket) once the cap elapses. Useful
    /// when a busy family like SDXL would otherwise pin the scanner for
    /// hours; default `None` preserves the unbounded behaviour.
    #[arg(long)]
    pub max_family_wallclock_secs: Option<u64>,
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

pub async fn run_refresh(args: RefreshArgs) -> Result<()> {
    let hf_base =
        std::env::var("MOLD_CATALOG_HF_BASE").unwrap_or_else(|_| "https://huggingface.co".into());
    let cv_base =
        std::env::var("MOLD_CATALOG_CIVITAI_BASE").unwrap_or_else(|_| "https://civitai.com".into());

    let mut opts = mold_catalog::scanner::ScanOptions::default();
    if let Some(family) = args.family.as_deref() {
        let fam = mold_catalog::families::Family::from_str(family)
            .map_err(|e| anyhow::anyhow!("unknown family: {e}"))?;
        opts.families = vec![fam];
    }
    opts.min_downloads = args.min_downloads;
    opts.include_nsfw = !args.no_nsfw;
    opts.hf_token = std::env::var("HF_TOKEN").ok();
    opts.civitai_token = std::env::var("CIVITAI_TOKEN").ok();
    opts.max_family_wallclock = args
        .max_family_wallclock_secs
        .map(std::time::Duration::from_secs);

    let report = mold_catalog::scanner::run_scan(&hf_base, &cv_base, &opts).await;

    println!(
        "scanned {} entries across {} families",
        report.total_entries,
        report.per_family.len()
    );
    for (fam, outcome) in &report.per_family {
        println!("  {:<12} {:?}", fam.as_str(), outcome);
    }

    if args.dry_run {
        println!("(dry-run; not writing shards or DB)");
        return Ok(());
    }

    // For commit_to_repo, write into the repo-relative shard dir; otherwise
    // into $MOLD_HOME/catalog/.
    let shard_dir = if args.commit_to_repo {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("mold-catalog")
            .join("data")
            .join("catalog")
    } else {
        mold_core::Config::mold_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot resolve MOLD_HOME"))?
            .join("catalog")
    };

    let conn = open_conn()?;
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Iso8601::DEFAULT)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());

    // Re-run the scanner per family so we can group entries by family
    // before sinking. The orchestrator already grouped them, but does not
    // currently return the grouped vec; for phase 1 the simpler path is a
    // second pass keyed off `family`.
    let mut by_family: std::collections::BTreeMap<
        mold_catalog::families::Family,
        Vec<mold_catalog::entry::CatalogEntry>,
    > = Default::default();
    // Re-run minimal scans family-by-family. This is the same pattern the
    // server's refresh endpoint uses; the duplicated network cost is OK
    // because manual refresh is rare and `--family` already narrows it.
    for fam in &opts.families {
        let mut single_opts = opts.clone();
        single_opts.families = vec![*fam];
        let report = mold_catalog::scanner::run_scan(&hf_base, &cv_base, &single_opts).await;
        by_family.insert(
            *fam,
            fetch_family_entries(&hf_base, &cv_base, &single_opts, *fam).await?,
        );
        let _ = report;
    }

    for (fam, entries) in by_family {
        let shard = mold_catalog::sink::build_shard(
            fam.as_str(),
            env!("CARGO_PKG_VERSION"),
            &now,
            entries.clone(),
        );
        mold_catalog::sink::write_shard_atomic(&shard_dir, &shard)?;
        mold_catalog::sink::upsert_family(&conn, fam, &entries)?;
    }
    println!("wrote shards to {}", shard_dir.display());
    Ok(())
}

async fn fetch_family_entries(
    hf_base: &str,
    cv_base: &str,
    opts: &mold_catalog::scanner::ScanOptions,
    family: mold_catalog::families::Family,
) -> Result<Vec<mold_catalog::entry::CatalogEntry>> {
    let mut entries: Vec<mold_catalog::entry::CatalogEntry> = Vec::new();
    if let Ok(hf) = mold_catalog::stages::hf::scan_family(
        hf_base,
        opts,
        family,
        mold_catalog::hf_seeds::seeds_for(family),
        None,
    )
    .await
    {
        entries.extend(hf);
    }

    let cv_keys: Vec<&'static str> = mold_catalog::civitai_map::CIVITAI_BASE_MODELS
        .iter()
        .copied()
        .filter(|k| {
            matches!(
                mold_catalog::civitai_map::map_base_model(k),
                Some((f, _, _)) if f == family
            )
        })
        .collect();
    if !cv_keys.is_empty() {
        if let Ok(cv) = mold_catalog::stages::civitai::scan(cv_base, opts, &cv_keys).await {
            entries.extend(cv);
        }
    }

    Ok(mold_catalog::filter::apply(entries, opts))
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
