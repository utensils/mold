//! `mold search` — find models in the Hugging Face and Civitai catalogs.
//!
//! Remote first, for the same reason `mold pull` is: the server holds the
//! stored catalog credentials, so a search asked of it sees gated and private
//! repositories an unauthenticated in-process query cannot, and the
//! `installed` column then answers about the machine that would do the
//! downloading. With no server reachable the query runs here instead, using
//! `HF_TOKEN` and `CIVITAI_TOKEN` from the environment — the same fallback
//! `mold pull` takes.
//!
//! A merged search whose one provider failed still answers with the healthy
//! provider's rows. Those failures go to stderr as warnings rather than
//! replacing the results, so `mold search … | head` still works while the
//! user is told what they are not seeing.

use anyhow::{Context, Result};
use colored::Colorize;
use mold_core::catalog_wire::{
    CatalogProviderErrorWire, CatalogSearchEntry, CatalogSearchPage, CatalogSearchQuery,
};
use mold_core::{classify_server_error, MoldClient, ServerAvailability};

use crate::theme;
use crate::ui::col_width;
use crate::{CatalogSortArg, CatalogSourceArg};

/// `mold search`, resolved from clap.
pub struct SearchArgs {
    pub query: Option<String>,
    pub family: Option<String>,
    pub kind: Option<String>,
    pub source: Option<CatalogSourceArg>,
    pub sort: Option<CatalogSortArg>,
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub include_nsfw: Option<bool>,
    pub json: bool,
}

impl SearchArgs {
    /// The wire query. Absent stays absent so the host applies its own
    /// defaults rather than this command carrying a second copy of them.
    pub fn to_query(&self) -> CatalogSearchQuery {
        CatalogSearchQuery {
            q: self.query.clone(),
            family: self.family.clone(),
            kind: self.kind.clone(),
            source: self.source.map(source_wire).map(str::to_string),
            sort: self.sort.map(sort_wire).map(str::to_string),
            page: self.page,
            page_size: self.page_size,
            include_nsfw: self.include_nsfw,
        }
    }
}

pub fn source_wire(source: CatalogSourceArg) -> &'static str {
    match source {
        CatalogSourceArg::Hf => "hf",
        CatalogSourceArg::Civitai => "civitai",
    }
}

pub fn sort_wire(sort: CatalogSortArg) -> &'static str {
    match sort {
        CatalogSortArg::Downloads => "downloads",
        CatalogSortArg::Recent => "recent",
        CatalogSortArg::Rating => "rating",
    }
}

pub async fn run(args: SearchArgs) -> Result<()> {
    let client = MoldClient::from_env();
    let query = args.to_query();
    let page = match client.search_catalog(&query).await {
        Ok(page) => {
            // A search proves the machine answered, so it becomes a `--host`
            // completion candidate. There are no ids to record: a catalog
            // entry is installed by name through `mold pull`, which completes
            // from the manifest.
            crate::completion_cache::record_reached_host(client.host(), |_| {});
            page
        }
        Err(error) => match classify_server_error(&error) {
            // The local fallback answered from this process, not from a
            // machine, so nothing is recorded.
            ServerAvailability::FallbackLocal => search_locally(&args).await?,
            ServerAvailability::SurfaceError => return Err(error),
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&page)?);
        return Ok(());
    }
    report_provider_errors(&page.provider_errors);
    print_page(&page);
    Ok(())
}

/// Provider failures are warnings beside the results, never a replacement for
/// them — and they go to stderr so a piped search still carries only rows.
fn report_provider_errors(errors: &[CatalogProviderErrorWire]) {
    for error in errors {
        let retry = match error.retry_after_seconds {
            Some(seconds) => format!(" (retry in {seconds}s)"),
            None => String::new(),
        };
        eprintln!(
            "{} {} search failed: {}{retry}",
            theme::icon_warn(),
            error.source,
            error.message
        );
    }
}

fn print_page(page: &CatalogSearchPage) {
    if page.entries.is_empty() {
        println!("{} Nothing matched.", theme::icon_neutral());
        return;
    }
    let id_width = col_width(page.entries.iter().map(|entry| entry.id.len()), 2, 2);
    let name_width = col_width(
        page.entries.iter().map(|entry| display_name(entry).len()),
        4,
        2,
    );
    let family_width = col_width(page.entries.iter().map(|entry| entry.family.len()), 6, 2);
    let kind_width = col_width(page.entries.iter().map(|entry| entry.kind.len()), 4, 2);

    println!(
        "{:<id_width$} {:<name_width$} {:<family_width$} {:<kind_width$} {:>8}  {:>10}",
        "ID".bold(),
        "NAME".bold(),
        "FAMILY".bold(),
        "KIND".bold(),
        "SIZE".bold(),
        "DOWNLOADS".bold(),
    );
    println!(
        "{}",
        "─"
            .repeat(id_width + name_width + family_width + kind_width + 24)
            .dimmed()
    );
    for entry in &page.entries {
        // Pad the plain text first; ANSI codes break `{:<N}`.
        let id = format!("{:<id_width$}", entry.id);
        let id = if entry.installed {
            id.green().to_string()
        } else {
            id
        };
        println!(
            "{} {:<name_width$} {:<family_width$} {:<kind_width$} {:>8}  {:>10}",
            id,
            truncate(&display_name(entry), name_width),
            entry.family,
            entry.kind,
            format_size(entry.size_bytes),
            format_count(entry.download_count),
        );
    }
    println!();
    println!(
        "{} page {} of results ({} total). Install one with {}.",
        theme::icon_neutral(),
        page.page.max(1),
        page.total,
        "mold pull <id>".bold()
    );
}

fn display_name(entry: &CatalogSearchEntry) -> String {
    match entry
        .author
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())
    {
        Some(author) => format!("{} · {author}", entry.name),
        None => entry.name.clone(),
    }
}

/// Keep a long title inside its column without breaking the row.
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let keep = width.saturating_sub(1);
    format!("{}…", text.chars().take(keep).collect::<String>())
}

fn format_size(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) if bytes > 0 => crate::ui::format_disk_size(bytes),
        _ => "—".to_string(),
    }
}

fn format_count(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// The no-server path: the same live query, run in this process.
///
/// Tokens come from the environment because there is no server whose stored
/// credentials could be used, and the bases honour `CIVITAI_BASE` / `HF_BASE`
/// so tests can point them somewhere else — exactly what
/// `catalog_bridge::lookup_catalog_entry_live` does for `mold pull`.
async fn search_locally(args: &SearchArgs) -> Result<CatalogSearchPage> {
    eprintln!(
        "{} No server answered; searching from this machine.",
        theme::icon_warn()
    );
    let opts = local_search_opts(args)?;
    let civitai_base =
        std::env::var("CIVITAI_BASE").unwrap_or_else(|_| "https://civitai.com".to_string());
    let hf_base = std::env::var("HF_BASE").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let cache = mold_catalog::live::LiveCache::new(std::time::Duration::from_secs(300), 64);
    let result = mold_catalog::live::search_page(&civitai_base, &hf_base, &cache, &opts)
        .await
        .context("catalog search failed")?;
    Ok(CatalogSearchPage {
        page: i64::from(opts.page),
        page_size: i64::from(opts.page_size),
        total: result.total as i64,
        entries: result.entries.iter().map(entry_to_wire).collect(),
        provider_errors: result
            .provider_errors
            .iter()
            .map(|error| CatalogProviderErrorWire {
                source: format!("{:?}", error.source).to_ascii_lowercase(),
                message: error.message.clone(),
                code: error.code.map(str::to_string),
                retry_after_seconds: error.retry_after_seconds,
            })
            .collect(),
    })
}

/// Translate the flags into the in-process query.
///
/// An unknown family, kind or source is refused here with the same vocabulary
/// the server refuses it with, rather than being dropped — a silently ignored
/// filter answers a question nobody asked.
fn local_search_opts(args: &SearchArgs) -> Result<mold_catalog::live::LiveSearchOpts> {
    let family = match args
        .family
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
    {
        Some(name) => Some(
            mold_catalog::families::Family::from_str(name)
                .map_err(|_| anyhow::anyhow!("unknown family: {name}"))?,
        ),
        None => None,
    };
    let kind = match args
        .kind
        .as_deref()
        .map(str::trim)
        .filter(|k| !k.is_empty())
    {
        Some(name) => Some(
            serde_json::from_value::<mold_catalog::entry::Kind>(serde_json::Value::String(
                name.to_ascii_lowercase(),
            ))
            .map_err(|_| anyhow::anyhow!("unknown kind: {name}"))?,
        ),
        None => None,
    };
    let source = args.source.map(|source| match source {
        CatalogSourceArg::Hf => mold_catalog::entry::Source::Hf,
        CatalogSourceArg::Civitai => mold_catalog::entry::Source::Civitai,
    });
    Ok(mold_catalog::live::LiveSearchOpts {
        q: args
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .map(str::to_string),
        family,
        kind,
        source,
        page: args.page.unwrap_or(1).max(1),
        page_size: args.page_size.unwrap_or(20).clamp(1, 100),
        include_nsfw: args.include_nsfw.unwrap_or(true),
        sort: args
            .sort
            .map(sort_wire)
            .and_then(mold_catalog::live::CatalogSort::from_wire)
            .unwrap_or_default(),
        civitai_token: std::env::var("CIVITAI_TOKEN").ok(),
        hf_token: std::env::var("HF_TOKEN").ok(),
    })
}

/// One locally-fetched row in the shape the host's own search answers with.
///
/// Written out field by field rather than reserialized, because the two types
/// are only wire-uniform on the fields the endpoint publishes: `installed`,
/// `primary_path` and the companion details are server-side answers a local
/// query has no way to give, and guessing them would be worse than leaving
/// them empty.
fn entry_to_wire(entry: &mold_catalog::entry::CatalogEntry) -> CatalogSearchEntry {
    CatalogSearchEntry {
        id: entry.id.as_str().to_string(),
        source: format!("{:?}", entry.source).to_ascii_lowercase(),
        source_id: entry.source_id.clone(),
        name: entry.name.clone(),
        author: entry.author.clone(),
        family: entry.family.as_str().to_string(),
        family_role: json_word(&entry.family_role),
        sub_family: entry.sub_family.clone(),
        modality: json_word(&entry.modality),
        kind: json_word(&entry.kind),
        file_format: json_word(&entry.file_format),
        bundling: json_word(&entry.bundling),
        size_bytes: entry.size_bytes,
        download_count: entry.download_count,
        rating: entry.rating.map(f64::from),
        likes: entry.likes,
        nsfw: Some(entry.nsfw),
        thumbnail_url: entry.thumbnail_url.clone(),
        description: entry.description.clone(),
        license: entry.license.clone(),
        license_flags: serde_json::to_value(&entry.license_flags).ok(),
        tags: entry.tags.clone(),
        companions: entry
            .companions
            .iter()
            .map(|name| name.to_string())
            .collect(),
        companion_details: Vec::new(),
        download_recipe: serde_json::to_value(&entry.download_recipe)
            .ok()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        supported: entry.supported,
        installed: false,
        primary_path: None,
        created_at: entry.created_at,
        updated_at: entry.updated_at,
        added_at: entry.added_at,
        trained_words: entry.trained_words.clone(),
        page_url: entry.page_url.clone(),
    }
}

/// The wire spelling of a small serde enum, for the fields this row carries
/// as plain strings.
fn json_word<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> SearchArgs {
        SearchArgs {
            query: None,
            family: None,
            kind: None,
            source: None,
            sort: None,
            page: None,
            page_size: None,
            include_nsfw: None,
            json: false,
        }
    }

    /// Only what the user named reaches the wire; the host owns every
    /// default, so an omitted flag must not become a value here.
    #[test]
    fn the_query_carries_only_what_was_asked_for() {
        assert!(args().to_query().query_pairs().is_empty());

        let pairs = SearchArgs {
            query: Some("flux".into()),
            source: Some(CatalogSourceArg::Civitai),
            sort: Some(CatalogSortArg::Rating),
            include_nsfw: Some(false),
            ..args()
        }
        .to_query()
        .query_pairs();
        assert_eq!(
            pairs,
            vec![
                ("q", "flux".to_string()),
                ("source", "civitai".to_string()),
                ("sort", "rating".to_string()),
                ("include_nsfw", "false".to_string()),
            ]
        );
    }

    /// The sort vocabulary is the catalog's own, so the two cannot drift.
    #[test]
    fn every_sort_flag_names_a_value_the_catalog_accepts() {
        for sort in [
            CatalogSortArg::Downloads,
            CatalogSortArg::Recent,
            CatalogSortArg::Rating,
        ] {
            assert!(
                mold_catalog::live::CatalogSort::from_wire(sort_wire(sort)).is_some(),
                "{} is not a catalog sort",
                sort_wire(sort)
            );
        }
        assert_eq!(
            mold_catalog::live::CatalogSort::WIRE_VALUES,
            ["downloads", "recent", "rating"]
        );
    }

    /// A filter the catalog does not know is refused by name on the local
    /// path too — the server answers 400 for the same input, and a dropped
    /// filter would answer a different question.
    #[test]
    fn an_unknown_filter_is_refused_rather_than_dropped() {
        let bad_family = local_search_opts(&SearchArgs {
            family: Some("nonesuch".into()),
            ..args()
        })
        .unwrap_err()
        .to_string();
        assert!(
            bad_family.contains("unknown family: nonesuch"),
            "{bad_family}"
        );

        let bad_kind = local_search_opts(&SearchArgs {
            kind: Some("nonesuch".into()),
            ..args()
        })
        .unwrap_err()
        .to_string();
        assert!(bad_kind.contains("unknown kind: nonesuch"), "{bad_kind}");
    }

    /// The local path's own defaults match the endpoint's documented ones,
    /// so the same command reads the same way with and without a server.
    #[test]
    fn the_local_path_defaults_match_the_endpoints() {
        let opts = local_search_opts(&args()).unwrap();
        assert_eq!(opts.page, 1);
        assert_eq!(opts.page_size, 20);
        assert!(opts.include_nsfw);
        assert_eq!(opts.sort, mold_catalog::live::CatalogSort::Downloads);

        let clamped = local_search_opts(&SearchArgs {
            page: Some(0),
            page_size: Some(1000),
            ..args()
        })
        .unwrap();
        assert_eq!(clamped.page, 1);
        assert_eq!(clamped.page_size, 100);
    }

    /// A long title is trimmed to its column rather than wrapping the row.
    #[test]
    fn a_long_title_stays_inside_its_column() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a very long model title", 10), "a very lo…");
    }

    #[test]
    fn download_counts_read_as_magnitudes() {
        assert_eq!(format_count(42), "42");
        assert_eq!(format_count(1_500), "1.5k");
        assert_eq!(format_count(2_400_000), "2.4M");
    }
}
