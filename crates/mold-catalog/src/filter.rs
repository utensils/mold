//! Quality + safety filter applied after scanner stages but before sink.
//!
//! NSFW filtering at the scanner is *only* the user's explicit
//! `--no-nsfw` request. The runtime UI also filters by the persisted
//! `catalog.show_nsfw` setting; the two layers don't have to agree.

use crate::entry::{CatalogEntry, Source};
use crate::scanner::ScanOptions;

pub fn apply(entries: Vec<CatalogEntry>, options: &ScanOptions) -> Vec<CatalogEntry> {
    entries
        .into_iter()
        .filter(|e| !e.download_recipe.files.is_empty())
        .filter(|e| match e.source {
            Source::Civitai => e.download_count >= options.min_downloads,
            // HF doesn't surface a stable per-repo download_count for every
            // model — don't penalize HF entries with the threshold.
            Source::Hf => true,
        })
        .filter(|e| options.include_nsfw || !e.nsfw)
        .collect()
}
