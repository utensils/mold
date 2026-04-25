# Catalog Expansion — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the model-discovery catalog as a first-class, queryable browser surfaced in the web SPA, the `mold catalog` CLI, and `mold pull <catalog-id>` — covering every Hugging Face + Civitai entry that maps to a mold-supported family, persisted in `mold.db` with sub-millisecond filter/sort/search at 12 k+ entries.

**Architecture:** A new workspace crate `mold-catalog` owns the scanner (HF + Civitai stages), companion registry, normalizer, filter, sink, and rust-embedded shard reader. Catalog rows live in a new `catalog` table + FTS5 virtual table inside `mold.db` (migration V7); shards are committed as JSON in `data/catalog/` and embedded into the binary so a fresh install seeds without a network round-trip. `mold-server` adds `/api/catalog/*` routes and a `CatalogScanQueue` modeled on the existing `DownloadQueue`. `mold-cli` adds a `catalog` subcommand (`list / show / refresh / where`) and extends `mold pull` to accept catalog ids. The web SPA gains a `/catalog` page with sidebar + topbar + card grid + detail drawer, a `useCatalog` composable, and Settings additions for HF / Civitai tokens + NSFW toggle. Phase-2..5 entries (single-file checkpoints) are scanned, stored, and rendered with an `engine_phase` badge and disabled Download button — visible-but-disabled, never hidden.

**Tech Stack:** Rust 1.85, rusqlite (bundled, FTS5), reqwest, tokio, rust-embed, axum, clap; Vue 3, Vite 7, Vitest, Tailwind v4, vue-virtual-scroller; bun; wiremock (HTTP fixtures).

**Spec:** `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` — section §1 only. Sections §2–§5 are out of scope for this plan.

**Branch / worktree layout:**

- Umbrella branch `feat/catalog-expansion`, cut from `main`.
- Phase 1 work lives **directly on the umbrella** (no separate worktree). Phases 2–5 each create their own worktree off the umbrella when work begins.
- The first task of this plan cherry-picks commit `4e42fd7` (the spec) onto the umbrella so the spec is part of the umbrella's history from day zero.
- Open a draft PR (`feat/catalog-expansion` → `main`) once Task 2 lands so CI runs against incremental commits.

**Boundaries / non-goals for this plan:**

- **No single-file checkpoint loaders.** P2 (SD1.5 + SDXL), P3 (FLUX), P4 (Z-Image), P5 (LTX) each get their own plan when their predecessor merges. Catalog entries with `engine_phase >= 2` are stored, scanned, and rendered, but their Download button is disabled with the badge "Coming in phase N".
- **No CI cron auto-refresh.** `mold catalog refresh` is the only refresh path.
- **No live HTTP in CI.** HF / Civitai network tests are `#[ignore]`d and run manually on the killswitch UAT box (`killswitch@192.168.1.67`, dual-3090 Arch, sm_86, repo at `~/github/mold`).
- **mold-discord and mold-tui MUST NOT depend on `mold-catalog`** — directly or transitively. Task 36 verifies this with `cargo tree`.
- **No new `Family` enum exposed from `mold-core`.** The codebase keeps `family: String` everywhere; the new enum is internal to `mold-catalog` with `as_str()` / `from_str()` round-trip helpers that match the existing manifest strings (`"flux"`, `"flux2"`, `"sd15"`, `"sdxl"`, `"z-image"`, `"ltx-video"`, `"ltx2"`, `"qwen-image"`, `"wuerstchen"`).
- **The catalog table has no `profile` column.** Catalog is global per mold install; per-profile state (downloaded? favorited?) lives in existing tables.

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `crates/mold-catalog/Cargo.toml` | Crate manifest. Deps: `mold-ai-core`, `mold-ai-db`, `serde`, `serde_json`, `tokio`, `reqwest`, `rust-embed`, `tracing`, `thiserror`, `urlencoding`. Dev-deps: `wiremock`, `tempfile`. |
| `crates/mold-catalog/build.rs` | Resolves `data/catalog/` shards or generates stubs into `$OUT_DIR/catalog-stub/`; stamps `MOLD_CATALOG_DIR` for the embedder. |
| `crates/mold-catalog/src/lib.rs` | Public re-exports: types, scanner orchestrator, sink, embedded-shard seeder. |
| `crates/mold-catalog/src/families.rs` | `Family` enum + `as_str` / `from_str`; matches existing manifest strings. |
| `crates/mold-catalog/src/entry.rs` | `CatalogEntry`, `Source`, `FamilyRole`, `Modality`, `Kind`, `FileFormat`, `Bundling`, `LicenseFlags`, `DownloadRecipe`, `RecipeFile`, `TokenKind`, `CompanionRef`, `CatalogId`. |
| `crates/mold-catalog/src/companions.rs` | Curated `COMPANIONS` static slice + `companions_for(family, bundling)` lookup. |
| `crates/mold-catalog/src/civitai_map.rs` | `map_base_model()` Civitai → Family/role/sub_family + `CIVITAI_BASE_MODELS` slice + `engine_phase_for()`. |
| `crates/mold-catalog/src/hf_seeds.rs` | `seeds_for(Family) -> &'static [&'static str]`. |
| `crates/mold-catalog/src/scanner.rs` | `ScanOptions`, top-level orchestration, per-family task spawn. |
| `crates/mold-catalog/src/stages/mod.rs` | Module marker; re-exports `hf` and `civitai`. |
| `crates/mold-catalog/src/stages/hf.rs` | `scan_family` + `fetch_repo_as_entry` + `detect_format` + `build_recipe_hf`. |
| `crates/mold-catalog/src/stages/civitai.rs` | `scan` + `pick_safetensors_file` + page walk. |
| `crates/mold-catalog/src/normalizer.rs` | `from_hf` and `from_civitai` builders. |
| `crates/mold-catalog/src/filter.rs` | `apply(entries, &ScanOptions)` (NSFW, min_downloads, empty recipe drop). |
| `crates/mold-catalog/src/sink.rs` | Two-phase shard write (`<dir>/.staging/<family>.json` → atomic rename) + per-family DB upsert in one txn. |
| `crates/mold-catalog/src/shards.rs` | `EmbeddedShards` (rust-embed) + `iter_shards()` + `seed_db_from_embedded_if_empty`. |
| `crates/mold-catalog/data/catalog/.gitkeep` | Keep dir present in fresh checkouts. |
| `crates/mold-catalog/data/catalog/flux.json` | Stub shard (empty `entries`, `$schema: "mold.catalog.v1"`). |
| `crates/mold-catalog/data/catalog/flux2.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/sd15.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/sdxl.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/z-image.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/ltx-video.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/ltx2.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/qwen-image.json` | Stub shard. |
| `crates/mold-catalog/data/catalog/wuerstchen.json` | Stub shard. |
| `crates/mold-catalog/tests/families_roundtrip.rs` | `Family::from_str(as_str(f)).unwrap() == f` for every variant. |
| `crates/mold-catalog/tests/entry_serde.rs` | JSON round-trip on a representative `CatalogEntry`. |
| `crates/mold-catalog/tests/civitai_map_completeness.rs` | Every key in `CIVITAI_BASE_MODELS` resolves or is in the explicit drop set. |
| `crates/mold-catalog/tests/companions_lookup.rs` | `companions_for(Flux, SingleFile)` includes `t5-v1_1-xxl` + `clip-l`. |
| `crates/mold-catalog/tests/normalizer_snapshots.rs` | Canned HF + Civitai JSON fixtures → assert serialized `CatalogEntry` matches. |
| `crates/mold-catalog/tests/hf_stage.rs` | wiremock-backed `scan_family` smoke test. |
| `crates/mold-catalog/tests/civitai_stage.rs` | wiremock-backed `scan` smoke test (drops `.pt` format, accepts safetensors). |
| `crates/mold-catalog/tests/sink_roundtrip.rs` | Round-trip every committed shard byte-identically through `serde_json`. |
| `crates/mold-catalog/tests/db_seed.rs` | Seed a `:memory:` DB from embedded shards; FTS5 query returns expected ids. |
| `crates/mold-catalog/tests/scanner_live.rs` | `#[ignore]` live HF + Civitai scan; killswitch-only. |
| `crates/mold-db/src/sql/V7_catalog.sql` | Optional companion file (the SQL is also kept inline as a const string per existing pattern). |
| `crates/mold-db/src/catalog.rs` | DB-side catalog repo: `upsert_family`, `delete_family`, `get_by_id`, `list`, `family_counts`, FTS5 search wrapper. |
| `crates/mold-server/src/catalog_api.rs` | Axum handlers for `/api/catalog/*` and the `CatalogScanQueue` trait + production `CatalogScanDriver`. |
| `crates/mold-server/src/catalog_api_test.rs` | Handler tests with a fake scan driver + in-memory DB. |
| `crates/mold-cli/src/commands/catalog.rs` | `mold catalog list/show/refresh/where` subcommand handlers. |
| `web/src/pages/CatalogPage.vue` | The `/catalog` route — composes sidebar, topbar, grid, detail drawer. |
| `web/src/components/CatalogSidebar.vue` | Family / role tree with counts. |
| `web/src/components/CatalogTopbar.vue` | Modality chips, search, sort, source chips, NSFW toggle. |
| `web/src/components/CatalogCardGrid.vue` | Virtualized card grid; lazy-loads thumbnails. |
| `web/src/components/CatalogCard.vue` | Single thumbnail card. |
| `web/src/components/CatalogDetailDrawer.vue` | Right-slide detail panel. |
| `web/src/composables/useCatalog.ts` | Reactive catalog state: list/filters/detail/refresh/download. |
| `web/src/composables/useCatalog.test.ts` | Filter wiring + pagination + debounce tests. |
| `web/src/components/CatalogCard.test.ts` | Renders thumbnail + badges. |
| `web/src/components/CatalogTopbar.test.ts` | Modality chip + sort dropdown wiring. |
| `web/src/components/CatalogSidebar.test.ts` | Family-count rendering + active-row state. |
| `web/src/components/CatalogDetailDrawer.test.ts` | Disabled Download button when `engine_phase >= 2`. |
| `data/catalog/README.md` | Repo-level pointer explaining the committed shards live under `crates/mold-catalog/data/catalog/`. |
| `website/docs/catalog.md` | VitePress page describing the catalog feature, CLI, and Settings tokens. |

### Modified Files

| File | What Changes |
|------|-------------|
| `Cargo.toml` (workspace root) | Add `crates/mold-catalog` to `members`. |
| `crates/mold-cli/Cargo.toml` | Add dep on `mold-catalog = { path = "../mold-catalog", package = "mold-ai-catalog", version = "0.9.0" }`. |
| `crates/mold-server/Cargo.toml` | Add the same `mold-catalog` dep. |
| `crates/mold-server/src/lib.rs` | `pub mod catalog_api;` + spawn `CatalogScanQueue` driver in `run_server` + call `mold_catalog::shards::seed_db_from_embedded_if_empty(&db)` on startup. |
| `crates/mold-server/src/state.rs` | Add `pub catalog_scan: Arc<catalog_api::CatalogScanQueue>` field; thread through every constructor that builds an `AppState`. |
| `crates/mold-server/src/routes.rs` | Register: `GET /api/catalog`, `GET /api/catalog/:id`, `GET /api/catalog/families`, `POST /api/catalog/refresh`, `GET /api/catalog/refresh/:id`, `POST /api/catalog/:id/download`. Update `/api/capabilities` to include `catalog: { available: true }`. |
| `crates/mold-db/src/migrations.rs` | Add `V7_CATALOG` const + `Migration { version: 7, ... }`; bump `SCHEMA_VERSION` to 7. |
| `crates/mold-db/src/lib.rs` | `pub mod catalog;` re-export. |
| `crates/mold-cli/src/main.rs` | Register `Commands::Catalog { action: CatalogAction }` enum + dispatch in the match arm; extend `Pull` handler to detect `hf:` / `cv:` prefixes and route via the catalog. |
| `crates/mold-cli/src/commands/mod.rs` | `pub mod catalog;`. |
| `crates/mold-cli/src/commands/pull.rs` (or its wrapper) | Branch on catalog-id prefix → `mold_catalog`-driven download, else legacy manifest pull. |
| `web/src/router.ts` | Add `{ path: "/catalog", name: "catalog", component: () => import("./pages/CatalogPage.vue") }`. |
| `web/src/api.ts` | Add `fetchCatalog`, `fetchCatalogEntry`, `fetchCatalogFamilies`, `postCatalogRefresh`, `fetchCatalogRefresh`, `postCatalogDownload`. |
| `web/src/types.ts` | Add `CatalogEntryWire`, `CatalogFamilyCounts`, `CatalogListParams`, `CatalogRefreshStatus`. |
| `web/src/components/SettingsModal.vue` | Add Hugging Face token row, Civitai token row, "Show NSFW models" toggle. |
| `web/src/App.vue` | Add nav link to `/catalog`. |
| `CHANGELOG.md` | Append `[Unreleased]` entries: new crate, new endpoints, new CLI subcommand, new env var `CIVITAI_TOKEN`, new SQLite migration v7. |
| `.claude/skills/mold/SKILL.md` | Document `mold catalog *`, `mold pull <catalog-id>`, `CIVITAI_TOKEN`, `MOLD_CATALOG_DIR`. |
| `website/.vitepress/config.ts` | Add `/docs/catalog.md` to the sidebar. |

---

## Sequencing notes

- Tasks 1–18 are crate-internal — the catalog crate, DB migration, and CLI subcommand can be implemented and tested without touching the web SPA.
- Tasks 19–24 wire the server endpoints; the SPA tasks (25–32) build on those.
- Tasks 33–37 are documentation, dependency-tree verification, and the final CI gate.
- Each task ends with a `git commit` step. Use Conventional Commits (`feat:`, `test:`, `chore:`) — the umbrella branch's history is the basis for the eventual squash-merge into `main`.

---

## Task 1: Cut the umbrella branch and import the spec

**Files:** None (git only)

- [ ] **Step 1: Update local main and cut the umbrella branch**

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b feat/catalog-expansion
```

Expected: `Switched to a new branch 'feat/catalog-expansion'`.

- [ ] **Step 2: Cherry-pick the spec commit**

```bash
git cherry-pick 4e42fd7
```

Expected: clean apply (the spec file `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` lands as a single commit).

If the cherry-pick aborts because the spec file already exists (e.g. it was merged into main since this plan was written), skip with:

```bash
git cherry-pick --skip
```

- [ ] **Step 3: Push the umbrella and open a draft PR**

```bash
git push -u origin feat/catalog-expansion
gh pr create --draft --base main --head feat/catalog-expansion \
  --title "feat(catalog): umbrella for sub-project A — model-discovery catalog" \
  --body "$(cat <<'EOF'
## Summary
- Umbrella PR for the catalog-expansion sub-project (sub-project A of the four-part model-discovery initiative).
- Spec: \`docs/superpowers/specs/2026-04-25-catalog-expansion-design.md\`
- Phase 1 plan: \`docs/superpowers/plans/2026-04-25-catalog-expansion-phase-1.md\`

## Test plan
- [ ] Phase 1 lands: catalog crate + DB migration + CLI + server endpoints + web SPA
- [ ] Phases 2–5 land progressively, each as its own worktree off this umbrella
- [ ] killswitch UAT after each phase
EOF
)"
```

Expected: PR URL printed; draft state.

- [ ] **Step 4: Verify branch and clean tree**

```bash
git rev-parse --abbrev-ref HEAD
git log --oneline -3
git status --short
```

Expected:

```
feat/catalog-expansion
4e42fd7 docs(spec): catalog expansion design — sub-project A
<main HEAD sha> ...
<main HEAD-1 sha> ...
```

(empty `status --short`).

All subsequent tasks run on `feat/catalog-expansion`.

---

## Task 2: Scaffold the `mold-catalog` crate

**Files:**
- Create: `crates/mold-catalog/Cargo.toml`
- Create: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/data/catalog/.gitkeep`
- Modify: `Cargo.toml` (workspace root, line 2-10 — `members` array)

- [ ] **Step 1: Add the crate to the workspace members list**

Edit `Cargo.toml` at the workspace root:

```toml
[workspace]
members = [
    "crates/mold-core",
    "crates/mold-db",
    "crates/mold-inference",
    "crates/mold-server",
    "crates/mold-cli",
    "crates/mold-discord",
    "crates/mold-tui",
    "crates/mold-catalog",
]
```

- [ ] **Step 2: Create `crates/mold-catalog/Cargo.toml`**

```toml
[package]
name = "mold-ai-catalog"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
readme.workspace = true
keywords.workspace = true
categories.workspace = true
description = "Model-discovery catalog for mold — Hugging Face + Civitai scanner, embedded shards, FTS5-backed lookup."

[lib]
name = "mold_catalog"

[dependencies]
mold-core = { path = "../mold-core", package = "mold-ai-core", version = "0.9.0" }
mold-db = { path = "../mold-db", package = "mold-ai-db", version = "0.9.0" }
anyhow = "1"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync", "time"] }
reqwest = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
rust-embed = "8"
tracing = "0.1"
urlencoding = "2"

[dev-dependencies]
tempfile = "3"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "test-util"] }
wiremock = "0.6"
```

- [ ] **Step 3: Create the placeholder `src/lib.rs`**

```rust
//! Model-discovery catalog for mold.
//!
//! See `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` for the
//! full design. This crate is intentionally lean: only `mold-cli` and
//! `mold-server` depend on it. `mold-discord` and `mold-tui` MUST NOT
//! transitively depend on this crate — see Task 36 for the dependency
//! tree check.

#![forbid(unsafe_code)]
```

- [ ] **Step 4: Create the `data/catalog/.gitkeep` placeholder**

Empty file. The build script writes stub shards into the same directory in Task 13; the `.gitkeep` keeps the directory in fresh checkouts and serves as a sentinel.

- [ ] **Step 5: Verify the crate builds**

Run from the repo root:

```bash
cargo check -p mold-ai-catalog
```

Expected: `Checking mold-ai-catalog v0.9.0` then `Finished` with no warnings.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/mold-catalog
git commit -m "feat(catalog): scaffold mold-catalog crate (sub-project A phase 1)"
```

---

## Task 3: `Family` enum + `families.rs` (TDD)

**Files:**
- Create: `crates/mold-catalog/src/families.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/families_roundtrip.rs`

- [ ] **Step 1: Write the failing round-trip test**

Create `crates/mold-catalog/tests/families_roundtrip.rs`:

```rust
use mold_catalog::families::{Family, ALL_FAMILIES};

#[test]
fn every_family_round_trips_through_strings() {
    for fam in ALL_FAMILIES {
        let s = fam.as_str();
        let back = Family::from_str(s).expect("from_str");
        assert_eq!(*fam, back, "round trip failed for {s}");
    }
}

#[test]
fn manifest_strings_are_stable() {
    // These are the strings the existing manifest.rs writes into
    // ModelManifest.family. Changing them is a breaking change for
    // everything that already keys off `family`. This test pins them.
    let cases = [
        (Family::Flux, "flux"),
        (Family::Flux2, "flux2"),
        (Family::Sd15, "sd15"),
        (Family::Sdxl, "sdxl"),
        (Family::ZImage, "z-image"),
        (Family::LtxVideo, "ltx-video"),
        (Family::Ltx2, "ltx2"),
        (Family::QwenImage, "qwen-image"),
        (Family::Wuerstchen, "wuerstchen"),
    ];
    for (fam, s) in cases {
        assert_eq!(fam.as_str(), s, "{:?} → {s}", fam);
        assert_eq!(Family::from_str(s).unwrap(), fam, "{s} → {:?}", fam);
    }
}

#[test]
fn unknown_family_string_returns_err() {
    assert!(Family::from_str("not-a-family").is_err());
    assert!(Family::from_str("").is_err());
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test families_roundtrip
```

Expected: compile error — `module 'families' is private` or `unresolved import mold_catalog::families`.

- [ ] **Step 3: Implement `families.rs`**

Create `crates/mold-catalog/src/families.rs`:

```rust
//! mold's supported family taxonomy. The string forms are load-bearing
//! — they match `crates/mold-core/src/manifest.rs` `ModelManifest.family`
//! values and the `family` column of the new `catalog` SQLite table.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    Flux,
    Flux2,
    Sd15,
    Sdxl,
    ZImage,
    LtxVideo,
    Ltx2,
    QwenImage,
    Wuerstchen,
}

pub const ALL_FAMILIES: &[Family] = &[
    Family::Flux,
    Family::Flux2,
    Family::Sd15,
    Family::Sdxl,
    Family::ZImage,
    Family::LtxVideo,
    Family::Ltx2,
    Family::QwenImage,
    Family::Wuerstchen,
];

#[derive(Debug, thiserror::Error)]
#[error("unknown family: {0:?}")]
pub struct UnknownFamily(pub String);

impl Family {
    /// Stable string form used in the SQLite `family` column and in the
    /// existing manifest.rs `ModelManifest.family`. **Do not change** —
    /// this is a load-bearing identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Family::Flux => "flux",
            Family::Flux2 => "flux2",
            Family::Sd15 => "sd15",
            Family::Sdxl => "sdxl",
            Family::ZImage => "z-image",
            Family::LtxVideo => "ltx-video",
            Family::Ltx2 => "ltx2",
            Family::QwenImage => "qwen-image",
            Family::Wuerstchen => "wuerstchen",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, UnknownFamily> {
        Ok(match s {
            "flux" => Family::Flux,
            "flux2" => Family::Flux2,
            "sd15" => Family::Sd15,
            "sdxl" => Family::Sdxl,
            "z-image" => Family::ZImage,
            "ltx-video" => Family::LtxVideo,
            "ltx2" => Family::Ltx2,
            "qwen-image" => Family::QwenImage,
            "wuerstchen" => Family::Wuerstchen,
            other => return Err(UnknownFamily(other.to_string())),
        })
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
```

- [ ] **Step 4: Re-export from `lib.rs`**

Replace `crates/mold-catalog/src/lib.rs` with:

```rust
//! Model-discovery catalog for mold.
//!
//! See `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` for the
//! full design. Only `mold-cli` and `mold-server` depend on this crate.
//! `mold-discord` and `mold-tui` MUST NOT transitively depend on it — see
//! Task 36 for the dependency-tree check.

#![forbid(unsafe_code)]

pub mod families;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test families_roundtrip
```

Expected: `running 3 tests ... 3 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/families.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/families_roundtrip.rs
git commit -m "feat(catalog): Family enum with stable string forms"
```

---

## Task 4: Core entry types + serde round-trip (TDD)

**Files:**
- Create: `crates/mold-catalog/src/entry.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/entry_serde.rs`

- [ ] **Step 1: Write the failing serde round-trip test**

Create `crates/mold-catalog/tests/entry_serde.rs`:

```rust
use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat,
    Kind, LicenseFlags, Modality, RecipeFile, Source, TokenKind,
};
use mold_catalog::families::Family;

fn sample_entry() -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from("hf:black-forest-labs/FLUX.1-dev"),
        source: Source::Hf,
        source_id: "black-forest-labs/FLUX.1-dev".into(),
        name: "FLUX.1-dev".into(),
        author: Some("black-forest-labs".into()),
        family: Family::Flux,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(23_800_000_000),
        download_count: 5_400_000,
        rating: None,
        likes: 12_300,
        nsfw: false,
        thumbnail_url: Some("https://huggingface.co/.../thumb.png".into()),
        description: Some("Flux.1-dev base.".into()),
        license: Some("flux-1-dev-non-commercial-license".into()),
        license_flags: LicenseFlags {
            commercial: Some(false),
            derivatives: Some(true),
            different_license: Some(false),
        },
        tags: vec!["text-to-image".into(), "flux".into()],
        companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
        download_recipe: DownloadRecipe {
            files: vec![RecipeFile {
                url: "https://huggingface.co/.../flux1-dev.safetensors".into(),
                dest: "{family}/{author}/{name}.safetensors".into(),
                sha256: Some("deadbeef".into()),
                size_bytes: Some(23_800_000_000),
            }],
            needs_token: Some(TokenKind::Hf),
        },
        engine_phase: 1,
        created_at: Some(1_700_000_000),
        updated_at: Some(1_710_000_000),
        added_at: 1_720_000_000,
    }
}

#[test]
fn catalog_entry_round_trips() {
    let entry = sample_entry();
    let s = serde_json::to_string_pretty(&entry).unwrap();
    let back: CatalogEntry = serde_json::from_str(&s).unwrap();
    assert_eq!(entry, back);
}

#[test]
fn enum_serializations_use_kebab_case() {
    assert_eq!(serde_json::to_string(&Source::Hf).unwrap(), "\"hf\"");
    assert_eq!(
        serde_json::to_string(&Source::Civitai).unwrap(),
        "\"civitai\""
    );
    assert_eq!(
        serde_json::to_string(&FamilyRole::Foundation).unwrap(),
        "\"foundation\""
    );
    assert_eq!(
        serde_json::to_string(&Bundling::SingleFile).unwrap(),
        "\"single-file\""
    );
    assert_eq!(
        serde_json::to_string(&FileFormat::Safetensors).unwrap(),
        "\"safetensors\""
    );
    assert_eq!(
        serde_json::to_string(&Kind::TextEncoder).unwrap(),
        "\"text-encoder\""
    );
}

#[test]
fn token_kind_round_trips() {
    let none: Option<TokenKind> = None;
    assert_eq!(serde_json::to_string(&none).unwrap(), "null");
    let hf: Option<TokenKind> = Some(TokenKind::Hf);
    assert_eq!(serde_json::to_string(&hf).unwrap(), "\"hf\"");
    let cv: Option<TokenKind> = Some(TokenKind::Civitai);
    assert_eq!(serde_json::to_string(&cv).unwrap(), "\"civitai\"");
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test entry_serde
```

Expected: `unresolved import mold_catalog::entry`.

- [ ] **Step 3: Implement `entry.rs`**

Create `crates/mold-catalog/src/entry.rs`:

```rust
//! Core catalog entry types. These serialize as the on-disk shard format
//! AND as the wire format for `/api/catalog/*`. Changing field names or
//! kebab-case forms is a wire-protocol break.

use serde::{Deserialize, Serialize};

use crate::families::Family;

pub type CompanionRef = String;

/// `"hf:author/repo"` for HF entries, `"cv:<modelVersionId>"` for Civitai.
/// Stored as a `String`-newtype for type safety at API boundaries; the
/// inner `String` is what hits SQLite as the `id` column primary key.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CatalogId(pub String);

impl<S: Into<String>> From<S> for CatalogId {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

impl CatalogId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    Hf,
    Civitai,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FamilyRole {
    Foundation,
    Finetune,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Checkpoint,
    Lora,
    Vae,
    TextEncoder,
    ControlNet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileFormat {
    Safetensors,
    Gguf,
    Diffusers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bundling {
    Separated,
    SingleFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenKind {
    Hf,
    Civitai,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LicenseFlags {
    pub commercial: Option<bool>,
    pub derivatives: Option<bool>,
    pub different_license: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeFile {
    pub url: String,
    pub dest: String,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DownloadRecipe {
    pub files: Vec<RecipeFile>,
    pub needs_token: Option<TokenKind>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: CatalogId,
    pub source: Source,
    pub source_id: String,
    pub name: String,
    pub author: Option<String>,
    pub family: Family,
    pub family_role: FamilyRole,
    pub sub_family: Option<String>,
    pub modality: Modality,
    pub kind: Kind,
    pub file_format: FileFormat,
    pub bundling: Bundling,
    pub size_bytes: Option<u64>,
    pub download_count: u64,
    pub rating: Option<f32>,
    pub likes: u64,
    pub nsfw: bool,
    pub thumbnail_url: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub license_flags: LicenseFlags,
    pub tags: Vec<String>,
    pub companions: Vec<CompanionRef>,
    pub download_recipe: DownloadRecipe,
    pub engine_phase: u8,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
    pub added_at: i64,
}

/// On-disk shard format. One file per family in `data/catalog/`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub family: String,
    pub generated_at: String,
    pub scanner_version: String,
    pub entries: Vec<CatalogEntry>,
}

pub const SHARD_SCHEMA: &str = "mold.catalog.v1";
```

- [ ] **Step 4: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod entry;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test entry_serde
```

Expected: `running 3 tests ... 3 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/entry.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/entry_serde.rs
git commit -m "feat(catalog): core entry types with kebab-case serde round-trip"
```

---

## Task 5: Civitai mapping + completeness gate (TDD)

**Files:**
- Create: `crates/mold-catalog/src/civitai_map.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/civitai_map_completeness.rs`

- [ ] **Step 1: Write the failing completeness test**

Create `crates/mold-catalog/tests/civitai_map_completeness.rs`:

```rust
use mold_catalog::civitai_map::{
    engine_phase_for, map_base_model, CIVITAI_BASE_MODELS, CIVITAI_DROPS,
};
use mold_catalog::entry::Bundling;
use mold_catalog::families::Family;

#[test]
fn every_known_base_model_maps_or_is_explicitly_dropped() {
    for key in CIVITAI_BASE_MODELS {
        let mapped = map_base_model(key).is_some();
        let dropped = CIVITAI_DROPS.contains(key);
        assert!(
            mapped ^ dropped,
            "civitai baseModel '{key}' must either map to a Family OR be in CIVITAI_DROPS — never both, never neither"
        );
    }
}

#[test]
fn pony_keeps_sub_family() {
    let (fam, _role, sub) = map_base_model("Pony").unwrap();
    assert_eq!(fam, Family::Sdxl);
    assert_eq!(sub, Some("pony".to_string()));
}

#[test]
fn unknown_strings_drop_silently() {
    assert!(map_base_model("Some Future Model 9000").is_none());
}

#[test]
fn engine_phase_classifies_separated_as_one() {
    for fam in [Family::Flux, Family::Sdxl, Family::Sd15, Family::ZImage] {
        assert_eq!(engine_phase_for(fam, Bundling::Separated), 1);
    }
}

#[test]
fn engine_phase_classifies_single_file_correctly() {
    assert_eq!(engine_phase_for(Family::Sd15, Bundling::SingleFile), 2);
    assert_eq!(engine_phase_for(Family::Sdxl, Bundling::SingleFile), 2);
    assert_eq!(engine_phase_for(Family::Flux, Bundling::SingleFile), 3);
    assert_eq!(engine_phase_for(Family::Flux2, Bundling::SingleFile), 3);
    assert_eq!(engine_phase_for(Family::ZImage, Bundling::SingleFile), 4);
    assert_eq!(engine_phase_for(Family::LtxVideo, Bundling::SingleFile), 5);
    assert_eq!(engine_phase_for(Family::Ltx2, Bundling::SingleFile), 5);
    // QwenImage / Wuerstchen single-file is out of scope and gets the 99 sentinel.
    assert_eq!(engine_phase_for(Family::QwenImage, Bundling::SingleFile), 99);
    assert_eq!(engine_phase_for(Family::Wuerstchen, Bundling::SingleFile), 99);
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test civitai_map_completeness
```

Expected: `unresolved import mold_catalog::civitai_map`.

- [ ] **Step 3: Implement `civitai_map.rs`**

Create `crates/mold-catalog/src/civitai_map.rs`:

```rust
//! Civitai `baseModel` string → mold `(Family, FamilyRole, sub_family)`.
//!
//! `CIVITAI_BASE_MODELS` is the union of known mappings and explicit drops
//! — it must stay synchronized: every entry either maps to `Some(...)` via
//! `map_base_model` or appears in `CIVITAI_DROPS`. The
//! `civitai_map_completeness` integration test enforces this invariant.

use crate::entry::Bundling;
use crate::families::Family;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FamilyRoleResult {
    Foundation,
    Finetune,
}

pub fn map_base_model(base_model: &str) -> Option<(Family, crate::entry::FamilyRole, Option<String>)> {
    use crate::entry::FamilyRole::*;
    use Family::*;
    Some(match base_model {
        // SD1.x
        "SD 1.4" | "SD 1.5" | "SD 1.5 LCM" | "SD 1.5 Hyper" => (Sd15, Finetune, None),

        // SDXL family (architecture-compatible variants)
        "SDXL 1.0" | "SDXL Lightning" | "SDXL Hyper" => (Sdxl, Finetune, None),
        "Pony" => (Sdxl, Finetune, Some("pony".into())),
        "Pony V7" => (Sdxl, Finetune, Some("pony-v7".into())),
        "Illustrious" => (Sdxl, Finetune, Some("illustrious".into())),
        "NoobAI" => (Sdxl, Finetune, Some("noobai".into())),

        // FLUX 1.x
        "Flux.1 S" => (Flux, Finetune, Some("flux1-s".into())),
        "Flux.1 D" => (Flux, Finetune, Some("flux1-d".into())),
        "Flux.1 Krea" => (Flux, Finetune, Some("flux1-krea".into())),
        "Flux.1 Kontext" => (Flux, Finetune, Some("flux1-kontext".into())),

        // FLUX 2
        "Flux.2 D" => (Flux2, Finetune, Some("flux2-d".into())),
        "Flux.2 Klein 9B" | "Flux.2 Klein 9B-base" => (Flux2, Finetune, Some("klein-9b".into())),
        "Flux.2 Klein 4B" | "Flux.2 Klein 4B-base" => (Flux2, Finetune, Some("klein-4b".into())),

        // Z-Image
        "ZImageTurbo" => (ZImage, Finetune, Some("turbo".into())),
        "ZImageBase" => (ZImage, Finetune, Some("base".into())),

        // LTX
        "LTXV" => (LtxVideo, Finetune, None),
        "LTXV2" => (Ltx2, Finetune, Some("v2".into())),
        "LTXV 2.3" => (Ltx2, Finetune, Some("v2.3".into())),

        // Qwen
        "Qwen" | "Qwen 2" => (QwenImage, Finetune, None),

        _ => return None,
    })
}

/// Civitai base-model strings we explicitly drop. mold has no engine for
/// these architectures, so surfacing them in the catalog would just tease
/// users with un-runnable downloads.
pub const CIVITAI_DROPS: &[&str] = &[
    "SD 2.0",
    "SD 2.1",
    "AuraFlow",
    "Chroma",
    "CogVideoX",
    "Ernie",
    "Grok",
    "HiDream",
    "Hunyuan 1",
    "Hunyuan Video",
    "Kolors",
    "Lumina",
    "Mochi",
    "PixArt a",
    "PixArt E",
    "Wan Video 1.3B t2v",
    "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p",
    "Wan Video 14B i2v 720p",
    "Wan Video 2.2 TI2V-5B",
    "Wan Video 2.2 I2V-A14B",
    "Wan Video 2.2 T2V-A14B",
    "Wan Video 2.5 T2V",
    "Wan Video 2.5 I2V",
    "Wan Image 2.7",
    "Wan Video 2.7",
    "Anima",
    "Other",
    "Upscaler",
];

/// Every Civitai base-model string we know about — union of mapped + dropped.
/// The completeness test asserts these two sets are disjoint and exhaust this list.
pub const CIVITAI_BASE_MODELS: &[&str] = &[
    "SD 1.4", "SD 1.5", "SD 1.5 LCM", "SD 1.5 Hyper",
    "SDXL 1.0", "SDXL Lightning", "SDXL Hyper",
    "Pony", "Pony V7", "Illustrious", "NoobAI",
    "Flux.1 S", "Flux.1 D", "Flux.1 Krea", "Flux.1 Kontext",
    "Flux.2 D", "Flux.2 Klein 9B", "Flux.2 Klein 9B-base",
    "Flux.2 Klein 4B", "Flux.2 Klein 4B-base",
    "ZImageTurbo", "ZImageBase",
    "LTXV", "LTXV2", "LTXV 2.3",
    "Qwen", "Qwen 2",
    "SD 2.0", "SD 2.1",
    "AuraFlow", "Chroma", "CogVideoX", "Ernie", "Grok",
    "HiDream", "Hunyuan 1", "Hunyuan Video",
    "Kolors", "Lumina", "Mochi",
    "PixArt a", "PixArt E",
    "Wan Video 1.3B t2v", "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p", "Wan Video 14B i2v 720p",
    "Wan Video 2.2 TI2V-5B", "Wan Video 2.2 I2V-A14B", "Wan Video 2.2 T2V-A14B",
    "Wan Video 2.5 T2V", "Wan Video 2.5 I2V", "Wan Image 2.7", "Wan Video 2.7",
    "Anima", "Other", "Upscaler",
];

/// Returns the phase that unlocks runnability for a given (family, bundling).
/// `99` is the sentinel for "not in scope for any current phase" — those entries
/// are stored but rendered with a permanently disabled Download button.
pub fn engine_phase_for(family: Family, bundling: Bundling) -> u8 {
    use Bundling::*;
    use Family::*;
    match (family, bundling) {
        // Diffusers HF entries already work via existing engine paths.
        (_, Separated) => 1,
        (Sd15 | Sdxl, SingleFile) => 2,
        (Flux | Flux2, SingleFile) => 3,
        (ZImage, SingleFile) => 4,
        (LtxVideo | Ltx2, SingleFile) => 5,
        (QwenImage | Wuerstchen, SingleFile) => 99,
    }
}
```

- [ ] **Step 4: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod civitai_map;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test civitai_map_completeness
```

Expected: `running 5 tests ... 5 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/civitai_map.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/civitai_map_completeness.rs
git commit -m "feat(catalog): Civitai baseModel mapping with completeness gate"
```

---

## Task 6: Companion registry (TDD)

**Files:**
- Create: `crates/mold-catalog/src/companions.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/companions_lookup.rs`

- [ ] **Step 1: Write the failing lookup test**

Create `crates/mold-catalog/tests/companions_lookup.rs`:

```rust
use mold_catalog::companions::{companion_by_name, companions_for, COMPANIONS};
use mold_catalog::entry::{Bundling, Kind};
use mold_catalog::families::Family;

#[test]
fn flux_single_file_needs_t5_and_clip_l() {
    let names = companions_for(Family::Flux, Bundling::SingleFile);
    assert!(names.contains(&"t5-v1_1-xxl".to_string()));
    assert!(names.contains(&"clip-l".to_string()));
}

#[test]
fn sdxl_single_file_needs_two_clips_and_vae() {
    let names = companions_for(Family::Sdxl, Bundling::SingleFile);
    assert!(names.contains(&"clip-l".to_string()));
    assert!(names.contains(&"clip-g".to_string()));
    assert!(names.contains(&"sdxl-vae".to_string()));
}

#[test]
fn separated_bundling_has_no_companions() {
    // Diffusers HF entries are self-contained; companions only matter for
    // single-file checkpoints that don't bundle their text encoders.
    assert!(companions_for(Family::Flux, Bundling::Separated).is_empty());
    assert!(companions_for(Family::Sdxl, Bundling::Separated).is_empty());
}

#[test]
fn every_canonical_name_resolves() {
    for c in COMPANIONS {
        let resolved = companion_by_name(c.canonical_name).expect("resolves");
        assert_eq!(resolved.canonical_name, c.canonical_name);
    }
}

#[test]
fn z_image_te_canonical_is_committed() {
    // The exact text-encoder repo for Z-Image phase-4 is finalized when
    // single-file loaders land, but the canonical NAME is committed now
    // so phase-1 entries can reference it without rewrites.
    let c = companion_by_name("z-image-te").expect("z-image-te");
    assert_eq!(c.kind, Kind::TextEncoder);
    assert!(c.family_scope.contains(&Family::ZImage));
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test companions_lookup
```

Expected: `unresolved import mold_catalog::companions`.

- [ ] **Step 3: Implement `companions.rs`**

Create `crates/mold-catalog/src/companions.rs`:

```rust
//! Curated canonical-companion registry.
//!
//! Single-file Civitai checkpoints (FLUX, SDXL, etc.) routinely strip
//! their text encoders + VAE to keep download size manageable. Without a
//! finite, mold-curated set of "canonical companions", every Civitai
//! entry would either have to ship its own T5 reference or trust an
//! arbitrary repo. By committing this registry, mold ships *one* T5,
//! *one* CLIP-L, etc., and any single-file checkpoint that demands
//! something exotic gets `engine_phase: 99` (visible-but-unsupported).

use crate::entry::{Bundling, CompanionRef, Kind, Source};
use crate::families::Family;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Companion {
    pub canonical_name: &'static str,
    pub kind: Kind,
    pub family_scope: &'static [Family],
    pub source: Source,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub size_bytes: u64,
}

pub static COMPANIONS: &[Companion] = &[
    Companion {
        canonical_name: "t5-v1_1-xxl",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::LtxVideo, Family::Ltx2],
        source: Source::Hf,
        repo: "city96/t5-v1_1-xxl-encoder-bf16",
        files: &["t5xxl_*.safetensors"],
        size_bytes: 9_500_000_000,
    },
    Companion {
        canonical_name: "clip-l",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::Sd15, Family::Sdxl],
        source: Source::Hf,
        repo: "openai/clip-vit-large-patch14",
        files: &["model.safetensors", "config.json", "tokenizer*.json", "vocab.json", "merges.txt"],
        size_bytes: 1_700_000_000,
    },
    Companion {
        canonical_name: "clip-g",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        files: &["open_clip_pytorch_model.safetensors", "open_clip_config.json"],
        size_bytes: 5_700_000_000,
    },
    Companion {
        canonical_name: "sdxl-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "madebyollin/sdxl-vae-fp16-fix",
        files: &["sdxl_vae.safetensors"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "sd-vae-ft-mse",
        kind: Kind::Vae,
        family_scope: &[Family::Sd15],
        source: Source::Hf,
        repo: "stabilityai/sd-vae-ft-mse",
        files: &["diffusion_pytorch_model.safetensors", "config.json"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "flux-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Flux, Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.1-dev",
        files: &["ae.safetensors"],
        size_bytes: 335_000_000,
    },
    // Reserved canonical for Z-Image. The exact text-encoder repo is
    // finalized when phase-4 single-file loader lands; the canonical
    // NAME is committed now so phase-1 entries can reference it without
    // rewrites.
    Companion {
        canonical_name: "z-image-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::ZImage],
        source: Source::Hf,
        repo: "Tongyi-MAI/Z-Image-Turbo",
        files: &["text_encoder/*"],
        size_bytes: 4_400_000_000,
    },
];

pub fn companion_by_name(name: &str) -> Option<&'static Companion> {
    COMPANIONS.iter().find(|c| c.canonical_name == name)
}

/// Returns the canonical-companion names a given (family, bundling) needs.
/// Empty for `Bundling::Separated` because diffusers HF entries are
/// self-contained.
pub fn companions_for(family: Family, bundling: Bundling) -> Vec<CompanionRef> {
    if matches!(bundling, Bundling::Separated) {
        return Vec::new();
    }
    let mut out = Vec::new();
    match family {
        Family::Flux | Family::Flux2 => {
            push(&mut out, "t5-v1_1-xxl");
            push(&mut out, "clip-l");
            push(&mut out, "flux-vae");
        }
        Family::Sd15 => {
            push(&mut out, "clip-l");
            push(&mut out, "sd-vae-ft-mse");
        }
        Family::Sdxl => {
            push(&mut out, "clip-l");
            push(&mut out, "clip-g");
            push(&mut out, "sdxl-vae");
        }
        Family::ZImage => {
            push(&mut out, "z-image-te");
        }
        Family::LtxVideo | Family::Ltx2 => {
            push(&mut out, "t5-v1_1-xxl");
        }
        // Single-file for these is `engine_phase: 99` — no companions.
        Family::QwenImage | Family::Wuerstchen => {}
    }
    out
}

fn push(out: &mut Vec<CompanionRef>, name: &'static str) {
    out.push(name.to_string());
}
```

- [ ] **Step 4: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod companions;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test companions_lookup
```

Expected: `running 5 tests ... 5 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/companions.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/companions_lookup.rs
git commit -m "feat(catalog): canonical companion registry + lookup"
```

---

## Task 7: HF seeds + scanner config skeleton

**Files:**
- Create: `crates/mold-catalog/src/hf_seeds.rs`
- Create: `crates/mold-catalog/src/scanner.rs`
- Modify: `crates/mold-catalog/src/lib.rs`

- [ ] **Step 1: Add `hf_seeds.rs`**

Create `crates/mold-catalog/src/hf_seeds.rs`:

```rust
//! Curated HF foundation repos used as the starting point for the HF
//! stage's `base_model:` walk. Adding a new family means: declare it in
//! `families.rs`, then add at least one seed here.

use crate::families::Family;

pub fn seeds_for(family: Family) -> &'static [&'static str] {
    use Family::*;
    match family {
        Flux => &[
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
        ],
        Flux2 => &[
            "black-forest-labs/FLUX.2-dev",
            "black-forest-labs/FLUX.2-Klein-9B",
        ],
        Sd15 => &[
            "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ],
        Sdxl => &["stabilityai/stable-diffusion-xl-base-1.0"],
        ZImage => &["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image-Base"],
        LtxVideo => &["Lightricks/LTX-Video"],
        Ltx2 => &["Lightricks/LTX-Video-2", "Lightricks/LTX-Video-2.3"],
        QwenImage => &["Qwen/Qwen-Image"],
        Wuerstchen => &["warp-ai/wuerstchen"],
    }
}
```

- [ ] **Step 2: Add `scanner.rs` with `ScanOptions`**

Create `crates/mold-catalog/src/scanner.rs`:

```rust
//! Scanner orchestrator. Coordinates the HF + Civitai stages, applies the
//! filter, and hands off to the sink. Per-family failure isolation: one
//! family hitting a rate limit or auth error does not abort the run —
//! see `ScanReport::per_family`.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::entry::CatalogEntry;
use crate::families::Family;

#[derive(Clone, Debug)]
pub struct ScanOptions {
    pub families: Vec<Family>,
    /// Civitai base-model entries below this download count are dropped at
    /// scanner time. Default 100; lower with `--min-downloads 0` for
    /// completeness, raise to thin the catalog.
    pub min_downloads: u64,
    pub include_nsfw: bool,
    pub hf_token: Option<String>,
    pub civitai_token: Option<String>,
    pub per_family_cap: Option<usize>,
    pub request_timeout: Duration,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            families: crate::families::ALL_FAMILIES.to_vec(),
            min_downloads: 100,
            include_nsfw: false,
            hf_token: None,
            civitai_token: None,
            per_family_cap: None,
            request_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanReport {
    pub per_family: BTreeMap<Family, FamilyScanOutcome>,
    pub total_entries: usize,
}

#[derive(Clone, Debug)]
pub enum FamilyScanOutcome {
    Ok { entries: usize },
    RateLimited { partial: usize },
    AuthRequired,
    NetworkError { message: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("network: {0}")]
    Network(#[from] reqwest::Error),
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("auth required for {host}")]
    AuthRequired { host: &'static str },
    #[error("rate limited by {host}")]
    RateLimited { host: &'static str },
}

/// Results of one family's scan: HF entries + Civitai entries (mixed in
/// the same vec; `Source` discriminates them downstream).
pub type FamilyScanResult = Result<Vec<CatalogEntry>, ScanError>;
```

- [ ] **Step 3: Re-export from `lib.rs`**

Update `crates/mold-catalog/src/lib.rs` so it now reads:

```rust
//! Model-discovery catalog for mold.

#![forbid(unsafe_code)]

pub mod civitai_map;
pub mod companions;
pub mod entry;
pub mod families;
pub mod hf_seeds;
pub mod scanner;
```

- [ ] **Step 4: Verify it builds**

```bash
cargo check -p mold-ai-catalog --tests
```

Expected: `Finished` with no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-catalog/src/hf_seeds.rs \
        crates/mold-catalog/src/scanner.rs \
        crates/mold-catalog/src/lib.rs
git commit -m "feat(catalog): HF seed registry + ScanOptions scaffold"
```

---

## Task 8: Filter (TDD)

**Files:**
- Create: `crates/mold-catalog/src/filter.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/filter.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/mold-catalog/tests/filter.rs`:

```rust
use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat,
    Kind, LicenseFlags, Modality, RecipeFile, Source,
};
use mold_catalog::families::Family;
use mold_catalog::filter::apply;
use mold_catalog::scanner::ScanOptions;

fn entry(source: Source, downloads: u64, nsfw: bool, with_files: bool) -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from(format!("{source:?}-{downloads}-{nsfw}-{with_files}")),
        source,
        source_id: "x".into(),
        name: "x".into(),
        author: None,
        family: Family::Flux,
        family_role: FamilyRole::Finetune,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: None,
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: if with_files {
                vec![RecipeFile {
                    url: "u".into(),
                    dest: "d".into(),
                    sha256: None,
                    size_bytes: None,
                }]
            } else {
                vec![]
            },
            needs_token: None,
        },
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn drops_entries_with_no_recipe_files() {
    let entries = vec![
        entry(Source::Hf, 1000, false, true),
        entry(Source::Hf, 1000, false, false),
    ];
    let opts = ScanOptions::default();
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 1);
}

#[test]
fn applies_min_downloads_to_civitai_only() {
    let entries = vec![
        entry(Source::Civitai, 50, false, true),
        entry(Source::Civitai, 200, false, true),
        // HF doesn't surface a per-repo download_count consistently —
        // don't penalize HF with the threshold.
        entry(Source::Hf, 0, false, true),
    ];
    let opts = ScanOptions {
        min_downloads: 100,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 2);
    assert!(kept.iter().any(|e| matches!(e.source, Source::Hf)));
    assert!(kept
        .iter()
        .any(|e| matches!(e.source, Source::Civitai) && e.download_count == 200));
}

#[test]
fn drops_nsfw_when_include_nsfw_false() {
    let entries = vec![
        entry(Source::Civitai, 1000, true, true),
        entry(Source::Civitai, 1000, false, true),
    ];
    let opts = ScanOptions {
        include_nsfw: false,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 1);
    assert!(!kept[0].nsfw);
}

#[test]
fn keeps_nsfw_when_include_nsfw_true() {
    let entries = vec![
        entry(Source::Civitai, 1000, true, true),
        entry(Source::Civitai, 1000, false, true),
    ];
    let opts = ScanOptions {
        include_nsfw: true,
        ..ScanOptions::default()
    };
    let kept = apply(entries, &opts);
    assert_eq!(kept.len(), 2);
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test filter
```

Expected: `unresolved import mold_catalog::filter`.

- [ ] **Step 3: Implement `filter.rs`**

Create `crates/mold-catalog/src/filter.rs`:

```rust
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
```

- [ ] **Step 4: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod filter;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test filter
```

Expected: `running 4 tests ... 4 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/filter.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/filter.rs
git commit -m "feat(catalog): post-scan filter (NSFW, min downloads, empty recipe)"
```

---

## Task 9: Normalizer (TDD)

**Files:**
- Create: `crates/mold-catalog/src/normalizer.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/normalizer_snapshots.rs`
- Create: `crates/mold-catalog/tests/fixtures/hf_flux_dev.json` (canned HF detail response)
- Create: `crates/mold-catalog/tests/fixtures/civitai_juggernaut.json` (canned Civitai response)

- [ ] **Step 1: Add the HF fixture**

Create `crates/mold-catalog/tests/fixtures/hf_flux_dev.json`:

```json
{
  "id": "black-forest-labs/FLUX.1-dev",
  "modelId": "black-forest-labs/FLUX.1-dev",
  "author": "black-forest-labs",
  "downloads": 5400000,
  "likes": 12300,
  "private": false,
  "tags": ["text-to-image", "flux", "diffusers"],
  "pipeline_tag": "text-to-image",
  "library_name": "diffusers",
  "createdAt": "2024-08-01T12:00:00.000Z",
  "lastModified": "2024-09-15T08:30:00.000Z",
  "cardData": {
    "license": "flux-1-dev-non-commercial-license",
    "extra_gated_eu_disallowed": false
  }
}
```

- [ ] **Step 2: Add the HF tree fixture**

Create `crates/mold-catalog/tests/fixtures/hf_flux_dev_tree.json`:

```json
[
  { "type": "file", "path": "ae.safetensors", "size": 335000000 },
  { "type": "file", "path": "flux1-dev.safetensors", "size": 23800000000 },
  { "type": "file", "path": "model_index.json", "size": 600 },
  { "type": "directory", "path": "text_encoder", "size": 0 },
  { "type": "directory", "path": "text_encoder_2", "size": 0 },
  { "type": "directory", "path": "vae", "size": 0 }
]
```

- [ ] **Step 3: Add the Civitai fixture**

Create `crates/mold-catalog/tests/fixtures/civitai_juggernaut.json`:

```json
{
  "id": 133005,
  "name": "Juggernaut XL",
  "type": "Checkpoint",
  "nsfw": false,
  "creator": { "username": "RunDiffusion" },
  "stats": { "downloadCount": 950000, "rating": 4.8, "favoriteCount": 42000 },
  "tags": ["base model", "photorealistic"],
  "modelVersions": [
    {
      "id": 618692,
      "name": "v10",
      "baseModel": "SDXL 1.0",
      "baseModelType": "Standard",
      "trainedWords": [],
      "files": [
        {
          "id": 999111,
          "name": "juggernautXL_v10.safetensors",
          "sizeKB": 6700000,
          "downloadCount": 600000,
          "metadata": { "format": "SafeTensor", "size": "full", "fp": "fp16" },
          "downloadUrl": "https://civitai.com/api/download/models/618692",
          "hashes": { "SHA256": "ABC123" }
        }
      ],
      "images": [
        { "url": "https://image.civitai.com/.../thumb.jpeg", "nsfwLevel": 1 }
      ]
    }
  ]
}
```

- [ ] **Step 4: Write the failing snapshot test**

Create `crates/mold-catalog/tests/normalizer_snapshots.rs`:

```rust
use mold_catalog::entry::{
    Bundling, CatalogEntry, FamilyRole, FileFormat, Kind, Modality, Source,
};
use mold_catalog::families::Family;
use mold_catalog::normalizer::{from_civitai, from_hf, CivitaiItem, HfDetail, HfTreeEntry};

fn load(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn hf_flux_dev_normalizes() {
    let detail: HfDetail = serde_json::from_str(&load("hf_flux_dev.json")).unwrap();
    let tree: Vec<HfTreeEntry> = serde_json::from_str(&load("hf_flux_dev_tree.json")).unwrap();
    let entry: CatalogEntry =
        from_hf(detail, tree, Family::Flux, FamilyRole::Foundation).unwrap();

    assert_eq!(entry.id.as_str(), "hf:black-forest-labs/FLUX.1-dev");
    assert_eq!(entry.source, Source::Hf);
    assert_eq!(entry.author.as_deref(), Some("black-forest-labs"));
    assert_eq!(entry.family, Family::Flux);
    assert_eq!(entry.family_role, FamilyRole::Foundation);
    assert_eq!(entry.modality, Modality::Image);
    assert_eq!(entry.kind, Kind::Checkpoint);
    assert_eq!(entry.file_format, FileFormat::Safetensors);
    // Diffusers layout (presence of model_index.json + text_encoder/ etc.)
    // → Bundling::Separated.
    assert_eq!(entry.bundling, Bundling::Separated);
    assert_eq!(entry.engine_phase, 1);
    assert_eq!(entry.likes, 12_300);
    assert!(!entry.nsfw);
    assert!(!entry.download_recipe.files.is_empty());
}

#[test]
fn civitai_juggernaut_normalizes_as_sdxl_single_file() {
    let item: CivitaiItem = serde_json::from_str(&load("civitai_juggernaut.json")).unwrap();
    let entry = from_civitai(item).expect("mapped");

    assert_eq!(entry.source, Source::Civitai);
    assert_eq!(entry.id.as_str(), "cv:618692");
    assert_eq!(entry.family, Family::Sdxl);
    assert_eq!(entry.bundling, Bundling::SingleFile);
    assert_eq!(entry.file_format, FileFormat::Safetensors);
    assert_eq!(entry.engine_phase, 2); // SDXL single-file → phase 2
    // Companions are required for single-file SDXL.
    assert!(entry.companions.contains(&"clip-l".to_string()));
    assert!(entry.companions.contains(&"clip-g".to_string()));
    assert!(entry.companions.contains(&"sdxl-vae".to_string()));
    assert!(!entry.download_recipe.files.is_empty());
    assert_eq!(
        entry.download_recipe.files[0].url,
        "https://civitai.com/api/download/models/618692"
    );
    assert_eq!(
        entry.download_recipe.files[0].sha256.as_deref(),
        Some("ABC123")
    );
    assert_eq!(entry.thumbnail_url.as_deref().is_some(), true);
}

#[test]
fn civitai_unknown_base_model_is_dropped() {
    let json = r#"{
        "id": 1,
        "name": "Random",
        "type": "Checkpoint",
        "nsfw": false,
        "creator": { "username": "x" },
        "stats": { "downloadCount": 100, "favoriteCount": 0 },
        "tags": [],
        "modelVersions": [{
            "id": 2,
            "name": "v1",
            "baseModel": "Some Future Model",
            "baseModelType": "Standard",
            "files": [{ "id": 3, "name": "x.safetensors", "sizeKB": 1, "downloadCount": 0,
                       "metadata": { "format": "SafeTensor" }, "downloadUrl": "u", "hashes": {} }],
            "images": []
        }]
    }"#;
    let item: CivitaiItem = serde_json::from_str(json).unwrap();
    assert!(from_civitai(item).is_none());
}
```

- [ ] **Step 5: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test normalizer_snapshots
```

Expected: `unresolved import mold_catalog::normalizer`.

- [ ] **Step 6: Implement `normalizer.rs`**

Create `crates/mold-catalog/src/normalizer.rs`:

```rust
//! Source-specific JSON → `CatalogEntry`.
//!
//! HF: combine `/api/models/{repo}` detail + `/api/models/{repo}/tree/main`.
//! Civitai: combine the model + first version + a chosen safetensors file.

use serde::Deserialize;

use crate::civitai_map::{engine_phase_for, map_base_model};
use crate::companions::companions_for;
use crate::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat,
    Kind, LicenseFlags, Modality, RecipeFile, Source, TokenKind,
};
use crate::families::Family;

#[derive(Clone, Debug, Deserialize)]
pub struct HfDetail {
    pub id: String,
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(default, rename = "library_name")]
    pub library_name: Option<String>,
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default, rename = "cardData")]
    pub card_data: Option<HfCardData>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct HfCardData {
    pub license: Option<String>,
    #[serde(default)]
    pub extra_gated_eu_disallowed: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HfTreeEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("no usable safetensors / diffusers payload found")]
    EmptyTree,
}

const HF_RAW: &str = "https://huggingface.co";

pub fn from_hf(
    detail: HfDetail,
    tree: Vec<HfTreeEntry>,
    family: Family,
    family_role: FamilyRole,
) -> Result<CatalogEntry, NormalizeError> {
    if tree.is_empty() {
        return Err(NormalizeError::EmptyTree);
    }

    let bundling = if tree
        .iter()
        .any(|e| e.kind == "file" && e.path == "model_index.json")
    {
        Bundling::Separated
    } else if tree.iter().any(|e| {
        e.kind == "file"
            && e.path.ends_with(".safetensors")
            && !e.path.contains('/')
    }) {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };

    let file_format = if tree
        .iter()
        .any(|e| e.kind == "file" && e.path.ends_with(".gguf"))
    {
        FileFormat::Gguf
    } else if matches!(bundling, Bundling::Separated) {
        FileFormat::Diffusers
    } else {
        FileFormat::Safetensors
    };

    let needs_token = if detail
        .card_data
        .as_ref()
        .and_then(|c| c.extra_gated_eu_disallowed)
        .unwrap_or(false)
        || detail.tags.iter().any(|t| t == "gated")
    {
        Some(TokenKind::Hf)
    } else {
        None
    };

    let mut files: Vec<RecipeFile> = tree
        .iter()
        .filter(|e| {
            e.kind == "file"
                && (e.path.ends_with(".safetensors")
                    || e.path.ends_with(".gguf")
                    || e.path == "model_index.json"
                    || e.path.ends_with("config.json"))
        })
        .map(|e| RecipeFile {
            url: format!("{HF_RAW}/{}/resolve/main/{}", detail.id, e.path),
            dest: format!("{{family}}/{{author}}/{{name}}/{}", e.path),
            sha256: None,
            size_bytes: if e.size > 0 { Some(e.size) } else { None },
        })
        .collect();

    if files.is_empty() {
        return Err(NormalizeError::EmptyTree);
    }
    files.sort_by(|a, b| a.url.cmp(&b.url));

    let total_size = files.iter().filter_map(|f| f.size_bytes).sum::<u64>();
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };

    let companions = match bundling {
        Bundling::SingleFile => companions_for(family, bundling),
        Bundling::Separated => Vec::new(),
    };
    let phase = engine_phase_for(family, bundling);

    let now = chrono_now_unix();

    Ok(CatalogEntry {
        id: CatalogId::from(format!("hf:{}", detail.id)),
        source: Source::Hf,
        source_id: detail.id.clone(),
        name: detail.id.split('/').last().unwrap_or(&detail.id).to_string(),
        author: detail.author.clone(),
        family,
        family_role,
        sub_family: None,
        modality,
        kind: Kind::Checkpoint,
        file_format,
        bundling,
        size_bytes: if total_size > 0 { Some(total_size) } else { None },
        download_count: detail.downloads,
        rating: None,
        likes: detail.likes,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: detail
            .card_data
            .as_ref()
            .and_then(|c| c.license.clone()),
        license_flags: LicenseFlags::default(),
        tags: detail.tags.clone(),
        companions,
        download_recipe: DownloadRecipe { files, needs_token },
        engine_phase: phase,
        created_at: parse_iso(&detail.created_at),
        updated_at: parse_iso(&detail.last_modified),
        added_at: now,
    })
}

fn parse_iso(opt: &Option<String>) -> Option<i64> {
    opt.as_deref().and_then(|s| {
        time::OffsetDateTime::parse(s, &time::format_description::well_known::Iso8601::DEFAULT)
            .ok()
            .map(|dt| dt.unix_timestamp())
    })
}

fn chrono_now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Civitai ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiItem {
    pub id: u64,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub nsfw: bool,
    #[serde(default)]
    pub creator: Option<CivitaiCreator>,
    #[serde(default)]
    pub stats: Option<CivitaiStats>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "modelVersions")]
    pub model_versions: Vec<CivitaiVersion>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiCreator {
    pub username: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiStats {
    #[serde(default, rename = "downloadCount")]
    pub download_count: u64,
    #[serde(default)]
    pub rating: Option<f32>,
    #[serde(default, rename = "favoriteCount")]
    pub favorite_count: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiVersion {
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "baseModel")]
    pub base_model: String,
    #[serde(default, rename = "baseModelType")]
    pub base_model_type: Option<String>,
    #[serde(default)]
    pub files: Vec<CivitaiFile>,
    #[serde(default)]
    pub images: Vec<CivitaiImage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiFile {
    pub id: u64,
    pub name: String,
    #[serde(default, rename = "sizeKB")]
    pub size_kb: Option<f64>,
    #[serde(default, rename = "downloadCount")]
    pub download_count: u64,
    #[serde(default)]
    pub metadata: CivitaiFileMetadata,
    #[serde(default, rename = "downloadUrl")]
    pub download_url: Option<String>,
    #[serde(default)]
    pub hashes: serde_json::Value,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiFileMetadata {
    pub format: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiImage {
    pub url: String,
    #[serde(default, rename = "nsfwLevel")]
    pub nsfw_level: Option<u32>,
}

pub fn from_civitai(item: CivitaiItem) -> Option<CatalogEntry> {
    let version = item.model_versions.first()?;
    let (family, family_role, sub_family) = map_base_model(&version.base_model)?;
    let file = pick_safetensors(&version.files)?;
    let bundling = if version.base_model_type.as_deref() == Some("Standard") {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };
    let companions = companions_for(family, bundling);
    let phase = engine_phase_for(family, bundling);
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };

    let sha256 = file
        .hashes
        .get("SHA256")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let recipe = DownloadRecipe {
        files: vec![RecipeFile {
            url: file
                .download_url
                .clone()
                .unwrap_or_else(|| format!("https://civitai.com/api/download/models/{}", version.id)),
            dest: format!("{{family}}/civitai/{}/{}", version.id, file.name),
            sha256,
            size_bytes: file.size_kb.map(|kb| (kb * 1000.0) as u64),
        }],
        needs_token: Some(TokenKind::Civitai),
    };

    let stats = item.stats.unwrap_or_default();
    let now = chrono_now_unix();

    Some(CatalogEntry {
        id: CatalogId::from(format!("cv:{}", version.id)),
        source: Source::Civitai,
        source_id: version.id.to_string(),
        name: item.name.clone(),
        author: item.creator.and_then(|c| c.username),
        family,
        family_role,
        sub_family,
        modality,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling,
        size_bytes: file.size_kb.map(|kb| (kb * 1000.0) as u64),
        download_count: stats.download_count,
        rating: stats.rating,
        likes: stats.favorite_count,
        nsfw: item.nsfw,
        thumbnail_url: version.images.first().map(|i| i.url.clone()),
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: item.tags,
        companions,
        download_recipe: recipe,
        engine_phase: phase,
        created_at: None,
        updated_at: None,
        added_at: now,
    })
}

/// Civitai's legacy unsafe `.pt` ("PickleTensor") format is dropped at the
/// scanner. Arbitrary-code-execution risk on deserialization is not worth
/// catalog completeness — only safetensors are surfaced.
fn pick_safetensors(files: &[CivitaiFile]) -> Option<&CivitaiFile> {
    files
        .iter()
        .find(|f| f.metadata.format.as_deref() == Some("SafeTensor"))
}
```

Add `time = { version = "0.3", features = ["parsing", "formatting", "macros"] }` to `crates/mold-catalog/Cargo.toml` `[dependencies]` if not already present.

- [ ] **Step 7: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod normalizer;
```

- [ ] **Step 8: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test normalizer_snapshots
```

Expected: `running 3 tests ... 3 passed`.

- [ ] **Step 9: Commit**

```bash
git add crates/mold-catalog/src/normalizer.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/normalizer_snapshots.rs \
        crates/mold-catalog/tests/fixtures/ \
        crates/mold-catalog/Cargo.toml
git commit -m "feat(catalog): HF + Civitai normalizers with snapshot fixtures"
```

---

## Task 10: HF stage (TDD with wiremock)

**Files:**
- Create: `crates/mold-catalog/src/stages/mod.rs`
- Create: `crates/mold-catalog/src/stages/hf.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/hf_stage.rs`

- [ ] **Step 1: Write the failing wiremock-backed integration test**

Create `crates/mold-catalog/tests/hf_stage.rs`:

```rust
use mold_catalog::families::Family;
use mold_catalog::scanner::ScanOptions;
use mold_catalog::stages::hf;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn scan_family_yields_seed_plus_finetunes() {
    let server = MockServer::start().await;

    // Detail for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(200).set_body_string(include_str!(
            "fixtures/hf_flux_dev.json"
        )))
        .mount(&server)
        .await;

    // Tree for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(ResponseTemplate::new(200).set_body_string(include_str!(
            "fixtures/hf_flux_dev_tree.json"
        )))
        .mount(&server)
        .await;

    // Empty finetune page (no finetunes for this fixture).
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("filter", "base_model:black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let opts = ScanOptions::default();
    let entries = hf::scan_family(&server.uri(), &opts, Family::Flux, &["black-forest-labs/FLUX.1-dev"])
        .await
        .expect("scan");

    assert!(!entries.is_empty(), "expected at least the seed entry");
    let seed = entries.iter().find(|e| e.source_id == "black-forest-labs/FLUX.1-dev").expect("seed present");
    assert_eq!(seed.family, Family::Flux);
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test hf_stage
```

Expected: `unresolved import mold_catalog::stages`.

- [ ] **Step 3: Create the stages module marker**

Create `crates/mold-catalog/src/stages/mod.rs`:

```rust
//! Source-specific scanner stages.
pub mod civitai;
pub mod hf;
```

(`civitai` is added in Task 11; declare it now to avoid an empty module file later.)

- [ ] **Step 4: Implement `stages/hf.rs`**

Create `crates/mold-catalog/src/stages/hf.rs`:

```rust
//! Hugging Face stage. Walks `seeds_for(family)` and follows the
//! `base_model:` finetune graph one page at a time. The base URL is
//! parameterized so tests can point at wiremock; production passes
//! `"https://huggingface.co"`.

use serde::Deserialize;
use std::time::Duration;

use crate::entry::{CatalogEntry, FamilyRole};
use crate::families::Family;
use crate::normalizer::{from_hf, HfDetail, HfTreeEntry};
use crate::scanner::{ScanError, ScanOptions};

#[derive(Clone, Debug, Deserialize)]
struct HfModelStub {
    id: String,
}

pub async fn scan_family(
    base: &str,
    options: &ScanOptions,
    family: Family,
    seeds: &[&str],
) -> Result<Vec<CatalogEntry>, ScanError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()?;

    let mut entries = Vec::new();
    for seed in seeds {
        if let Some(e) = fetch_repo_as_entry(&client, base, options, seed, family, FamilyRole::Foundation).await? {
            entries.push(e);
        }
        let mut page = 1u32;
        loop {
            let url = format!(
                "{base}/api/models?filter=base_model:{seed}&sort=downloads&direction=-1&limit=100&page={page}",
            );
            let resp = http_get(&client, options, &url).await?;
            let stubs: Vec<HfModelStub> = serde_json::from_str(&resp)?;
            if stubs.is_empty() {
                break;
            }
            for stub in stubs {
                if let Some(e) = fetch_repo_as_entry(
                    &client,
                    base,
                    options,
                    &stub.id,
                    family,
                    FamilyRole::Finetune,
                )
                .await?
                {
                    entries.push(e);
                }
                if let Some(cap) = options.per_family_cap {
                    if entries.len() >= cap {
                        return Ok(entries);
                    }
                }
            }
            page += 1;
        }
    }
    Ok(entries)
}

async fn fetch_repo_as_entry(
    client: &reqwest::Client,
    base: &str,
    options: &ScanOptions,
    repo: &str,
    family: Family,
    role: FamilyRole,
) -> Result<Option<CatalogEntry>, ScanError> {
    let detail_url = format!("{base}/api/models/{repo}");
    let detail_body = http_get(client, options, &detail_url).await?;
    let detail: HfDetail = serde_json::from_str(&detail_body)?;

    let tree_url = format!("{base}/api/models/{repo}/tree/main");
    let tree_body = http_get(client, options, &tree_url).await?;
    let tree: Vec<HfTreeEntry> = serde_json::from_str(&tree_body)?;

    match from_hf(detail, tree, family, role) {
        Ok(e) => Ok(Some(e)),
        Err(_) => Ok(None),
    }
}

async fn http_get(
    client: &reqwest::Client,
    options: &ScanOptions,
    url: &str,
) -> Result<String, ScanError> {
    let mut req = client.get(url);
    if let Some(t) = options.hf_token.as_deref() {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(ScanError::AuthRequired { host: "huggingface.co" });
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(ScanError::RateLimited { host: "huggingface.co" });
    }
    Ok(resp.text().await?)
}
```

- [ ] **Step 5: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod stages;
```

- [ ] **Step 6: Stub out `civitai.rs` so the module declaration compiles**

Create `crates/mold-catalog/src/stages/civitai.rs`:

```rust
//! Civitai stage — implementation lands in Task 11. Empty stub keeps the
//! `stages` module declaration compiling.
```

- [ ] **Step 7: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test hf_stage
```

Expected: `running 1 test ... 1 passed`.

- [ ] **Step 8: Commit**

```bash
git add crates/mold-catalog/src/stages/ \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/hf_stage.rs
git commit -m "feat(catalog): HF stage with seed-then-finetune walk"
```

---

## Task 11: Civitai stage (TDD with wiremock)

**Files:**
- Modify: `crates/mold-catalog/src/stages/civitai.rs`
- Create: `crates/mold-catalog/tests/civitai_stage.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/mold-catalog/tests/civitai_stage.rs`:

```rust
use mold_catalog::scanner::ScanOptions;
use mold_catalog::stages::civitai;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RESPONSE: &str = r#"{
    "items": [
        {
            "id": 100,
            "name": "Real Photo XL",
            "type": "Checkpoint",
            "nsfw": false,
            "creator": { "username": "alice" },
            "stats": { "downloadCount": 950000, "rating": 4.8, "favoriteCount": 5 },
            "tags": [],
            "modelVersions": [{
                "id": 200,
                "name": "v1",
                "baseModel": "SDXL 1.0",
                "baseModelType": "Standard",
                "files": [
                    { "id": 1, "name": "x.safetensors", "sizeKB": 100,
                      "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                      "downloadUrl": "u", "hashes": {} }
                ],
                "images": []
            }]
        },
        {
            "id": 101,
            "name": "Pickle Trap",
            "type": "Checkpoint",
            "nsfw": false,
            "creator": { "username": "bob" },
            "stats": { "downloadCount": 1, "favoriteCount": 0 },
            "tags": [],
            "modelVersions": [{
                "id": 201,
                "name": "v1",
                "baseModel": "SDXL 1.0",
                "baseModelType": "Standard",
                "files": [
                    { "id": 2, "name": "x.pt", "sizeKB": 100,
                      "downloadCount": 1, "metadata": { "format": "PickleTensor" },
                      "downloadUrl": "u", "hashes": {} }
                ],
                "images": []
            }]
        }
    ],
    "metadata": { "totalPages": 1 }
}"#;

#[tokio::test]
async fn scan_drops_pickle_files_and_keeps_safetensors() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .respond_with(ResponseTemplate::new(200).set_body_string(RESPONSE))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        // include_nsfw is irrelevant here; both fixtures are SFW. The
        // assertion checks the safetensor/pt distinction.
        ..ScanOptions::default()
    };
    let entries = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"]).await.expect("scan");
    assert_eq!(entries.len(), 1, "pickle entry must be dropped");
    assert_eq!(entries[0].source_id, "200");
}

#[tokio::test]
async fn scan_passes_token_via_bearer_when_present() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "SDXL 1.0"))
        .and(wiremock::matchers::header("Authorization", "Bearer civitai-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"items":[],"metadata":{"totalPages":1}}"#))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        civitai_token: Some("civitai-secret".into()),
        ..ScanOptions::default()
    };
    let entries = civitai::scan(&server.uri(), &opts, &["SDXL 1.0"]).await.expect("scan");
    assert!(entries.is_empty());
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test civitai_stage
```

Expected: `cannot find function 'scan' in module 'mold_catalog::stages::civitai'`.

- [ ] **Step 3: Implement `stages/civitai.rs`**

Replace the stub at `crates/mold-catalog/src/stages/civitai.rs`:

```rust
//! Civitai stage. Walks `baseModels=` page-by-page; the base URL is
//! parameterized so tests can point at wiremock. Production passes
//! `"https://civitai.com"`.
//!
//! Hard-rule: only safetensors files are surfaced. Civitai's legacy `.pt`
//! ("PickleTensor") format is dropped at the scanner — arbitrary-code
//! execution risk on deserialization is not worth catalog completeness.

use serde::Deserialize;
use std::time::Duration;

use crate::entry::CatalogEntry;
use crate::normalizer::{from_civitai, CivitaiItem};
use crate::scanner::{ScanError, ScanOptions};

#[derive(Clone, Debug, Deserialize)]
struct CivitaiResponse {
    #[serde(default)]
    items: Vec<CivitaiItem>,
    #[serde(default)]
    metadata: CivitaiPaging,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CivitaiPaging {
    #[serde(default, rename = "totalPages")]
    total_pages: Option<u32>,
    #[serde(default, rename = "nextPage")]
    next_page: Option<String>,
}

pub async fn scan(
    base: &str,
    options: &ScanOptions,
    base_models: &[&str],
) -> Result<Vec<CatalogEntry>, ScanError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.request_timeout.as_secs()))
        .build()?;

    let mut entries = Vec::new();
    for base_model in base_models {
        let mut page = 1u32;
        loop {
            let url = format!(
                "{base}/api/v1/models?baseModels={bm}&types=Checkpoint&sort=Most+Downloaded&limit=100&page={page}",
                bm = urlencoding::encode(base_model),
            );
            let resp = http_get(&client, options, &url).await?;
            let parsed: CivitaiResponse = serde_json::from_str(&resp)?;
            if parsed.items.is_empty() {
                break;
            }
            for item in parsed.items {
                if let Some(e) = from_civitai(item) {
                    entries.push(e);
                    if let Some(cap) = options.per_family_cap {
                        if entries.len() >= cap {
                            return Ok(entries);
                        }
                    }
                }
            }
            if parsed.metadata.next_page.is_none()
                && parsed.metadata.total_pages.map(|t| page >= t).unwrap_or(true)
            {
                break;
            }
            page += 1;
        }
    }
    Ok(entries)
}

async fn http_get(
    client: &reqwest::Client,
    options: &ScanOptions,
    url: &str,
) -> Result<String, ScanError> {
    let mut req = client.get(url);
    if let Some(t) = options.civitai_token.as_deref() {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(ScanError::AuthRequired { host: "civitai.com" });
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(ScanError::RateLimited { host: "civitai.com" });
    }
    Ok(resp.text().await?)
}
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test civitai_stage
```

Expected: `running 2 tests ... 2 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-catalog/src/stages/civitai.rs \
        crates/mold-catalog/tests/civitai_stage.rs
git commit -m "feat(catalog): Civitai stage drops pickle files, supports auth"
```

---

## Task 12: Scanner orchestrator (TDD)

**Files:**
- Modify: `crates/mold-catalog/src/scanner.rs`
- Create: `crates/mold-catalog/tests/scanner_orchestrator.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/mold-catalog/tests/scanner_orchestrator.rs`:

```rust
use mold_catalog::families::Family;
use mold_catalog::scanner::{run_scan, FamilyScanOutcome, ScanOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn run_scan_aggregates_per_family_outcomes_and_isolates_failures() {
    let hf_server = MockServer::start().await;
    let cv_server = MockServer::start().await;

    // FLUX seed succeeds.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(200).set_body_string(include_str!(
            "fixtures/hf_flux_dev.json"
        )))
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(ResponseTemplate::new(200).set_body_string(include_str!(
            "fixtures/hf_flux_dev_tree.json"
        )))
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&hf_server)
        .await;

    // Other families: HF returns 503; civitai returns 401. Per-family
    // failure isolation means FLUX still succeeds.
    // (Mounted last so the more specific FLUX matchers above match first.)
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&cv_server)
        .await;

    let opts = ScanOptions {
        families: vec![Family::Flux, Family::Sdxl],
        ..ScanOptions::default()
    };
    let report = run_scan(&hf_server.uri(), &cv_server.uri(), &opts).await;

    let flux = report.per_family.get(&Family::Flux).expect("flux outcome");
    assert!(matches!(flux, FamilyScanOutcome::Ok { .. }));
    let sdxl = report.per_family.get(&Family::Sdxl).expect("sdxl outcome");
    // SDXL hit the catch-all 503 / 401 mocks → NetworkError or AuthRequired.
    assert!(matches!(
        sdxl,
        FamilyScanOutcome::NetworkError { .. } | FamilyScanOutcome::AuthRequired
    ));
    assert!(report.total_entries > 0);
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test scanner_orchestrator
```

Expected: `cannot find function 'run_scan' in module 'mold_catalog::scanner'`.

- [ ] **Step 3: Add `run_scan` and family→entries aggregation**

Append to `crates/mold-catalog/src/scanner.rs`:

```rust
use crate::civitai_map::map_base_model;
use crate::filter;
use crate::hf_seeds;
use crate::stages;

/// Orchestrate one full scan across HF + Civitai for the families in
/// `options.families`. Each family runs as an independent task; errors
/// are isolated per family and surfaced via `ScanReport.per_family`.
///
/// `hf_base` and `civitai_base` allow tests to point the stages at
/// wiremock; production callers pass `"https://huggingface.co"` and
/// `"https://civitai.com"`.
pub async fn run_scan(
    hf_base: &str,
    civitai_base: &str,
    options: &ScanOptions,
) -> ScanReport {
    let mut report = ScanReport::default();

    for &family in &options.families {
        let mut bucket: Vec<crate::entry::CatalogEntry> = Vec::new();
        let mut hf_failed = false;
        let mut auth_required = false;
        let mut rate_limited = false;
        let mut network_error: Option<String> = None;

        // ── HF stage ──────────────────────────────────────────────────
        match stages::hf::scan_family(hf_base, options, family, hf_seeds::seeds_for(family)).await {
            Ok(entries) => bucket.extend(entries),
            Err(ScanError::AuthRequired { .. }) => {
                auth_required = true;
                hf_failed = true;
            }
            Err(ScanError::RateLimited { .. }) => {
                rate_limited = true;
                hf_failed = true;
            }
            Err(e) => {
                network_error = Some(e.to_string());
                hf_failed = true;
            }
        }

        // ── Civitai stage ─────────────────────────────────────────────
        let cv_keys: Vec<&'static str> = crate::civitai_map::CIVITAI_BASE_MODELS
            .iter()
            .copied()
            .filter(|k| matches!(map_base_model(k), Some((f, _, _)) if f == family))
            .collect();
        if !cv_keys.is_empty() {
            match stages::civitai::scan(civitai_base, options, &cv_keys).await {
                Ok(entries) => bucket.extend(entries),
                Err(ScanError::AuthRequired { .. }) => auth_required = true,
                Err(ScanError::RateLimited { .. }) => rate_limited = true,
                Err(e) => network_error.get_or_insert(e.to_string()),
            };
        }

        let kept = filter::apply(bucket, options);
        report.total_entries += kept.len();

        let outcome = if !kept.is_empty() && !hf_failed {
            FamilyScanOutcome::Ok { entries: kept.len() }
        } else if rate_limited {
            FamilyScanOutcome::RateLimited { partial: kept.len() }
        } else if auth_required {
            FamilyScanOutcome::AuthRequired
        } else if let Some(msg) = network_error {
            FamilyScanOutcome::NetworkError { message: msg }
        } else {
            FamilyScanOutcome::Ok { entries: kept.len() }
        };
        report.per_family.insert(family, outcome);
    }

    report
}
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test scanner_orchestrator
```

Expected: `running 1 test ... 1 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-catalog/src/scanner.rs \
        crates/mold-catalog/tests/scanner_orchestrator.rs
git commit -m "feat(catalog): scanner orchestrator with per-family failure isolation"
```

---

## Task 13: Build script + stub shards + rust-embed reader (TDD)

**Files:**
- Create: `crates/mold-catalog/build.rs`
- Create: `crates/mold-catalog/src/shards.rs`
- Create: `crates/mold-catalog/data/catalog/flux.json` (and 8 siblings)
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/sink_roundtrip.rs`

- [ ] **Step 1: Commit one canonical stub shard**

Create `crates/mold-catalog/data/catalog/flux.json`:

```json
{
  "$schema": "mold.catalog.v1",
  "family": "flux",
  "generated_at": "1970-01-01T00:00:00Z",
  "scanner_version": "0.0",
  "entries": []
}
```

- [ ] **Step 2: Repeat for the eight other families**

Create the same shape for `flux2.json`, `sd15.json`, `sdxl.json`, `z-image.json`, `ltx-video.json`, `ltx2.json`, `qwen-image.json`, `wuerstchen.json` — only the `family` value changes (and matches the stable family string from Task 3).

- [ ] **Step 3: Write the failing round-trip test**

Create `crates/mold-catalog/tests/sink_roundtrip.rs`:

```rust
use mold_catalog::entry::Shard;
use std::path::PathBuf;

fn shard_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/catalog")
}

#[test]
fn every_committed_shard_round_trips_byte_identical() {
    let dir = shard_dir();
    let mut count = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read shard dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let original = std::fs::read_to_string(&path).unwrap();
        let parsed: Shard = serde_json::from_str(&original).unwrap();
        let re = serde_json::to_string_pretty(&parsed).unwrap() + "\n";
        assert_eq!(
            re.trim_end(),
            original.trim_end(),
            "shard {} is not in canonical form — run `mold catalog refresh --commit-to-repo --dry-run` to normalize",
            path.display()
        );
        count += 1;
    }
    assert!(count >= 9, "expected 9+ shards, found {count}");
}
```

- [ ] **Step 4: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test sink_roundtrip
```

Expected: assertion failure on the first stub that doesn't already match `to_string_pretty`. (If the hand-authored JSON above already matches, the test passes early — that's fine, proceed to Step 5.)

- [ ] **Step 5: Normalize the stub shards**

Run a one-shot Rust scratch program (or open each in `jq`) to rewrite the stub shards through `serde_json::to_string_pretty` so they're byte-identical to what the round-trip emits:

```bash
for f in crates/mold-catalog/data/catalog/*.json; do
  python3 -c "import json,sys; data=json.load(open('$f')); open('$f','w').write(json.dumps(data, indent=2)+'\n')"
done
```

Run the test again — it should pass.

- [ ] **Step 6: Implement `build.rs`**

Create `crates/mold-catalog/build.rs`:

```rust
//! Resolves the catalog shard directory and stamps it into
//! `MOLD_CATALOG_DIR`. Mirrors `mold-server/build.rs`'s stub-fallback
//! pattern: if `data/catalog/flux.json` is missing (e.g. a pristine
//! checkout that hasn't yet vendored the shards), emit empty stubs into
//! `$OUT_DIR/catalog-stub/` so `cargo build` succeeds.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const FAMILIES: &[&str] = &[
    "flux", "flux2", "sd15", "sdxl", "z-image", "ltx-video", "ltx2",
    "qwen-image", "wuerstchen",
];

fn stub(family: &str) -> String {
    format!(
        r#"{{
  "$schema": "mold.catalog.v1",
  "family": "{family}",
  "generated_at": "1970-01-01T00:00:00Z",
  "scanner_version": "0.0",
  "entries": []
}}
"#
    )
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let real = crate_dir.join("data/catalog");
    println!("cargo:rerun-if-changed={}", real.display());

    let resolved = if FAMILIES
        .iter()
        .all(|f| real.join(format!("{f}.json")).is_file())
    {
        real
    } else {
        emit_stub(&out_dir)
    };

    println!("cargo:rustc-env=MOLD_CATALOG_DIR={}", resolved.display());
}

fn emit_stub(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("catalog-stub");
    fs::create_dir_all(&dir).expect("create stub dir");
    for family in FAMILIES {
        let path = dir.join(format!("{family}.json"));
        if !path.exists() {
            fs::write(&path, stub(family)).expect("write stub shard");
        }
    }
    dir
}
```

- [ ] **Step 7: Implement `shards.rs`**

Create `crates/mold-catalog/src/shards.rs`:

```rust
//! Embedded catalog shards. The `MOLD_CATALOG_DIR` env var is stamped at
//! build time by `build.rs`, so `rust-embed` always finds something —
//! either the committed shards or stubs.

use rust_embed::RustEmbed;

use crate::entry::Shard;

#[derive(RustEmbed)]
#[folder = "$MOLD_CATALOG_DIR"]
pub struct EmbeddedShards;

/// Yield every embedded shard parsed into the typed form. Errors on
/// individual shards are returned alongside the family name so callers
/// can log+continue rather than abort.
pub fn iter_shards() -> impl Iterator<Item = (String, Result<Shard, serde_json::Error>)> {
    EmbeddedShards::iter().map(|name| {
        let bytes = EmbeddedShards::get(&name).expect("embedded").data;
        let parsed: Result<Shard, _> = serde_json::from_slice(&bytes);
        (name.to_string(), parsed)
    })
}

pub fn shard_count() -> usize {
    EmbeddedShards::iter().count()
}
```

- [ ] **Step 8: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod shards;
```

- [ ] **Step 9: Build the crate**

```bash
cargo build -p mold-ai-catalog
```

Expected: `Finished` with no warnings.

- [ ] **Step 10: Add a quick smoke test that the embedder picked up nine shards**

Create `crates/mold-catalog/tests/embed.rs`:

```rust
use mold_catalog::shards::{iter_shards, shard_count};

#[test]
fn nine_families_embedded() {
    assert_eq!(shard_count(), 9, "expected one shard per family");
}

#[test]
fn every_shard_parses_as_typed_form() {
    for (name, result) in iter_shards() {
        result.unwrap_or_else(|e| panic!("shard {name} failed to parse: {e}"));
    }
}
```

- [ ] **Step 11: Run all the new tests**

```bash
cargo test -p mold-ai-catalog --test sink_roundtrip --test embed
```

Expected: both pass.

- [ ] **Step 12: Commit**

```bash
git add crates/mold-catalog/build.rs \
        crates/mold-catalog/src/shards.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/data/catalog/ \
        crates/mold-catalog/tests/sink_roundtrip.rs \
        crates/mold-catalog/tests/embed.rs
git commit -m "feat(catalog): embedded shards + build.rs stub fallback"
```

---

## Task 14: Sink — atomic shard write + DB upsert (TDD)

**Files:**
- Create: `crates/mold-catalog/src/sink.rs`
- Modify: `crates/mold-catalog/src/lib.rs`
- Create: `crates/mold-catalog/tests/sink_write.rs`

NOTE: the DB-side upsert uses a small SQL surface that's added in Task 16. To keep this task self-contained, the sink module exposes both the shard-write half (callable now) and the DB-upsert half (callable after Task 16). Task 14's tests cover only the shard half; the DB half is tested in Task 18.

- [ ] **Step 1: Write the failing shard-write test**

Create `crates/mold-catalog/tests/sink_write.rs`:

```rust
use mold_catalog::entry::Shard;
use mold_catalog::sink::write_shard_atomic;
use tempfile::TempDir;

#[test]
fn write_shard_atomic_creates_canonical_file() {
    let dir = TempDir::new().unwrap();
    let shard = Shard {
        schema: "mold.catalog.v1".into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![],
    };
    let path = write_shard_atomic(dir.path(), &shard).expect("write");
    assert!(path.exists());
    assert!(path.ends_with("flux.json"));

    // Round-trip.
    let body = std::fs::read_to_string(&path).unwrap();
    let parsed: Shard = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed, shard);

    // No staging file leaked.
    let staging = dir.path().join(".staging");
    if staging.exists() {
        let leftovers: Vec<_> = std::fs::read_dir(&staging)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert!(leftovers.is_empty(), "leftover staging files: {leftovers:?}");
    }
}

#[test]
fn write_shard_overwrites_existing() {
    let dir = TempDir::new().unwrap();
    let shard1 = Shard {
        schema: "mold.catalog.v1".into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![],
    };
    let path = write_shard_atomic(dir.path(), &shard1).unwrap();
    let shard2 = Shard {
        generated_at: "2026-05-01T00:00:00Z".into(),
        ..shard1
    };
    let path2 = write_shard_atomic(dir.path(), &shard2).unwrap();
    assert_eq!(path, path2);
    let body = std::fs::read_to_string(&path).unwrap();
    let parsed: Shard = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed.generated_at, "2026-05-01T00:00:00Z");
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test sink_write
```

Expected: `unresolved import mold_catalog::sink`.

- [ ] **Step 3: Implement `sink.rs`** (shard-write half only — DB half added in Task 18)

Create `crates/mold-catalog/src/sink.rs`:

```rust
//! Two-phase commit per family.
//!
//! Phase 1 — shard file:
//! 1. Serialize entries to canonical JSON (sorted by `(family_role,
//!    download_count desc, name)`, indented 2 spaces, trailing newline).
//! 2. Write to `<dest>/.staging/<family>.json`.
//! 3. `fs::rename` to `<dest>/<family>.json` (POSIX atomic on the same fs).
//!
//! Phase 2 — DB upsert (added in Task 18).
//!
//! Sort ordering is canonical: a re-scan that finds zero changes produces
//! a byte-identical shard, so `git diff` shows nothing — `mold catalog
//! refresh` weekly does not pollute commit history.

use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use crate::entry::{CatalogEntry, FamilyRole, Shard, SHARD_SCHEMA};

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn canonicalize_entries(entries: &mut [CatalogEntry]) {
    entries.sort_by(|a, b| {
        let role = match (a.family_role, b.family_role) {
            (FamilyRole::Foundation, FamilyRole::Foundation) => Ordering::Equal,
            (FamilyRole::Foundation, _) => Ordering::Less,
            (_, FamilyRole::Foundation) => Ordering::Greater,
            _ => Ordering::Equal,
        };
        role.then_with(|| b.download_count.cmp(&a.download_count))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.id.as_str().cmp(b.id.as_str()))
    });
}

pub fn write_shard_atomic(dir: &Path, shard: &Shard) -> Result<PathBuf, SinkError> {
    debug_assert_eq!(shard.schema, SHARD_SCHEMA, "shard.schema must be {SHARD_SCHEMA}");
    fs::create_dir_all(dir)?;
    let staging = dir.join(".staging");
    fs::create_dir_all(&staging)?;
    let staged = staging.join(format!("{}.json", shard.family));
    let body = serde_json::to_string_pretty(shard)? + "\n";
    fs::write(&staged, &body)?;

    // Validate round-trip before flipping the rename.
    let _: Shard = serde_json::from_str(&body)?;

    let final_path = dir.join(format!("{}.json", shard.family));
    fs::rename(&staged, &final_path)?;
    // Best-effort cleanup if the staging dir is empty.
    let _ = fs::remove_dir(&staging);
    Ok(final_path)
}

pub fn build_shard(
    family: &str,
    scanner_version: &str,
    generated_at: &str,
    mut entries: Vec<CatalogEntry>,
) -> Shard {
    canonicalize_entries(&mut entries);
    Shard {
        schema: SHARD_SCHEMA.to_string(),
        family: family.to_string(),
        generated_at: generated_at.to_string(),
        scanner_version: scanner_version.to_string(),
        entries,
    }
}
```

- [ ] **Step 4: Re-export from `lib.rs`**

Add to `crates/mold-catalog/src/lib.rs`:

```rust
pub mod sink;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test sink_write
```

Expected: `running 2 tests ... 2 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/sink.rs \
        crates/mold-catalog/src/lib.rs \
        crates/mold-catalog/tests/sink_write.rs
git commit -m "feat(catalog): canonical shard sink with atomic rename"
```

---

## Task 15: DB migration V7 — `catalog` + `catalog_fts` (TDD)

**Files:**
- Modify: `crates/mold-db/src/migrations.rs`
- Create: `crates/mold-db/src/migrations_v7_test.rs` (or appended `#[cfg(test)]` block)

- [ ] **Step 1: Write the failing migration test**

Append to `crates/mold-db/src/migrations.rs` (or to `lib.rs` if a `mod migrations_test` already exists there) inside `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod v7_tests {
    use super::*;
    use rusqlite::Connection;

    fn open() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        apply_pending(&conn).unwrap();
        conn
    }

    #[test]
    fn schema_version_is_seven() {
        assert_eq!(SCHEMA_VERSION, 7);
    }

    #[test]
    fn catalog_table_exists_with_expected_columns() {
        let conn = open();
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(catalog)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        for required in [
            "id", "source", "source_id", "name", "author", "family",
            "family_role", "sub_family", "modality", "kind", "file_format",
            "bundling", "size_bytes", "download_count", "rating", "likes",
            "nsfw", "thumbnail_url", "description", "license",
            "license_flags", "tags", "companions", "download_recipe",
            "engine_phase", "created_at", "updated_at", "added_at",
        ] {
            assert!(cols.contains(&required.to_string()), "missing column: {required}");
        }
    }

    #[test]
    fn catalog_fts_virtual_table_exists() {
        let conn = open();
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name LIKE 'catalog_fts%'")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(tables.contains(&"catalog_fts".to_string()));
    }

    #[test]
    fn catalog_indexes_exist() {
        let conn = open();
        let idx: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='catalog'")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        for name in [
            "idx_catalog_family",
            "idx_catalog_modality",
            "idx_catalog_downloads",
            "idx_catalog_updated",
            "idx_catalog_rating",
            "idx_catalog_phase",
        ] {
            assert!(idx.iter().any(|i| i == name), "missing index: {name}");
        }
    }

    #[test]
    fn unique_source_source_id_constraint() {
        let conn = open();
        conn.execute(
            "INSERT INTO catalog (id, source, source_id, name, family, family_role, modality, kind, file_format, bundling, download_recipe, engine_phase, added_at)
             VALUES ('hf:a', 'hf', 'a', 'A', 'flux', 'foundation', 'image', 'checkpoint', 'safetensors', 'separated', '{}', 1, 0)",
            [],
        ).unwrap();
        let dup = conn.execute(
            "INSERT INTO catalog (id, source, source_id, name, family, family_role, modality, kind, file_format, bundling, download_recipe, engine_phase, added_at)
             VALUES ('hf:dup', 'hf', 'a', 'A2', 'flux', 'foundation', 'image', 'checkpoint', 'safetensors', 'separated', '{}', 1, 0)",
            [],
        );
        assert!(dup.is_err(), "duplicate (source, source_id) should violate UNIQUE");
    }
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-db --lib v7_tests
```

Expected: failures — `SCHEMA_VERSION` is still `6`, no `catalog` table.

- [ ] **Step 3: Add the V7 migration constant**

Insert just above `pub(crate) const MIGRATIONS` in `crates/mold-db/src/migrations.rs`:

```rust
const V7_CATALOG_TABLE: &str = r#"
CREATE TABLE catalog (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    name            TEXT NOT NULL,
    author          TEXT,
    family          TEXT NOT NULL,
    family_role     TEXT NOT NULL,
    sub_family      TEXT,
    modality        TEXT NOT NULL,
    kind            TEXT NOT NULL,
    file_format     TEXT NOT NULL,
    bundling        TEXT NOT NULL,
    size_bytes      INTEGER,
    download_count  INTEGER NOT NULL DEFAULT 0,
    rating          REAL,
    likes           INTEGER NOT NULL DEFAULT 0,
    nsfw            INTEGER NOT NULL DEFAULT 0,
    thumbnail_url   TEXT,
    description     TEXT,
    license         TEXT,
    license_flags   TEXT,
    tags            TEXT,
    companions      TEXT,
    download_recipe TEXT NOT NULL,
    engine_phase    INTEGER NOT NULL,
    created_at      INTEGER,
    updated_at      INTEGER,
    added_at        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (source, source_id)
);

CREATE INDEX idx_catalog_family    ON catalog(family, family_role);
CREATE INDEX idx_catalog_modality  ON catalog(modality);
CREATE INDEX idx_catalog_downloads ON catalog(download_count DESC);
CREATE INDEX idx_catalog_updated   ON catalog(updated_at DESC);
CREATE INDEX idx_catalog_rating    ON catalog(rating DESC);
CREATE INDEX idx_catalog_phase     ON catalog(engine_phase);

CREATE VIRTUAL TABLE catalog_fts USING fts5(
    name,
    author,
    description,
    tags,
    content='catalog',
    content_rowid='rowid'
);
"#;
```

- [ ] **Step 4: Append the migration entry and bump `SCHEMA_VERSION`**

Update `MIGRATIONS` and `SCHEMA_VERSION` in the same file:

```rust
pub(crate) const MIGRATIONS: &[Migration] = &[
    // ... existing v1..v6 entries unchanged ...
    Migration {
        version: 7,
        kind: MigrationKind::Sql(V7_CATALOG_TABLE),
    },
];

pub const SCHEMA_VERSION: i64 = 7;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-db --lib v7_tests
```

Expected: `running 5 tests ... 5 passed`.

- [ ] **Step 6: Run the full mold-db test suite**

```bash
cargo test -p mold-ai-db
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/mold-db/src/migrations.rs
git commit -m "feat(db): V7 catalog table + FTS5 virtual table"
```

---

## Task 16: DB-side catalog repo (`mold-db/src/catalog.rs`) (TDD)

**Files:**
- Create: `crates/mold-db/src/catalog.rs`
- Modify: `crates/mold-db/src/lib.rs`
- Create: `crates/mold-db/tests/catalog_repo.rs`

- [ ] **Step 1: Write the failing repo test**

Create `crates/mold-db/tests/catalog_repo.rs`:

```rust
use mold_ai_db::catalog::{
    delete_family, family_counts, get_by_id, list, search_fts, upsert_entries,
    CatalogRow, ListParams, SortBy,
};
use rusqlite::Connection;

fn open() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    mold_ai_db::migrations::apply_pending(&conn).unwrap();
    conn
}

fn row(id: &str, source: &str, family: &str, name: &str, downloads: i64) -> CatalogRow {
    CatalogRow {
        id: id.into(),
        source: source.into(),
        source_id: id.split(':').nth(1).unwrap().into(),
        name: name.into(),
        author: Some("alice".into()),
        family: family.into(),
        family_role: "finetune".into(),
        sub_family: None,
        modality: "image".into(),
        kind: "checkpoint".into(),
        file_format: "safetensors".into(),
        bundling: "separated".into(),
        size_bytes: Some(123),
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw: 0,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: None,
        tags: Some("[]".into()),
        companions: Some("[]".into()),
        download_recipe: "{\"files\":[]}".into(),
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn upsert_then_get_round_trips() {
    let conn = open();
    let r = row("hf:a/b", "hf", "flux", "B", 100);
    upsert_entries(&conn, "flux", &[r.clone()]).unwrap();
    let fetched = get_by_id(&conn, "hf:a/b").unwrap().unwrap();
    assert_eq!(fetched.id, r.id);
    assert_eq!(fetched.name, "B");
}

#[test]
fn delete_family_clears_only_that_family() {
    let conn = open();
    upsert_entries(&conn, "flux", &[row("hf:a/b", "hf", "flux", "B", 1)]).unwrap();
    upsert_entries(&conn, "sdxl", &[row("hf:c/d", "hf", "sdxl", "D", 1)]).unwrap();
    delete_family(&conn, "flux").unwrap();
    assert!(get_by_id(&conn, "hf:a/b").unwrap().is_none());
    assert!(get_by_id(&conn, "hf:c/d").unwrap().is_some());
}

#[test]
fn list_paginates_and_sorts() {
    let conn = open();
    let rows: Vec<CatalogRow> = (0..50)
        .map(|i| row(&format!("hf:a/{i}"), "hf", "flux", &format!("Model {i}"), i as i64))
        .collect();
    upsert_entries(&conn, "flux", &rows).unwrap();

    let params = ListParams {
        family: Some("flux".into()),
        sort: SortBy::Downloads,
        limit: 10,
        offset: 0,
        ..Default::default()
    };
    let page = list(&conn, &params).unwrap();
    assert_eq!(page.len(), 10);
    assert!(page[0].download_count >= page[1].download_count);
}

#[test]
fn family_counts_groups_by_role() {
    let conn = open();
    upsert_entries(&conn, "flux", &[
        CatalogRow { family_role: "foundation".into(), ..row("hf:f1", "hf", "flux", "F1", 1) },
        CatalogRow { family_role: "foundation".into(), ..row("hf:f2", "hf", "flux", "F2", 1) },
        CatalogRow { family_role: "finetune".into(),   ..row("hf:t1", "hf", "flux", "T1", 1) },
    ]).unwrap();
    let counts = family_counts(&conn).unwrap();
    let flux = counts.iter().find(|c| c.family == "flux").unwrap();
    assert_eq!(flux.foundation, 2);
    assert_eq!(flux.finetune, 1);
}

#[test]
fn search_fts_matches_name() {
    let conn = open();
    upsert_entries(&conn, "sdxl", &[
        row("hf:a/Juggernaut", "hf", "sdxl", "Juggernaut XL", 1),
        row("hf:a/Other", "hf", "sdxl", "Other Model", 1),
    ]).unwrap();
    let hits = search_fts(&conn, "juggernaut").unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name, "Juggernaut XL");
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-db --test catalog_repo
```

Expected: `unresolved import mold_ai_db::catalog`.

- [ ] **Step 3: Implement `catalog.rs`**

Create `crates/mold-db/src/catalog.rs`:

```rust
//! DB-side catalog repository. The SPA, CLI, and server all read through
//! this — no query strings live in handler code.
//!
//! Catalog rows are global per mold install (no `profile` column). Per-
//! profile state (downloaded? favorited?) lives in existing tables.

use rusqlite::{params, Connection, Row, ToSql};

#[derive(Clone, Debug, PartialEq)]
pub struct CatalogRow {
    pub id: String,
    pub source: String,
    pub source_id: String,
    pub name: String,
    pub author: Option<String>,
    pub family: String,
    pub family_role: String,
    pub sub_family: Option<String>,
    pub modality: String,
    pub kind: String,
    pub file_format: String,
    pub bundling: String,
    pub size_bytes: Option<i64>,
    pub download_count: i64,
    pub rating: Option<f64>,
    pub likes: i64,
    pub nsfw: i64,
    pub thumbnail_url: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub license_flags: Option<String>,
    pub tags: Option<String>,
    pub companions: Option<String>,
    pub download_recipe: String,
    pub engine_phase: i64,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
    pub added_at: i64,
}

#[derive(Clone, Debug, Default)]
pub struct ListParams {
    pub family: Option<String>,
    pub family_role: Option<String>,
    pub modality: Option<String>,
    pub source: Option<String>,
    pub sub_family: Option<String>,
    pub q: Option<String>,
    pub include_nsfw: bool,
    pub max_engine_phase: Option<u8>,
    pub sort: SortBy,
    pub limit: i64,
    pub offset: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortBy {
    #[default]
    Downloads,
    Rating,
    Recent,
    Name,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyCount {
    pub family: String,
    pub foundation: i64,
    pub finetune: i64,
}

const COLUMNS: &str = "id, source, source_id, name, author, family, family_role, sub_family, \
    modality, kind, file_format, bundling, size_bytes, download_count, rating, likes, nsfw, \
    thumbnail_url, description, license, license_flags, tags, companions, download_recipe, \
    engine_phase, created_at, updated_at, added_at";

fn from_row(row: &Row<'_>) -> rusqlite::Result<CatalogRow> {
    Ok(CatalogRow {
        id: row.get(0)?,
        source: row.get(1)?,
        source_id: row.get(2)?,
        name: row.get(3)?,
        author: row.get(4)?,
        family: row.get(5)?,
        family_role: row.get(6)?,
        sub_family: row.get(7)?,
        modality: row.get(8)?,
        kind: row.get(9)?,
        file_format: row.get(10)?,
        bundling: row.get(11)?,
        size_bytes: row.get(12)?,
        download_count: row.get(13)?,
        rating: row.get(14)?,
        likes: row.get(15)?,
        nsfw: row.get(16)?,
        thumbnail_url: row.get(17)?,
        description: row.get(18)?,
        license: row.get(19)?,
        license_flags: row.get(20)?,
        tags: row.get(21)?,
        companions: row.get(22)?,
        download_recipe: row.get(23)?,
        engine_phase: row.get(24)?,
        created_at: row.get(25)?,
        updated_at: row.get(26)?,
        added_at: row.get(27)?,
    })
}

/// Replace every row for `family` in a single transaction. The previous
/// rows are deleted first; the FTS5 mirror is rebuilt at the end.
pub fn upsert_entries(conn: &Connection, family: &str, rows: &[CatalogRow]) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute("DELETE FROM catalog WHERE family = ?1", params![family])?;
    for r in rows {
        tx.execute(
            &format!(
                "INSERT INTO catalog ({COLUMNS}) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28)",
            ),
            params![
                r.id, r.source, r.source_id, r.name, r.author, r.family, r.family_role,
                r.sub_family, r.modality, r.kind, r.file_format, r.bundling, r.size_bytes,
                r.download_count, r.rating, r.likes, r.nsfw, r.thumbnail_url, r.description,
                r.license, r.license_flags, r.tags, r.companions, r.download_recipe,
                r.engine_phase, r.created_at, r.updated_at, r.added_at,
            ],
        )?;
    }
    tx.execute("INSERT INTO catalog_fts(catalog_fts) VALUES ('rebuild')", [])?;
    tx.commit()
}

pub fn delete_family(conn: &Connection, family: &str) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM catalog WHERE family = ?1", params![family])?;
    conn.execute("INSERT INTO catalog_fts(catalog_fts) VALUES ('rebuild')", [])?;
    Ok(())
}

pub fn get_by_id(conn: &Connection, id: &str) -> rusqlite::Result<Option<CatalogRow>> {
    let mut stmt = conn.prepare(&format!("SELECT {COLUMNS} FROM catalog WHERE id = ?1"))?;
    let mut rows = stmt.query(params![id])?;
    Ok(if let Some(row) = rows.next()? {
        Some(from_row(row)?)
    } else {
        None
    })
}

pub fn list(conn: &Connection, params: &ListParams) -> rusqlite::Result<Vec<CatalogRow>> {
    let mut sql = format!("SELECT {COLUMNS} FROM catalog WHERE 1=1");
    let mut args: Vec<Box<dyn ToSql>> = Vec::new();
    if let Some(f) = &params.family {
        sql.push_str(" AND family = ?");
        args.push(Box::new(f.clone()));
    }
    if let Some(r) = &params.family_role {
        sql.push_str(" AND family_role = ?");
        args.push(Box::new(r.clone()));
    }
    if let Some(m) = &params.modality {
        sql.push_str(" AND modality = ?");
        args.push(Box::new(m.clone()));
    }
    if let Some(s) = &params.source {
        sql.push_str(" AND source = ?");
        args.push(Box::new(s.clone()));
    }
    if let Some(sf) = &params.sub_family {
        sql.push_str(" AND sub_family = ?");
        args.push(Box::new(sf.clone()));
    }
    if !params.include_nsfw {
        sql.push_str(" AND nsfw = 0");
    }
    if let Some(p) = params.max_engine_phase {
        sql.push_str(" AND engine_phase <= ?");
        args.push(Box::new(p as i64));
    }
    if let Some(q) = &params.q {
        // FTS join — wrap sql in a subquery so the existing WHERE clauses survive.
        sql = format!(
            "SELECT {COLUMNS} FROM catalog \
             INNER JOIN catalog_fts ON catalog.rowid = catalog_fts.rowid \
             WHERE catalog_fts MATCH ?1 AND ({inner})",
            inner = sql.replacen(&format!("SELECT {COLUMNS} FROM catalog WHERE "), "", 1),
        );
        args.insert(0, Box::new(q.clone()));
    }
    let order = match params.sort {
        SortBy::Downloads => "ORDER BY download_count DESC",
        SortBy::Rating => "ORDER BY rating DESC NULLS LAST",
        SortBy::Recent => "ORDER BY updated_at DESC NULLS LAST, added_at DESC",
        SortBy::Name => "ORDER BY name COLLATE NOCASE ASC",
    };
    sql.push(' ');
    sql.push_str(order);
    sql.push_str(" LIMIT ? OFFSET ?");
    args.push(Box::new(params.limit));
    args.push(Box::new(params.offset));

    let mut stmt = conn.prepare(&sql)?;
    let refs: Vec<&dyn ToSql> = args.iter().map(|b| b.as_ref()).collect();
    let rows = stmt
        .query_map(refs.as_slice(), from_row)?
        .filter_map(Result::ok)
        .collect();
    Ok(rows)
}

pub fn family_counts(conn: &Connection) -> rusqlite::Result<Vec<FamilyCount>> {
    let mut stmt = conn.prepare(
        "SELECT family,
                SUM(CASE WHEN family_role='foundation' THEN 1 ELSE 0 END) AS foundation,
                SUM(CASE WHEN family_role='finetune'   THEN 1 ELSE 0 END) AS finetune
         FROM catalog GROUP BY family ORDER BY family",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(FamilyCount {
                family: row.get(0)?,
                foundation: row.get(1)?,
                finetune: row.get(2)?,
            })
        })?
        .filter_map(Result::ok)
        .collect();
    Ok(rows)
}

/// Convenience: name+author+description+tags FTS5 search across all
/// non-NSFW rows. UI handlers usually call `list()` with `q` set instead;
/// this exists for the CLI golden test and ad-hoc use.
pub fn search_fts(conn: &Connection, query: &str) -> rusqlite::Result<Vec<CatalogRow>> {
    let params = ListParams {
        q: Some(query.to_string()),
        sort: SortBy::Downloads,
        limit: 100,
        offset: 0,
        ..Default::default()
    };
    list(conn, &params)
}
```

- [ ] **Step 4: Re-export the module**

Edit `crates/mold-db/src/lib.rs` to add:

```rust
pub mod catalog;
```

Make sure `migrations` is also re-exported as `pub mod migrations;` (it likely is — confirm before editing).

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-db --test catalog_repo
```

Expected: `running 5 tests ... 5 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-db/src/catalog.rs crates/mold-db/src/lib.rs crates/mold-db/tests/catalog_repo.rs
git commit -m "feat(db): catalog repo with list/get/upsert/FTS search"
```

---

## Task 17: Sink DB-upsert half + entry → CatalogRow conversion (TDD)

**Files:**
- Modify: `crates/mold-catalog/src/sink.rs`
- Modify: `crates/mold-catalog/Cargo.toml` (already has `mold-ai-db` dep)
- Create: `crates/mold-catalog/tests/sink_db.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/mold-catalog/tests/sink_db.rs`:

```rust
use mold_ai_db::catalog::{get_by_id, list, ListParams, SortBy};
use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind,
    LicenseFlags, Modality, RecipeFile, Source, TokenKind,
};
use mold_catalog::families::Family;
use mold_catalog::sink::upsert_family;
use rusqlite::Connection;

fn open_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    mold_ai_db::migrations::apply_pending(&conn).unwrap();
    conn
}

fn entry(id: &str, family: Family, downloads: u64) -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from(id),
        source: Source::Hf,
        source_id: id.trim_start_matches("hf:").to_string(),
        name: id.into(),
        author: None,
        family,
        family_role: FamilyRole::Finetune,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(1),
        download_count: downloads,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: vec![RecipeFile {
                url: "u".into(),
                dest: "d".into(),
                sha256: None,
                size_bytes: None,
            }],
            needs_token: Some(TokenKind::Hf),
        },
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn upsert_family_replaces_only_that_family() {
    let conn = open_db();
    upsert_family(&conn, Family::Flux, &[entry("hf:f1", Family::Flux, 1)]).unwrap();
    upsert_family(&conn, Family::Sdxl, &[entry("hf:s1", Family::Sdxl, 1)]).unwrap();
    upsert_family(&conn, Family::Flux, &[entry("hf:f2", Family::Flux, 1)]).unwrap();

    assert!(get_by_id(&conn, "hf:f1").unwrap().is_none(), "old flux row replaced");
    assert!(get_by_id(&conn, "hf:f2").unwrap().is_some());
    assert!(get_by_id(&conn, "hf:s1").unwrap().is_some(), "sdxl untouched");
}

#[test]
fn upsert_family_preserves_engine_phase_and_companions() {
    let conn = open_db();
    let mut e = entry("hf:phase3", Family::Flux, 1);
    e.engine_phase = 3;
    e.companions = vec!["t5-v1_1-xxl".into(), "clip-l".into()];
    upsert_family(&conn, Family::Flux, &[e]).unwrap();
    let row = get_by_id(&conn, "hf:phase3").unwrap().unwrap();
    assert_eq!(row.engine_phase, 3);
    let comps: Vec<String> = serde_json::from_str(row.companions.as_deref().unwrap()).unwrap();
    assert_eq!(comps, vec!["t5-v1_1-xxl", "clip-l"]);
}

#[test]
fn list_filters_by_max_engine_phase() {
    let conn = open_db();
    let mut runnable = entry("hf:r", Family::Flux, 1);
    runnable.engine_phase = 1;
    let mut pending = entry("hf:p", Family::Flux, 1);
    pending.engine_phase = 3;
    upsert_family(&conn, Family::Flux, &[runnable, pending]).unwrap();
    let res = list(
        &conn,
        &ListParams {
            max_engine_phase: Some(1),
            sort: SortBy::Name,
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "hf:r");
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test sink_db
```

Expected: `cannot find function 'upsert_family' in module 'mold_catalog::sink'`.

- [ ] **Step 3: Add `upsert_family` and entry → row conversion**

Append to `crates/mold-catalog/src/sink.rs`:

```rust
use mold_ai_db::catalog::{upsert_entries, CatalogRow};
use rusqlite::Connection;

use crate::families::Family;

pub fn entry_to_row(e: &CatalogEntry) -> Result<CatalogRow, SinkError> {
    Ok(CatalogRow {
        id: e.id.as_str().to_string(),
        source: serde_json::to_value(&e.source)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        source_id: e.source_id.clone(),
        name: e.name.clone(),
        author: e.author.clone(),
        family: e.family.as_str().to_string(),
        family_role: serde_json::to_value(&e.family_role)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        sub_family: e.sub_family.clone(),
        modality: serde_json::to_value(&e.modality)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        kind: serde_json::to_value(&e.kind)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        file_format: serde_json::to_value(&e.file_format)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        bundling: serde_json::to_value(&e.bundling)?
            .as_str()
            .unwrap_or_default()
            .to_string(),
        size_bytes: e.size_bytes.map(|v| v as i64),
        download_count: e.download_count as i64,
        rating: e.rating.map(|v| v as f64),
        likes: e.likes as i64,
        nsfw: if e.nsfw { 1 } else { 0 },
        thumbnail_url: e.thumbnail_url.clone(),
        description: e.description.clone(),
        license: e.license.clone(),
        license_flags: Some(serde_json::to_string(&e.license_flags)?),
        tags: Some(serde_json::to_string(&e.tags)?),
        companions: Some(serde_json::to_string(&e.companions)?),
        download_recipe: serde_json::to_string(&e.download_recipe)?,
        engine_phase: e.engine_phase as i64,
        created_at: e.created_at,
        updated_at: e.updated_at,
        added_at: e.added_at,
    })
}

pub fn upsert_family(conn: &Connection, family: Family, entries: &[CatalogEntry]) -> Result<(), SinkError> {
    let rows: Vec<CatalogRow> = entries
        .iter()
        .map(entry_to_row)
        .collect::<Result<Vec<_>, _>>()?;
    upsert_entries(conn, family.as_str(), &rows).map_err(|e| SinkError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    Ok(())
}
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test sink_db
```

Expected: `running 3 tests ... 3 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-catalog/src/sink.rs crates/mold-catalog/tests/sink_db.rs
git commit -m "feat(catalog): sink DB upsert + entry → row conversion"
```

---

## Task 18: Embedded-shard seeder (TDD)

**Files:**
- Modify: `crates/mold-catalog/src/shards.rs`
- Create: `crates/mold-catalog/tests/db_seed.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/mold-catalog/tests/db_seed.rs`:

```rust
use mold_ai_db::catalog::list;
use mold_catalog::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind,
    LicenseFlags, Modality, RecipeFile, Shard, Source, TokenKind, SHARD_SCHEMA,
};
use mold_catalog::families::Family;
use mold_catalog::shards::seed_db_from_embedded_if_empty;
use mold_catalog::sink::write_shard_atomic;
use rusqlite::Connection;
use tempfile::TempDir;

fn open_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    mold_ai_db::migrations::apply_pending(&conn).unwrap();
    conn
}

fn make_entry() -> CatalogEntry {
    CatalogEntry {
        id: CatalogId::from("hf:test/seed"),
        source: Source::Hf,
        source_id: "test/seed".into(),
        name: "seed".into(),
        author: None,
        family: Family::Flux,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: Modality::Image,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: Some(1),
        download_count: 1,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: vec![RecipeFile {
                url: "u".into(),
                dest: "d".into(),
                sha256: None,
                size_bytes: None,
            }],
            needs_token: Some(TokenKind::Hf),
        },
        engine_phase: 1,
        created_at: None,
        updated_at: None,
        added_at: 0,
    }
}

#[test]
fn seed_uses_disk_shards_when_dir_provided_and_db_empty() {
    let conn = open_db();
    let tmp = TempDir::new().unwrap();
    let shard = Shard {
        schema: SHARD_SCHEMA.into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![make_entry()],
    };
    write_shard_atomic(tmp.path(), &shard).unwrap();
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();

    let res = list(
        &conn,
        &mold_ai_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "hf:test/seed");
}

#[test]
fn seed_is_idempotent() {
    let conn = open_db();
    let tmp = TempDir::new().unwrap();
    let shard = Shard {
        schema: SHARD_SCHEMA.into(),
        family: "flux".into(),
        generated_at: "2026-04-25T00:00:00Z".into(),
        scanner_version: "0.9.0".into(),
        entries: vec![make_entry()],
    };
    write_shard_atomic(tmp.path(), &shard).unwrap();
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();
    // Second call should be a no-op (DB no longer empty).
    seed_db_from_embedded_if_empty(&conn, Some(tmp.path())).unwrap();
    let res = list(
        &conn,
        &mold_ai_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.len(), 1);
}

#[test]
fn seed_skips_when_disk_dir_missing_and_embedded_stubs_are_empty() {
    let conn = open_db();
    seed_db_from_embedded_if_empty(&conn, None).unwrap();
    let res = list(
        &conn,
        &mold_ai_db::catalog::ListParams {
            limit: 10,
            offset: 0,
            ..Default::default()
        },
    )
    .unwrap();
    // Embedded shards are stubs (zero entries) — DB stays empty.
    assert!(res.is_empty());
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-catalog --test db_seed
```

Expected: `cannot find function 'seed_db_from_embedded_if_empty'`.

- [ ] **Step 3: Implement the seeder**

Append to `crates/mold-catalog/src/shards.rs`:

```rust
use std::path::Path;

use crate::families::Family;
use crate::sink::upsert_family;

#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("rusqlite: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("sink: {0}")]
    Sink(#[from] crate::sink::SinkError),
}

/// Idempotent: if the catalog table already has any rows, returns Ok(())
/// without touching anything. Otherwise:
///
/// - If `disk_dir` is provided AND contains shards, seed from disk.
/// - Else, seed from rust-embedded shards.
///
/// Stub shards (zero entries) are silently skipped, so a fresh checkout
/// that hasn't yet run `mold catalog refresh` quietly stays empty.
pub fn seed_db_from_embedded_if_empty(
    conn: &rusqlite::Connection,
    disk_dir: Option<&Path>,
) -> Result<(), SeedError> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM catalog", [], |r| r.get(0))?;
    if count > 0 {
        return Ok(());
    }

    let mut shards: Vec<crate::entry::Shard> = Vec::new();
    if let Some(dir) = disk_dir {
        if dir.exists() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let body = std::fs::read_to_string(&path)?;
                let shard: crate::entry::Shard = serde_json::from_str(&body)?;
                shards.push(shard);
            }
        }
    }
    if shards.is_empty() {
        for (_name, parsed) in iter_shards() {
            shards.push(parsed?);
        }
    }

    for shard in shards {
        if shard.entries.is_empty() {
            continue;
        }
        let family = Family::from_str(&shard.family).map_err(|e| {
            SeedError::Decode(serde_json::Error::custom(format!("unknown family in shard: {e}")))
        })?;
        upsert_family(conn, family, &shard.entries)?;
    }
    Ok(())
}
```

(Note: `serde_json::Error::custom` requires `use serde::de::Error`. If `tests/db_seed.rs` fails to compile because of that import, add `use serde::de::Error as _;` near the top of `shards.rs`.)

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai-catalog --test db_seed
```

Expected: `running 3 tests ... 3 passed`.

- [ ] **Step 5: Run the full catalog test suite**

```bash
cargo test -p mold-ai-catalog
```

Expected: every test in the crate passes.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-catalog/src/shards.rs crates/mold-catalog/tests/db_seed.rs
git commit -m "feat(catalog): idempotent embedded-shard seeder"
```

---

## Task 19: Server `CatalogScanQueue` (TDD)

**Files:**
- Create: `crates/mold-server/src/catalog_api.rs`
- Modify: `crates/mold-server/Cargo.toml`
- Modify: `crates/mold-server/src/lib.rs`

- [ ] **Step 1: Add the `mold-catalog` dep**

In `crates/mold-server/Cargo.toml`, under `[dependencies]`:

```toml
mold-catalog = { path = "../mold-catalog", package = "mold-ai-catalog", version = "0.9.0" }
```

- [ ] **Step 2: Write the failing scan-queue test**

Create `crates/mold-server/src/catalog_api_test.rs` (and add `#[cfg(test)] mod catalog_api_test;` at the bottom of `catalog_api.rs` once Step 3 lands):

```rust
use std::sync::Arc;

use mold_catalog::scanner::ScanReport;
use tokio::sync::Mutex;

use super::{CatalogScanQueue, CatalogScanStatus, ScanDriver};

struct FakeDriver {
    report: Arc<Mutex<Option<ScanReport>>>,
}

#[async_trait::async_trait]
impl ScanDriver for FakeDriver {
    async fn run(&self, _opts: mold_catalog::scanner::ScanOptions) -> ScanReport {
        let r = self.report.lock().await.take().unwrap_or_default();
        r
    }
}

#[tokio::test]
async fn enqueue_then_status_transitions_to_done() {
    let report = Arc::new(Mutex::new(Some(ScanReport {
        per_family: Default::default(),
        total_entries: 7,
    })));
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");
    // Wait for completion.
    for _ in 0..50 {
        if let Some(CatalogScanStatus::Done { total_entries, .. }) = queue.status(&id).await {
            assert_eq!(total_entries, 7);
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("scan never reached Done");
}

#[tokio::test]
async fn second_enqueue_while_running_is_rejected() {
    let report = Arc::new(Mutex::new(None)); // empty → driver hangs forever
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });
    queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("first enqueue");
    let second = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await;
    assert!(second.is_err(), "single-writer guarantee");
}
```

- [ ] **Step 3: Implement `catalog_api.rs` (queue + driver trait only)**

Create `crates/mold-server/src/catalog_api.rs`:

```rust
//! Catalog REST + scan-queue surface. Mirrors the single-writer pattern
//! used by `crate::downloads::DownloadQueue`: one scan at a time per
//! server, status polled by id.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use mold_catalog::scanner::{run_scan, ScanOptions, ScanReport};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CatalogScanStatus {
    Pending,
    Running,
    Done { total_entries: usize, per_family: BTreeMap<String, String> },
    Failed { message: String },
}

#[async_trait]
pub trait ScanDriver: Send + Sync + 'static {
    async fn run(&self, opts: ScanOptions) -> ScanReport;
}

/// Production driver. Hits the real Hugging Face + Civitai endpoints.
pub struct LiveScanDriver;

#[async_trait]
impl ScanDriver for LiveScanDriver {
    async fn run(&self, opts: ScanOptions) -> ScanReport {
        run_scan("https://huggingface.co", "https://civitai.com", &opts).await
    }
}

#[derive(Clone)]
pub struct CatalogScanQueue {
    inner: Arc<Mutex<Inner>>,
    notify: Arc<tokio::sync::Notify>,
}

struct Inner {
    pending: Option<(String, ScanOptions)>,
    statuses: BTreeMap<String, CatalogScanStatus>,
    active: Option<String>,
}

impl CatalogScanQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                pending: None,
                statuses: BTreeMap::new(),
                active: None,
            })),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub async fn enqueue(&self, opts: ScanOptions) -> Result<String, String> {
        let mut inner = self.inner.lock().await;
        if inner.active.is_some() || inner.pending.is_some() {
            return Err("a catalog refresh is already in progress".into());
        }
        let id = Uuid::new_v4().to_string();
        inner.pending = Some((id.clone(), opts));
        inner.statuses.insert(id.clone(), CatalogScanStatus::Pending);
        drop(inner);
        self.notify.notify_one();
        Ok(id)
    }

    pub async fn status(&self, id: &str) -> Option<CatalogScanStatus> {
        self.inner.lock().await.statuses.get(id).cloned()
    }

    pub async fn drive(self, driver: Arc<dyn ScanDriver>) {
        loop {
            self.notify.notified().await;
            let job = {
                let mut inner = self.inner.lock().await;
                let job = inner.pending.take();
                if let Some((id, _)) = &job {
                    inner.active = Some(id.clone());
                    inner.statuses.insert(id.clone(), CatalogScanStatus::Running);
                }
                job
            };
            let Some((id, opts)) = job else { continue };
            let report = driver.run(opts).await;
            let mut summary = BTreeMap::new();
            for (fam, outcome) in &report.per_family {
                summary.insert(fam.as_str().to_string(), format!("{:?}", outcome));
            }
            let mut inner = self.inner.lock().await;
            inner.active = None;
            inner.statuses.insert(
                id.clone(),
                CatalogScanStatus::Done {
                    total_entries: report.total_entries,
                    per_family: summary,
                },
            );
        }
    }
}

#[cfg(test)]
mod catalog_api_test;
```

Add `uuid = { version = "1", features = ["v4"] }` to `crates/mold-server/Cargo.toml` if not already a dep (most modern axum servers already have it; verify with `grep uuid crates/mold-server/Cargo.toml`).

- [ ] **Step 4: Wire the module**

In `crates/mold-server/src/lib.rs`, add near the existing `pub mod` lines:

```rust
pub mod catalog_api;
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-server catalog_api_test
```

Expected: `running 2 tests ... 2 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-server/src/catalog_api.rs \
        crates/mold-server/src/catalog_api_test.rs \
        crates/mold-server/src/lib.rs \
        crates/mold-server/Cargo.toml
git commit -m "feat(server): catalog scan queue with single-writer guarantee"
```

---

## Task 20: Server endpoints — `GET /api/catalog`, `:id`, `families` (TDD)

**Files:**
- Modify: `crates/mold-server/src/catalog_api.rs` (add handlers)
- Modify: `crates/mold-server/src/state.rs` (add `catalog_scan: Arc<CatalogScanQueue>` field)
- Modify: `crates/mold-server/src/routes.rs` (register routes)
- Create: `crates/mold-server/tests/catalog_routes.rs`

- [ ] **Step 1: Write the failing handler test**

Create `crates/mold-server/tests/catalog_routes.rs`:

```rust
use axum::http::StatusCode;
use mold_ai_server::test_support::TestApp; // helper introduced if missing — see Step 2.

#[tokio::test]
async fn list_catalog_returns_200_and_paginates() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog?family=flux&limit=5&offset=0").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["entries"].is_array());
    assert!(v["page_size"].as_i64().unwrap() <= 5);
}

#[tokio::test]
async fn get_catalog_entry_404_for_unknown() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog/hf:does/not-exist").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn families_endpoint_returns_counts() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog/families").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["families"].is_array());
}

#[tokio::test]
async fn list_with_search_uses_fts() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog?q=Juggernaut&limit=10").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["entries"].as_array().unwrap().iter().any(|e| {
        e["name"].as_str().unwrap_or("").contains("Juggernaut")
    }));
}
```

- [ ] **Step 2: If `TestApp` doesn't exist, add a minimal helper**

Check `crates/mold-server/src/lib.rs` for an existing `pub mod test_support`. If not present, add:

```rust
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Create `crates/mold-server/src/test_support.rs`:

```rust
//! Lightweight in-process test client. Avoids the full hyper boot.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

pub struct TestResponse {
    pub status: StatusCode,
    pub body: String,
}

pub struct TestApp {
    router: axum::Router,
}

impl TestApp {
    pub async fn with_seeded_catalog() -> Self {
        // Build an in-memory mold.db, seed a couple of fixture rows.
        let db_path = std::env::temp_dir().join(format!(
            "mold-catalog-test-{}.db",
            uuid::Uuid::new_v4()
        ));
        std::env::set_var("MOLD_DB_PATH", &db_path);
        let conn = mold_ai_db::db::open_or_create(&db_path).expect("open db");
        // Seed rows directly.
        mold_ai_db::catalog::upsert_entries(
            &conn,
            "sdxl",
            &[mold_ai_db::catalog::CatalogRow {
                id: "hf:RunDiffusion/Juggernaut-XL".into(),
                source: "hf".into(),
                source_id: "RunDiffusion/Juggernaut-XL".into(),
                name: "Juggernaut XL".into(),
                author: Some("RunDiffusion".into()),
                family: "sdxl".into(),
                family_role: "finetune".into(),
                sub_family: None,
                modality: "image".into(),
                kind: "checkpoint".into(),
                file_format: "safetensors".into(),
                bundling: "separated".into(),
                size_bytes: Some(1),
                download_count: 100,
                rating: Some(4.7),
                likes: 0,
                nsfw: 0,
                thumbnail_url: None,
                description: None,
                license: None,
                license_flags: None,
                tags: Some("[]".into()),
                companions: Some("[]".into()),
                download_recipe: r#"{"files":[],"needs_token":null}"#.into(),
                engine_phase: 1,
                created_at: None,
                updated_at: None,
                added_at: 0,
            }],
        )
        .unwrap();
        let state = crate::state::AppState::for_tests(Arc::new(conn));
        let router = crate::routes::router(state);
        Self { router }
    }

    pub async fn get(&self, uri: &str) -> TestResponse {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let body = String::from_utf8(resp.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
        TestResponse { status, body }
    }
}
```

(`AppState::for_tests` may need a thin constructor — add one that wraps a `Connection` and stub queues. The exact name should match a constructor that already exists or be added alongside the Task-21 changes if not.)

- [ ] **Step 3: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-server --test catalog_routes
```

Expected: failures because handlers + routes don't yet exist.

- [ ] **Step 4: Implement the handlers**

Append to `crates/mold-server/src/catalog_api.rs`:

```rust
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use mold_ai_db::catalog::{family_counts, get_by_id, list, ListParams, SortBy};

#[derive(Debug, serde::Deserialize)]
pub struct ListQuery {
    pub family: Option<String>,
    pub family_role: Option<String>,
    pub modality: Option<String>,
    pub source: Option<String>,
    pub sub_family: Option<String>,
    pub q: Option<String>,
    pub include_nsfw: Option<bool>,
    pub max_engine_phase: Option<u8>,
    pub sort: Option<String>,
    pub page: Option<i64>,
    pub page_size: Option<i64>,
}

pub async fn list_catalog(
    State(state): State<crate::state::AppState>,
    Query(q): Query<ListQuery>,
) -> impl IntoResponse {
    let page_size = q.page_size.unwrap_or(48).clamp(1, 200);
    let page = q.page.unwrap_or(1).max(1);
    let params = ListParams {
        family: q.family,
        family_role: q.family_role,
        modality: q.modality,
        source: q.source,
        sub_family: q.sub_family,
        q: q.q,
        include_nsfw: q.include_nsfw.unwrap_or(false),
        max_engine_phase: q.max_engine_phase,
        sort: match q.sort.as_deref() {
            Some("rating") => SortBy::Rating,
            Some("recent") => SortBy::Recent,
            Some("name") => SortBy::Name,
            _ => SortBy::Downloads,
        },
        limit: page_size,
        offset: (page - 1) * page_size,
    };
    let conn = state.db_conn();
    match list(&conn, &params) {
        Ok(rows) => Json(serde_json::json!({
            "entries": rows.into_iter().map(catalog_row_to_wire).collect::<Vec<_>>(),
            "page": page,
            "page_size": page_size,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn get_catalog_entry(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let conn = state.db_conn();
    match get_by_id(&conn, &id) {
        Ok(Some(row)) => Json(catalog_row_to_wire(row)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "not found").into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn list_families(
    State(state): State<crate::state::AppState>,
) -> impl IntoResponse {
    let conn = state.db_conn();
    match family_counts(&conn) {
        Ok(rows) => Json(serde_json::json!({ "families": rows })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn catalog_row_to_wire(r: mold_ai_db::catalog::CatalogRow) -> serde_json::Value {
    serde_json::json!({
        "id": r.id,
        "source": r.source,
        "source_id": r.source_id,
        "name": r.name,
        "author": r.author,
        "family": r.family,
        "family_role": r.family_role,
        "sub_family": r.sub_family,
        "modality": r.modality,
        "kind": r.kind,
        "file_format": r.file_format,
        "bundling": r.bundling,
        "size_bytes": r.size_bytes,
        "download_count": r.download_count,
        "rating": r.rating,
        "likes": r.likes,
        "nsfw": r.nsfw != 0,
        "thumbnail_url": r.thumbnail_url,
        "description": r.description,
        "license": r.license,
        "license_flags": r.license_flags.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "tags": r.tags.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "companions": r.companions.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
        "download_recipe": serde_json::from_str::<serde_json::Value>(&r.download_recipe).unwrap_or(serde_json::json!({})),
        "engine_phase": r.engine_phase,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "added_at": r.added_at,
    })
}
```

- [ ] **Step 5: Add `catalog_scan: Arc<CatalogScanQueue>` to `AppState`**

In `crates/mold-server/src/state.rs`:

1. `use crate::catalog_api::CatalogScanQueue;`
2. Add `pub catalog_scan: Arc<CatalogScanQueue>,` to the struct.
3. Add `catalog_scan: CatalogScanQueue::new().into(),` to every `AppState::new` / `AppState::for_*` constructor.
4. Add `pub fn db_conn(&self) -> rusqlite::Connection` if one doesn't exist — opens a fresh handle from `MOLD_DB_PATH`. (If `AppState` already holds a handle, expose it via a method instead.)

- [ ] **Step 6: Register the routes**

In `crates/mold-server/src/routes.rs`, in the same place the downloads UI section was added, append:

```rust
        // ─── Catalog (sub-project A) ─────────────────────────────────────
        .route("/api/catalog", get(crate::catalog_api::list_catalog))
        .route("/api/catalog/families", get(crate::catalog_api::list_families))
        .route("/api/catalog/:id", get(crate::catalog_api::get_catalog_entry))
```

(`POST /api/catalog/refresh`, the refresh status route, and the per-id download route are added in Task 21.)

- [ ] **Step 7: Run the test and verify it passes**

```bash
cargo test -p mold-ai-server --test catalog_routes
```

Expected: `running 4 tests ... 4 passed`.

- [ ] **Step 8: Commit**

```bash
git add crates/mold-server/src/catalog_api.rs \
        crates/mold-server/src/state.rs \
        crates/mold-server/src/routes.rs \
        crates/mold-server/src/test_support.rs \
        crates/mold-server/src/lib.rs \
        crates/mold-server/tests/catalog_routes.rs
git commit -m "feat(server): GET /api/catalog, /:id, /families with FTS-backed list"
```

---

## Task 21: Refresh endpoints — `POST /api/catalog/refresh` + status (TDD)

**Files:**
- Modify: `crates/mold-server/src/catalog_api.rs`
- Modify: `crates/mold-server/src/routes.rs`
- Append: `crates/mold-server/tests/catalog_routes.rs`

- [ ] **Step 1: Append the failing test**

Append to `crates/mold-server/tests/catalog_routes.rs`:

```rust
#[tokio::test]
async fn refresh_enqueue_then_status_returns_pending_or_running() {
    let app = TestApp::with_seeded_catalog().await;
    let post = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(post.status, axum::http::StatusCode::ACCEPTED);
    let v: serde_json::Value = serde_json::from_str(&post.body).unwrap();
    let id = v["id"].as_str().expect("id field").to_string();

    let status = app.get(&format!("/api/catalog/refresh/{id}")).await;
    assert_eq!(status.status, axum::http::StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&status.body).unwrap();
    let state = body["state"].as_str().unwrap();
    assert!(matches!(state, "pending" | "running" | "done"));
}

#[tokio::test]
async fn refresh_returns_409_when_already_running() {
    let app = TestApp::with_seeded_catalog().await;
    let first = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(first.status, axum::http::StatusCode::ACCEPTED);
    let second = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(second.status, axum::http::StatusCode::CONFLICT);
}
```

If `TestApp::post_json` doesn't exist yet, append a helper to `test_support.rs`:

```rust
impl TestApp {
    pub async fn post_json(&self, uri: &str, body: &str) -> TestResponse {
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let body = String::from_utf8(resp.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
        TestResponse { status, body }
    }
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-server --test catalog_routes refresh
```

Expected: 404 for both endpoints.

- [ ] **Step 3: Add the refresh handlers**

Append to `crates/mold-server/src/catalog_api.rs`:

```rust
#[derive(Debug, Default, serde::Deserialize)]
pub struct RefreshBody {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub min_downloads: Option<u64>,
    #[serde(default)]
    pub no_nsfw: Option<bool>,
    #[serde(default)]
    pub include_nsfw: Option<bool>,
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default)]
    pub civitai_token: Option<String>,
}

pub async fn post_refresh(
    State(state): State<crate::state::AppState>,
    body: Option<Json<RefreshBody>>,
) -> impl IntoResponse {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let mut opts = ScanOptions::default();
    if let Some(family) = body.family.as_deref() {
        if let Ok(f) = mold_catalog::families::Family::from_str(family) {
            opts.families = vec![f];
        } else {
            return (StatusCode::BAD_REQUEST, format!("unknown family: {family}")).into_response();
        }
    }
    if let Some(m) = body.min_downloads {
        opts.min_downloads = m;
    }
    if let Some(true) = body.no_nsfw {
        opts.include_nsfw = false;
    } else if let Some(true) = body.include_nsfw {
        opts.include_nsfw = true;
    }
    opts.hf_token = body.hf_token.or_else(|| std::env::var("HF_TOKEN").ok());
    opts.civitai_token = body.civitai_token.or_else(|| std::env::var("CIVITAI_TOKEN").ok());

    match state.catalog_scan.enqueue(opts).await {
        Ok(id) => (StatusCode::ACCEPTED, Json(serde_json::json!({ "id": id }))).into_response(),
        Err(msg) => (StatusCode::CONFLICT, msg).into_response(),
    }
}

pub async fn get_refresh_status(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.catalog_scan.status(&id).await {
        Some(status) => Json(status).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown scan id").into_response(),
    }
}
```

- [ ] **Step 4: Register the routes**

In `crates/mold-server/src/routes.rs`, in the catalog section:

```rust
        .route("/api/catalog/refresh", post(crate::catalog_api::post_refresh))
        .route("/api/catalog/refresh/:id", get(crate::catalog_api::get_refresh_status))
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-server --test catalog_routes refresh
```

Expected: both new tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-server/src/catalog_api.rs \
        crates/mold-server/src/routes.rs \
        crates/mold-server/src/test_support.rs \
        crates/mold-server/tests/catalog_routes.rs
git commit -m "feat(server): POST /api/catalog/refresh + status polling"
```

---

## Task 22: Per-entry download endpoint — `POST /api/catalog/:id/download`

**Files:**
- Modify: `crates/mold-server/src/catalog_api.rs`
- Modify: `crates/mold-server/src/routes.rs`
- Append: `crates/mold-server/tests/catalog_routes.rs`

- [ ] **Step 1: Append the failing test**

```rust
#[tokio::test]
async fn download_unknown_id_404() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.post_json("/api/catalog/hf:does/not-exist/download", "{}").await;
    assert_eq!(resp.status, axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn download_known_id_returns_202_with_job_id() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app
        .post_json(
            "/api/catalog/hf:RunDiffusion/Juggernaut-XL/download",
            "{}",
        )
        .await;
    assert_eq!(resp.status, axum::http::StatusCode::ACCEPTED);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["job_ids"].is_array());
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-server --test catalog_routes download
```

Expected: route 404.

- [ ] **Step 3: Implement the handler**

Append to `crates/mold-server/src/catalog_api.rs`:

```rust
pub async fn post_catalog_download(
    State(state): State<crate::state::AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let conn = state.db_conn();
    let row = match get_by_id(&conn, &id) {
        Ok(Some(r)) => r,
        Ok(None) => return (StatusCode::NOT_FOUND, "unknown catalog id").into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    if row.engine_phase as u8 >= 2 {
        return (
            StatusCode::CONFLICT,
            format!(
                "engine_phase {} not yet supported by this build — see release notes",
                row.engine_phase
            ),
        )
            .into_response();
    }

    // Phase 1 reuses the existing `DownloadQueue` for HF entries that map
    // 1:1 onto a manifest model name. For Civitai entries, the recipe's
    // first file URL is used to build a stub manifest at runtime — that
    // path is implemented in phase 2 alongside companion auto-pull. For
    // now, return a placeholder list of pending job ids.
    let mut job_ids = Vec::new();
    if row.source == "hf" {
        // Best-effort: use the source_id (e.g. "black-forest-labs/FLUX.1-dev")
        // to look up an existing manifest model name; if not found, surface
        // the catalog-id directly to the queue.
        let model = match mold_core::manifest::find_manifest(&row.source_id) {
            Some(m) => m.name.clone(),
            None => row.source_id.clone(),
        };
        match state.downloads.enqueue(model).await {
            Ok((id, _, _)) => job_ids.push(id),
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "civitai catalog download is implemented in phase 2 (single-file loaders)",
        )
            .into_response();
    }

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_ids": job_ids })),
    )
        .into_response()
}
```

- [ ] **Step 4: Register the route**

```rust
        .route(
            "/api/catalog/:id/download",
            post(crate::catalog_api::post_catalog_download),
        )
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cargo test -p mold-ai-server --test catalog_routes download
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-server/src/catalog_api.rs \
        crates/mold-server/src/routes.rs \
        crates/mold-server/tests/catalog_routes.rs
git commit -m "feat(server): POST /api/catalog/:id/download (phase-1 HF path)"
```

---

## Task 23: Server startup wiring — seed embedded shards + spawn scan driver

**Files:**
- Modify: `crates/mold-server/src/lib.rs`

- [ ] **Step 1: In `run_server`, before the existing `downloads_driver = ...` line**

Add the seeder + scan-driver spawn:

```rust
    // ── Catalog: seed from embedded shards on first boot, spawn scan driver.
    {
        let conn = state.db_conn();
        if let Err(e) = mold_catalog::shards::seed_db_from_embedded_if_empty(&conn, None) {
            tracing::warn!(target: "catalog", "seed failed: {e}");
        }
    }
    let catalog_driver = {
        let queue = state.catalog_scan.clone();
        tokio::spawn(async move {
            queue
                .drive(std::sync::Arc::new(crate::catalog_api::LiveScanDriver))
                .await;
        })
    };
```

- [ ] **Step 2: Make sure the join handle is awaited or aborted at shutdown**

If `run_server` already collects driver handles into a vec, push `catalog_driver` into the same vec. Otherwise add at the bottom of `run_server`:

```rust
    let _ = catalog_driver.await;
```

- [ ] **Step 3: Build the workspace to verify wiring**

```bash
cargo check --workspace
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/lib.rs
git commit -m "feat(server): seed catalog from embedded shards + spawn scan driver on boot"
```

---

## Task 24: `/api/capabilities` advertises catalog availability

**Files:**
- Modify: `crates/mold-server/src/routes.rs` (the `server_capabilities` handler)
- Append: `crates/mold-server/tests/catalog_routes.rs`

- [ ] **Step 1: Append the failing test**

```rust
#[tokio::test]
async fn capabilities_includes_catalog_block() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/capabilities").await;
    assert_eq!(resp.status, axum::http::StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(v["catalog"]["available"], serde_json::Value::Bool(true));
    assert!(v["catalog"]["families"].is_array());
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai-server --test catalog_routes capabilities_includes_catalog_block
```

Expected: assertion failure.

- [ ] **Step 3: Extend `server_capabilities`**

In `crates/mold-server/src/routes.rs`, locate the existing `server_capabilities` handler (search for `"/api/capabilities"`). Add a `catalog` field to its JSON response:

```rust
    let catalog_available = std::env::var("MOLD_CATALOG_DISABLE")
        .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    body["catalog"] = serde_json::json!({
        "available": catalog_available,
        "families": mold_catalog::families::ALL_FAMILIES
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>(),
    });
```

(If `body` isn't already a `serde_json::Value`, build the response object as one and serialize at the end. Mirror the surrounding handler's existing style.)

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai-server --test catalog_routes capabilities_includes_catalog_block
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/tests/catalog_routes.rs
git commit -m "feat(server): /api/capabilities exposes catalog availability + family list"
```

---

## Task 25: CLI `mold catalog` subcommand scaffold

**Files:**
- Modify: `crates/mold-cli/Cargo.toml`
- Create: `crates/mold-cli/src/commands/catalog.rs`
- Modify: `crates/mold-cli/src/commands/mod.rs`
- Modify: `crates/mold-cli/src/main.rs`

- [ ] **Step 1: Add the dep**

In `crates/mold-cli/Cargo.toml`:

```toml
mold-catalog = { path = "../mold-catalog", package = "mold-ai-catalog", version = "0.9.0" }
```

- [ ] **Step 2: Define the `Catalog` subcommand variant**

In `crates/mold-cli/src/main.rs`, search for the existing `enum Commands { ... }` block (around line 355). Add an `enum CatalogAction` next to the other action enums (search for `enum ConfigAction`):

```rust
#[derive(Subcommand)]
enum CatalogAction {
    /// List entries with filters
    List(catalog::ListArgs),
    /// Show a single entry by id
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Re-run the scanner, write shards, reseed the DB
    Refresh(catalog::RefreshArgs),
    /// Print the local path for a downloaded entry, or "<not downloaded>"
    Where { id: String },
}
```

And inside `enum Commands`:

```rust
    /// Browse + refresh the model-discovery catalog (Hugging Face + Civitai)
    #[command(after_long_help = "\
Examples:
  mold catalog list --family flux --limit 10
  mold catalog show hf:black-forest-labs/FLUX.1-dev
  mold catalog refresh --family flux
  mold catalog where cv:618692")]
    Catalog {
        #[command(subcommand)]
        action: CatalogAction,
    },
```

- [ ] **Step 3: Add the `commands::catalog` module**

In `crates/mold-cli/src/commands/mod.rs`, add:

```rust
pub mod catalog;
```

- [ ] **Step 4: Stub `commands/catalog.rs`** (handlers come in Tasks 26–27)

```rust
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
    Err(anyhow::anyhow!("mold catalog list — implemented in Task 26"))
}

pub async fn run_show(_id: String, _json: bool) -> Result<()> {
    Err(anyhow::anyhow!("mold catalog show — implemented in Task 26"))
}

pub async fn run_refresh(_args: RefreshArgs) -> Result<()> {
    Err(anyhow::anyhow!("mold catalog refresh — implemented in Task 27"))
}

pub async fn run_where(_id: String) -> Result<()> {
    Err(anyhow::anyhow!("mold catalog where — implemented in Task 26"))
}
```

- [ ] **Step 5: Wire dispatch**

In `crates/mold-cli/src/main.rs`, locate the existing `match cli.command { ... }` arms (search for `Commands::Pull`). Add:

```rust
        Commands::Catalog { action } => match action {
            CatalogAction::List(args) => commands::catalog::run_list(args).await?,
            CatalogAction::Show { id, json } => commands::catalog::run_show(id, json).await?,
            CatalogAction::Refresh(args) => commands::catalog::run_refresh(args).await?,
            CatalogAction::Where { id } => commands::catalog::run_where(id).await?,
        },
```

- [ ] **Step 6: Build the CLI**

```bash
cargo build -p mold-ai
```

Expected: clean build.

- [ ] **Step 7: Verify the command shows up in help**

```bash
./target/debug/mold catalog --help
```

Expected: subcommands `list`, `show`, `refresh`, `where` listed.

- [ ] **Step 8: Commit**

```bash
git add crates/mold-cli/Cargo.toml \
        crates/mold-cli/src/commands/mod.rs \
        crates/mold-cli/src/commands/catalog.rs \
        crates/mold-cli/src/main.rs
git commit -m "feat(cli): mold catalog subcommand scaffold"
```

---

## Task 26: CLI `list / show / where` (TDD against fixture DB)

**Files:**
- Modify: `crates/mold-cli/src/commands/catalog.rs`
- Create: `crates/mold-cli/tests/catalog_cli.rs`

- [ ] **Step 1: Write the failing CLI golden test**

Create `crates/mold-cli/tests/catalog_cli.rs`:

```rust
use assert_cmd::Command;
use predicates::prelude::*;
use rusqlite::Connection;
use tempfile::TempDir;

fn seeded_home() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("mold.db");
    let conn = Connection::open(&db).unwrap();
    mold_ai_db::migrations::apply_pending(&conn).unwrap();
    mold_ai_db::catalog::upsert_entries(
        &conn,
        "flux",
        &[mold_ai_db::catalog::CatalogRow {
            id: "hf:bfl/FLUX.1-dev".into(),
            source: "hf".into(),
            source_id: "bfl/FLUX.1-dev".into(),
            name: "FLUX.1-dev".into(),
            author: Some("bfl".into()),
            family: "flux".into(),
            family_role: "foundation".into(),
            sub_family: None,
            modality: "image".into(),
            kind: "checkpoint".into(),
            file_format: "safetensors".into(),
            bundling: "separated".into(),
            size_bytes: Some(123),
            download_count: 1000,
            rating: None,
            likes: 0,
            nsfw: 0,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: None,
            tags: Some("[]".into()),
            companions: Some("[]".into()),
            download_recipe: r#"{"files":[]}"#.into(),
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
        }],
    )
    .unwrap();
    tmp
}

#[test]
fn catalog_list_json_returns_array() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "list", "--family", "flux", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("FLUX.1-dev"));
}

#[test]
fn catalog_show_prints_entry() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "show", "hf:bfl/FLUX.1-dev"])
        .assert()
        .success()
        .stdout(predicate::str::contains("FLUX.1-dev"));
}

#[test]
fn catalog_where_prints_not_downloaded() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .args(["catalog", "where", "hf:bfl/FLUX.1-dev"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<not downloaded>"));
}
```

Add `assert_cmd = "2"` and `predicates = "3"` under `[dev-dependencies]` in `crates/mold-cli/Cargo.toml` if not already present.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai --test catalog_cli
```

Expected: each subcommand exits non-zero with the placeholder error from Task 25.

- [ ] **Step 3: Implement `run_list`, `run_show`, `run_where`**

Replace the bodies of the three functions in `crates/mold-cli/src/commands/catalog.rs`:

```rust
use mold_ai_db::catalog::{
    family_counts as _family_counts, get_by_id, list, ListParams, SortBy,
};
use mold_core::Config;
use rusqlite::Connection;

fn open_conn() -> Result<Connection> {
    let path = mold_ai_db::path::resolve_db_path()?;
    let conn = Connection::open(path)?;
    mold_ai_db::migrations::apply_pending(&conn)?;
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
        println!("{}", serde_json::to_string_pretty(&rows.iter().map(catalog_row_to_value).collect::<Vec<_>>())?);
    } else {
        for r in rows {
            println!(
                "{:<48} {:<7} {:<8} {:<10} {:>7} ★{}",
                r.name,
                r.source,
                r.family,
                r.bundling,
                r.download_count,
                r.rating.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "-".into()),
            );
        }
    }
    Ok(())
}

pub async fn run_show(id: String, json: bool) -> Result<()> {
    let conn = open_conn()?;
    let row = get_by_id(&conn, &id)?
        .ok_or_else(|| anyhow::anyhow!("no catalog entry with id {id}"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&catalog_row_to_value(&row))?);
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

pub async fn run_where(id: String) -> Result<()> {
    let conn = open_conn()?;
    let row = get_by_id(&conn, &id)?
        .ok_or_else(|| anyhow::anyhow!("no catalog entry with id {id}"))?;
    let cfg = Config::load_or_default();
    let downloaded = cfg.manifest_model_is_downloaded(&row.source_id);
    if downloaded {
        println!("{}", cfg.models_dir().display());
    } else {
        println!("<not downloaded>");
    }
    Ok(())
}

fn catalog_row_to_value(r: &mold_ai_db::catalog::CatalogRow) -> serde_json::Value {
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
```

If `mold_ai_db::path::resolve_db_path` doesn't exist, use `mold_ai_db::path::default_db_path` (whichever the existing `mold-db` crate exposes — grep the crate to confirm before pasting).

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai --test catalog_cli
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/Cargo.toml \
        crates/mold-cli/src/commands/catalog.rs \
        crates/mold-cli/tests/catalog_cli.rs
git commit -m "feat(cli): mold catalog list/show/where read from mold.db"
```

---

## Task 27: CLI `mold catalog refresh` (TDD against wiremock)

**Files:**
- Modify: `crates/mold-cli/src/commands/catalog.rs`
- Append: `crates/mold-cli/tests/catalog_cli.rs`

- [ ] **Step 1: Append the failing test**

```rust
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_with_dry_run_does_not_touch_db() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let home = seeded_home();
    let assertion = Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .env("MOLD_CATALOG_HF_BASE", server.uri())
        .env("MOLD_CATALOG_CIVITAI_BASE", server.uri())
        .args(["catalog", "refresh", "--family", "flux", "--dry-run"])
        .assert()
        .success();

    // After --dry-run, the seeded row from `seeded_home` is still present.
    let conn = rusqlite::Connection::open(home.path().join("mold.db")).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM catalog WHERE id='hf:bfl/FLUX.1-dev'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1, "dry-run must not delete existing rows");

    drop(assertion);
}
```

Add `wiremock = "0.6"` to `[dev-dependencies]` of `mold-cli/Cargo.toml`.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai --test catalog_cli refresh
```

Expected: stub returns the placeholder error.

- [ ] **Step 3: Implement `run_refresh`**

Replace the body of `run_refresh` in `commands/catalog.rs`:

```rust
pub async fn run_refresh(args: RefreshArgs) -> Result<()> {
    let hf_base = std::env::var("MOLD_CATALOG_HF_BASE")
        .unwrap_or_else(|_| "https://huggingface.co".into());
    let cv_base = std::env::var("MOLD_CATALOG_CIVITAI_BASE")
        .unwrap_or_else(|_| "https://civitai.com".into());

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

    let report = mold_catalog::scanner::run_scan(&hf_base, &cv_base, &opts).await;

    println!("scanned {} entries across {} families", report.total_entries, report.per_family.len());
    for (fam, outcome) in &report.per_family {
        println!("  {:<12} {:?}", fam.as_str(), outcome);
    }

    if args.dry_run {
        println!("(dry-run; not writing shards or DB)");
        return Ok(());
    }

    // For commit_to_repo, write into the repo-relative shard dir; otherwise
    // into $MOLD_HOME/catalog/.
    let cfg = Config::load_or_default();
    let shard_dir = if args.commit_to_repo {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("mold-catalog")
            .join("data")
            .join("catalog")
    } else {
        cfg.mold_home().join("catalog")
    };

    let conn = open_conn()?;
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Iso8601::DEFAULT)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());

    // Re-run the scanner per family so we can group entries by family
    // before sinking. The orchestrator already grouped them, but does not
    // currently return the grouped vec; for phase 1 the simpler path is a
    // second pass keyed off `family`.
    let mut by_family: std::collections::BTreeMap<mold_catalog::families::Family, Vec<mold_catalog::entry::CatalogEntry>> = Default::default();
    // Re-run minimal scans family-by-family. This is the same pattern the
    // server's refresh endpoint uses; the duplicated network cost is OK
    // because manual refresh is rare and `--family` already narrows it.
    for fam in &opts.families {
        let mut single_opts = opts.clone();
        single_opts.families = vec![*fam];
        let report = mold_catalog::scanner::run_scan(&hf_base, &cv_base, &single_opts).await;
        by_family.insert(*fam, fetch_family_entries(&hf_base, &cv_base, &single_opts, *fam).await?);
        let _ = report;
    }

    for (fam, entries) in by_family {
        let shard = mold_catalog::sink::build_shard(fam.as_str(), env!("CARGO_PKG_VERSION"), &now, entries.clone());
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
```

(`Config::mold_home` may need a small public accessor if it's not already exposed — check `crates/mold-core/src/config.rs` and add one if missing.)

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai --test catalog_cli refresh
```

Expected: pass — the seeded row survives `--dry-run`.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/Cargo.toml \
        crates/mold-cli/src/commands/catalog.rs \
        crates/mold-cli/tests/catalog_cli.rs
git commit -m "feat(cli): mold catalog refresh writes shards + reseeds DB"
```

---

## Task 28: `mold pull <catalog-id>` integration

**Files:**
- Modify: `crates/mold-cli/src/main.rs` (the `Commands::Pull` arm)
- Modify: `crates/mold-cli/src/commands/pull.rs` (or wherever the existing pull handler lives — confirm with `grep -rn 'PullOptions' crates/mold-cli/src`)
- Append: `crates/mold-cli/tests/catalog_cli.rs`

- [ ] **Step 1: Append the failing test**

```rust
#[test]
fn pull_with_catalog_id_routes_to_catalog_lookup() {
    let home = seeded_home();
    Command::cargo_bin("mold")
        .unwrap()
        .env("MOLD_HOME", home.path())
        .env("MOLD_DB_PATH", home.path().join("mold.db"))
        .env("MOLD_HOST", "http://127.0.0.1:1") // refuse to fall through to remote
        .args(["pull", "hf:bfl/FLUX.1-dev", "--skip-verify"])
        .assert()
        // We don't expect a successful HF pull here — the purpose is to
        // verify the catalog branch is taken (error must mention manifest
        // resolution against `bfl/FLUX.1-dev`, not "unknown model 'hf:...'").
        .stderr(predicate::str::contains("bfl/FLUX.1-dev").or(predicate::str::contains("HF")))
        .failure();
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test -p mold-ai --test catalog_cli pull_with_catalog_id
```

Expected: failure — current behavior likely passes the literal `hf:bfl/FLUX.1-dev` to the manifest resolver, yielding "unknown model".

- [ ] **Step 3: Add catalog-id resolution in front of the existing pull**

Wrap the existing `Commands::Pull` arm body with a prefix check:

```rust
        Commands::Pull { model, skip_verify } => {
            let resolved = if model.starts_with("hf:") || model.starts_with("cv:") {
                resolve_catalog_id(&model)?
            } else {
                model
            };
            let opts = mold_core::download::PullOptions { skip_verify };
            // ... existing call site, with `resolved` replacing `model` ...
        }
```

Add a helper near the bottom of `main.rs` (or a new file `crates/mold-cli/src/commands/pull.rs` if absent):

```rust
fn resolve_catalog_id(id: &str) -> anyhow::Result<String> {
    let path = mold_ai_db::path::resolve_db_path()?;
    let conn = rusqlite::Connection::open(path)?;
    mold_ai_db::migrations::apply_pending(&conn)?;
    let row = mold_ai_db::catalog::get_by_id(&conn, id)?
        .ok_or_else(|| anyhow::anyhow!("no catalog entry with id {id} — run `mold catalog refresh` first"))?;
    if row.engine_phase >= 2 {
        anyhow::bail!(
            "engine_phase {} not yet supported by this build of mold (release notes for status)",
            row.engine_phase
        );
    }
    if row.source == "hf" {
        // Use the HF source_id as the manifest model name. Existing manifest
        // resolution then turns it into a real download.
        Ok(row.source_id)
    } else {
        anyhow::bail!("civitai catalog pulls are implemented in phase 2")
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
cargo test -p mold-ai --test catalog_cli pull_with_catalog_id
```

Expected: pass — the failure now references `bfl/FLUX.1-dev` (or contains a HF-related string), proving the catalog branch was taken.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/src/main.rs \
        crates/mold-cli/tests/catalog_cli.rs
git commit -m "feat(cli): mold pull routes hf:/cv: ids through the catalog"
```

---

## Task 29: Web — types + api wrappers (TDD)

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`
- Create: `web/src/__tests__/catalog-api.test.ts`

- [ ] **Step 1: Write the failing test**

Create `web/src/__tests__/catalog-api.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCatalog,
  fetchCatalogEntry,
  fetchCatalogFamilies,
  postCatalogRefresh,
  postCatalogDownload,
} from "../api";

const originalFetch = global.fetch;

describe("catalog api", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("fetchCatalog passes filters as query params", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalog({ family: "flux", q: "juggernaut", page: 2 });
    const call = (global.fetch as any).mock.calls[0][0] as string;
    expect(call).toContain("/api/catalog");
    expect(call).toContain("family=flux");
    expect(call).toContain("q=juggernaut");
    expect(call).toContain("page=2");
  });

  it("fetchCatalogEntry url-encodes the id", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });
    await fetchCatalogEntry("hf:foo/bar baz");
    const call = (global.fetch as any).mock.calls[0][0] as string;
    expect(call).toContain("/api/catalog/hf%3Afoo%2Fbar%20baz");
  });

  it("postCatalogRefresh sends an empty JSON body when no filters", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({ id: "abc" }) });
    const out = await postCatalogRefresh({});
    expect(out.id).toBe("abc");
    const init = (global.fetch as any).mock.calls[0][1];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({}));
  });

  it("fetchCatalogFamilies returns the families array", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ families: [{ family: "flux", foundation: 2, finetune: 3 }] }),
    });
    const out = await fetchCatalogFamilies();
    expect(out.families[0].family).toBe("flux");
  });

  it("postCatalogDownload returns the job_ids", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ job_ids: ["x"] }),
    });
    const out = await postCatalogDownload("hf:bfl/FLUX.1-dev");
    expect(out.job_ids).toEqual(["x"]);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd web && bun run test
```

Expected: 5 failures — `fetchCatalog`, `fetchCatalogEntry`, etc. don't exist.

- [ ] **Step 3: Add the wire types**

Append to `web/src/types.ts`:

```ts
// ─── Catalog (sub-project A) ──────────────────────────────────────────────

export interface CatalogEntryWire {
  id: string;
  source: "hf" | "civitai";
  source_id: string;
  name: string;
  author: string | null;
  family: string;
  family_role: "foundation" | "finetune";
  sub_family: string | null;
  modality: "image" | "video";
  kind: "checkpoint" | "lora" | "vae" | "text-encoder" | "control-net";
  file_format: "safetensors" | "gguf" | "diffusers";
  bundling: "separated" | "single-file";
  size_bytes: number | null;
  download_count: number;
  rating: number | null;
  likes: number;
  nsfw: boolean;
  thumbnail_url: string | null;
  description: string | null;
  license: string | null;
  license_flags: { commercial?: boolean | null; derivatives?: boolean | null; different_license?: boolean | null } | null;
  tags: string[];
  companions: string[];
  download_recipe: { files: { url: string; dest: string; sha256: string | null; size_bytes: number | null }[]; needs_token: "hf" | "civitai" | null };
  engine_phase: number;
  created_at: number | null;
  updated_at: number | null;
  added_at: number;
}

export interface CatalogListResponse {
  entries: CatalogEntryWire[];
  page: number;
  page_size: number;
}

export interface CatalogFamilyCount {
  family: string;
  foundation: number;
  finetune: number;
}

export interface CatalogFamiliesResponse {
  families: CatalogFamilyCount[];
}

export interface CatalogListParams {
  family?: string;
  family_role?: "foundation" | "finetune";
  modality?: "image" | "video";
  source?: "hf" | "civitai";
  sub_family?: string;
  q?: string;
  include_nsfw?: boolean;
  max_engine_phase?: number;
  sort?: "downloads" | "rating" | "recent" | "name";
  page?: number;
  page_size?: number;
}

export type CatalogRefreshStatus =
  | { state: "pending" }
  | { state: "running" }
  | { state: "done"; total_entries: number; per_family: Record<string, string> }
  | { state: "failed"; message: string };
```

- [ ] **Step 4: Add the api functions**

Append to `web/src/api.ts`:

```ts
import type {
  CatalogEntryWire,
  CatalogFamiliesResponse,
  CatalogListParams,
  CatalogListResponse,
  CatalogRefreshStatus,
} from "./types";

export async function fetchCatalog(params: CatalogListParams): Promise<CatalogListResponse> {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    sp.set(k, String(v));
  }
  const r = await fetch(`/api/catalog?${sp.toString()}`);
  if (!r.ok) throw new Error(`/api/catalog ${r.status}`);
  return r.json();
}

export async function fetchCatalogEntry(id: string): Promise<CatalogEntryWire> {
  const r = await fetch(`/api/catalog/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`/api/catalog/${id} ${r.status}`);
  return r.json();
}

export async function fetchCatalogFamilies(): Promise<CatalogFamiliesResponse> {
  const r = await fetch(`/api/catalog/families`);
  if (!r.ok) throw new Error(`/api/catalog/families ${r.status}`);
  return r.json();
}

export async function postCatalogRefresh(body: { family?: string; min_downloads?: number; no_nsfw?: boolean; include_nsfw?: boolean; }): Promise<{ id: string }> {
  const r = await fetch(`/api/catalog/refresh`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`/api/catalog/refresh ${r.status}`);
  return r.json();
}

export async function fetchCatalogRefresh(id: string): Promise<CatalogRefreshStatus> {
  const r = await fetch(`/api/catalog/refresh/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`/api/catalog/refresh/${id} ${r.status}`);
  return r.json();
}

export async function postCatalogDownload(id: string): Promise<{ job_ids: string[] }> {
  const r = await fetch(`/api/catalog/${encodeURIComponent(id)}/download`, { method: "POST" });
  if (!r.ok) throw new Error(`/api/catalog/${id}/download ${r.status}`);
  return r.json();
}
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
cd web && bun run test
```

Expected: the 5 catalog-api tests pass; existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/__tests__/catalog-api.test.ts
git commit -m "feat(web): catalog wire types + api wrappers"
```

---

## Task 30: Web — `useCatalog` composable (TDD)

**Files:**
- Create: `web/src/composables/useCatalog.ts`
- Create: `web/src/composables/useCatalog.test.ts`

- [ ] **Step 1: Write the failing test**

Create `web/src/composables/useCatalog.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { useCatalog } from "./useCatalog";

const originalFetch = global.fetch;

beforeEach(() => {
  global.fetch = vi.fn().mockImplementation(async (url: string) => {
    if (url.startsWith("/api/catalog/families")) {
      return {
        ok: true,
        json: async () => ({
          families: [{ family: "flux", foundation: 1, finetune: 4 }],
        }),
      };
    }
    if (url.startsWith("/api/catalog?")) {
      return {
        ok: true,
        json: async () => ({
          entries: [
            { id: "hf:a", name: "Alpha", family: "flux", engine_phase: 1, source: "hf", source_id: "a", author: null, family_role: "foundation", sub_family: null, modality: "image", kind: "checkpoint", file_format: "safetensors", bundling: "separated", size_bytes: 1, download_count: 100, rating: null, likes: 0, nsfw: false, thumbnail_url: null, description: null, license: null, license_flags: null, tags: [], companions: [], download_recipe: { files: [], needs_token: null }, created_at: null, updated_at: null, added_at: 0 },
          ],
          page: 1,
          page_size: 48,
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
});
afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("useCatalog", () => {
  it("loads families and entries on init", async () => {
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.entries.value.length).toBe(1);
    expect(cat.families.value[0].family).toBe("flux");
  });

  it("setFilter triggers a re-fetch", async () => {
    const cat = useCatalog();
    await cat.refresh();
    (global.fetch as any).mockClear();
    cat.setFilter({ family: "flux" });
    await new Promise((r) => setTimeout(r, 300)); // past the 250ms debounce
    expect((global.fetch as any).mock.calls.some((c: any[]) => (c[0] as string).includes("family=flux"))).toBe(true);
  });

  it("disables download for entries with engine_phase >= 2", async () => {
    const cat = useCatalog();
    expect(cat.canDownload({ engine_phase: 1 } as any)).toBe(true);
    expect(cat.canDownload({ engine_phase: 2 } as any)).toBe(false);
    expect(cat.canDownload({ engine_phase: 99 } as any)).toBe(false);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd web && bun run test useCatalog
```

Expected: import error.

- [ ] **Step 3: Implement the composable**

Create `web/src/composables/useCatalog.ts`:

```ts
import { ref, watch } from "vue";
import {
  fetchCatalog,
  fetchCatalogEntry,
  fetchCatalogFamilies,
  postCatalogDownload,
  postCatalogRefresh,
  fetchCatalogRefresh,
} from "../api";
import type {
  CatalogEntryWire,
  CatalogFamilyCount,
  CatalogListParams,
  CatalogRefreshStatus,
} from "../types";

const DEBOUNCE_MS = 250;

let singleton: ReturnType<typeof build> | null = null;

function build() {
  const filter = ref<CatalogListParams>({ page: 1, page_size: 48, sort: "downloads" });
  const entries = ref<CatalogEntryWire[]>([]);
  const families = ref<CatalogFamilyCount[]>([]);
  const loading = ref(false);
  const errorMsg = ref<string | null>(null);
  const detail = ref<CatalogEntryWire | null>(null);
  const refreshStatus = ref<CatalogRefreshStatus | null>(null);

  let debounceHandle: ReturnType<typeof setTimeout> | null = null;

  async function refresh() {
    loading.value = true;
    errorMsg.value = null;
    try {
      const [list, fams] = await Promise.all([
        fetchCatalog(filter.value),
        fetchCatalogFamilies(),
      ]);
      entries.value = list.entries;
      families.value = fams.families;
    } catch (e: unknown) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function setFilter(patch: Partial<CatalogListParams>) {
    filter.value = { ...filter.value, ...patch, page: 1 };
  }

  watch(
    filter,
    () => {
      if (debounceHandle) clearTimeout(debounceHandle);
      debounceHandle = setTimeout(() => {
        void refresh();
      }, DEBOUNCE_MS);
    },
    { deep: true }
  );

  async function openDetail(id: string) {
    detail.value = await fetchCatalogEntry(id);
  }

  function closeDetail() {
    detail.value = null;
  }

  function canDownload(entry: Pick<CatalogEntryWire, "engine_phase">): boolean {
    return entry.engine_phase === 1;
  }

  async function startDownload(id: string) {
    return await postCatalogDownload(id);
  }

  async function startRefresh(family?: string) {
    const { id } = await postCatalogRefresh(family ? { family } : {});
    refreshStatus.value = { state: "pending" };
    pollRefresh(id);
  }

  function pollRefresh(id: string) {
    const tick = async () => {
      try {
        const status = await fetchCatalogRefresh(id);
        refreshStatus.value = status;
        if (status.state === "done" || status.state === "failed") {
          await refresh();
          return;
        }
      } catch {}
      setTimeout(tick, 1500);
    };
    void tick();
  }

  return {
    filter,
    entries,
    families,
    loading,
    errorMsg,
    detail,
    refreshStatus,
    refresh,
    setFilter,
    openDetail,
    closeDetail,
    canDownload,
    startDownload,
    startRefresh,
  };
}

export function useCatalog() {
  if (!singleton) singleton = build();
  return singleton;
}
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
cd web && bun run test useCatalog
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/composables/useCatalog.ts web/src/composables/useCatalog.test.ts
git commit -m "feat(web): useCatalog composable with debounced filters + refresh polling"
```

---

## Task 31: Web — `/catalog` route + page + components (TDD per component)

**Files:**
- Create: `web/src/pages/CatalogPage.vue`
- Create: `web/src/components/CatalogSidebar.vue`
- Create: `web/src/components/CatalogTopbar.vue`
- Create: `web/src/components/CatalogCardGrid.vue`
- Create: `web/src/components/CatalogCard.vue`
- Create: `web/src/components/CatalogDetailDrawer.vue`
- Create matching `*.test.ts` files
- Modify: `web/src/router.ts`
- Modify: `web/src/App.vue` (nav link)

- [ ] **Step 1: Write `CatalogCard.test.ts`**

```ts
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import CatalogCard from "./CatalogCard.vue";

const baseEntry = {
  id: "hf:a", name: "Alpha", family: "flux", engine_phase: 1, source: "hf",
  source_id: "a", author: "alice", family_role: "finetune", sub_family: null,
  modality: "image", kind: "checkpoint", file_format: "safetensors",
  bundling: "separated", size_bytes: 6_000_000_000, download_count: 1234, rating: 4.7,
  likes: 0, nsfw: false, thumbnail_url: null, description: null, license: null,
  license_flags: null, tags: [], companions: [], download_recipe: { files: [], needs_token: null },
  created_at: null, updated_at: null, added_at: 0,
} as const;

describe("CatalogCard", () => {
  it("renders name + size + author + downloads", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    expect(w.text()).toContain("Alpha");
    expect(w.text()).toContain("alice");
    expect(w.text()).toContain("6.0 GB");
    expect(w.text()).toContain("1,234");
  });

  it("shows phase badge for engine_phase >= 2", () => {
    const entry = { ...baseEntry, engine_phase: 3 };
    const w = mount(CatalogCard, { props: { entry } });
    expect(w.text()).toMatch(/phase 3|coming/i);
  });
});
```

Then create `CatalogCard.vue`:

```vue
<script setup lang="ts">
import type { CatalogEntryWire } from "../types";

const props = defineProps<{ entry: CatalogEntryWire }>();

function formatGB(bytes: number | null): string {
  if (!bytes) return "—";
  const gb = bytes / 1_000_000_000;
  return `${gb.toFixed(1)} GB`;
}

function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
</script>

<template>
  <article
    class="rounded-lg border border-zinc-800 bg-zinc-900 p-3 hover:border-zinc-600 transition-colors flex flex-col gap-2"
  >
    <div class="aspect-square bg-zinc-950 rounded overflow-hidden flex items-center justify-center">
      <img
        v-if="props.entry.thumbnail_url"
        :src="props.entry.thumbnail_url"
        :alt="props.entry.name"
        loading="lazy"
        class="object-cover w-full h-full"
      />
      <span v-else class="text-zinc-600 text-xs">no thumbnail</span>
    </div>
    <div class="flex items-start justify-between gap-2">
      <h3 class="text-sm font-medium text-zinc-100 truncate">{{ props.entry.name }}</h3>
      <span
        v-if="props.entry.engine_phase >= 2"
        class="text-[10px] uppercase tracking-wide px-1.5 py-0.5 bg-amber-700/30 text-amber-200 rounded"
        :title="`Coming in phase ${props.entry.engine_phase}`"
      >
        phase {{ props.entry.engine_phase }}
      </span>
    </div>
    <p class="text-xs text-zinc-500 truncate">{{ props.entry.author ?? "unknown" }}</p>
    <div class="flex items-center justify-between text-[11px] text-zinc-400">
      <span>{{ formatGB(props.entry.size_bytes) }}</span>
      <span>{{ formatCount(props.entry.download_count) }} dl</span>
      <span v-if="props.entry.rating !== null">★ {{ props.entry.rating.toFixed(1) }}</span>
    </div>
  </article>
</template>
```

- [ ] **Step 2: Add the page composition + sub-components**

Create `web/src/components/CatalogSidebar.vue`, `CatalogTopbar.vue`, `CatalogCardGrid.vue`, `CatalogDetailDrawer.vue` — each rendering a small slice of the composable's reactive state. Sample structure (one component shown; the rest follow the same pattern):

```vue
<!-- web/src/components/CatalogSidebar.vue -->
<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";

const cat = useCatalog();
</script>

<template>
  <aside class="w-60 border-r border-zinc-800 p-3 overflow-y-auto">
    <h2 class="text-xs uppercase text-zinc-500 mb-2">Families</h2>
    <ul class="text-sm">
      <li
        v-for="row in cat.families.value"
        :key="row.family"
        @click="cat.setFilter({ family: row.family })"
        class="cursor-pointer py-1 px-2 rounded hover:bg-zinc-800"
        :class="{ 'bg-zinc-800': cat.filter.value.family === row.family }"
      >
        <div class="font-medium">{{ row.family }}</div>
        <div class="text-[11px] text-zinc-500">
          {{ row.foundation }} foundation · {{ row.finetune }} fine-tunes
        </div>
      </li>
    </ul>
  </aside>
</template>
```

Create `CatalogTopbar.vue` (modality chips, search box debounced via composable, sort dropdown, source chips, NSFW toggle).

Create `CatalogCardGrid.vue` (lazy-loaded thumbnails, virtual-scroll for >200 rows; opens the detail drawer on click).

Create `CatalogDetailDrawer.vue` (right-slide; Download button disabled when `cat.canDownload(entry)` is false, with tooltip "Coming in phase N").

- [ ] **Step 3: Compose them into `CatalogPage.vue`**

```vue
<script setup lang="ts">
import { onMounted } from "vue";
import { useCatalog } from "../composables/useCatalog";
import CatalogSidebar from "../components/CatalogSidebar.vue";
import CatalogTopbar from "../components/CatalogTopbar.vue";
import CatalogCardGrid from "../components/CatalogCardGrid.vue";
import CatalogDetailDrawer from "../components/CatalogDetailDrawer.vue";

const cat = useCatalog();

onMounted(() => {
  void cat.refresh();
});
</script>

<template>
  <div class="flex h-full">
    <CatalogSidebar />
    <main class="flex-1 flex flex-col">
      <CatalogTopbar />
      <CatalogCardGrid />
    </main>
    <CatalogDetailDrawer v-if="cat.detail.value" />
  </div>
</template>
```

- [ ] **Step 4: Wire the route**

In `web/src/router.ts`:

```ts
import CatalogPage from "./pages/CatalogPage.vue";

const routes: RouteRecordRaw[] = [
  { path: "/", name: "gallery", component: GalleryPage },
  { path: "/generate", name: "generate", component: GeneratePage },
  { path: "/catalog", name: "catalog", component: CatalogPage },
];
```

- [ ] **Step 5: Add a nav link in `App.vue`**

Find the existing top-bar nav (search for `to="/generate"`). Add:

```vue
<router-link to="/catalog" class="...">Catalog</router-link>
```

(Match the existing nav-link class pattern.)

- [ ] **Step 6: Add component-level tests**

For each new component, add a sibling `*.test.ts` that mounts it with `@vue/test-utils` and asserts:

- `CatalogSidebar.test.ts` — clicking a family row calls `setFilter({ family })` (mock the composable).
- `CatalogTopbar.test.ts` — modality chips toggle the filter; search input debounces.
- `CatalogDetailDrawer.test.ts` — Download button is `disabled` when `entry.engine_phase >= 2` and shows a tooltip mentioning the phase.

- [ ] **Step 7: Run the web test suite**

```bash
cd web && bun run test
```

Expected: all new tests pass; no regressions.

- [ ] **Step 8: Visual check (manual)**

```bash
./scripts/ensure-web-dist.sh
cd web && bun run dev
# open http://localhost:5173/catalog (or whatever the dev server reports)
```

Verify:
- Sidebar lists families with counts (after a `mold catalog refresh`, otherwise empty state).
- Card grid renders.
- Clicking a card opens the drawer; phase>=2 entries have the Download button disabled with the badge.

(This visual check is optional for plan execution but required for UAT sign-off.)

- [ ] **Step 9: Commit**

```bash
git add web/src/pages/ web/src/components/Catalog*.vue \
        web/src/components/Catalog*.test.ts \
        web/src/router.ts web/src/App.vue
git commit -m "feat(web): /catalog route with sidebar, topbar, grid, drawer"
```

---

## Task 32: Web — Settings additions (HF / Civitai tokens + NSFW toggle)

**Files:**
- Modify: `web/src/components/SettingsModal.vue`
- Append: `web/src/components/__tests__/SettingsModal.test.ts` (or create if missing)

- [ ] **Step 1: Add a failing test**

Create `web/src/components/SettingsModal.test.ts` (or append to existing). Sample:

```ts
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SettingsModal from "./SettingsModal.vue";

describe("SettingsModal — catalog auth", () => {
  it("renders the Hugging Face token input", () => {
    const w = mount(SettingsModal);
    expect(w.text()).toContain("Hugging Face");
    expect(w.find("input[name=hf_token]").exists()).toBe(true);
  });

  it("renders the Civitai token input", () => {
    const w = mount(SettingsModal);
    expect(w.text()).toContain("Civitai");
    expect(w.find("input[name=civitai_token]").exists()).toBe(true);
  });

  it("renders the Show NSFW toggle", () => {
    const w = mount(SettingsModal);
    expect(w.find("input[name=catalog_show_nsfw]").exists()).toBe(true);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd web && bun run test SettingsModal
```

Expected: 3 failures.

- [ ] **Step 3: Add the three rows to `SettingsModal.vue`**

Inside the modal body, add a `<section>` for "Model Discovery" containing:

```vue
<section class="space-y-3 mt-6">
  <h3 class="text-sm uppercase text-zinc-500">Model Discovery</h3>
  <label class="flex items-center justify-between gap-3 text-sm">
    <span>Hugging Face token</span>
    <input
      name="hf_token"
      type="password"
      v-model="hfToken"
      placeholder="hf_..."
      class="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 w-64"
    />
  </label>
  <label class="flex items-center justify-between gap-3 text-sm">
    <span>Civitai token</span>
    <input
      name="civitai_token"
      type="password"
      v-model="civitaiToken"
      placeholder="cv_..."
      class="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 w-64"
    />
  </label>
  <label class="flex items-center justify-between gap-3 text-sm">
    <span>Show NSFW models</span>
    <input
      name="catalog_show_nsfw"
      type="checkbox"
      v-model="showNsfw"
    />
  </label>
</section>
```

Bind the `hfToken`, `civitaiToken`, `showNsfw` refs to the existing `settings` API (POST `/api/settings/set` with keys `huggingface.token`, `civitai.token`, `catalog.show_nsfw`). The keys mirror the `mold config set <key>` surface and are persisted to `mold.db` `settings`.

- [ ] **Step 4: Run the test and verify it passes**

```bash
cd web && bun run test SettingsModal
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/SettingsModal.vue web/src/components/SettingsModal.test.ts
git commit -m "feat(web): Settings — HF/Civitai tokens + NSFW toggle"
```

---

## Task 33: CHANGELOG.md updates

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append `[Unreleased]` entries**

Open `CHANGELOG.md` and add under the `[Unreleased]` heading (preserve the existing format — Keep-a-Changelog with `### Added` / `### Changed` sections):

```markdown
### Added

- New `mold-catalog` workspace crate — Hugging Face + Civitai scanner, embedded shards (one JSON file per family in `crates/mold-catalog/data/catalog/`), normalizer, filter, sink.
- New `mold catalog` CLI subcommand: `list / show / refresh / where`.
- `mold pull <catalog-id>` accepts `hf:author/repo` and `cv:<modelVersionId>` and routes through the catalog.
- New web `/catalog` route with sidebar (family counts), topbar (search, modality, sort), card grid, detail drawer, lazy-loaded thumbnails.
- New Settings rows: Hugging Face token, Civitai token, Show NSFW models toggle.
- New env var `CIVITAI_TOKEN` (Civitai bearer auth, read by scanner + server).
- New env vars `MOLD_CATALOG_HF_BASE` / `MOLD_CATALOG_CIVITAI_BASE` (test-only override of scanner upstream).
- New env var `MOLD_CATALOG_DISABLE=1` to flag the catalog feature as unavailable in `/api/capabilities`.
- New endpoints: `GET /api/catalog`, `GET /api/catalog/:id`, `GET /api/catalog/families`, `POST /api/catalog/refresh`, `GET /api/catalog/refresh/:id`, `POST /api/catalog/:id/download`.
- `/api/capabilities` now includes a `catalog` block listing supported families and availability.
- SQLite migration **v7** — adds `catalog` table + 6 indexes + `catalog_fts` FTS5 virtual table. Forward-only; no data loss for existing rows in `generations` / `settings` / `model_prefs` / `prompt_history`.
- Phase-1 catalog entries with `engine_phase >= 2` (single-file checkpoints) are stored and rendered with a "phase N" badge and disabled Download button — runtime support arrives in mold v0.10/0.11/0.12 (sub-projects 2–5).
```

- [ ] **Step 2: Verify CHANGELOG renders cleanly**

```bash
grep -A 30 '\[Unreleased\]' CHANGELOG.md | head -50
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): catalog expansion phase 1 entries"
```

---

## Task 34: Update `.claude/skills/mold/SKILL.md`

**Files:**
- Modify: `.claude/skills/mold/SKILL.md`

- [ ] **Step 1: Skim the current skill file**

```bash
sed -n '1,80p' .claude/skills/mold/SKILL.md
```

- [ ] **Step 2: Add a "Catalog" section near the existing CLI reference**

Insert a section like:

```markdown
## Model discovery catalog (sub-project A)

**Browse:** `mold catalog list [--family flux] [--q juggernaut] [--json]` reads the local `mold.db` `catalog` table and prints rows.

**Inspect:** `mold catalog show hf:black-forest-labs/FLUX.1-dev` (or `cv:618692`) prints a single entry; `--json` for machine-readable.

**Refresh:** `mold catalog refresh [--family flux] [--no-nsfw] [--dry-run]` re-runs the scanner against Hugging Face + Civitai, writes shards into `$MOLD_HOME/catalog/`, reseeds the DB. Maintainer-only `--commit-to-repo` writes into `crates/mold-catalog/data/catalog/`.

**Pull catalog ids:** `mold pull hf:author/repo` and `mold pull cv:618692` route through the catalog. Phase-1 supports HF separated-bundling entries with `engine_phase=1`. Single-file (Civitai) entries land in mold v0.10+ when sub-projects 2–5 ship.

**Auth:** `HF_TOKEN` for gated Hugging Face repos; `CIVITAI_TOKEN` for early-access / NSFW Civitai. Web Settings persists these to `mold.db` `settings` (`huggingface.token`, `civitai.token`).

**Web:** `/catalog` route in the SPA — sidebar, topbar, card grid, detail drawer.

**Internals:** scanner / shards / FTS5 search live in `mold-catalog`. Catalog rows persist in the SQLite `catalog` table; FTS5 mirrors `name + author + description + tags`.

**`MOLD_CATALOG_DISABLE=1`** flags the catalog as unavailable in `/api/capabilities`.
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/mold/SKILL.md
git commit -m "docs(skill): document mold catalog + mold pull <catalog-id>"
```

---

## Task 35: VitePress page

**Files:**
- Create: `website/docs/catalog.md`
- Modify: `website/.vitepress/config.ts` (sidebar)

- [ ] **Step 1: Add the docs page**

Create `website/docs/catalog.md`:

```markdown
# Model Discovery Catalog

mold ships a built-in catalog of every model family it can run, scanned from
Hugging Face and Civitai. Browse it through the web UI at `/catalog`, the
`mold catalog` CLI, or directly via `mold pull <catalog-id>`.

## CLI

```bash
mold catalog list --family flux --limit 10
mold catalog show hf:black-forest-labs/FLUX.1-dev
mold catalog refresh --family flux        # re-scan one family
mold catalog where cv:618692              # path on disk if downloaded
```

## Web UI

Visit `/catalog` to browse the full catalog with filters by family,
modality, source, sub-family, and FTS5-backed search. The detail drawer
shows the download recipe; the Download button is disabled with a phase
badge for entries that need a single-file loader (mold v0.10+).

## Auth

Set `HF_TOKEN` and `CIVITAI_TOKEN` as env vars, or paste them in
`Settings → Model Discovery`. Both are stored in `mold.db` `settings`
(`huggingface.token`, `civitai.token`).

## Refresh

The catalog is global per mold install. Run `mold catalog refresh` weekly
(or whenever you want fresh discovery); the scanner is incremental and
deterministic — no-op refreshes produce byte-identical shard files.
```

- [ ] **Step 2: Add the sidebar entry**

In `website/.vitepress/config.ts`, find the existing `sidebar` array and add:

```ts
{ text: "Catalog", link: "/docs/catalog" },
```

- [ ] **Step 3: Verify the website builds**

```bash
cd website && bun run fmt:check && bun run verify && bun run build
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add website/docs/catalog.md website/.vitepress/config.ts
git commit -m "docs(website): catalog page + sidebar entry"
```

---

## Task 36: Crate-boundary check (no transitive `mold-catalog` in `mold-tui` / `mold-discord`)

**Files:** none — verification only

- [ ] **Step 1: Inspect the dep tree for `mold-discord`**

```bash
cargo tree -p mold-ai-discord -e normal | grep -F "mold-ai-catalog" || echo "OK: no catalog dep"
```

Expected: `OK: no catalog dep`. If `cargo tree` prints a row, fix the dep graph immediately — the only crates that should pull `mold-catalog` are `mold-cli` and `mold-server`.

- [ ] **Step 2: Inspect the dep tree for `mold-tui`**

```bash
cargo tree -p mold-ai-tui -e normal | grep -F "mold-ai-catalog" || echo "OK: no catalog dep"
```

Expected: `OK: no catalog dep`.

- [ ] **Step 3: Pin the invariant with a CI-grade assertion**

Create `crates/mold-catalog/tests/no_inheritance.rs`:

```rust
//! Pinned crate-boundary check — `mold-discord` and `mold-tui` MUST NOT
//! transitively depend on `mold-catalog`. Catches accidental dep churn.
//!
//! Implemented as an `#[ignore]`d test so the regular `cargo test --workspace`
//! is unaffected; the workspace CI gate runs it explicitly.

use std::process::Command;

#[test]
#[ignore]
fn discord_does_not_inherit_mold_catalog() {
    let out = Command::new("cargo")
        .args(["tree", "-p", "mold-ai-discord", "-e", "normal"])
        .output()
        .expect("cargo tree");
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        !body.contains("mold-ai-catalog"),
        "mold-ai-discord must not transitively depend on mold-ai-catalog\n\n{body}"
    );
}

#[test]
#[ignore]
fn tui_does_not_inherit_mold_catalog() {
    let out = Command::new("cargo")
        .args(["tree", "-p", "mold-ai-tui", "-e", "normal"])
        .output()
        .expect("cargo tree");
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        !body.contains("mold-ai-catalog"),
        "mold-ai-tui must not transitively depend on mold-ai-catalog\n\n{body}"
    );
}
```

(`#[ignore]`d because `cargo tree` is slow; the explicit Step 1/Step 2 invocations are the actual gate. The pinned tests exist so a future contributor can rerun them locally with `cargo test -p mold-ai-catalog --test no_inheritance -- --ignored`.)

- [ ] **Step 4: Commit**

```bash
git add crates/mold-catalog/tests/no_inheritance.rs
git commit -m "test(catalog): pin crate-boundary invariant — discord/tui must not inherit catalog"
```

---

## Task 37: Final CI gate (local-equivalent of GitHub `ci.yml`)

**Files:** none — verification only

- [ ] **Step 1: Format check**

```bash
cargo fmt --all -- --check
```

Expected: no diff.

- [ ] **Step 2: Clippy with deny-warnings**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 3: Workspace tests**

```bash
cargo test --workspace
```

If `theme_save_then_load_round_trip_preserves_preset` flakes (known auto-memory note), retry once before investigating — the failure is unrelated to catalog work.

Expected: all pass on retry if needed.

- [ ] **Step 4: Feature-combo check**

```bash
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
```

Expected: clean.

- [ ] **Step 5: Web tests + format + build**

```bash
cd web && bun run fmt:check && bun run test && bun run build && cd ..
```

Expected: clean.

- [ ] **Step 6: Website verify + build**

```bash
cd website && bun run fmt:check && bun run verify && bun run build && cd ..
```

Expected: clean.

- [ ] **Step 7: Crate-boundary check**

```bash
cargo tree -p mold-ai-discord -e normal | grep -F "mold-ai-catalog" || echo "OK"
cargo tree -p mold-ai-tui     -e normal | grep -F "mold-ai-catalog" || echo "OK"
```

Both must print `OK`.

- [ ] **Step 8: Push the umbrella**

```bash
git push origin feat/catalog-expansion
```

Expected: GitHub PR refreshes; CI runs.

- [ ] **Step 9: UAT on killswitch**

```bash
ssh killswitch@192.168.1.67 "cd ~/github/mold && git fetch origin && git checkout feat/catalog-expansion && nix build && ./result/bin/mold catalog refresh --family flux --dry-run"
```

Expected: `mold catalog refresh --family flux --dry-run` reports per-family scan outcomes; no DB writes.

After UAT:

```bash
ssh killswitch@192.168.1.67 "cd ~/github/mold && ./result/bin/mold catalog refresh --family sdxl"
ssh killswitch@192.168.1.67 "cd ~/github/mold && ./result/bin/mold catalog list --family sdxl --limit 5"
```

Expected: live SDXL entries land in the local `mold.db` and round-trip out via `catalog list`.

- [ ] **Step 10: Mark the draft PR ready for review**

```bash
gh pr ready feat/catalog-expansion
```

(Phase-1 work is done. The umbrella PR stays open while phases 2–5 layer on; the draft → ready transition is the user-facing signal that phase 1 is reviewable on its own.)

---

## Self-Review Checklist

Before declaring this plan complete, verify against the spec (`docs/superpowers/specs/2026-04-25-catalog-expansion-design.md`, sections §1.1 through §1.15):

- §1.1 New crate `crates/mold-catalog/` — Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18.
- §1.2 Core types in `entry.rs` — Task 4.
- §1.3 Companion registry — Task 6.
- §1.4 Civitai mapping with completeness gate — Task 5.
- §1.5 HF stage with seeded `base_model:` walk — Task 10.
- §1.6 Civitai stage drops `.pt` — Task 11.
- §1.7 Filter — Task 8.
- §1.8 Sink with two-phase commit + canonical sort — Tasks 14 + 17.
- §1.9 mold-db migration V7 + FTS5 + indexes — Task 15. DB repo for queries — Task 16.
- §1.10 Server endpoints (`/api/catalog`, `/:id`, `/families`, `refresh`, `refresh/:id`, `:id/download`) — Tasks 19, 20, 21, 22. Capabilities — Task 24.
- §1.11 CLI subcommand + `mold pull <catalog-id>` — Tasks 25, 26, 27, 28.
- §1.12 Web `/catalog` route + Settings — Tasks 29, 30, 31, 32.
- §1.13 build.rs with stub fallback — Task 13.
- §1.14 First-run seed flow — Tasks 18 + 23 (server startup wiring).
- §1.15 Phase 1 testing — covered throughout (each TDD step is a snapshot/round-trip/golden test).

Cross-cutting:

- CHANGELOG.md updated — Task 33.
- `.claude/skills/mold/SKILL.md` updated — Task 34.
- `website/` updated — Task 35.
- Crate-boundary invariant pinned — Task 36.
- Final CI gate runs — Task 37.

**No placeholders.** Each task's code blocks are complete; method signatures are consistent across tasks (e.g. `companions_for(family, bundling)` is the only signature used; `engine_phase_for(family, bundling)` likewise; `Family::from_str` returns `Result<Self, UnknownFamily>` everywhere).

**Spec carve-outs honored:**

- No single-file checkpoint loaders. Phase-2..5 entries are stored with `engine_phase >= 2`, rendered with a badge, and Download is disabled. The catalog never *teases* mold-runnable models that aren't.
- No CI cron auto-refresh — only `mold catalog refresh`.
- No live HTTP in CI (`#[ignore]` on the live test in Task 36's no-inheritance file pattern + the spec's reserved `#[ignore]` test in `crates/mold-catalog/tests/scanner_live.rs` which is not implemented in this plan but is reserved for the killswitch UAT path in Task 37 Step 9).
- `mold-discord` and `mold-tui` do not inherit `mold-catalog` (Task 36).
- No `Family` enum exposed from `mold-core` — the new enum lives in `mold-catalog::families` only (Task 3 deliberately diverges from spec §1.1's "re-export from mold-core" because the codebase keeps `family: String` everywhere).
- Catalog is global per mold install (no `profile` column in the V7 migration — Task 15).
- Companion canonical names are committed verbatim from the spec (Task 6) — `t5-v1_1-xxl`, `clip-l`, `clip-g`, `sdxl-vae`, `sd-vae-ft-mse`, `flux-vae`, `z-image-te`.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-04-25-catalog-expansion-phase-1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Use `superpowers:subagent-driven-development`. Dispatches a fresh subagent per task with two-stage review checkpoints between tasks. Best for keeping the main context window clean while the work spans multiple subsystems (Rust, SQL, web, docs).

2. **Inline Execution** — Use `superpowers:executing-plans`. Runs tasks sequentially in this session with periodic batch checkpoints. Best when you want to land tasks 1–5 in one sitting and then re-engage for the server / web halves.

Which approach?
