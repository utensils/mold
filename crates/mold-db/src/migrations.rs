//! Schema migration framework for the gallery metadata DB.
//!
//! Each migration is applied in order inside its own transaction. The
//! current schema version is tracked via SQLite's built-in `PRAGMA
//! user_version` so we don't need a sidecar table.
//!
//! Migrations are one-way (no `down` step): if you need to walk back, write
//! a forward-only migration that undoes the change.
//!
//! ## Two kinds of migration
//!
//! Most migrations are pure DDL — a string of SQL statements. Some need
//! programmatic rewrites of existing rows (e.g. v2 canonicalizes
//! `output_dir` values written under raw paths by the v0.8.x release).
//! The [`MigrationKind`] enum covers both.
//!
//! ## Adding a new migration
//!
//! Append a new entry to [`MIGRATIONS`] with the next sequential version.
//! Example DDL-only migration:
//!
//! ```ignore
//! Migration {
//!     version: 3,
//!     kind: MigrationKind::Sql(r#"
//!         ALTER TABLE generations ADD COLUMN controlnet_model TEXT;
//!         ALTER TABLE generations ADD COLUMN controlnet_scale REAL;
//!     "#),
//! },
//! ```

use anyhow::{bail, Result};
use rusqlite::{Connection, Transaction};

use crate::path::canonical_dir_string;

/// What a migration does. SQL migrations are applied by
/// [`Connection::execute_batch`]; Rust migrations receive the active
/// transaction so they can both read + rewrite existing rows in place.
pub(crate) enum MigrationKind {
    Sql(&'static str),
    Rust(fn(&Transaction<'_>) -> Result<()>),
}

/// A single forward-only migration.
pub(crate) struct Migration {
    pub version: i64,
    pub kind: MigrationKind,
}

/// The initial schema — what v0.8.x shipped. Kept as a single block so a
/// fresh DB needs only one transaction to become a v1 DB.
const V1_INITIAL_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS generations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT    NOT NULL,
    output_dir         TEXT    NOT NULL,
    created_at_ms      INTEGER NOT NULL,
    file_mtime_ms      INTEGER,
    file_size_bytes    INTEGER,

    format             TEXT    NOT NULL,

    model              TEXT    NOT NULL,
    prompt             TEXT    NOT NULL DEFAULT '',
    negative_prompt    TEXT,
    original_prompt    TEXT,
    seed               INTEGER NOT NULL DEFAULT 0,
    steps              INTEGER NOT NULL DEFAULT 0,
    guidance           REAL    NOT NULL DEFAULT 0.0,
    width              INTEGER NOT NULL DEFAULT 0,
    height             INTEGER NOT NULL DEFAULT 0,
    strength           REAL,
    scheduler          TEXT,
    lora               TEXT,
    lora_scale         REAL,
    frames             INTEGER,
    fps                INTEGER,
    metadata_version   TEXT    NOT NULL DEFAULT '',

    generation_time_ms INTEGER,
    backend            TEXT,
    hostname           TEXT,
    source             TEXT    NOT NULL DEFAULT 'unknown',
    metadata_synthetic INTEGER NOT NULL DEFAULT 0,

    UNIQUE(output_dir, filename)
);

CREATE INDEX IF NOT EXISTS idx_gen_created_at ON generations(created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_gen_mtime      ON generations(file_mtime_ms DESC);
CREATE INDEX IF NOT EXISTS idx_gen_model      ON generations(model);
CREATE INDEX IF NOT EXISTS idx_gen_format     ON generations(format);
CREATE INDEX IF NOT EXISTS idx_gen_filename   ON generations(filename);
CREATE INDEX IF NOT EXISTS idx_gen_output_dir ON generations(output_dir);
"#;

/// v3 → add the global KV `settings` table. Used for TUI + user-preference
/// state that previously lived in `tui-session.json` and the user-facing
/// portions of `config.toml`.
const V3_SETTINGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS settings (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    value_type    TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"#;

/// v4 → add the per-model preferences table. One row per resolved model
/// tag; every column is nullable because a fresh install has nothing to
/// remember yet.
const V4_MODEL_PREFS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS model_prefs (
    model           TEXT PRIMARY KEY,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    guidance        REAL,
    scheduler       TEXT,
    seed_mode       TEXT,
    batch           INTEGER,
    format          TEXT,
    lora_path       TEXT,
    lora_scale      REAL,
    expand          INTEGER,
    offload         INTEGER,
    strength        REAL,
    control_scale   REAL,
    frames          INTEGER,
    fps             INTEGER,
    last_prompt     TEXT,
    last_negative   TEXT,
    updated_at_ms   INTEGER NOT NULL
);
"#;

/// v5 → prompt history. Replaces `prompt-history.jsonl`; bounded-size via
/// the caller-driven `trim_to()` API.
const V5_PROMPT_HISTORY_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS prompt_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt        TEXT NOT NULL,
    negative      TEXT,
    model         TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_hist_created
    ON prompt_history(created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_hist_model
    ON prompt_history(model);
"#;

/// v6 → add a `profile` column to `settings` and `model_prefs` so the
/// same DB can host multiple independent user preference sets (`default`,
/// `dev`, `portrait`, …). All v5 rows land under `profile = 'default'`
/// so existing installs keep working untouched.
///
/// SQLite can't change a PK in-place, so each table is recreated and the
/// data is copied. Both steps run in the same v6 transaction — a crash
/// mid-migration leaves the DB at v5.
const V6_PROFILE_SCOPING: &str = r#"
CREATE TABLE settings_v6 (
    profile       TEXT NOT NULL DEFAULT 'default',
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    value_type    TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (profile, key)
);
INSERT INTO settings_v6 (profile, key, value, value_type, updated_at_ms)
    SELECT 'default', key, value, value_type, updated_at_ms FROM settings;
DROP TABLE settings;
ALTER TABLE settings_v6 RENAME TO settings;

CREATE TABLE model_prefs_v6 (
    profile         TEXT NOT NULL DEFAULT 'default',
    model           TEXT NOT NULL,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    guidance        REAL,
    scheduler       TEXT,
    seed_mode       TEXT,
    batch           INTEGER,
    format          TEXT,
    lora_path       TEXT,
    lora_scale      REAL,
    expand          INTEGER,
    offload         INTEGER,
    strength        REAL,
    control_scale   REAL,
    frames          INTEGER,
    fps             INTEGER,
    last_prompt     TEXT,
    last_negative   TEXT,
    updated_at_ms   INTEGER NOT NULL,
    PRIMARY KEY (profile, model)
);
INSERT INTO model_prefs_v6
    SELECT 'default', model, width, height, steps, guidance, scheduler, seed_mode,
           batch, format, lora_path, lora_scale, expand, offload, strength,
           control_scale, frames, fps, last_prompt, last_negative, updated_at_ms
    FROM model_prefs;
DROP TABLE model_prefs;
ALTER TABLE model_prefs_v6 RENAME TO model_prefs;
"#;

/// v7 → add the `catalog` table for the model-catalog expansion, plus a
/// companion `catalog_fts` FTS5 virtual table for full-text search over
/// `name`, `author`, `description`, and `tags`. Six covering indexes are
/// added to support the most common browse/sort patterns.
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
    supported    INTEGER NOT NULL,
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
CREATE INDEX idx_catalog_supported ON catalog(supported);

CREATE VIRTUAL TABLE catalog_fts USING fts5(
    name,
    author,
    description,
    tags,
    content='catalog',
    content_rowid='rowid'
);
"#;

/// v8 → add `trained_words` to `catalog`. Civitai LoRA versions advertise
/// trigger phrases (`trainedWords` on `/api/v1/models`); the web UI surfaces
/// them as click-to-insert chips next to the LoRA picker. Stored as a JSON
/// array of strings to avoid a separate side table for what is typically
/// 0–8 short tokens per LoRA. Default `'[]'` keeps pre-v8 rows parseable
/// without a backfill pass.
const V8_CATALOG_TRAINED_WORDS: &str = r#"
ALTER TABLE catalog ADD COLUMN trained_words TEXT NOT NULL DEFAULT '[]';
"#;

/// v9 → drop the bulk-scrape catalog. The SPA, CLI, and server all
/// read from live HF + Civitai now (with sidecars next to each
/// installed file as the source of truth for "downloaded"). The
/// catalog DB and its FTS5 sidekick haven't been on the read path
/// for several releases; v9 reclaims the space.
const V9_DROP_CATALOG: &str = r#"
DROP TABLE IF EXISTS catalog_fts;
DROP TABLE IF EXISTS catalog;
"#;

/// v10 → persist the full `mold:parameters` JSON next to the indexed
/// compatibility columns. Newer generate controls should round-trip through
/// gallery Recreate without adding one SQLite column per option.
const V10_GENERATION_METADATA_JSON: &str = r#"
ALTER TABLE generations ADD COLUMN metadata_json TEXT;
"#;

/// v11 → durable chain-job index. The manifest in each job directory is
/// the portable source of truth; these tables provide queryable state and
/// startup reconciliation scans.
const V11_CHAIN_JOBS: &str = r#"
CREATE TABLE chain_jobs (
    id            TEXT PRIMARY KEY,
    state         TEXT NOT NULL,
    model         TEXT NOT NULL,
    request_json  TEXT NOT NULL,
    job_dir       TEXT NOT NULL,
    stage_count   INTEGER NOT NULL,
    current_stage INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    finalized_at  INTEGER
);

CREATE TABLE chain_job_stages (
    job_id             TEXT NOT NULL REFERENCES chain_jobs(id) ON DELETE CASCADE,
    stage_idx          INTEGER NOT NULL,
    state              TEXT NOT NULL,
    seed               INTEGER NOT NULL,
    frames_emitted     INTEGER,
    generation_time_ms INTEGER,
    segment_rel_path   TEXT,
    error              TEXT,
    updated_at         INTEGER NOT NULL,
    PRIMARY KEY (job_id, stage_idx)
);
"#;

/// v12 → machine-wide desired device enablement. This table is deliberately
/// not profile-scoped: hardware administration applies to the whole server.
/// Missing rows mean enabled-by-default, so merely discovering a device never
/// writes to the database.
const V12_DEVICE_PREFERENCES: &str = r#"
CREATE TABLE device_preferences (
    device_id       TEXT PRIMARY KEY,
    desired_enabled INTEGER NOT NULL CHECK (desired_enabled IN (0, 1)),
    updated_at      INTEGER NOT NULL
);
"#;

/// v13 → learned scheduler timing and memory observations.
const V13_SCHEDULER_ESTIMATES: &str = r#"
CREATE TABLE scheduler_estimates (
    estimate_key                 TEXT PRIMARY KEY,
    device_class                TEXT NOT NULL,
    model_fingerprint           TEXT NOT NULL,
    work_kind                   TEXT NOT NULL,
    shape_bucket                TEXT NOT NULL,
    execution_fingerprint       TEXT NOT NULL,
    sample_count                INTEGER NOT NULL,
    ewma_total_ms               REAL NOT NULL,
    ewma_load_ms                REAL,
    vram_high_water_bytes       INTEGER,
    host_high_water_bytes       INTEGER,
    last_observed_at            INTEGER NOT NULL
);
CREATE INDEX idx_scheduler_estimates_last_observed
ON scheduler_estimates(last_observed_at);
"#;

/// v14 → semantic model families, phase timing, and explicit outcomes.
///
/// Existing v13 rows remain valid: scheduler lookup explicitly probes the
/// empty-family exact identity before using v14 semantic-family normalization.
const V14_SCHEDULER_ESTIMATE_EVIDENCE: &str = r#"
ALTER TABLE scheduler_estimates ADD COLUMN model_family TEXT NOT NULL DEFAULT '';
ALTER TABLE scheduler_estimates ADD COLUMN ewma_warm_reload_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_prompt_encode_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_denoise_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_vae_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_upscale_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE scheduler_estimates ADD COLUMN invalidated_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE scheduler_estimates ADD COLUMN last_outcome TEXT NOT NULL DEFAULT 'success';
ALTER TABLE scheduler_estimates ADD COLUMN last_fallback_reason TEXT;
ALTER TABLE scheduler_estimates ADD COLUMN last_invalidated_plan_reason TEXT;
"#;

/// v15 → setup-independent scheduler runtime evidence.
///
/// NULL preserves the exact v13/v14 fallback until a successful observation
/// records total minus the setup disposition that actually occurred.
const V15_SCHEDULER_ESTIMATE_RUNTIME: &str = r#"
ALTER TABLE scheduler_estimates ADD COLUMN ewma_runtime_ms REAL;
"#;

/// v16 → individually revocable credentials created by mobile pairing.
///
/// Only SHA-256 digests of the high-entropy bearer credentials are retained.
/// Revocation deletes the row, so a copied client credential stops working
/// immediately and stays revoked across server restarts.
const V16_PAIRED_CLIENTS: &str = r#"
CREATE TABLE paired_clients (
    id                   TEXT PRIMARY KEY,
    server_instance_id   TEXT NOT NULL,
    name                 TEXT NOT NULL,
    client_kind          TEXT NOT NULL,
    credential_hash      BLOB NOT NULL UNIQUE CHECK (length(credential_hash) = 32),
    created_at_ms        INTEGER NOT NULL,
    last_used_at_ms      INTEGER
);
CREATE INDEX idx_paired_clients_instance_created_at
ON paired_clients(server_instance_id, created_at_ms DESC);
"#;

/// v17 → synchronized audio/video output-phase timing.
///
/// The legacy `ewma_vae_ms` column remains authoritative for families that
/// report one undifferentiated VAE phase. NULL additive fields preserve every
/// existing estimate until a runtime emits the more specific typed phases.
const V17_SCHEDULER_AV_PHASES: &str = r#"
ALTER TABLE scheduler_estimates ADD COLUMN ewma_visual_decode_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_audio_decode_ms REAL;
ALTER TABLE scheduler_estimates ADD COLUMN ewma_mux_ms REAL;
"#;

/// v18 → durable singleton generation admission queue.
///
/// Unlike `chain_jobs` there is no companion manifest: a queued singleton owns
/// no artifacts, so the row is the whole state. A row present at startup means
/// this installation died owing that output.
///
/// `owner_uuid` is a port-independent journal identity persisted in `settings`
/// — deliberately NOT `instance_id`, which is scoped to `(data dir, port)` and
/// would make a server that came back on a different port orphan its own
/// queue. Deliberately not profile-scoped, matching `chain_jobs`: queued jobs
/// are server-wide.
///
/// The row never carries a secret. Reference-upload handles and resolved
/// reference paths are excluded at admission rather than redacted here.
const V18_GENERATION_QUEUE: &str = r#"
CREATE TABLE generation_queue (
    id                 TEXT PRIMARY KEY,
    owner_uuid         TEXT NOT NULL,
    state              TEXT NOT NULL,
    model              TEXT NOT NULL,
    request_json       TEXT NOT NULL,
    output_dir         TEXT NOT NULL,
    target_gpu         INTEGER,
    completion_payload TEXT NOT NULL,
    seed_pinned        INTEGER NOT NULL DEFAULT 0,
    dispatch_attempts  INTEGER NOT NULL DEFAULT 0,
    replay_seen        INTEGER NOT NULL DEFAULT 0,
    held_reason        TEXT,
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL,
    started_at         INTEGER
);

-- `rowid` breaks ties for same-millisecond inserts in the replay query but
-- cannot appear in an index (SQLite rejects it as a column here), so the
-- index covers the selective prefix and the tiebreak rides on the ordering.
CREATE INDEX generation_queue_replay
ON generation_queue(owner_uuid, state, created_at);
"#;

/// v19 → remember a queued job's STABLE device pin, not just the ordinal it
/// resolved to.
///
/// `PATCH /api/queue/:id` accepts `hard_pinned_device_id` and resolves it to an
/// ordinal for dispatch, but ordinals are an enumeration artifact: MIG
/// reconfiguration or a changed `MOLD_GPUS` renumbers them across a restart.
/// Replaying a stale ordinal runs the job on the wrong device, or fails while
/// the device the user actually pinned is present. NULL means Auto or a
/// legacy ordinal-only pin.
const V19_GENERATION_QUEUE_DEVICE_PIN: &str = r#"
ALTER TABLE generation_queue ADD COLUMN target_device_id TEXT;
"#;

/// v20 → library organization: titles, favorites, tags, collections, trash.
///
/// `title`, `favorite`, and `trashed_at_ms` are user-owned columns on
/// `generations`. The upsert path seeds `title` on insert and otherwise
/// leaves all three alone on conflict, so a reconcile refresh can never
/// reset them. Side tables FK `generations(id) ON DELETE CASCADE`, so a
/// hard delete drops tag and collection membership with the row.
///
/// A trashed row keeps its `(output_dir, filename)` identity; the bytes move
/// to `<output_dir>/.trash/<filename>` and `trashed_at_ms` records when.
const V20_LIBRARY_ORGANIZATION: &str = r#"
ALTER TABLE generations ADD COLUMN title TEXT;
ALTER TABLE generations ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0;
ALTER TABLE generations ADD COLUMN trashed_at_ms INTEGER;
CREATE INDEX IF NOT EXISTS idx_gen_trashed ON generations(trashed_at_ms);
CREATE INDEX IF NOT EXISTS idx_gen_favorite ON generations(favorite);

CREATE TABLE tags (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE generation_tags (
    generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    tag_id        INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (generation_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_generation_tags_tag ON generation_tags(tag_id);

CREATE TABLE collections (
    id             TEXT    PRIMARY KEY,
    name           TEXT    NOT NULL,
    slug           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    description    TEXT,
    cover_filename TEXT,
    created_at_ms  INTEGER NOT NULL,
    updated_at_ms  INTEGER NOT NULL
);

CREATE TABLE collection_items (
    collection_id TEXT    NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    added_at_ms   INTEGER NOT NULL,
    PRIMARY KEY (collection_id, generation_id)
);
CREATE INDEX IF NOT EXISTS idx_collection_items_generation ON collection_items(generation_id);
"#;

/// Hidden collections stay visible on the Collections shelf, but clients omit
/// their members from the default Library grid and its search results.
const V21_HIDDEN_COLLECTIONS: &str = r#"
ALTER TABLE collections ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0;
"#;

/// Ordered list of schema migrations. Version numbers must be strictly
/// increasing — [`apply_pending`] validates this at startup.
pub(crate) const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        kind: MigrationKind::Sql(V1_INITIAL_SCHEMA),
    },
    Migration {
        version: 2,
        kind: MigrationKind::Rust(canonicalize_existing_output_dirs),
    },
    Migration {
        version: 3,
        kind: MigrationKind::Sql(V3_SETTINGS_TABLE),
    },
    Migration {
        version: 4,
        kind: MigrationKind::Sql(V4_MODEL_PREFS_TABLE),
    },
    Migration {
        version: 5,
        kind: MigrationKind::Sql(V5_PROMPT_HISTORY_TABLE),
    },
    Migration {
        version: 6,
        kind: MigrationKind::Sql(V6_PROFILE_SCOPING),
    },
    Migration {
        version: 7,
        kind: MigrationKind::Sql(V7_CATALOG_TABLE),
    },
    Migration {
        version: 8,
        kind: MigrationKind::Sql(V8_CATALOG_TRAINED_WORDS),
    },
    Migration {
        version: 9,
        kind: MigrationKind::Sql(V9_DROP_CATALOG),
    },
    Migration {
        version: 10,
        kind: MigrationKind::Sql(V10_GENERATION_METADATA_JSON),
    },
    Migration {
        version: 11,
        kind: MigrationKind::Sql(V11_CHAIN_JOBS),
    },
    Migration {
        version: 12,
        kind: MigrationKind::Sql(V12_DEVICE_PREFERENCES),
    },
    Migration {
        version: 13,
        kind: MigrationKind::Sql(V13_SCHEDULER_ESTIMATES),
    },
    Migration {
        version: 14,
        kind: MigrationKind::Sql(V14_SCHEDULER_ESTIMATE_EVIDENCE),
    },
    Migration {
        version: 15,
        kind: MigrationKind::Sql(V15_SCHEDULER_ESTIMATE_RUNTIME),
    },
    Migration {
        version: 16,
        kind: MigrationKind::Sql(V16_PAIRED_CLIENTS),
    },
    Migration {
        version: 17,
        kind: MigrationKind::Sql(V17_SCHEDULER_AV_PHASES),
    },
    Migration {
        version: 18,
        kind: MigrationKind::Sql(V18_GENERATION_QUEUE),
    },
    Migration {
        version: 19,
        kind: MigrationKind::Sql(V19_GENERATION_QUEUE_DEVICE_PIN),
    },
    Migration {
        version: 20,
        kind: MigrationKind::Sql(V20_LIBRARY_ORGANIZATION),
    },
    Migration {
        version: 21,
        kind: MigrationKind::Sql(V21_HIDDEN_COLLECTIONS),
    },
    Migration {
        version: 22,
        kind: MigrationKind::Sql(V22_IDENTITY_EXTRACT_ESTIMATE),
    },
    Migration {
        version: 23,
        kind: MigrationKind::Sql(V23_HETEROGENEOUS_GENERATION_BATCHES),
    },
    Migration {
        version: 24,
        kind: MigrationKind::Sql(V24_DURABLE_QUEUE_CLAIMS),
    },
    Migration {
        version: 25,
        kind: MigrationKind::Rust(add_generation_job_lookup_indexes),
    },
    Migration {
        version: 26,
        kind: MigrationKind::Sql(V26_GENERATION_QUEUE_MEDIA),
    },
    Migration {
        version: 27,
        kind: MigrationKind::Sql(V27_GENERATION_QUEUE_OWNER_ORDER),
    },
    Migration {
        version: 28,
        kind: MigrationKind::Sql(V28_GENERATION_QUEUE_RETRYABLE),
    },
    Migration {
        version: 29,
        kind: MigrationKind::Sql(V29_GENERATION_BATCH_CHILD_REVISION),
    },
    Migration {
        version: 30,
        kind: MigrationKind::Sql(V30_GENERATION_BATCH_CHILD_ERROR_CODE),
    },
    Migration {
        version: 31,
        kind: MigrationKind::Sql(V31_GENERATION_DIR_RECENCY_INDEX),
    },
    Migration {
        version: 32,
        kind: MigrationKind::Sql(V32_GENERATION_QUEUE_EXPLICIT_PAUSE),
    },
    Migration {
        version: 33,
        kind: MigrationKind::Sql(V33_GALLERY_RETAINED_MEDIA),
    },
    Migration {
        version: 34,
        kind: MigrationKind::Sql(V34_VIDEO_UPSCALE_JOBS),
    },
];

/// The gallery listing is `WHERE output_dir = ? ORDER BY
/// COALESCE(file_mtime_ms, created_at_ms) DESC`; without an index matching
/// that exact expression SQLite filtered on `idx_gen_output_dir` and then
/// sorted every row of the directory in a temp B-tree on every `GET
/// /api/gallery` (and every 15 s poll). An expression index lets the planner
/// walk rows already in listing order.
const V31_GENERATION_DIR_RECENCY_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_gen_dir_recency
    ON generations(output_dir, COALESCE(file_mtime_ms, created_at_ms) DESC);
"#;

/// Distinguish a user's explicit one-job pause from restart recovery. Global
/// Resume releases only restart-paused work; it must never wake a row the user
/// deliberately parked through `/api/queue/{id}/pause`.
const V32_GENERATION_QUEUE_EXPLICIT_PAUSE: &str = r#"
ALTER TABLE generation_queue
    ADD COLUMN explicitly_paused INTEGER NOT NULL DEFAULT 0;
"#;

/// Durable, independently resumable framewise video-upscale jobs. Source
/// paths are resolved from authenticated gallery/upload authority at
/// admission; callers never provide arbitrary server paths.
const V34_VIDEO_UPSCALE_JOBS: &str = r#"
CREATE TABLE video_upscale_jobs (
    id                TEXT PRIMARY KEY,
    state             TEXT NOT NULL CHECK (state IN
                         ('queued','running','finalizing','paused','completed','failed','cancelled')),
    source_json       TEXT NOT NULL,
    source_path       TEXT NOT NULL,
    model             TEXT NOT NULL,
    scale_factor      INTEGER NOT NULL,
    tile_size         INTEGER,
    completed_frames  INTEGER NOT NULL DEFAULT 0,
    total_frames      INTEGER NOT NULL DEFAULT 0,
    source_facts_json TEXT,
    output_facts_json TEXT,
    output_filename   TEXT,
    error             TEXT,
    work_dir          TEXT NOT NULL,
    created_at_ms     INTEGER NOT NULL,
    updated_at_ms     INTEGER NOT NULL
);
CREATE INDEX video_upscale_jobs_state_created
    ON video_upscale_jobs(state, created_at_ms);
"#;

/// Repairable SQLite projection of encrypted gallery-owned source-media
/// pins. The committed gallery checkpoint remains lifecycle authority.
const V33_GALLERY_RETAINED_MEDIA: &str = r#"
CREATE TABLE gallery_media_sets (
    media_set_id TEXT NOT NULL CHECK (length(media_set_id) > 0),
    owner_uuid   TEXT NOT NULL CHECK (length(owner_uuid) > 0),
    job_id       TEXT NOT NULL CHECK (length(job_id) > 0),
    PRIMARY KEY (media_set_id, owner_uuid, job_id)
);

CREATE TABLE gallery_media_bindings (
    output_dir   TEXT NOT NULL,
    filename     TEXT NOT NULL CHECK (length(filename) > 0),
    pin_id       TEXT NOT NULL CHECK (length(pin_id) = 64),
    media_set_id TEXT NOT NULL,
    owner_uuid   TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    FOREIGN KEY (media_set_id, owner_uuid, job_id)
        REFERENCES gallery_media_sets(media_set_id, owner_uuid, job_id)
        ON DELETE CASCADE,
    PRIMARY KEY (output_dir, filename, pin_id, media_set_id, owner_uuid, job_id)
);

CREATE INDEX gallery_media_bindings_set
ON gallery_media_bindings(media_set_id, owner_uuid, job_id, output_dir, filename);
"#;
/// #1227 phase 2 moved face-identity extraction from admission onto the
/// render's leased device, where it became a typed scheduler phase
/// (`ProgressPhase::IdentityExtract`). Its EWMA needs a column: unlike an
/// additive Rust `Option` an old row already reads as `NULL`, SQLite needs the
/// column to exist before the new `INSERT`/`UPDATE` can name it.
///
/// Appended after every existing column so no `SELECT` index in
/// `scheduler_estimates.rs` moves.
const V22_IDENTITY_EXTRACT_ESTIMATE: &str = r#"
ALTER TABLE scheduler_estimates ADD COLUMN ewma_identity_extract_ms REAL;
"#;

/// Durable grouping and idempotency for one heterogeneous client admission.
/// Child requests continue to live in `generation_queue` as the sole replay
/// authority; terminal summaries survive after those queue rows are removed.
const V23_HETEROGENEOUS_GENERATION_BATCHES: &str = r#"
CREATE TABLE generation_batches (
    id             TEXT PRIMARY KEY,
    client_batch_id TEXT NOT NULL,
    owner_uuid     TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    created_at_ms  INTEGER NOT NULL,
    UNIQUE(owner_uuid, client_batch_id)
);

CREATE TABLE generation_batch_children (
    batch_id      TEXT NOT NULL REFERENCES generation_batches(id) ON DELETE CASCADE,
    job_id        TEXT PRIMARY KEY,
    batch_index   INTEGER NOT NULL,
    state         TEXT NOT NULL,
    error         TEXT,
    updated_at_ms INTEGER NOT NULL,
    UNIQUE(batch_id, batch_index)
);

CREATE INDEX generation_batch_children_parent
ON generation_batch_children(batch_id, batch_index);

CREATE TABLE gallery_mutation_receipts (
    operation_id   TEXT PRIMARY KEY,
    request_sha256 TEXT NOT NULL,
    response_json  TEXT NOT NULL,
    created_at_ms  INTEGER NOT NULL
);
"#;

/// Token-fenced runtime claims and reconnectable batch-child outcomes.
///
/// Queue order remains `(created_at, rowid)`: v24 deliberately does not
/// rewrite the existing admission rows or introduce a second ordering
/// authority. A partial claimable index lets the feeder reserve one oldest
/// row without reading the backlog into memory. The token is runtime-only and
/// cleared during startup recovery.
///
/// Structured terminal payloads are opaque JSON at this layer. Their schemas
/// belong to the server/core wire contract; SQLite only commits them beside
/// the legacy `state`/`error` columns and queue-row deletion.
const V24_DURABLE_QUEUE_CLAIMS: &str = r#"
ALTER TABLE generation_queue ADD COLUMN claim_token TEXT;

CREATE UNIQUE INDEX generation_queue_claim_token
ON generation_queue(claim_token)
WHERE claim_token IS NOT NULL;

CREATE INDEX generation_queue_claimable
ON generation_queue(owner_uuid, state, created_at)
WHERE claim_token IS NULL;

ALTER TABLE generation_batch_children ADD COLUMN terminal_error_json TEXT;
ALTER TABLE generation_batch_children ADD COLUMN result_json TEXT;
ALTER TABLE generation_batch_children ADD COLUMN completed_at_ms INTEGER;
"#;

/// The highest migration version this build ships. Exposed publicly so
/// operators / tests can assert what schema level they're running against.
pub const SCHEMA_VERSION: i64 = 34;

/// Opaque staged-media ownership for durable queue rows.
///
/// SQLite records only the random set identifier and cleanup obligation. The
/// staging format, members, roles, names, digests, paths, and byte counts stay
/// outside the database. A partial unique index makes one non-NULL marker
/// belong to one queue row while preserving every legacy/media-free NULL.
///
/// Cleanup cannot cascade from `generation_queue`: the obligation must
/// survive authority deletion until the filesystem cleanup has succeeded.
/// The AFTER DELETE trigger is therefore the singular retirement authority
/// for direct, bulk, legacy, batch, and claimed deletion paths.
const V26_GENERATION_QUEUE_MEDIA: &str = r#"
CREATE TABLE generation_queue_media (
    media_set_id  TEXT PRIMARY KEY NOT NULL CHECK (length(media_set_id) > 0),
    owner_uuid    TEXT NOT NULL CHECK (length(owner_uuid) > 0),
    state         TEXT NOT NULL CHECK (state IN ('active', 'gc_pending')),
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE INDEX generation_queue_media_owner_state
ON generation_queue_media(owner_uuid, state, created_at_ms, media_set_id);

ALTER TABLE generation_queue ADD COLUMN media_set_id TEXT
    REFERENCES generation_queue_media(media_set_id);

CREATE UNIQUE INDEX generation_queue_media_set
ON generation_queue(media_set_id)
WHERE media_set_id IS NOT NULL;

CREATE TRIGGER generation_queue_media_reference_insert
BEFORE INSERT ON generation_queue
WHEN NEW.media_set_id IS NOT NULL
 AND NOT EXISTS (
        SELECT 1 FROM generation_queue_media AS media
         WHERE media.media_set_id = NEW.media_set_id
           AND media.owner_uuid = NEW.owner_uuid
           AND media.state = 'active'
 )
BEGIN
    SELECT RAISE(ABORT, 'queue media obligation is not active for this owner');
END;

CREATE TRIGGER generation_queue_media_marker_immutable
BEFORE UPDATE OF media_set_id ON generation_queue
WHEN OLD.media_set_id IS NOT NEW.media_set_id
BEGIN
    SELECT RAISE(ABORT, 'queue media marker is immutable');
END;

CREATE TRIGGER generation_queue_media_owner_update
BEFORE UPDATE OF owner_uuid ON generation_queue
WHEN NEW.media_set_id IS NOT NULL
 AND NOT EXISTS (
        SELECT 1 FROM generation_queue_media AS media
         WHERE media.media_set_id = NEW.media_set_id
           AND media.owner_uuid = NEW.owner_uuid
           AND media.state = 'active'
 )
BEGIN
    SELECT RAISE(ABORT, 'queue media obligation is not active for this owner');
END;

CREATE TRIGGER generation_queue_media_retire
AFTER DELETE ON generation_queue
WHEN OLD.media_set_id IS NOT NULL
BEGIN
    UPDATE generation_queue_media
       SET state = 'gc_pending',
           updated_at_ms = MAX(
               updated_at_ms,
               CAST((julianday('now') - 2440587.5) * 86400000.0 AS INTEGER)
           )
     WHERE media_set_id = OLD.media_set_id
       AND state = 'active';
END;
"#;

/// Owner-first stable ordering for payload-free queue pagination.
///
/// SQLite appends the table rowid to every ordinary secondary-index entry,
/// so `(owner_uuid, created_at)` also satisfies the queue's authoritative
/// `(created_at, rowid)` order without a temporary sort. Unlike the older
/// replay index, no state column interrupts the owner/order prefix.
const V27_GENERATION_QUEUE_OWNER_ORDER: &str = r#"
CREATE INDEX generation_queue_owner_order
ON generation_queue(owner_uuid, created_at);
"#;

/// Retry authority for parked durable generations.
///
/// Existing held rows remain operator-visible but cannot be requeued blindly:
/// corrupt media and invalid publication authority need repair, not another
/// automatic attempt. Deferred dependency preparation marks only its own
/// recoverable holds retryable, and the retry endpoint is fenced by this bit.
const V28_GENERATION_QUEUE_RETRYABLE: &str = r#"
ALTER TABLE generation_queue ADD COLUMN retryable INTEGER NOT NULL DEFAULT 0;
ALTER TABLE generation_queue ADD COLUMN admission_authority TEXT
    CHECK (
        admission_authority IS NULL OR (
            length(admission_authority) BETWEEN 56 AND 2048
        )
    );
"#;

/// A monotonic per-child version, because a wall-clock millisecond is not one.
///
/// `updated_at_ms` was carrying two jobs at once: an honest human-facing
/// timestamp, and the ordering token every client reducer compares to decide
/// whether an incoming snapshot supersedes the one it holds. Those are
/// different concerns, and the timestamp is the weaker of the two — SQLite
/// commits several transitions inside one millisecond routinely, and
/// `POST /api/queue/:id/retry` is the first route that moves a child
/// BACKWARD through the browser's `FORWARD_PHASE_RANK` (held -> accepted),
/// so a same-millisecond collision there is not a tie to be broken
/// arbitrarily: it decides whether the retry is visible at all.
///
/// `revision` is that ordering token, incremented by every authoritative
/// state transition and by nothing else. It also answers a question no
/// timestamp can: after an ambiguous retry POST whose response was lost,
/// "did my retry land?" is exactly "is the child's revision above the one I
/// captured before the POST" — a still-`held` child at a HIGHER revision
/// was retried and re-held, which is a different fact from a retry that
/// never arrived.
///
/// Existing rows start at 0 and take their first increment on their next
/// transition. `updated_at_ms` keeps its own monotonic bump on retry for
/// clients that predate this column and still order by it.
const V29_GENERATION_BATCH_CHILD_REVISION: &str = r#"
ALTER TABLE generation_batch_children ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
"#;

/// A held child's typed refusal code beside its human sentence.
///
/// The feeder holds a child with the preparation error's `code`
/// (`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`, …) but only ever persisted the
/// sentence, so every client's missing-model pull offer — which classifies
/// on the code — had nothing to read. Additive and nullable: a row held
/// before this migration reads `NULL`, which a client treats as "no typed
/// cause". Cleared by the retry route with the sentence.
const V30_GENERATION_BATCH_CHILD_ERROR_CODE: &str = r#"
ALTER TABLE generation_batch_children ADD COLUMN error_code TEXT;
"#;

/// Build a serde-compatible reverse lookup for durable publication recovery.
///
/// SQLite JSON projections are intentionally not the authority here: their
/// duplicate-key and coercion behavior need not match the serde parse used by
/// gallery metadata. The migration classifies every retained row with the
/// exact Rust parser, and current writers maintain the same projection.
fn add_generation_job_lookup_indexes(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        r#"
        ALTER TABLE generations ADD COLUMN queue_job_id TEXT;
        ALTER TABLE generations ADD COLUMN queue_job_metadata_state INTEGER;
        "#,
    )?;

    let rows = {
        let mut stmt = tx.prepare("SELECT id, metadata_json FROM generations ORDER BY id")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for (id, metadata_json) in rows {
        let (job_id, state) = match metadata_json {
            None => (None, 1_i64),
            Some(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(value) => match value.get("job_id") {
                    None => (None, 1),
                    Some(serde_json::Value::String(job_id)) => (Some(job_id.clone()), 1),
                    Some(_) => (None, 0),
                },
                Err(_) => (None, 0),
            },
        };
        tx.execute(
            "UPDATE generations
                SET queue_job_id = ?1, queue_job_metadata_state = ?2
              WHERE id = ?3",
            rusqlite::params![job_id, state, id],
        )?;
    }

    tx.execute_batch(
        r#"
        CREATE INDEX generations_queue_job_id
        ON generations(queue_job_id, id)
        WHERE queue_job_metadata_state = 1 AND queue_job_id IS NOT NULL;

        CREATE INDEX generations_output_queue_job_id
        ON generations(output_dir, queue_job_id, id)
        WHERE queue_job_metadata_state = 1 AND queue_job_id IS NOT NULL;

        CREATE INDEX generations_invalid_queue_metadata
        ON generations(queue_job_metadata_state, id)
        WHERE queue_job_metadata_state = 0;

        CREATE INDEX generations_output_invalid_queue_metadata
        ON generations(output_dir, queue_job_metadata_state, created_at_ms, id)
        WHERE queue_job_metadata_state = 0;

        CREATE INDEX generations_unknown_queue_metadata
        ON generations(queue_job_metadata_state, id)
        WHERE queue_job_metadata_state IS NULL;

        CREATE INDEX generations_output_unknown_queue_metadata
        ON generations(output_dir, queue_job_metadata_state, created_at_ms, id)
        WHERE queue_job_metadata_state IS NULL;

        CREATE TRIGGER generations_queue_job_projection_dirty
        AFTER UPDATE OF metadata_json ON generations
        WHEN NEW.metadata_json IS NOT OLD.metadata_json
        BEGIN
            UPDATE generations
               SET queue_job_id = NULL, queue_job_metadata_state = NULL
             WHERE id = NEW.id;
        END;
        "#,
    )?;
    Ok(())
}

/// v1 → v2: rewrite every `output_dir` value to its canonical form so
/// rows written by the v0.8.x release (which keyed on raw paths) keep
/// matching the new canonicalized lookups. Without this, an upgraded
/// install would see every row written under `/tmp/...` or a symlinked
/// directory stop matching queries, and reconcile would insert fresh
/// duplicates under the canonical key.
///
/// The rewrite is conflict-safe: if a row already exists under the
/// canonical key (e.g. because both forms somehow got written), we
/// prefer the canonical row and drop the legacy one.
fn canonicalize_existing_output_dirs(tx: &Transaction<'_>) -> Result<()> {
    #[derive(Debug)]
    struct Row {
        id: i64,
        output_dir: String,
        filename: String,
    }

    // Pull the full set up front — the table is tiny (rows measured in
    // thousands at most) and avoids holding a statement open while we
    // run UPDATE/DELETE against the same table.
    let mut stmt = tx.prepare("SELECT id, output_dir, filename FROM generations")?;
    let rows: Vec<Row> = stmt
        .query_map([], |r| {
            Ok(Row {
                id: r.get(0)?,
                output_dir: r.get(1)?,
                filename: r.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    let mut rewritten = 0u64;
    let mut dropped_legacy_dup = 0u64;
    for row in rows {
        let canonical = canonical_dir_string(std::path::Path::new(&row.output_dir));
        if canonical == row.output_dir {
            continue;
        }
        // Is there already a row under the canonical key + same filename?
        let conflict: Option<i64> = tx
            .query_row(
                "SELECT id FROM generations WHERE output_dir = ?1 AND filename = ?2",
                rusqlite::params![canonical, row.filename],
                |r| r.get(0),
            )
            .ok();
        if conflict.is_some_and(|id| id != row.id) {
            // Canonical row wins — drop the legacy one to keep UNIQUE happy.
            tx.execute(
                "DELETE FROM generations WHERE id = ?1",
                rusqlite::params![row.id],
            )?;
            dropped_legacy_dup += 1;
        } else {
            tx.execute(
                "UPDATE generations SET output_dir = ?1 WHERE id = ?2",
                rusqlite::params![canonical, row.id],
            )?;
            rewritten += 1;
        }
    }
    if rewritten > 0 || dropped_legacy_dup > 0 {
        tracing::info!(
            rewritten,
            dropped_legacy_dup,
            "v2 migration canonicalized existing output_dir keys"
        );
    }
    Ok(())
}

/// Apply every migration whose version is greater than the DB's current
/// `user_version` pragma. Runs each migration in its own transaction —
/// partial failures leave the DB at the previous version instead of a
/// half-migrated state. A catastrophic crash between migrations is safe
/// because each transaction commits the `user_version` bump alongside
/// the DDL.
pub fn apply_pending(conn: &mut Connection) -> Result<i64> {
    // Sanity-check the migration list in debug builds — the SCHEMA_VERSION
    // constant must match the last entry and versions must be monotonic.
    debug_assert!(!MIGRATIONS.is_empty(), "migration list cannot be empty");
    debug_assert_eq!(
        MIGRATIONS.last().map(|m| m.version),
        Some(SCHEMA_VERSION),
        "SCHEMA_VERSION must match the last migration"
    );
    for win in MIGRATIONS.windows(2) {
        debug_assert!(
            win[0].version < win[1].version,
            "migration versions must be strictly increasing"
        );
    }

    // Concurrency: multiple processes open mold.db simultaneously in the
    // default setup (`mold tui` beside its auto-spawned `mold serve`).
    // Each migration therefore runs under an IMMEDIATE transaction — the
    // write lock is taken up front — and re-reads `user_version` inside
    // that lock. A connection that lost the race sees the bumped version
    // and skips, instead of re-applying DDL and corrupting the schema
    // ("duplicate column" on every subsequent open).
    let mut current;
    loop {
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        current = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        let Some(m) = MIGRATIONS.iter().find(|m| m.version > current) else {
            break;
        };
        if m.version != current + 1 {
            bail!(
                "migration gap: DB at v{}, next migration is v{}, expected v{}",
                current,
                m.version,
                current + 1
            );
        }
        match &m.kind {
            MigrationKind::Sql(sql) => tx.execute_batch(sql)?,
            MigrationKind::Rust(run) => run(&tx)?,
        }
        // `user_version` pragma doesn't bind parameters — safe because
        // `m.version` is compile-time constant from our own source.
        tx.execute_batch(&format!("PRAGMA user_version = {};", m.version))?;
        tx.commit()?;
        tracing::info!(version = m.version, "applied metadata DB migration");
    }
    Ok(current)
}

/// Read the DB's current schema version from the `user_version` pragma.
/// A freshly-created DB returns `0`.
pub(crate) fn current_version(conn: &Connection) -> Result<i64> {
    let v: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_opens_of_a_fresh_db_all_migrate_safely() {
        // Regression: `mold tui` + its auto-spawned `mold serve` open the
        // same mold.db at boot. Racing connections both read the old
        // user_version and both applied the same ALTER TABLE — the loser
        // hit "duplicate column" and the file failed every later open.
        let tmp = std::env::temp_dir().join(format!(
            "mold-db-migrate-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("mold.db");

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || crate::MetadataDb::open(&path).map(|_| ()))
            })
            .collect();
        for h in handles {
            h.join()
                .expect("thread panicked")
                .expect("every racing open must migrate or observe cleanly");
        }
        // And the file must still open fine afterwards.
        crate::MetadataDb::open(&path).expect("post-race open must succeed");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn migration_list_invariants_hold() {
        assert!(!MIGRATIONS.is_empty());
        assert_eq!(MIGRATIONS.last().unwrap().version, SCHEMA_VERSION);
        for win in MIGRATIONS.windows(2) {
            assert!(win[0].version < win[1].version);
        }
    }

    #[test]
    fn apply_pending_on_fresh_db_reaches_schema_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        let v = apply_pending(&mut conn).unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
    }

    #[test]
    fn apply_pending_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let v1 = current_version(&conn).unwrap();
        apply_pending(&mut conn).unwrap();
        let v2 = current_version(&conn).unwrap();
        assert_eq!(v1, v2);
    }

    /// Synthesize an ad-hoc DDL migration at runtime (without touching the
    /// real MIGRATIONS list) to prove the transaction wrapping + ordering
    /// works. This is the pattern future `ALTER TABLE ADD COLUMN`
    /// migrations will follow.
    #[test]
    fn transaction_wraps_each_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();

        let sql = "ALTER TABLE generations ADD COLUMN test_col TEXT;\n\
                   PRAGMA user_version = 99;";
        let tx = conn.transaction().unwrap();
        tx.execute_batch(sql).unwrap();
        tx.commit().unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('generations') WHERE name = 'test_col'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(current_version(&conn).unwrap(), 99);
    }

    /// v2 migration (Codex finding): simulate the pre-upgrade state where
    /// v1 rows were written under a non-canonical path. After v2 runs,
    /// the rows must sit at the canonical key so new queries can find
    /// them — no orphans, no duplicates.
    #[cfg(target_os = "macos")]
    #[test]
    fn v2_canonicalizes_legacy_tmp_rows_on_macos() {
        // Apply v1 only so we control the pre-v2 row layout.
        let mut conn = Connection::open_in_memory().unwrap();
        let v1_only = Migration {
            version: 1,
            kind: MigrationKind::Sql(V1_INITIAL_SCHEMA),
        };
        let tx = conn.transaction().unwrap();
        match &v1_only.kind {
            MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
            _ => unreachable!(),
        }
        tx.execute_batch("PRAGMA user_version = 1;").unwrap();
        tx.commit().unwrap();

        // Seed a row under a non-canonical /tmp alias.
        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let legacy_path = tmp.path().to_string_lossy().into_owned();
        assert!(legacy_path.starts_with("/tmp/"), "test setup sanity");
        conn.execute(
            "INSERT INTO generations
                (filename, output_dir, created_at_ms, format, model)
             VALUES (?1, ?2, 0, 'png', 'm')",
            rusqlite::params!["legacy.png", legacy_path],
        )
        .unwrap();

        // Now run the real migration pipeline — v2 should rewrite the row.
        let final_v = apply_pending(&mut conn).unwrap();
        assert_eq!(final_v, SCHEMA_VERSION);

        let stored: String = conn
            .query_row(
                "SELECT output_dir FROM generations WHERE filename = 'legacy.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let canonical = canonical_dir_string(tmp.path());
        assert_eq!(stored, canonical, "v2 must rewrite legacy /tmp key");
        assert_ne!(stored, legacy_path, "must be the canonical form");
    }

    /// v2 edge case: if a row already exists under the canonical key when
    /// the legacy row is encountered, the migration must drop the legacy
    /// row rather than blow up the UNIQUE constraint.
    #[cfg(target_os = "macos")]
    #[test]
    fn v2_drops_legacy_dup_when_canonical_already_present() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch("PRAGMA user_version = 1;").unwrap();
        tx.commit().unwrap();

        let tmp = tempfile::tempdir_in("/tmp").unwrap();
        let legacy = tmp.path().to_string_lossy().into_owned();
        let canonical = canonical_dir_string(tmp.path());
        assert_ne!(legacy, canonical);

        // Seed both a legacy-keyed and a canonical-keyed row with the same filename.
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model)
             VALUES ('dup.png', ?1, 0, 'png', 'm')",
            rusqlite::params![legacy],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model)
             VALUES ('dup.png', ?1, 0, 'png', 'm')",
            rusqlite::params![canonical],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM generations WHERE filename = 'dup.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "legacy row should be dropped, canonical kept");
        let kept: String = conn
            .query_row(
                "SELECT output_dir FROM generations WHERE filename = 'dup.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kept, canonical);
    }

    // ------------------------------------------------------------------
    // v3 / v4 / v5 migration tests — added for feat/sqlite-settings.
    // These assert the shape of the new tables before the migrations
    // land, so a regression in any future refactor is caught early.
    // ------------------------------------------------------------------

    fn column_names(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info('{table}')"))
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn table_exists(conn: &Connection, table: &str) -> bool {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |r| r.get(0),
            )
            .unwrap();
        n == 1
    }

    /// The Library listing must walk the new expression index in listing
    /// order rather than filter-then-sort: a temp B-tree over every row of
    /// the directory ran on every `GET /api/gallery` and every poll.
    #[test]
    fn gallery_listing_uses_the_recency_index_instead_of_a_temp_sort() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let plan: Vec<String> = conn
            .prepare(
                "EXPLAIN QUERY PLAN SELECT id FROM generations \
                 WHERE output_dir = ?1 \
                 ORDER BY COALESCE(file_mtime_ms, created_at_ms) DESC",
            )
            .unwrap()
            .query_map(["/out"], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let joined = plan.join(" | ");
        assert!(
            joined.contains("idx_gen_dir_recency"),
            "listing must use idx_gen_dir_recency, got: {joined}"
        );
        assert!(
            !joined.contains("TEMP B-TREE"),
            "listing must not sort in a temp B-tree, got: {joined}"
        );
    }

    fn index_exists(conn: &Connection, index: &str) -> bool {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                rusqlite::params![index],
                |r| r.get(0),
            )
            .unwrap();
        n == 1
    }

    #[test]
    fn fresh_db_reaches_latest_schema_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert_eq!(
            current_version(&conn).unwrap(),
            SCHEMA_VERSION,
            "fresh DB must end at the latest SCHEMA_VERSION",
        );
        assert_eq!(SCHEMA_VERSION, 34);
        assert!(table_exists(&conn, "device_preferences"));
        assert_eq!(
            column_names(&conn, "device_preferences"),
            vec!["device_id", "desired_enabled", "updated_at"]
        );
    }

    #[test]
    fn v14_preserves_v13_scheduler_estimates_with_compatible_defaults() {
        let mut conn = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 13)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(run) => run(&tx).unwrap(),
            }
            tx.execute_batch(&format!("PRAGMA user_version = {};", migration.version))
                .unwrap();
            tx.commit().unwrap();
        }
        conn.execute(
            "INSERT INTO scheduler_estimates (
                estimate_key, device_class, model_fingerprint, work_kind, shape_bucket,
                execution_fingerprint, sample_count, ewma_total_ms, ewma_load_ms,
                vram_high_water_bytes, host_high_water_bytes, last_observed_at
             ) VALUES ('legacy', 'cuda:sm86', 'cv:3143864', 'generation', '1024',
                       'bf16', 7, 1000.0, 100.0, 12000, NULL, 10)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        let migrated: (String, i64, i64, String) = conn
            .query_row(
                "SELECT model_family, failure_count, invalidated_count, last_outcome
                 FROM scheduler_estimates WHERE estimate_key = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(migrated, ("".into(), 0, 0, "success".into()));
    }

    /// v6: `settings` keeps every existing row under `profile = 'default'`
    /// and the composite PK `(profile, key)` is in place.
    #[test]
    fn v6_moves_existing_settings_to_default_profile() {
        // Seed a v5 DB with real settings rows, then let apply_pending
        // migrate it forward.
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch(V3_SETTINGS_TABLE).unwrap();
        tx.execute_batch(V4_MODEL_PREFS_TABLE).unwrap();
        tx.execute_batch(V5_PROMPT_HISTORY_TABLE).unwrap();
        tx.execute_batch("PRAGMA user_version = 5;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO settings (key, value, value_type, updated_at_ms)
             VALUES ('tui.theme', 'mocha', 'string', 123)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);

        let cols = column_names(&conn, "settings");
        assert!(
            cols.iter().any(|c| c == "profile"),
            "settings must gain a profile column, got {cols:?}"
        );

        let (profile, value): (String, String) = conn
            .query_row(
                "SELECT profile, value FROM settings WHERE key = 'tui.theme'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(profile, "default");
        assert_eq!(value, "mocha");
    }

    /// v6: `model_prefs` keeps every existing row under `profile = 'default'`.
    #[test]
    fn v6_moves_existing_model_prefs_to_default_profile() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch(V3_SETTINGS_TABLE).unwrap();
        tx.execute_batch(V4_MODEL_PREFS_TABLE).unwrap();
        tx.execute_batch(V5_PROMPT_HISTORY_TABLE).unwrap();
        tx.execute_batch("PRAGMA user_version = 5;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO model_prefs
                (model, width, height, steps, updated_at_ms)
             VALUES ('flux-dev:q4', 1024, 1024, 20, 123)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        let cols = column_names(&conn, "model_prefs");
        assert!(
            cols.iter().any(|c| c == "profile"),
            "model_prefs must gain a profile column, got {cols:?}"
        );
        let (profile, width): (String, i64) = conn
            .query_row(
                "SELECT profile, width FROM model_prefs WHERE model = 'flux-dev:q4'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(profile, "default");
        assert_eq!(width, 1024);
    }

    /// v18: a v17 database gains the durable generation queue without
    /// disturbing anything already recorded.
    #[test]
    fn v18_upgrade_adds_the_generation_queue_and_preserves_existing_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 17)
        {
            match &migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(run) => run(&tx).unwrap(),
            }
        }
        tx.execute_batch("PRAGMA user_version = 17;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO generations
                (filename, output_dir, created_at_ms, format, model)
             VALUES ('kept.png', '/gallery', 1, 'png', 'flux-dev:q4')",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(SCHEMA_VERSION, 34);
        assert!(table_exists(&conn, "generation_queue"));
        let columns = column_names(&conn, "generation_queue");
        for expected in [
            "id",
            "owner_uuid",
            "state",
            "model",
            "request_json",
            "output_dir",
            "target_gpu",
            "completion_payload",
            "seed_pinned",
            "dispatch_attempts",
            "replay_seen",
            "held_reason",
            "retryable",
            "admission_authority",
            "created_at",
            "updated_at",
            "started_at",
        ] {
            assert!(
                columns.iter().any(|column| column == expected),
                "generation_queue must carry {expected}, got {columns:?}"
            );
        }
        let kept: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM generations WHERE filename = 'kept.png'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(kept, 1, "the upgrade must not disturb existing rows");
    }

    #[test]
    fn v24_adds_claim_and_terminal_fields_without_changing_queue_order() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 23)
        {
            match &migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(run) => run(&tx).unwrap(),
            }
        }
        tx.execute_batch("PRAGMA user_version = 23;").unwrap();
        tx.commit().unwrap();
        for id in ["first", "second", "third"] {
            conn.execute(
                "INSERT INTO generation_queue (
                    id, owner_uuid, state, model, request_json, output_dir,
                    completion_payload, created_at, updated_at
                 ) VALUES (?1, 'owner', 'queued', 'model', '{}', '/gallery', 'full', 7, 7)",
                rusqlite::params![id],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO generation_batches
                (id, client_batch_id, owner_uuid, request_sha256, created_at_ms)
             VALUES ('batch', 'client', 'owner', 'hash', 7)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO generation_batch_children
                (batch_id, job_id, batch_index, state, updated_at_ms)
             VALUES ('batch', 'first', 1, 'accepted', 7)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        let ids = conn
            .prepare(
                "SELECT id FROM generation_queue
                 WHERE owner_uuid = 'owner' ORDER BY created_at, rowid",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(ids, vec!["first", "second", "third"]);
        let queue_columns = column_names(&conn, "generation_queue");
        assert!(queue_columns.iter().any(|column| column == "claim_token"));
        let child_columns = column_names(&conn, "generation_batch_children");
        for expected in ["terminal_error_json", "result_json", "completed_at_ms"] {
            assert!(child_columns.iter().any(|column| column == expected));
        }
        let fields: (Option<String>, Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT terminal_error_json, result_json, completed_at_ms
                   FROM generation_batch_children WHERE job_id = 'first'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(fields, (None, None, None));
    }

    /// v20/v21: a v19 database gains organization plus hidden collections
    /// while every existing gallery row survives at neutral defaults.
    #[test]
    fn v20_upgrade_adds_library_organization_and_preserves_existing_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
        let tx = conn.transaction().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 19)
        {
            match &migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(run) => run(&tx).unwrap(),
            }
        }
        tx.execute_batch("PRAGMA user_version = 19;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO generations
                (filename, output_dir, created_at_ms, format, model, prompt)
             VALUES ('kept.png', '/gallery', 1, 'png', 'flux-dev:q4', 'a cat')",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(SCHEMA_VERSION, 34);
        let columns = column_names(&conn, "generations");
        for expected in ["title", "favorite", "trashed_at_ms"] {
            assert!(
                columns.iter().any(|column| column == expected),
                "generations must carry {expected}, got {columns:?}"
            );
        }
        for table in ["tags", "generation_tags", "collections", "collection_items"] {
            assert!(table_exists(&conn, table), "missing table {table}");
        }
        let collection_columns = column_names(&conn, "collections");
        assert!(collection_columns.iter().any(|column| column == "hidden"));
        for index in [
            "idx_gen_trashed",
            "idx_gen_favorite",
            "idx_generation_tags_tag",
            "idx_collection_items_generation",
            "idx_gen_dir_recency",
        ] {
            assert!(index_exists(&conn, index), "missing index {index}");
        }
        let (prompt, title, favorite, trashed): (String, Option<String>, i64, Option<i64>) = conn
            .query_row(
                "SELECT prompt, title, favorite, trashed_at_ms
                 FROM generations WHERE filename = 'kept.png'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(prompt, "a cat");
        assert_eq!(title, None);
        assert_eq!(favorite, 0);
        assert_eq!(trashed, None);

        // Side tables cascade from the gallery row.
        conn.execute(
            "INSERT INTO tags (name, created_at_ms) VALUES ('owls', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO generation_tags (generation_id, tag_id)
             SELECT g.id, t.id FROM generations g, tags t
             WHERE g.filename = 'kept.png' AND t.name = 'owls'",
            [],
        )
        .unwrap();
        conn.execute("DELETE FROM generations WHERE filename = 'kept.png'", [])
            .unwrap();
        let links: i64 = conn
            .query_row("SELECT COUNT(*) FROM generation_tags", [], |row| row.get(0))
            .unwrap();
        assert_eq!(links, 0, "generation_tags must cascade with the row");
    }

    #[test]
    fn v11_upgrade_adds_empty_machine_wide_device_preferences() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 11)
        {
            match &migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(run) => run(&tx).unwrap(),
            }
        }
        tx.execute_batch("PRAGMA user_version = 11;").unwrap();
        tx.commit().unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert!(table_exists(&conn, "device_preferences"));
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM device_preferences", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rows, 0, "discovery must not create preference rows");
        assert!(
            !column_names(&conn, "device_preferences")
                .iter()
                .any(|column| column == "profile"),
            "device preferences are machine-wide, not profile-scoped"
        );
    }

    /// v6: two rows with the same `key` but different `profile` values
    /// coexist under the composite PK.
    #[test]
    fn v6_allows_same_key_across_profiles() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
             VALUES ('default', 'tui.theme', 'mocha', 'string', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO settings (profile, key, value, value_type, updated_at_ms)
             VALUES ('dev', 'tui.theme', 'nord', 'string', 1)",
            [],
        )
        .unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key = 'tui.theme'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2, "same key under distinct profiles must coexist");
    }

    #[test]
    fn v3_creates_settings_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "settings"));
        let cols = column_names(&conn, "settings");
        for expected in &["key", "value", "value_type", "updated_at_ms"] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "settings table missing column {expected}; got {cols:?}"
            );
        }
    }

    #[test]
    fn v4_creates_model_prefs_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "model_prefs"));
        let cols = column_names(&conn, "model_prefs");
        // Spot-check the invariants — every field we persist must exist.
        for expected in &[
            "model",
            "width",
            "height",
            "steps",
            "guidance",
            "scheduler",
            "seed_mode",
            "batch",
            "format",
            "lora_path",
            "lora_scale",
            "expand",
            "offload",
            "strength",
            "control_scale",
            "frames",
            "fps",
            "last_prompt",
            "last_negative",
            "updated_at_ms",
        ] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "model_prefs missing column {expected}; got {cols:?}"
            );
        }
    }

    #[test]
    fn v5_creates_prompt_history_table_with_indexes() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        assert!(table_exists(&conn, "prompt_history"));
        let cols = column_names(&conn, "prompt_history");
        for expected in &["id", "prompt", "negative", "model", "created_at_ms"] {
            assert!(
                cols.iter().any(|c| c == *expected),
                "prompt_history missing column {expected}; got {cols:?}"
            );
        }
        assert!(
            index_exists(&conn, "idx_prompt_hist_created"),
            "missing created-desc index on prompt_history"
        );
        assert!(
            index_exists(&conn, "idx_prompt_hist_model"),
            "missing model index on prompt_history"
        );
    }

    /// Upgrading a v2 DB with existing `generations` rows must not clobber
    /// those rows. The whole point of additive migrations is that prod data
    /// survives a version bump.
    #[test]
    fn upgrade_from_v2_preserves_generations_table() {
        // Manually seed a v2 DB (v1 schema + v2 user_version).
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch("PRAGMA user_version = 2;").unwrap();
        tx.commit().unwrap();

        // Seed a representative row.
        conn.execute(
            "INSERT INTO generations (filename, output_dir, created_at_ms, format, model, prompt)
             VALUES ('legacy.png', '/out', 1000, 'png', 'flux-dev:q4', 'a cat')",
            [],
        )
        .unwrap();

        // Apply the pending v3/v4/v5/v6 migrations.
        apply_pending(&mut conn).unwrap();
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);

        // Original row intact.
        let prompt: String = conn
            .query_row(
                "SELECT prompt FROM generations WHERE filename = 'legacy.png'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(prompt, "a cat");

        // New tables exist and are empty.
        assert!(table_exists(&conn, "settings"));
        assert!(table_exists(&conn, "model_prefs"));
        assert!(table_exists(&conn, "prompt_history"));
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM settings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    /// A migration whose SQL is malformed must not advance the version.
    #[test]
    fn failed_migration_rolls_back_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let before = current_version(&conn).unwrap();

        let tx = conn.transaction().unwrap();
        let err = tx.execute_batch("THIS IS NOT SQL;");
        assert!(err.is_err());
        // `tx` drops here and rolls back. Version should stay put.
        drop(tx);
        assert_eq!(current_version(&conn).unwrap(), before);
    }
}

#[cfg(test)]
mod v9_tests {
    //! v7 + v8 added the `catalog` + `catalog_fts` tables; v9 dropped
    //! them once the SPA, CLI, and server moved to live HF/Civitai.
    //! These tests pin the drop so a future re-add doesn't silently
    //! reintroduce the bulk-scrape DB.

    use super::*;
    use rusqlite::Connection;

    #[test]
    fn schema_version_is_current() {
        assert_eq!(SCHEMA_VERSION, 34);
    }

    #[test]
    fn fresh_db_does_not_have_catalog_tables() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_pending(&mut conn).unwrap();
        let tables: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name LIKE 'catalog%'",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(
            tables.is_empty(),
            "v9 must drop catalog* tables, found: {tables:?}"
        );
    }

    /// Forward-only migration from a pre-v9 DB: a v8 install with rows
    /// in the catalog table must end up at v9 with the table gone.
    #[test]
    fn v8_to_v9_drops_catalog_data() {
        let mut conn = Connection::open_in_memory().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        tx.execute_batch(V3_SETTINGS_TABLE).unwrap();
        tx.execute_batch(V4_MODEL_PREFS_TABLE).unwrap();
        tx.execute_batch(V5_PROMPT_HISTORY_TABLE).unwrap();
        tx.execute_batch(V6_PROFILE_SCOPING).unwrap();
        tx.execute_batch(V7_CATALOG_TABLE).unwrap();
        tx.execute_batch(V8_CATALOG_TRAINED_WORDS).unwrap();
        tx.execute_batch("PRAGMA user_version = 8;").unwrap();
        tx.commit().unwrap();
        conn.execute(
            "INSERT INTO catalog (id, source, source_id, name, family, family_role, modality, kind, file_format, bundling, download_recipe, supported, added_at)
             VALUES ('hf:legacy', 'hf', 'legacy', 'L', 'flux', 'foundation', 'image', 'checkpoint', 'safetensors', 'separated', '{}', 1, 0)",
            [],
        ).unwrap();

        apply_pending(&mut conn).unwrap();
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='catalog'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(exists, 0, "catalog table must be dropped after v9");
    }
}

#[cfg(test)]
mod v15_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn v14_runtime_migration_preserves_old_rows_with_null_runtime() {
        let mut conn = Connection::open_in_memory().unwrap();
        // Every real v14 database carries the gallery table; v20 alters it.
        conn.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        conn.execute_batch(V10_GENERATION_METADATA_JSON).unwrap();
        conn.execute_batch(V13_SCHEDULER_ESTIMATES).unwrap();
        conn.execute_batch(V14_SCHEDULER_ESTIMATE_EVIDENCE).unwrap();
        conn.execute(
            "INSERT INTO scheduler_estimates (
                estimate_key, device_class, model_fingerprint, work_kind,
                shape_bucket, execution_fingerprint, sample_count,
                ewma_total_ms, ewma_load_ms, last_observed_at
             ) VALUES ('legacy', 'cuda:sm86', 'flux', 'generation',
                '512x512', 'bf16', 4, 1200.0, 200.0, 100)",
            [],
        )
        .unwrap();
        conn.execute_batch("PRAGMA user_version = 14;").unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        let runtime: Option<f64> = conn
            .query_row(
                "SELECT ewma_runtime_ms FROM scheduler_estimates WHERE estimate_key = 'legacy'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(runtime, None);
    }
}

#[cfg(test)]
mod v17_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn v16_av_phase_migration_preserves_legacy_vae_evidence() {
        let mut conn = Connection::open_in_memory().unwrap();
        // Every real v16 database carries the gallery table; v20 alters it.
        conn.execute_batch(V1_INITIAL_SCHEMA).unwrap();
        conn.execute_batch(V10_GENERATION_METADATA_JSON).unwrap();
        conn.execute_batch(V13_SCHEDULER_ESTIMATES).unwrap();
        conn.execute_batch(V14_SCHEDULER_ESTIMATE_EVIDENCE).unwrap();
        conn.execute_batch(V15_SCHEDULER_ESTIMATE_RUNTIME).unwrap();
        conn.execute_batch(V16_PAIRED_CLIENTS).unwrap();
        conn.execute(
            "INSERT INTO scheduler_estimates (
                estimate_key, device_class, model_fingerprint, work_kind,
                shape_bucket, execution_fingerprint, sample_count,
                ewma_total_ms, ewma_vae_ms, last_observed_at
             ) VALUES ('legacy-vae', 'cuda:sm86', 'wan', 'generation',
                '544x960', 'bf16', 4, 1200.0, 321.0, 100)",
            [],
        )
        .unwrap();
        conn.execute_batch("PRAGMA user_version = 16;").unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        let phases: (Option<f64>, Option<f64>, Option<f64>, Option<f64>) = conn
            .query_row(
                "SELECT ewma_vae_ms, ewma_visual_decode_ms,
                        ewma_audio_decode_ms, ewma_mux_ms
                 FROM scheduler_estimates WHERE estimate_key = 'legacy-vae'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(phases, (Some(321.0), None, None, None));
    }
}

#[cfg(test)]
mod v25_tests {
    use super::*;
    use rusqlite::Connection;

    fn migrate_through(conn: &mut Connection, version: i64) {
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= version)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(function) => function(&tx).unwrap(),
            }
            tx.pragma_update(None, "user_version", migration.version)
                .unwrap();
            tx.commit().unwrap();
        }
    }

    #[test]
    fn v25_backfills_exact_job_projection_and_dirties_old_writer_updates() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate_through(&mut conn, 24);
        for (filename, metadata) in [
            ("valid.png", Some(r#"{"job_id":"job-a"}"#)),
            ("absent.png", Some(r#"{"seed":7}"#)),
            ("malformed.png", Some(r#"{"job_id":"broken""#)),
            ("nonstring.png", Some(r#"{"job_id":7}"#)),
            (
                "duplicate.png",
                Some(r#"{"job_id":"first","job_id":"last"}"#),
            ),
            ("none.png", None),
        ] {
            conn.execute(
                "INSERT INTO generations
                    (filename, output_dir, created_at_ms, format, model, metadata_json)
                 VALUES (?1, '/gallery', 1, 'png', 'model', ?2)",
                rusqlite::params![filename, metadata],
            )
            .unwrap();
        }

        apply_pending(&mut conn).unwrap();
        let projected = |filename: &str| {
            conn.query_row(
                "SELECT queue_job_id, queue_job_metadata_state
                   FROM generations WHERE filename = ?1",
                [filename],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                    ))
                },
            )
            .unwrap()
        };
        assert_eq!(projected("valid.png"), (Some("job-a".into()), Some(1)));
        assert_eq!(projected("absent.png"), (None, Some(1)));
        assert_eq!(projected("malformed.png"), (None, Some(0)));
        assert_eq!(projected("nonstring.png"), (None, Some(0)));
        assert_eq!(projected("duplicate.png"), (Some("last".into()), Some(1)));
        assert_eq!(projected("none.png"), (None, Some(1)));

        conn.execute(
            "UPDATE generations SET metadata_json = ?1 WHERE filename = 'valid.png'",
            [r#"{"job_id":"old-writer"}"#],
        )
        .unwrap();
        assert_eq!(
            projected("valid.png"),
            (None, None),
            "an older writer must make the projection explicitly reparsable, never stale"
        );
    }
}

#[cfg(test)]
mod v26_tests {
    use super::*;
    use rusqlite::Connection;

    fn column_names(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn migrate_through(conn: &mut Connection, version: i64) {
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= version)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(function) => function(&tx).unwrap(),
            }
            tx.pragma_update(None, "user_version", migration.version)
                .unwrap();
            tx.commit().unwrap();
        }
    }

    fn insert_legacy_queue_row(conn: &Connection, id: &str) {
        conn.execute(
            "INSERT INTO generation_queue (
                id, owner_uuid, state, model, request_json, output_dir,
                completion_payload, created_at, updated_at
             ) VALUES (?1, 'owner', 'queued', 'model', ?2, '/gallery', ?3, 7, 7)",
            rusqlite::params![id, r#"{"prompt":"unchanged"}"#, "metadata_only"],
        )
        .unwrap();
    }

    fn assert_legacy_row_is_payload_identical(conn: &Connection, id: &str) {
        let row: (String, String, String, Option<String>) = conn
            .query_row(
                "SELECT request_json, output_dir, completion_payload, media_set_id
                   FROM generation_queue WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            (
                r#"{"prompt":"unchanged"}"#.to_string(),
                "/gallery".to_string(),
                "metadata_only".to_string(),
                None,
            )
        );
        let obligations: i64 = conn
            .query_row("SELECT COUNT(*) FROM generation_queue_media", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(obligations, 0, "legacy rows must not invent media work");
    }

    #[test]
    fn fresh_v26_schema_is_opaque_and_payload_free() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(
            column_names(&conn, "generation_queue_media"),
            [
                "media_set_id",
                "owner_uuid",
                "state",
                "created_at_ms",
                "updated_at_ms",
            ]
        );
        assert!(column_names(&conn, "generation_queue")
            .iter()
            .any(|column| column == "media_set_id"));
        for invalid in [None, Some("")] {
            assert!(conn
                .execute(
                    "INSERT INTO generation_queue_media
                        (media_set_id, owner_uuid, state, created_at_ms, updated_at_ms)
                     VALUES (?1, 'owner', 'active', 1, 1)",
                    [invalid],
                )
                .is_err());
        }
        let trigger: String = conn
            .query_row(
                "SELECT sql FROM sqlite_schema
                  WHERE type = 'trigger' AND name = 'generation_queue_media_retire'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(trigger.contains("AFTER DELETE ON generation_queue"));

        let normalized = conn
            .query_row(
                "SELECT sql FROM sqlite_schema
                  WHERE type = 'table' AND name = 'generation_queue_media'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
            .to_ascii_lowercase();
        for forbidden in [
            "path", "byte", "digest", "role", "name", "member", "payload", "request",
        ] {
            assert!(
                !normalized.contains(forbidden),
                "queue-media schema leaked forbidden field {forbidden}: {normalized}"
            );
        }
    }

    #[test]
    fn shipped_v23_upgrades_through_v26_without_touching_legacy_payloads() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        migrate_through(&mut conn, 23);
        insert_legacy_queue_row(&conn, "from-v23");

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_legacy_row_is_payload_identical(&conn, "from-v23");
    }

    #[test]
    fn feature_v25_upgrades_to_v26_without_touching_legacy_payloads() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        migrate_through(&mut conn, 25);
        insert_legacy_queue_row(&conn, "from-v25");

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_legacy_row_is_payload_identical(&conn, "from-v25");
    }
}

#[cfg(test)]
mod v27_tests {
    use super::*;
    use rusqlite::Connection;

    fn migrate_through(conn: &mut Connection, version: i64) {
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= version)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(function) => function(&tx).unwrap(),
            }
            tx.pragma_update(None, "user_version", migration.version)
                .unwrap();
            tx.commit().unwrap();
        }
    }

    #[test]
    fn v26_upgrade_adds_owner_order_index_without_rewriting_queue_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate_through(&mut conn, 26);
        for (id, owner) in [
            ("first", "owner-a"),
            ("second", "owner-a"),
            ("foreign", "owner-b"),
        ] {
            conn.execute(
                "INSERT INTO generation_queue (
                    id, owner_uuid, state, model, request_json, output_dir,
                    completion_payload, created_at, updated_at
                 ) VALUES (?1, ?2, 'queued', 'model', ?3, '/gallery', 'metadata_only', 7, 7)",
                rusqlite::params![id, owner, r#"{"prompt":"unchanged"}"#],
            )
            .unwrap();
        }

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        let index_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_schema
                  WHERE type = 'index' AND name = 'generation_queue_owner_order'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(index_sql.contains("owner_uuid, created_at"));
        let rows = conn
            .prepare(
                "SELECT id, owner_uuid, request_json, created_at, rowid
                   FROM generation_queue ORDER BY rowid",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].0, "first");
        assert_eq!(rows[1].0, "second");
        assert_eq!(rows[2].1, "owner-b");
        assert!(rows.iter().all(|(_, _, request, created_at, _)| request
            == r#"{"prompt":"unchanged"}"#
            && *created_at == 7));
        assert!(rows.windows(2).all(|pair| pair[0].4 < pair[1].4));
    }
}

#[cfg(test)]
mod v32_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn v31_upgrade_marks_existing_paused_rows_as_restart_recovery() {
        let mut conn = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 31)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(function) => function(&tx).unwrap(),
            }
            tx.pragma_update(None, "user_version", migration.version)
                .unwrap();
            tx.commit().unwrap();
        }
        conn.execute(
            "INSERT INTO generation_queue (
                id, owner_uuid, state, model, request_json, output_dir,
                completion_payload, created_at, updated_at
             ) VALUES ('paused', 'owner', 'paused', 'model', '{}', '/gallery', 'full', 1, 1)",
            [],
        )
        .unwrap();

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(
            conn.query_row(
                "SELECT explicitly_paused FROM generation_queue WHERE id = 'paused'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            0
        );
    }
}

#[cfg(test)]
mod v34_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn current_main_v33_upgrades_without_colliding_and_adds_video_jobs() {
        let mut conn = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= 33)
        {
            let tx = conn.transaction().unwrap();
            match migration.kind {
                MigrationKind::Sql(sql) => tx.execute_batch(sql).unwrap(),
                MigrationKind::Rust(function) => function(&tx).unwrap(),
            }
            tx.pragma_update(None, "user_version", migration.version)
                .unwrap();
            tx.commit().unwrap();
        }

        assert_eq!(current_version(&conn).unwrap(), 33);
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='gallery_media_sets'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1,
        );

        apply_pending(&mut conn).unwrap();

        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='video_upscale_jobs'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1,
        );
    }
}
