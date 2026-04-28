//! DB-side catalog repository. The SPA, CLI, and server all read through
//! this — no query strings live in handler code.
//!
//! Catalog rows are global per mold install (no `profile` column). Per-
//! profile state (downloaded? favorited?) lives in existing tables.

use rusqlite::{params, Connection, Row, ToSql};

use crate::MetadataDb;

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

/// Same columns but table-qualified for queries that JOIN catalog_fts,
/// where both tables expose `name`, `author`, `description`, and `tags`.
const QUALIFIED_COLUMNS: &str =
    "catalog.id, catalog.source, catalog.source_id, catalog.name, catalog.author, \
     catalog.family, catalog.family_role, catalog.sub_family, catalog.modality, catalog.kind, \
     catalog.file_format, catalog.bundling, catalog.size_bytes, catalog.download_count, \
     catalog.rating, catalog.likes, catalog.nsfw, catalog.thumbnail_url, catalog.description, \
     catalog.license, catalog.license_flags, catalog.tags, catalog.companions, \
     catalog.download_recipe, catalog.engine_phase, catalog.created_at, catalog.updated_at, \
     catalog.added_at";

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

/// Replace every row for `family` in a single transaction. Rows for this
/// family are deleted first; the new batch is then written via
/// `INSERT OR REPLACE` so cross-family overlap (an entry the scanner now
/// classifies under `family` but that already exists under a different
/// family from a prior scan or the embedded seed shards) reassigns rather
/// than blowing up on the `UNIQUE (source, source_id)` constraint.
/// Within a single batch, duplicates by `id` (primary key) similarly
/// collapse to the last write. The FTS5 mirror is rebuilt at the end.
pub fn upsert_entries(
    conn: &Connection,
    family: &str,
    rows: &[CatalogRow],
) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute("DELETE FROM catalog WHERE family = ?1", params![family])?;
    for r in rows {
        tx.execute(
            &format!(
                "INSERT OR REPLACE INTO catalog ({COLUMNS}) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28)",
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
    tx.execute(
        "INSERT INTO catalog_fts(catalog_fts) VALUES ('rebuild')",
        [],
    )?;
    tx.commit()
}

pub fn delete_family(conn: &Connection, family: &str) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM catalog WHERE family = ?1", params![family])?;
    conn.execute(
        "INSERT INTO catalog_fts(catalog_fts) VALUES ('rebuild')",
        [],
    )?;
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
        // FTS join — use table-qualified columns to avoid ambiguity with
        // catalog_fts which also exposes name/author/description/tags.
        let inner = sql.replacen(&format!("SELECT {COLUMNS} FROM catalog WHERE "), "", 1);
        sql = format!(
            "SELECT {QUALIFIED_COLUMNS} FROM catalog \
             INNER JOIN catalog_fts ON catalog.rowid = catalog_fts.rowid \
             WHERE catalog_fts MATCH ?1 AND ({inner})",
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

// ── MetadataDb convenience methods for catalog access ─────────────────────────

impl MetadataDb {
    /// List catalog rows matching `params`. Acquires the DB lock once.
    pub fn catalog_list(&self, params: &ListParams) -> anyhow::Result<Vec<CatalogRow>> {
        self.with_conn(|conn| Ok(list(conn, params)?))
    }

    /// Fetch a single catalog row by its `id`. Returns `None` when not found.
    pub fn catalog_get(&self, id: &str) -> anyhow::Result<Option<CatalogRow>> {
        self.with_conn(|conn| Ok(get_by_id(conn, id)?))
    }

    /// Aggregate foundation/finetune counts per family.
    pub fn catalog_family_counts(&self) -> anyhow::Result<Vec<FamilyCount>> {
        self.with_conn(|conn| Ok(family_counts(conn)?))
    }

    /// Upsert a batch of catalog rows for a family.
    pub fn catalog_upsert(&self, family: &str, rows: &[CatalogRow]) -> anyhow::Result<()> {
        self.with_conn(|conn| Ok(upsert_entries(conn, family, rows)?))
    }
}
