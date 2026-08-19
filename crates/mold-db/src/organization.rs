//! Library organization: titles, favorites, tags, and collections.
//!
//! Everything here is user-owned state layered over the gallery rows in
//! `generations`. Identity is the same `(output_dir, filename)` pair the
//! gallery uses; side tables (`tags`, `generation_tags`, `collections`,
//! `collection_items`) FK `generations(id) ON DELETE CASCADE`, so a hard
//! delete of a print drops its memberships with it.
//!
//! Tag names are matched case-insensitively (the `tags.name` column is
//! `COLLATE NOCASE`, which folds ASCII only) and are normalized before they
//! reach SQLite: trimmed, interior whitespace runs collapsed to one space,
//! at most [`MAX_TAG_CHARS`] characters, no control characters. Empty names
//! are ignored. Tags that end up with zero uses are garbage-collected.
//!
//! Collections are identified by a UUID `id`; their `slug` (derived from the
//! name by [`collection_slug`]) is what clients use to merge collections of
//! the same name across hosts, and it is unique per DB.
//!
//! Errors are typed ([`OrganizationError`]) so the server can map them:
//! `NotFound` → 404, `Conflict` → 409, `Invalid` → 422, `Db` → 500.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::path::canonical_dir_string;
use crate::MetadataDb;

/// Longest tag name accepted, in characters (after normalization).
pub const MAX_TAG_CHARS: usize = 64;
/// Longest collection name accepted, in characters (after whitespace
/// normalization).
pub const MAX_COLLECTION_NAME_CHARS: usize = 120;
/// Longest collection slug produced by [`collection_slug`].
pub const MAX_COLLECTION_SLUG_CHARS: usize = 80;

/// Typed outcome of an organization mutation.
#[derive(Debug, thiserror::Error)]
pub enum OrganizationError {
    /// The print (by `(output_dir, filename)`), collection, or tag does not
    /// exist. Maps to HTTP 404.
    #[error("not found")]
    NotFound,
    /// A uniqueness rule would be violated (a collection slug already
    /// taken by another collection). Maps to HTTP 409.
    #[error("conflict: {0}")]
    Conflict(String),
    /// The input is malformed (empty collection name, over-long tag, control
    /// characters, ...). Maps to HTTP 422.
    #[error("invalid: {0}")]
    Invalid(String),
    /// An underlying SQLite failure. Maps to HTTP 500.
    #[error("metadata DB error: {0}")]
    Db(#[from] rusqlite::Error),
}

pub type OrgResult<T> = Result<T, OrganizationError>;

/// Per-print organization state, joined in Rust from the side tables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrintOrganization {
    pub title: Option<String>,
    pub favorite: bool,
    /// Tag names, sorted case-insensitively.
    pub tags: Vec<String>,
    /// Collection ids (UUIDs) the print belongs to, sorted.
    pub collections: Vec<String>,
    pub trashed_at_ms: Option<i64>,
}

/// One row of [`MetadataDb::list_tags`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagCountRow {
    pub name: String,
    /// Number of prints carrying the tag (trashed ones included).
    pub count: u64,
}

/// One collection plus its item count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionRow {
    pub id: String,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub cover_filename: Option<String>,
    /// Number of prints in the collection.
    pub count: u64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// Bulk mutation applied to a set of prints in one transaction. `None`
/// fields are left untouched.
#[derive(Debug, Clone, Default)]
pub struct BulkOrganize<'a> {
    pub favorite: Option<bool>,
    pub add_tags: Option<&'a [String]>,
    pub remove_tags: Option<&'a [String]>,
    /// Collection ids.
    pub add_to_collections: Option<&'a [String]>,
    /// Collection ids.
    pub remove_from_collections: Option<&'a [String]>,
}

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

/// Normalize a tag name: trim, collapse interior whitespace runs to a single
/// space. Returns `None` for an empty result (callers ignore those) and an
/// `Invalid` error for control characters or over-long names.
pub fn normalize_tag_name(raw: &str) -> OrgResult<Option<String>> {
    // Whitespace controls (tab, newline) are collapsed below; anything else
    // (NUL, escape, ...) has no place in a tag.
    if raw.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return Err(OrganizationError::Invalid(
            "tag names must not contain control characters".into(),
        ));
    }
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return Ok(None);
    }
    if collapsed.chars().count() > MAX_TAG_CHARS {
        return Err(OrganizationError::Invalid(format!(
            "tag names must be at most {MAX_TAG_CHARS} characters"
        )));
    }
    Ok(Some(collapsed))
}

/// Normalize a list of tag names, dropping empties and case-insensitive
/// duplicates while preserving first-seen order.
fn normalize_tag_list(raw: &[String]) -> OrgResult<Vec<String>> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for name in raw {
        if let Some(n) = normalize_tag_name(name)? {
            if seen.insert(n.to_lowercase()) {
                out.push(n);
            }
        }
    }
    Ok(out)
}

/// Slug for a collection name: lowercase ASCII, `[a-z0-9]` kept, every
/// other character becomes `-`, runs collapsed, ends trimmed, at most
/// [`MAX_COLLECTION_SLUG_CHARS`]. Same algorithm as `mold_core::title_slug`
/// with a longer cap. `None` when nothing survives.
pub fn collection_slug(name: &str) -> Option<String> {
    let mut slug = String::with_capacity(name.len());
    let mut pending_dash = false;
    for ch in name.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(lowered);
        } else {
            pending_dash = true;
        }
        if slug.len() >= MAX_COLLECTION_SLUG_CHARS {
            break;
        }
    }
    let slug: String = slug.chars().take(MAX_COLLECTION_SLUG_CHARS).collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        None
    } else {
        Some(slug)
    }
}

fn validate_collection_name(raw: &str) -> OrgResult<(String, String)> {
    if raw.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return Err(OrganizationError::Invalid(
            "collection names must not contain control characters".into(),
        ));
    }
    let name = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        return Err(OrganizationError::Invalid(
            "collection name must not be empty".into(),
        ));
    }
    if name.chars().count() > MAX_COLLECTION_NAME_CHARS {
        return Err(OrganizationError::Invalid(format!(
            "collection names must be at most {MAX_COLLECTION_NAME_CHARS} characters"
        )));
    }
    let slug = collection_slug(&name).ok_or_else(|| {
        OrganizationError::Invalid(
            "collection name must contain at least one letter or digit".into(),
        )
    })?;
    Ok((name, slug))
}

fn normalize_title(title: Option<&str>) -> Option<String> {
    title
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

fn now_ms() -> i64 {
    mold_core::time::now_epoch_ms()
}

fn sort_case_insensitive(names: &mut [String]) {
    names.sort_by(|a, b| {
        a.to_lowercase()
            .cmp(&b.to_lowercase())
            .then_with(|| a.cmp(b))
    });
}

// ---------------------------------------------------------------------------
// Connection-level primitives (shared with reconcile's trash import)
// ---------------------------------------------------------------------------

/// Resolve a print's row id, or `NotFound`.
pub(crate) fn generation_id(conn: &Connection, dir_key: &str, filename: &str) -> OrgResult<i64> {
    conn.query_row(
        "SELECT id FROM generations WHERE output_dir = ?1 AND filename = ?2",
        params![dir_key, filename],
        |r| r.get::<_, i64>(0),
    )
    .optional()?
    .ok_or(OrganizationError::NotFound)
}

/// Resolve every filename to its row id, failing on the first unknown one.
fn generation_ids(conn: &Connection, dir_key: &str, filenames: &[String]) -> OrgResult<Vec<i64>> {
    let mut seen = HashSet::new();
    let mut ids = Vec::with_capacity(filenames.len());
    for filename in filenames {
        if !seen.insert(filename.as_str()) {
            continue;
        }
        ids.push(generation_id(conn, dir_key, filename)?);
    }
    Ok(ids)
}

/// Find-or-create a tag by (already normalized) name; returns its id.
pub(crate) fn ensure_tag(conn: &Connection, name: &str, now_ms: i64) -> OrgResult<i64> {
    conn.execute(
        "INSERT INTO tags (name, created_at_ms) VALUES (?1, ?2)
         ON CONFLICT(name) DO NOTHING",
        params![name, now_ms],
    )?;
    // `tags.name` is COLLATE NOCASE, so this lookup folds case.
    let id = conn.query_row("SELECT id FROM tags WHERE name = ?1", params![name], |r| {
        r.get::<_, i64>(0)
    })?;
    Ok(id)
}

/// Attach (already normalized) tags to a generation. Idempotent.
pub(crate) fn attach_tags(
    conn: &Connection,
    generation_id: i64,
    names: &[String],
    now_ms: i64,
) -> OrgResult<()> {
    for name in names {
        let tag_id = ensure_tag(conn, name, now_ms)?;
        conn.execute(
            "INSERT INTO generation_tags (generation_id, tag_id) VALUES (?1, ?2)
             ON CONFLICT DO NOTHING",
            params![generation_id, tag_id],
        )?;
    }
    Ok(())
}

fn detach_tags(conn: &Connection, generation_id: i64, names: &[String]) -> OrgResult<()> {
    for name in names {
        conn.execute(
            "DELETE FROM generation_tags
             WHERE generation_id = ?1
               AND tag_id IN (SELECT id FROM tags WHERE name = ?2)",
            params![generation_id, name],
        )?;
    }
    Ok(())
}

/// Drop tags that no print references any more.
pub(crate) fn gc_orphan_tags(conn: &Connection) -> OrgResult<usize> {
    Ok(conn.execute(
        "DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM generation_tags)",
        [],
    )?)
}

fn collection_exists(conn: &Connection, id: &str) -> OrgResult<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM collections WHERE id = ?1",
        params![id],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

/// Resolve a collection id from its slug, if such a collection exists.
pub(crate) fn collection_id_for_slug(conn: &Connection, slug: &str) -> OrgResult<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT id FROM collections WHERE slug = ?1",
            params![slug],
            |r| r.get::<_, String>(0),
        )
        .optional()?)
}

/// Append generations to a collection (idempotent; positions continue after
/// the current max). Returns how many rows were newly added.
pub(crate) fn collection_add_ids(
    conn: &Connection,
    collection_id: &str,
    generation_ids: &[i64],
    now_ms: i64,
) -> OrgResult<usize> {
    if !collection_exists(conn, collection_id)? {
        return Err(OrganizationError::NotFound);
    }
    let mut next_position: i64 = conn.query_row(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM collection_items WHERE collection_id = ?1",
        params![collection_id],
        |r| r.get(0),
    )?;
    let mut added = 0usize;
    for gid in generation_ids {
        let n = conn.execute(
            "INSERT INTO collection_items (collection_id, generation_id, position, added_at_ms)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(collection_id, generation_id) DO NOTHING",
            params![collection_id, gid, next_position, now_ms],
        )?;
        if n > 0 {
            added += 1;
            next_position += 1;
        }
    }
    if added > 0 {
        touch_collection(conn, collection_id, now_ms)?;
    }
    Ok(added)
}

fn collection_remove_ids(
    conn: &Connection,
    collection_id: &str,
    generation_ids: &[i64],
    now_ms: i64,
) -> OrgResult<usize> {
    if !collection_exists(conn, collection_id)? {
        return Err(OrganizationError::NotFound);
    }
    let mut removed = 0usize;
    for gid in generation_ids {
        removed += conn.execute(
            "DELETE FROM collection_items WHERE collection_id = ?1 AND generation_id = ?2",
            params![collection_id, gid],
        )?;
    }
    if removed > 0 {
        // A cover that left the collection is no longer a valid cover.
        conn.execute(
            "UPDATE collections SET cover_filename = NULL
             WHERE id = ?1
               AND cover_filename IS NOT NULL
               AND cover_filename NOT IN (
                   SELECT g.filename FROM collection_items ci
                   JOIN generations g ON g.id = ci.generation_id
                   WHERE ci.collection_id = ?1)",
            params![collection_id],
        )?;
        touch_collection(conn, collection_id, now_ms)?;
    }
    Ok(removed)
}

fn touch_collection(conn: &Connection, collection_id: &str, now_ms: i64) -> OrgResult<()> {
    conn.execute(
        "UPDATE collections SET updated_at_ms = ?2 WHERE id = ?1",
        params![collection_id, now_ms],
    )?;
    Ok(())
}

fn read_collection(conn: &Connection, id: &str) -> OrgResult<Option<CollectionRow>> {
    Ok(conn
        .query_row(
            "SELECT c.id, c.name, c.slug, c.description, c.cover_filename,
                    (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id),
                    c.created_at_ms, c.updated_at_ms
             FROM collections c WHERE c.id = ?1",
            params![id],
            collection_from_row,
        )
        .optional()?)
}

fn collection_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<CollectionRow> {
    Ok(CollectionRow {
        id: r.get(0)?,
        name: r.get(1)?,
        slug: r.get(2)?,
        description: r.get(3)?,
        cover_filename: r.get(4)?,
        count: r.get::<_, i64>(5)?.max(0) as u64,
        created_at_ms: r.get(6)?,
        updated_at_ms: r.get(7)?,
    })
}

fn is_unique_violation(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(code, _)
            if code.code == rusqlite::ErrorCode::ConstraintViolation
    )
}

// ---------------------------------------------------------------------------
// MetadataDb API
// ---------------------------------------------------------------------------

impl MetadataDb {
    /// Organization state for every print in `dir`, keyed by filename. One
    /// query per table, joined in Rust. Includes trashed prints (the
    /// `trashed_at_ms` field says which).
    pub fn organization_for_dir(
        &self,
        dir: &Path,
    ) -> OrgResult<HashMap<String, PrintOrganization>> {
        let dir_key = canonical_dir_string(dir);
        self.with_conn_typed(|conn| {
            let mut by_id: HashMap<i64, String> = HashMap::new();
            let mut out: HashMap<String, PrintOrganization> = HashMap::new();
            {
                let mut stmt = conn.prepare(
                    "SELECT id, filename, title, favorite, trashed_at_ms
                     FROM generations WHERE output_dir = ?1",
                )?;
                let rows = stmt.query_map(params![dir_key], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, i64>(3)? != 0,
                        r.get::<_, Option<i64>>(4)?,
                    ))
                })?;
                for row in rows {
                    let (id, filename, title, favorite, trashed_at_ms) = row?;
                    by_id.insert(id, filename.clone());
                    out.insert(
                        filename,
                        PrintOrganization {
                            title,
                            favorite,
                            tags: Vec::new(),
                            collections: Vec::new(),
                            trashed_at_ms,
                        },
                    );
                }
            }
            {
                let mut stmt = conn.prepare(
                    "SELECT gt.generation_id, t.name
                     FROM generation_tags gt
                     JOIN tags t ON t.id = gt.tag_id
                     JOIN generations g ON g.id = gt.generation_id
                     WHERE g.output_dir = ?1",
                )?;
                let rows = stmt.query_map(params![dir_key], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (gid, name) = row?;
                    if let Some(org) = by_id.get(&gid).and_then(|f| out.get_mut(f)) {
                        org.tags.push(name);
                    }
                }
            }
            {
                let mut stmt = conn.prepare(
                    "SELECT ci.generation_id, ci.collection_id
                     FROM collection_items ci
                     JOIN generations g ON g.id = ci.generation_id
                     WHERE g.output_dir = ?1",
                )?;
                let rows = stmt.query_map(params![dir_key], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (gid, cid) = row?;
                    if let Some(org) = by_id.get(&gid).and_then(|f| out.get_mut(f)) {
                        org.collections.push(cid);
                    }
                }
            }
            for org in out.values_mut() {
                sort_case_insensitive(&mut org.tags);
                org.collections.sort();
            }
            Ok(out)
        })
    }

    /// Organization state for one print, or `None` when it has no row.
    pub fn print_organization(
        &self,
        dir: &Path,
        filename: &str,
    ) -> OrgResult<Option<PrintOrganization>> {
        let dir_key = canonical_dir_string(dir);
        self.with_conn_typed(|conn| {
            let Some((id, title, favorite, trashed_at_ms)) = conn
                .query_row(
                    "SELECT id, title, favorite, trashed_at_ms
                     FROM generations WHERE output_dir = ?1 AND filename = ?2",
                    params![dir_key, filename],
                    |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, Option<String>>(1)?,
                            r.get::<_, i64>(2)? != 0,
                            r.get::<_, Option<i64>>(3)?,
                        ))
                    },
                )
                .optional()?
            else {
                return Ok(None);
            };
            let mut tags: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT t.name FROM generation_tags gt JOIN tags t ON t.id = gt.tag_id
                     WHERE gt.generation_id = ?1",
                )?;
                let rows = stmt.query_map(params![id], |r| r.get::<_, String>(0))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            sort_case_insensitive(&mut tags);
            let mut collections: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT collection_id FROM collection_items WHERE generation_id = ?1",
                )?;
                let rows = stmt.query_map(params![id], |r| r.get::<_, String>(0))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            collections.sort();
            Ok(Some(PrintOrganization {
                title,
                favorite,
                tags,
                collections,
                trashed_at_ms,
            }))
        })
    }

    /// Set (or clear with `None` / empty) a print's title. The title is
    /// stored trimmed; length and character validation is the caller's job
    /// (`mold_core::validate_print_title`).
    pub fn set_title(&self, dir: &Path, filename: &str, title: Option<&str>) -> OrgResult<()> {
        let dir_key = canonical_dir_string(dir);
        let title = normalize_title(title);
        self.transact_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            conn.execute(
                "UPDATE generations SET title = ?2 WHERE id = ?1",
                params![id, title],
            )?;
            Ok(())
        })
    }

    /// Set a print's favorite flag.
    pub fn set_favorite(&self, dir: &Path, filename: &str, favorite: bool) -> OrgResult<()> {
        let dir_key = canonical_dir_string(dir);
        self.transact_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            conn.execute(
                "UPDATE generations SET favorite = ?2 WHERE id = ?1",
                params![id, favorite as i64],
            )?;
            Ok(())
        })
    }

    /// Replace a print's tag set. Returns the normalized, sorted tag list.
    pub fn replace_tags(
        &self,
        dir: &Path,
        filename: &str,
        tags: &[String],
    ) -> OrgResult<Vec<String>> {
        let dir_key = canonical_dir_string(dir);
        let names = normalize_tag_list(tags)?;
        let now = now_ms();
        self.transact_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            conn.execute(
                "DELETE FROM generation_tags WHERE generation_id = ?1",
                params![id],
            )?;
            attach_tags(conn, id, &names, now)?;
            gc_orphan_tags(conn)?;
            tags_for_generation(conn, id)
        })
    }

    /// Add tags to a print (idempotent). Returns the resulting tag list.
    pub fn add_tags(&self, dir: &Path, filename: &str, tags: &[String]) -> OrgResult<Vec<String>> {
        let dir_key = canonical_dir_string(dir);
        let names = normalize_tag_list(tags)?;
        let now = now_ms();
        self.transact_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            attach_tags(conn, id, &names, now)?;
            tags_for_generation(conn, id)
        })
    }

    /// Remove tags from a print (unknown names are ignored). Returns the
    /// resulting tag list.
    pub fn remove_tags(
        &self,
        dir: &Path,
        filename: &str,
        tags: &[String],
    ) -> OrgResult<Vec<String>> {
        let dir_key = canonical_dir_string(dir);
        let names = normalize_tag_list(tags)?;
        self.transact_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            detach_tags(conn, id, &names)?;
            gc_orphan_tags(conn)?;
            tags_for_generation(conn, id)
        })
    }

    /// Every tag with its use count, sorted case-insensitively by name.
    pub fn list_tags(&self) -> OrgResult<Vec<TagCountRow>> {
        self.with_conn_typed(|conn| {
            let mut stmt = conn.prepare(
                "SELECT t.name, COUNT(gt.generation_id)
                 FROM tags t LEFT JOIN generation_tags gt ON gt.tag_id = t.id
                 GROUP BY t.id",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok(TagCountRow {
                    name: r.get(0)?,
                    count: r.get::<_, i64>(1)?.max(0) as u64,
                })
            })?;
            let mut out = rows.collect::<rusqlite::Result<Vec<_>>>()?;
            out.sort_by(|a, b| {
                a.name
                    .to_lowercase()
                    .cmp(&b.name.to_lowercase())
                    .then_with(|| a.name.cmp(&b.name))
            });
            Ok(out)
        })
    }

    /// Rename a tag. When `new` already exists (case-insensitively) the two
    /// are merged: every print tagged `old` gains `new` and `old` is
    /// deleted. Renaming a tag to a different casing of itself just updates
    /// the stored name. Returns the stored name.
    pub fn rename_tag(&self, old: &str, new: &str) -> OrgResult<String> {
        let old = normalize_tag_name(old)?.ok_or(OrganizationError::NotFound)?;
        let new = normalize_tag_name(new)?
            .ok_or_else(|| OrganizationError::Invalid("tag name must not be empty".into()))?;
        self.transact_typed(|conn| {
            let old_id: i64 = conn
                .query_row("SELECT id FROM tags WHERE name = ?1", params![old], |r| {
                    r.get(0)
                })
                .optional()?
                .ok_or(OrganizationError::NotFound)?;
            let existing_new: Option<i64> = conn
                .query_row("SELECT id FROM tags WHERE name = ?1", params![new], |r| {
                    r.get(0)
                })
                .optional()?;
            match existing_new {
                Some(new_id) if new_id == old_id => {
                    // Same tag, possibly different casing.
                    conn.execute(
                        "UPDATE tags SET name = ?2 WHERE id = ?1",
                        params![old_id, new],
                    )?;
                }
                Some(new_id) => {
                    // Merge: move every link, then drop the old tag.
                    conn.execute(
                        "INSERT INTO generation_tags (generation_id, tag_id)
                         SELECT generation_id, ?2 FROM generation_tags WHERE tag_id = ?1
                         ON CONFLICT DO NOTHING",
                        params![old_id, new_id],
                    )?;
                    conn.execute("DELETE FROM tags WHERE id = ?1", params![old_id])?;
                }
                None => {
                    conn.execute(
                        "UPDATE tags SET name = ?2 WHERE id = ?1",
                        params![old_id, new],
                    )?;
                }
            }
            Ok(new.clone())
        })
    }

    /// Delete a tag everywhere. Returns `true` when it existed.
    pub fn delete_tag(&self, name: &str) -> OrgResult<bool> {
        let Some(name) = normalize_tag_name(name)? else {
            return Ok(false);
        };
        self.transact_typed(|conn| {
            let n = conn.execute("DELETE FROM tags WHERE name = ?1", params![name])?;
            Ok(n > 0)
        })
    }

    // ---- collections -------------------------------------------------------

    /// Create a collection. `name` is trimmed and must slug to something
    /// non-empty; an existing collection with the same slug is a `Conflict`.
    pub fn create_collection(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> OrgResult<CollectionRow> {
        let (name, slug) = validate_collection_name(name)?;
        let description = normalize_title(description);
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        self.transact_typed(|conn| {
            if collection_id_for_slug(conn, &slug)?.is_some() {
                return Err(OrganizationError::Conflict(format!(
                    "a collection with slug '{slug}' already exists"
                )));
            }
            let inserted = conn.execute(
                "INSERT INTO collections
                    (id, name, slug, description, cover_filename, created_at_ms, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?5)",
                params![id, name, slug, description, now],
            );
            match inserted {
                Ok(_) => {}
                Err(e) if is_unique_violation(&e) => {
                    return Err(OrganizationError::Conflict(format!(
                        "a collection with slug '{slug}' already exists"
                    )));
                }
                Err(e) => return Err(e.into()),
            }
            read_collection(conn, &id)?.ok_or(OrganizationError::NotFound)
        })
    }

    /// Every collection with its item count, newest-updated first.
    pub fn list_collections(&self) -> OrgResult<Vec<CollectionRow>> {
        self.with_conn_typed(|conn| {
            let mut stmt = conn.prepare(
                "SELECT c.id, c.name, c.slug, c.description, c.cover_filename,
                        (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id),
                        c.created_at_ms, c.updated_at_ms
                 FROM collections c
                 ORDER BY c.updated_at_ms DESC, c.created_at_ms DESC, c.name COLLATE NOCASE",
            )?;
            let rows = stmt.query_map([], collection_from_row)?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
    }

    /// One collection by id.
    pub fn get_collection(&self, id: &str) -> OrgResult<Option<CollectionRow>> {
        self.with_conn_typed(|conn| read_collection(conn, id))
    }

    /// Update a collection. `None` leaves a field alone; `Some("")` clears
    /// `description` / `cover_filename`. Renaming re-derives the slug and a
    /// slug collision is a `Conflict`. A cover must name a print in the
    /// collection.
    pub fn update_collection(
        &self,
        id: &str,
        name: Option<&str>,
        description: Option<&str>,
        cover_filename: Option<&str>,
    ) -> OrgResult<CollectionRow> {
        let renamed = name.map(validate_collection_name).transpose()?;
        let description = description.map(|d| normalize_title(Some(d)));
        let cover = cover_filename.map(|c| normalize_title(Some(c)));
        let now = now_ms();
        self.transact_typed(|conn| {
            if !collection_exists(conn, id)? {
                return Err(OrganizationError::NotFound);
            }
            if let Some((name, slug)) = &renamed {
                if let Some(other) = collection_id_for_slug(conn, slug)? {
                    if other != id {
                        return Err(OrganizationError::Conflict(format!(
                            "a collection with slug '{slug}' already exists"
                        )));
                    }
                }
                conn.execute(
                    "UPDATE collections SET name = ?2, slug = ?3 WHERE id = ?1",
                    params![id, name, slug],
                )?;
            }
            if let Some(description) = &description {
                conn.execute(
                    "UPDATE collections SET description = ?2 WHERE id = ?1",
                    params![id, description],
                )?;
            }
            if let Some(cover) = &cover {
                if let Some(filename) = cover {
                    let member: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM collection_items ci
                         JOIN generations g ON g.id = ci.generation_id
                         WHERE ci.collection_id = ?1 AND g.filename = ?2",
                        params![id, filename],
                        |r| r.get(0),
                    )?;
                    if member == 0 {
                        return Err(OrganizationError::Invalid(format!(
                            "cover '{filename}' is not in the collection"
                        )));
                    }
                }
                conn.execute(
                    "UPDATE collections SET cover_filename = ?2 WHERE id = ?1",
                    params![id, cover],
                )?;
            }
            touch_collection(conn, id, now)?;
            read_collection(conn, id)?.ok_or(OrganizationError::NotFound)
        })
    }

    /// Delete a collection (never its prints). Returns `true` when it existed.
    pub fn delete_collection(&self, id: &str) -> OrgResult<bool> {
        self.transact_typed(|conn| {
            let n = conn.execute("DELETE FROM collections WHERE id = ?1", params![id])?;
            Ok(n > 0)
        })
    }

    /// Append prints to a collection (idempotent). Unknown filenames or an
    /// unknown collection fail the whole call with `NotFound`. Returns how
    /// many were newly added.
    pub fn collection_add(&self, id: &str, dir: &Path, filenames: &[String]) -> OrgResult<usize> {
        let dir_key = canonical_dir_string(dir);
        let now = now_ms();
        self.transact_typed(|conn| {
            let ids = generation_ids(conn, &dir_key, filenames)?;
            collection_add_ids(conn, id, &ids, now)
        })
    }

    /// Remove prints from a collection. Unknown filenames fail with
    /// `NotFound`; prints that were not members are simply not counted.
    pub fn collection_remove(
        &self,
        id: &str,
        dir: &Path,
        filenames: &[String],
    ) -> OrgResult<usize> {
        let dir_key = canonical_dir_string(dir);
        let now = now_ms();
        self.transact_typed(|conn| {
            let ids = generation_ids(conn, &dir_key, filenames)?;
            collection_remove_ids(conn, id, &ids, now)
        })
    }

    /// Filenames in a collection ordered by position (insertion order).
    pub fn collection_filenames(&self, id: &str) -> OrgResult<Vec<String>> {
        self.with_conn_typed(|conn| {
            if !collection_exists(conn, id)? {
                return Err(OrganizationError::NotFound);
            }
            let mut stmt = conn.prepare(
                "SELECT g.filename FROM collection_items ci
                 JOIN generations g ON g.id = ci.generation_id
                 WHERE ci.collection_id = ?1
                 ORDER BY ci.position ASC, ci.added_at_ms ASC",
            )?;
            let rows = stmt.query_map(params![id], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
    }

    /// Apply a bulk mutation to several prints in one transaction. Any
    /// unknown filename or collection rolls the whole batch back.
    pub fn organize_bulk(
        &self,
        dir: &Path,
        filenames: &[String],
        op: BulkOrganize<'_>,
    ) -> OrgResult<()> {
        let dir_key = canonical_dir_string(dir);
        let add_tags = op.add_tags.map(normalize_tag_list).transpose()?;
        let remove_tags = op.remove_tags.map(normalize_tag_list).transpose()?;
        let now = now_ms();
        self.transact_typed(|conn| {
            let ids = generation_ids(conn, &dir_key, filenames)?;
            if let Some(favorite) = op.favorite {
                for id in &ids {
                    conn.execute(
                        "UPDATE generations SET favorite = ?2 WHERE id = ?1",
                        params![id, favorite as i64],
                    )?;
                }
            }
            if let Some(names) = &add_tags {
                for id in &ids {
                    attach_tags(conn, *id, names, now)?;
                }
            }
            if let Some(names) = &remove_tags {
                for id in &ids {
                    detach_tags(conn, *id, names)?;
                }
            }
            if let Some(collections) = op.add_to_collections {
                for cid in collections {
                    collection_add_ids(conn, cid, &ids, now)?;
                }
            }
            if let Some(collections) = op.remove_from_collections {
                for cid in collections {
                    collection_remove_ids(conn, cid, &ids, now)?;
                }
            }
            if remove_tags.is_some() {
                gc_orphan_tags(conn)?;
            }
            Ok(())
        })
    }

    /// Slugs of every collection a print belongs to, sorted. Used to build
    /// trash tombstones, which carry slugs rather than ids so a different
    /// host (or a rebuilt DB) can re-home the print by name.
    pub fn collection_slugs_for_print(&self, dir: &Path, filename: &str) -> OrgResult<Vec<String>> {
        let dir_key = canonical_dir_string(dir);
        self.with_conn_typed(|conn| {
            let id = generation_id(conn, &dir_key, filename)?;
            let mut stmt = conn.prepare(
                "SELECT c.slug FROM collection_items ci
                 JOIN collections c ON c.id = ci.collection_id
                 WHERE ci.generation_id = ?1 ORDER BY c.slug",
            )?;
            let rows = stmt.query_map(params![id], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
    }
}

fn tags_for_generation(conn: &Connection, generation_id: i64) -> OrgResult<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT t.name FROM generation_tags gt JOIN tags t ON t.id = gt.tag_id
         WHERE gt.generation_id = ?1",
    )?;
    let rows = stmt.query_map(params![generation_id], |r| r.get::<_, String>(0))?;
    let mut names = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    sort_case_insensitive(&mut names);
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{GenerationRecord, RecordSource};
    use mold_core::{OutputFormat, OutputMetadata};

    const DIR: &str = "/tmp/org-tests";

    fn meta() -> OutputMetadata {
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"an owl","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        OutputMetadata::from_generate_request(&req, 1, None, "test")
    }

    fn seed(db: &MetadataDb, names: &[&str]) {
        for name in names {
            let rec = GenerationRecord::from_save(
                Path::new(DIR),
                *name,
                OutputFormat::Png,
                meta(),
                RecordSource::Server,
                1,
            );
            db.upsert(&rec).unwrap();
        }
    }

    fn dir() -> &'static Path {
        Path::new(DIR)
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    // ---- normalization ----------------------------------------------------

    #[test]
    fn tag_names_are_trimmed_collapsed_and_bounded() {
        assert_eq!(
            normalize_tag_name("  Night   Owls \t").unwrap().as_deref(),
            Some("Night Owls")
        );
        assert_eq!(normalize_tag_name("   ").unwrap(), None);
        assert!(matches!(
            normalize_tag_name(&"x".repeat(65)),
            Err(OrganizationError::Invalid(_))
        ));
        assert!(normalize_tag_name(&"x".repeat(64)).unwrap().is_some());
        assert!(matches!(
            normalize_tag_name("bad\u{7}tag"),
            Err(OrganizationError::Invalid(_))
        ));
    }

    #[test]
    fn collection_slug_matches_title_slug_rules_with_longer_cap() {
        assert_eq!(
            collection_slug("  Night Owls: Vol. 2! ").as_deref(),
            Some("night-owls-vol-2")
        );
        assert_eq!(collection_slug("---").as_deref(), None);
        assert_eq!(collection_slug("Ünïcode"), Some("n-code".into()));
        let long = "a".repeat(200);
        assert_eq!(
            collection_slug(&long).unwrap().len(),
            MAX_COLLECTION_SLUG_CHARS
        );
        // A cap landing on a dash must still trim it.
        let mut s = "b".repeat(MAX_COLLECTION_SLUG_CHARS - 1);
        s.push_str("-zzz");
        let slug = collection_slug(&s).unwrap();
        assert!(!slug.ends_with('-'));
        assert!(slug.len() <= MAX_COLLECTION_SLUG_CHARS);
    }

    // ---- titles / favorites ------------------------------------------------

    #[test]
    fn set_title_and_favorite_round_trip_and_reject_unknown_prints() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        db.set_title(dir(), "a.png", Some("  Owl study  ")).unwrap();
        db.set_favorite(dir(), "a.png", true).unwrap();
        let org = db.print_organization(dir(), "a.png").unwrap().unwrap();
        assert_eq!(org.title.as_deref(), Some("Owl study"));
        assert!(org.favorite);

        db.set_title(dir(), "a.png", Some("")).unwrap();
        assert_eq!(
            db.print_organization(dir(), "a.png")
                .unwrap()
                .unwrap()
                .title,
            None
        );

        assert!(matches!(
            db.set_title(dir(), "missing.png", Some("x")),
            Err(OrganizationError::NotFound)
        ));
        assert!(matches!(
            db.set_favorite(dir(), "missing.png", true),
            Err(OrganizationError::NotFound)
        ));
        assert!(db
            .print_organization(dir(), "missing.png")
            .unwrap()
            .is_none());
    }

    // ---- tags -------------------------------------------------------------

    #[test]
    fn tags_are_case_insensitive_and_deduplicated() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png"]);
        let got = db
            .add_tags(dir(), "a.png", &s(&["Owls", "owls", " OWLS "]))
            .unwrap();
        assert_eq!(got, vec!["Owls"]);
        db.add_tags(dir(), "b.png", &s(&["OWLS"])).unwrap();
        let tags = db.list_tags().unwrap();
        assert_eq!(tags.len(), 1, "{tags:?}");
        assert_eq!(tags[0].name, "Owls", "first spelling wins");
        assert_eq!(tags[0].count, 2);
        // Removing with a different casing still detaches.
        let after = db.remove_tags(dir(), "b.png", &s(&["owls"])).unwrap();
        assert!(after.is_empty());
        assert_eq!(db.list_tags().unwrap()[0].count, 1);
    }

    #[test]
    fn replace_tags_sets_exact_list_and_gcs_orphans() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        db.add_tags(dir(), "a.png", &s(&["one", "two"])).unwrap();
        let got = db
            .replace_tags(dir(), "a.png", &s(&["Zeta", "alpha", ""]))
            .unwrap();
        assert_eq!(got, vec!["alpha", "Zeta"], "sorted case-insensitively");
        let names: Vec<_> = db
            .list_tags()
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["alpha", "Zeta"], "orphans one/two are gone");
        let got = db
            .remove_tags(dir(), "a.png", &s(&["alpha", "nope"]))
            .unwrap();
        assert_eq!(got, vec!["Zeta"]);
        assert_eq!(db.list_tags().unwrap().len(), 1);
    }

    #[test]
    fn rename_tag_merges_into_existing_and_delete_tag_detaches_everywhere() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png", "c.png"]);
        db.add_tags(dir(), "a.png", &s(&["cats"])).unwrap();
        db.add_tags(dir(), "b.png", &s(&["cats", "kittens"]))
            .unwrap();
        db.add_tags(dir(), "c.png", &s(&["kittens"])).unwrap();

        // Plain rename.
        assert_eq!(db.rename_tag("cats", "Felines").unwrap(), "Felines");
        assert_eq!(
            db.print_organization(dir(), "a.png").unwrap().unwrap().tags,
            vec!["Felines"]
        );
        // Case-only rename keeps the id.
        assert_eq!(db.rename_tag("felines", "FELINES").unwrap(), "FELINES");
        // Merge into an existing tag.
        assert_eq!(db.rename_tag("FELINES", "kittens").unwrap(), "kittens");
        let tags = db.list_tags().unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "kittens");
        assert_eq!(tags[0].count, 3);
        assert_eq!(
            db.print_organization(dir(), "b.png").unwrap().unwrap().tags,
            vec!["kittens"],
            "b had both; merge must not duplicate the link"
        );
        assert!(matches!(
            db.rename_tag("ghost", "x"),
            Err(OrganizationError::NotFound)
        ));
        assert!(matches!(
            db.rename_tag("kittens", "  "),
            Err(OrganizationError::Invalid(_))
        ));

        assert!(db.delete_tag("KITTENS").unwrap());
        assert!(!db.delete_tag("kittens").unwrap());
        assert!(db.list_tags().unwrap().is_empty());
        assert!(db
            .print_organization(dir(), "c.png")
            .unwrap()
            .unwrap()
            .tags
            .is_empty());
    }

    #[test]
    fn hard_deleting_a_print_cascades_its_tags_and_memberships() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        db.add_tags(dir(), "a.png", &s(&["solo"])).unwrap();
        let c = db.create_collection("Keepers", None).unwrap();
        db.collection_add(&c.id, dir(), &s(&["a.png"])).unwrap();
        assert!(db.delete(dir(), "a.png").unwrap());
        let c = db.get_collection(&c.id).unwrap().unwrap();
        assert_eq!(c.count, 0);
        assert!(db.collection_filenames(&c.id).unwrap().is_empty());
        // The tag row itself survives until the next gc; it reports zero uses.
        let tags = db.list_tags().unwrap();
        assert!(tags.iter().all(|t| t.count == 0), "{tags:?}");
    }

    // ---- collections --------------------------------------------------------

    #[test]
    fn collections_crud_with_slug_conflicts() {
        let db = MetadataDb::open_in_memory().unwrap();
        let c = db
            .create_collection("  Night Owls ", Some(" best ones "))
            .unwrap();
        assert_eq!(c.name, "Night Owls");
        assert_eq!(c.slug, "night-owls");
        assert_eq!(c.description.as_deref(), Some("best ones"));
        assert_eq!(c.count, 0);
        assert!(uuid::Uuid::parse_str(&c.id).is_ok());

        assert!(matches!(
            db.create_collection("night   owls", None),
            Err(OrganizationError::Conflict(_))
        ));
        assert!(matches!(
            db.create_collection("   ", None),
            Err(OrganizationError::Invalid(_))
        ));
        assert!(matches!(
            db.create_collection("!!!", None),
            Err(OrganizationError::Invalid(_))
        ));

        let other = db.create_collection("Day Owls", None).unwrap();
        assert!(matches!(
            db.update_collection(&other.id, Some("NIGHT-OWLS"), None, None),
            Err(OrganizationError::Conflict(_))
        ));
        // Renaming to a different casing of itself is fine.
        let renamed = db
            .update_collection(&c.id, Some("NIGHT OWLS"), Some(""), None)
            .unwrap();
        assert_eq!(renamed.name, "NIGHT OWLS");
        assert_eq!(renamed.slug, "night-owls");
        assert_eq!(renamed.description, None);

        assert_eq!(db.list_collections().unwrap().len(), 2);
        assert!(db.delete_collection(&other.id).unwrap());
        assert!(!db.delete_collection(&other.id).unwrap());
        assert!(db.get_collection(&other.id).unwrap().is_none());
        assert!(matches!(
            db.update_collection(&other.id, Some("x"), None, None),
            Err(OrganizationError::NotFound)
        ));
    }

    #[test]
    fn collection_items_keep_insertion_order_and_are_idempotent() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png", "c.png"]);
        let c = db.create_collection("Shelf", None).unwrap();
        assert_eq!(
            db.collection_add(&c.id, dir(), &s(&["b.png", "a.png"]))
                .unwrap(),
            2
        );
        assert_eq!(
            db.collection_add(&c.id, dir(), &s(&["a.png", "c.png"]))
                .unwrap(),
            1
        );
        assert_eq!(
            db.collection_filenames(&c.id).unwrap(),
            vec!["b.png", "a.png", "c.png"]
        );
        assert_eq!(db.get_collection(&c.id).unwrap().unwrap().count, 3);

        // Unknown filename rolls the whole add back.
        assert!(matches!(
            db.collection_add(&c.id, dir(), &s(&["zzz.png"])),
            Err(OrganizationError::NotFound)
        ));
        assert!(matches!(
            db.collection_add("no-such-collection", dir(), &s(&["a.png"])),
            Err(OrganizationError::NotFound)
        ));

        // Cover must be a member; removing the cover clears it.
        assert!(matches!(
            db.update_collection(&c.id, None, None, Some("nope.png")),
            Err(OrganizationError::Invalid(_))
        ));
        let with_cover = db
            .update_collection(&c.id, None, None, Some("a.png"))
            .unwrap();
        assert_eq!(with_cover.cover_filename.as_deref(), Some("a.png"));
        assert_eq!(
            db.collection_remove(&c.id, dir(), &s(&["a.png", "b.png"]))
                .unwrap(),
            2
        );
        let after = db.get_collection(&c.id).unwrap().unwrap();
        assert_eq!(after.cover_filename, None);
        assert_eq!(after.count, 1);
        assert_eq!(db.collection_filenames(&c.id).unwrap(), vec!["c.png"]);
        // Removing a non-member is a no-op, not an error.
        assert_eq!(
            db.collection_remove(&c.id, dir(), &s(&["a.png"])).unwrap(),
            0
        );

        let org = db.print_organization(dir(), "c.png").unwrap().unwrap();
        assert_eq!(org.collections, vec![c.id.clone()]);
        assert_eq!(
            db.collection_slugs_for_print(dir(), "c.png").unwrap(),
            vec!["shelf"]
        );

        // Deleting the collection never touches prints.
        assert!(db.delete_collection(&c.id).unwrap());
        assert_eq!(db.count().unwrap(), 3);
        assert!(db
            .print_organization(dir(), "c.png")
            .unwrap()
            .unwrap()
            .collections
            .is_empty());
    }

    // ---- organization_for_dir ----------------------------------------------

    #[test]
    fn organization_for_dir_joins_every_table_and_scopes_by_dir() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png"]);
        let other = GenerationRecord::from_save(
            Path::new("/tmp/elsewhere"),
            "a.png",
            OutputFormat::Png,
            meta(),
            RecordSource::Server,
            1,
        );
        db.upsert(&other).unwrap();
        db.set_title(dir(), "a.png", Some("Titled")).unwrap();
        db.set_favorite(dir(), "a.png", true).unwrap();
        db.add_tags(dir(), "a.png", &s(&["zeta", "Alpha"])).unwrap();
        let c = db.create_collection("Col", None).unwrap();
        db.collection_add(&c.id, dir(), &s(&["a.png"])).unwrap();
        assert!(db.mark_trashed(dir(), "b.png", 99).unwrap());

        let map = db.organization_for_dir(dir()).unwrap();
        assert_eq!(map.len(), 2);
        let a = &map["a.png"];
        assert_eq!(a.title.as_deref(), Some("Titled"));
        assert!(a.favorite);
        assert_eq!(a.tags, vec!["Alpha", "zeta"]);
        assert_eq!(a.collections, vec![c.id.clone()]);
        assert_eq!(a.trashed_at_ms, None);
        let b = &map["b.png"];
        assert_eq!(
            *b,
            PrintOrganization {
                trashed_at_ms: Some(99),
                ..Default::default()
            }
        );

        let elsewhere = db
            .organization_for_dir(Path::new("/tmp/elsewhere"))
            .unwrap();
        assert_eq!(elsewhere["a.png"], PrintOrganization::default());
    }

    // ---- bulk -------------------------------------------------------------

    #[test]
    fn organize_bulk_applies_everything_in_one_transaction() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png", "b.png", "c.png"]);
        db.add_tags(dir(), "a.png", &s(&["old"])).unwrap();
        let keep = db.create_collection("Keep", None).unwrap();
        let drop_c = db.create_collection("Drop", None).unwrap();
        db.collection_add(&drop_c.id, dir(), &s(&["a.png", "b.png"]))
            .unwrap();

        db.organize_bulk(
            dir(),
            &s(&["a.png", "b.png"]),
            BulkOrganize {
                favorite: Some(true),
                add_tags: Some(&s(&["new", "New"])),
                remove_tags: Some(&s(&["old"])),
                add_to_collections: Some(std::slice::from_ref(&keep.id)),
                remove_from_collections: Some(std::slice::from_ref(&drop_c.id)),
            },
        )
        .unwrap();

        let map = db.organization_for_dir(dir()).unwrap();
        for name in ["a.png", "b.png"] {
            let org = &map[name];
            assert!(org.favorite, "{name}");
            assert_eq!(org.tags, vec!["new"], "{name}");
            assert_eq!(org.collections, vec![keep.id.clone()], "{name}");
        }
        assert_eq!(map["c.png"], PrintOrganization::default());
        assert_eq!(db.get_collection(&drop_c.id).unwrap().unwrap().count, 0);
        let names: Vec<_> = db
            .list_tags()
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["new"], "orphaned 'old' was gc'd");

        // An unknown filename rolls back every effect.
        let err = db
            .organize_bulk(
                dir(),
                &s(&["c.png", "ghost.png"]),
                BulkOrganize {
                    favorite: Some(true),
                    add_tags: Some(&s(&["nope"])),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, OrganizationError::NotFound));
        let c = db.print_organization(dir(), "c.png").unwrap().unwrap();
        assert!(!c.favorite);
        assert!(c.tags.is_empty());
        assert!(db.list_tags().unwrap().iter().all(|t| t.name != "nope"));

        // An unknown collection id rolls back too.
        let err = db
            .organize_bulk(
                dir(),
                &s(&["c.png"]),
                BulkOrganize {
                    favorite: Some(true),
                    add_to_collections: Some(&s(&["missing-collection"])),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, OrganizationError::NotFound));
        assert!(
            !db.print_organization(dir(), "c.png")
                .unwrap()
                .unwrap()
                .favorite
        );
    }

    #[test]
    fn invalid_tags_fail_before_touching_the_db() {
        let db = MetadataDb::open_in_memory().unwrap();
        seed(&db, &["a.png"]);
        let long = "y".repeat(MAX_TAG_CHARS + 1);
        assert!(matches!(
            db.add_tags(dir(), "a.png", &[long]),
            Err(OrganizationError::Invalid(_))
        ));
        assert!(db.list_tags().unwrap().is_empty());
    }
}
