//! Durable replay receipts for idempotent client-side gallery mutation queues.

use anyhow::Result;
use std::path::Path;

use rusqlite::{params, OptionalExtension};

use crate::organization::{
    collection_id_for_slug, create_collection_with_conn, mutate_bulk_on_conn, normalize_tag_list,
    validate_collection_name, OrgResult, OrganizationError,
};
use crate::path::canonical_dir_string;
use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GalleryMutationReceipt {
    pub request_sha256: String,
    pub response_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GalleryMutationApply {
    Applied,
    Replayed(String),
}

impl MetadataDb {
    /// Apply the complete mutation and store its replay receipt under one
    /// `BEGIN IMMEDIATE` transaction. A conflicting operation id cannot race
    /// between receipt lookup, collection creation, print edits, and receipt
    /// publication.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_gallery_mutation_once(
        &self,
        operation_id: &str,
        request_sha256: &str,
        response_json: &str,
        created_at_ms: i64,
        dir: &Path,
        titles: &[(String, Option<String>)],
        filenames: &[String],
        favorite: Option<bool>,
        add_tags: &[String],
        remove_tags: &[String],
        add_collection_id: Option<&str>,
        add_collection_name: Option<&str>,
        remove_collection_slug: Option<&str>,
    ) -> OrgResult<GalleryMutationApply> {
        let dir_key = canonical_dir_string(dir);
        let add_tags = normalize_tag_list(add_tags)?;
        let remove_tags = normalize_tag_list(remove_tags)?;
        self.transact_typed(|conn| {
            if let Some(receipt) = conn
                .query_row(
                    "SELECT request_sha256, response_json
                     FROM gallery_mutation_receipts WHERE operation_id = ?1",
                    params![operation_id],
                    |row| {
                        Ok(GalleryMutationReceipt {
                            request_sha256: row.get(0)?,
                            response_json: row.get(1)?,
                        })
                    },
                )
                .optional()?
            {
                if receipt.request_sha256 != request_sha256 {
                    return Err(OrganizationError::Conflict(
                        "operation id was already used for a different gallery mutation".into(),
                    ));
                }
                return Ok(GalleryMutationApply::Replayed(receipt.response_json));
            }

            let now = created_at_ms;
            let mut add_ids = Vec::new();
            if let Some(id) = add_collection_id {
                add_ids.push(id.to_string());
            } else if let Some(raw_name) = add_collection_name {
                let (name, slug) = validate_collection_name(raw_name)?;
                let id = match collection_id_for_slug(conn, &slug)? {
                    Some(id) => id,
                    None => create_collection_with_conn(conn, &name, &slug, now)?,
                };
                add_ids.push(id);
            }
            let mut remove_ids = Vec::new();
            if let Some(slug) = remove_collection_slug {
                if let Some(id) = collection_id_for_slug(conn, slug)? {
                    remove_ids.push(id);
                }
            }

            mutate_bulk_on_conn(
                conn,
                &dir_key,
                titles,
                filenames,
                favorite,
                Some(&add_tags),
                Some(&remove_tags),
                Some(&add_ids),
                Some(&remove_ids),
                now,
            )?;
            conn.execute(
                "INSERT INTO gallery_mutation_receipts
                    (operation_id, request_sha256, response_json, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4)",
                params![operation_id, request_sha256, response_json, created_at_ms],
            )?;
            Ok(GalleryMutationApply::Applied)
        })
    }

    pub fn gallery_mutation_receipt(
        &self,
        operation_id: &str,
    ) -> Result<Option<GalleryMutationReceipt>> {
        self.with_conn(|conn| {
            Ok(conn
                .query_row(
                    "SELECT request_sha256, response_json
                     FROM gallery_mutation_receipts WHERE operation_id = ?1",
                    params![operation_id],
                    |row| {
                        Ok(GalleryMutationReceipt {
                            request_sha256: row.get(0)?,
                            response_json: row.get(1)?,
                        })
                    },
                )
                .optional()?)
        })
    }

    pub fn record_gallery_mutation_receipt(
        &self,
        operation_id: &str,
        request_sha256: &str,
        response_json: &str,
        created_at_ms: i64,
    ) -> Result<()> {
        self.transact_immediate(|conn| {
            conn.execute(
                "INSERT INTO gallery_mutation_receipts
                    (operation_id, request_sha256, response_json, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(operation_id) DO NOTHING",
                params![operation_id, request_sha256, response_json, created_at_ms],
            )?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_survives_reopen_and_first_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        {
            let db = MetadataDb::open(&path).unwrap();
            db.record_gallery_mutation_receipt("op-1", "hash-1", r#"{"changed":30}"#, 1)
                .unwrap();
            db.record_gallery_mutation_receipt("op-1", "hash-2", "different", 2)
                .unwrap();
        }
        let db = MetadataDb::open(&path).unwrap();
        assert_eq!(
            db.gallery_mutation_receipt("op-1").unwrap(),
            Some(GalleryMutationReceipt {
                request_sha256: "hash-1".into(),
                response_json: r#"{"changed":30}"#.into(),
            })
        );
    }
}
