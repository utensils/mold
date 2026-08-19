//! Library organization over HTTP: titles, favorites, tags, collections, and
//! the post-overlay enrichment that folds that state into `/api/gallery`
//! rows.
//!
//! Everything here mutates only the metadata DB — never gallery bytes — so
//! the handlers take the READ side of the gallery publication gate: they
//! never race an atomic batch publication (which holds the writer) and they
//! never block ordinary listings. Trash, which moves files, lives in
//! [`crate::gallery_trash`] and takes the writer.
//!
//! Errors from `mold_db::organization` map one-to-one: `NotFound` → 404,
//! `Conflict` → 409, `Invalid` → 422, `Db` → 500. With the metadata DB
//! disabled every route answers 501 and `GET /api/capabilities` advertises
//! `gallery.organize = false`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    Json,
};
use mold_core::{
    Collection, CollectionCreateRequest, CollectionDetail, CollectionItemsRequest,
    CollectionUpdateRequest, GalleryImage, GalleryOrganizeRequest, GalleryPatchRequest,
    ServerEvent, TagCount, TagRenameRequest,
};
use mold_db::{CollectionRow, MetadataDb, OrganizationError, PrintOrganization};

use crate::batch_transaction::CommittedArchiveIndex;
use crate::routes::ApiError;
use crate::state::AppState;

/// Error code for organization routes on a host whose metadata DB is
/// disabled (`MOLD_DB_DISABLE=1`).
pub(crate) const GALLERY_ORGANIZATION_UNAVAILABLE: &str = "GALLERY_ORGANIZATION_UNAVAILABLE";

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Resolve the gallery directory, or the same 404 every gallery route returns
/// when output is disabled.
pub(crate) async fn gallery_output_dir(state: &AppState) -> Result<PathBuf, ApiError> {
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    Ok(config.effective_output_dir())
}

/// The effective `gallery.trash_retention_days` (env overrides DB/config).
pub(crate) async fn current_retention_days(state: &AppState) -> u32 {
    state
        .config
        .read()
        .await
        .gallery
        .effective_trash_retention_days()
}

/// The metadata DB, or the 501 that every organization route returns when
/// it is disabled.
pub(crate) fn require_metadata_db(db: &Arc<Option<MetadataDb>>) -> Result<&MetadataDb, ApiError> {
    db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "library organization requires the metadata DB, which is disabled on this host",
            GALLERY_ORGANIZATION_UNAVAILABLE,
            StatusCode::NOT_IMPLEMENTED,
        )
    })
}

/// Reject anything that is not one bare filename component.
pub(crate) fn clean_gallery_filename(filename: &str) -> Result<String, ApiError> {
    let clean = Path::new(filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    if clean.is_empty() || clean != filename {
        return Err(ApiError::validation("invalid filename"));
    }
    Ok(clean)
}

pub(crate) fn map_org_error(error: OrganizationError) -> ApiError {
    match error {
        OrganizationError::NotFound => ApiError::with_code(
            "gallery print, collection, or tag not found",
            "GALLERY_NOT_FOUND",
            StatusCode::NOT_FOUND,
        ),
        OrganizationError::Conflict(message) => ApiError::with_code(
            message,
            "GALLERY_ORGANIZATION_CONFLICT",
            StatusCode::CONFLICT,
        ),
        OrganizationError::Invalid(message) => ApiError::validation(message),
        OrganizationError::Db(error) => {
            ApiError::internal(format!("metadata DB organization query failed: {error}"))
        }
    }
}

fn org_not_found(what: &str) -> ApiError {
    ApiError::with_code(what, "GALLERY_NOT_FOUND", StatusCode::NOT_FOUND)
}

/// Fold one print's organization state into its wire row. `purge_at` is
/// derived from the retention in force right now, never stored.
pub(crate) fn apply_organization(
    image: &mut GalleryImage,
    org: Option<&PrintOrganization>,
    retention_days: u32,
) {
    let Some(org) = org else {
        // No row: the archive/scan row stands on its own.
        return;
    };
    image.title = org.title.clone().or_else(|| image.metadata.title.clone());
    image.tags = org.tags.clone();
    image.favorite = org.favorite;
    image.collections = org.collections.clone();
    image.trashed_at = org.trashed_at_ms.map(|ms| (ms / 1000).max(0) as u64);
    image.purge_at = org
        .trashed_at_ms
        .and_then(|ms| mold_db::purge_at_ms(ms, retention_days))
        .map(|ms| (ms / 1000).max(0) as u64);
}

/// Post-overlay enrichment for `GET /api/gallery?view=library`: applies
/// organization by filename and drops every trashed row, even one the
/// committed-archive overlay re-added.
pub(crate) fn enrich_library_listing(
    images: Vec<GalleryImage>,
    org: &HashMap<String, PrintOrganization>,
    retention_days: u32,
) -> Vec<GalleryImage> {
    images
        .into_iter()
        .filter_map(|mut image| {
            let print = org.get(&image.filename);
            if print.is_some_and(|p| p.trashed_at_ms.is_some()) {
                return None;
            }
            apply_organization(&mut image, print, retention_days);
            Some(image)
        })
        .collect()
}

/// Build the enriched wire row for one print: the committed archive record
/// when one is live (matching the listing's preference), else the DB row,
/// with organization folded in. `None` when the print has no DB row.
pub(crate) fn enriched_gallery_image(
    db: &MetadataDb,
    archive: Option<&CommittedArchiveIndex>,
    dir: &Path,
    filename: &str,
    retention_days: u32,
) -> anyhow::Result<Option<GalleryImage>> {
    let Some(row) = db.get(dir, filename)? else {
        return Ok(None);
    };
    let mut image = archive
        .and_then(|index| index.get(filename))
        .map(|entry| entry.record().to_gallery_image())
        .unwrap_or_else(|| row.to_gallery_image());
    let org = db
        .print_organization(dir, filename)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    apply_organization(&mut image, org.as_ref(), retention_days);
    Ok(Some(image))
}

fn collection_to_wire(row: CollectionRow) -> Collection {
    Collection {
        id: row.id,
        name: row.name,
        slug: row.slug,
        description: row.description,
        cover_filename: row.cover_filename,
        count: row.count,
        created_at: (row.created_at_ms / 1000).max(0) as u64,
        updated_at: (row.updated_at_ms / 1000).max(0) as u64,
    }
}

fn tag_to_wire(row: mold_db::TagCountRow) -> TagCount {
    TagCount {
        name: row.name,
        count: row.count,
    }
}

/// Filenames in `dir` currently carrying `tag` (case-insensitive), read
/// BEFORE a rename/delete so the right `gallery_updated` events fire.
fn filenames_with_tag(db: &MetadataDb, dir: &Path, tag: &str) -> Result<Vec<String>, ApiError> {
    let wanted = tag.trim().to_lowercase();
    let org = db.organization_for_dir(dir).map_err(map_org_error)?;
    let mut names: Vec<String> = org
        .into_iter()
        .filter(|(_, print)| print.tags.iter().any(|t| t.to_lowercase() == wanted))
        .map(|(filename, _)| filename)
        .collect();
    names.sort();
    Ok(names)
}

// ── PATCH /api/gallery/image/:filename ──────────────────────────────────────

/// Edit one print's title, favorite flag, and tags.
#[utoipa::path(
    patch,
    path = "/api/gallery/image/{filename}",
    tag = "gallery",
    params(("filename" = String, Path, description = "Gallery filename")),
    request_body = GalleryPatchRequest,
    responses(
        (status = 200, description = "The refreshed gallery entry (GalleryImage) with title, tags, favorite, collections"),
        (status = 404, description = "No gallery row for the filename"),
        (status = 422, description = "Invalid filename, title, or tag"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn patch_gallery_image(
    State(state): State<AppState>,
    AxumPath(filename): AxumPath<String>,
    Json(request): Json<GalleryPatchRequest>,
) -> Result<Json<GalleryImage>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let dir = gallery_output_dir(&state).await?;
    let name = clean_gallery_filename(&filename)?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let title = request
        .title
        .as_deref()
        .map(mold_core::validate_print_title)
        .transpose()
        .map_err(ApiError::validation)?;
    let retention = current_retention_days(&state).await;
    let archive = state.gallery_publication_gate.clone();
    let image = tokio::task::spawn_blocking(move || -> Result<GalleryImage, ApiError> {
        let db = require_metadata_db(&db)?;
        if db
            .get(&dir, &name)
            .map_err(|e| ApiError::internal(format!("metadata DB read failed: {e:#}")))?
            .is_none()
        {
            return Err(org_not_found("gallery print not found"));
        }
        if let Some(title) = &title {
            db.set_title(&dir, &name, title.as_deref())
                .map_err(map_org_error)?;
        }
        if let Some(favorite) = request.favorite {
            db.set_favorite(&dir, &name, favorite)
                .map_err(map_org_error)?;
        }
        if let Some(tags) = &request.tags {
            db.replace_tags(&dir, &name, tags).map_err(map_org_error)?;
        } else {
            if let Some(add) = &request.add_tags {
                db.add_tags(&dir, &name, add).map_err(map_org_error)?;
            }
            if let Some(remove) = &request.remove_tags {
                db.remove_tags(&dir, &name, remove).map_err(map_org_error)?;
            }
        }
        let index = archive
            .committed_archive_index(&dir)
            .map_err(|e| ApiError::internal(format!("gallery archive read failed: {e:#}")))?;
        enriched_gallery_image(db, Some(&index), &dir, &name, retention)
            .map_err(|e| ApiError::internal(format!("gallery enrichment failed: {e:#}")))?
            .ok_or_else(|| org_not_found("gallery print not found"))
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery patch task failed: {e}")))??;

    state.events.publish(ServerEvent::GalleryUpdated {
        filename: image.filename.clone(),
        image: Some(Box::new(image.clone())),
    });
    Ok(Json(image))
}

// ── POST /api/gallery/organize ──────────────────────────────────────────────

/// Apply one organization edit to many prints in a single transaction.
#[utoipa::path(
    post,
    path = "/api/gallery/organize",
    tag = "gallery",
    request_body = GalleryOrganizeRequest,
    responses(
        (status = 204, description = "Applied to every listed filename"),
        (status = 404, description = "A filename or collection id is unknown; nothing was changed"),
        (status = 422, description = "Empty filename list or invalid tag"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn organize_gallery(
    State(state): State<AppState>,
    Json(request): Json<GalleryOrganizeRequest>,
) -> Result<StatusCode, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let dir = gallery_output_dir(&state).await?;
    if request.filenames.is_empty() {
        return Err(ApiError::validation("filenames must not be empty"));
    }
    let filenames = request
        .filenames
        .iter()
        .map(|f| clean_gallery_filename(f))
        .collect::<Result<Vec<_>, _>>()?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let collections_changed = request
        .add_to_collections
        .as_ref()
        .is_some_and(|c| !c.is_empty())
        || request
            .remove_from_collections
            .as_ref()
            .is_some_and(|c| !c.is_empty());
    let names_for_task = filenames.clone();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        let db = require_metadata_db(&db)?;
        // Name the offending filename: the DB's NotFound is deliberately
        // uniform, so pre-check each row before the transaction.
        for name in &names_for_task {
            if db
                .get(&dir, name)
                .map_err(|e| ApiError::internal(format!("metadata DB read failed: {e:#}")))?
                .is_none()
            {
                return Err(org_not_found(&format!("gallery print not found: {name}")));
            }
        }
        let op = mold_db::BulkOrganize {
            favorite: request.favorite,
            add_tags: request.add_tags.as_deref(),
            remove_tags: request.remove_tags.as_deref(),
            add_to_collections: request.add_to_collections.as_deref(),
            remove_from_collections: request.remove_from_collections.as_deref(),
        };
        db.organize_bulk(&dir, &names_for_task, op)
            .map_err(|error| match error {
                OrganizationError::NotFound => org_not_found("collection not found"),
                other => map_org_error(other),
            })
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery organize task failed: {e}")))??;

    // Bulk edits publish without the row (`image: None` ⇒ refetch): building
    // N enriched rows here would serialize a large selection behind one
    // request for little gain.
    for filename in filenames {
        state.events.publish(ServerEvent::GalleryUpdated {
            filename,
            image: None,
        });
    }
    if collections_changed {
        state
            .events
            .publish(ServerEvent::GalleryCollectionsChanged {});
    }
    Ok(StatusCode::NO_CONTENT)
}

// ── /api/gallery/collections ────────────────────────────────────────────────

/// List every collection with its item count.
#[utoipa::path(
    get,
    path = "/api/gallery/collections",
    tag = "gallery",
    responses(
        (status = 200, description = "Collections, newest-updated first", body = Vec<Collection>),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn list_collections(
    State(state): State<AppState>,
) -> Result<Json<Vec<Collection>>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let rows = tokio::task::spawn_blocking(move || -> Result<Vec<CollectionRow>, ApiError> {
        require_metadata_db(&db)?
            .list_collections()
            .map_err(map_org_error)
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    Ok(Json(rows.into_iter().map(collection_to_wire).collect()))
}

/// Create a collection.
#[utoipa::path(
    post,
    path = "/api/gallery/collections",
    tag = "gallery",
    request_body = CollectionCreateRequest,
    responses(
        (status = 201, description = "Created", body = Collection),
        (status = 409, description = "A collection with the same slug exists"),
        (status = 422, description = "Empty or invalid name"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn create_collection(
    State(state): State<AppState>,
    Json(request): Json<CollectionCreateRequest>,
) -> Result<(StatusCode, Json<Collection>), ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let row = tokio::task::spawn_blocking(move || -> Result<CollectionRow, ApiError> {
        require_metadata_db(&db)?
            .create_collection(&request.name, request.description.as_deref())
            .map_err(map_org_error)
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    state
        .events
        .publish(ServerEvent::GalleryCollectionsChanged {});
    Ok((StatusCode::CREATED, Json(collection_to_wire(row))))
}

/// One collection plus its member filenames in order.
#[utoipa::path(
    get,
    path = "/api/gallery/collections/{id}",
    tag = "gallery",
    params(("id" = String, Path, description = "Collection id")),
    responses(
        (status = 200, description = "Collection and ordered member filenames", body = CollectionDetail),
        (status = 404, description = "Unknown collection"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn get_collection(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<CollectionDetail>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let detail = tokio::task::spawn_blocking(move || -> Result<CollectionDetail, ApiError> {
        let db = require_metadata_db(&db)?;
        let row = db
            .get_collection(&id)
            .map_err(map_org_error)?
            .ok_or_else(|| org_not_found("collection not found"))?;
        let filenames = db.collection_filenames(&id).map_err(map_org_error)?;
        Ok(CollectionDetail {
            collection: collection_to_wire(row),
            filenames,
        })
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    Ok(Json(detail))
}

/// Rename / describe / re-cover a collection.
#[utoipa::path(
    patch,
    path = "/api/gallery/collections/{id}",
    tag = "gallery",
    params(("id" = String, Path, description = "Collection id")),
    request_body = CollectionUpdateRequest,
    responses(
        (status = 200, description = "Updated", body = Collection),
        (status = 404, description = "Unknown collection"),
        (status = 409, description = "Renaming would collide with another collection's slug"),
        (status = 422, description = "Invalid name or a cover that is not a member"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn update_collection(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<CollectionUpdateRequest>,
) -> Result<Json<Collection>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let row = tokio::task::spawn_blocking(move || -> Result<CollectionRow, ApiError> {
        require_metadata_db(&db)?
            .update_collection(
                &id,
                request.name.as_deref(),
                request.description.as_deref(),
                request.cover_filename.as_deref(),
            )
            .map_err(map_org_error)
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    state
        .events
        .publish(ServerEvent::GalleryCollectionsChanged {});
    Ok(Json(collection_to_wire(row)))
}

/// Delete a collection. Its prints are untouched.
#[utoipa::path(
    delete,
    path = "/api/gallery/collections/{id}",
    tag = "gallery",
    params(("id" = String, Path, description = "Collection id")),
    responses(
        (status = 204, description = "Deleted (prints are never touched)"),
        (status = 404, description = "Unknown collection"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn delete_collection(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let existed = tokio::task::spawn_blocking(move || -> Result<bool, ApiError> {
        require_metadata_db(&db)?
            .delete_collection(&id)
            .map_err(map_org_error)
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    if !existed {
        return Err(org_not_found("collection not found"));
    }
    state
        .events
        .publish(ServerEvent::GalleryCollectionsChanged {});
    Ok(StatusCode::NO_CONTENT)
}

/// Add / remove prints in a collection.
#[utoipa::path(
    put,
    path = "/api/gallery/collections/{id}/items",
    tag = "gallery",
    params(("id" = String, Path, description = "Collection id")),
    request_body = CollectionItemsRequest,
    responses(
        (status = 200, description = "The collection after the change", body = Collection),
        (status = 404, description = "Unknown collection or filename; nothing was changed"),
        (status = 422, description = "Invalid filename"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn put_collection_items(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<CollectionItemsRequest>,
) -> Result<Json<Collection>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let dir = gallery_output_dir(&state).await?;
    let add = request
        .add
        .iter()
        .map(|f| clean_gallery_filename(f))
        .collect::<Result<Vec<_>, _>>()?;
    let remove = request
        .remove
        .iter()
        .map(|f| clean_gallery_filename(f))
        .collect::<Result<Vec<_>, _>>()?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let touched: Vec<String> = add.iter().chain(remove.iter()).cloned().collect();
    let row = tokio::task::spawn_blocking(move || -> Result<CollectionRow, ApiError> {
        let db = require_metadata_db(&db)?;
        if db.get_collection(&id).map_err(map_org_error)?.is_none() {
            return Err(org_not_found("collection not found"));
        }
        for name in add.iter().chain(remove.iter()) {
            if db
                .get(&dir, name)
                .map_err(|e| ApiError::internal(format!("metadata DB read failed: {e:#}")))?
                .is_none()
            {
                return Err(org_not_found(&format!("gallery print not found: {name}")));
            }
        }
        if !add.is_empty() {
            db.collection_add(&id, &dir, &add).map_err(map_org_error)?;
        }
        if !remove.is_empty() {
            db.collection_remove(&id, &dir, &remove)
                .map_err(map_org_error)?;
        }
        db.get_collection(&id)
            .map_err(map_org_error)?
            .ok_or_else(|| org_not_found("collection not found"))
    })
    .await
    .map_err(|e| ApiError::internal(format!("collections task failed: {e}")))??;
    state
        .events
        .publish(ServerEvent::GalleryCollectionsChanged {});
    for filename in touched {
        state.events.publish(ServerEvent::GalleryUpdated {
            filename,
            image: None,
        });
    }
    Ok(Json(collection_to_wire(row)))
}

// ── /api/gallery/tags ───────────────────────────────────────────────────────

/// Every tag with its use count.
#[utoipa::path(
    get,
    path = "/api/gallery/tags",
    tag = "gallery",
    responses(
        (status = 200, description = "Tags sorted case-insensitively", body = Vec<TagCount>),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn list_tags(
    State(state): State<AppState>,
) -> Result<Json<Vec<TagCount>>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let rows =
        tokio::task::spawn_blocking(move || -> Result<Vec<mold_db::TagCountRow>, ApiError> {
            require_metadata_db(&db)?.list_tags().map_err(map_org_error)
        })
        .await
        .map_err(|e| ApiError::internal(format!("tags task failed: {e}")))??;
    Ok(Json(rows.into_iter().map(tag_to_wire).collect()))
}

/// Rename a tag (merging into an existing one when the new name is taken).
#[utoipa::path(
    patch,
    path = "/api/gallery/tags/{name}",
    tag = "gallery",
    params(("name" = String, Path, description = "Current tag name")),
    request_body = TagRenameRequest,
    responses(
        (status = 200, description = "The resulting tag and its count", body = TagCount),
        (status = 404, description = "Unknown tag"),
        (status = 422, description = "Empty or invalid new name"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn rename_tag(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(request): Json<TagRenameRequest>,
) -> Result<Json<TagCount>, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let dir = gallery_output_dir(&state).await?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let (tag, affected) =
        tokio::task::spawn_blocking(move || -> Result<(TagCount, Vec<String>), ApiError> {
            let db = require_metadata_db(&db)?;
            let affected = filenames_with_tag(db, &dir, &name)?;
            let stored = db.rename_tag(&name, &request.name).map_err(map_org_error)?;
            let count = db
                .list_tags()
                .map_err(map_org_error)?
                .into_iter()
                .find(|row| row.name.eq_ignore_ascii_case(&stored))
                .map(|row| row.count)
                .unwrap_or(0);
            Ok((
                TagCount {
                    name: stored,
                    count,
                },
                affected,
            ))
        })
        .await
        .map_err(|e| ApiError::internal(format!("tags task failed: {e}")))??;
    for filename in affected {
        state.events.publish(ServerEvent::GalleryUpdated {
            filename,
            image: None,
        });
    }
    Ok(Json(tag))
}

/// Delete a tag from every print.
#[utoipa::path(
    delete,
    path = "/api/gallery/tags/{name}",
    tag = "gallery",
    params(("name" = String, Path, description = "Tag name")),
    responses(
        (status = 204, description = "Deleted everywhere"),
        (status = 404, description = "Unknown tag"),
        (status = 501, description = "Metadata DB disabled — organization unavailable"),
    )
)]
pub(crate) async fn delete_tag(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let dir = gallery_output_dir(&state).await?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let (existed, affected) =
        tokio::task::spawn_blocking(move || -> Result<(bool, Vec<String>), ApiError> {
            let db = require_metadata_db(&db)?;
            let affected = filenames_with_tag(db, &dir, &name)?;
            let existed = db.delete_tag(&name).map_err(map_org_error)?;
            Ok((existed, affected))
        })
        .await
        .map_err(|e| ApiError::internal(format!("tags task failed: {e}")))??;
    if !existed {
        return Err(org_not_found("tag not found"));
    }
    for filename in affected {
        state.events.publish(ServerEvent::GalleryUpdated {
            filename,
            image: None,
        });
    }
    Ok(StatusCode::NO_CONTENT)
}
