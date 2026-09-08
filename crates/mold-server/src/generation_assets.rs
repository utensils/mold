//! Durable inventory and path-safe download of files associated with a print.

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use mold_db::generation_assets::{AssetStorage, GenerationAssetRecord};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::{routes::ApiError, state::AppState};

fn media_version(record: &mold_db::GenerationRecord) -> String {
    format!(
        "{}:{}",
        record.file_mtime_ms.unwrap_or(record.created_at_ms),
        record.file_size_bytes.unwrap_or_default()
    )
}

fn print_path(output_dir: &Path, record: &mold_db::GenerationRecord) -> PathBuf {
    if record.trashed_at_ms.is_some() {
        mold_db::trash::trash_dir(output_dir).join(&record.filename)
    } else {
        output_dir.join(&record.filename)
    }
}

fn inventory_from_glb(filename: &str, bytes: &[u8]) -> anyhow::Result<Vec<GenerationAssetRecord>> {
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("mesh");
    Ok(
        mold_inference::hunyuan3d::glb::embedded_material_assets(bytes)?
            .into_iter()
            .map(|embedded| GenerationAssetRecord {
                asset: mold_core::GenerationAsset {
                    asset_id: embedded.role.to_string(),
                    role: embedded.role.to_string(),
                    display_name: format!("{stem}-{}", embedded.display_suffix),
                    media_type: embedded.media_type.to_string(),
                    size_bytes: embedded.bytes.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(&embedded.bytes)),
                    width: Some(embedded.width),
                    height: Some(embedded.height),
                },
                storage: AssetStorage::EmbeddedGlb {
                    role: embedded.role.to_string(),
                },
            })
            .collect(),
    )
}

fn ensure_indexed(
    db: &mold_db::MetadataDb,
    output_dir: &Path,
    record: &mold_db::GenerationRecord,
) -> anyhow::Result<()> {
    let Some(id) = record.id else {
        return Ok(());
    };
    let version = media_version(record);
    if mold_db::generation_assets::scan_is_current(db, id, &version)? {
        return Ok(());
    }
    let records = if record.format == mold_core::OutputFormat::Glb {
        let bytes = std::fs::read(print_path(output_dir, record))?;
        inventory_from_glb(&record.filename, &bytes)?
    } else {
        Vec::new()
    };
    mold_db::generation_assets::replace_for_generation(db, id, &version, &records)
}

pub(crate) fn attach_to_images(
    db: &mold_db::MetadataDb,
    output_dir: &Path,
    records: &[mold_db::GenerationRecord],
    images: &mut [mold_core::GalleryImage],
) -> anyhow::Result<()> {
    let by_name = records
        .iter()
        .map(|record| (record.filename.as_str(), record))
        .collect::<std::collections::HashMap<_, _>>();
    for image in images {
        let Some(record) = by_name.get(image.filename.as_str()) else {
            continue;
        };
        ensure_indexed(db, output_dir, record)?;
        if let Some(id) = record.id {
            image.assets = mold_db::generation_assets::list_for_generation(db, id)?
                .into_iter()
                .map(|record| record.asset)
                .collect();
        }
    }
    Ok(())
}

fn safe_download_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .take(255)
        .collect()
}

pub(crate) async fn download(
    State(state): State<AppState>,
    AxumPath((filename, asset_id)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    crate::routes::validate_gallery_filename(&filename)?;
    if asset_id.is_empty() || asset_id.len() > 128 || asset_id.contains('/') {
        return Err(ApiError::validation("invalid generation asset id"));
    }
    let config = state.config.read().await;
    let output_dir = config.effective_output_dir();
    drop(config);
    let db = state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::structured(
            "generation assets require the metadata database",
            "GENERATION_ASSETS_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
            None,
            None,
        )
    })?;
    let record = db
        .get(&output_dir, &filename)
        .map_err(|error| ApiError::internal(format!("generation asset lookup failed: {error:#}")))?
        .ok_or_else(|| {
            ApiError::structured(
                "gallery print not found",
                "GALLERY_NOT_FOUND",
                StatusCode::NOT_FOUND,
                None,
                None,
            )
        })?;
    let path = print_path(&output_dir, &record);
    let (asset, bytes) = tokio::task::spawn_blocking({
        let db = state.metadata_db.clone();
        let output_dir = output_dir.clone();
        move || -> anyhow::Result<Option<(mold_core::GenerationAsset, Vec<u8>)>> {
            let db = db.as_ref().as_ref().expect("checked before spawn");
            ensure_indexed(db, &output_dir, &record)?;
            let Some(id) = record.id else { return Ok(None) };
            let Some(stored) = mold_db::generation_assets::list_for_generation(db, id)?
                .into_iter()
                .find(|stored| stored.asset.asset_id == asset_id)
            else {
                return Ok(None);
            };
            let bytes = match &stored.storage {
                AssetStorage::EmbeddedGlb { role } => {
                    let glb = std::fs::read(&path)?;
                    mold_inference::hunyuan3d::glb::embedded_material_assets(&glb)?
                        .into_iter()
                        .find(|embedded| embedded.role == role)
                        .map(|embedded| embedded.bytes)
                        .ok_or_else(|| anyhow::anyhow!("embedded generation asset is missing"))?
                }
                AssetStorage::Sidecar { filename } => {
                    anyhow::ensure!(
                        Path::new(filename).components().count() == 1
                            && Path::new(filename).file_name().is_some(),
                        "invalid generation asset sidecar locator"
                    );
                    std::fs::read(output_dir.join(filename))?
                }
            };
            anyhow::ensure!(
                bytes.len() as u64 == stored.asset.size_bytes,
                "generation asset size changed"
            );
            anyhow::ensure!(
                format!("{:x}", Sha256::digest(&bytes)) == stored.asset.sha256,
                "generation asset digest changed"
            );
            Ok(Some((stored.asset, bytes)))
        }
    })
    .await
    .map_err(|error| ApiError::internal(format!("generation asset task failed: {error}")))?
    .map_err(|error| ApiError::internal(format!("generation asset read failed: {error:#}")))?
    .ok_or_else(|| {
        ApiError::structured(
            "generation asset not found",
            "GENERATION_ASSET_NOT_FOUND",
            StatusCode::NOT_FOUND,
            None,
            None,
        )
    })?;

    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&asset.media_type)
            .map_err(|_| ApiError::internal("invalid stored generation asset media type"))?,
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&asset.size_bytes.to_string())
            .map_err(|_| ApiError::internal("invalid generation asset size"))?,
    );
    let disposition = format!(
        "attachment; filename=\"{}\"",
        safe_download_name(&asset.display_name)
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&disposition)
            .map_err(|_| ApiError::internal("invalid generation asset filename"))?,
    );
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glb_inventory_preserves_encoded_maps_and_uses_stable_roles() {
        use mold_inference::hunyuan3d::{glb, mesh::Mesh};
        let png = |rgb| {
            let image = image::RgbImage::from_pixel(2, 2, image::Rgb(rgb));
            let mut bytes = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut bytes),
                    image::ImageFormat::Png,
                )
                .unwrap();
            bytes
        };
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            normals: None,
            vertex_colors: None,
            uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        };
        let bytes = glb::write_glb(
            &mesh,
            &glb::GlbMaterial {
                base_color_texture: Some(png([1, 2, 3])),
                metallic_roughness_texture: Some(png([4, 5, 6])),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let records = inventory_from_glb("chair.glb", &bytes).unwrap();
        assert_eq!(
            records
                .iter()
                .map(|r| r.asset.asset_id.as_str())
                .collect::<Vec<_>>(),
            ["base_color", "metallic_roughness"]
        );
        assert!(records
            .iter()
            .all(|r| r.asset.display_name.starts_with("chair-")));
    }
}
