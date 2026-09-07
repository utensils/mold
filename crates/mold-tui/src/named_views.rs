use mold_core::{
    GenerationImageReferenceRole, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance,
};
use std::{io::Cursor, path::Path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NamedViewPath {
    pub role: GenerationImageReferenceRole,
    pub path: String,
}

pub(crate) fn parse_named_view_paths(input: &str) -> Result<Vec<NamedViewPath>, String> {
    let mut parsed = Vec::new();
    for raw in input
        .split(';')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let (role, path) = raw
            .split_once('=')
            .ok_or_else(|| format!("Named view needs role=path: {raw}"))?;
        let role = match role.trim().to_ascii_lowercase().as_str() {
            "front" => GenerationImageReferenceRole::Front,
            "left" => GenerationImageReferenceRole::Left,
            "back" => GenerationImageReferenceRole::Back,
            "right" => GenerationImageReferenceRole::Right,
            other => return Err(format!("Unknown named view role '{other}'")),
        };
        if parsed.iter().any(|item: &NamedViewPath| item.role == role) {
            return Err(format!(
                "Named view role '{}' appears more than once",
                role_name(role)
            ));
        }
        let path = crate::source_image::validate_source_image_path(path.trim())?;
        parsed.push(NamedViewPath { role, path });
    }
    parsed.sort_by_key(|item| role_index(item.role));
    Ok(parsed)
}

pub(crate) fn display_named_view_paths(paths: &[NamedViewPath]) -> String {
    paths
        .iter()
        .map(|item| format!("{}={}", role_name(item.role), item.path))
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn prepare_named_views(
    paths: &[NamedViewPath],
) -> Result<Vec<GenerationReference>, String> {
    paths.iter().map(prepare_named_view).collect()
}

fn prepare_named_view(item: &NamedViewPath) -> Result<GenerationReference, String> {
    let role = role_name(item.role);
    let bytes = std::fs::read(&item.path)
        .map_err(|error| format!("Named view {role} could not be read: {error}"))?;
    let reader = image::ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|error| format!("Named view {role} is not an image: {error}"))?;
    let format = reader
        .format()
        .ok_or_else(|| format!("Named view {role} has an unknown format"))?;
    let mime_type = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        _ => return Err(format!("Named view {role} must be PNG or JPEG")),
    };
    let (width, height) =
        mold_inference::img_utils::oriented_image_dimensions(reader, image_limits())
            .map_err(|error| format!("Named view {role} could not be decoded: {error}"))?;
    Ok(GenerationReference::NamedImage {
        role: item.role,
        media: GenerationReferenceAuthority::Inline { data: bytes },
        provenance: GenerationReferenceProvenance {
            name: Path::new(&item.path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
            sha256: None,
            crop: None,
        },
        mime_type: mime_type.to_string(),
        width,
        height,
    })
}

fn image_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(mold_core::minimax_h3::MAX_REFERENCE_DIMENSION);
    limits.max_image_height = Some(mold_core::minimax_h3::MAX_REFERENCE_DIMENSION);
    limits.max_alloc = Some(mold_core::minimax_h3::MAX_REFERENCE_IMAGE_PIXELS.saturating_mul(4));
    limits
}

fn role_index(role: GenerationImageReferenceRole) -> u8 {
    match role {
        GenerationImageReferenceRole::Front => 0,
        GenerationImageReferenceRole::Left => 1,
        GenerationImageReferenceRole::Back => 2,
        GenerationImageReferenceRole::Right => 3,
    }
}

fn role_name(role: GenerationImageReferenceRole) -> &'static str {
    match role {
        GenerationImageReferenceRole::Front => "front",
        GenerationImageReferenceRole::Left => "left",
        GenerationImageReferenceRole::Back => "back",
        GenerationImageReferenceRole::Right => "right",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_orders_semantic_slots_and_rejects_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let front = dir.path().join("front.png");
        let right = dir.path().join("right.jpg");
        image::DynamicImage::new_rgba8(8, 8).save(&front).unwrap();
        image::DynamicImage::new_rgb8(8, 8).save(&right).unwrap();

        let parsed = parse_named_view_paths(&format!(
            "right={}; front={}",
            right.display(),
            front.display()
        ))
        .unwrap();
        assert_eq!(
            parsed.iter().map(|item| item.role).collect::<Vec<_>>(),
            [
                GenerationImageReferenceRole::Front,
                GenerationImageReferenceRole::Right
            ]
        );
        let error = parse_named_view_paths(&format!(
            "front={}; front={}",
            front.display(),
            front.display()
        ))
        .unwrap_err();
        assert!(error.contains("appears more than once"));
    }

    #[test]
    fn preparation_keeps_role_dimensions_bytes_and_basename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("left.png");
        image::DynamicImage::new_rgba8(32, 16).save(&path).unwrap();
        let prepared = prepare_named_views(&[NamedViewPath {
            role: GenerationImageReferenceRole::Left,
            path: path.to_string_lossy().into_owned(),
        }])
        .unwrap();
        assert!(matches!(
            &prepared[0],
            GenerationReference::NamedImage {
                role: GenerationImageReferenceRole::Left,
                media: GenerationReferenceAuthority::Inline { data },
                provenance,
                mime_type,
                width: 32,
                height: 16,
            } if !data.is_empty()
                && provenance.name.as_deref() == Some("left.png")
                && mime_type == "image/png"
        ));
    }
}
