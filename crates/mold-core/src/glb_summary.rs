//! Mesh statistics read off a binary glTF's JSON chunk, with no geometry
//! decode.
//!
//! A client that hydrates a stored `.glb` from the gallery — the MCP server
//! on the durable path — has the bytes and the gallery row, and the row
//! records the controls that shaped the mesh but not what came out: vertex
//! and triangle counts and the bounds live in the file. glTF keeps all of
//! them in the JSON chunk (`accessors[].count`, and `min`/`max`, which the
//! spec REQUIRES on every `POSITION` accessor), so they are a header parse
//! away and nobody has to link a geometry reader to caption a mesh.

const GLB_MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;

/// What a caption needs to know about a stored mesh, mirroring the fields
/// [`crate::MeshData`] promises.
#[derive(Debug, Clone, PartialEq)]
pub struct GlbSummary {
    pub vertex_count: u32,
    pub face_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Whether a material samples a base-colour texture, as opposed to bare
    /// geometry with a default material.
    pub textured: bool,
}

/// Summarise the first primitive of the first mesh in a binary glTF.
///
/// `None` for anything that is not a version-2 GLB with a JSON chunk naming
/// a `POSITION` accessor with `min`/`max` — a malformed or foreign file is
/// "no summary", never a panic, because the caller has already accepted the
/// bytes as an output and is only captioning them.
pub fn summarize_glb(bytes: &[u8]) -> Option<GlbSummary> {
    let u32_at = |offset: usize| -> Option<u32> {
        bytes
            .get(offset..offset + 4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("4 bytes")))
    };
    if bytes.get(0..4)? != GLB_MAGIC || u32_at(4)? != 2 {
        return None;
    }
    let json_len = u32_at(12)? as usize;
    if u32_at(16)? != CHUNK_JSON {
        return None;
    }
    let json_bytes = bytes.get(20..20usize.checked_add(json_len)?)?;
    let json: serde_json::Value = serde_json::from_slice(json_bytes).ok()?;

    let primitive = json.get("meshes")?.get(0)?.get("primitives")?.get(0)?;
    let accessors = json.get("accessors")?;
    let position =
        accessors.get(primitive.get("attributes")?.get("POSITION")?.as_u64()? as usize)?;
    let vertex_count = u32::try_from(position.get("count")?.as_u64()?).ok()?;
    let vec3 = |key: &str| -> Option<[f32; 3]> {
        let values = position.get(key)?.as_array()?;
        if values.len() != 3 {
            return None;
        }
        Some([
            values[0].as_f64()? as f32,
            values[1].as_f64()? as f32,
            values[2].as_f64()? as f32,
        ])
    };
    let bounds_min = vec3("min")?;
    let bounds_max = vec3("max")?;
    let index_count = match primitive.get("indices").and_then(serde_json::Value::as_u64) {
        Some(index) => accessors.get(index as usize)?.get("count")?.as_u64()?,
        None => u64::from(vertex_count),
    };
    let face_count = u32::try_from(index_count / 3).ok()?;
    let textured = json
        .get("materials")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|materials| {
            materials.iter().any(|material| {
                material
                    .get("pbrMetallicRoughness")
                    .and_then(|pbr| pbr.get("baseColorTexture"))
                    .is_some()
            })
        });
    Some(GlbSummary {
        vertex_count,
        face_count,
        bounds_min,
        bounds_max,
        textured,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GLB with the given JSON chunk and an empty BIN chunk — the summary
    /// never reads the binary chunk, so its contents are irrelevant here.
    fn glb_with_json(json: &serde_json::Value) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(json).unwrap();
        json_bytes.extend(std::iter::repeat_n(b' ', (4 - (json_bytes.len() % 4)) % 4));
        let total = 12 + 8 + json_bytes.len() + 8;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out
    }

    fn indexed_mesh_json() -> serde_json::Value {
        serde_json::json!({
            "asset": { "version": "2.0" },
            "accessors": [
                { "type": "VEC3", "componentType": 5126, "count": 24576,
                  "min": [-0.5, -0.4, -0.3], "max": [0.5, 0.4, 0.3] },
                { "type": "SCALAR", "componentType": 5125, "count": 147456 }
            ],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }]
        })
    }

    #[test]
    fn summary_reads_counts_and_bounds_off_the_json_chunk() {
        let summary = summarize_glb(&glb_with_json(&indexed_mesh_json())).unwrap();
        assert_eq!(
            summary,
            GlbSummary {
                vertex_count: 24_576,
                face_count: 49_152,
                bounds_min: [-0.5, -0.4, -0.3],
                bounds_max: [0.5, 0.4, 0.3],
                textured: false,
            }
        );
    }

    #[test]
    fn a_non_indexed_primitive_counts_a_face_per_three_vertices() {
        let mut json = indexed_mesh_json();
        json["meshes"][0]["primitives"][0]
            .as_object_mut()
            .unwrap()
            .remove("indices");
        json["accessors"][0]["count"] = serde_json::json!(9);
        let summary = summarize_glb(&glb_with_json(&json)).unwrap();
        assert_eq!((summary.vertex_count, summary.face_count), (9, 3));
    }

    #[test]
    fn a_base_colour_texture_marks_the_mesh_textured() {
        let mut json = indexed_mesh_json();
        json["materials"] = serde_json::json!([{
            "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } }
        }]);
        assert!(summarize_glb(&glb_with_json(&json)).unwrap().textured);
        json["materials"] = serde_json::json!([{ "pbrMetallicRoughness": {} }]);
        assert!(!summarize_glb(&glb_with_json(&json)).unwrap().textured);
    }

    #[test]
    fn foreign_or_broken_bytes_summarise_to_none_without_panicking() {
        assert_eq!(summarize_glb(b""), None);
        assert_eq!(summarize_glb(b"glT"), None);
        assert_eq!(summarize_glb(b"not a glb at all, just bytes"), None);
        let good = glb_with_json(&indexed_mesh_json());
        let mut wrong_version = good.clone();
        wrong_version[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(summarize_glb(&wrong_version), None);
        let mut truncated = good.clone();
        truncated.truncate(40);
        assert_eq!(summarize_glb(&truncated), None);
        let mut huge_json_len = good.clone();
        huge_json_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(summarize_glb(&huge_json_len), None);

        let mut no_bounds = indexed_mesh_json();
        no_bounds["accessors"][0]
            .as_object_mut()
            .unwrap()
            .remove("min");
        assert_eq!(summarize_glb(&glb_with_json(&no_bounds)), None);
        let mut no_position = indexed_mesh_json();
        no_position["meshes"][0]["primitives"][0]["attributes"] = serde_json::json!({});
        assert_eq!(summarize_glb(&glb_with_json(&no_position)), None);
    }
}
