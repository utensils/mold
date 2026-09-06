//! Binary glTF (GLB), Wavefront OBJ, STL and PLY serialization for extracted
//! meshes, plus the narrow GLB reader the gallery exports transcode from.
//!
//! Dependency-free by construction: the GLB container is 12 bytes of header
//! plus two length-prefixed chunks, and everything else is `serde_json`. Ported
//! from ComfyUI's `save_glb` (`comfy_extras/nodes_save_3d.py:109-494`), which
//! is the oracle for this family and is itself written without a glTF library.
//!
//! Textures are wired up now even though the texturing phase lands later, so
//! that phase is a change to the CALLER (fill in [`GlbMaterial`]) rather than a
//! change to the writer.

use crate::hunyuan3d::mesh::{Mesh, MeshError};

/// Why a mesh cannot be written as GLB.
#[derive(Debug, thiserror::Error)]
pub enum GlbError {
    /// glTF has no representation for a mesh with no vertices, and upstream
    /// raises here too (`nodes_save_3d.py:150-151`). Returning an error beats
    /// writing a file that every viewer rejects.
    #[error("cannot write GLB: the mesh has no vertices")]
    EmptyMesh,
    /// The mesh failed [`Mesh::validate`] — out-of-range face index, non-finite
    /// coordinate, or a mismatched per-vertex attribute.
    #[error("cannot write GLB: {0}")]
    InvalidMesh(#[from] MeshError),
}

/// PBR material for the exported primitive.
///
/// The three `Option<f32>` / `Option<[f32; 4]>` factors mean "auto": leave them
/// `None` and the writer picks the value upstream picks for that combination of
/// attributes (`nodes_save_3d.py:387-402`, where the same idea is spelled as a
/// negative sentinel). Set one to override.
#[derive(Debug, Clone)]
pub struct GlbMaterial {
    /// baseColor texture, as encoded PNG bytes. Needs `mesh.uvs`.
    pub base_color_texture: Option<Vec<u8>>,
    /// glTF metallicRoughness texture (R unused or AO, G roughness, B metallic).
    pub metallic_roughness_texture: Option<Vec<u8>>,
    /// Tangent-space normal map (glTF/OpenGL +Y).
    pub normal_texture: Option<Vec<u8>>,
    /// `None` = auto: neutral gray for bare geometry, white when a baseColor
    /// texture or vertex colors are present so they pass through unscaled.
    pub base_color_factor: Option<[f32; 4]>,
    /// `None` = auto: 0.0, or 1.0 when a metallicRoughness texture is present
    /// (the factors scale the texture, so 1.0 passes it through).
    pub metallic_factor: Option<f32>,
    /// `None` = auto: 0.5 bare, 1.0 with a baseColor texture or vertex colors.
    pub roughness_factor: Option<f32>,
    /// `normalTexture.scale`.
    pub normal_scale: f32,
    /// ORM packing: the R channel of `metallic_roughness_texture` holds ambient
    /// occlusion, and `occlusionTexture` points at that same image
    /// (`nodes_save_3d.py:404-408`).
    pub occlusion_in_metallic_roughness: bool,
    /// `occlusionTexture.strength`.
    pub occlusion_strength: f32,
    pub double_sided: bool,
}

impl Default for GlbMaterial {
    fn default() -> Self {
        Self {
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            base_color_factor: None,
            metallic_factor: None,
            roughness_factor: None,
            normal_scale: 1.0,
            occlusion_in_metallic_roughness: false,
            occlusion_strength: 1.0,
            // Extracted surfaces are not reliably closed, and a one-sided
            // material makes the inside of an open shell invisible.
            double_sided: true,
        }
    }
}

// glTF constants, spelled out so the JSON below reads as glTF and not as magic.
const COMPONENT_FLOAT: u32 = 5126;
const COMPONENT_UNSIGNED_INT: u32 = 5125;
const COMPONENT_UNSIGNED_SHORT: u32 = 5123;
const COMPONENT_UNSIGNED_BYTE: u32 = 5121;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;
const MODE_TRIANGLES: u32 = 4;
const FILTER_LINEAR: u32 = 9729;
const WRAP_CLAMP_TO_EDGE: u32 = 33071;

const GLB_MAGIC: &[u8; 4] = b"glTF";
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

/// Accumulates 4-byte-aligned blobs into one glTF buffer, handing back each
/// blob's byte offset.
#[derive(Default)]
struct BinBuffer {
    data: Vec<u8>,
}

impl BinBuffer {
    /// Appends `bytes`, zero-pads to the next 4-byte boundary, and returns
    /// `(byte_offset, byte_length)` of the unpadded content.
    fn push(&mut self, bytes: &[u8]) -> (usize, usize) {
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let pad = (4 - (bytes.len() % 4)) % 4;
        self.data.extend(std::iter::repeat_n(0u8, pad));
        (offset, bytes.len())
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Serialize `mesh` as a complete binary glTF file.
///
/// `metadata`, when present, becomes `asset.extras` — the same slot upstream
/// uses (`nodes_save_3d.py:470-471`).
pub fn write_glb(
    mesh: &Mesh,
    material: &GlbMaterial,
    metadata: Option<&serde_json::Value>,
) -> anyhow::Result<Vec<u8>> {
    if mesh.vertices.is_empty() {
        return Err(GlbError::EmptyMesh.into());
    }
    // Catches out-of-range face indices before a single byte is written, so a
    // bad mesh is an error rather than a corrupt file.
    mesh.validate().map_err(GlbError::InvalidMesh)?;

    let n_verts = mesh.vertices.len();
    let mut bin = BinBuffer::default();

    // Blob order matches upstream's `_blobs` list (`nodes_save_3d.py:214-219`)
    // so a byte diff against the oracle lines up.
    let positions: Vec<f32> = mesh.vertices.iter().flatten().copied().collect();
    let (pos_off, pos_len) = bin.push(&f32_bytes(&positions));

    let mut index_bytes = Vec::with_capacity(mesh.faces.len() * 12);
    for face in &mesh.faces {
        for i in face {
            index_bytes.extend_from_slice(&i.to_le_bytes());
        }
    }
    let (idx_off, idx_len) = bin.push(&index_bytes);

    let uv_view = mesh.uvs.as_ref().filter(|u| !u.is_empty()).map(|uvs| {
        let flat: Vec<f32> = uvs.iter().flatten().copied().collect();
        bin.push(&f32_bytes(&flat))
    });
    let color_view = mesh
        .vertex_colors
        .as_ref()
        .filter(|c| !c.is_empty())
        .map(|colors| {
            // Upstream clips vertex colors into [0, 1] (`nodes_save_3d.py:146`).
            let flat: Vec<f32> = colors.iter().flatten().map(|c| c.clamp(0.0, 1.0)).collect();
            bin.push(&f32_bytes(&flat))
        });
    let normal_view = mesh
        .normals
        .as_ref()
        .filter(|n| !n.is_empty())
        .map(|normals| {
            let flat: Vec<f32> = normals.iter().flatten().copied().collect();
            bin.push(&f32_bytes(&flat))
        });
    let base_color_png = material
        .base_color_texture
        .as_ref()
        .filter(|b| !b.is_empty())
        .map(|png| bin.push(png));
    let mr_png = material
        .metallic_roughness_texture
        .as_ref()
        .filter(|b| !b.is_empty())
        .map(|png| bin.push(png));
    let normal_png = material
        .normal_texture
        .as_ref()
        .filter(|b| !b.is_empty())
        .map(|png| bin.push(png));

    let mut buffer_views = vec![
        serde_json::json!({
            "buffer": 0,
            "byteOffset": pos_off,
            "byteLength": pos_len,
            "target": TARGET_ARRAY_BUFFER,
        }),
        serde_json::json!({
            "buffer": 0,
            "byteOffset": idx_off,
            "byteLength": idx_len,
            "target": TARGET_ELEMENT_ARRAY_BUFFER,
        }),
    ];

    let (min, max) = mesh.bounds();
    let mut accessors = vec![
        // glTF REQUIRES min/max on the POSITION accessor.
        serde_json::json!({
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": COMPONENT_FLOAT,
            "count": n_verts,
            "type": "VEC3",
            "min": [min[0], min[1], min[2]],
            "max": [max[0], max[1], max[2]],
        }),
        serde_json::json!({
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": COMPONENT_UNSIGNED_INT,
            "count": mesh.faces.len() * 3,
            "type": "SCALAR",
        }),
    ];

    let mut attributes = serde_json::Map::new();
    attributes.insert("POSITION".into(), serde_json::json!(0));

    let mut add_vertex_attribute =
        |name: &str, view: Option<(usize, usize)>, ty: &str, count: usize| {
            let Some((offset, length)) = view else {
                return;
            };
            buffer_views.push(serde_json::json!({
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": length,
                "target": TARGET_ARRAY_BUFFER,
            }));
            accessors.push(serde_json::json!({
                "bufferView": buffer_views.len() - 1,
                "byteOffset": 0,
                "componentType": COMPONENT_FLOAT,
                "count": count,
                "type": ty,
            }));
            attributes.insert(name.into(), serde_json::json!(accessors.len() - 1));
        };
    add_vertex_attribute("TEXCOORD_0", uv_view, "VEC2", n_verts);
    add_vertex_attribute("COLOR_0", color_view, "VEC3", n_verts);
    add_vertex_attribute("NORMAL", normal_view, "VEC3", n_verts);

    let has_uv = attributes.contains_key("TEXCOORD_0");
    let has_colors = attributes.contains_key("COLOR_0");

    // Embedded PNGs: one bufferView + one image + one texture each, sharing a
    // single sampler (`add_image_texture`, `nodes_save_3d.py:359-366`).
    let mut images: Vec<serde_json::Value> = Vec::new();
    let mut textures: Vec<serde_json::Value> = Vec::new();
    let mut samplers: Vec<serde_json::Value> = Vec::new();
    let mut add_texture = |png: (usize, usize)| -> usize {
        buffer_views.push(serde_json::json!({
            "buffer": 0,
            "byteOffset": png.0,
            "byteLength": png.1,
        }));
        images.push(serde_json::json!({
            "bufferView": buffer_views.len() - 1,
            "mimeType": "image/png",
        }));
        if samplers.is_empty() {
            samplers.push(serde_json::json!({
                "magFilter": FILTER_LINEAR,
                "minFilter": FILTER_LINEAR,
                "wrapS": WRAP_CLAMP_TO_EDGE,
                "wrapT": WRAP_CLAMP_TO_EDGE,
            }));
        }
        textures.push(serde_json::json!({ "source": images.len() - 1, "sampler": 0 }));
        textures.len() - 1
    };

    // Auto factors, matching the bare-geometry defaults upstream picks
    // (`nodes_save_3d.py:373-386`).
    let mut auto_base_color = [0.22f32, 0.22, 0.22, 1.0];
    let mut auto_metallic = 0.0f32;
    let mut auto_roughness = 0.5f32;

    let mut pbr = serde_json::Map::new();
    let base_color_texture_index = base_color_png.filter(|_| has_uv).map(&mut add_texture);
    if let Some(index) = base_color_texture_index {
        pbr.insert(
            "baseColorTexture".into(),
            serde_json::json!({ "index": index, "texCoord": 0 }),
        );
    }
    if base_color_texture_index.is_some() || has_colors {
        auto_base_color = [1.0, 1.0, 1.0, 1.0];
        auto_roughness = 1.0;
    }
    let mr_texture_index = mr_png.filter(|_| has_uv).map(&mut add_texture);
    if let Some(index) = mr_texture_index {
        pbr.insert(
            "metallicRoughnessTexture".into(),
            serde_json::json!({ "index": index, "texCoord": 0 }),
        );
        // With an MR texture the factors SCALE it, so 1.0 passes it through.
        auto_metallic = 1.0;
        auto_roughness = 1.0;
    }

    let base_color = material.base_color_factor.unwrap_or(auto_base_color);
    pbr.insert(
        "baseColorFactor".into(),
        serde_json::json!([base_color[0], base_color[1], base_color[2], base_color[3]]),
    );
    pbr.insert(
        "metallicFactor".into(),
        serde_json::json!(material.metallic_factor.unwrap_or(auto_metallic)),
    );
    pbr.insert(
        "roughnessFactor".into(),
        serde_json::json!(material.roughness_factor.unwrap_or(auto_roughness)),
    );

    let mut gltf_material = serde_json::Map::new();
    gltf_material.insert(
        "pbrMetallicRoughness".into(),
        serde_json::Value::Object(pbr),
    );
    gltf_material.insert(
        "doubleSided".into(),
        serde_json::json!(material.double_sided),
    );
    if material.occlusion_in_metallic_roughness {
        if let Some(index) = mr_texture_index {
            gltf_material.insert(
                "occlusionTexture".into(),
                serde_json::json!({
                    "index": index,
                    "texCoord": 0,
                    "strength": material.occlusion_strength,
                }),
            );
        }
    }
    if let Some(png) = normal_png.filter(|_| has_uv) {
        let index = add_texture(png);
        gltf_material.insert(
            "normalTexture".into(),
            serde_json::json!({
                "index": index,
                "texCoord": 0,
                "scale": material.normal_scale,
            }),
        );
    }

    let primitive = serde_json::json!({
        "attributes": serde_json::Value::Object(attributes),
        "indices": 1,
        "mode": MODE_TRIANGLES,
        "material": 0,
    });

    let mut asset = serde_json::Map::new();
    asset.insert("version".into(), serde_json::json!("2.0"));
    asset.insert("generator".into(), serde_json::json!("mold"));
    if let Some(metadata) = metadata {
        asset.insert("extras".into(), metadata.clone());
    }

    let mut gltf = serde_json::Map::new();
    gltf.insert("asset".into(), serde_json::Value::Object(asset));
    gltf.insert(
        "buffers".into(),
        serde_json::json!([{ "byteLength": bin.data.len() }]),
    );
    gltf.insert("bufferViews".into(), serde_json::Value::Array(buffer_views));
    gltf.insert("accessors".into(), serde_json::Value::Array(accessors));
    gltf.insert(
        "meshes".into(),
        serde_json::json!([{ "primitives": [primitive] }]),
    );
    gltf.insert("nodes".into(), serde_json::json!([{ "mesh": 0 }]));
    gltf.insert("scenes".into(), serde_json::json!([{ "nodes": [0] }]));
    gltf.insert("scene".into(), serde_json::json!(0));
    if !images.is_empty() {
        gltf.insert("images".into(), serde_json::Value::Array(images));
    }
    if !samplers.is_empty() {
        gltf.insert("samplers".into(), serde_json::Value::Array(samplers));
    }
    if !textures.is_empty() {
        gltf.insert("textures".into(), serde_json::Value::Array(textures));
    }
    gltf.insert(
        "materials".into(),
        serde_json::Value::Array(vec![serde_json::Value::Object(gltf_material)]),
    );

    let mut json = serde_json::to_vec(&serde_json::Value::Object(gltf))?;
    // The JSON chunk pads with SPACES so the padding stays inside valid JSON
    // whitespace; the BIN chunk pads with zeros.
    json.extend(std::iter::repeat_n(b' ', (4 - (json.len() % 4)) % 4));

    let total = 12 + 8 + json.len() + 8 + bin.data.len();
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(GLB_MAGIC);
    glb.extend_from_slice(&GLB_VERSION.to_le_bytes());
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.data.len() as u32).to_le_bytes());
    glb.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    glb.extend_from_slice(&bin.data);
    debug_assert_eq!(glb.len(), total);
    Ok(glb)
}

/// Why a stored `.glb` cannot be read back for export.
///
/// Deliberately specific. An export refusal is a user-facing message about a
/// file they can see in their gallery, so "not a GLB" and "a GLB this reader
/// does not cover" have to be different sentences.
#[derive(Debug, thiserror::Error)]
pub enum GlbReadError {
    #[error("not a binary glTF file: {0}")]
    NotGlb(&'static str),
    #[error("unsupported binary glTF: {0}")]
    Unsupported(String),
    #[error("malformed binary glTF: {0}")]
    Malformed(String),
}

#[path = "glb_scene.rs"]
mod scene;

/// Flatten the selected static binary glTF scene into a [`Mesh`].
///
/// Applies parent transforms, inverse-transpose normals and reflected winding
/// to every triangle primitive. Reads embedded float attributes and unsigned
/// indices. Unsupported compression, posed geometry and external resources are
/// refused explicitly. Material images are not part of the returned geometry.
pub fn read_glb(bytes: &[u8]) -> Result<Mesh, GlbReadError> {
    let (json, bin) = split_glb_chunks(bytes)?;
    scene::read_mesh(&json, bin)
}

fn read_primitive(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
) -> Result<Mesh, GlbReadError> {
    if let Some(mode) = primitive.get("mode").and_then(serde_json::Value::as_u64) {
        if mode != u64::from(MODE_TRIANGLES) {
            return Err(GlbReadError::Unsupported(format!(
                "primitive mode {mode} is not triangles"
            )));
        }
    }
    let attributes = primitive
        .get("attributes")
        .and_then(serde_json::Value::as_object)
        .ok_or(GlbReadError::Malformed(
            "the primitive has no attributes".to_string(),
        ))?;

    let position_index = attributes
        .get("POSITION")
        .and_then(serde_json::Value::as_u64)
        .ok_or(GlbReadError::Unsupported(
            "the primitive has no POSITION attribute".to_string(),
        ))?;
    let vertices = read_vec3(json, bin, position_index, "POSITION")?;
    let normals = match attributes.get("NORMAL").and_then(serde_json::Value::as_u64) {
        Some(index) => Some(read_vec3(json, bin, index, "NORMAL")?),
        None => None,
    };
    let vertex_colors = match attributes
        .get("COLOR_0")
        .and_then(serde_json::Value::as_u64)
    {
        Some(index) => Some(read_vec3(json, bin, index, "COLOR_0")?),
        None => None,
    };
    let uvs = match attributes
        .get("TEXCOORD_0")
        .and_then(serde_json::Value::as_u64)
    {
        Some(index) => Some(read_vec2(json, bin, index, "TEXCOORD_0")?),
        None => None,
    };
    let faces = match primitive.get("indices").and_then(serde_json::Value::as_u64) {
        Some(index) => read_indices(json, bin, index)?,
        // An unindexed primitive is a legal glTF: consecutive triples.
        None => (0..vertices.len() as u32 / 3)
            .map(|triangle| [triangle * 3, triangle * 3 + 1, triangle * 3 + 2])
            .collect(),
    };

    for attribute in [
        ("NORMAL", normals.as_ref().map(Vec::len)),
        ("COLOR_0", vertex_colors.as_ref().map(Vec::len)),
    ]
    .into_iter()
    .filter_map(|(name, len)| len.map(|len| (name, len)))
    .chain(uvs.as_ref().map(|uvs| ("TEXCOORD_0", uvs.len())))
    {
        if attribute.1 != vertices.len() {
            return Err(GlbReadError::Malformed(format!(
                "{} has {} entries but the mesh has {} vertices",
                attribute.0,
                attribute.1,
                vertices.len()
            )));
        }
    }
    let mesh = Mesh {
        vertices,
        faces,
        normals,
        uvs,
        vertex_colors,
    };
    mesh.validate()
        .map_err(|error| GlbReadError::Malformed(error.to_string()))?;
    Ok(mesh)
}

fn split_glb_chunks(bytes: &[u8]) -> Result<(serde_json::Value, &[u8]), GlbReadError> {
    if bytes.len() < 12 {
        return Err(GlbReadError::NotGlb("the file is shorter than a header"));
    }
    if &bytes[0..4] != GLB_MAGIC {
        return Err(GlbReadError::NotGlb("the 'glTF' magic is missing"));
    }
    let u32_at = |offset: usize| -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
    };
    let version = u32_at(4);
    if version != GLB_VERSION {
        return Err(GlbReadError::Unsupported(format!(
            "glTF container version {version} (only version {GLB_VERSION} is read)"
        )));
    }
    let declared = u32_at(8) as usize;
    if declared > bytes.len() {
        return Err(GlbReadError::Malformed(format!(
            "the header declares {declared} bytes but the file is {}",
            bytes.len()
        )));
    }
    let bytes = &bytes[..declared];

    let mut json = None;
    let mut bin: &[u8] = &[];
    let mut cursor = 12usize;
    while cursor + 8 <= bytes.len() {
        let length = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .expect("4 bytes of chunk length"),
        ) as usize;
        let kind = u32::from_le_bytes(
            bytes[cursor + 4..cursor + 8]
                .try_into()
                .expect("4 bytes of chunk type"),
        );
        let start = cursor + 8;
        let end = start
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                GlbReadError::Malformed(format!("a chunk claims {length} bytes past the file end"))
            })?;
        match kind {
            CHUNK_JSON if json.is_none() => {
                json = Some(serde_json::from_slice(&bytes[start..end]).map_err(|error| {
                    GlbReadError::Malformed(format!("the JSON chunk does not parse: {error}"))
                })?);
            }
            CHUNK_BIN if bin.is_empty() => bin = &bytes[start..end],
            // Unknown chunk types are legal and must be skipped, per the spec.
            _ => {}
        }
        cursor = end;
    }
    let json = json.ok_or(GlbReadError::Malformed(
        "the file has no JSON chunk".to_string(),
    ))?;
    Ok((json, bin))
}

/// Resolve one accessor to its bytes, checking every constraint the reader
/// depends on rather than trusting the offsets.
fn accessor_bytes<'a>(
    json: &serde_json::Value,
    bin: &'a [u8],
    accessor_index: u64,
    expected_type: &str,
    expected_components: usize,
    component_size: usize,
) -> Result<(&'a [u8], usize, usize), GlbReadError> {
    let accessor = json
        .get("accessors")
        .and_then(|accessors| accessors.get(accessor_index as usize))
        .ok_or_else(|| GlbReadError::Malformed(format!("accessor {accessor_index} is missing")))?;
    let kind = accessor
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if kind != expected_type {
        return Err(GlbReadError::Unsupported(format!(
            "accessor {accessor_index} is {kind}, expected {expected_type}"
        )));
    }
    if accessor.get("sparse").is_some() {
        return Err(GlbReadError::Unsupported(
            "sparse accessors are not read".to_string(),
        ));
    }
    if accessor
        .get("normalized")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return Err(GlbReadError::Unsupported(
            "normalized accessors are not read".to_string(),
        ));
    }
    let count = accessor
        .get("count")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            GlbReadError::Malformed(format!("accessor {accessor_index} has no count"))
        })?;
    let view_index = accessor
        .get("bufferView")
        .and_then(serde_json::Value::as_u64)
        .ok_or(GlbReadError::Unsupported(
            "an accessor with no bufferView is not read".to_string(),
        ))?;
    let view = json
        .get("bufferViews")
        .and_then(|views| views.get(view_index as usize))
        .ok_or_else(|| GlbReadError::Malformed(format!("bufferView {view_index} is missing")))?;
    if view.get("buffer").and_then(serde_json::Value::as_u64) != Some(0) {
        return Err(GlbReadError::Unsupported(
            "only the embedded binary buffer is read".to_string(),
        ));
    }
    let stride = view
        .get("byteStride")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    let element = expected_components * component_size;
    if stride != 0
        && (stride < element
            || stride > 252
            || !stride.is_multiple_of(4)
            || expected_type == "SCALAR")
    {
        return Err(GlbReadError::Unsupported(format!(
            "invalid byteStride {stride}"
        )));
    }
    let stride = if stride == 0 { element } else { stride };
    // Every figure here is a JSON-supplied u64, so the arithmetic is checked
    // end to end: a foreign file with an absurd count or offset is a
    // Malformed error, never an overflow panic that `spawn_blocking` turns
    // into an anonymous 500.
    let view_offset = view
        .get("byteOffset")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let accessor_offset = accessor
        .get("byteOffset")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let out_of_range = || {
        GlbReadError::Malformed("an accessor reads past the end of the binary chunk".to_string())
    };
    let offset = view_offset
        .checked_add(accessor_offset)
        .ok_or_else(out_of_range)?;
    let view_length = view
        .get("byteLength")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| GlbReadError::Malformed("bufferView has no byteLength".into()))?;
    let view_end = view_offset
        .checked_add(view_length)
        .filter(|end| *end <= bin.len() as u64)
        .ok_or_else(out_of_range)?;
    if !offset.is_multiple_of(component_size as u64) {
        return Err(GlbReadError::Malformed("unaligned accessor offset".into()));
    }
    let span = if count == 0 {
        Some(0)
    } else {
        (count - 1)
            .checked_mul(stride as u64)
            .and_then(|size| size.checked_add(element as u64))
    }
    .ok_or_else(out_of_range)?;
    let end = offset
        .checked_add(span)
        .filter(|end| *end <= view_end)
        .ok_or_else(out_of_range)?;
    // Both fit in usize now: `end` is bounded by `bin.len()`.
    let (offset, end) = (offset as usize, end as usize);
    Ok((&bin[offset..end], count as usize, stride))
}

fn component_type(json: &serde_json::Value, accessor_index: u64) -> u32 {
    json.get("accessors")
        .and_then(|accessors| accessors.get(accessor_index as usize))
        .and_then(|accessor| accessor.get("componentType"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32
}

fn read_f32(bytes: &[u8], index: usize) -> f32 {
    let start = index * 4;
    f32::from_le_bytes(bytes[start..start + 4].try_into().expect("4 bytes of f32"))
}

fn read_vec3(
    json: &serde_json::Value,
    bin: &[u8],
    accessor_index: u64,
    name: &str,
) -> Result<Vec<[f32; 3]>, GlbReadError> {
    if component_type(json, accessor_index) != COMPONENT_FLOAT {
        return Err(GlbReadError::Unsupported(format!(
            "{name} is not stored as float"
        )));
    }
    let (bytes, count, stride) = accessor_bytes(json, bin, accessor_index, "VEC3", 3, 4)?;
    Ok((0..count)
        .map(|i| {
            let element = &bytes[i * stride..i * stride + 12];
            [
                read_f32(element, 0),
                read_f32(element, 1),
                read_f32(element, 2),
            ]
        })
        .collect())
}

fn read_vec2(
    json: &serde_json::Value,
    bin: &[u8],
    accessor_index: u64,
    name: &str,
) -> Result<Vec<[f32; 2]>, GlbReadError> {
    if component_type(json, accessor_index) != COMPONENT_FLOAT {
        return Err(GlbReadError::Unsupported(format!(
            "{name} is not stored as float"
        )));
    }
    let (bytes, count, stride) = accessor_bytes(json, bin, accessor_index, "VEC2", 2, 4)?;
    Ok((0..count)
        .map(|i| {
            let element = &bytes[i * stride..i * stride + 8];
            [read_f32(element, 0), read_f32(element, 1)]
        })
        .collect())
}

fn read_indices(
    json: &serde_json::Value,
    bin: &[u8],
    accessor_index: u64,
) -> Result<Vec<[u32; 3]>, GlbReadError> {
    let component = component_type(json, accessor_index);
    let size = match component {
        COMPONENT_UNSIGNED_INT => 4,
        COMPONENT_UNSIGNED_SHORT => 2,
        COMPONENT_UNSIGNED_BYTE => 1,
        other => {
            return Err(GlbReadError::Unsupported(format!(
                "index component type {other} is not read (expected unsigned byte, short, or int)"
            )))
        }
    };
    let (bytes, count, stride) = accessor_bytes(json, bin, accessor_index, "SCALAR", 1, size)?;
    if !count.is_multiple_of(3) {
        return Err(GlbReadError::Malformed(format!(
            "the index count ({count}) is not a whole number of triangles"
        )));
    }
    let at = |i: usize| -> u32 {
        let start = i * stride;
        match size {
            4 => u32::from_le_bytes(bytes[start..start + 4].try_into().expect("4 bytes")),
            2 => u32::from(u16::from_le_bytes(
                bytes[start..start + 2].try_into().expect("2 bytes"),
            )),
            _ => u32::from(bytes[start]),
        }
    };
    Ok((0..count / 3)
        .map(|triangle| [at(triangle * 3), at(triangle * 3 + 1), at(triangle * 3 + 2)])
        .collect())
}

/// Serialize `mesh` as binary STL.
///
/// STL has no vertex identity, no UVs and no colour — every triangle carries
/// its three positions and one facet normal — so this is the LOSSIEST of the
/// exports and exists because it is what 3-D printers and CAD tools read. The
/// facet normal is recomputed from the triangle's own winding rather than
/// averaged from the vertex normals, which is what the format means by it; a
/// degenerate triangle gets a zero normal, the conventional "ask the viewer to
/// derive it" value.
pub fn write_stl(mesh: &Mesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + mesh.faces.len() * 50);
    // The 80-byte header is free-form and must NOT begin with "solid", which
    // is how readers detect the ASCII dialect.
    let mut header = [0u8; 80];
    let banner = b"mold";
    header[..banner.len()].copy_from_slice(banner);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(mesh.faces.len() as u32).to_le_bytes());
    for face in &mesh.faces {
        let vertices = face.map(|index| mesh.vertices[index as usize]);
        let normal = facet_normal(&vertices);
        for value in normal {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for vertex in vertices {
            for value in vertex {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        // Attribute byte count. Zero everywhere outside the unofficial
        // per-face colour extensions.
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// The unit normal of one triangle, or zero for a degenerate one.
fn facet_normal(triangle: &[[f32; 3]; 3]) -> [f32; 3] {
    let [a, b, c] = triangle;
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length.is_finite() && length > 0.0 {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Serialize `mesh` as binary little-endian PLY.
///
/// Keeps shared vertices, unlike STL, and carries per-vertex normals when the
/// mesh has them. The header is ASCII and the payload is packed little-endian
/// with no padding, which is what every PLY reader expects from
/// `format binary_little_endian 1.0`.
pub fn write_ply(mesh: &Mesh) -> Vec<u8> {
    let normals = mesh
        .normals
        .as_ref()
        .filter(|normals| normals.len() == mesh.vertices.len());
    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\ncomment generated by mold\n");
    header.push_str(&format!("element vertex {}\n", mesh.vertices.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    if normals.is_some() {
        header.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    header.push_str(&format!("element face {}\n", mesh.faces.len()));
    header.push_str("property list uchar int vertex_indices\nend_header\n");

    let vertex_stride = if normals.is_some() { 24 } else { 12 };
    let mut out = Vec::with_capacity(
        header.len() + mesh.vertices.len() * vertex_stride + mesh.faces.len() * 13,
    );
    out.extend_from_slice(header.as_bytes());
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        for value in vertex {
            out.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(normals) = normals {
            for value in &normals[index] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    for face in &mesh.faces {
        out.push(3);
        for index in face {
            // `property list uchar int` — a SIGNED 32-bit index, which is what
            // every reader expects here even though an index is never
            // negative.
            out.extend_from_slice(&(*index as i32).to_le_bytes());
        }
    }
    out
}

/// Material name emitted by [`write_mtl`] and referenced by
/// [`write_obj_with_mtl`].
pub const OBJ_MATERIAL_NAME: &str = "mold";

/// Serialize `mesh` as Wavefront OBJ. Indices are 1-based, as the format
/// requires.
pub fn write_obj(mesh: &Mesh) -> String {
    write_obj_inner(mesh, None)
}

/// [`write_obj`] with a leading `mtllib`/`usemtl` pair pointing at the `.mtl`
/// companion written by [`write_mtl`].
pub fn write_obj_with_mtl(mesh: &Mesh, mtl_filename: &str) -> String {
    write_obj_inner(mesh, Some(mtl_filename))
}

fn write_obj_inner(mesh: &Mesh, mtl_filename: Option<&str>) -> String {
    let mut out = String::with_capacity(mesh.vertices.len() * 40 + mesh.faces.len() * 24);
    out.push_str("# generated by mold\n");
    if let Some(mtl) = mtl_filename {
        out.push_str(&format!("mtllib {mtl}\nusemtl {OBJ_MATERIAL_NAME}\n"));
    }
    for (i, v) in mesh.vertices.iter().enumerate() {
        // The `v x y z r g b` vertex-color extension is understood by Blender,
        // MeshLab and assimp; plain readers ignore the trailing three floats.
        match mesh.vertex_colors.as_ref().and_then(|c| c.get(i)) {
            Some(c) => out.push_str(&format!(
                "v {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n",
                v[0], v[1], v[2], c[0], c[1], c[2]
            )),
            None => out.push_str(&format!("v {:.6} {:.6} {:.6}\n", v[0], v[1], v[2])),
        }
    }
    let has_uvs = mesh
        .uvs
        .as_ref()
        .is_some_and(|u| u.len() == mesh.vertices.len());
    if has_uvs {
        for uv in mesh.uvs.as_ref().expect("checked above") {
            out.push_str(&format!("vt {:.6} {:.6}\n", uv[0], uv[1]));
        }
    }
    let has_normals = mesh
        .normals
        .as_ref()
        .is_some_and(|n| n.len() == mesh.vertices.len());
    if has_normals {
        for n in mesh.normals.as_ref().expect("checked above") {
            out.push_str(&format!("vn {:.6} {:.6} {:.6}\n", n[0], n[1], n[2]));
        }
    }
    for face in &mesh.faces {
        out.push('f');
        for &idx in face {
            let i = idx as usize + 1;
            match (has_uvs, has_normals) {
                (true, true) => out.push_str(&format!(" {i}/{i}/{i}")),
                (true, false) => out.push_str(&format!(" {i}/{i}")),
                (false, true) => out.push_str(&format!(" {i}//{i}")),
                (false, false) => out.push_str(&format!(" {i}")),
            }
        }
        out.push('\n');
    }
    out
}

/// The `.mtl` companion for [`write_obj_with_mtl`].
///
/// `base_color_texture_file` is the filename the caller will write the
/// baseColor PNG to alongside the OBJ; OBJ cannot embed images the way GLB
/// can, so the texture has to live in a sibling file.
pub fn write_mtl(material: &GlbMaterial, base_color_texture_file: Option<&str>) -> String {
    let base = material
        .base_color_factor
        .unwrap_or([0.22, 0.22, 0.22, 1.0]);
    let roughness = material.roughness_factor.unwrap_or(0.5);
    let mut out = String::new();
    out.push_str("# generated by mold\n");
    out.push_str(&format!("newmtl {OBJ_MATERIAL_NAME}\n"));
    out.push_str(&format!(
        "Kd {:.6} {:.6} {:.6}\n",
        base[0], base[1], base[2]
    ));
    out.push_str("Ka 0.000000 0.000000 0.000000\n");
    out.push_str("Ks 0.000000 0.000000 0.000000\n");
    out.push_str(&format!("d {:.6}\n", base[3]));
    // illum 2 = colour on, ambient on, specular on.
    out.push_str("illum 2\n");
    // PBR extension keywords (Pr/Pm), understood by Blender's OBJ importer.
    out.push_str(&format!("Pr {roughness:.6}\n"));
    out.push_str(&format!(
        "Pm {:.6}\n",
        material.metallic_factor.unwrap_or(0.0)
    ));
    if let Some(file) = base_color_texture_file {
        out.push_str(&format!("map_Kd {file}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan3d::mesh::{extract, MeshAlgorithm, OccupancyGrid};

    /// Minimal GLB reader, written here rather than pulled in as a dependency
    /// so the test exercises the bytes and not a shared helper.
    struct ParsedGlb {
        version: u32,
        total_len: u32,
        json: serde_json::Value,
        bin: Vec<u8>,
    }

    fn parse_glb(bytes: &[u8]) -> ParsedGlb {
        assert!(bytes.len() >= 12, "truncated header");
        assert_eq!(&bytes[0..4], b"glTF", "bad magic");
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let version = u32_at(4);
        let total_len = u32_at(8);
        assert_eq!(
            total_len as usize,
            bytes.len(),
            "header length != file size"
        );

        let json_len = u32_at(12) as usize;
        assert_eq!(u32_at(16), 0x4E4F_534A, "chunk 0 is not JSON");
        assert_eq!(json_len % 4, 0, "JSON chunk not 4-byte aligned");
        let json_start = 20;
        let json_bytes = &bytes[json_start..json_start + json_len];
        let json: serde_json::Value =
            serde_json::from_slice(json_bytes).expect("JSON chunk does not parse");

        let bin_header = json_start + json_len;
        let bin_len =
            u32::from_le_bytes(bytes[bin_header..bin_header + 4].try_into().unwrap()) as usize;
        assert_eq!(
            u32::from_le_bytes(bytes[bin_header + 4..bin_header + 8].try_into().unwrap()),
            0x004E_4942,
            "chunk 1 is not BIN"
        );
        assert_eq!(bin_len % 4, 0, "BIN chunk not 4-byte aligned");
        let bin = bytes[bin_header + 8..bin_header + 8 + bin_len].to_vec();
        assert_eq!(bin_header + 8 + bin_len, bytes.len(), "trailing bytes");
        ParsedGlb {
            version,
            total_len,
            json,
            bin,
        }
    }

    fn sphere_mesh() -> Mesh {
        let n = 20usize;
        let c = (n as f32 - 1.0) / 2.0;
        let mut logits = Vec::with_capacity(n * n * n);
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    let d = ((i0 as f32 - c).powi(2)
                        + (i1 as f32 - c).powi(2)
                        + (i2 as f32 - c).powi(2))
                    .sqrt();
                    logits.push(7.0 - d);
                }
            }
        }
        let grid = OccupancyGrid::new(logits, [n, n, n]).unwrap();
        extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut |_, _| Ok(())).unwrap()
    }

    /// The writer stores f32 factors, which widen to f64 in JSON, so an exact
    /// comparison against a decimal literal fails on values like 0.22.
    fn assert_factor_array(actual: &serde_json::Value, expected: [f64; 4]) {
        let got: Vec<f64> = actual
            .as_array()
            .unwrap_or_else(|| panic!("expected an array, got {actual}"))
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(got.len(), 4, "expected 4 factors, got {got:?}");
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {got:?}, expected {expected:?}");
        }
    }

    fn triangle_mesh() -> Mesh {
        Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        }
    }

    #[test]
    fn glb_round_trips_header_json_and_accessors() {
        let mesh = sphere_mesh();
        assert!(mesh.vertex_count() > 0);
        let bytes = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = parse_glb(&bytes);

        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.total_len as usize, bytes.len());

        let json = &parsed.json;
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["scene"], 0);

        let accessors = json["accessors"].as_array().unwrap();
        assert_eq!(accessors[0]["count"], mesh.vertex_count());
        assert_eq!(accessors[0]["type"], "VEC3");
        assert_eq!(accessors[1]["count"], mesh.face_count() * 3);
        assert_eq!(accessors[1]["componentType"], 5125); // UNSIGNED_INT

        // glTF requires min/max on POSITION, and they must be the real bounds.
        let (min, max) = mesh.bounds();
        let json_min: Vec<f64> = accessors[0]["min"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let json_max: Vec<f64> = accessors[0]["max"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for a in 0..3 {
            assert!((json_min[a] - min[a] as f64).abs() < 1e-6);
            assert!((json_max[a] - max[a] as f64).abs() < 1e-6);
        }

        // The declared buffer length must equal the BIN chunk we actually
        // wrote, and every bufferView must fit inside it.
        let declared = json["buffers"][0]["byteLength"].as_u64().unwrap() as usize;
        assert_eq!(declared, parsed.bin.len());
        for view in json["bufferViews"].as_array().unwrap() {
            let off = view["byteOffset"].as_u64().unwrap() as usize;
            let len = view["byteLength"].as_u64().unwrap() as usize;
            assert!(off + len <= parsed.bin.len());
            assert_eq!(off % 4, 0, "bufferView offset must be 4-byte aligned");
        }

        // Positions round-trip through the BIN chunk unchanged.
        let pos_view = &json["bufferViews"][0];
        let off = pos_view["byteOffset"].as_u64().unwrap() as usize;
        let first = f32::from_le_bytes(parsed.bin[off..off + 4].try_into().unwrap());
        assert_eq!(first, mesh.vertices[0][0]);
    }

    #[test]
    fn glb_writes_normals_and_declares_them() {
        let mut mesh = sphere_mesh();
        crate::hunyuan3d::mesh::compute_smooth_normals(&mut mesh);
        let bytes = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = parse_glb(&bytes);
        let attrs = &parsed.json["meshes"][0]["primitives"][0]["attributes"];
        let normal_accessor = attrs["NORMAL"].as_u64().unwrap() as usize;
        assert_eq!(
            parsed.json["accessors"][normal_accessor]["count"],
            mesh.vertex_count()
        );
        assert_eq!(parsed.json["accessors"][normal_accessor]["type"], "VEC3");
    }

    #[test]
    fn glb_embeds_textures_and_wires_the_material() {
        let mut mesh = triangle_mesh();
        mesh.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        // Not a real PNG — the writer stores the bytes verbatim and never
        // decodes them, so any payload proves the plumbing.
        let material = GlbMaterial {
            base_color_texture: Some(vec![0xAB; 7]),
            metallic_roughness_texture: Some(vec![0xCD; 5]),
            normal_texture: Some(vec![0xEF; 3]),
            occlusion_in_metallic_roughness: true,
            ..GlbMaterial::default()
        };
        let bytes = write_glb(&mesh, &material, None).unwrap();
        let parsed = parse_glb(&bytes);
        let json = &parsed.json;

        assert_eq!(json["images"].as_array().unwrap().len(), 3);
        assert_eq!(json["textures"].as_array().unwrap().len(), 3);
        assert_eq!(json["samplers"].as_array().unwrap().len(), 1);
        let mat = &json["materials"][0];
        assert!(mat["pbrMetallicRoughness"]["baseColorTexture"].is_object());
        assert!(mat["pbrMetallicRoughness"]["metallicRoughnessTexture"].is_object());
        assert!(mat["normalTexture"].is_object());
        // ORM packing points occlusionTexture at the MR image.
        assert_eq!(
            mat["occlusionTexture"]["index"],
            mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
        );
        // A baseColor texture flips the auto factors to pass-through white.
        assert_factor_array(
            &mat["pbrMetallicRoughness"]["baseColorFactor"],
            [1.0, 1.0, 1.0, 1.0],
        );
        // An MR texture makes both factors 1.0 so the texture is unscaled.
        assert_eq!(mat["pbrMetallicRoughness"]["metallicFactor"], 1.0);
        assert_eq!(mat["pbrMetallicRoughness"]["roughnessFactor"], 1.0);
    }

    #[test]
    fn glb_textures_need_uvs() {
        let mesh = triangle_mesh(); // no uvs
        let material = GlbMaterial {
            base_color_texture: Some(vec![0xAB; 7]),
            ..GlbMaterial::default()
        };
        let parsed = parse_glb(&write_glb(&mesh, &material, None).unwrap());
        // Without TEXCOORD_0 there is nothing to sample the texture with, so it
        // is dropped rather than referenced from an unusable material.
        assert!(parsed.json.get("images").is_none());
        assert_factor_array(
            &parsed.json["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            [0.22, 0.22, 0.22, 1.0],
        );
    }

    #[test]
    fn glb_material_factors_override_the_auto_values() {
        let material = GlbMaterial {
            base_color_factor: Some([0.1, 0.2, 0.3, 0.4]),
            metallic_factor: Some(0.75),
            roughness_factor: Some(0.25),
            double_sided: false,
            ..GlbMaterial::default()
        };
        let parsed = parse_glb(&write_glb(&triangle_mesh(), &material, None).unwrap());
        let pbr = &parsed.json["materials"][0]["pbrMetallicRoughness"];
        assert_factor_array(&pbr["baseColorFactor"], [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(pbr["metallicFactor"].as_f64().unwrap(), 0.75);
        assert_eq!(pbr["roughnessFactor"].as_f64().unwrap(), 0.25);
        assert_eq!(parsed.json["materials"][0]["doubleSided"], false);
    }

    #[test]
    fn glb_carries_metadata_as_asset_extras() {
        let metadata = serde_json::json!({ "model": "hunyuan3d-2", "seed": 42 });
        let parsed = parse_glb(
            &write_glb(&triangle_mesh(), &GlbMaterial::default(), Some(&metadata)).unwrap(),
        );
        assert_eq!(parsed.json["asset"]["extras"], metadata);
    }

    #[test]
    fn glb_rejects_an_empty_mesh_instead_of_writing_a_corrupt_file() {
        let err = write_glb(&Mesh::default(), &GlbMaterial::default(), None).unwrap_err();
        assert!(
            err.downcast_ref::<GlbError>()
                .is_some_and(|e| matches!(e, GlbError::EmptyMesh)),
            "expected GlbError::EmptyMesh, got {err}"
        );
    }

    #[test]
    fn glb_rejects_out_of_range_face_indices() {
        let mesh = Mesh {
            vertices: vec![[0.0; 3]; 3],
            faces: vec![[0, 1, 9]],
            ..Mesh::default()
        };
        let err = write_glb(&mesh, &GlbMaterial::default(), None).unwrap_err();
        assert!(
            err.downcast_ref::<GlbError>()
                .is_some_and(|e| matches!(e, GlbError::InvalidMesh(_))),
            "expected GlbError::InvalidMesh, got {err}"
        );
    }

    #[test]
    fn obj_is_one_based_and_line_counts_match() {
        let mesh = triangle_mesh();
        let obj = write_obj(&mesh);
        let v_lines: Vec<&str> = obj.lines().filter(|l| l.starts_with("v ")).collect();
        let f_lines: Vec<&str> = obj.lines().filter(|l| l.starts_with('f')).collect();
        assert_eq!(v_lines.len(), mesh.vertex_count());
        assert_eq!(f_lines.len(), mesh.face_count());
        assert_eq!(f_lines[0], "f 1 2 3");
        assert!(!obj.contains("mtllib"));
    }

    #[test]
    fn obj_emits_uv_normal_and_color_channels() {
        let mut mesh = triangle_mesh();
        mesh.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        mesh.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        mesh.vertex_colors = Some(vec![[1.0, 0.0, 0.0]; 3]);
        let obj = write_obj_with_mtl(&mesh, "model.mtl");
        assert!(obj.contains("mtllib model.mtl"));
        assert!(obj.contains(&format!("usemtl {OBJ_MATERIAL_NAME}")));
        assert_eq!(obj.lines().filter(|l| l.starts_with("vt ")).count(), 3);
        assert_eq!(obj.lines().filter(|l| l.starts_with("vn ")).count(), 3);
        assert!(obj.contains("v 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000"));
        assert!(obj.lines().any(|l| l == "f 1/1/1 2/2/2 3/3/3"));
    }

    #[test]
    fn mtl_names_the_material_and_optional_texture() {
        let mtl = write_mtl(&GlbMaterial::default(), None);
        assert!(mtl.contains(&format!("newmtl {OBJ_MATERIAL_NAME}")));
        assert!(mtl.contains("Kd 0.220000 0.220000 0.220000"));
        assert!(!mtl.contains("map_Kd"));

        let textured = write_mtl(&GlbMaterial::default(), Some("basecolor.png"));
        assert!(textured.contains("map_Kd basecolor.png"));
    }

    /// Two triangles sharing an edge: small enough to write the expected
    /// bytes out by hand, big enough to exercise shared vertices.
    fn two_triangle_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
            uvs: None,
            vertex_colors: None,
        }
    }

    /// Rebuild a GLB container around an edited JSON chunk, so a test can
    /// hand `read_glb` a layout the writer would never produce.
    fn rebuild_glb(json: &serde_json::Value, bin: &[u8]) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(json).unwrap();
        json_bytes.extend(std::iter::repeat_n(b' ', (4 - (json_bytes.len() % 4)) % 4));
        let mut bin = bin.to_vec();
        bin.extend(std::iter::repeat_n(0u8, (4 - (bin.len() % 4)) % 4));
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    /// The export path is `write_glb` -> stored file -> `read_glb` -> writer,
    /// so the reader has to recover exactly what the writer put in.
    #[test]
    fn glb_round_trips_through_read_glb() {
        let mut mesh = two_triangle_mesh();
        mesh.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        mesh.vertex_colors = Some(vec![[0.25, 0.5, 0.75]; 4]);
        let bytes = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = read_glb(&bytes).unwrap();
        assert_eq!(parsed, mesh);
    }

    #[test]
    fn read_glb_flattens_all_primitives_through_parent_transforms() {
        let mesh = triangle_mesh();
        let parsed = parse_glb(&write_glb(&mesh, &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        let primitive = json["meshes"][0]["primitives"][0].clone();
        json["meshes"][0]["primitives"] = serde_json::json!([primitive, primitive]);
        json["nodes"] = serde_json::json!([
            {"translation": [10., 0., 0.], "children": [1]},
            {"mesh": 0, "translation": [0., 3., 0.], "scale": [2., 2., 2.]}
        ]);
        json["scenes"] = serde_json::json!([{"nodes": [0]}]);
        json["scene"] = serde_json::json!(0);
        let output = read_glb(&rebuild_glb(&json, &parsed.bin)).unwrap();
        assert_eq!(output.faces.len(), mesh.faces.len() * 2);
        assert_eq!(output.vertices.len(), mesh.vertices.len() * 2);
        for (input, output) in mesh.vertices.iter().zip(&output.vertices) {
            assert_eq!(
                *output,
                [input[0] * 2. + 10., input[1] * 2. + 3., input[2] * 2.]
            );
        }
    }

    #[test]
    fn read_glb_respects_the_selected_scene() {
        let parsed =
            parse_glb(&write_glb(&triangle_mesh(), &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        json["nodes"] = serde_json::json!([
            {"mesh": 0, "translation": [9., 0., 0.]}, {"mesh": 0, "translation": [2., 0., 0.]}
        ]);
        json["scenes"] = serde_json::json!([{"nodes": [0]}, {"nodes": [1]}]);
        json["scene"] = serde_json::json!(1);
        let output = read_glb(&rebuild_glb(&json, &parsed.bin)).unwrap();
        assert_eq!(output.vertices[0][0], triangle_mesh().vertices[0][0] + 2.);
    }

    #[test]
    fn read_glb_refuses_cycles_skinning_and_unknown_required_extensions() {
        let parsed =
            parse_glb(&write_glb(&triangle_mesh(), &GlbMaterial::default(), None).unwrap());
        for change in [
            (|json: &mut serde_json::Value| {
                json["nodes"][0]["children"] = serde_json::json!([0]);
            }) as fn(&mut serde_json::Value),
            |json| {
                json["nodes"][0]["skin"] = serde_json::json!(0);
            },
            |json| {
                json["extensionsRequired"] = serde_json::json!(["KHR_draco_mesh_compression"]);
            },
            |json| {
                json["nodes"][0]["matrix"] =
                    serde_json::json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
                json["nodes"][0]["translation"] = serde_json::json!([1, 2, 3]);
            },
        ] {
            let mut json = parsed.json.clone();
            change(&mut json);
            assert!(read_glb(&rebuild_glb(&json, &parsed.bin)).is_err());
        }
    }

    #[test]
    fn read_glb_transforms_normals_and_reflected_winding() {
        let mut mesh = triangle_mesh();
        let n = 1.0 / 2.0f32.sqrt();
        mesh.normals = Some(vec![[n, n, 0.]; mesh.vertices.len()]);
        let parsed = parse_glb(&write_glb(&mesh, &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        json["nodes"][0]["scale"] = serde_json::json!([-1., 2., 3.]);
        let output = read_glb(&rebuild_glb(&json, &parsed.bin)).unwrap();
        assert_eq!(
            output.faces[0],
            [mesh.faces[0][0], mesh.faces[0][2], mesh.faces[0][1]]
        );
        let actual = output.normals.unwrap()[0];
        assert!((actual[0] + 2.0 / 5.0f32.sqrt()).abs() < 1e-6);
        assert!((actual[1] - 1.0 / 5.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn read_glb_refuses_shear_and_nonfinite_vertex_attributes() {
        let mut mesh = triangle_mesh();
        mesh.normals = Some(vec![[0., 0., 1.]; mesh.vertices.len()]);
        let parsed = parse_glb(&write_glb(&mesh, &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        json["nodes"][0]["matrix"] =
            serde_json::json!([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        assert!(read_glb(&rebuild_glb(&json, &parsed.bin)).is_err());
        let accessor = parsed.json["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
            .as_u64()
            .unwrap() as usize;
        let view = parsed.json["accessors"][accessor]["bufferView"]
            .as_u64()
            .unwrap() as usize;
        let offset = parsed.json["bufferViews"][view]["byteOffset"]
            .as_u64()
            .unwrap() as usize;
        let mut bin = parsed.bin.clone();
        bin[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_glb(&rebuild_glb(&parsed.json, &bin)).is_err());
    }

    /// A real extracted mesh, not a hand-built one: the writer is fed the
    /// surface-net output and the reader must recover every vertex.
    #[test]
    fn an_extracted_surface_round_trips_through_read_glb() {
        let mut values = vec![0.0f32; 6 * 6 * 6];
        for z in 1..5 {
            for y in 1..5 {
                for x in 1..5 {
                    values[(z * 6 + y) * 6 + x] = 1.0;
                }
            }
        }
        let grid = OccupancyGrid::new(values, [6, 6, 6]).unwrap();
        let mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.5, &mut |_, _| Ok(())).unwrap();
        assert!(mesh.face_count() > 0);
        let bytes = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = read_glb(&bytes).unwrap();
        assert_eq!(parsed.vertices, mesh.vertices);
        assert_eq!(parsed.faces, mesh.faces);
    }

    /// Every refusal is by NAME. An export error is shown to a user looking
    /// at a file in their own gallery, so "not a GLB" and "a GLB shape this
    /// reader does not cover" must not collapse into one message.
    #[test]
    fn read_glb_accessor_cannot_escape_its_buffer_view() {
        let parsed =
            parse_glb(&write_glb(&triangle_mesh(), &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        json["bufferViews"][0]["byteLength"] = serde_json::json!(4);
        assert!(read_glb(&rebuild_glb(&json, &parsed.bin)).is_err());
    }

    #[test]
    fn read_glb_reads_interleaved_positions_without_padding() {
        let mesh = triangle_mesh();
        let parsed = parse_glb(&write_glb(&mesh, &GlbMaterial::default(), None).unwrap());
        let mut json = parsed.json.clone();
        let mut bin = parsed.bin.clone();
        let offset = bin.len();
        for position in &mesh.vertices {
            for component in position {
                bin.extend_from_slice(&component.to_le_bytes());
            }
            bin.extend_from_slice(&f32::NAN.to_le_bytes());
        }
        json["bufferViews"][0]["byteOffset"] = serde_json::json!(offset);
        json["bufferViews"][0]["byteLength"] = serde_json::json!(mesh.vertices.len() * 16);
        json["bufferViews"][0]["byteStride"] = serde_json::json!(16);
        json["buffers"][0]["byteLength"] = serde_json::json!(bin.len());
        assert_eq!(
            read_glb(&rebuild_glb(&json, &bin)).unwrap().vertices,
            mesh.vertices
        );
    }

    #[test]
    fn read_glb_refuses_foreign_layouts() {
        assert!(matches!(
            read_glb(b"not a glb at all"),
            Err(GlbReadError::NotGlb(_))
        ));
        assert!(matches!(read_glb(&[]), Err(GlbReadError::NotGlb(_))));

        let mesh = two_triangle_mesh();
        let good = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();

        let mut wrong_version = good.clone();
        wrong_version[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            read_glb(&wrong_version),
            Err(GlbReadError::Unsupported(_))
        ));

        let mut truncated = good.clone();
        truncated.truncate(good.len() - 8);
        assert!(read_glb(&truncated).is_err());

        let parsed = parse_glb(&good);
        for mutate in [
            (|json: &mut serde_json::Value| {
                json["meshes"][0]["primitives"][0]["mode"] = serde_json::json!(1);
            }) as fn(&mut serde_json::Value),
            |json: &mut serde_json::Value| {
                json["meshes"][0]["primitives"][0]["attributes"]
                    .as_object_mut()
                    .unwrap()
                    .remove("POSITION");
            },
            |json: &mut serde_json::Value| {
                json["accessors"][0]["componentType"] = serde_json::json!(5120);
            },
            |json: &mut serde_json::Value| {
                json["bufferViews"][0]["byteStride"] = serde_json::json!(10);
            },
        ] {
            let mut json = parsed.json.clone();
            mutate(&mut json);
            let rebuilt = rebuild_glb(&json, &parsed.bin);
            assert!(
                matches!(read_glb(&rebuilt), Err(GlbReadError::Unsupported(_))),
                "a foreign layout must be refused as unsupported"
            );
        }
    }

    /// A foreign file can name any count or offset a u64 holds. Each of
    /// these used to be unchecked arithmetic — a debug overflow panic, or a
    /// release wrap into an empty slice and a capacity-overflow panic in the
    /// collect — which `spawn_blocking` surfaced as an anonymous 500. They
    /// are all the named Malformed error now.
    #[test]
    fn read_glb_refuses_absurd_accessor_counts_and_offsets_without_panicking() {
        let mesh = two_triangle_mesh();
        let good = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = parse_glb(&good);
        for mutate in [
            (|json: &mut serde_json::Value| {
                json["accessors"][0]["count"] = serde_json::json!(u32::MAX);
            }) as fn(&mut serde_json::Value),
            |json: &mut serde_json::Value| {
                json["accessors"][0]["count"] = serde_json::json!(u64::MAX);
            },
            |json: &mut serde_json::Value| {
                json["bufferViews"][0]["byteOffset"] = serde_json::json!(u64::MAX - 4);
                json["accessors"][0]["byteOffset"] = serde_json::json!(8);
            },
            |json: &mut serde_json::Value| {
                json["bufferViews"][0]["byteOffset"] = serde_json::json!(u64::MAX);
            },
            |json: &mut serde_json::Value| {
                json["accessors"][0]["byteOffset"] = serde_json::json!(u64::MAX);
            },
        ] {
            let mut json = parsed.json.clone();
            mutate(&mut json);
            let rebuilt = rebuild_glb(&json, &parsed.bin);
            assert!(
                matches!(read_glb(&rebuilt), Err(GlbReadError::Malformed(_))),
                "an out-of-range accessor must be refused as malformed"
            );
        }
    }

    /// `u16` indices are what most exporters emit for a small mesh, so the
    /// reader takes them even though mold itself always writes `u32`.
    #[test]
    fn read_glb_accepts_short_indices() {
        let mesh = two_triangle_mesh();
        let good = write_glb(&mesh, &GlbMaterial::default(), None).unwrap();
        let parsed = parse_glb(&good);
        let mut json = parsed.json.clone();
        json["accessors"][1]["componentType"] = serde_json::json!(5123);
        let mut bin = parsed.bin.clone();
        let offset = json["bufferViews"][1]["byteOffset"].as_u64().unwrap() as usize;
        let mut shorts = Vec::new();
        for face in &mesh.faces {
            for index in face {
                shorts.extend_from_slice(&(*index as u16).to_le_bytes());
            }
        }
        json["bufferViews"][1]["byteLength"] = serde_json::json!(shorts.len());
        bin[offset..offset + shorts.len()].copy_from_slice(&shorts);
        let rebuilt = rebuild_glb(&json, &bin);
        assert_eq!(read_glb(&rebuilt).unwrap().faces, mesh.faces);
    }

    /// Binary STL is a fixed 84 + 50n layout, so the whole file is pinned
    /// byte for byte rather than sampled.
    #[test]
    fn write_stl_emits_the_exact_binary_layout() {
        let stl = write_stl(&two_triangle_mesh());
        assert_eq!(stl.len(), 84 + 2 * 50);
        assert_eq!(&stl[..4], b"mold");
        // The header must NOT start with "solid" — that is how a reader
        // detects the ASCII dialect.
        assert!(!stl.starts_with(b"solid"));
        assert!(stl[4..80].iter().all(|byte| *byte == 0));
        assert_eq!(u32::from_le_bytes(stl[80..84].try_into().unwrap()), 2);

        let mut expected = Vec::new();
        for triangle in [
            [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0f32, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ] {
            // Both faces wind counter-clockwise in the z = 0 plane, so the
            // facet normal is +z.
            for value in [0.0f32, 0.0, 1.0] {
                expected.extend_from_slice(&value.to_le_bytes());
            }
            for vertex in triangle {
                for value in vertex {
                    expected.extend_from_slice(&value.to_le_bytes());
                }
            }
            expected.extend_from_slice(&0u16.to_le_bytes());
        }
        assert_eq!(&stl[84..], &expected[..]);
    }

    #[test]
    fn write_stl_gives_a_degenerate_triangle_a_zero_normal() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            faces: vec![[0, 1, 2]],
            normals: None,
            uvs: None,
            vertex_colors: None,
        };
        let stl = write_stl(&mesh);
        for lane in 0..3 {
            let start = 84 + lane * 4;
            assert_eq!(
                f32::from_le_bytes(stl[start..start + 4].try_into().unwrap()),
                0.0
            );
        }
    }

    fn ply_header_len(ply: &[u8]) -> usize {
        let terminator = b"end_header\n";
        ply.windows(terminator.len())
            .position(|window| window == terminator)
            .expect("the header terminator is present")
            + terminator.len()
    }

    #[test]
    fn write_ply_emits_the_exact_binary_layout() {
        let ply = write_ply(&two_triangle_mesh());
        let split = ply_header_len(&ply);
        let header = std::str::from_utf8(&ply[..split]).unwrap();
        assert_eq!(
            header,
            "ply\nformat binary_little_endian 1.0\ncomment generated by mold\n\
             element vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
             property float nx\nproperty float ny\nproperty float nz\n\
             element face 2\nproperty list uchar int vertex_indices\nend_header\n"
        );

        let mut expected = Vec::new();
        for (vertex, normal) in [
            ([0.0f32, 0.0, 0.0], [0.0f32, 0.0, 1.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ] {
            for value in vertex.iter().chain(normal.iter()) {
                expected.extend_from_slice(&value.to_le_bytes());
            }
        }
        for face in [[0i32, 1, 2], [0, 2, 3]] {
            expected.push(3);
            for index in face {
                expected.extend_from_slice(&index.to_le_bytes());
            }
        }
        assert_eq!(&ply[split..], &expected[..]);
    }

    /// Without normals the vertex record is three floats, and the header
    /// must not promise properties the payload does not carry.
    #[test]
    fn write_ply_omits_normals_it_does_not_have() {
        let mut mesh = two_triangle_mesh();
        mesh.normals = None;
        let ply = write_ply(&mesh);
        let split = ply_header_len(&ply);
        let header = std::str::from_utf8(&ply[..split]).unwrap();
        assert!(!header.contains("property float nx"), "{header}");
        assert_eq!(ply.len(), split + 4 * 12 + 2 * 13);
    }
}
