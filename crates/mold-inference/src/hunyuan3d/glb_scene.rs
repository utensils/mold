//! Static scene flattening, following glTF 2.0 §3.5.3 and §5.25:
//! https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#transformations
//! Local TRS is T*R*S; world is parent*local. Matrices are column-major.
use super::{read_primitive, GlbReadError, Mesh};
use serde_json::Value;

type Result<T> = std::result::Result<T, GlbReadError>;
type Matrix = [[f64; 4]; 4];
const IDENTITY: Matrix = [
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
];
const MAX_NODES: usize = 8192;
const MAX_FACES: usize = 8_000_000;
const MAX_VERTICES: usize = MAX_FACES * 3;
fn malformed(message: impl Into<String>) -> GlbReadError {
    GlbReadError::Malformed(message.into())
}
fn unsupported(message: impl Into<String>) -> GlbReadError {
    GlbReadError::Unsupported(message.into())
}

fn index(value: &Value, name: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| malformed(format!("{name} must be a nonnegative index")))
}

pub(super) fn read_mesh(json: &Value, bin: &[u8]) -> Result<Mesh> {
    if let Some(required) = json.get("extensionsRequired") {
        let required = required
            .as_array()
            .ok_or_else(|| malformed("extensionsRequired must be an array"))?;
        if !required.is_empty() {
            return Err(unsupported(format!(
                "required glTF extensions: {required:?}"
            )));
        }
    }
    for kind in ["buffers", "images"] {
        if let Some(resources) = json.get(kind).and_then(Value::as_array) {
            if resources
                .iter()
                .any(|resource| resource.get("uri").is_some())
            {
                return Err(unsupported(format!(
                    "{kind} with URI resources; provide a self-contained GLB"
                )));
            }
        }
    }
    let mut output = Mesh::default();
    let Some(nodes_value) = json.get("nodes") else {
        let meshes = json
            .get("meshes")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("no meshes"))?;
        for mesh in meshes {
            append_primitives(&mut output, mesh, &IDENTITY, json, bin)?;
        }
        return finish(output);
    };
    let nodes = nodes_value
        .as_array()
        .ok_or_else(|| malformed("nodes must be an array"))?;
    if nodes.len() > MAX_NODES {
        return Err(unsupported("scene exceeds the node budget"));
    }
    let mut parents = vec![None; nodes.len()];
    for (parent, node) in nodes.iter().enumerate() {
        if let Some(children) = node.get("children") {
            for child in children
                .as_array()
                .ok_or_else(|| malformed("children must be an array"))?
            {
                let child = index(child, "child")?;
                let slot = parents
                    .get_mut(child)
                    .ok_or_else(|| malformed("child node is missing"))?;
                if slot.replace(parent).is_some() {
                    return Err(malformed("node has more than one parent"));
                }
            }
        }
    }
    // Validate all parent chains, including unselected scenes, without recursion.
    for node in 0..nodes.len() {
        let mut cursor = Some(node);
        let mut depth = 0;
        while let Some(current) = cursor {
            depth += 1;
            if depth > 128 {
                return Err(unsupported("cyclic or excessively deep node hierarchy"));
            }
            cursor = parents[current];
        }
    }
    let roots: Vec<usize> = if let Some(scenes_value) = json.get("scenes") {
        let scenes = scenes_value
            .as_array()
            .ok_or_else(|| malformed("scenes must be an array"))?;
        let selected = json
            .get("scene")
            .map(|value| index(value, "scene"))
            .transpose()?
            .unwrap_or(0);
        let scene = scenes
            .get(selected)
            .ok_or_else(|| malformed("selected scene is missing"))?;
        scene
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("scene has no root nodes"))?
            .iter()
            .map(|node| index(node, "root node"))
            .collect::<Result<_>>()?
    } else {
        parents
            .iter()
            .enumerate()
            .filter_map(|(i, parent)| parent.is_none().then_some(i))
            .collect()
    };
    let mut visited = vec![false; nodes.len()];
    let mut pending: Vec<_> = roots
        .into_iter()
        .rev()
        .map(|node| (node, IDENTITY))
        .collect();
    while let Some((node_index, parent)) = pending.pop() {
        let node = nodes
            .get(node_index)
            .ok_or_else(|| malformed("root node is missing"))?;
        if std::mem::replace(&mut visited[node_index], true) {
            return Err(malformed("node appears twice in scene"));
        }
        if node.get("skin").is_some() || node.get("weights").is_some() {
            return Err(unsupported(
                "skinning and morph weights require a posed static mesh",
            ));
        }
        let world = multiply(&parent, &local_matrix(node)?);
        if let Some(mesh) = node.get("mesh") {
            let mesh = index(mesh, "mesh")?;
            let mesh = json
                .get("meshes")
                .and_then(Value::as_array)
                .and_then(|meshes| meshes.get(mesh))
                .ok_or_else(|| malformed("node mesh is missing"))?;
            append_primitives(&mut output, mesh, &world, json, bin)?;
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for child in children.iter().rev() {
                pending.push((index(child, "child")?, world));
            }
        }
    }
    finish(output)
}

fn finish(mesh: Mesh) -> Result<Mesh> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err(unsupported("scene has no triangle geometry"));
    }
    mesh.validate()
        .map_err(|error| malformed(error.to_string()))?;
    Ok(mesh)
}

fn components<const N: usize>(node: &Value, name: &str, default: [f64; N]) -> Result<[f64; N]> {
    let Some(value) = node.get(name) else {
        return Ok(default);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == N)
        .ok_or_else(|| malformed(format!("{name} must contain {N} components")))?;
    let mut output = [0.; N];
    for (slot, value) in output.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| malformed(format!("nonfinite or nonnumeric {name}")))?;
    }
    Ok(output)
}

fn local_matrix(node: &Value) -> Result<Matrix> {
    if node.get("matrix").is_some() {
        if ["translation", "rotation", "scale"]
            .iter()
            .any(|key| node.get(key).is_some())
        {
            return Err(malformed("a node cannot contain both matrix and TRS"));
        }
        let values = components(node, "matrix", [0.; 16])?;
        let mut matrix = IDENTITY;
        for row in 0..4 {
            for column in 0..4 {
                matrix[row][column] = values[column * 4 + row];
            }
        }
        if matrix[3] != [0., 0., 0., 1.] {
            return Err(unsupported("node matrix is not affine"));
        }
        // A local glTF matrix must decompose into TRS. Parent composition
        // can legitimately introduce shear, so check only the local matrix.
        let columns = [0, 1, 2].map(|i| [matrix[0][i], matrix[1][i], matrix[2][i]]);
        for i in 0..3 {
            for j in i + 1..3 {
                let scale = dot(columns[i], columns[i]).sqrt() * dot(columns[j], columns[j]).sqrt();
                if dot(columns[i], columns[j]).abs() > 1e-6 * scale {
                    return Err(unsupported("node matrix contains shear"));
                }
            }
        }
        return Ok(matrix);
    }
    let translation = components(node, "translation", [0.; 3])?;
    let scale = components(node, "scale", [1.; 3])?;
    let [x, y, z, w] = components(node, "rotation", [0., 0., 0., 1.])?;
    if (x * x + y * y + z * z + w * w - 1.).abs() > 1e-4 {
        return Err(malformed("rotation must be a unit quaternion"));
    }
    let mut matrix = [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
            translation[0],
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
            translation[1],
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
            translation[2],
        ],
        [0., 0., 0., 1.],
    ];
    for row in matrix.iter_mut().take(3) {
        for (component, scale) in row.iter_mut().zip(scale) {
            *component *= scale;
        }
    }
    Ok(matrix)
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut result = [[0.; 4]; 4];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    result
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn unit(vector: [f64; 3]) -> [f32; 3] {
    let norm = dot(vector, vector).sqrt();
    if norm == 0. {
        return [0., 0., 1.];
    }
    vector.map(|x| (x / norm) as f32)
}

fn transform(mesh: &mut Mesh, matrix: &Matrix) -> Result<()> {
    if *matrix == IDENTITY {
        return Ok(());
    }
    let columns = [0, 1, 2].map(|i| [matrix[0][i], matrix[1][i], matrix[2][i]]);
    let cofactors = [
        cross(columns[1], columns[2]),
        cross(columns[2], columns[0]),
        cross(columns[0], columns[1]),
    ];
    let determinant = dot(columns[0], cofactors[0]);
    if !determinant.is_finite() || determinant == 0. {
        return Err(unsupported("singular node transform"));
    }
    for vertex in &mut mesh.vertices {
        let p = vertex.map(f64::from);
        *vertex = [0, 1, 2].map(|i| {
            (matrix[i][0] * p[0] + matrix[i][1] * p[1] + matrix[i][2] * p[2] + matrix[i][3]) as f32
        });
    }
    if let Some(normals) = &mut mesh.normals {
        for normal in normals {
            let n = normal.map(f64::from);
            *normal = unit([0, 1, 2].map(|i| {
                (cofactors[0][i] * n[0] + cofactors[1][i] * n[1] + cofactors[2][i] * n[2])
                    / determinant
            }));
        }
    }
    if determinant < 0. {
        for face in &mut mesh.faces {
            face.swap(1, 2);
        }
    }
    mesh.validate()
        .map_err(|error| malformed(error.to_string()))
}

fn flat_normals(mesh: &mut Mesh) {
    let source = std::mem::take(mesh);
    mesh.normals = Some(Vec::new());
    mesh.uvs = source.uvs.as_ref().map(|_| Vec::new());
    mesh.vertex_colors = source.vertex_colors.as_ref().map(|_| Vec::new());
    for face in &source.faces {
        let p = face.map(|i| source.vertices[i as usize].map(f64::from));
        let n = unit(cross(
            [0, 1, 2].map(|i| p[1][i] - p[0][i]),
            [0, 1, 2].map(|i| p[2][i] - p[0][i]),
        ));
        let offset = mesh.vertices.len() as u32;
        for &i in face {
            mesh.vertices.push(source.vertices[i as usize]);
            mesh.normals.as_mut().unwrap().push(n);
            if let (Some(dst), Some(src)) = (&mut mesh.uvs, &source.uvs) {
                dst.push(src[i as usize]);
            }
            if let (Some(dst), Some(src)) = (&mut mesh.vertex_colors, &source.vertex_colors) {
                dst.push(src[i as usize]);
            }
        }
        mesh.faces.push([offset, offset + 1, offset + 2]);
    }
}

fn append_primitives(
    output: &mut Mesh,
    definition: &Value,
    world: &Matrix,
    json: &Value,
    bin: &[u8],
) -> Result<()> {
    let primitives = definition
        .get("primitives")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("mesh has no primitives"))?;
    for primitive in primitives {
        if primitive.get("targets").is_some() || primitive.get("extensions").is_some() {
            return Err(unsupported(
                "morph targets or primitive extensions require a static uncompressed mesh",
            ));
        }
        let mut mesh = read_primitive(json, bin, primitive)?;
        let finite = mesh
            .normals
            .iter()
            .flatten()
            .flatten()
            .chain(mesh.vertex_colors.iter().flatten().flatten())
            .chain(mesh.uvs.iter().flatten().flatten())
            .all(|component| component.is_finite());
        if !finite {
            return Err(malformed("nonfinite vertex attribute"));
        }
        if mesh.vertices.len() > MAX_VERTICES {
            return Err(unsupported("scene exceeds the vertex budget"));
        }
        if output.faces.len().saturating_add(mesh.faces.len()) > MAX_FACES {
            return Err(unsupported("scene exceeds the triangle budget"));
        }
        transform(&mut mesh, world)?;
        if output.vertices.is_empty() {
            *output = mesh;
            continue;
        }
        if output.normals.is_some() && mesh.normals.is_none() {
            flat_normals(&mut mesh);
        }
        if output.normals.is_none() && mesh.normals.is_some() {
            flat_normals(output);
        }
        let old = output.vertices.len();
        let added = mesh.vertices.len();
        if old.saturating_add(added) > MAX_VERTICES {
            return Err(unsupported("scene exceeds the vertex budget"));
        }
        let offset = old as u32;
        output.vertices.extend(mesh.vertices);
        output
            .faces
            .extend(mesh.faces.into_iter().map(|face| face.map(|i| i + offset)));
        if let (Some(dst), Some(src)) = (&mut output.normals, mesh.normals) {
            dst.extend(src);
        }
        match (&mut output.uvs, mesh.uvs) {
            (Some(dst), Some(src)) => dst.extend(src),
            _ => output.uvs = None, // Only a COMPLETE UV set may be preserved.
        }
        match (&mut output.vertex_colors, mesh.vertex_colors) {
            (Some(dst), Some(src)) => dst.extend(src),
            (Some(dst), None) => dst.extend(std::iter::repeat_n([1.; 3], added)),
            (None, Some(src)) => {
                let mut dst = vec![[1.; 3]; old];
                dst.extend(src);
                output.vertex_colors = Some(dst);
            }
            (None, None) => {}
        }
    }
    Ok(())
}
