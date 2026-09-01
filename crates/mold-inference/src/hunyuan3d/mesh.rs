//! Surface extraction from the Hunyuan3D shape-VAE occupancy grid.
//!
//! Pure CPU, no candle, no GPU. Runs after the GPU lock is released, so every
//! long loop reports progress through a caller-supplied closure that can also
//! cancel the extraction by returning `Err`.
//!
//! Ported from ComfyUI (the executable oracle for this family):
//!   - `comfy_extras/nodes_hunyuan3d.py:120-224`  `voxel_to_mesh`          ([`MeshAlgorithm::Basic`])
//!   - `comfy_extras/nodes_hunyuan3d.py:225-413`  `voxel_to_mesh_surfnet`  ([`MeshAlgorithm::SurfaceNet`], the default)
//!
//! # Grid axis order
//!
//! [`OccupancyGrid`] holds the tensor exactly as `ShapeVAE.decode` returns it,
//! row-major over `dim = [d0, d1, d2]`, i.e. flat index `(i0 * d1 + i1) * d2 + i2`.
//! ComfyUI names those three axes `D, H, W` and its loop variables `z, y, x`
//! (`nodes_hunyuan3d.py:231-240`), so "row-major `[z][y][x]`" is the ComfyUI
//! spelling of the same layout.
//!
//! What those axes *mean* is worth writing down, because THREE upstream
//! transposes cancel out in a way that is easy to get backwards:
//!
//! 1. `VanillaVolumeDecoder.__call__` (`comfy/ldm/hunyuan3d/vae.py:427-458`)
//!    builds the query points with `meshgrid(x, y, z, indexing="ij")` and
//!    flattens row-major, then reshapes the logits to `(B, N+1, N+1, N+1)`.
//!    At that point the axes are `[qx][qy][qz]` — query-space x, y, z.
//! 2. `ShapeVAE.decode` returns `grid_logits.movedim(-2, -1)`
//!    (`comfy/ldm/hunyuan3d/vae.py:976`), which swaps the last two axes:
//!    `[qx][qz][qy]`.
//! 3. That call is reached through the generic `comfy.sd.VAE.decode` wrapper,
//!    which finishes every decode with the channels-last `movedim(1, -1)`
//!    meant for images (`comfy/sd.py:1277`). On the voxel grid it moves `qx`
//!    to the end: the tensor `VoxelToMesh` actually receives is `[qz][qy][qx]`.
//!
//! So `dim0 = query z`, `dim1 = query y`, `dim2 = query x`. Both mesh functions
//! then emit vertex columns in `(dim0, dim1, dim2)` order and finish with
//! `torch.fliplr` (`nodes_hunyuan3d.py:226` and `:412`), which reverses the
//! columns to `(dim2, dim1, dim0)` = `(query x, query y, query z)`: the final
//! vertex IS the raw query coordinate, every transpose cancelled. glTF is
//! Y-up, so the query grid's `+y` is "up" in every viewer, and the handedness
//! works out for a right-handed one. `ShapeVae::reshape_grid_logits` owns
//! moves 2 and 3, this module owns the flip, and each is reproduced verbatim;
//! "simplifying" any of them away rotates every mesh. The first real render
//! shipped with move 3 missing and came out as the cyclic permutation
//! `(qy, qz, qx)` — a chair lying on its side — which is what the
//! orientation test at the bottom of this file now pins.
//!
//! The grid is cubic in practice (`octree_resolution + 1` on every axis), so a
//! transposition bug is invisible in the shapes and only shows up as a rotated
//! model — hence the long comment instead of a shape assertion.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;

/// A triangle mesh in normalized model space (roughly the unit cube).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub vertex_colors: Option<Vec<[f32; 3]>>,
}

/// Why a [`Mesh`] cannot be exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// A face references a vertex that does not exist.
    FaceIndexOutOfRange {
        face: usize,
        index: u32,
        vertex_count: usize,
    },
    /// A coordinate is NaN or infinite.
    NonFiniteVertex { vertex: usize },
    /// A per-vertex attribute has a different length than `vertices`.
    AttributeLengthMismatch {
        attribute: &'static str,
        len: usize,
        vertex_count: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FaceIndexOutOfRange {
                face,
                index,
                vertex_count,
            } => write!(
                f,
                "face {face} references vertex {index} but the mesh has {vertex_count} vertices"
            ),
            Self::NonFiniteVertex { vertex } => {
                write!(f, "vertex {vertex} has a non-finite coordinate")
            }
            Self::AttributeLengthMismatch {
                attribute,
                len,
                vertex_count,
            } => write!(
                f,
                "{attribute} has {len} entries but the mesh has {vertex_count} vertices"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    /// Axis-aligned bounds as `(min, max)`. An empty mesh reports the origin
    /// for both, which is what the glTF POSITION accessor would need anyway —
    /// though an empty mesh never reaches the writer, which rejects it.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        if self.vertices.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &self.vertices {
            for a in 0..3 {
                if v[a] < min[a] {
                    min[a] = v[a];
                }
                if v[a] > max[a] {
                    max[a] = v[a];
                }
            }
        }
        (min, max)
    }

    /// Rejects the two things that silently produce a corrupt export: a face
    /// index past the end of the vertex array, and a NaN/inf coordinate.
    ///
    /// Mirrors the `raise ValueError` checks in `save_glb`
    /// (`comfy_extras/nodes_save_3d.py:149-176`), but runs on the mesh rather
    /// than inside the writer so callers can fail before doing any work.
    pub fn validate(&self) -> Result<(), MeshError> {
        let n = self.vertices.len();
        for (i, v) in self.vertices.iter().enumerate() {
            if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
                return Err(MeshError::NonFiniteVertex { vertex: i });
            }
        }
        for (fi, face) in self.faces.iter().enumerate() {
            for &idx in face {
                if idx as usize >= n {
                    return Err(MeshError::FaceIndexOutOfRange {
                        face: fi,
                        index: idx,
                        vertex_count: n,
                    });
                }
            }
        }
        let check = |attribute: &'static str, len: Option<usize>| -> Result<(), MeshError> {
            match len {
                Some(len) if len != n => Err(MeshError::AttributeLengthMismatch {
                    attribute,
                    len,
                    vertex_count: n,
                }),
                _ => Ok(()),
            }
        };
        check("normals", self.normals.as_ref().map(Vec::len))?;
        check("uvs", self.uvs.as_ref().map(Vec::len))?;
        check("vertex_colors", self.vertex_colors.as_ref().map(Vec::len))?;
        Ok(())
    }
}

/// The dense scalar field the shape VAE evaluates on the query grid.
///
/// `logits` is row-major over `dim`; see the module docs for what the three
/// axes mean. Values are raw logits, not probabilities — `threshold` is
/// compared against them directly, exactly as upstream does.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyGrid {
    pub logits: Vec<f32>,
    pub dim: [usize; 3],
}

impl OccupancyGrid {
    pub fn new(logits: Vec<f32>, dim: [usize; 3]) -> anyhow::Result<Self> {
        let expected = dim[0]
            .checked_mul(dim[1])
            .and_then(|n| n.checked_mul(dim[2]))
            .ok_or_else(|| anyhow::anyhow!("occupancy grid dimensions {dim:?} overflow usize"))?;
        anyhow::ensure!(
            logits.len() == expected,
            "occupancy grid has {} values but dimensions {dim:?} need {expected}",
            logits.len()
        );
        Ok(Self { logits, dim })
    }

    pub fn len(&self) -> usize {
        self.logits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.logits.is_empty()
    }

    /// Row-major sample at `[i0][i1][i2]`.
    #[inline]
    pub fn at(&self, i0: usize, i1: usize, i2: usize) -> f32 {
        self.logits[(i0 * self.dim[1] + i1) * self.dim[2] + i2]
    }

    /// Sample in the once-zero-padded index space upstream builds with
    /// `F.pad(voxels, (1, 1, 1, 1, 1, 1), 'constant', 0)`
    /// (`nodes_hunyuan3d.py:126` and `:230`).
    ///
    /// Padded index `p` maps to unpadded `p - 1`; `p == 0` is the pad and reads
    /// zero. Callers only ever pass `p <= dim`, so the upper pad is never hit.
    #[inline]
    fn padded(&self, p0: usize, p1: usize, p2: usize) -> f32 {
        if p0 == 0 || p1 == 0 || p2 == 0 {
            return 0.0;
        }
        let (i0, i1, i2) = (p0 - 1, p1 - 1, p2 - 1);
        if i0 >= self.dim[0] || i1 >= self.dim[1] || i2 >= self.dim[2] {
            return 0.0;
        }
        self.at(i0, i1, i2)
    }
}

/// Which extraction algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshAlgorithm {
    /// Surface nets / naive dual contouring. Smooth, far fewer triangles.
    /// ComfyUI's default (`VoxelToMesh`, `nodes_hunyuan3d.py:451-486`).
    #[default]
    SurfaceNet,
    /// Blocky per-voxel quads. ComfyUI's deprecated `VoxelToMeshBasic`.
    Basic,
}

/// Reports `(current, total)` and lets the caller cancel by returning `Err`.
type ProgressFn<'a> = &'a mut dyn FnMut(u32, u32) -> anyhow::Result<()>;

struct Ticker<'a> {
    cb: ProgressFn<'a>,
    current: u32,
    total: u32,
}

impl Ticker<'_> {
    fn tick(&mut self) -> anyhow::Result<()> {
        if self.current < self.total {
            self.current += 1;
        }
        (self.cb)(self.current, self.total)
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        self.current = self.total;
        (self.cb)(self.current, self.total)
    }
}

/// Corner offsets of a surface-net cell, in `(dim0, dim1, dim2)` column order.
///
/// `nodes_hunyuan3d.py:242-245`.
const SURFNET_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// The 12 cube edges as corner-index pairs. `nodes_hunyuan3d.py:262-265`.
const SURFNET_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [0, 2],
    [0, 4],
    [1, 3],
    [1, 5],
    [2, 3],
    [2, 6],
    [3, 7],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
];

/// Extract a triangle mesh from `grid`.
///
/// `progress` is invoked tens of times over the whole run (never per cell); a
/// non-`Ok` return aborts the extraction immediately.
pub fn extract(
    grid: &OccupancyGrid,
    algorithm: MeshAlgorithm,
    threshold: f32,
    progress: ProgressFn<'_>,
) -> anyhow::Result<Mesh> {
    match algorithm {
        MeshAlgorithm::SurfaceNet => extract_surface_net(grid, threshold, progress),
        MeshAlgorithm::Basic => extract_basic(grid, threshold, progress),
    }
}

/// Number of progress ticks each algorithm emits. Chosen so the caller sees
/// tens of updates rather than millions.
const SCAN_TICKS: u32 = 16;
const FACE_TICKS_PER_PAIR: u32 = 8;
const SURFNET_TICKS: u32 = SCAN_TICKS + SURFNET_EDGES.len() as u32 + 3 * FACE_TICKS_PER_PAIR;
const BASIC_TICKS: u32 = SCAN_TICKS + 6;

/// One cell of the padded grid that straddles the iso-surface.
struct ActiveCell {
    /// Position in padded-cell space, `(dim0, dim1, dim2)`.
    pos: [u32; 3],
    /// The 8 corner logits, in [`SURFNET_CORNERS`] order.
    values: [f32; 8],
    /// Bit `i` set means corner `i` is inside (`value > threshold`).
    signs: u8,
}

fn extract_surface_net(
    grid: &OccupancyGrid,
    threshold: f32,
    progress: ProgressFn<'_>,
) -> anyhow::Result<Mesh> {
    let mut ticker = Ticker {
        cb: progress,
        current: 0,
        total: SURFNET_TICKS,
    };
    let [d0, d1, d2] = grid.dim;
    if d0 == 0 || d1 == 0 || d2 == 0 {
        ticker.finish()?;
        return Ok(Mesh::default());
    }

    // Pass 1: collect the cells whose 8 corners are not all on one side.
    // Upstream materialises every cell's corners as a (D*H*W, 8) tensor
    // (`nodes_hunyuan3d.py:247-256`); at octree resolution 256 that is 138M
    // floats, so we stream it instead and keep only the sparse surface.
    let mut active: Vec<ActiveCell> = Vec::new();
    let slice_step = ((d0 as u32).div_ceil(SCAN_TICKS)).max(1);
    let mut next_tick_slice = slice_step;
    for p0 in 0..d0 {
        for p1 in 0..d1 {
            for p2 in 0..d2 {
                let mut values = [0.0f32; 8];
                let mut signs = 0u8;
                for (c, off) in SURFNET_CORNERS.iter().enumerate() {
                    let v = grid.padded(p0 + off[0], p1 + off[1], p2 + off[2]);
                    values[c] = v;
                    if v > threshold {
                        signs |= 1 << c;
                    }
                }
                // has_inside & has_outside (`nodes_hunyuan3d.py:258-260`).
                if signs != 0 && signs != 0xFF {
                    active.push(ActiveCell {
                        pos: [p0 as u32, p1 as u32, p2 as u32],
                        values,
                        signs,
                    });
                }
            }
        }
        if p0 as u32 + 1 >= next_tick_slice {
            next_tick_slice += slice_step;
            ticker.tick()?;
        }
    }
    while ticker.current < SCAN_TICKS {
        ticker.tick()?;
    }

    if active.is_empty() {
        ticker.finish()?;
        return Ok(Mesh::default());
    }

    // Pass 2: one dual vertex per active cell, at the mean of its edge
    // crossings (`nodes_hunyuan3d.py:267-311`).
    //
    // Upstream keys a dict by active-cell index and later iterates it in
    // insertion order, so a cell's vertex index is decided by the FIRST edge
    // that crosses it, not by the cell's position. Reproducing that ordering
    // costs nothing and keeps our vertex buffer byte-comparable with the
    // oracle's.
    let n_active = active.len();
    let mut sums = vec![[0.0f32; 3]; n_active];
    let mut counts = vec![0u32; n_active];
    let mut order: Vec<u32> = Vec::with_capacity(n_active);
    for [e1, e2] in SURFNET_EDGES {
        let p1 = SURFNET_CORNERS[e1];
        let p2 = SURFNET_CORNERS[e2];
        for (ci, cell) in active.iter().enumerate() {
            let s1 = cell.signs >> e1 & 1;
            let s2 = cell.signs >> e2 & 1;
            if s1 == s2 {
                continue;
            }
            let v1 = cell.values[e1];
            let v2 = cell.values[e2];
            let denom = v2 - v1;
            // `t[~valid] = 0.5` when the two corner values are identical
            // (`nodes_hunyuan3d.py:280-284`).
            let t = if denom != 0.0 {
                (threshold - v1) / denom
            } else {
                0.5
            };
            if counts[ci] == 0 {
                order.push(ci as u32);
            }
            counts[ci] += 1;
            for a in 0..3 {
                let a1 = p1[a] as f32;
                let a2 = p2[a] as f32;
                sums[ci][a] += a1 + t * (a2 - a1);
            }
        }
        ticker.tick()?;
    }

    // `vertex_lookup` upstream is a dict keyed by cell position. A flat table
    // over the cell grid gives O(1) neighbour probes without hashing; it costs
    // 4 bytes per cell (~68 MB at octree resolution 256, alongside the ~68 MB
    // grid itself) and is dropped as soon as the faces are built.
    let cell_count = d0 * d1 * d2;
    let mut lookup = vec![u32::MAX; cell_count];
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(order.len());
    for &ci in &order {
        let ci = ci as usize;
        let cell = &active[ci];
        let inv = 1.0 / counts[ci] as f32;
        let v = [
            sums[ci][0] * inv + cell.pos[0] as f32,
            sums[ci][1] * inv + cell.pos[1] as f32,
            sums[ci][2] * inv + cell.pos[2] as f32,
        ];
        let flat = (cell.pos[0] as usize * d1 + cell.pos[1] as usize) * d2 + cell.pos[2] as usize;
        lookup[flat] = vertices.len() as u32;
        vertices.push(v);
    }

    // Per-cell surface gradient: mean inside-corner position minus mean
    // outside-corner position (`nodes_hunyuan3d.py:320-341`). Used only to
    // orient each quad.
    let mut gradients = vec![[0.0f32; 3]; n_active];
    for (ci, cell) in active.iter().enumerate() {
        let mut inside = [0.0f32; 3];
        let mut outside = [0.0f32; 3];
        let mut n_in = 0.0f32;
        let mut n_out = 0.0f32;
        for (c, off) in SURFNET_CORNERS.iter().enumerate() {
            let target = if cell.signs >> c & 1 == 1 {
                n_in += 1.0;
                &mut inside
            } else {
                n_out += 1.0;
                &mut outside
            };
            for a in 0..3 {
                target[a] += off[a] as f32;
            }
        }
        // Both counts are non-zero: an active cell has corners on both sides.
        for a in 0..3 {
            gradients[ci][a] = inside[a] / n_in - outside[a] / n_out;
        }
    }

    // Quads between four cells sharing an edge, one axis pair at a time
    // (`nodes_hunyuan3d.py:343-397`). `pos_dirs` is the identity basis, so
    // `cross_products` is cross(e_i, e_j) for the three pairs below.
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for (pair_idx, (i, j)) in PAIRS.into_iter().enumerate() {
        let mut dir_i = [0usize; 3];
        dir_i[i] = 1;
        let mut dir_j = [0usize; 3];
        dir_j[j] = 1;
        let cross = cross_basis(i, j);

        let chunk = (n_active as u32).div_ceil(FACE_TICKS_PER_PAIR).max(1);
        let mut next_tick = chunk;
        let ticks_before = ticker.current;
        for (ci, cell) in active.iter().enumerate() {
            let probe = |delta: [usize; 3]| -> Option<u32> {
                let mut p = [0usize; 3];
                for a in 0..3 {
                    p[a] = cell.pos[a] as usize + delta[a];
                    if p[a] >= grid.dim[a] {
                        return None;
                    }
                }
                let v = lookup[(p[0] * d1 + p[1]) * d2 + p[2]];
                (v != u32::MAX).then_some(v)
            };
            let diag = [
                dir_i[0] + dir_j[0],
                dir_i[1] + dir_j[1],
                dir_i[2] + dir_j[2],
            ];
            if let (Some(v0), Some(v1), Some(v2), Some(v3)) =
                (probe([0, 0, 0]), probe(dir_i), probe(dir_j), probe(diag))
            {
                let alignment = gradients[ci][0] * cross[0]
                    + gradients[ci][1] * cross[1]
                    + gradients[ci][2] * cross[2];
                // Strictly greater, so a zero alignment takes the else branch —
                // matching `if alignments[cell_idx] > 0` upstream.
                if alignment > 0.0 {
                    faces.push([v0, v1, v3]);
                    faces.push([v0, v3, v2]);
                } else {
                    faces.push([v0, v3, v1]);
                    faces.push([v0, v2, v3]);
                }
            }
            if ci as u32 + 1 >= next_tick {
                next_tick += chunk;
                ticker.tick()?;
            }
        }
        let _ = pair_idx;
        while ticker.current < ticks_before + FACE_TICKS_PER_PAIR {
            ticker.tick()?;
        }
    }

    normalize_and_flip(&mut vertices, grid.dim);
    ticker.finish()?;
    Ok(Mesh {
        vertices,
        faces,
        ..Mesh::default()
    })
}

/// `cross(e_i, e_j)` for the standard basis vectors.
fn cross_basis(i: usize, j: usize) -> [f32; 3] {
    let mut a = [0.0f32; 3];
    let mut b = [0.0f32; 3];
    a[i] = 1.0;
    b[j] = 1.0;
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Neighbour directions and the quad corners of the face they expose.
/// `nodes_hunyuan3d.py:130-171`, in the same order.
const BASIC_NEIGHBORS: [[i64; 3]; 6] = [
    [0, 0, 1],
    [0, 0, -1],
    [0, 1, 0],
    [0, -1, 0],
    [1, 0, 0],
    [-1, 0, 0],
];

const BASIC_FACE_CORNERS: [[[u32; 3]; 4]; 6] = [
    [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
    [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
    [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
];

fn extract_basic(
    grid: &OccupancyGrid,
    threshold: f32,
    progress: ProgressFn<'_>,
) -> anyhow::Result<Mesh> {
    let mut ticker = Ticker {
        cb: progress,
        current: 0,
        total: BASIC_TICKS,
    };
    let [d0, d1, d2] = grid.dim;
    if d0 == 0 || d1 == 0 || d2 == 0 {
        ticker.finish()?;
        return Ok(Mesh::default());
    }

    // `binary = (voxels > threshold)` then the solid cells in row-major order
    // (`nodes_hunyuan3d.py:125,144-145`).
    let mut solid: Vec<[u32; 3]> = Vec::new();
    let slice_step = ((d0 as u32).div_ceil(SCAN_TICKS)).max(1);
    let mut next_tick_slice = slice_step;
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                if grid.at(i0, i1, i2) > threshold {
                    solid.push([i0 as u32, i1 as u32, i2 as u32]);
                }
            }
        }
        if i0 as u32 + 1 >= next_tick_slice {
            next_tick_slice += slice_step;
            ticker.tick()?;
        }
    }
    while ticker.current < SCAN_TICKS {
        ticker.tick()?;
    }

    let solid_at = |c: [i64; 3]| -> bool {
        if c[0] < 0 || c[1] < 0 || c[2] < 0 {
            return false;
        }
        let (a, b, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        if a >= d0 || b >= d1 || d >= d2 {
            return false;
        }
        grid.at(a, b, d) > threshold
    };

    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for (face_idx, offset) in BASIC_NEIGHBORS.into_iter().enumerate() {
        let corners = BASIC_FACE_CORNERS[face_idx];
        for cell in &solid {
            let neighbor = [
                cell[0] as i64 + offset[0],
                cell[1] as i64 + offset[1],
                cell[2] as i64 + offset[2],
            ];
            // `is_exposed = padded[...] == 0`: a solid cell contributes a quad
            // only where its neighbour is empty or off-grid.
            if solid_at(neighbor) {
                continue;
            }
            let base = vertices.len() as u32;
            for c in corners {
                vertices.push([
                    (cell[0] + c[0]) as f32,
                    (cell[1] + c[1]) as f32,
                    (cell[2] + c[2]) as f32,
                ]);
            }
            faces.push([base, base + 1, base + 2]);
            faces.push([base, base + 2, base + 3]);
        }
        ticker.tick()?;
    }

    // DELIBERATE DEVIATION. When nothing is solid, upstream returns
    // `torch.zeros((1, 3))` for both vertices and faces
    // (`nodes_hunyuan3d.py:210-212`) — one vertex at the origin and one
    // degenerate face `(0, 0, 0)`. That fallback exists purely so the
    // downstream `torch.cat`/`torch.stack` sees a well-shaped tensor; it is
    // not geometry. Writing it out produces a GLB with a zero-area triangle,
    // so we return a genuinely empty mesh and let the caller/writer reject it.
    if vertices.is_empty() {
        ticker.finish()?;
        return Ok(Mesh::default());
    }

    normalize_and_flip(&mut vertices, grid.dim);
    ticker.finish()?;
    Ok(Mesh {
        vertices,
        faces,
        ..Mesh::default()
    })
}

/// Centre on the grid, scale to roughly the unit cube, then reverse the vertex
/// columns.
///
/// Both upstream functions end with the same three steps
/// (`nodes_hunyuan3d.py:214-223` and `:399-411`):
///
/// ```text
/// v_min = 0; v_max = max(shape)
/// vertices = (vertices - (v_min + v_max) / 2) / ((v_max - v_min) / 2)
/// vertices = torch.fliplr(vertices)
/// ```
///
/// Two quirks worth naming, both preserved:
///
/// * `v_min` is hard-coded to 0 rather than the actual minimum vertex
///   coordinate, and `v_max` to the largest grid dimension rather than the
///   actual maximum. So the mesh is centred on the *grid*, not on itself, and
///   a model that does not fill its bounding volume sits off-centre. Using the
///   real extent would recentre and rescale every mesh away from the oracle.
/// * `fliplr` reverses the columns, converting `(dim0, dim1, dim2)` to
///   `(dim2, dim1, dim0)`. See the module docs for why this is a coordinate
///   convention and not a bug.
fn normalize_and_flip(vertices: &mut [[f32; 3]], dim: [usize; 3]) {
    let v_max = dim.iter().copied().max().unwrap_or(0) as f32;
    let center = v_max / 2.0;
    let scale = v_max / 2.0;
    for v in vertices.iter_mut() {
        let mut a = [v[0] - center, v[1] - center, v[2] - center];
        if scale > 0.0 {
            a = [a[0] / scale, a[1] / scale, a[2] / scale];
        }
        *v = [a[2], a[1], a[0]];
    }
}

/// Fill in area-weighted per-vertex normals.
///
/// Surface-net output carries none, and a glTF viewer with no NORMAL attribute
/// falls back to flat per-face shading, which makes a smooth extraction look
/// like the blocky one. The face cross product's magnitude is twice the
/// triangle area, so summing the raw cross products weights each face by its
/// area for free.
///
/// Degenerate vertices (no incident face, or perfectly cancelling faces) get
/// `+Y`, so every normal is unit length as glTF requires.
pub fn compute_smooth_normals(mesh: &mut Mesh) {
    let mut normals = vec![[0.0f32; 3]; mesh.vertices.len()];
    for face in &mesh.faces {
        let (a, b, c) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if a >= mesh.vertices.len() || b >= mesh.vertices.len() || c >= mesh.vertices.len() {
            continue;
        }
        let p0 = mesh.vertices[a];
        let p1 = mesh.vertices[b];
        let p2 = mesh.vertices[c];
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for idx in [a, b, c] {
            for k in 0..3 {
                normals[idx][k] += n[k];
            }
        }
    }
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-20 && len.is_finite() {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }
    mesh.normals = Some(normals);
}

/// Decimate `mesh` to at most `target_faces` triangles by quadric-error edge
/// collapse (Garland & Heckbert, "Surface Simplification Using Quadric Error
/// Metrics", SIGGRAPH '97).
///
/// Each vertex carries the sum of the fundamental error quadrics of its
/// incident faces, area-weighted so large flat regions dominate small ones.
/// Collapsing an edge sums the two quadrics; the new vertex goes wherever that
/// sum is minimised, which is what lets a flat region collapse to almost
/// nothing while a crease holds its shape.
///
/// Guarantees, in order of how much they cost you when violated:
///
/// * The result always passes [`Mesh::validate`]. Every collapse is checked
///   BEFORE it is applied, and a collapse that would invert a face normal,
///   create a zero-area triangle, or produce two faces with the same three
///   vertices is rejected rather than emitted.
/// * `target_faces >= mesh.face_count()` is a no-op that clones the mesh whole,
///   attributes included — asking for no decimation is not an error.
/// * It terminates. When no legal collapse remains (a tetrahedron is the small
///   example: every one of its six edges would fold the solid into a doubled
///   triangle) the best mesh reached so far is returned, even if that is still
///   above `target_faces`. An explicit iteration cap backstops that.
///
/// `normals`, `uvs` and `vertex_colors` are DROPPED. A collapse places the new
/// vertex at the quadric minimum, which is generally not either endpoint and
/// often not even on the original surface, so there is no correct barycentric
/// weight to interpolate an attribute with — carrying them across would
/// smear a texture and produce normals that disagree with the new geometry.
/// The caller recomputes normals afterwards
/// (`engine.rs:453-457` does exactly this), and UV/color assignment for this
/// family happens after decimation, not before.
pub fn simplify(mesh: &Mesh, target_faces: usize) -> anyhow::Result<Mesh> {
    // Never hand back something worse than we were given, and never start from
    // indices that would panic the adjacency build.
    mesh.validate()?;
    if mesh.faces.is_empty() || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }

    let mut state = Simplifier::new(mesh);
    state.run(target_faces);
    let out = state.into_mesh();
    // Cheap next to the decimation itself, and the one thing this function
    // must never get wrong.
    out.validate()?;
    Ok(out)
}

/// Symmetric 4x4 error quadric, packed as the upper triangle:
/// `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]`.
///
/// f64 throughout: the products of plane coefficients that build these lose
/// several digits, and an f32 quadric puts visible noise into the collapse
/// ordering on meshes this size.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    /// The fundamental quadric `K_p = p p^T` of the plane `n·x + d = 0`,
    /// scaled by `weight` (we pass twice the triangle area).
    fn from_plane(n: [f64; 3], d: f64, weight: f64) -> Self {
        let [a, b, c] = n;
        Self([
            a * a * weight,
            a * b * weight,
            a * c * weight,
            a * d * weight,
            b * b * weight,
            b * c * weight,
            b * d * weight,
            c * c * weight,
            c * d * weight,
            d * d * weight,
        ])
    }

    fn add_assign(&mut self, other: &Quadric) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b;
        }
    }

    fn sum(a: &Quadric, b: &Quadric) -> Quadric {
        let mut out = *a;
        out.add_assign(b);
        out
    }

    /// `v^T Q v` for the homogeneous point `(x, y, z, 1)`.
    fn error(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let [x, y, z] = p;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    /// The point minimising [`Quadric::error`], i.e. the solution of
    /// `A v = -b` for the upper-left 3x3 block `A` and the translation row `b`.
    /// `None` when `A` is singular, which happens on planar and symmetric
    /// neighbourhoods — very common, so the caller must have a fallback.
    fn optimal_position(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        let a = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        // Scale-aware: a quadric built from a tiny triangle has a tiny
        // determinant without being singular, so compare against the matrix
        // magnitude rather than an absolute epsilon.
        let scale = a
            .iter()
            .flatten()
            .fold(0.0f64, |acc, v| acc.max(v.abs()))
            .max(1e-30);
        if det.abs() < 1e-10 * scale * scale * scale {
            return None;
        }
        let b = [-q[3], -q[6], -q[8]];
        // Cramer's rule: three 3x3 determinants with b substituted per column.
        let col_det = |c: usize| -> f64 {
            let mut m = a;
            for (r, row) in m.iter_mut().enumerate() {
                row[c] = b[r];
            }
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        Some([col_det(0) / det, col_det(1) / det, col_det(2) / det])
    }
}

/// One pending collapse. `vu`/`vv` stamp the endpoint versions at the time the
/// cost was computed; a stamp that no longer matches means the neighbourhood
/// has moved since and the entry is stale.
///
/// Lazy invalidation like this is why there is no "decrease-key" here: a
/// collapse touches an unbounded number of incident edges, and re-pushing them
/// is cheaper than finding and rewriting their heap slots.
struct Candidate {
    cost: f64,
    u: u32,
    v: u32,
    vu: u64,
    vv: u64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed so `BinaryHeap`, a MAX-heap, pops the cheapest collapse.
        // `total_cmp` because quadric costs are f64 and a NaN must not make
        // the ordering inconsistent and corrupt the heap.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.u.cmp(&self.u))
            .then_with(|| other.v.cmp(&self.v))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Candidate {}

/// Reject a collapse whose new normal disagrees with the old one. Zero rather
/// than a positive cosine: the goal is to forbid actual inversions, not to
/// forbid the surface from curving, and a stricter bound stalls decimation on
/// exactly the rounded surfaces this family produces.
const MIN_NORMAL_DOT: f64 = 0.0;

/// Below this the triangle has no usable normal and we treat it as degenerate.
/// Vertices live in roughly `[-1, 1]`, so even a 257-cell grid's triangles have
/// cross-product magnitudes near 1e-5 — twelve orders of magnitude clear.
const DEGENERATE_AREA: f64 = 1e-12;

/// Weight on the virtual planes that hold open boundaries in place. Large
/// enough that a boundary loop is effectively pinned against ordinary interior
/// error, which stops an open shell from eating its own rim.
const BOUNDARY_WEIGHT: f64 = 1000.0;

struct Simplifier {
    positions: Vec<[f64; 3]>,
    quadrics: Vec<Quadric>,
    /// Bumped whenever a vertex moves, merges, or dies. See [`Candidate`].
    version: Vec<u64>,
    alive_vertex: Vec<bool>,
    faces: Vec<[u32; 3]>,
    alive_face: Vec<bool>,
    /// Incident faces per vertex. May hold dead entries; readers filter.
    vertex_faces: Vec<Vec<u32>>,
    live_faces: usize,
}

impl Simplifier {
    fn new(mesh: &Mesh) -> Self {
        let positions: Vec<[f64; 3]> = mesh
            .vertices
            .iter()
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let n = positions.len();
        let mut vertex_faces = vec![Vec::new(); n];
        for (fi, face) in mesh.faces.iter().enumerate() {
            for &v in face {
                vertex_faces[v as usize].push(fi as u32);
            }
        }
        let mut state = Self {
            positions,
            quadrics: vec![Quadric::default(); n],
            version: vec![0; n],
            alive_vertex: vec![true; n],
            faces: mesh.faces.clone(),
            alive_face: vec![true; mesh.faces.len()],
            vertex_faces,
            live_faces: mesh.faces.len(),
        };
        // A vertex with no incident face can never be collapsed and would
        // otherwise sit in the output as an orphan.
        for v in 0..n {
            if state.vertex_faces[v].is_empty() {
                state.alive_vertex[v] = false;
            }
        }
        state.build_quadrics();
        state
    }

    fn face_plane(&self, face: [u32; 3]) -> Option<([f64; 3], f64, f64)> {
        let a = self.positions[face[0] as usize];
        let b = self.positions[face[1] as usize];
        let c = self.positions[face[2] as usize];
        let n = cross(sub(b, a), sub(c, a));
        let len = norm(n);
        if len < DEGENERATE_AREA {
            return None;
        }
        let unit = [n[0] / len, n[1] / len, n[2] / len];
        let d = -dot(unit, a);
        // `len` is twice the triangle area — the standard area weighting.
        Some((unit, d, len))
    }

    fn build_quadrics(&mut self) {
        for fi in 0..self.faces.len() {
            let face = self.faces[fi];
            let Some((n, d, area2)) = self.face_plane(face) else {
                continue;
            };
            let q = Quadric::from_plane(n, d, area2);
            for &v in &face {
                self.quadrics[v as usize].add_assign(&q);
            }
        }
        self.add_boundary_quadrics();
    }

    /// Open edges get a virtual plane perpendicular to their one incident face,
    /// passing through the edge. Without it the rim of an open shell is the
    /// cheapest thing in the mesh to collapse and decimation eats inward.
    fn add_boundary_quadrics(&mut self) {
        let mut edge_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            for k in 0..3 {
                let key = edge_key(face[k], face[(k + 1) % 3]);
                edge_faces.entry(key).or_default().push(fi as u32);
            }
        }
        for ((a, b), faces) in &edge_faces {
            if faces.len() != 1 {
                continue;
            }
            let Some((n, _, _)) = self.face_plane(self.faces[faces[0] as usize]) else {
                continue;
            };
            let pa = self.positions[*a as usize];
            let pb = self.positions[*b as usize];
            let along = sub(pb, pa);
            let perp = cross(along, n);
            let len = norm(perp);
            if len < DEGENERATE_AREA {
                continue;
            }
            let unit = [perp[0] / len, perp[1] / len, perp[2] / len];
            let q = Quadric::from_plane(unit, -dot(unit, pa), BOUNDARY_WEIGHT * norm(along));
            self.quadrics[*a as usize].add_assign(&q);
            self.quadrics[*b as usize].add_assign(&q);
        }
    }

    fn live_incident_faces(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        self.vertex_faces[v as usize]
            .iter()
            .copied()
            .filter(move |&fi| self.alive_face[fi as usize])
    }

    fn neighbors(&self, v: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for fi in self.live_incident_faces(v) {
            for &w in &self.faces[fi as usize] {
                if w != v && !out.contains(&w) {
                    out.push(w);
                }
            }
        }
        out
    }

    /// Where the merged vertex would go, and what that costs.
    fn placement(&self, u: u32, v: u32) -> ([f64; 3], f64) {
        let q = Quadric::sum(&self.quadrics[u as usize], &self.quadrics[v as usize]);
        let pu = self.positions[u as usize];
        let pv = self.positions[v as usize];
        let mid = [
            (pu[0] + pv[0]) * 0.5,
            (pu[1] + pv[1]) * 0.5,
            (pu[2] + pv[2]) * 0.5,
        ];
        if let Some(p) = q.optimal_position() {
            // A near-singular solve can put the minimum arbitrarily far away
            // and spike the surface. Anything past a few edge lengths is not a
            // simplification of this edge, whatever its algebraic error says.
            let edge_len = norm(sub(pv, pu));
            if norm(sub(p, mid)) <= 4.0 * edge_len.max(1e-12) {
                return (p, q.error(p).max(0.0));
            }
        }
        // Best of the three obvious candidates; ties keep the midpoint.
        let mut best = (mid, q.error(mid));
        for cand in [pu, pv] {
            let e = q.error(cand);
            if e < best.1 {
                best = (cand, e);
            }
        }
        (best.0, best.1.max(0.0))
    }

    /// Every check that must pass before a collapse is applied.
    ///
    /// Returns the faces that die (they contain both endpoints) and the faces
    /// that survive with `v` rewritten to `u`.
    fn check_collapse(&self, u: u32, v: u32, p: [f64; 3]) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut dying = Vec::new();
        let mut surviving = Vec::new();
        for fi in self
            .live_incident_faces(u)
            .chain(self.live_incident_faces(v))
        {
            let face = self.faces[fi as usize];
            let has_u = face.contains(&u);
            let has_v = face.contains(&v);
            if has_u && has_v {
                if !dying.contains(&fi) {
                    dying.push(fi);
                }
            } else if !surviving.contains(&fi) {
                surviving.push(fi);
            }
        }
        if dying.is_empty() {
            // Not actually an edge of the mesh any more.
            return None;
        }

        // Topology: `u` and `v` may share only the vertices opposite the faces
        // that die. A shared neighbour that is NOT opposite a dying face means
        // the collapse would pinch the surface into a non-manifold seam. This
        // is the check that rejects a tetrahedron.
        let mut opposite = Vec::new();
        for &fi in &dying {
            for &w in &self.faces[fi as usize] {
                if w != u && w != v {
                    opposite.push(w);
                }
            }
        }
        let nu = self.neighbors(u);
        let nv = self.neighbors(v);
        for w in &nu {
            if *w != v && nv.contains(w) && !opposite.contains(w) {
                return None;
            }
        }

        // Geometry: no inverted or degenerate triangles.
        for &fi in &surviving {
            let face = self.faces[fi as usize];
            let old = self.face_plane(face);
            let mut pts = [[0.0f64; 3]; 3];
            for (k, &w) in face.iter().enumerate() {
                pts[k] = if w == u || w == v {
                    p
                } else {
                    self.positions[w as usize]
                };
            }
            let n_new = cross(sub(pts[1], pts[0]), sub(pts[2], pts[0]));
            let len_new = norm(n_new);
            if len_new < DEGENERATE_AREA {
                return None;
            }
            if let Some((n_old, _, _)) = old {
                let d = dot(
                    n_old,
                    [n_new[0] / len_new, n_new[1] / len_new, n_new[2] / len_new],
                );
                if d <= MIN_NORMAL_DOT {
                    return None;
                }
            }
        }

        // No two surviving faces may end up with the same three vertices. Any
        // such pair must contain the merged vertex, so the incident set above
        // is the whole search space.
        let mut seen: Vec<[u32; 3]> = Vec::with_capacity(surviving.len());
        for &fi in &surviving {
            let mut key = self.faces[fi as usize];
            for w in key.iter_mut() {
                if *w == v {
                    *w = u;
                }
            }
            key.sort_unstable();
            if key[0] == key[1] || key[1] == key[2] {
                return None;
            }
            if seen.contains(&key) {
                return None;
            }
            seen.push(key);
        }

        Some((dying, surviving))
    }

    fn apply_collapse(&mut self, u: u32, v: u32, p: [f64; 3], dying: &[u32], surviving: &[u32]) {
        for &fi in dying {
            self.alive_face[fi as usize] = false;
            self.live_faces -= 1;
        }
        for &fi in surviving {
            let face = &mut self.faces[fi as usize];
            let mut touched = false;
            for w in face.iter_mut() {
                if *w == v {
                    *w = u;
                    touched = true;
                }
            }
            if touched && !self.vertex_faces[u as usize].contains(&fi) {
                self.vertex_faces[u as usize].push(fi);
            }
        }
        self.positions[u as usize] = p;
        let qv = self.quadrics[v as usize];
        self.quadrics[u as usize].add_assign(&qv);
        self.alive_vertex[v as usize] = false;
        self.vertex_faces[v as usize].clear();
        // Drop dead faces so adjacency scans stay proportional to live degree.
        let alive = &self.alive_face;
        self.vertex_faces[u as usize].retain(|&fi| alive[fi as usize]);
        self.version[u as usize] += 1;
        self.version[v as usize] += 1;
    }

    fn push_edges_around(&self, heap: &mut BinaryHeap<Candidate>, v: u32) {
        for w in self.neighbors(v) {
            let (u, x) = edge_key(v, w);
            let (_, cost) = self.placement(u, x);
            heap.push(Candidate {
                cost,
                u,
                v: x,
                vu: self.version[u as usize],
                vv: self.version[x as usize],
            });
        }
    }

    fn run(&mut self, target_faces: usize) {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        // A set, not a `Vec` + linear scan: a 300k-face mesh has ~450k unique
        // edges, and deduplicating those quadratically is the difference
        // between milliseconds and hours.
        let mut seeded: HashSet<(u32, u32)> = HashSet::with_capacity(self.faces.len() * 2);
        for face in &self.faces {
            for k in 0..3 {
                seeded.insert(edge_key(face[k], face[(k + 1) % 3]));
            }
        }
        for (u, v) in seeded {
            let (_, cost) = self.placement(u, v);
            heap.push(Candidate {
                cost,
                u,
                v,
                vu: self.version[u as usize],
                vv: self.version[v as usize],
            });
        }

        // Hard bound on work. Each legal collapse removes at least one face, so
        // the useful iterations are bounded by the face count; the slack
        // absorbs stale pops and rejected candidates. Hitting the cap returns
        // the best mesh reached rather than spinning.
        let max_iterations = 64 * self.faces.len() + 4096;
        let mut iterations = 0usize;

        while self.live_faces > target_faces {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }
            let Some(candidate) = heap.pop() else {
                // No legal collapse remains; a tetrahedron ends up here.
                break;
            };
            let (u, v) = (candidate.u, candidate.v);
            if !self.alive_vertex[u as usize] || !self.alive_vertex[v as usize] {
                continue;
            }
            if self.version[u as usize] != candidate.vu || self.version[v as usize] != candidate.vv
            {
                continue; // Stale: the neighbourhood moved after this was queued.
            }
            let (p, _) = self.placement(u, v);
            let Some((dying, surviving)) = self.check_collapse(u, v, p) else {
                continue;
            };
            self.apply_collapse(u, v, p, &dying, &surviving);
            self.push_edges_around(&mut heap, u);
        }
    }

    fn into_mesh(self) -> Mesh {
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut vertices = Vec::new();
        let mut faces = Vec::with_capacity(self.live_faces);
        for (fi, face) in self.faces.iter().enumerate() {
            if !self.alive_face[fi] {
                continue;
            }
            let mut out = [0u32; 3];
            for (k, &v) in face.iter().enumerate() {
                let slot = &mut remap[v as usize];
                if *slot == u32::MAX {
                    *slot = vertices.len() as u32;
                    let p = self.positions[v as usize];
                    vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                }
                out[k] = *slot;
            }
            faces.push(out);
        }
        Mesh {
            vertices,
            faces,
            ..Mesh::default()
        }
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Progress sink that records the calls and never cancels.
    #[derive(Default)]
    struct Calls(Vec<(u32, u32)>);

    fn noop() -> impl FnMut(u32, u32) -> anyhow::Result<()> {
        |_, _| Ok(())
    }

    /// A sphere displaced along ONE query axis must come out displaced along
    /// the SAME glTF axis. The grid is laid out `[qz][qy][qx]` exactly as
    /// `ShapeVae::reshape_grid_logits` hands it over (`comfy/sd.py:1277`
    /// after `vae.py:976`), and the mesher's `fliplr` must bring the columns
    /// back to `(x, y, z)`. With the wrapper's move missing, a `+x` offset
    /// surfaced on glTF `+Z` and every mesh was rotated.
    #[test]
    fn a_displacement_along_query_x_lands_on_gltf_x() {
        let n = 24usize;
        let c = (n as f32 - 1.0) / 2.0;
        let offset = 6.0f32;
        let mut logits = Vec::with_capacity(n * n * n);
        for qz in 0..n {
            for qy in 0..n {
                for qx in 0..n {
                    let d = ((qx as f32 - c - offset).powi(2)
                        + (qy as f32 - c).powi(2)
                        + (qz as f32 - c).powi(2))
                    .sqrt();
                    logits.push(4.0 - d);
                }
            }
        }
        let grid = OccupancyGrid::new(logits, [n, n, n]).unwrap();
        let mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut |_, _| Ok(())).unwrap();
        assert!(!mesh.vertices.is_empty());
        let count = mesh.vertices.len() as f32;
        let centroid = mesh.vertices.iter().fold([0.0f32; 3], |acc, v| {
            [
                acc[0] + v[0] / count,
                acc[1] + v[1] / count,
                acc[2] + v[2] / count,
            ]
        });
        // Normalised by `v_max = n`, so `offset` cells is `2 * offset / n`.
        let expected = 2.0 * offset / n as f32;
        assert!(
            (centroid[0] - expected).abs() < 0.05,
            "the +x displacement must appear on glTF x, centroid = {centroid:?}"
        );
        assert!(
            centroid[1].abs() < 0.05 && centroid[2].abs() < 0.05,
            "no displacement may leak onto y or z, centroid = {centroid:?}"
        );
    }

    /// The same displacement, but starting from the DECODER's flat logit
    /// order and going through `ShapeVae::reshape_grid_logits` before the
    /// mesher — the two conventions composed, exactly as `decode_occupancy`
    /// composes them. The test above pins the mesher's half and the
    /// `reshape_grid_logits` test pins the reshape's half; this one refuses
    /// a future change that flips both in the same direction and leaves each
    /// half-test green.
    #[test]
    fn a_decoder_order_displacement_along_x_survives_reshape_and_extraction() {
        use crate::hunyuan3d::shape_vae::ShapeVae;
        use candle_core::{Device, Tensor};

        let octree = 23usize;
        let n = octree + 1;
        let c = (n as f32 - 1.0) / 2.0;
        let offset = 6.0f32;
        // `query_grid` order: x slowest, z fastest (`vae.py:440-441`).
        let mut flat = Vec::with_capacity(n * n * n);
        for qx in 0..n {
            for qy in 0..n {
                for qz in 0..n {
                    let d = ((qx as f32 - c - offset).powi(2)
                        + (qy as f32 - c).powi(2)
                        + (qz as f32 - c).powi(2))
                    .sqrt();
                    flat.push(4.0 - d);
                }
            }
        }
        let logits = Tensor::from_vec(flat, (1, n * n * n), &Device::Cpu).unwrap();
        let ordered = ShapeVae::reshape_grid_logits(&logits, octree)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let grid = OccupancyGrid::new(ordered, [n, n, n]).unwrap();
        let mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut |_, _| Ok(())).unwrap();
        assert!(!mesh.vertices.is_empty());
        let count = mesh.vertices.len() as f32;
        let centroid = mesh.vertices.iter().fold([0.0f32; 3], |acc, v| {
            [
                acc[0] + v[0] / count,
                acc[1] + v[1] / count,
                acc[2] + v[2] / count,
            ]
        });
        let expected = 2.0 * offset / n as f32;
        assert!(
            (centroid[0] - expected).abs() < 0.05,
            "a +x query displacement must reach glTF x through the whole pipeline, centroid = {centroid:?}"
        );
        assert!(
            centroid[1].abs() < 0.05 && centroid[2].abs() < 0.05,
            "no displacement may leak onto y or z, centroid = {centroid:?}"
        );
    }

    /// Signed-distance-ish logits for a sphere of `radius` (in cells) centred
    /// in an `n^3` grid: positive inside, negative outside.
    fn sphere_grid(n: usize, radius: f32) -> OccupancyGrid {
        let c = (n as f32 - 1.0) / 2.0;
        let mut logits = Vec::with_capacity(n * n * n);
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    let d = ((i0 as f32 - c).powi(2)
                        + (i1 as f32 - c).powi(2)
                        + (i2 as f32 - c).powi(2))
                    .sqrt();
                    logits.push(radius - d);
                }
            }
        }
        OccupancyGrid::new(logits, [n, n, n]).unwrap()
    }

    /// Axis-aligned solid cube occupying the middle of an `n^3` grid.
    fn cube_grid(n: usize, half: usize) -> OccupancyGrid {
        let c = n / 2;
        let mut logits = vec![-1.0f32; n * n * n];
        for i0 in c - half..=c + half {
            for i1 in c - half..=c + half {
                for i2 in c - half..=c + half {
                    logits[(i0 * n + i1) * n + i2] = 1.0;
                }
            }
        }
        OccupancyGrid::new(logits, [n, n, n]).unwrap()
    }

    #[test]
    fn grid_rejects_mismatched_dimensions() {
        assert!(OccupancyGrid::new(vec![0.0; 7], [2, 2, 2]).is_err());
        assert!(OccupancyGrid::new(vec![0.0; 8], [2, 2, 2]).is_ok());
    }

    #[test]
    fn grid_padded_reads_shift_by_one_and_zero_the_low_pad() {
        let grid = OccupancyGrid::new((0..8).map(|v| v as f32).collect(), [2, 2, 2]).unwrap();
        assert_eq!(grid.at(1, 0, 1), 5.0);
        // Padded index 0 on any axis is the constant-zero pad.
        assert_eq!(grid.padded(0, 1, 1), 0.0);
        // Padded index p reads unpadded p - 1.
        assert_eq!(grid.padded(2, 1, 2), 5.0);
    }

    #[test]
    fn surface_net_sphere_is_plausible_and_in_bounds() {
        let grid = sphere_grid(24, 8.0);
        let mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut noop()).unwrap();

        assert!(mesh.vertex_count() > 100, "got {}", mesh.vertex_count());
        assert!(mesh.face_count() > 100, "got {}", mesh.face_count());
        mesh.validate().unwrap();

        let (min, max) = mesh.bounds();
        for a in 0..3 {
            assert!(min[a] >= -1.0, "min[{a}] = {}", min[a]);
            assert!(max[a] <= 1.0, "max[{a}] = {}", max[a]);
        }
        // The sphere has radius 8 in a 24-cell grid, so it spans roughly
        // two-thirds of the normalized cube — not a degenerate speck.
        for a in 0..3 {
            assert!(max[a] - min[a] > 0.5, "extent[{a}] = {}", max[a] - min[a]);
        }
    }

    #[test]
    fn progress_is_reported_and_bounded() {
        let mut calls = Calls::default();
        let grid = sphere_grid(16, 5.0);
        {
            let mut cb = |c: u32, t: u32| {
                calls.0.push((c, t));
                Ok(())
            };
            extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut cb).unwrap();
        }
        assert!(!calls.0.is_empty());
        // Tens of updates, not millions: 16^3 = 4096 cells.
        assert!(calls.0.len() < 200, "got {} progress calls", calls.0.len());
        for (c, t) in &calls.0 {
            assert_eq!(*t, SURFNET_TICKS);
            assert!(c <= t);
        }
        assert_eq!(
            calls.0.last().copied(),
            Some((SURFNET_TICKS, SURFNET_TICKS))
        );
    }

    #[test]
    fn progress_error_cancels_extraction() {
        let grid = sphere_grid(16, 5.0);
        let mut n = 0;
        let mut cb = |_: u32, _: u32| {
            n += 1;
            if n > 2 {
                anyhow::bail!("cancelled");
            }
            Ok(())
        };
        let err = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut cb).unwrap_err();
        assert!(err.to_string().contains("cancelled"));
    }

    #[test]
    fn empty_grid_yields_an_empty_mesh_for_both_algorithms() {
        let grid = OccupancyGrid::new(vec![-1.0; 8 * 8 * 8], [8, 8, 8]).unwrap();
        for algorithm in [MeshAlgorithm::SurfaceNet, MeshAlgorithm::Basic] {
            let mesh = extract(&grid, algorithm, 0.0, &mut noop()).unwrap();
            assert_eq!(mesh.vertex_count(), 0, "{algorithm:?}");
            assert_eq!(mesh.face_count(), 0, "{algorithm:?}");
            assert!(mesh.is_empty());
        }
    }

    #[test]
    fn full_grid_has_no_interior_surface() {
        // Every sample is inside, so surface nets find no sign change at all
        // and the blocky mesher only sees the grid boundary as "exposed".
        let grid = OccupancyGrid::new(vec![1.0; 8 * 8 * 8], [8, 8, 8]).unwrap();

        let sn = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut noop()).unwrap();
        // The zero pad around the grid IS a sign change, so the outer shell
        // survives; nothing interior does.
        sn.validate().unwrap();

        let basic = extract(&grid, MeshAlgorithm::Basic, 0.0, &mut noop()).unwrap();
        basic.validate().unwrap();
        // 6 faces of an 8x8 grid, 2 triangles each.
        assert_eq!(basic.face_count(), 6 * 8 * 8 * 2);
    }

    #[test]
    fn surface_net_shares_vertices_where_basic_duplicates_them() {
        let grid = sphere_grid(24, 8.0);
        let sn = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut noop()).unwrap();
        let basic = extract(&grid, MeshAlgorithm::Basic, 0.0, &mut noop()).unwrap();
        sn.validate().unwrap();
        basic.validate().unwrap();
        assert!(sn.face_count() > 0);

        // Both algorithms emit exactly one quad per grid edge that crosses the
        // surface — `basic` as the exposed face of a solid voxel, surface nets
        // as the quad joining the four cells around that edge — so their
        // TRIANGLE counts coincide for a shape that does not touch the grid
        // boundary. Surface nets is not a decimation pass; the saving is in
        // vertices, and in the surface being smooth instead of blocky.
        assert!(
            sn.face_count() <= basic.face_count(),
            "surface net {} faces vs basic {}",
            sn.face_count(),
            basic.face_count()
        );

        // `basic` emits four unshared vertices per quad; surface nets emits one
        // shared vertex per cell, so the buffer is roughly 4x smaller.
        assert!(
            sn.vertex_count() * 3 < basic.vertex_count(),
            "surface net {} vertices vs basic {}",
            sn.vertex_count(),
            basic.vertex_count()
        );
        assert_eq!(basic.vertex_count(), basic.face_count() * 2);
    }

    #[test]
    fn validate_rejects_out_of_range_faces_and_nan() {
        let bad_index = Mesh {
            vertices: vec![[0.0; 3]; 3],
            faces: vec![[0, 1, 3]],
            ..Mesh::default()
        };
        assert!(matches!(
            bad_index.validate(),
            Err(MeshError::FaceIndexOutOfRange { index: 3, .. })
        ));

        let nan = Mesh {
            vertices: vec![[0.0, f32::NAN, 0.0]],
            faces: vec![],
            ..Mesh::default()
        };
        assert!(matches!(
            nan.validate(),
            Err(MeshError::NonFiniteVertex { vertex: 0 })
        ));

        let short_normals = Mesh {
            vertices: vec![[0.0; 3]; 3],
            faces: vec![],
            normals: Some(vec![[0.0, 1.0, 0.0]]),
            ..Mesh::default()
        };
        assert!(matches!(
            short_normals.validate(),
            Err(MeshError::AttributeLengthMismatch {
                attribute: "normals",
                ..
            })
        ));
    }

    #[test]
    fn smooth_normals_are_unit_length_and_point_outward() {
        let grid = cube_grid(16, 4);
        let mut mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut noop()).unwrap();
        compute_smooth_normals(&mut mesh);
        let normals = mesh.normals.clone().unwrap();
        assert_eq!(normals.len(), mesh.vertex_count());

        // The cube is centred in the grid and `normalize_and_flip` centres on
        // the grid, so the mesh centroid is very close to the origin; use the
        // actual centroid so the test does not depend on that.
        let mut centroid = [0.0f32; 3];
        for v in &mesh.vertices {
            for a in 0..3 {
                centroid[a] += v[a] / mesh.vertex_count() as f32;
            }
        }

        let mut outward = 0usize;
        for (v, n) in mesh.vertices.iter().zip(&normals) {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "normal length {len}");
            let r = [v[0] - centroid[0], v[1] - centroid[1], v[2] - centroid[2]];
            if r[0] * n[0] + r[1] * n[1] + r[2] * n[2] > 0.0 {
                outward += 1;
            }
        }
        // Every normal on a convex closed surface must face away from the
        // centre; a globally inverted winding would score 0 here.
        assert_eq!(
            outward,
            mesh.vertex_count(),
            "{}/{} normals point outward",
            outward,
            mesh.vertex_count()
        );
    }

    /// A closed, manifold, uniformly-triangulated sphere: an icosahedron
    /// subdivided `subdivisions` times with every vertex pushed onto the unit
    /// sphere. 20 * 4^n faces. Independent of the surface-net extractor, so a
    /// decimation bug cannot be masked by an extraction bug.
    fn icosphere(subdivisions: u32) -> Mesh {
        let t = (1.0 + 5.0f32.sqrt()) / 2.0;
        let mut vertices: Vec<[f32; 3]> = vec![
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        let normalize = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        for v in vertices.iter_mut() {
            *v = normalize(*v);
        }
        for _ in 0..subdivisions {
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next = Vec::with_capacity(faces.len() * 4);
            for f in &faces {
                let mut mid = [0u32; 3];
                for k in 0..3 {
                    let (a, b) = (f[k], f[(k + 1) % 3]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    mid[k] = *cache.entry(key).or_insert_with(|| {
                        let pa = vertices[a as usize];
                        let pb = vertices[b as usize];
                        vertices.push(normalize([
                            (pa[0] + pb[0]) * 0.5,
                            (pa[1] + pb[1]) * 0.5,
                            (pa[2] + pb[2]) * 0.5,
                        ]));
                        vertices.len() as u32 - 1
                    });
                }
                next.push([f[0], mid[0], mid[2]]);
                next.push([f[1], mid[1], mid[0]]);
                next.push([f[2], mid[2], mid[1]]);
                next.push([mid[0], mid[1], mid[2]]);
            }
            faces = next;
        }
        Mesh {
            vertices,
            faces,
            ..Mesh::default()
        }
    }

    /// No two faces may share the same three vertices, and no face may repeat a
    /// vertex. `validate` cannot see either; both are what a bad collapse
    /// produces.
    fn assert_no_degenerate_or_duplicate_faces(mesh: &Mesh) {
        let mut seen: Vec<[u32; 3]> = Vec::with_capacity(mesh.face_count());
        for face in &mesh.faces {
            let mut key = *face;
            key.sort_unstable();
            assert!(
                key[0] != key[1] && key[1] != key[2],
                "degenerate face {face:?}"
            );
            assert!(!seen.contains(&key), "duplicate face {face:?}");
            seen.push(key);
        }
    }

    #[test]
    fn simplify_decimates_a_sphere_to_the_target_without_moving_the_surface() {
        let sphere = icosphere(3);
        assert_eq!(sphere.face_count(), 1280);
        sphere.validate().unwrap();

        let target = sphere.face_count() / 4;
        let out = simplify(&sphere, target).unwrap();

        out.validate().unwrap();
        assert_no_degenerate_or_duplicate_faces(&out);
        // Every collapse on a closed manifold removes exactly two faces, so the
        // loop can only overshoot the target by one.
        assert!(
            out.face_count() <= target && out.face_count() >= target - 1,
            "got {} faces for a target of {target}",
            out.face_count()
        );
        // No orphan vertices: the output is compacted.
        let referenced: std::collections::HashSet<u32> =
            out.faces.iter().flatten().copied().collect();
        assert_eq!(referenced.len(), out.vertex_count());

        // Decimation reshapes the tessellation; it must not move the surface.
        let (min_in, max_in) = sphere.bounds();
        let (min_out, max_out) = out.bounds();
        for a in 0..3 {
            assert!(
                (min_in[a] - min_out[a]).abs() < 0.05,
                "min[{a}] moved {} -> {}",
                min_in[a],
                min_out[a]
            );
            assert!(
                (max_in[a] - max_out[a]).abs() < 0.05,
                "max[{a}] moved {} -> {}",
                max_in[a],
                max_out[a]
            );
        }
        // Still recognisably a unit sphere: no vertex has drifted off it.
        for v in &out.vertices {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - 1.0).abs() < 0.1, "vertex at radius {r}");
        }
    }

    #[test]
    fn simplify_at_or_above_the_current_count_is_a_no_op() {
        let mut sphere = icosphere(1);
        compute_smooth_normals(&mut sphere);
        for target in [sphere.face_count(), sphere.face_count() + 1, usize::MAX] {
            let out = simplify(&sphere, target).unwrap();
            assert_eq!(out.vertices, sphere.vertices);
            assert_eq!(out.faces, sphere.faces);
            // The no-op path is a whole clone, so attributes survive — there is
            // no decimation to invalidate them.
            assert_eq!(out.normals, sphere.normals);
        }
    }

    #[test]
    fn simplify_to_zero_or_one_face_stays_valid() {
        let sphere = icosphere(2);
        for target in [0usize, 1] {
            let out = simplify(&sphere, target).unwrap();
            out.validate().unwrap();
            assert_no_degenerate_or_duplicate_faces(&out);
            // A closed surface cannot legally decimate below a tetrahedron;
            // what matters is that it terminates and stays a valid mesh.
            assert!(out.face_count() < sphere.face_count());
            assert!(
                out.face_count() >= 4 || out.face_count() == 0,
                "got {} faces",
                out.face_count()
            );
        }
    }

    #[test]
    fn simplify_terminates_on_a_mesh_that_cannot_be_decimated() {
        // Every edge of a tetrahedron would fold it into two coincident
        // triangles, so all six collapses are rejected and the mesh comes back
        // untouched instead of the loop spinning.
        let tetra = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            ..Mesh::default()
        };
        tetra.validate().unwrap();
        let out = simplify(&tetra, 2).unwrap();
        out.validate().unwrap();
        assert_no_degenerate_or_duplicate_faces(&out);
        assert_eq!(out.face_count(), 4);
        assert_eq!(out.vertex_count(), 4);
    }

    #[test]
    fn simplify_handles_an_empty_mesh_and_rejects_an_invalid_one() {
        let empty = simplify(&Mesh::default(), 0).unwrap();
        assert_eq!(empty.face_count(), 0);
        assert_eq!(empty.vertex_count(), 0);

        let bad = Mesh {
            vertices: vec![[0.0; 3]; 3],
            faces: vec![[0, 1, 7]],
            ..Mesh::default()
        };
        assert!(simplify(&bad, 1).is_err());
    }

    #[test]
    fn simplify_preserves_an_open_boundary() {
        // A flat 4x4 grid of quads. The interior is perfectly planar, so its
        // quadric error is zero and it decimates freely; the rim is held by the
        // boundary quadrics and must not retreat.
        let n = 5usize;
        let mut vertices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                vertices.push([i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32, 0.0]);
            }
        }
        let mut faces = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = (j * n + i) as u32;
                faces.push([a, a + 1, a + n as u32]);
                faces.push([a + 1, a + n as u32 + 1, a + n as u32]);
            }
        }
        let plane = Mesh {
            vertices,
            faces,
            ..Mesh::default()
        };
        plane.validate().unwrap();

        let out = simplify(&plane, 8).unwrap();
        out.validate().unwrap();
        assert_no_degenerate_or_duplicate_faces(&out);
        assert!(out.face_count() < plane.face_count());
        let (min, max) = out.bounds();
        for a in 0..2 {
            assert!(
                min[a].abs() < 1e-5,
                "boundary min[{a}] pulled in to {}",
                min[a]
            );
            assert!(
                (max[a] - 1.0).abs() < 1e-5,
                "boundary max[{a}] pulled in to {}",
                max[a]
            );
        }
    }

    #[test]
    fn simplify_decimates_real_surface_net_output() {
        let grid = sphere_grid(24, 8.0);
        let mut mesh = extract(&grid, MeshAlgorithm::SurfaceNet, 0.0, &mut noop()).unwrap();
        compute_smooth_normals(&mut mesh);
        let target = mesh.face_count() / 5;
        let out = simplify(&mesh, target).unwrap();
        out.validate().unwrap();
        assert_no_degenerate_or_duplicate_faces(&out);
        assert!(out.face_count() <= target, "got {}", out.face_count());
        // Attributes are dropped: the merged vertex is at the quadric minimum,
        // which is not a point any original attribute was sampled at.
        assert!(out.normals.is_none());
        assert!(out.uvs.is_none());
        assert!(out.vertex_colors.is_none());
    }

    #[test]
    fn smooth_normals_survive_a_vertex_with_no_faces() {
        let mut mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        };
        compute_smooth_normals(&mut mesh);
        for n in mesh.normals.as_ref().unwrap() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal length {len}");
        }
    }
}
