//! CPU xatlas unwrap; no GPU residency is needed for this stage.

use super::mesh::Mesh;
use anyhow::{ensure, Result};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

unsafe extern "C" {
    fn mold_xatlas_generate(
        positions: *const f32,
        vertices: u32,
        indices: *const u32,
        index_count: u32,
        proceed: extern "C" fn(*const c_void, u32, u32) -> bool,
        state: *const c_void,
        out_vertices: *mut u32,
        out_indices: *mut u32,
    ) -> *mut c_void;
    fn mold_xatlas_copy(
        handle: *const c_void,
        mapping: *mut u32,
        uv: *mut f32,
        indices: *mut u32,
        vertex_capacity: u32,
        index_capacity: u32,
    ) -> bool;
    fn mold_xatlas_destroy(handle: *mut c_void);
}

struct Atlas(*mut c_void);
impl Drop for Atlas {
    fn drop(&mut self) {
        // SAFETY: the only constructor receives a nonnull owned xatlas handle.
        unsafe { mold_xatlas_destroy(self.0) };
    }
}

/// The four phases xatlas walks, in the order it runs them.
///
/// `xatlas.h:241-247`. They are reported so a long unwrap says which one it is
/// in rather than sitting on a single opaque stage name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwrapPhase {
    AddMesh,
    ComputeCharts,
    PackCharts,
    BuildOutputMeshes,
}

impl UnwrapPhase {
    /// How many phases precede this one, so a caller can lay the four out on
    /// one 0..=400 bar.
    pub const COUNT: u32 = 4;

    fn from_category(category: u32) -> Option<Self> {
        Some(match category {
            0 => Self::AddMesh,
            1 => Self::ComputeCharts,
            2 => Self::PackCharts,
            3 => Self::BuildOutputMeshes,
            // `MOLD_XATLAS_NO_CATEGORY`: a bare cancellation probe.
            _ => return None,
        })
    }

    pub fn index(self) -> u32 {
        match self {
            Self::AddMesh => 0,
            Self::ComputeCharts => 1,
            Self::PackCharts => 2,
            Self::BuildOutputMeshes => 3,
        }
    }
}

/// What the native callback is handed. `Sync` because xatlas may call it from
/// a worker thread.
struct UnwrapCallback<'a> {
    cancelled: &'a AtomicBool,
    report: &'a (dyn Fn(UnwrapPhase, u32) + Sync),
}

extern "C" fn proceed(state: *const c_void, category: u32, percent: u32) -> bool {
    // SAFETY: unwrap keeps this borrow alive until the blocking native call
    // joins all workers. Atomic reads are valid from any callback thread, and
    // the reporter is `Sync`.
    let callback = unsafe { &*state.cast::<UnwrapCallback<'_>>() };
    if let Some(phase) = UnwrapPhase::from_category(category) {
        (callback.report)(phase, percent);
    }
    !callback.cancelled.load(Ordering::Acquire)
}

/// Unwrap with the exact xatlas-python 0.0.9 defaults used by Tencent. Existing
/// UVs and normals do not influence charting; all corner attributes are remapped
/// to duplicated seam vertices. This CPU stage polls cancellation in xatlas.
pub fn unwrap(source: &Mesh, cancelled: &AtomicBool) -> Result<Mesh> {
    unwrap_reporting(source, cancelled, &|_, _| {})
}

/// [`unwrap`], reporting xatlas's own phase and percent as it goes.
///
/// Cancellation and progress share one native callback because xatlas has
/// exactly one — `ProgressFunc` (`xatlas.h:250`) — and it is the only hook
/// into a stage that is otherwise opaque for its whole duration.
pub fn unwrap_reporting(
    source: &Mesh,
    cancelled: &AtomicBool,
    report: &(dyn Fn(UnwrapPhase, u32) + Sync),
) -> Result<Mesh> {
    ensure!(!cancelled.load(Ordering::Acquire), "UV unwrap cancelled");
    source.validate()?;
    ensure!(
        !source.faces.is_empty() && !source.vertices.is_empty(),
        "cannot unwrap an empty mesh"
    );
    ensure!(
        source.faces.len() <= mold_core::validation::MESH_MAX_TARGET_FACES as usize,
        "mesh exceeds UV face budget"
    );
    ensure!(
        source.vertices.len() <= source.faces.len() * 3,
        "mesh has too many unused vertices"
    );
    ensure!(
        source
            .vertices
            .iter()
            .flatten()
            .all(|v| v.abs() <= 1_000_000.),
        "normalize mesh coordinates before UV unwrapping"
    );
    for values in [&source.normals, &source.vertex_colors]
        .into_iter()
        .flatten()
    {
        ensure!(
            values.iter().flatten().all(|v| v.is_finite()),
            "nonfinite mesh attribute"
        );
    }
    for face in &source.faces {
        let [a, b, c] = face.map(|i| source.vertices[i as usize].map(f64::from));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        ensure!(
            cross.iter().any(|value| *value != 0.),
            "degenerate triangle cannot be unwrapped"
        );
    }
    let vertices = u32::try_from(source.vertices.len())?;
    let indices = u32::try_from(source.faces.len() * 3)?;
    let (mut out_vertices, mut out_indices) = (0, 0);
    let callback = UnwrapCallback { cancelled, report };
    // SAFETY: arrays of f32/u32 have contiguous layout; input lengths were
    // checked, the immutable input and cancellation flag outlive every worker.
    let handle = unsafe {
        mold_xatlas_generate(
            source.vertices.as_ptr().cast(),
            vertices,
            source.faces.as_ptr().cast(),
            indices,
            proceed,
            (&callback as *const UnwrapCallback<'_>).cast(),
            &mut out_vertices,
            &mut out_indices,
        )
    };
    ensure!(
        !handle.is_null(),
        "xatlas failed or UV unwrap was cancelled"
    );
    let atlas = Atlas(handle);
    ensure!(
        out_vertices > 0 && out_vertices <= indices && out_indices == indices,
        "invalid xatlas output dimensions"
    );
    let mut mapping = vec![0u32; out_vertices as usize];
    let mut uv = vec![[0f32; 2]; out_vertices as usize];
    let mut faces = vec![[0u32; 3]; source.faces.len()];
    // SAFETY: owned output buffers have the exact element counts reported by
    // this still-live atlas; the native bridge checks those counts again.
    ensure!(
        unsafe {
            mold_xatlas_copy(
                atlas.0,
                mapping.as_mut_ptr(),
                uv.as_mut_ptr().cast(),
                faces.as_mut_ptr().cast(),
                out_vertices,
                out_indices,
            )
        },
        "invalid xatlas copy"
    );
    ensure!(
        mapping.iter().all(|&i| i < vertices),
        "xatlas vertex mapping out of range"
    );
    ensure!(
        uv.iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "invalid xatlas UV coordinate"
    );
    let remap = |values: &Vec<[f32; 3]>| mapping.iter().map(|&i| values[i as usize]).collect();
    let mesh = Mesh {
        vertices: remap(&source.vertices),
        faces,
        normals: source.normals.as_ref().map(remap),
        vertex_colors: source.vertex_colors.as_ref().map(remap),
        uvs: Some(uv),
    };
    mesh.validate()?;
    ensure!(!cancelled.load(Ordering::Acquire), "UV unwrap cancelled");
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::super::mesh::Mesh;
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn tetrahedron() -> Mesh {
        Mesh {
            vertices: vec![[1., 1., 1.], [-1., -1., 1.], [-1., 1., -1.], [1., -1., -1.]],
            faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            normals: Some(vec![
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [-1., 0., 0.],
            ]),
            vertex_colors: Some(vec![[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
            ..Default::default()
        }
    }

    #[test]
    fn unwrap_duplicates_seams_and_preserves_every_corner_attribute() {
        let source = tetrahedron();
        let output = unwrap(&source, &AtomicBool::new(false)).unwrap();
        assert!(output.vertices.len() > source.vertices.len());
        assert_eq!(output.faces.len(), source.faces.len());
        let uvs = output.uvs.as_ref().unwrap();
        assert!(uvs
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        for (old, new) in source.faces.iter().zip(&output.faces) {
            for (&old, &new) in old.iter().zip(new) {
                assert_eq!(source.vertices[old as usize], output.vertices[new as usize]);
                assert_eq!(
                    source.normals.as_ref().unwrap()[old as usize],
                    output.normals.as_ref().unwrap()[new as usize]
                );
                assert_eq!(
                    source.vertex_colors.as_ref().unwrap()[old as usize],
                    output.vertex_colors.as_ref().unwrap()[new as usize]
                );
            }
        }
        assert_eq!(output, unwrap(&source, &AtomicBool::new(false)).unwrap());
    }

    #[test]
    fn unwrap_refuses_invalid_geometry_before_native_code() {
        let mut source = tetrahedron();
        source.faces[0] = [0, 0, 1];
        assert!(unwrap(&source, &AtomicBool::new(false)).is_err());
        source = tetrahedron();
        source.vertices[0][0] = f32::NAN;
        assert!(unwrap(&source, &AtomicBool::new(false)).is_err());
        assert!(unwrap(&Mesh::default(), &AtomicBool::new(false)).is_err());
    }

    #[test]
    fn unwrap_matches_the_executable_xatlas_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hunyuan3d/uv-tetrahedron.json"
        ))
        .unwrap();
        let source = tetrahedron();
        let output = unwrap(&source, &AtomicBool::new(false)).unwrap();
        let mapping: Vec<usize> = serde_json::from_value(fixture["mapping"].clone()).unwrap();
        let uv: Vec<[f32; 2]> = serde_json::from_value(fixture["uv"].clone()).unwrap();
        let indices: Vec<[u32; 3]> = serde_json::from_value(fixture["indices"].clone()).unwrap();
        assert_eq!(
            output.vertices,
            mapping
                .iter()
                .map(|&i| source.vertices[i])
                .collect::<Vec<_>>()
        );
        assert_eq!(output.faces, indices);
        for (actual, expected) in output
            .uvs
            .unwrap()
            .iter()
            .flatten()
            .zip(uv.iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn unwrap_preserves_geometry_with_unused_source_vertices() {
        let mut source = tetrahedron();
        source.vertices.push([0., 0., 0.]);
        source.normals.as_mut().unwrap().push([0., 0., 1.]);
        source.vertex_colors.as_mut().unwrap().push([1., 1., 1.]);
        let output = unwrap(&source, &AtomicBool::new(false)).unwrap();
        assert_eq!(output.faces.len(), source.faces.len());
        for (old, new) in source.faces.iter().zip(&output.faces) {
            for (&old, &new) in old.iter().zip(new) {
                assert_eq!(source.vertices[old as usize], output.vertices[new as usize]);
            }
        }
    }

    #[test]
    fn unwrap_preserves_slivers_that_xatlas_leaves_out_of_charts() {
        let mut source = tetrahedron();
        source
            .vertices
            .extend([[0., 0., 0.], [1e-8, 0., 0.], [0., 1e-8, 0.]]);
        source.faces.push([4, 5, 6]);
        source.normals.as_mut().unwrap().extend([[0., 0., 1.]; 3]);
        source
            .vertex_colors
            .as_mut()
            .unwrap()
            .extend([[1., 1., 1.]; 3]);
        let output = unwrap(&source, &AtomicBool::new(false)).unwrap();
        assert_eq!(output.faces.len(), source.faces.len());
        let last = output.faces.last().unwrap();
        for (&old, &new) in source.faces.last().unwrap().iter().zip(last) {
            assert_eq!(source.vertices[old as usize], output.vertices[new as usize]);
            assert_eq!(output.uvs.as_ref().unwrap()[new as usize], [0., 0.]);
        }
    }

    #[test]
    fn unwrap_honors_cancellation() {
        assert!(unwrap(&tetrahedron(), &AtomicBool::new(true)).is_err());
    }

    /// xatlas hands the bridge a category and a percent on every callback and
    /// they used to be dropped on the floor, which is why an unwrap that ran
    /// for an hour reported nothing at all (#1666).
    #[test]
    fn unwrap_reports_xatlas_phases_in_order() {
        let seen = std::sync::Mutex::new(Vec::new());
        let output = unwrap_reporting(&tetrahedron(), &AtomicBool::new(false), &|phase, pct| {
            seen.lock().unwrap().push((phase, pct));
        })
        .unwrap();
        assert!(!output.faces.is_empty());

        let seen = seen.into_inner().unwrap();
        assert!(!seen.is_empty(), "the unwrap reported no progress at all");
        assert!(seen.iter().all(|&(_, pct)| pct <= 100));
        // The phases run in the order `xatlas.h:241-247` declares them, so the
        // 0..=400 bar the paint stage lays them on never goes backwards.
        assert!(
            seen.windows(2)
                .all(|pair| pair[0].0.index() <= pair[1].0.index()),
            "phases were reported out of order: {seen:?}"
        );
        for expected in [UnwrapPhase::ComputeCharts, UnwrapPhase::PackCharts] {
            assert!(
                seen.iter().any(|&(phase, _)| phase == expected),
                "{expected:?} was never reported"
            );
        }
    }

    /// The progress callback is also the ONLY cancellation hook xatlas has, so
    /// a cancel raised while native code owns the thread has to travel back
    /// through it. Flipping the flag from inside the callback tests exactly
    /// that path without racing a second thread against a fast unwrap.
    #[test]
    fn unwrap_stops_when_the_native_callback_observes_a_cancel() {
        let cancelled = AtomicBool::new(false);
        let error = unwrap_reporting(&tetrahedron(), &cancelled, &|_, _| {
            cancelled.store(true, Ordering::Release);
        })
        .expect_err("a cancel observed inside xatlas must stop the unwrap");
        assert!(error.to_string().contains("cancel"), "{error}");
    }
}
