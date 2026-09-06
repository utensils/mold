//! Paint-specific view policy, independent of poster/turntable cameras.
use anyhow::{ensure, Result};

/// Tencent `hy3dpaint/utils/pipeline_utils.py:40-109`, pinned at
/// 82920d643c0dc2f7bfd7255f45f62d386edfe60c. Visibility uses zero-based face
/// indices, without the rasterizer's background sentinel. The first six views
/// always survive. Strict comparison preserves the earliest candidate on ties.
pub fn select_views(areas: &[f32], visible: &[Vec<u32>], limit: usize) -> Result<Vec<usize>> {
    ensure!(
        visible.len() == 30,
        "paint requires the 30 canonical candidate views"
    );
    ensure!(
        (6..=30).contains(&limit),
        "paint view limit must be 6 through 30"
    );
    ensure!(
        !areas.is_empty() && areas.len() <= 8_000_000,
        "invalid paint face count"
    );
    ensure!(
        areas.iter().all(|area| area.is_finite() && *area >= 0.),
        "invalid face area"
    );
    let total: f32 = areas.iter().sum();
    ensure!(
        total.is_finite() && total > 0.,
        "mesh has no finite positive surface area"
    );
    let ratios: Vec<_> = areas.iter().map(|area| area / total).collect();
    let mut unique = Vec::with_capacity(30);
    for faces in visible {
        ensure!(
            faces.len() <= areas.len(),
            "visibility must contain at most one entry per face"
        );
        ensure!(
            faces.iter().all(|&face| (face as usize) < areas.len()),
            "visibility face is out of range"
        );
        let mut faces = faces.clone();
        faces.sort_unstable();
        faces.dedup();
        unique.push(faces);
    }
    let mut covered = vec![false; areas.len()];
    let mut selected: Vec<_> = (0..6).collect();
    let mut used = [false; 30];
    for index in &selected {
        used[*index] = true;
        for &face in &unique[*index] {
            covered[face as usize] = true;
        }
    }
    while selected.len() < limit {
        let mut best = None;
        let mut best_gain = 0.;
        for (index, faces) in unique.iter().enumerate() {
            if used[index] {
                continue;
            }
            let gain: f32 = faces
                .iter()
                .filter(|&&face| !covered[face as usize])
                .map(|&face| ratios[face as usize])
                .sum();
            if gain > best_gain {
                best = Some(index);
                best_gain = gain;
            }
        }
        if best_gain <= 0.01 {
            break;
        }
        let index = best.expect("positive gain has a candidate");
        selected.push(index);
        used[index] = true;
        for &face in &unique[index] {
            covered[face as usize] = true;
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        areas: Vec<f32>,
        visible: Vec<Vec<u32>>,
        limit: usize,
        selected: Vec<usize>,
    }

    #[test]
    fn matches_executable_tencent_selection_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hunyuan3d/view-selection.json"
        ))
        .unwrap();
        for case in fixture.cases {
            assert_eq!(
                select_views(&case.areas, &case.visible, case.limit).unwrap(),
                case.selected
            );
        }
    }

    #[test]
    fn invalid_coverage_is_refused() {
        assert!(select_views(&[1.], &vec![vec![1]; 30], 6).is_err());
        assert!(select_views(&[f32::NAN], &vec![vec![]; 30], 6).is_err());
        assert!(select_views(&[0.], &vec![vec![]; 30], 6).is_err());
        assert!(select_views(&[1.], &vec![vec![]; 30], 5).is_err());
    }
}
