//! OpenCV-compatible depth edges for Tencent paint's reliability mask.

use anyhow::{ensure, Result};

/// Tencent MeshRender.py:1105-1107 calls Canny on the depth bytes with
/// thresholds 30/80, aperture 3 and L1 magnitude. Port of OpenCV 4.10.0
/// modules/imgproc/src/canny.cpp:293-299, 380-384, 595-639, 914-927.
/// Sobel uses replicated input borders; nonmax suppression uses zero magnitude
/// outside the image. All intermediate storage belongs to this cancellable call.
pub fn depth_edges(
    pixels: &[u8],
    width: usize,
    height: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<bool>> {
    ensure!(
        (1..=2048).contains(&width) && (1..=2048).contains(&height),
        "invalid paint depth edge dimensions"
    );
    ensure!(
        pixels.len() == width * height,
        "paint depth byte count differs"
    );
    checkpoint()?;
    let mut gradient = vec![[0i16; 2]; pixels.len()];
    checkpoint()?;
    let mut magnitude = vec![0i32; pixels.len()];
    for y in 0..height {
        checkpoint()?;
        let top = y.saturating_sub(1) * width;
        let mid = y * width;
        let bottom = (y + 1).min(height - 1) * width;
        for x in 0..width {
            let left = x.saturating_sub(1);
            let right = (x + 1).min(width - 1);
            let sample = |row, col| i16::from(pixels[row + col]);
            let dx = sample(top, right) - sample(top, left)
                + 2 * (sample(mid, right) - sample(mid, left))
                + sample(bottom, right)
                - sample(bottom, left);
            let dy = sample(bottom, left) - sample(top, left)
                + 2 * (sample(bottom, x) - sample(top, x))
                + sample(bottom, right)
                - sample(top, right);
            gradient[mid + x] = [dx, dy];
            magnitude[mid + x] = i32::from(dx.abs()) + i32::from(dy.abs());
        }
    }
    checkpoint()?;
    // 0 = weak candidate, 1 = suppressed, 2 = connected strong edge.
    let mut state = vec![1u8; pixels.len()];
    let mut stack = Vec::new();
    let mag = |x: isize, y: isize| {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            0
        } else {
            magnitude[y as usize * width + x as usize]
        }
    };
    for y in 0..height {
        checkpoint()?;
        for x in 0..width {
            let index = y * width + x;
            let m = magnitude[index];
            if m <= 30 {
                continue;
            }
            let [dx, dy] = gradient[index].map(i32::from);
            let ax = dx.abs();
            let ay = dy.abs() << 15;
            let tg22x = ax * 13573;
            let (x, y) = (x as isize, y as isize);
            let maximum = if ay < tg22x {
                m > mag(x - 1, y) && m >= mag(x + 1, y)
            } else if ay > tg22x + (ax << 16) {
                m > mag(x, y - 1) && m >= mag(x, y + 1)
            } else {
                let sign = if (dx ^ dy) < 0 { -1 } else { 1 };
                m > mag(x - sign, y - 1) && m > mag(x + sign, y + 1)
            };
            if maximum {
                if m > 80 {
                    state[index] = 2;
                    stack.push(index);
                } else {
                    state[index] = 0;
                }
            }
        }
    }
    let mut visited = 0usize;
    while let Some(index) = stack.pop() {
        if visited.is_multiple_of(4096) {
            checkpoint()?;
        }
        visited += 1;
        let x = index % width;
        let y = index / width;
        for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let next = ny * width + nx;
                if state[next] == 0 {
                    state[next] = 2;
                    stack.push(next);
                }
            }
        }
    }
    checkpoint()?;
    let mut result = Vec::with_capacity(pixels.len());
    for (index, value) in state.into_iter().enumerate() {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        result.push(value == 2);
    }
    checkpoint()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    #[test]
    fn depth_edges_match_opencv() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-edges.safetensors"),
            &Device::Cpu,
        )?;
        for name in [
            "single",
            "row",
            "column",
            "noise",
            "weak",
            "directions",
            "ties",
            "bridge",
        ] {
            let source = &fixture[&format!("{name}.pixels")];
            let (height, width) = source.dims2()?;
            let pixels = source.flatten_all()?.to_vec1::<u8>()?;
            let expected = fixture[&format!("{name}.edges")]
                .flatten_all()?
                .to_vec1::<u8>()?;
            let actual = super::depth_edges(&pixels, width, height, &mut || Ok(()))?;
            assert_eq!(
                actual,
                expected.iter().map(|v| *v != 0).collect::<Vec<_>>(),
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn depth_edges_validate_and_cancel() -> anyhow::Result<()> {
        use super::depth_edges;
        assert!(depth_edges(&[], 0, 1, &mut || Ok(())).is_err());
        assert!(depth_edges(&[], 2049, 1, &mut || Ok(())).is_err());
        assert!(depth_edges(&[0], 2, 2, &mut || Ok(())).is_err());
        let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 71) % 256) as u8).collect();
        let mut calls = 0;
        depth_edges(&pixels, 64, 64, &mut || {
            calls += 1;
            Ok(())
        })?;
        for cancel_at in 0..calls {
            let mut remaining = cancel_at;
            let result = depth_edges(&pixels, 64, 64, &mut || {
                anyhow::ensure!(remaining > 0, "cancelled");
                remaining -= 1;
                Ok(())
            });
            assert_eq!(result.unwrap_err().to_string(), "cancelled");
        }
        Ok(())
    }
}
