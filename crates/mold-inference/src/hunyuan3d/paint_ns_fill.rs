//! OpenCV-compatible radius-three Navier–Stokes fill for RGB paint textures.

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

use anyhow::{ensure, Result};
use std::{cmp::Ordering, collections::BinaryHeap};

const INSIDE: u8 = 2;
#[derive(PartialEq)]
struct Front {
    distance: f32,
    order: usize,
    row: usize,
    col: usize,
}
impl Eq for Front {}
impl Ord for Front {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.order.cmp(&self.order))
    }
}
impl PartialOrd for Front {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn solve(a: usize, b: usize, state: &[u8], time: &[f32]) -> f32 {
    let x = f64::from(time[a]);
    let y = f64::from(time[b]);
    let value = match (state[a] != INSIDE, state[b] != INSIDE) {
        (true, true) if (x - y).abs() < 1.0 => (x + y + (2.0 - (x - y) * (x - y)).sqrt()) * 0.5,
        (true, true) | (false, false) => 1.0 + x.min(y),
        (true, false) => 1.0 + x,
        (false, true) => 1.0 + y,
    };
    value as f32
}

fn direction(dot: f32, product: f32) -> f32 {
    if dot.abs() <= 0.01 {
        0.000001
    } else {
        // Global-scope C++ sqrt/fabs use double; the product rounded in F32.
        (f64::from(dot) / f64::from(product).sqrt()).abs() as f32
    }
}

/// Port of OpenCV 4.10 modules/photo/src/inpaint.cpp:483-595, 733-786.
/// `trust` uses Tencent's convention: 255 keeps a texel; every other value is
/// missing (OpenCV receives 255-trust). Radius is fixed at Tencent's three.
/// All work is owned by this cancellable call, with no partial publication.
pub fn fill_rgb(
    pixels: &[[u8; 3]],
    trust: &[u8],
    width: usize,
    height: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<[u8; 3]>> {
    checkpoint()?;
    ensure!(
        (2..=4096).contains(&width) && (2..=4096).contains(&height),
        "invalid NS paint dimensions"
    );
    ensure!(
        pixels.len() == width * height && trust.len() == pixels.len(),
        "invalid NS paint image lengths"
    );
    let stride = width + 2;
    let mut state = vec![0u8; stride * (height + 2)];
    let mut time = vec![1e6f32; state.len()];
    for row in 1..=height {
        checkpoint()?;
        for col in 1..=width {
            if trust[(row - 1) * width + col - 1] != 255 {
                state[row * stride + col] = INSIDE;
            }
        }
    }
    let mut queue = BinaryHeap::new();
    let mut order = 0;
    // Cross dilation minus the original mask, row-major heap insertion.
    for row in 1..=height {
        checkpoint()?;
        for col in 1..=width {
            let p = row * stride + col;
            if state[p] != INSIDE
                && [p - stride, p - 1, p + stride, p + 1]
                    .iter()
                    .any(|&q| state[q] == INSIDE)
            {
                time[p] = 0.;
                queue.push(Front {
                    distance: 0.,
                    order,
                    row,
                    col,
                });
                order += 1;
            }
        }
    }
    let mut output = pixels.to_vec();
    while let Some(front) = queue.pop() {
        checkpoint()?;
        let row = front.row;
        let col = front.col;
        state[row * stride + col] = 0;
        for (i, j) in [
            (row - 1, col),
            (row, col - 1),
            (row + 1, col),
            (row, col + 1),
        ] {
            if i == 0 || j == 0 || i > height || j > width {
                continue;
            }
            let p = i * stride + j;
            if state[p] != INSIDE {
                continue;
            }
            let distance = solve(p - stride, p - 1, &state, &time)
                .min(solve(p + stride, p - 1, &state, &time))
                .min(solve(p - stride, p + 1, &state, &time))
                .min(solve(p + stride, p + 1, &state, &time));
            time[p] = distance;
            let mut sum = [0f32; 3];
            let mut weights = [1e-20f32; 3];
            for k in i.saturating_sub(3).max(1)..=(i + 3).min(height) {
                let km = k - 1 + usize::from(k == 1);
                let kp = k - 1 - usize::from(k == height);
                for l in j.saturating_sub(3).max(1)..=(j + 3).min(width) {
                    let lm = l - 1 + usize::from(l == 1);
                    let lp = l - 1 - usize::from(l == width);
                    let q = k * stride + l;
                    let ry = k as f32 - i as f32;
                    let rx = l as f32 - j as f32;
                    let length = rx * rx + ry * ry;
                    if state[q] == INSIDE || length > 9. {
                        continue;
                    }
                    let dst = 1.0 / (length * length + 1.0);
                    for channel in 0..3 {
                        let sample = |y: usize, x: usize| i32::from(output[y * width + x][channel]);
                        let diff = |a: i32, b: i32| (a - b).abs() as f32;
                        let mut gx =
                            match (state[q + stride] != INSIDE, state[q - stride] != INSIDE) {
                                (true, true) => {
                                    diff(sample(kp + 1, lm), sample(kp, lm))
                                        + diff(sample(kp, lm), sample(km - 1, lm))
                                }
                                (true, false) => diff(sample(kp + 1, lm), sample(kp, lm)) * 2.0,
                                (false, true) => diff(sample(kp, lm), sample(km - 1, lm)) * 2.0,
                                _ => 0.0,
                            };
                        let gy = match (state[q + 1] != INSIDE, state[q - 1] != INSIDE) {
                            (true, true) => {
                                diff(sample(km, lp + 1), sample(km, lm))
                                    + diff(sample(km, lm), sample(km, lm - 1))
                            }
                            (true, false) => diff(sample(km, lp + 1), sample(km, lm)) * 2.0,
                            (false, true) => diff(sample(km, lm), sample(km, lm - 1)) * 2.0,
                            _ => 0.0,
                        };
                        gx = -gx;
                        let dot = rx * gx + ry * gy;
                        let dir = direction(dot, length * (gx * gx + gy * gy));
                        let weight = dst * dir;
                        sum[channel] +=
                            weight * f32::from(output[(k - 1) * width + l - 1][channel]);
                        weights[channel] += weight;
                    }
                }
            }
            output[(i - 1) * width + j - 1] = std::array::from_fn(|c| {
                (f64::from(sum[c]) / f64::from(weights[c]))
                    .round_ties_even()
                    .clamp(0., 255.) as u8
            });
            state[p] = 1;
            queue.push(Front {
                distance,
                order,
                row: i,
                col: j,
            });
            order += 1;
        }
    }
    checkpoint()?;
    Ok(output)
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
        name: String,
        width: usize,
        height: usize,
        pixels: Vec<[u8; 3]>,
        trust: Vec<u8>,
        expected: Vec<[u8; 3]>,
    }
    #[test]
    fn ns_direction_retains_cpp_double_sqrt_boundary() {
        // Compiled upstream expression: r=(1,2), gradient=(-1,15).
        // A float sqrt/divide incorrectly rounds to 1063049671 instead.
        assert_eq!(direction(29.0, 1130.0).to_bits(), 1063049670);
        assert_eq!(direction(0.0, 0.0), 0.000001);
    }

    #[test]
    fn ns_fill_matches_opencv_rgb() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hunyuan3d/ns-fill.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let mut calls = 0;
            let actual = fill_rgb(
                &case.pixels,
                &case.trust,
                case.width,
                case.height,
                &mut || {
                    calls += 1;
                    Ok(())
                },
            )
            .unwrap();
            for cancel_at in [1, calls / 2, calls] {
                let mut seen = 0;
                let error = fill_rgb(
                    &case.pixels,
                    &case.trust,
                    case.width,
                    case.height,
                    &mut || {
                        seen += 1;
                        anyhow::ensure!(seen != cancel_at, "cancelled");
                        Ok(())
                    },
                )
                .unwrap_err();
                assert_eq!(error.to_string(), "cancelled");
            }
            assert_eq!(actual, case.expected, "{}", case.name);
        }
    }
}
