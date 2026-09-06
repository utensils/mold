//! Preserve the CPU paint conditions for comparison with retained Tencent maps.
use anyhow::{ensure, Result};
use mold_inference::hunyuan3d::{glb::read_glb, paint_raster, paint_views::candidate_views};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: paint_conditions INPUT.glb NEW_DIRECTORY"
    );
    let directory = std::path::Path::new(&args[1]);
    std::fs::create_dir(directory)?;
    let mesh = paint_raster::prepare_mesh(&read_glb(&std::fs::read(&args[0])?)?)?;
    let started = std::time::Instant::now();
    for (index, view) in candidate_views().into_iter().take(6).enumerate() {
        let buffers = paint_raster::render(&mesh, view.elevation, view.azimuth, 2048)?;
        for (position, output_index) in [(false, index), (true, index + 6)] {
            let mut pixels = vec![255; 2048 * 2048 * 3];
            for (pixel, rgb) in pixels.chunks_exact_mut(3).enumerate() {
                if !buffers.mask[pixel] {
                    continue;
                }
                for (axis, channel) in rgb.iter_mut().enumerate() {
                    let value = if position {
                        0.5 - buffers.position[pixel][axis] / 1.15
                    } else {
                        (buffers.normal[pixel][axis] + 1.) * 0.5
                    };
                    *channel = (value * 255.) as u8;
                }
            }
            image::save_buffer(
                directory.join(format!("condition-{output_index:02}.png")),
                &pixels,
                2048,
                2048,
                image::ColorType::Rgb8,
            )?;
        }
        println!("view={index} covered={}", buffers.covered_pixels());
    }
    println!("elapsed_seconds={}", started.elapsed().as_secs_f64());
    Ok(())
}
