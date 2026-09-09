//! Run the CPU UV stage on a retained GLB and preserve its exact arrays.
//!
//! An optional third argument decimates the mesh first, so one invocation
//! measures the two CPU stages a textured run pays for — `mesh::simplify`
//! and `uv::unwrap` — against the same input.
#[cfg(feature = "mesh-texture")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use mold_inference::hunyuan3d::{glb::read_glb, mesh};
    use std::sync::atomic::AtomicBool;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: unwrap_mesh INPUT.glb NEW_RESULT.json [TARGET_FACES]"
    );
    let target_faces = match args.get(2) {
        Some(value) => Some(
            value
                .to_str()
                .and_then(|text| text.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("TARGET_FACES must be a positive integer"))?,
        ),
        None => None,
    };
    let mut mesh = read_glb(&std::fs::read(&args[0])?)?;
    let input_faces = mesh.faces.len();
    let mut simplify_seconds = 0.0;
    if let Some(target) = target_faces {
        let started = std::time::Instant::now();
        mesh = mesh::simplify(&mesh, target)?;
        simplify_seconds = started.elapsed().as_secs_f64();
    }
    let simplified_faces = mesh.faces.len();
    let started = std::time::Instant::now();
    // `MOLD_UNWRAP_CANCEL_AFTER_SECS` cancels mid-unwrap, which is how the
    // native cancellation path is exercised on a real mesh: the interesting
    // phases are the ones that run for minutes inside one native call.
    let deadline = std::env::var("MOLD_UNWRAP_CANCEL_AFTER_SECS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .map(std::time::Duration::from_secs_f64);
    let cancelled = AtomicBool::new(false);
    let output =
        mold_inference::hunyuan3d::uv::unwrap_reporting(&mesh, &cancelled, &|phase, pct| {
            if let Some(deadline) = deadline {
                if started.elapsed() >= deadline {
                    eprintln!("cancelling during {phase:?} at {pct}%");
                    cancelled.store(true, std::sync::atomic::Ordering::Release);
                }
            }
        });
    if deadline.is_some() {
        println!(
            "cancel requested; unwrap returned after {:.2}s: {}",
            started.elapsed().as_secs_f64(),
            output.as_ref().err().map_or("Ok".into(), |e| e.to_string())
        );
        output?;
        return Ok(());
    }
    let output = output?;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(
        &mut writer,
        &serde_json::json!({
            "vertices": output.vertices, "faces": output.faces, "uv": output.uvs,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "input_faces": input_faces,
            "simplified_faces": simplified_faces,
            "simplify_seconds": simplify_seconds,
        }),
    )?;
    use std::io::Write;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(not(feature = "mesh-texture"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("unwrap_mesh requires --features mesh-texture")
}
