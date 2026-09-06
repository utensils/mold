//! Run the CPU UV stage on a retained GLB and preserve its exact arrays.
#[cfg(feature = "mesh-texture")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use mold_inference::hunyuan3d::{glb::read_glb, uv::unwrap};
    use std::sync::atomic::AtomicBool;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: unwrap_mesh INPUT.glb NEW_RESULT.json"
    );
    let mesh = read_glb(&std::fs::read(&args[0])?)?;
    let started = std::time::Instant::now();
    let output = unwrap(&mesh, &AtomicBool::new(false))?;
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
