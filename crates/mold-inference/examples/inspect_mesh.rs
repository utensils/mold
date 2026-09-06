//! Render a stored mesh with mold's own poster camera for retained qualification.
use anyhow::{ensure, Context, Result};
use mold_inference::hunyuan3d::{glb::read_glb, poster::render_poster};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: inspect_mesh INPUT.glb NEW_POSTER.png"
    );
    let mesh = read_glb(&std::fs::read(&args[0])?).context("read mesh")?;
    let png = render_poster(&mesh, 768)?;
    use std::io::Write;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    output.write_all(&png)?;
    output.sync_all()?;
    Ok(())
}
