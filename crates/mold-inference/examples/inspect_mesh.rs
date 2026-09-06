//! Render a stored mesh with mold's own poster camera for retained qualification.
use anyhow::{ensure, Context, Result};
use mold_inference::hunyuan3d::{glb::read_glb, obj::read_obj, poster::render_poster};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: inspect_mesh INPUT.glb|INPUT.obj NEW_POSTER.png"
    );
    let bytes = std::fs::read(&args[0])?;
    let mesh = if std::path::Path::new(&args[0])
        .extension()
        .is_some_and(|ext| ext == "obj")
    {
        read_obj(std::str::from_utf8(&bytes)?).context("read OBJ mesh")?
    } else {
        read_glb(&bytes).context("read GLB mesh")?
    };
    println!(
        "vertices={} triangles={}",
        mesh.vertices.len(),
        mesh.faces.len()
    );
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
