fn main() {
    // generate_context! embeds the frontend at Rust compile time. Make Cargo
    // rebuild the crate whenever Vite refreshes the packaged mobile assets.
    println!("cargo:rerun-if-changed=../../../desktop/dist-mobile");
    tauri_build::build();
}
