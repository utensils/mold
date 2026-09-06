fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;

        println!("cargo::rerun-if-changed=src/comfy_int8/cuda/int8_linear.cu");
        let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
            .join("comfy_int8_cuda.rs");
        let bindings = cudaforge::KernelBuilder::new()
            .source_files(vec!["src/comfy_int8/cuda/int8_linear.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .build_ptx()
            .expect("compile Comfy INT8 CUDA kernels");
        bindings
            .write(output)
            .expect("write Comfy INT8 PTX bindings");

        println!("cargo::rerun-if-changed=src/stable_diffusion/vae/group_norm.cu");
        cudaforge::KernelBuilder::new()
            .source_files(vec!["src/stable_diffusion/vae/group_norm.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .build_ptx()
            .expect("compile paint VAE normalization")
            .write(
                PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
                    .join("vae_precision_cuda.rs"),
            )
            .expect("write paint VAE normalization PTX bindings");
    }
}
