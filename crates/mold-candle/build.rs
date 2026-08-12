fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;

        println!("cargo::rerun-if-changed=src/minimax_h3/cuda/int8_linear.cu");
        let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
            .join("h3_int8_cuda.rs");
        let bindings = cudaforge::KernelBuilder::new()
            .source_files(vec!["src/minimax_h3/cuda/int8_linear.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .build_ptx()
            .expect("compile MiniMax H3 INT8 CUDA kernels");
        bindings
            .write(output)
            .expect("write MiniMax H3 INT8 PTX bindings");
    }
}
