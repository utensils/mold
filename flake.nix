{
  description = "mold - like ollama, but for diffusion models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        isLinux = builtins.elem system [ "x86_64-linux" "aarch64-linux" ];
        isDarwin = builtins.elem system [ "x86_64-darwin" "aarch64-darwin" ];
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
          config.cudaSupport = isLinux;
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rustfmt" "clippy" ];
        };

        commonInputs = [
          rustToolchain
          pkgs.pkg-config
          pkgs.openssl
          pkgs.git
        ];

        linuxInputs = pkgs.lib.optionals isLinux [
          pkgs.cudaPackages.cuda_nvcc
          pkgs.cudaPackages.cuda_cudart
          pkgs.cudaPackages.libcublas
          pkgs.cudaPackages.cuda_nvtx
        ];

        darwinInputs = pkgs.lib.optionals isDarwin [
          pkgs.libiconv
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = commonInputs ++ linuxInputs ++ darwinInputs;

          env = pkgs.lib.optionalAttrs isLinux {
            CUDA_PATH = "${pkgs.cudaPackages.cuda_cudart}";
            CUDA_COMPUTE_CAP = "89"; # RTX 4090 = sm_89
          };

          LD_LIBRARY_PATH = pkgs.lib.optionalString isLinux (
            (pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas
            ]) + ":/run/opengl-driver/lib"
          );

          shellHook = ''
            echo "mold dev shell (${system})"
          '' + pkgs.lib.optionalString isLinux ''
            echo "  CUDA enabled - build with: cargo build --release -p mold-server --features cuda"
          '' + pkgs.lib.optionalString isDarwin ''
            echo "  macOS - build with: cargo build --release -p mold-server --features metal"
          '';
        };
      }
    );
}
