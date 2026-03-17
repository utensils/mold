{
  description = "mold — local AI image generation CLI for FLUX, SD1.5, SDXL & Z-Image diffusion models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devshell.flakeModule
        inputs.treefmt-nix.flakeModule
      ];

      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        {
          system,
          lib,
          ...
        }:
        let
          isLinux = system == "x86_64-linux";
          isDarwin = system == "aarch64-darwin";

          # CUDA compute capability — override for different GPU architectures.
          # Default "89" targets RTX 4090 (Ada Lovelace).
          # Common values: "75" (Turing), "80" (Ampere A100), "86" (Ampere 3090),
          # "89" (Ada 4090), "90" (Hopper H100).
          cudaComputeCap = "89";

          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
            config.allowUnfree = true;
          };

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rustfmt"
              "clippy"
            ];
          };

          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;

          src = craneLib.path ./.;

          commonArgs = {
            inherit src;
            pname = "mold";
            version = "0.1.0";
            strictDeps = true;
            cargoVendorDir = craneLib.vendorCargoDeps {
              inherit src;
            };
            nativeBuildInputs = [
              pkgs.pkg-config
            ]
            ++ lib.optionals isLinux [
              pkgs.cudaPackages.cuda_nvcc
            ];
            buildInputs = [
              pkgs.openssl
            ]
            ++ lib.optionals isDarwin [
              pkgs.libiconv
            ]
            ++ lib.optionals isLinux [
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas.lib
              pkgs.cudaPackages.cuda_nvtx.lib
              pkgs.cudaPackages.cuda_nvrtc.lib
              pkgs.cudaPackages.libcurand.lib
            ];
          }
          // lib.optionalAttrs isLinux {
            CUDA_PATH = "${pkgs.cudaPackages.cuda_nvcc}";
            CUDA_COMPUTE_CAP = cudaComputeCap;
            NIX_LDFLAGS = "-L${pkgs.cudaPackages.cuda_cudart}/lib/stubs";
          };

          cargoArtifacts = craneLib.buildDepsOnly commonArgs;

          gpuFeature =
            if isLinux then
              "cuda"
            else if isDarwin then
              "metal"
            else
              "";

          meta = with lib; {
            description = "Local AI image generation CLI for FLUX, SD1.5, SDXL & Z-Image diffusion models";
            homepage = "https://github.com/utensils/mold";
            license = licenses.mit;
            maintainers = [ ];
          };

          mold = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts meta;
              cargoExtraArgs = "-p mold-cli" + lib.optionalString (gpuFeature != "") " --features ${gpuFeature}";
            }
          );
        in
        {
          _module.args.pkgs = pkgs;

          packages = {
            inherit mold;
            default = mold;
          };

          apps.default = {
            type = "app";
            program = "${mold}/bin/mold";
            meta.description = meta.description;
          };

          devshells.default = {
            motd = ''
              {202}mold{reset} — local AI image generation for FLUX, SD1.5, SDXL & Z-Image ({bold}${system}{reset})
              $(type menu &>/dev/null && menu)
            '';

            packages = [
              rustToolchain
              pkgs.pkg-config
              pkgs.openssl
              pkgs.git
              pkgs.viu
            ]
            ++ lib.optionals isDarwin [
              pkgs.libiconv
            ]
            ++ lib.optionals isLinux [
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas.lib
              pkgs.cudaPackages.cuda_nvtx.lib
              pkgs.cudaPackages.cuda_nvrtc.lib
              pkgs.cudaPackages.libcurand.lib
            ];

            env = [
              {
                name = "RUST_BACKTRACE";
                value = "1";
              }
            ]
            ++ lib.optionals isDarwin [
              {
                name = "LIBRARY_PATH";
                value = lib.makeLibraryPath [
                  pkgs.libiconv
                  pkgs.openssl
                ];
              }
            ]
            ++ lib.optionals isLinux [
              {
                name = "CUDA_PATH";
                value = "${pkgs.cudaPackages.cuda_nvcc}";
              }
              {
                name = "CUDA_COMPUTE_CAP";
                value = "89";
              }
              {
                name = "CPATH";
                value = "${pkgs.cudaPackages.cuda_cudart}/include:${pkgs.cudaPackages.cuda_cccl}/include";
              }
              {
                name = "LIBRARY_PATH";
                value =
                  lib.makeLibraryPath [
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.libcublas.lib
                    pkgs.cudaPackages.cuda_nvrtc.lib
                    pkgs.cudaPackages.libcurand.lib
                  ]
                  + ":${pkgs.cudaPackages.cuda_cudart}/lib/stubs"
                  + ":/run/opengl-driver/lib";
              }
              {
                name = "LD_LIBRARY_PATH";
                value =
                  "/run/opengl-driver/lib:"
                  + lib.makeLibraryPath [
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.libcublas.lib
                    pkgs.cudaPackages.cuda_nvrtc.lib
                    pkgs.cudaPackages.libcurand.lib
                  ];
              }
            ];

            commands = [
              {
                category = "build";
                name = "build";
                help = "cargo build (debug, all crates)";
                command = "cargo build \"$@\"";
              }
              {
                category = "build";
                name = "build-release";
                help = "cargo build --release";
                command = "cargo build --release \"$@\"";
              }
              {
                category = "build";
                name = "build-server";
                help = "cargo build -p mold-cli --features ${gpuFeature} (single binary with GPU)";
                command = "cargo build -p mold-cli --features ${gpuFeature} \"$@\"";
              }
              {
                category = "check";
                name = "check";
                help = "cargo check";
                command = "cargo check \"$@\"";
              }
              {
                category = "check";
                name = "clippy";
                help = "cargo clippy";
                command = "cargo clippy \"$@\"";
              }
              {
                category = "check";
                name = "run-tests";
                help = "cargo test";
                command = "cargo test \"$@\"";
              }
              {
                category = "check";
                name = "fmt";
                help = "cargo fmt";
                command = "cargo fmt \"$@\"";
              }
              {
                category = "check";
                name = "fmt-check";
                help = "cargo fmt --check";
                command = "cargo fmt --check \"$@\"";
              }
              {
                category = "run";
                name = "mold";
                help = "run mold CLI (e.g. mold list, mold ps, mold pull)";
                command = "cargo run -p mold-cli --features ${gpuFeature} -- \"$@\"";
              }
              {
                category = "run";
                name = "serve";
                help = "start the mold server";
                command = "cargo run -p mold-cli --features ${gpuFeature} -- serve \"$@\"";
              }
              {
                category = "run";
                name = "generate";
                help = "generate an image from a prompt";
                command = "cargo run -p mold-cli --features ${gpuFeature} -- run \"$@\"";
              }
            ];
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
            programs.rustfmt.enable = true;
          };
        };
    };
}
