{
  description = "mold — like ollama, but for diffusion models";

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
            ];
            buildInputs =
              [
                pkgs.openssl
              ]
              ++ lib.optionals isDarwin [
                pkgs.libiconv
              ]
              ++ lib.optionals isLinux [
                pkgs.cudaPackages.cuda_nvcc
                pkgs.cudaPackages.cuda_cudart
                pkgs.cudaPackages.libcublas
                pkgs.cudaPackages.cuda_nvtx
                pkgs.cudaPackages.cuda_nvrtc
                pkgs.cudaPackages.libcurand
              ];
          };

          cargoArtifacts = craneLib.buildDepsOnly commonArgs;

          gpuFeature = if isLinux then "cuda" else if isDarwin then "metal" else "";

          meta = with lib; {
            description = "Like ollama, but for diffusion models";
            homepage = "https://github.com/utensils/mold";
            license = licenses.mit;
            maintainers = [ ];
          };

          mold = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts meta;
              cargoExtraArgs = "-p mold-cli";
            }
          );

          mold-server = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts meta;
              cargoExtraArgs = "-p mold-server --features ${gpuFeature}";
            }
          );
        in
        {
          _module.args.pkgs = pkgs;

          packages = {
            inherit mold mold-server;
            default = mold;
          };

          apps = {
            default = {
              type = "app";
              program = "${mold}/bin/mold";
            };
            mold-server = {
              type = "app";
              program = "${mold-server}/bin/mold-server";
            };
          };

          devshells.default = {
            motd = ''
              {202}mold{reset} — like ollama, but for diffusion models ({bold}${system}{reset})
              $(type menu &>/dev/null && menu)
            '';

            packages =
              [
                rustToolchain
                pkgs.pkg-config
                pkgs.openssl
                pkgs.git
              ]
              ++ lib.optionals isDarwin [
                pkgs.libiconv
              ]
              ++ lib.optionals isLinux [
                pkgs.cudaPackages.cuda_nvcc
                pkgs.cudaPackages.cuda_cudart
                pkgs.cudaPackages.libcublas
                pkgs.cudaPackages.cuda_nvtx
                pkgs.cudaPackages.cuda_nvrtc
                pkgs.cudaPackages.libcurand
              ];

            env = [
              {
                name = "RUST_BACKTRACE";
                value = "1";
              }
            ] ++ lib.optionals isDarwin [
              {
                name = "LIBRARY_PATH";
                value = lib.makeLibraryPath [
                  pkgs.libiconv
                  pkgs.openssl
                ];
              }
            ] ++ lib.optionals isLinux [
              {
                name = "CUDA_PATH";
                value = "${pkgs.cudaPackages.cuda_cudart}";
              }
              {
                name = "CUDA_COMPUTE_CAP";
                value = "89";
              }
              {
                name = "LD_LIBRARY_PATH";
                value = lib.makeLibraryPath [
                  pkgs.cudaPackages.cuda_cudart
                  pkgs.cudaPackages.libcublas
                  pkgs.cudaPackages.cuda_nvrtc
                  pkgs.cudaPackages.libcurand
                ] + ":/run/opengl-driver/lib";
              }
            ];

            commands = [
              {
                category = "build";
                name = "build";
                help = "cargo build (debug, all crates)";
                command = "cargo build $@";
              }
              {
                category = "build";
                name = "build-release";
                help = "cargo build --release";
                command = "cargo build --release $@";
              }
              {
                category = "build";
                name = "build-server";
                help = "cargo build -p mold-server --features ${gpuFeature}";
                command = "cargo build -p mold-server --features ${gpuFeature} $@";
              }
              {
                category = "check";
                name = "check";
                help = "cargo check";
                command = "cargo check $@";
              }
              {
                category = "check";
                name = "clippy";
                help = "cargo clippy";
                command = "cargo clippy $@";
              }
              {
                category = "check";
                name = "run-tests";
                help = "cargo test";
                command = "cargo test $@";
              }
              {
                category = "check";
                name = "fmt";
                help = "cargo fmt";
                command = "cargo fmt $@";
              }
              {
                category = "check";
                name = "fmt-check";
                help = "cargo fmt --check";
                command = "cargo fmt --check $@";
              }
              {
                category = "run";
                name = "mold";
                help = "run mold CLI (e.g. mold list, mold ps, mold pull)";
                command = "cargo run -p mold-cli -- $@";
              }
              {
                category = "run";
                name = "serve";
                help = "start the mold server";
                command = "cargo run -p mold-cli -- serve $@";
              }
              {
                category = "run";
                name = "generate";
                help = "generate an image from a prompt";
                command = "cargo run -p mold-cli -- generate $@";
              }
              {
                category = "run";
                name = "tui";
                help = "interactive TUI session";
                command = "cargo run -p mold-cli -- run $@";
              }
              {
                category = "deploy";
                name = "deploy";
                help = "deploy to hal9000";
                command = "./scripts/deploy.sh";
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
