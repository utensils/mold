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
    let
      # Git metadata for build info — available from the flake's self reference.
      gitShortRev = inputs.self.shortRev or inputs.self.dirtyShortRev or "unknown";
      gitDate =
        let
          raw = toString (inputs.self.lastModifiedDate or "unknown");
        in
        if builtins.stringLength raw >= 8 then
          "${builtins.substring 0 4 raw}-${builtins.substring 4 2 raw}-${builtins.substring 6 2 raw}"
        else
          "unknown";
    in
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devshell.flakeModule
        inputs.treefmt-nix.flakeModule
      ];

      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      flake.nixosModules = {
        default = ./nix/module.nix;
        mold = ./nix/module.nix;
      };

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
            localSystem = system;
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
            version = "0.2.0";
            strictDeps = true;

            # Pass git metadata so build.rs can embed it (no .git in Nix sandbox).
            MOLD_GIT_SHA = gitShortRev;
            MOLD_BUILD_DATE = gitDate;
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
            CUDA_PATH = "${cudaToolkit}";
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

          # Features string for devshell commands: GPU + preview + discord + expand
          devFeatures =
            if gpuFeature != "" then "${gpuFeature},preview,discord,expand" else "preview,discord,expand";

          # Merged CUDA toolkit so bindgen_cuda can find both bin/nvcc and include/cuda.h
          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit-merged";
            paths = [
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cuda_cudart
            ];
          };

          meta = with lib; {
            description = "Local AI image generation CLI for FLUX, SD1.5, SDXL & Z-Image diffusion models";
            homepage = "https://github.com/utensils/mold";
            license = licenses.mit;
            mainProgram = "mold";
            maintainers = [ ];
          };

          mold = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts meta;
              cargoExtraArgs =
                "-p mold-ai --features preview,discord,expand"
                + lib.optionalString (gpuFeature != "") ",${gpuFeature}";
              postInstall = ''
                installShellCompletion --cmd mold \
                  --bash <($out/bin/mold completions bash) \
                  --zsh <($out/bin/mold completions zsh) \
                  --fish <($out/bin/mold completions fish)
              '';
              nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.installShellFiles ];
            }
          );

          moldDiscord = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              pname = "mold-discord";
              cargoExtraArgs = "-p mold-ai-discord";
              meta = with lib; {
                description = "Discord bot for mold — AI image generation via slash commands";
                homepage = "https://github.com/utensils/mold";
                license = licenses.mit;
                mainProgram = "mold-discord";
                maintainers = [ ];
              };
            }
          );
        in
        {
          _module.args.pkgs = pkgs;

          packages = {
            inherit mold;
            mold-discord = moldDiscord;
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
              pkgs.cargo-llvm-cov
            ]
            ++ lib.optionals isDarwin [
              pkgs.libiconv
              pkgs.llvmPackages.libcxxClang
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
                  pkgs.llvmPackages.libcxx
                ];
              }
            ]
            ++ lib.optionals isLinux [
              {
                name = "CUDA_PATH";
                value = "${cudaToolkit}";
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
                  # /run/opengl-driver/lib MUST come before cuda_cudart/lib/stubs
                  # so the real libcuda.so (NVIDIA driver) is found before the
                  # stub placeholder. Without this, debug builds link against
                  # the stub and fail at runtime with CUDA_ERROR_STUB_LIBRARY.
                  "/run/opengl-driver/lib:"
                  + lib.makeLibraryPath [
                    pkgs.cudaPackages.cuda_cudart
                    pkgs.cudaPackages.libcublas.lib
                    pkgs.cudaPackages.cuda_nvrtc.lib
                    pkgs.cudaPackages.libcurand.lib
                  ]
                  + ":${pkgs.cudaPackages.cuda_cudart}/lib/stubs";
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
                help = "cargo build -p mold-ai --features ${devFeatures} (single binary with GPU + preview)";
                command = "cargo build -p mold-ai --features ${devFeatures} \"$@\"";
              }
              {
                category = "build";
                name = "build-discord";
                help = "cargo build -p mold-ai --features discord";
                command = "cargo build -p mold-ai --features ${devFeatures} \"$@\"";
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
                category = "check";
                name = "coverage";
                help = "test coverage report (--html for browsable report)";
                command = ''
                  LLVM_COV="$(find /nix/store -maxdepth 3 -name llvm-cov 2>/dev/null | head -1)"
                  LLVM_PROFDATA="$(find /nix/store -maxdepth 3 -name llvm-profdata 2>/dev/null | head -1)"
                  export LLVM_COV LLVM_PROFDATA
                  if [ "''${1:-}" = "--html" ]; then
                    cargo llvm-cov --workspace --html --no-cfg-coverage --output-dir target/coverage
                    echo "Report: target/coverage/html/index.html"
                  else
                    cargo llvm-cov --workspace --no-cfg-coverage --skip-functions
                  fi
                '';
              }
              {
                category = "run";
                name = "mold";
                help = "run mold CLI (e.g. mold list, mold ps, mold pull)";
                command = "cargo run -p mold-ai --features ${devFeatures} -- \"$@\"";
              }
              {
                category = "run";
                name = "serve";
                help = "start the mold server";
                command = "cargo run -p mold-ai --features ${devFeatures} -- serve \"$@\"";
              }
              {
                category = "run";
                name = "generate";
                help = "generate an image from a prompt";
                command = "cargo run -p mold-ai --features ${devFeatures} -- run \"$@\"";
              }
              {
                category = "run";
                name = "discord-bot";
                help = "start the mold Discord bot";
                command = "cargo run -p mold-ai --features ${devFeatures} -- discord \"$@\"";
              }
            ];
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
            programs.rustfmt = {
              enable = true;
              edition = "2021";
            };
          };
        };
    };
}
