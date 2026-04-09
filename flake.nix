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
          # "89" (Ada 4090), "90" (Hopper H100), "120" (Blackwell RTX 5090).
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
            version = "0.5.0";
            strictDeps = true;

            # Pass git metadata so build.rs can embed it (no .git in Nix sandbox).
            MOLD_GIT_SHA = gitShortRev;
            MOLD_BUILD_DATE = gitDate;
            cargoVendorDir = craneLib.vendorCargoDeps {
              inherit src;
            };
            nativeBuildInputs = [
              pkgs.pkg-config
              pkgs.nasm
              pkgs.clang
              pkgs.llvmPackages.libclang.lib
            ]
            ++ lib.optionals isLinux [
              pkgs.cudaPackages.cuda_nvcc
            ];
            buildInputs = [
              pkgs.openssl
              pkgs.libwebp
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

          opensslPkgConfigPath = "${pkgs.openssl.dev}/lib/pkgconfig";
          opensslLibDir = "${pkgs.lib.getLib pkgs.openssl}/lib";
          opensslIncludeDir = "${pkgs.openssl.dev}/include";

          cargoArtifacts = craneLib.buildDepsOnly commonArgs;

          gpuFeature =
            if isLinux then
              "cuda"
            else if isDarwin then
              "metal"
            else
              "";

          # Features string for devshell commands: GPU + preview + discord + expand + tui + video formats
          devFeatures =
            if gpuFeature != "" then
              "${gpuFeature},preview,discord,expand,tui,webp,mp4"
            else
              "preview,discord,expand,tui,webp,mp4";

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

          # Build a mold package for a given CUDA compute capability.
          mkMold =
            computeCap:
            craneLib.buildPackage (
              commonArgs
              // {
                inherit cargoArtifacts meta;
                cargoExtraArgs =
                  "-p mold-ai --features preview,discord,expand,tui,webp,mp4,metrics"
                  + lib.optionalString (gpuFeature != "") ",${gpuFeature}";
                postInstall = ''
                  installShellCompletion --cmd mold \
                    --bash <($out/bin/mold completions bash) \
                    --zsh <($out/bin/mold completions zsh) \
                    --fish <($out/bin/mold completions fish)
                '';
                nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.installShellFiles ];
              }
              // lib.optionalAttrs isLinux {
                CUDA_COMPUTE_CAP = computeCap;
              }
            );

          mold = mkMold cudaComputeCap;

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
          }
          // lib.optionalAttrs isLinux {
            mold-sm120 = mkMold "120"; # Blackwell (RTX 50-series)
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
              pkgs.nasm
              pkgs.git
              pkgs.gh
              pkgs.jq
              pkgs.python3
              pkgs.uv
              pkgs.viu
              pkgs.cargo-llvm-cov
              pkgs.ffmpeg
              pkgs.imagemagick
              pkgs.bun
              pkgs.nodePackages.prettier
              pkgs.tmux
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
              {
                name = "MOLD_LTX_DEBUG";
                value = "1";
              }
              {
                name = "PKG_CONFIG_PATH";
                value = opensslPkgConfigPath;
              }
              {
                name = "OPENSSL_DIR";
                value = "${pkgs.openssl.dev}";
              }
              {
                name = "OPENSSL_LIB_DIR";
                value = opensslLibDir;
              }
              {
                name = "OPENSSL_INCLUDE_DIR";
                value = opensslIncludeDir;
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
                value = cudaComputeCap;
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
                help = "cargo build --release -p mold-ai --features ${devFeatures}";
                command = "cargo build --release -p mold-ai --features ${devFeatures} \"$@\"";
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
                category = "build";
                name = "build-candle-wuerstchen";
                help = "build the official Candle Wuerstchen example in the devshell";
                command = ''
                  set -euo pipefail
                  repo_dir="''${CANDLE_UPSTREAM_DIR:-$PWD/.cache/candle-upstream}"
                  if [ ! -d "$repo_dir/.git" ]; then
                    mkdir -p "$(dirname "$repo_dir")"
                    git clone https://github.com/huggingface/candle "$repo_dir"
                  fi
                  git -C "$repo_dir" fetch --tags origin
                  git -C "$repo_dir" checkout main
                  git -C "$repo_dir" pull --ff-only
                  cd "$repo_dir/candle-examples"
                  cargo build --example wuerstchen --features ${gpuFeature}
                '';
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
                name = "test-ltx2";
                help = "targeted LTX-2 / LTX-2.3 tests";
                command = "cargo test \"$@\" ltx2";
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
              {
                category = "run";
                name = "build-ltx2";
                help = "build mold with the full feature set for LTX-2 work";
                command = "cargo build -p mold-ai --features ${devFeatures} \"$@\"";
              }
              {
                category = "run";
                name = "smoke-ltx2";
                help = "run a local LTX-2 / LTX-2.3 smoke inference";
                command = "cargo run -p mold-ai --features ${devFeatures} -- run --local \"$@\"";
              }
              {
                category = "run";
                name = "contact-sheet";
                help = "build a contact sheet from a clip via ffmpeg";
                command = ''
                  set -euo pipefail
                  if [ "$#" -lt 2 ]; then
                    echo "usage: contact-sheet <input-video-or-gif> <output-png> [tile]"
                    exit 1
                  fi
                  input="$1"
                  output="$2"
                  tile="''${3:-4x5}"
                  cols="''${tile%x*}"
                  rows="''${tile#*x}"
                  if [ -z "$cols" ] || [ "$cols" = "$tile" ]; then
                    echo "tile must be in CxR format, for example 4x5"
                    exit 1
                  fi
                  if [ -z "$rows" ] || [ "$rows" = "$tile" ]; then
                    rows=5
                  fi
                  ffmpeg -y -v error -i "$input" -vf "tile=''${cols}x''${rows}" -frames:v 1 "$output"
                '';
              }
              {
                category = "run";
                name = "issue-note";
                help = "post a progress update to GitHub issue #187";
                command = ''
                  set -euo pipefail
                  if [ "$#" -lt 1 ]; then
                    echo "usage: issue-note <message>"
                    exit 1
                  fi
                  gh issue comment 187 --repo utensils/mold --body "$*"
                '';
              }
              {
                category = "docs";
                name = "docs-dev";
                help = "start VitePress dev server for docs";
                command = "cd website && bun install && bun run dev \"$@\"";
              }
              {
                category = "docs";
                name = "docs-build";
                help = "build the documentation site";
                command = "cd website && bun install && bun run build";
              }
              {
                category = "docs";
                name = "docs-preview";
                help = "preview the built documentation site";
                command = "cd website && bun run preview \"$@\"";
              }
              {
                category = "docs";
                name = "docs-fmt";
                help = "format documentation with prettier";
                command = "cd website && bun run fmt";
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
