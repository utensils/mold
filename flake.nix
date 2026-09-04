{
  description = "mold — local AI image generation CLI for FLUX, SD1.5, SDXL & Z-Image diffusion models";

  nixConfig = {
    extra-substituters = [ "https://mold.cachix.org" ];
    extra-trusted-public-keys = [
      "mold.cachix.org-1:9HBc/bEXDdpbxMjOwpaIDpjZqBh9JYg0h5Fipm+D8m4="
    ];
  };

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
    bun2nix = {
      url = "github:nix-community/bun2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    let
      # Git metadata for build info — available from the flake's self reference.
      gitRev = inputs.self.rev or "unknown";
      gitDate =
        let
          raw = toString (inputs.self.lastModifiedDate or "unknown");
        in
        if builtins.stringLength raw >= 8 then
          "${builtins.substring 0 4 raw}-${builtins.substring 4 2 raw}-${builtins.substring 6 2 raw}"
        else
          "unknown";
      workspaceVersion = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
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
          config,
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
            overlays = [
              inputs.rust-overlay.overlays.default
              inputs.bun2nix.overlays.default
            ];
            config.allowUnfree = true;
          };

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rustfmt"
              "clippy"
            ];
            targets =
              (lib.optionals isDarwin [
                "aarch64-apple-ios"
                "aarch64-apple-ios-sim"
              ])
              ++ [
                "aarch64-linux-android"
                "armv7-linux-androideabi"
                "i686-linux-android"
                "x86_64-linux-android"
              ];
          };

          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;

          src = craneLib.path ./.;

          h3CutlassCommit = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
          h3CutlassSource = pkgs.fetchFromGitHub {
            owner = "NVIDIA";
            repo = "cutlass";
            rev = h3CutlassCommit;
            hash = "sha256-D/s7eYsa5l/mfx73tE4mnFcTQdYqGmXa9d9TCryw4e4=";
          };
          prepareCudaforgeCache = ''
            export CUDAFORGE_HOME="$TMPDIR/cudaforge"
            cutlass_checkout="$CUDAFORGE_HOME/git/checkouts/cutlass-${builtins.substring 0 16 h3CutlassCommit}"
            mkdir -p "$cutlass_checkout"
            cp -R ${h3CutlassSource}/. "$cutlass_checkout/"
            chmod -R u+w "$CUDAFORGE_HOME"
            git -C "$cutlass_checkout" init --quiet
            base64 --decode ${./nix/cutlass-7d49e6c.commit.b64} > "$CUDAFORGE_HOME/cutlass.commit"
            cutlass_commit="$(git -C "$cutlass_checkout" hash-object -t commit -w "$CUDAFORGE_HOME/cutlass.commit")"
            test "$cutlass_commit" = ${h3CutlassCommit}
            git -C "$cutlass_checkout" update-ref refs/heads/cudaforge "$cutlass_commit"
            git -C "$cutlass_checkout" symbolic-ref HEAD refs/heads/cudaforge
          '';

          commonArgs = {
            inherit src;
            pname = "mold";
            version = workspaceVersion;
            strictDeps = true;

            # Pass git metadata so build.rs can embed it (no .git in Nix sandbox).
            MOLD_GIT_SHA = gitRev;
            MOLD_BUILD_DATE = gitDate;
            cargoVendorDir = craneLib.vendorCargoDeps {
              inherit src;
            };
            preBuild = lib.optionalString isLinux prepareCudaforgeCache;
            nativeBuildInputs = [
              pkgs.pkg-config
              pkgs.nasm
              pkgs.clang
              pkgs.llvmPackages.libclang.lib
              # `candle-onnx`'s build script drives `prost-build`, which shells
              # out to `protoc`. The `pulid` feature pulls that crate in and is
              # now in every release recipe, so this is a build requirement
              # rather than a devshell convenience.
              pkgs.protobuf
            ]
            ++ lib.optionals isLinux [
              pkgs.gitMinimal
              pkgs.lld
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
              pkgs.stdenv.cc.cc.lib
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas.lib
              pkgs.cudaPackages.cuda_nvtx.lib
              pkgs.cudaPackages.cuda_nvrtc.lib
              pkgs.cudaPackages.libcurand.lib
              pkgs.cudaPackages.cudnn.lib
            ];
          }
          // lib.optionalAttrs isLinux {
            CUDA_PATH = "${cudaToolkit}";
            CUDA_COMPUTE_CAP = cudaComputeCap;
            NIX_LDFLAGS = "-L${pkgs.cudaPackages.cuda_cudart}/lib/stubs -L${pkgs.cudaPackages.cudnn.lib}/lib";
          };

          opensslLibDir = "${pkgs.lib.getLib pkgs.openssl}/lib";
          opensslIncludeDir = "${pkgs.openssl.dev}/include";

          desktopLinuxBuildInputs = with pkgs; [
            dbus
            gst_all_1.gst-libav
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-ugly
            gst_all_1.gstreamer
            gtk3
            libayatana-appindicator
            librsvg
            libsoup_3
            xdotool
            webkitgtk_4_1
            zlib
          ];

          desktopPkgConfigInputs = lib.closePropagation (
            [ pkgs.openssl ] ++ lib.optionals isLinux desktopLinuxBuildInputs
          );

          desktopLinuxRuntimeInputs = lib.closePropagation (
            desktopLinuxBuildInputs
            ++ (with pkgs; [
              atk
              cairo
              gdk-pixbuf
              glib
              pango
            ])
          );

          desktopPkgConfigPath = lib.concatStringsSep ":" [
            (lib.makeSearchPath "lib/pkgconfig" (map lib.getDev desktopPkgConfigInputs))
            (lib.makeSearchPath "share/pkgconfig" (map lib.getDev desktopPkgConfigInputs))
          ];

          # Every shared library a `releaseFeatures` binary ends up NEEDED
          # against. `cudnn` is in that set (#1483), so `cudnn.lib` belongs here
          # beside the rest rather than being left to chance.
          #
          # This list is about the LOAD, not the link. cudarc's build script
          # emits `-L $CUDA_PATH/lib` and the merged `cudaToolkit` carries
          # `libcudnn.so`, so a devshell `cudnn` build links clean either way —
          # the issue's reported `-lcudnn` link error does not reproduce. It
          # then dies in the dynamic loader, because the binary carries no
          # RUNPATH at all: nixpkgs' `ld` wrapper is what turns a store `-L`
          # into `-rpath`, and `.cargo/config.toml` links this target with
          # `-fuse-ld=lld`, which goes around that wrapper. LD_LIBRARY_PATH is
          # then the only thing left that can resolve `libcudnn.so.9`.
          #
          # `nix build` is immune for a reason that does NOT apply here:
          # `mkMold` runs `autoPatchelfHook` + `autoAddDriverRunpath`, which
          # rewrite RUNPATH from `buildInputs` — and those carry `cudnn.lib`.
          # (Not `NIX_LDFLAGS`: it also names `cuda_cudart/lib/stubs`, which
          # appears in no built `mold`'s RUNPATH.) The devshell gets no such
          # pass, so omitting cuDNN here made the one feature combination that
          # matches a release artifact the one a developer could build but not
          # run (#1510). `devshell-cuda-load-path` in `checks` is the guard.
          devshellLinuxCudaLibs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.libcublas.lib
            pkgs.cudaPackages.cuda_nvrtc.lib
            pkgs.cudaPackages.libcurand.lib
            pkgs.cudaPackages.cudnn.lib
          ];

          # /run/opengl-driver/lib MUST come before cuda_cudart/lib/stubs so the
          # real libcuda.so (NVIDIA driver) is found before the stub
          # placeholder. Without this, debug builds link against the stub and
          # fail at runtime with CUDA_ERROR_STUB_LIBRARY.
          devshellLinuxLibraryPath =
            "/run/opengl-driver/lib:"
            + lib.makeLibraryPath (desktopLinuxRuntimeInputs ++ devshellLinuxCudaLibs)
            + ":${pkgs.cudaPackages.cuda_cudart}/lib/stubs";

          devshellLinuxLdLibraryPath =
            "/run/opengl-driver/lib:"
            + lib.makeLibraryPath (desktopLinuxRuntimeInputs ++ devshellLinuxCudaLibs);

          # SM89 names `h3-cuda`, never `cuda,h3` -- since #1164 the bare `h3`
          # feature implies neither CUDA nor the SM89 attention kernel, and
          # `h3-cuda` implies `cuda` so it replaces the device feature.
          desktopFeatureFor =
            computeCap: if isLinux then if computeCap == "89" then "h3-cuda" else "cuda" else "metal,h3";
          # The desktop app's complete feature recipe. `pulid` rides every
          # desktop build for the same reason it rides every `mold` release
          # recipe (#1223): the embedded This-device server advertises
          # `supports_identity` only when the feature is compiled, and the
          # identity photo well is gated on that advertisement, so a desktop
          # build without it hides face identity permanently.
          desktopFeaturesFor =
            computeCap:
            [
              (desktopFeatureFor computeCap)
              "pulid"
              "webp"
            ]
            # The desktop app embeds the same server, so it takes the same
            # convolution backend on Linux CUDA (#1483).
            ++ lib.optionals isLinux [ "cudnn" ];
          desktopFeatures = lib.concatStringsSep "," (desktopFeaturesFor cudaComputeCap);

          gpuFeature =
            if isLinux then
              "cuda"
            else if isDarwin then
              "metal"
            else
              "";

          devProfile = "dev-fast";

          # Full shipping feature set used for release builds and feature coverage.
          #
          # `pulid` ships ON. The feature only decides whether the binary LINKS
          # the PuLID stack; the capability stays unadvertised unless a
          # qualified checkpoint is installed AND the four-file bundle has been
          # pulled with the InsightFace licence explicitly accepted, so a user
          # who never asks for a face never sees it. Shipping it off would make
          # building from source the only route to identity conditioning, which
          # is not a licence decision — the licence gate is the acceptance
          # record, and it is enforced at download time.
          releaseFeaturesFor =
            computeCap:
            if isLinux then
              # `cudnn` is Linux-CUDA only and deliberately not implied by
              # `cuda`: it links libcudnn and needs its headers, which a plain
              # `cargo check --features cuda` must not require (#1483).
              "${
                if computeCap == "89" then "h3-cuda" else "cuda"
              },cudnn,preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid"
            else if gpuFeature != "" then
              "${gpuFeature},h3,preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid"
            else
              "preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid";

          # Shell completion generation only needs CLI shape, not GPU linkage.
          # Keep this CUDA-free so Linux sandbox builds can generate completion
          # scripts without loading the host-only NVIDIA driver library.
          completionFeatures = "preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid";

          # Devshell defaults compile the full shipping feature set so that
          # `mold tui`, `mold discord`, WebP/MP4 output, Prometheus metrics,
          # and local prompt expansion are all available from the interactive
          # `mold`, `serve`, and `generate` commands without the user having
          # to know which features to flip. CI and `nix build` use the same
          # list via `releaseFeatures`, so there's a single feature matrix.
          releaseFeatures = releaseFeaturesFor cudaComputeCap;
          devFeatures = releaseFeatures;

          cargoArtifacts = craneLib.buildDepsOnly (
            commonArgs
            // {
              cargoExtraArgs = "-p mold-ai --features ${releaseFeatures}";
            }
          );

          webEmbedSetup = ''
            export SCCACHE_DIR="''${MOLD_SCCACHE_DIR:-$PWD/.cache/sccache}"
            export MOLD_WEB_DIST="$PWD/web/dist"
            ./scripts/ensure-web-dist.sh
          '';

          # Tauri desktop app (desktop/): its cargo root is excluded from the
          # workspace, so every command targets its manifest explicitly. On
          # Darwin the Apple linker must be used — the Nix linker breaks
          # objc2/system-framework linking and produces Team-ID-rejected
          # binaries (pattern proven in the Aethon project).
          desktopSetup = ''
            export SCCACHE_DIR="''${MOLD_SCCACHE_DIR:-$PWD/.cache/sccache}"
          ''
          + lib.optionalString isDarwin ''
            export CC=/usr/bin/cc
            export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER=/usr/bin/cc
            export RUSTC_LINKER=/usr/bin/cc
          '';

          assertMoldRunpathScriptFor =
            {
              ccLib,
              cudaCudart,
              libcublas,
              libcurand,
            }:
            ''
              assertMoldRunpath() {
                rpath="$(patchelf --print-rpath $out/bin/mold)"
                needed="$(patchelf --print-needed $out/bin/mold)"
                echo "mold RUNPATH: $rpath"
                echo "mold NEEDED: $needed"

                case ":$rpath:" in
                  *":${ccLib}/lib:"*) ;;
                  *) echo "missing libstdc++ RUNPATH entry" >&2; exit 1 ;;
                esac
                case "$needed" in
                  *libcudart.so*)
                    case ":$rpath:" in
                      *":${cudaCudart}/lib:"*) ;;
                      *) echo "missing CUDA runtime RUNPATH entry" >&2; exit 1 ;;
                    esac
                    ;;
                esac
                case ":$rpath:" in
                  *":${libcublas}/lib:"*) ;;
                  *) echo "missing libcublas RUNPATH entry" >&2; exit 1 ;;
                esac
                case ":$rpath:" in
                  *":${libcurand}/lib:"*) ;;
                  *) echo "missing libcurand RUNPATH entry" >&2; exit 1 ;;
                esac
                case ":$rpath:" in
                  *":/run/opengl-driver/lib:"*) ;;
                  *) echo "missing NVIDIA driver RUNPATH entry" >&2; exit 1 ;;
                esac
              }
            '';

          assertMoldRunpathScript = assertMoldRunpathScriptFor {
            ccLib = "${pkgs.stdenv.cc.cc.lib}";
            cudaCudart = "${pkgs.cudaPackages.cuda_cudart}";
            libcublas = "${pkgs.cudaPackages.libcublas.lib}";
            libcurand = "${pkgs.cudaPackages.libcurand.lib}";
          };

          moldRunpathAssertHook = pkgs.writeTextFile {
            name = "mold-runpath-assert-hook";
            destination = "/nix-support/setup-hook";
            text = ''
              ${assertMoldRunpathScript}

              postFixupHooks+=(assertMoldRunpath)
            '';
          };

          desktopDriverRunpathScript = ''
            fixupDesktopDriverRunpath() {
              local root="$1"
              while IFS= read -r -d $'\0' candidate; do
                needed="$(patchelf --print-needed "$candidate" 2>/dev/null)" || continue
                if printf '%s\n' "$needed" | grep -Fxq libcuda.so.1; then
                  rpath="$(patchelf --print-rpath "$candidate")"
                  case ":$rpath:" in
                    *":/run/opengl-driver/lib:"*) ;;
                    *) patchelf --add-rpath /run/opengl-driver/lib "$candidate" ;;
                  esac
                fi
              done < <(find "$root" -type f -print0)
            }

            assertDesktopDriverRunpath() {
              local root="$1"
              local cuda_consumers=0
              while IFS= read -r -d $'\0' candidate; do
                needed="$(patchelf --print-needed "$candidate" 2>/dev/null)" || continue
                if printf '%s\n' "$needed" | grep -Fxq libcuda.so.1; then
                  cuda_consumers=$((cuda_consumers + 1))
                  rpath="$(patchelf --print-rpath "$candidate")"
                  case ":$rpath:" in
                    *":/run/opengl-driver/lib:"*) ;;
                    *)
                      echo "CUDA consumer is missing the NVIDIA driver RUNPATH: $candidate" >&2
                      return 1
                      ;;
                  esac
                fi
              done < <(find "$root" -type f -print0)
              if [ "$cuda_consumers" -eq 0 ]; then
                echo "no libcuda.so.1 consumer found under $root" >&2
                return 1
              fi
            }
          '';

          # This hook is deliberately listed after autoPatchelfHook and
          # wrapGAppsHook3. autoPatchelf replaces RUNPATHs and wrapGApps renames
          # the real executable, so the driver path must be added after both.
          desktopDriverRunpathHook = pkgs.writeTextFile {
            name = "mold-desktop-driver-runpath-hook";
            destination = "/nix-support/setup-hook";
            text = ''
              ${desktopDriverRunpathScript}

              fixupAndAssertDesktopDriverRunpath() {
                fixupDesktopDriverRunpath "$out"
                assertDesktopDriverRunpath "$out"
              }

              postFixupHooks+=(fixupAndAssertDesktopDriverRunpath)
            '';
          };

          # Merged CUDA toolkit so bindgen_cuda can find both bin/nvcc and
          # include/cuda.h, and so cudarc's build script finds cudnn_version.h
          # and libcudnn for the `cudnn` feature (#1483). cuDNN adds ~1.2 GB to
          # the closure; it is what makes a Wan VAE decode 4.4x cheaper on its
          # convolutions, and `crate::conv_policy` still decides per family
          # whether a render uses it.
          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit-merged";
            paths = [
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cuda_cudart
            ]
            ++ lib.optionals isLinux [
              pkgs.cudaPackages.cudnn.dev
              pkgs.cudaPackages.cudnn.include
              pkgs.cudaPackages.cudnn.lib
            ];
          };

          meta = with lib; {
            description = "Local AI image generation CLI for FLUX, SD1.5, SDXL & Z-Image diffusion models";
            homepage = "https://github.com/utensils/mold";
            license = licenses.mit;
            mainProgram = "mold";
            maintainers = [ ];
          };

          # Web gallery SPA (Vue 3 + Vite + Tailwind v4). Built via bun2nix so
          # the `node_modules` cache is reproducibly derived from the root bun.lock.
          # Output layout: `$out/index.html` + `$out/assets/...` — consumed at
          # Rust compile time via `MOLD_WEB_DIST`, then embedded into the
          # `mold` binary by `rust-embed` (see `crates/mold-server/build.rs`).
          mold-web = pkgs.stdenv.mkDerivation {
            pname = "mold-web";
            version = workspaceVersion;
            # Keep the workspace root so Bun can link @mold/studio and @mold/ui.
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./package.json
                ./bun.lock
                ./desktop/package.json
                ./web
                ./studio
                ./ui
              ];
            };
            sourceRoot = "source";
            nativeBuildInputs = [ pkgs.bun2nix.hook ];
            bunDeps = pkgs.bun2nix.fetchBunDeps {
              bunNix = ./bun.nix;
            };
            # Keep the install path fully offline and platform-stable. Do not rely
            # on bun2nix hook defaults here: bun2nix revs have differed on whether
            # they pass --backend=symlink, and without it Bun ignores
            # BUN_INSTALL_CACHE_DIR and attempts live npm fetches inside the Nix
            # sandbox (issues #286 and #330).
            bunInstallFlags = [
              "--linker=isolated" # avoids AccessDenied from the hoisted linker creating nested estree-walker dirs.
              "--backend=symlink" # resolves packages from BUN_INSTALL_CACHE_DIR via symlinks.
            ];

            # Skip the lifecycle-scripts phase (second bun install without
            # --ignore-scripts). All native-binary packages (esbuild, rollup,
            # @tailwindcss/oxide, lightningcss) ship their platform-specific tarballs
            # as explicit entries in bunDeps already; their postinstall download
            # scripts would attempt network access the sandbox blocks.
            dontRunLifecycleScripts = true;
            buildPhase = ''
              runHook preBuild
              bun run build:web
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              cp -R web/dist/. $out/
              runHook postInstall
            '';
          };

          # Desktop app frontend (Vue SPA under desktop/), built like mold-web.
          mold-desktop-web = pkgs.stdenv.mkDerivation {
            pname = "mold-desktop-web";
            version = workspaceVersion;
            # Same root workspace layout as mold-web.
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./package.json
                ./bun.lock
                ./desktop
                ./web/package.json
                ./studio
                ./ui
              ];
            };
            sourceRoot = "source";
            nativeBuildInputs = [ pkgs.bun2nix.hook ];
            bunDeps = pkgs.bun2nix.fetchBunDeps {
              bunNix = ./bun.nix;
            };
            bunInstallFlags = [
              "--linker=isolated"
              "--backend=symlink"
            ];
            dontRunLifecycleScripts = true;
            buildPhase = ''
              runHook preBuild
              bun run build:desktop
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              cp -R desktop/dist/. $out/
              runHook postInstall
            '';
          };

          # Tauri desktop app. The Nix package uses a native app bundle on
          # Darwin and a deb intermediate on Linux, which cargo-tauri.hook
          # installs into a regular Nix output. AppImage builds remain a
          # developer/CI distribution path outside the Nix sandbox.
          mkMoldDesktop =
            computeCap:
            pkgs.rustPlatform.buildRustPackage {
              pname = "mold-desktop";
              version = workspaceVersion;
              src = craneLib.path ./.;
              cargoRoot = "desktop/src-tauri";
              buildAndTestSubdir = "desktop/src-tauri";
              cargoLock = {
                lockFile = ./desktop/src-tauri/Cargo.lock;
                outputHashes = {
                  "candle-core-mold-0.11.1" = "sha256-givk1MIAncZN+YO/XI6DPMPaIOHw6G68DRR40y5Oims=";
                  "cudarc-0.19.8" = "sha256-ARnabIhBCzahrk/kVCt5084gftGDyCBme3jxg+mvkUA=";
                };
              };
              buildFeatures = desktopFeaturesFor computeCap;

              MOLD_GIT_SHA = gitRev;
              MOLD_BUILD_DATE = gitDate;
              # The embedded engine serves the regular web SPA to browsers.
              MOLD_WEB_DIST = "${mold-web}";
              MOLD_BUNDLED_FFMPEG = "${pkgs.ffmpeg}/bin/ffmpeg";
              MOLD_BUNDLED_FFPROBE = "${pkgs.ffmpeg}/bin/ffprobe";
              CUDA_PATH = lib.optionalString isLinux "${cudaToolkit}";
              CUDA_COMPUTE_CAP = lib.optionalString isLinux computeCap;
              NIX_LDFLAGS = lib.optionalString isLinux "-L${pkgs.cudaPackages.cuda_cudart}/lib/stubs";

              nativeBuildInputs = [
                pkgs.cargo-tauri.hook
                pkgs.pkg-config
                pkgs.nasm
                # `candle-onnx`'s build script drives `prost-build`, which
                # shells out to `protoc`. The `pulid` feature pulls that crate
                # into the desktop graph too.
                pkgs.protobuf
                pkgs.makeBinaryWrapper
              ]
              ++ lib.optionals isLinux [
                pkgs.autoPatchelfHook
                pkgs.clang
                pkgs.cudaPackages.cuda_nvcc
                pkgs.gitMinimal
                pkgs.lld
                pkgs.wrapGAppsHook3
                desktopDriverRunpathHook
              ];
              buildInputs = [
                pkgs.openssl
                pkgs.libwebp
              ]
              ++ lib.optionals isLinux (
                desktopLinuxBuildInputs
                ++ [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.cudaPackages.cuda_cudart
                  pkgs.cudaPackages.libcublas.lib
                  pkgs.cudaPackages.cuda_nvrtc.lib
                  pkgs.cudaPackages.libcurand.lib
                  pkgs.cudaPackages.cudnn.lib
                ]
              );

              autoPatchelfIgnoreMissingDeps = lib.optionals isLinux [ "libcuda.so.1" ];

              postPatch = ''
                mkdir -p desktop/dist
                cp -R ${mold-desktop-web}/. desktop/dist/
                ${pkgs.jq}/bin/jq '.build.beforeBuildCommand = ""' \
                  desktop/src-tauri/tauri.conf.json > tauri.conf.tmp
                mv tauri.conf.tmp desktop/src-tauri/tauri.conf.json
              '';

              preBuild = lib.optionalString isLinux prepareCudaforgeCache;

              tauriBundleType = lib.optionalString isLinux "deb";
              tauriBuildFlags = lib.optionals isDarwin [
                "--bundles"
                "app"
              ];
              doCheck = false;

              postInstall =
                if isLinux then
                  ''
                    desktop_bin="$(find "$out" -type f -name mold-desktop -print -quit)"
                    if [ -z "$desktop_bin" ]; then
                      echo "installed Mold desktop executable is missing" >&2
                      exit 1
                    fi
                    ${pkgs.bash}/bin/bash ${./scripts/verify-h3-release-exclusion.sh} "$desktop_bin"
                  ''
                else
                  ''
                    if [ -d "$out/Applications/Mold.app" ]; then
                      mkdir -p $out/bin
                      makeBinaryWrapper "$out/Applications/Mold.app/Contents/MacOS/mold-desktop" \
                        $out/bin/mold-desktop
                    fi
                  '';

              # A signed/notarized bundle must not load /nix/store dylibs
              # (dyld Team-ID rejection); libiconv links in via stdenv — point
              # it at the system copy and re-sign ad hoc.
              postFixup = lib.optionalString isDarwin ''
                app_bin="$out/Applications/Mold.app/Contents/MacOS/mold-desktop"
                if [ -f "$app_bin" ]; then
                  for ref in $(${pkgs.darwin.cctools}/bin/otool -L "$app_bin" \
                    | awk '/\/nix\/store\/.*libiconv/ {print $1}'); do
                    ${pkgs.darwin.cctools}/bin/install_name_tool \
                      -change "$ref" /usr/lib/libiconv.2.dylib "$app_bin"
                  done
                  # install_name_tool invalidates the ad-hoc signature; re-sign
                  # with the sigtool codesign shim (sandbox has no Apple codesign).
                  ${pkgs.darwin.sigtool}/bin/codesign -f -s - "$app_bin"
                  # tail +2: otool's header line is the binary's own store path.
                  if ${pkgs.darwin.cctools}/bin/otool -L "$app_bin" | tail -n +2 | grep -q "/nix/store"; then
                    echo "mold-desktop still references /nix/store dylibs:" >&2
                    ${pkgs.darwin.cctools}/bin/otool -L "$app_bin" >&2
                    exit 1
                  fi
                fi
              '';

              passthru.moldCudaComputeCapability = computeCap;

              meta = with lib; {
                description = "Mold — native desktop app for local AI image/video generation";
                homepage = "https://github.com/utensils/mold";
                license = licenses.mit;
                mainProgram = "mold-desktop";
                platforms = [
                  "aarch64-darwin"
                  "x86_64-linux"
                ];
              };
            };

          mold-desktop = mkMoldDesktop cudaComputeCap;

          # Build a mold package for a given CUDA compute capability.
          # `MOLD_WEB_DIST` is read by `crates/mold-server/build.rs`, which
          # stages the SPA into a directory that `rust-embed` bakes into the
          # binary at compile time. The result is a true single-file install:
          # `$out/bin/mold` serves the gallery UI with no runtime dependency
          # on `share/mold/web` or any external assets.
          mkMold =
            computeCap:
            craneLib.buildPackage (
              commonArgs
              // {
                inherit cargoArtifacts meta;
                MOLD_WEB_DIST = "${mold-web}";
                # The codec bridge stays process-isolated, but the packaged
                # server does not depend on the caller's PATH.
                MOLD_BUNDLED_FFMPEG = "${pkgs.ffmpeg}/bin/ffmpeg";
                MOLD_BUNDLED_FFPROBE = "${pkgs.ffmpeg}/bin/ffprobe";
                cargoExtraArgs = "-p mold-ai --features ${releaseFeaturesFor computeCap}";
                postInstall =
                  if isLinux then
                    ''
                      ${pkgs.bash}/bin/bash ${./scripts/verify-h3-release-exclusion.sh} "$out/bin/mold"

                      # Build a CUDA-free helper for completion generation, then
                      # patch its RUNPATH before execing it. The installed binary
                      # is CUDA-linked and cannot run in the sandbox because the
                      # host-only NVIDIA driver (`libcuda.so.1`) is absent.
                      cargoWithProfile build -p mold-ai --features ${completionFeatures}
                      patchelf --set-rpath ${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]} target/release/mold

                      installShellCompletion --cmd mold \
                        --bash <(target/release/mold completions bash) \
                        --zsh <(target/release/mold completions zsh) \
                        --fish <(target/release/mold completions fish)
                    ''
                  else
                    ''
                      installShellCompletion --cmd mold \
                        --bash <($out/bin/mold completions bash) \
                        --zsh <($out/bin/mold completions zsh) \
                        --fish <($out/bin/mold completions fish)
                    '';
                nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.installShellFiles ];
              }
              // lib.optionalAttrs isLinux {
                CUDA_COMPUTE_CAP = computeCap;
                nativeBuildInputs = commonArgs.nativeBuildInputs ++ [
                  pkgs.installShellFiles
                  pkgs.autoPatchelfHook
                  pkgs.autoAddDriverRunpath
                  moldRunpathAssertHook
                ];

                # `libcuda.so.1` comes from the host NVIDIA driver and is not
                # available in the Nix build sandbox. Resolve CUDA toolkit libs
                # with autoPatchelf, then let autoAddDriverRunpath add the stable
                # NixOS driver RUNPATH after autoPatchelf's `--set-rpath` pass.
                autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];

                # Sandboxed Linux builders do not provide the host NVIDIA
                # driver (`libcuda.so.1`). The CUDA-linked CLI can therefore
                # fail in the dynamic loader before reaching `main()` when
                # integration tests exec `target/release/mold`. Keep the Nix
                # check hermetic by running only binary unit tests; exec-based
                # CLI smoke tests remain covered by non-sandbox CI.
                cargoTestExtraArgs = "--bins";

                # The H3 release-provenance marker deliberately keeps the
                # CUDA/C++ dependency path live in the binary test harness.
                # It runs before autoPatchelf, so expose the pinned toolkit
                # libraries and inert driver stub only during checkPhase.
                preCheck = ''
                  export LD_LIBRARY_PATH=${
                    lib.makeLibraryPath [
                      pkgs.stdenv.cc.cc.lib
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudaPackages.libcublas.lib
                      pkgs.cudaPackages.libcurand.lib
                      pkgs.cudaPackages.cudnn.lib
                    ]
                  }:${pkgs.cudaPackages.cuda_cudart}/lib/stubs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
                '';
              }
              // {
                passthru.moldCudaComputeCapability = computeCap;
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
            inherit
              mold
              mold-desktop
              mold-desktop-web
              mold-web
              ;
            mold-discord = moldDiscord;
            default = mold;
          }
          // lib.optionalAttrs isLinux {
            mold-sm86 = mkMold "86"; # Ampere (RTX 3090/A40)
            mold-sm100 = mkMold "100"; # Datacenter Blackwell (B200/B300)
            mold-sm120 = mkMold "120"; # Consumer Blackwell (RTX 50-series)
            mold-desktop-sm86 = mkMoldDesktop "86";
            mold-desktop-sm120 = mkMoldDesktop "120";
          };

          checks = {
            runpath-assertion = pkgs.runCommand "mold-runpath-assertion-check" { } ''
              set -eu

              mkdir -p "$out/bin" "$TMPDIR/bin"
              touch "$out/bin/mold"

              cat > "$TMPDIR/bin/patchelf" <<'EOF'
              #!${pkgs.runtimeShell}
              set -eu
              case "$1" in
                --print-rpath)
                  printf '%s\n' "$MOLD_TEST_RPATH"
                  ;;
                --print-needed)
                  printf '%s\n' "$MOLD_TEST_NEEDED"
                  ;;
                *)
                  echo "unexpected patchelf args: $*" >&2
                  exit 2
                  ;;
              esac
              EOF
              chmod +x "$TMPDIR/bin/patchelf"
              export PATH="$TMPDIR/bin:$PATH"

              ${assertMoldRunpathScriptFor {
                ccLib = "/nix/store/test-libstdcxx";
                cudaCudart = "/nix/store/test-cuda-cudart";
                libcublas = "/nix/store/test-libcublas";
                libcurand = "/nix/store/test-libcurand";
              }}

              base_rpath="/nix/store/test-libstdcxx/lib:/nix/store/test-libcublas/lib:/nix/store/test-libcurand/lib:/run/opengl-driver/lib"

              MOLD_TEST_RPATH="$base_rpath"
              MOLD_TEST_NEEDED="$(printf '%s\n' \
                "libcublas.so.12" \
                "libcurand.so.10")"
              export MOLD_TEST_RPATH MOLD_TEST_NEEDED
              assertMoldRunpath

              MOLD_TEST_NEEDED="$(printf '%s\n' \
                "libcudart.so.12" \
                "libcublas.so.12" \
                "libcurand.so.10")"
              export MOLD_TEST_NEEDED
              if ( assertMoldRunpath ); then
                echo "expected missing libcudart RUNPATH to fail when libcudart is needed" >&2
                exit 1
              fi

              MOLD_TEST_RPATH="$base_rpath:/nix/store/test-cuda-cudart/lib"
              export MOLD_TEST_RPATH
              assertMoldRunpath
            '';
          }
          // lib.optionalAttrs isLinux {
            # The devshell's own advertised feature set must be RUNNABLE in it.
            # `releaseFeatures` includes `cudnn` on Linux and the binary carries
            # no RUNPATH, so every library it links has to be on LD_LIBRARY_PATH
            # or `mold` dies in the dynamic loader (#1510); LIBRARY_PATH is held
            # to the same set so the link path cannot drift from the load path.
            #
            # Read back out of `config.devshells.default.env` rather than off
            # the `let` bindings, so re-inlining either `value =` as a
            # hand-rolled list — the exact regression shape — is still caught
            # instead of quietly orphaning the bindings this would have tested.
            devshell-cuda-load-path =
              let
                # `builtins.match`, under `hasPrefix`/`hasSuffix`, refuses a
                # pattern carrying string context, so compare as plain text.
                plain = builtins.unsafeDiscardStringContext;

                envValue =
                  name:
                  let
                    matches = lib.filter (entry: entry.name == name) config.devshells.default.env;
                  in
                  assert lib.assertMsg (matches != [ ]) "the devshell sets no ${name}";
                  plain (lib.head matches).value;

                libraryPath = envValue "LIBRARY_PATH";
                ldLibraryPath = envValue "LD_LIBRARY_PATH";

                driverPath = "/run/opengl-driver/lib";
                stubsPath = plain "${pkgs.cudaPackages.cuda_cudart}/lib/stubs";

                # `lib.getLib` mirrors what `makeLibraryPath` resolves, and the
                # comparison is per `:`-segment rather than by substring:
                # `${cuda_cudart}/lib` is a prefix of `${cuda_cudart}/lib/stubs`,
                # so an infix test would call cudart present on a path carrying
                # only its stub directory.
                required = lib.mapAttrs (_: drv: plain "${lib.getLib drv}/lib") {
                  "libstdc++" = pkgs.stdenv.cc.cc.lib;
                  cudart = pkgs.cudaPackages.cuda_cudart;
                  cublas = pkgs.cudaPackages.libcublas.lib;
                  nvrtc = pkgs.cudaPackages.cuda_nvrtc.lib;
                  curand = pkgs.cudaPackages.libcurand.lib;
                  cudnn = pkgs.cudaPackages.cudnn.lib;
                };
                missingFrom =
                  value:
                  let
                    entries = lib.splitString ":" value;
                  in
                  lib.filter (name: !builtins.elem required.${name} entries) (lib.attrNames required);

                missingLink = missingFrom libraryPath;
                missingLoad = missingFrom ldLibraryPath;
              in
              assert lib.assertMsg (missingLoad == [ ]) (
                "devshell LD_LIBRARY_PATH is missing libraries the release feature set links: "
                + lib.concatStringsSep ", " missingLoad
              );
              assert lib.assertMsg (missingLink == [ ]) (
                "devshell LIBRARY_PATH is missing libraries the release feature set links: "
                + lib.concatStringsSep ", " missingLink
              );
              # The driver directory has to lead both paths so the real
              # libcuda.so wins over the cudart stub, which has to stay last on
              # LIBRARY_PATH. Losing that order costs CUDA_ERROR_STUB_LIBRARY at
              # runtime, and a membership test alone cannot see it.
              assert lib.assertMsg (lib.hasPrefix "${driverPath}:" libraryPath) (
                "devshell LIBRARY_PATH must start with ${driverPath}"
              );
              assert lib.assertMsg (lib.hasPrefix "${driverPath}:" ldLibraryPath) (
                "devshell LD_LIBRARY_PATH must start with ${driverPath}"
              );
              assert lib.assertMsg (lib.hasSuffix ":${stubsPath}" libraryPath) (
                "devshell LIBRARY_PATH must end with the cudart stubs directory"
              );
              pkgs.runCommand "mold-devshell-cuda-load-path-check" { } "touch $out";

            artifact-attestation-private-state =
              let
                evaluated = inputs.nixpkgs.lib.nixosSystem {
                  inherit system;
                  modules = [
                    ./nix/module.nix
                    {
                      services.mold = {
                        enable = true;
                        package = mold;
                      };
                    }
                  ];
                };
                service = evaluated.config.systemd.services.mold;
              in
              assert
                service.environment.MOLD_ARTIFACT_ATTESTATIONS_DIR == "/var/lib/mold-artifact-attestations-v1";
              assert service.serviceConfig.StateDirectory == "mold-artifact-attestations-v1";
              assert service.serviceConfig.StateDirectoryMode == "0700";
              assert builtins.elem "d /var/lib/mold 0775 mold mold -" evaluated.config.systemd.tmpfiles.rules;
              pkgs.runCommand "mold-artifact-attestation-private-state-check" { } ''
                touch "$out"
              '';

            cuda-package-consistency =
              let
                moduleWarnings =
                  cudaArch: package:
                  (inputs.nixpkgs.lib.nixosSystem {
                    inherit system;
                    modules = [
                      ./nix/module.nix
                      {
                        services.mold = {
                          enable = true;
                          inherit cudaArch package;
                        };
                      }
                    ];
                  }).config.warnings;
                correctPairs = [
                  {
                    cudaArch = "ampere";
                    package = mkMold "86";
                  }
                  {
                    cudaArch = "ada";
                    package = mold;
                  }
                  {
                    cudaArch = "blackwell-datacenter";
                    package = mkMold "100";
                  }
                  {
                    cudaArch = "blackwell";
                    package = mkMold "120";
                  }
                ];
                correct = builtins.all (pair: moduleWarnings pair.cudaArch pair.package == [ ]) correctPairs;
                mismatch = builtins.length (moduleWarnings "ada" (mkMold "86")) == 1;
                unknownPackage =
                  builtins.length (moduleWarnings "ada" (pkgs.writeShellScriptBin "mold" "exit 0")) == 1;
              in
              assert correct;
              assert mismatch;
              assert unknownPackage;
              pkgs.runCommand "mold-cuda-package-consistency-check" { } ''
                touch "$out"
              '';

            desktop-runtime-closure = pkgs.runCommand "mold-desktop-runtime-closure-check" { } ''
              set -eu
              runtime_path=${lib.escapeShellArg (lib.makeLibraryPath desktopLinuxRuntimeInputs)}
              for library in libgdk_pixbuf-2.0.so.0 libcairo.so.2 libglib-2.0.so.0 libgio-2.0.so.0; do
                found=
                old_ifs=$IFS
                IFS=:
                for directory in $runtime_path; do
                  if [ -e "$directory/$library" ]; then
                    found=1
                    break
                  fi
                done
                IFS=$old_ifs
                if [ -z "$found" ]; then
                  echo "desktop runtime closure is missing $library" >&2
                  exit 1
                fi
              done

              ${desktopDriverRunpathScript}
              assertDesktopDriverRunpath ${mold-desktop}

              touch "$out"
            '';
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
              # `candle-onnx`'s build script drives `prost-build`, which shells
              # out to `protoc` to parse `onnx.proto3`. The `pulid` feature
              # pulls that crate in; it is in the shipping feature set, so the
              # crane `nativeBuildInputs` carry it too.
              pkgs.protobuf
              pkgs.sccache
              pkgs.git
              pkgs.gh
              pkgs.jq
              pkgs.tokei
              pkgs.lsof
              pkgs.curl
              pkgs.viu
              pkgs.mpv
              pkgs.cargo-llvm-cov
              pkgs.ffmpeg
              pkgs.imagemagick
              pkgs.bun
              pkgs.bun2nix
              pkgs.cargo-tauri
              pkgs.nodejs_22
              pkgs.nodePackages.prettier
              pkgs.pnpm
              pkgs.tmux
              pkgs.runpodctl
            ]
            ++ lib.optionals isDarwin [
              pkgs.libiconv
              pkgs.llvmPackages.libcxxClang
            ]
            ++ lib.optionals isLinux [
              pkgs.clang
              pkgs.file
              pkgs.lld
              pkgs.wget
              pkgs.xdg-utils
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.libcublas.lib
              pkgs.cudaPackages.cuda_nvtx.lib
              pkgs.cudaPackages.cuda_nvrtc.lib
              pkgs.cudaPackages.libcurand.lib
            ]
            ++ lib.optionals isLinux desktopLinuxBuildInputs;

            env = [
              {
                name = "RUST_BACKTRACE";
                value = "1";
              }
              {
                name = "MOLD_LTX_DEBUG";
                value = "1";
              }
              # `sccache` (below) refuses to run whenever `CARGO_INCREMENTAL`
              # is set in the environment — it rejects *any* value, so we can't
              # paper over the issue with `CARGO_INCREMENTAL=0`. The devshell
              # therefore deliberately leaves the variable unset and relies on
              # the project-wide `sccache` cache instead of per-crate
              # incremental builds. Direct `cargo` users outside the devshell
              # still get cargo's default incremental behavior.
              {
                name = "RUSTC_WRAPPER";
                value = "sccache";
              }
              {
                name = "OPENSSL_DIR";
                value = "${pkgs.openssl.dev}";
              }
              {
                name = "PKG_CONFIG_PATH";
                value = desktopPkgConfigPath;
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
                value = devshellLinuxLibraryPath;
              }
              {
                name = "LD_LIBRARY_PATH";
                value = devshellLinuxLdLibraryPath;
              }
            ];

            commands = [
              {
                category = "build";
                name = "build";
                help = "fast local mold build with the web bundle embedded";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo build --profile ${devProfile} -p mold-ai --features ${devFeatures} "$@"
                '';
              }
              {
                category = "build";
                name = "build-workspace";
                help = "cargo build the full workspace in debug mode";
                command = "cargo build \"$@\"";
              }
              {
                category = "build";
                name = "build-release";
                help = "shipping mold build with the full feature set and embedded web UI";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo build --release -p mold-ai --features ${releaseFeatures} "$@"
                '';
              }
              {
                category = "build";
                name = "build-server";
                help = "fast local server build with GPU + preview + expand and embedded web UI";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo build --profile ${devProfile} -p mold-ai --features ${devFeatures} "$@"
                '';
              }
              {
                category = "build";
                name = "build-discord";
                help = "fast local Discord-bot build";
                command = "cargo build --profile ${devProfile} -p mold-ai --features discord \"$@\"";
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
                help = "cargo check --workspace (matches CI)";
                command = "cargo check --workspace \"$@\"";
              }
              {
                category = "check";
                name = "clippy";
                help = "cargo clippy --workspace --all-targets -- -D warnings (matches CI)";
                command = "cargo clippy --workspace --all-targets \"$@\" -- -D warnings";
              }
              {
                category = "check";
                name = "run-tests";
                help = "cargo test --workspace (matches CI)";
                command = "cargo test --workspace \"$@\"";
              }
              {
                category = "check";
                name = "ci-local";
                help = "run main's CI gates locally in a clean env (rust/web/docs/contracts; --list, -k)";
                command = ''
                  repo_dir="''${PRJ_ROOT:-$(git rev-parse --show-toplevel)}"
                  exec "$repo_dir/scripts/ci-local.sh" "$@"
                '';
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
                help = "run mold CLI with the fast local feature set";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo run --profile ${devProfile} -p mold-ai --features ${devFeatures} -- "$@"
                '';
              }
              {
                category = "run";
                name = "serve";
                help = "start the mold server";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo run --profile ${devProfile} -p mold-ai --features ${devFeatures} -- serve "$@"
                '';
              }
              {
                category = "run";
                name = "generate";
                help = "generate an image from a prompt";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo run --profile ${devProfile} -p mold-ai --features ${devFeatures} -- run "$@"
                '';
              }
              {
                category = "run";
                name = "discord-bot";
                help = "start the mold Discord bot";
                command = "cargo run --profile ${devProfile} -p mold-ai --features discord -- discord \"$@\"";
              }
              {
                category = "runpod";
                name = "runpod-doctor";
                help = "mold runpod doctor — verify RunPod auth";
                command = "cargo run -p mold-ai --features ${devFeatures} -- runpod doctor \"$@\"";
              }
              {
                category = "runpod";
                name = "runpod-list";
                help = "mold runpod list — list your RunPod pods";
                command = "cargo run -p mold-ai --features ${devFeatures} -- runpod list \"$@\"";
              }
              {
                category = "runpod";
                name = "runpod-create";
                help = "mold runpod create — create a new pod";
                command = "cargo run -p mold-ai --features ${devFeatures} -- runpod create \"$@\"";
              }
              {
                category = "runpod";
                name = "runpod-run";
                help = "mold runpod run <prompt> — generate on a RunPod pod end-to-end";
                command = "cargo run -p mold-ai --features ${devFeatures} -- runpod run \"$@\"";
              }
              {
                category = "runpod";
                name = "runpod-usage";
                help = "mold runpod usage — balance and spend summary";
                command = "cargo run -p mold-ai --features ${devFeatures} -- runpod usage \"$@\"";
              }
              {
                category = "run";
                name = "build-ltx2";
                help = "build mold with the full feature set for LTX-2 work";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo build --profile ${devProfile} -p mold-ai --features ${releaseFeatures} "$@"
                '';
              }
              {
                category = "run";
                name = "smoke-ltx2";
                help = "run a local LTX-2 / LTX-2.3 smoke inference";
                command = ''
                  set -euo pipefail
                  ${webEmbedSetup}
                  cargo run --profile ${devProfile} -p mold-ai --features ${releaseFeatures} -- run --local "$@"
                '';
              }
              {
                category = "run";
                name = "contact-sheet";
                help = "build native review artifacts from a clip via the Rust ltx2_review tool";
                command = ''
                  set -euo pipefail
                  if [ "$#" -lt 1 ]; then
                    echo "usage: contact-sheet <input.mp4> [more.mp4...]"
                    exit 1
                  fi
                  cargo run -p mold-ai-inference --features dev-bins --bin ltx2_review -- "$@"
                '';
              }
              {
                category = "ios";
                name = "ios-dev";
                help = "run the iPhone app with Tauri hot reload (defaults to an iPhone simulator)";
                command = "./scripts/ios.sh dev \"$@\"";
              }
              {
                category = "ios";
                name = "ios-run";
                help = "build and run the production app on an iPhone or simulator";
                command = "./scripts/ios.sh run \"$@\"";
              }
              {
                category = "ios";
                name = "ios-check";
                help = "cross-check the thin Tauri shell for the Apple Silicon simulator";
                command = "./scripts/ios.sh check \"$@\"";
              }
              {
                category = "ios";
                name = "ios-build";
                help = "archive and export the iPhone app for App Store Connect";
                command = "./scripts/ios.sh build \"$@\"";
              }
              {
                category = "android";
                name = "android-dev";
                help = "run the Android app with Tauri hot reload";
                command = "./scripts/android.sh dev \"$@\"";
              }
              {
                category = "android";
                name = "android-run";
                help = "build and run the production app on Android";
                command = "./scripts/android.sh run \"$@\"";
              }
              {
                category = "android";
                name = "android-check";
                help = "build a debug ARM64 APK from the shared mobile shell";
                command = "./scripts/android.sh check \"$@\"";
              }
              {
                category = "android";
                name = "android-test";
                help = "build Android and run native instrumentation tests on an emulator";
                command = "./scripts/android.sh test \"$@\"";
              }
              {
                category = "android";
                name = "android-build";
                help = "build Android ARM64/ARMv7 app bundles for Google Play";
                command = "./scripts/android.sh build \"$@\"";
              }
              {
                category = "android";
                name = "android-emulator";
                help = "boot the external-storage Mold_API_37 emulator";
                command = "./scripts/android.sh emulator \"$@\"";
              }
              {
                category = "android";
                name = "android-doctor";
                help = "verify Android Studio, SDK, NDK, AVD, and cache paths";
                command = "./scripts/android.sh doctor \"$@\"";
              }
              {
                category = "desktop";
                name = "desktop-dev";
                help = "run the native Tauri desktop app with hot reload (Vite on :1430)";
                command = ''
                  set -euo pipefail
                  ${desktopSetup}
                  # Killing a previous desktop-dev mid-build orphans its Vite
                  # child, which keeps :1430 bound and fails the next run with
                  # "Port 1430 is already in use" — reap stale listeners first.
                  stale=$(lsof -ti tcp:1430 || true)
                  if [ -n "$stale" ]; then
                    echo "desktop-dev: killing stale listener(s) on :1430 (pid $stale)"
                    kill $stale 2>/dev/null || true
                    sleep 1
                    stale=$(lsof -ti tcp:1430 || true)
                    [ -z "$stale" ] || kill -9 $stale 2>/dev/null || true
                  fi
                  bun install --frozen-lockfile
                  cd desktop
                  cargo tauri dev --features ${desktopFeatures} "$@"
                '';
              }
              {
                category = "desktop";
                name = "desktop-build";
                help = "build the native desktop bundle (Mold.app, AppImage, or Nix package on NixOS)";
                command = ''
                  set -euo pipefail
                  ${desktopSetup}
                  if [ "$(uname -s)" = "Darwin" ] && [ -f .secrets/signing.env ]; then
                    # shellcheck disable=SC1091
                    source .secrets/signing.env
                  fi
                  bun install --frozen-lockfile
                  cd desktop
                  ${
                    if isLinux then
                      ''
                        if [ -x /usr/bin/xdg-open ]; then
                          export XDG_CACHE_HOME="''${MOLD_DESKTOP_CACHE_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}/mold-desktop}"
                          ../scripts/prepare-desktop-linuxdeploy.sh
                          cargo tauri build --features ${desktopFeatures} --bundles appimage "$@"
                        else
                          # Tauri's downloaded linuxdeploy tools require an FHS
                          # host. NixOS has a first-class native package instead.
                          cd ..
                          nix build .#mold-desktop "$@"
                        fi
                      ''
                    else
                      ''cargo tauri build --features ${desktopFeatures} --bundles app "$@"''
                  }
                '';
              }
              {
                category = "desktop";
                name = "desktop-release";
                help = "build, notarize, staple, and verify the Mold app + DMG";
                command =
                  if isDarwin then
                    ''
                      set -euo pipefail
                      ${desktopSetup}
                      if [ ! -f .secrets/signing.env ]; then
                        echo "missing .secrets/signing.env (see website/guide/desktop.md)" >&2
                        exit 1
                      fi
                      # shellcheck disable=SC1091
                      source .secrets/signing.env
                      for name in APPLE_SIGNING_IDENTITY APPLE_API_ISSUER APPLE_API_KEY APPLE_API_KEY_PATH; do
                        if [ -z "''${!name:-}" ]; then
                          echo "missing $name in .secrets/signing.env" >&2
                          exit 1
                        fi
                      done
                      bun install --frozen-lockfile
                      cd desktop
                      cargo tauri build --features ${desktopFeatures} --bundles app,dmg "$@"
                      cd ..
                      app="desktop/src-tauri/target/release/bundle/macos/Mold.app"
                      dmg=$(find desktop/src-tauri/target/release/bundle/dmg -maxdepth 1 -name '*.dmg' -print -quit)
                      scripts/notarize-desktop-dmg.sh "$dmg"
                      scripts/verify-desktop-release.sh "$app" "$dmg"
                    ''
                  else
                    ''
                      echo "desktop-release is the signed macOS distribution path; use desktop-build for a Linux AppImage" >&2
                      exit 1
                    '';
              }
              {
                category = "desktop";
                name = "desktop-check";
                help = "desktop CI gate: rustfmt, clippy -D warnings, vue-tsc, prettier";
                command = ''
                  set -euo pipefail
                  ${desktopSetup}
                  cargo fmt --manifest-path desktop/src-tauri/Cargo.toml -- --check
                  cargo clippy --manifest-path desktop/src-tauri/Cargo.toml --all-targets -- -D warnings
                  # `pulid` and `webp` are in every shipped desktop recipe, so
                  # their lints belong here rather than first on a release runner.
                  cargo clippy --manifest-path desktop/src-tauri/Cargo.toml --all-targets --features pulid,webp -- -D warnings
                  bun install --frozen-lockfile
                  cd desktop
                  bunx vue-tsc -b
                  bun run fmt:check
                '';
              }
              {
                category = "desktop";
                name = "desktop-test";
                help = "desktop tests: cargo test (CPU) + vitest";
                command = ''
                  set -euo pipefail
                  ${desktopSetup}
                  cargo test --manifest-path desktop/src-tauri/Cargo.toml "$@"
                  bun install --frozen-lockfile
                  cd desktop
                  bun run test
                '';
              }
              {
                category = "desktop";
                name = "desktop-ui";
                help = "frontend-only Vite dev server (pair with a running `serve`)";
                command = ''
                  set -euo pipefail
                  bun install --frozen-lockfile
                  cd desktop
                  bun run dev "$@"
                '';
              }
              {
                category = "desktop";
                name = "frontend-bun-lock";
                help = "regenerate the root bun.lock and bun.nix frontend dependency set";
                command = ''
                  set -euo pipefail
                  bun install
                  bun2nix -o bun.nix
                '';
              }
              {
                category = "docs";
                name = "understand-dashboard";
                help = "open the Understand Anything knowledge-graph dashboard";
                command = "./scripts/understand-dashboard.sh \"$@\"";
              }
              {
                category = "docs";
                name = "code-report";
                help = "generate the gitignored HTML code-metrics report";
                command = ''
                  repo_dir="''${PRJ_ROOT:-$(git rev-parse --show-toplevel)}"
                  exec "$repo_dir/scripts/code-report.sh" "$@"
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
            # The standalone iOS crate is Rust 2024 and has its own cargo-fmt CI gate.
            settings.formatter.rustfmt.excludes = [ "apps/mobile/src-tauri/**/*.rs" ];
          };
        };
    };
}
