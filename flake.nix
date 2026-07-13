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
    bun2nix = {
      url = "github:nix-community/bun2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
          };

          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;

          src = craneLib.path ./.;

          commonArgs = {
            inherit src;
            pname = "mold";
            version = "0.10.0";
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
            ];
          }
          // lib.optionalAttrs isLinux {
            CUDA_PATH = "${cudaToolkit}";
            CUDA_COMPUTE_CAP = cudaComputeCap;
            NIX_LDFLAGS = "-L${pkgs.cudaPackages.cuda_cudart}/lib/stubs";
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

          desktopFeature = if isLinux then "cuda" else "metal";

          gpuFeature =
            if isLinux then
              "cuda"
            else if isDarwin then
              "metal"
            else
              "";

          devProfile = "dev-fast";

          # Full shipping feature set used for release builds and feature coverage.
          releaseFeatures =
            if gpuFeature != "" then
              "${gpuFeature},preview,discord,expand,tui,webp,mp4,metrics,mdns"
            else
              "preview,discord,expand,tui,webp,mp4,metrics,mdns";

          # Shell completion generation only needs CLI shape, not GPU linkage.
          # Keep this CUDA-free so Linux sandbox builds can generate completion
          # scripts without loading the host-only NVIDIA driver library.
          completionFeatures = "preview,discord,expand,tui,webp,mp4,metrics,mdns";

          # Devshell defaults compile the full shipping feature set so that
          # `mold tui`, `mold discord`, WebP/MP4 output, Prometheus metrics,
          # and local prompt expansion are all available from the interactive
          # `mold`, `serve`, and `generate` commands without the user having
          # to know which features to flip. CI and `nix build` use the same
          # list via `releaseFeatures`, so there's a single feature matrix.
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

          # Web gallery SPA (Vue 3 + Vite + Tailwind v4). Built via bun2nix so
          # the `node_modules` cache is reproducibly derived from `web/bun.lock`.
          # Output layout: `$out/index.html` + `$out/assets/...` — consumed at
          # Rust compile time via `MOLD_WEB_DIST`, then embedded into the
          # `mold` binary by `rust-embed` (see `crates/mold-server/build.rs`).
          mold-web = pkgs.stdenv.mkDerivation {
            pname = "mold-web";
            version = "0.10.0";
            src = ./web;
            nativeBuildInputs = [ pkgs.bun2nix.hook ];
            bunDeps = pkgs.bun2nix.fetchBunDeps {
              bunNix = ./web/bun.nix;
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
              bun run build
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              cp -R dist/. $out/
              runHook postInstall
            '';
          };

          # Desktop app frontend (Vue SPA under desktop/), built like mold-web.
          mold-desktop-web = pkgs.stdenv.mkDerivation {
            pname = "mold-desktop-web";
            version = "0.16.0";
            src = ./desktop;
            nativeBuildInputs = [ pkgs.bun2nix.hook ];
            bunDeps = pkgs.bun2nix.fetchBunDeps {
              bunNix = ./desktop/bun.nix;
            };
            bunInstallFlags = [
              "--linker=isolated"
              "--backend=symlink"
            ];
            dontRunLifecycleScripts = true;
            buildPhase = ''
              runHook preBuild
              bun run build
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              cp -R dist/. $out/
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
              version = "0.16.0";
              src = craneLib.path ./.;
              cargoRoot = "desktop/src-tauri";
              buildAndTestSubdir = "desktop/src-tauri";
              cargoLock.lockFile = ./desktop/src-tauri/Cargo.lock;
              buildFeatures = [ desktopFeature ];

              MOLD_GIT_SHA = gitShortRev;
              MOLD_BUILD_DATE = gitDate;
              # The embedded engine serves the regular web SPA to browsers.
              MOLD_WEB_DIST = "${mold-web}";
              CUDA_PATH = lib.optionalString isLinux "${cudaToolkit}";
              CUDA_COMPUTE_CAP = lib.optionalString isLinux computeCap;
              NIX_LDFLAGS = lib.optionalString isLinux "-L${pkgs.cudaPackages.cuda_cudart}/lib/stubs";

              nativeBuildInputs = [
                pkgs.cargo-tauri.hook
                pkgs.pkg-config
                pkgs.nasm
                pkgs.makeBinaryWrapper
              ]
              ++ lib.optionals isLinux [
                pkgs.autoPatchelfHook
                pkgs.clang
                pkgs.cudaPackages.cuda_nvcc
                pkgs.lld
                pkgs.wrapGAppsHook3
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

              tauriBundleType = lib.optionalString isLinux "deb";
              tauriBuildFlags = lib.optionals isDarwin [
                "--bundles"
                "app"
              ];
              doCheck = false;

              postInstall = lib.optionalString isDarwin ''
                if [ -d "$out/Applications/Mold.app" ]; then
                  mkdir -p $out/bin
                  makeBinaryWrapper "$out/Applications/Mold.app/Contents/MacOS/mold-desktop" \
                    $out/bin/mold-desktop
                fi
              '';

              # A signed/notarized bundle must not load /nix/store dylibs
              # (dyld Team-ID rejection); libiconv links in via stdenv — point
              # it at the system copy and re-sign ad hoc.
              postFixup =
                lib.optionalString isDarwin ''
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
                ''
                + lib.optionalString isLinux ''
                  patchelf --add-rpath /run/opengl-driver/lib "$out/bin/mold-desktop"
                '';

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
                cargoExtraArgs = "-p mold-ai --features ${releaseFeatures}";
                postInstall =
                  if isLinux then
                    ''
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
            mold-sm120 = mkMold "120"; # Blackwell (RTX 50-series)
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
              pkgs.sccache
              pkgs.git
              pkgs.gh
              pkgs.jq
              pkgs.viu
              pkgs.mpv
              pkgs.cargo-llvm-cov
              pkgs.ffmpeg
              pkgs.imagemagick
              pkgs.bun
              pkgs.bun2nix
              pkgs.cargo-tauri
              pkgs.nodePackages.prettier
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
                value =
                  # /run/opengl-driver/lib MUST come before cuda_cudart/lib/stubs
                  # so the real libcuda.so (NVIDIA driver) is found before the
                  # stub placeholder. Without this, debug builds link against
                  # the stub and fail at runtime with CUDA_ERROR_STUB_LIBRARY.
                  "/run/opengl-driver/lib:"
                  + lib.makeLibraryPath (
                    desktopLinuxRuntimeInputs
                    ++ [
                      pkgs.stdenv.cc.cc.lib
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudaPackages.libcublas.lib
                      pkgs.cudaPackages.cuda_nvrtc.lib
                      pkgs.cudaPackages.libcurand.lib
                    ]
                  )
                  + ":${pkgs.cudaPackages.cuda_cudart}/lib/stubs";
              }
              {
                name = "LD_LIBRARY_PATH";
                value =
                  "/run/opengl-driver/lib:"
                  + lib.makeLibraryPath (
                    desktopLinuxRuntimeInputs
                    ++ [
                      pkgs.stdenv.cc.cc.lib
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudaPackages.libcublas.lib
                      pkgs.cudaPackages.cuda_nvrtc.lib
                      pkgs.cudaPackages.libcurand.lib
                    ]
                  );
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
                help = "run the same sequence CI runs: fmt-check, check, clippy, test";
                command = ''
                  set -euo pipefail
                  cargo fmt --all -- --check
                  cargo check --workspace
                  cargo clippy --workspace --all-targets -- -D warnings
                  cargo test --workspace
                  cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4,mdns
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
                  cd desktop
                  bun install --frozen-lockfile
                  cargo tauri dev --features ${desktopFeature} "$@"
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
                  cd desktop
                  bun install --frozen-lockfile
                  ${
                    if isLinux then
                      ''
                        if [ -x /usr/bin/xdg-open ]; then
                          cargo tauri build --features ${desktopFeature} --bundles appimage "$@"
                        else
                          # Tauri's downloaded linuxdeploy tools require an FHS
                          # host. NixOS has a first-class native package instead.
                          cd ..
                          nix build .#mold-desktop "$@"
                        fi
                      ''
                    else
                      ''cargo tauri build --features ${desktopFeature} --bundles app "$@"''
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
                      cd desktop
                      bun install --frozen-lockfile
                      cargo tauri build --features metal --bundles app,dmg "$@"
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
                  cd desktop
                  bun install --frozen-lockfile
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
                  cd desktop
                  bun install --frozen-lockfile
                  bun run test
                '';
              }
              {
                category = "desktop";
                name = "desktop-ui";
                help = "frontend-only Vite dev server (pair with a running `serve`)";
                command = ''
                  set -euo pipefail
                  cd desktop
                  bun install --frozen-lockfile
                  bun run dev "$@"
                '';
              }
              {
                category = "desktop";
                name = "desktop-bun-lock";
                help = "regenerate desktop/bun.nix from bun.lock (bun2nix)";
                command = ''
                  set -euo pipefail
                  cd desktop
                  bun install
                  bun2nix -o bun.nix
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
