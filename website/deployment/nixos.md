# NixOS

mold provides a NixOS module for declarative server deployment.

## Flake Setup

```nix
{
  inputs.mold.url = "github:utensils/mold";

  outputs = { self, nixpkgs, mold, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        mold.nixosModules.default
        {
          services.mold = {
            enable = true;
            port = 7680;
          };
        }
      ];
    };
  };
}
```

## Build Variants

```bash
# Ada / RTX 40-series (default, sm_89)
nix build github:utensils/mold

# Blackwell / RTX 50-series (sm_120)
nix build github:utensils/mold#mold-sm120
```

## Development Shell

```bash
nix develop github:utensils/mold
```

The devshell includes Rust toolchain, CUDA toolkit, and convenience commands:

| Command         | Description                        |
| --------------- | ---------------------------------- |
| `build`         | `cargo build` (debug)              |
| `build-release` | `cargo build --release`            |
| `build-server`  | Build with GPU + preview + discord |
| `serve`         | Start the mold server              |
| `generate`      | Generate an image                  |
| `mold`          | Run any mold CLI command           |
| `check`         | `cargo check`                      |
| `clippy`        | `cargo clippy`                     |
| `fmt`           | `cargo fmt`                        |
| `run-tests`     | `cargo test`                       |
| `coverage`      | Test coverage report               |

## CUDA on NixOS

The systemd service needs access to the NVIDIA driver:

```ini
Environment=LD_LIBRARY_PATH=/run/opengl-driver/lib
```

This is handled automatically by the NixOS module.
