# NixOS

mold provides a NixOS module for declarative server and Discord bot deployment.

## Flake Setup

Add mold to your flake inputs and import the module:

```nix
{
  inputs.mold.url = "github:utensils/mold";

  outputs = { self, nixpkgs, mold, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        mold.nixosModules.default
        ./mold.nix  # your mold config (see below)
      ];
    };
  };
}
```

## Minimal Configuration

```nix
{ inputs, system, ... }:
{
  services.mold = {
    enable = true;
    package = inputs.mold.packages.${system}.default;  # Ada / RTX 40-series
  };
}
```

This starts `mold serve` on port 7680 with sensible defaults, creates a `mold`
system user, and manages the data directory at `/var/lib/mold`.

## Full Configuration Example

```nix{37-43}
{ inputs, system, config, ... }:
{
  services.mold = {
    enable = true;

    # Package — must match your GPU architecture
    package = inputs.mold.packages.${system}.default;     # Ada (RTX 4090, sm_89)
    # package = inputs.mold.packages.${system}.mold-sm120; # Blackwell (RTX 5090, sm_120)

    # Advisory hint — emits a build warning if package doesn't match
    # cudaArch = "blackwell";

    # Server
    port = 7680;
    bindAddress = "0.0.0.0";
    logLevel = "info";         # trace, debug, info, warn, error
    openFirewall = false;      # set true to allow LAN access

    # Directories
    homeDir = "/var/lib/mold";           # MOLD_HOME
    # modelsDir = "/var/lib/mold/models"; # defaults to homeDir/models

    # Models
    defaultModel = "flux2-klein:q8";

    # Image persistence — save copies of all server-generated images
    # outputDir = "/srv/mold/gallery";

    # CORS — restrict to specific origin (null = permissive)
    # corsOrigin = "https://mysite.example.com";

    # HuggingFace auth — for gated model repos
    # Points to a file containing the token (e.g. agenix secret)
    hfTokenFile = config.age.secrets.hf-token.path;

    # Extra environment variables
    environment = {
      MOLD_EAGER = "1";        # keep all components loaded
      MOLD_T5_VARIANT = "q4";  # use Q4 T5 encoder
    };

    # Discord bot
    discord = {
      enable = true;
      # Must be an EnvironmentFile: MOLD_DISCORD_TOKEN=your-token-here
      tokenFile = config.age.secrets.discord-token.path;
      # moldHost = "http://localhost:7680";  # defaults to main server
      cooldownSeconds = 10;
      logLevel = "info";
    };
  };
}
```

## Module Options Reference

### Server Options

| Option         | Type        | Default             | Description                                        |
| -------------- | ----------- | ------------------- | -------------------------------------------------- |
| `enable`       | bool        | `false`             | Enable the mold server                             |
| `package`      | package     | —                   | The mold package (must set explicitly)             |
| `cudaArch`     | null/enum   | `null`              | `"ada"` or `"blackwell"` — advisory warning only   |
| `port`         | port        | `7680`              | HTTP server port                                   |
| `bindAddress`  | string      | `"0.0.0.0"`         | Address to bind                                    |
| `homeDir`      | string      | `"/var/lib/mold"`   | Base directory (MOLD_HOME)                         |
| `modelsDir`    | string      | `homeDir + /models` | Model storage directory                            |
| `logLevel`     | enum        | `"info"`            | Log level (trace/debug/info/warn/error)            |
| `corsOrigin`   | null/string | `null`              | CORS origin restriction (null = permissive)        |
| `openFirewall` | bool        | `false`             | Open firewall port                                 |
| `defaultModel` | null/string | `null`              | Default model name                                 |
| `outputDir`    | null/string | `null`              | Image output directory (default: `homeDir/output`) |
| `hfTokenFile`  | null/path   | `null`              | Path to file with HuggingFace token                |
| `environment`  | attrs       | `{}`                | Extra environment variables                        |

### Discord Bot Options

| Option                    | Type    | Default                        | Description                  |
| ------------------------- | ------- | ------------------------------ | ---------------------------- |
| `discord.enable`          | bool    | `false`                        | Enable Discord bot service   |
| `discord.package`         | package | `config.services.mold.package` | Package for the bot          |
| `discord.tokenFile`       | path    | —                              | File containing bot token    |
| `discord.moldHost`        | string  | `"http://localhost:{port}"`    | mold server URL              |
| `discord.cooldownSeconds` | int     | `10`                           | Per-user generation cooldown |
| `discord.logLevel`        | enum    | `"info"`                       | Bot log level                |

## What the Module Creates

- **System user** `mold:mold` with home at `homeDir`
- **Directories** via tmpfiles: `homeDir`, `modelsDir`, and `outputDir` (if set)
- **Systemd service** `mold.service` — runs `mold serve` with:
  - `LD_LIBRARY_PATH=/run/opengl-driver/lib` for NixOS CUDA driver access
  - `video` and `render` supplementary groups for GPU access
  - Hardened: `NoNewPrivileges`, `ProtectSystem=full`, `ProtectHome`,
    `PrivateTmp`
  - HuggingFace token loaded via `EnvironmentFile` (never in process env)
- **Systemd service** `mold-discord.service` (if `discord.enable`) — runs
  `mold discord`, depends on `mold.service`, further hardened with
  `ProtectSystem=strict` and `PrivateDevices` (no GPU needed)
- **Firewall rule** if `openFirewall = true`

## GPU Architecture

The module **cannot auto-select** the flake package — you must set `package` to
match your GPU:

| GPU                       | Package                                     |
| ------------------------- | ------------------------------------------- |
| RTX 40-series (Ada)       | `inputs.mold.packages.${system}.default`    |
| RTX 50-series (Blackwell) | `inputs.mold.packages.${system}.mold-sm120` |

Set `cudaArch = "blackwell"` as a reminder — it emits a build warning if you
forget to switch the package.

## Build Variants

::: code-group

```bash [Ada]
nix build github:utensils/mold
```

```bash [Blackwell]
nix build github:utensils/mold#mold-sm120
```

:::

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
| `docs-dev`      | Start VitePress docs dev server    |
| `docs-build`    | Build the documentation site       |
| `docs-fmt`      | Format docs with prettier          |
