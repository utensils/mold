# NixOS

mold provides a NixOS module for declarative server and Discord bot deployment.

The flake follows a source revision, not GitHub's prebuilt binary release
channel. Pin the input revision for reproducible deployments. Nix-built cloud
clients default to mutable `latest*` container images because a Cargo package
version is not evidence that a corresponding stable image was published; only
official stable release builds embed a release version and resolve its exact
published image digest.

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

## Binary Cache

Direct `nix build github:utensils/mold` commands use Mold's signed public
binary cache automatically after Nix accepts the flake configuration. When
Mold is an input of another flake, that input's `nixConfig` is not inherited.
Configure the consuming NixOS system explicitly so deployments can substitute
the exact CI-built package instead of compiling Mold locally:

```nix
{
  nix.settings = {
    extra-substituters = [ "https://mold.cachix.org" ];
    extra-trusted-public-keys = [
      "mold.cachix.org-1:9HBc/bEXDdpbxMjOwpaIDpjZqBh9JYg0h5Fipm+D8m4="
    ];
  };
}
```

Only store paths signed by that key are accepted. The cache is an optimization:
Nix falls back to a local source build when CI has not published the exact
revision and package variant. CI currently publishes the default Ada/sm89
package; the other architecture variants fall back to a local build.

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

::: tip Web gallery is bundled

Since v0.8.1 the Vue 3 gallery SPA is embedded directly into the `mold` binary
at compile time -- visiting `http://<host>:7680/` opens the gallery with no
extra configuration. Earlier versions required staging `web/dist/` into
`~/.mold/web` or pointing `MOLD_WEB_DIR` at a built SPA. That override still
works for SPA hot-iteration without recompiling Rust.

:::

## Full Configuration Example

```nix{37-43}
{ inputs, system, config, ... }:
{
  services.mold = {
    enable = true;

    # Package -- must match your GPU architecture
    package = inputs.mold.packages.${system}.default;     # Ada (RTX 4090, sm_89)
    # package = inputs.mold.packages.${system}.mold-sm86;  # RTX 3090/A40, sm_86
    # package = inputs.mold.packages.${system}.mold-sm100; # B200/B300, sm_100
    # package = inputs.mold.packages.${system}.mold-sm120; # RTX 5090, sm_120

    # Advisory hint -- emits a build warning if package doesn't match
    # cudaArch = "blackwell";

    # Server
    port = 7680;
    bindAddress = "0.0.0.0";
    logLevel = "info";         # trace, debug, info, warn, error
    openFirewall = false;      # set true to allow LAN access
    # mdns = true;             # advertise + browse _mold._tcp (set false for MOLD_MDNS=0)

    # Directories
    homeDir = "/var/lib/mold";           # MOLD_HOME
    # modelsDir = "/var/lib/mold/models"; # defaults to homeDir/models

    # Models
    defaultModel = "flux2-klein:q8";

    # Multi-GPU -- pin the server to specific cards (null = use all visible)
    # gpus = "0,1";
    # queueSize = 200; # max queued jobs; overflow returns HTTP 503

    # Stop budget -- the running generation is aborted at its next checkpoint
    # and requeued; queued work is retained and replayed on the next start.
    # shutdown.abortSeconds = 45;

    # Image persistence -- save copies of all server-generated images
    # outputDir = "/srv/mold/gallery";

    # CORS -- restrict to specific origin (null = permissive)
    # corsOrigin = "https://mysite.example.com";

    # Catalog auth defaults -- users can override these in web Settings
    # Points to files containing tokens (e.g. agenix secrets)
    hfTokenFile = config.age.secrets.hf-token.path;
    civitaiTokenFile = config.age.secrets.civitai-token.path;

    # API key authentication -- file with one key per line (e.g. agenix secret)
    # When set, all API requests require an X-Api-Key header
    # apiKeyFile = config.age.secrets.mold-api-key.path;

    # Rate limiting -- per-IP, generation endpoints at configured rate, reads at 10x
    # rateLimit = "10/min";
    # rateLimitBurst = 20;

    # Extra environment variables
    environment = {
      MOLD_EAGER = "1";        # keep all components loaded
      MOLD_T5_VARIANT = "q4";  # use Q4 T5 encoder
      # MOLD_THUMBNAIL_WARMUP = "1"; # opt in to startup gallery thumbnail warmup
    };

    # Discord bot
    discord = {
      enable = true;
      # Must be an EnvironmentFile: MOLD_DISCORD_TOKEN=your-token-here
      tokenFile = config.age.secrets.discord-token.path;
      # moldHost = "http://localhost:7680";  # defaults to main server
      cooldownSeconds = 10;
      # allowedRoles = "artist, 1234567890";  # restrict to specific roles
      # dailyQuota = 20;                       # max generations per user per day
      logLevel = "info";
    };
  };
}
```

## Module Options Reference

### Server Options

| Option                  | Type        | Default             | Description                                                          |
| ----------------------- | ----------- | ------------------- | -------------------------------------------------------------------- |
| `enable`                | bool        | `false`             | Enable the mold server                                               |
| `package`               | package     | --                  | The mold package (must set explicitly)                               |
| `cudaArch`              | null/enum   | `null`              | See the exact advisory architecture-to-package mapping below         |
| `port`                  | port        | `7680`              | HTTP server port                                                     |
| `bindAddress`           | string      | `"0.0.0.0"`         | Address to bind                                                      |
| `homeDir`               | string      | `"/var/lib/mold"`   | Base directory (MOLD_HOME)                                           |
| `modelsDir`             | string      | `homeDir + /models` | Model storage directory                                              |
| `logLevel`              | enum        | `"info"`            | Log level (trace/debug/info/warn/error)                              |
| `corsOrigin`            | null/string | `null`              | CORS origin restriction (null = permissive)                          |
| `openFirewall`          | bool        | `false`             | Open firewall port (also UDP 5353 when `mdns` is on)                 |
| `mdns`                  | bool        | `true`              | Advertise and browse `_mold._tcp`; `false` sets `MOLD_MDNS=0`        |
| `defaultModel`          | null/string | `null`              | Default model name                                                   |
| `gpus`                  | null/string | `null`              | `all`, `none`, ordinals, or stable CUDA/Metal/NVIDIA UUID IDs        |
| `queueSize`             | null/int    | `null`              | Max queued generation jobs (null = default 200)                      |
| `shutdown.abortSeconds` | int         | `45`                | Seconds the server waits for its GPU workers on stop (see below)     |
| `outputDir`             | null/string | `null`              | Image output directory (default: `homeDir/output`)                   |
| `hfTokenFile`           | null/path   | `null`              | Path to overridable default HuggingFace token                        |
| `civitaiTokenFile`      | null/path   | `null`              | Path to overridable default Civitai token                            |
| `apiKeyFile`            | null/path   | `null`              | Path to file with API key(s) for authentication (e.g. agenix secret) |
| `rateLimit`             | null/string | `null`              | Per-IP rate limit (e.g. `"10/min"`)                                  |
| `rateLimitBurst`        | null/int    | `null`              | Override burst allowance (defaults to 2x rate)                       |
| `logToFile`             | bool        | `false`             | Enable file logging (in addition to journal)                         |
| `logDir`                | string      | `homeDir + /logs`   | Directory for log files when `logToFile` is enabled                  |
| `logRetentionDays`      | int         | `7`                 | Days to retain rotated log files                                     |
| `environment`           | attrs       | `{}`                | Extra environment variables                                          |

`cudaArch` does not select a package automatically. Set `package` to the
matching flake output:

- `"ampere"` → `packages.${system}.mold-sm86` (RTX 3090/A40, sm_86)
- `"ada"` → `packages.${system}.mold` (RTX 40-series, sm_89)
- `"blackwell-datacenter"` → `packages.${system}.mold-sm100` (B200/B300, sm_100)
- `"blackwell"` → `packages.${system}.mold-sm120` (RTX 50-series, sm_120)

### Restarts and the durable queue

Queued generations are recorded in `mold.db` and replayed automatically after a
restart, under their original job ids, so `systemctl restart mold` during a busy
queue no longer loses work. `shutdown.abortSeconds` (default 45) is a hard
deadline on the whole shutdown, not just a wait: the running generation is
aborted at its next inference checkpoint and requeued, and if shutdown overruns
the server ends the process itself (status 0 for an ordinary stop, 1 after a
fatal CUDA error so `Restart=on-failure` brings it back). A cold model load is
not interruptible, so without that the unit would sit until systemd SIGKILLed
it.

The unit's `TimeoutStopSec` is **derived** from that option -- `abortSeconds + 60`
-- so systemd is never the component that decides. Do not set
`TimeoutStopSec=infinity` through `environment` or an override: with a durable
queue the correct response to a wedged worker is to exit and replay, not to hang
the deploy. Raising `abortSeconds` past a cold model load is likewise the wrong
lever; the job comes back either way.

A job that repeatedly fails to finish is **held** rather than retried forever.
Held rows appear in `GET /api/queue` with `state: "held"` and a reason and are
never started automatically. See
[Durable queue and shutdown](/guide/configuration#durable-queue-and-shutdown)
for the environment knobs behind this.

### Monitoring

Nix builds include the `metrics` feature. The server exposes `GET /metrics` in
Prometheus text exposition format (HTTP request rates, generation duration, queue
depth, GPU memory, uptime). The endpoint is excluded from auth and rate limiting,
so Prometheus/Grafana Agent can scrape it without an API key.

### Discord Bot Options

| Option                    | Type    | Default                        | Description                                           |
| ------------------------- | ------- | ------------------------------ | ----------------------------------------------------- |
| `discord.enable`          | bool    | `false`                        | Enable Discord bot service                            |
| `discord.package`         | package | `config.services.mold.package` | Package for the bot                                   |
| `discord.tokenFile`       | path    | --                             | File containing bot token                             |
| `discord.moldHost`        | string  | `"http://localhost:{port}"`    | mold server URL                                       |
| `discord.cooldownSeconds` | int     | `10`                           | Per-user generation cooldown                          |
| `discord.allowedRoles`    | string? | `null`                         | Comma-separated role names/IDs (`null` = all)         |
| `discord.dailyQuota`      | int?    | `null`                         | Max generations per user per day (`null` = unlimited) |
| `discord.logLevel`        | enum    | `"info"`                       | Bot log level                                         |
| `discord.environment`     | attrs   | `{}`                           | Extra environment variables for the Discord bot       |

## What the Module Creates

- **System user** `mold:mold` with home at `homeDir`
- **Directories** via tmpfiles: `homeDir`, `modelsDir`, and `outputDir` (if set)
- **Systemd service** `mold.service` -- runs `mold serve` with:
  - `video` and `render` supplementary groups for GPU access
  - Hardened: `NoNewPrivileges`, `ProtectSystem=full`, `ProtectHome`,
    `PrivateTmp`
  - HuggingFace token loaded via `EnvironmentFile` (never in process env)
- **Systemd service** `mold-discord.service` (if `discord.enable`) -- runs
  `mold discord`, depends on `mold.service`, further hardened with
  `ProtectSystem=strict` and `PrivateDevices` (no GPU needed)
- **Firewall rule** if `openFirewall = true`

## GPU Architecture

The module **cannot auto-select** the flake package -- you must set `package` to
match your GPU:

| GPU                                | Package                                     |
| ---------------------------------- | ------------------------------------------- |
| RTX 3090 / A40 (Ampere)            | `inputs.mold.packages.${system}.mold-sm86`  |
| RTX 40-series (Ada)                | `inputs.mold.packages.${system}.mold`       |
| B200 / B300 (datacenter Blackwell) | `inputs.mold.packages.${system}.mold-sm100` |
| RTX 50-series (consumer Blackwell) | `inputs.mold.packages.${system}.mold-sm120` |

Set `cudaArch` to the matching `ampere`, `ada`, `blackwell-datacenter`, or
`blackwell` value. This is an advisory consistency check: it emits a build
warning if the package's Mold CUDA capability metadata does not match, but
never switches the package itself. All four official package variants carry
that metadata; custom packages without it warn rather than being assumed safe.

## Build Variants

::: code-group

```bash [Ada]
nix build github:utensils/mold
```

```bash [RTX 3090 / A40]
nix build github:utensils/mold#mold-sm86
```

```bash [B200 / B300]
nix build github:utensils/mold#mold-sm100
```

```bash [RTX 50-series]
nix build github:utensils/mold#mold-sm120
```

:::

B200/B300 support is simulated, not hardware-qualified. Hosted release CI
builds the sm_100 server package alongside sm_86 and the sm_86 desktop output;
real B200 qualification remains deferred. There is intentionally no sm_100
desktop package.
GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.
The current Linux flake outputs are x86_64 and must not be selected for Grace
systems.

## Development Shell

```bash
nix develop github:utensils/mold
```

The devshell includes Rust toolchain, CUDA toolkit, and convenience commands:

| Command           | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `build`           | Fast local `mold` build (`dev-fast`) with embedded web UI |
| `build-workspace` | `cargo build` (debug, all crates)                         |
| `build-release`   | Shipping release build with the full feature set          |
| `build-server`    | Fast local server build with GPU + preview + expand       |
| `serve`           | Start the mold server                                     |
| `generate`        | Generate an image                                         |
| `mold`            | Run any mold CLI command                                  |
| `check`           | `cargo check`                                             |
| `clippy`          | `cargo clippy`                                            |
| `fmt`             | `cargo fmt`                                               |
| `run-tests`       | `cargo test`                                              |
| `coverage`        | Test coverage report                                      |
| `docs-dev`        | Start VitePress docs dev server                           |
| `docs-build`      | Build the documentation site                              |
| `docs-fmt`        | Format docs with prettier                                 |
