---
paths:
  - "crates/mold-core/src/config*.rs"
  - "crates/mold-db/src/config_sync.rs"
  - "crates/mold-db/src/settings*.rs"
  - "crates/mold-cli/src/commands/config.rs"
  - "crates/mold-server/src/routes_config.rs"
---

# Config: two stores, one logical view

Moved from the root CLAUDE.md; loaded only when working under the paths above.

Two stores, one logical `Config` view:

| Surface                                                                                                                          | Owns                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `$MOLD_HOME/config.toml` (default `~/.mold/config.toml`; `$XDG_CONFIG_HOME/mold/home` is a bootstrap pointer file, never config) | Bootstrap: paths, ports, credentials, `[logging]`, `[runpod]`, per-model component paths |
| `mold.db` `settings` + `model_prefs`                                                                                             | User prefs: `expand.*`, `generate.*`, per-model defaults                                 |
| `MOLD_*` env vars                                                                                                                | Runtime override (highest precedence)                                                    |

Every `main()` calls `mold_db::config_sync::install_config_post_load_hook()`, which runs a one-shot idempotent `config.toml → DB` migration on first boot (renames original to `config.toml.migrated`) and overlays DB onto every `Config::load_or_default()`. Consumers still read `cfg.expand.*` unchanged.

`mold config set <key> <val>` routes by key prefix (`expand.*` → DB, `models_dir` → TOML). `mold config where <key>` prints the surface. `mold config list --json` tags each row `[db]` / `[file]` / `[env]`. Multi-profile: `settings` and `model_prefs` are keyed on `(profile, key)`; active profile resolves `MOLD_PROFILE` → `settings.profile.active` → `"default"`.
