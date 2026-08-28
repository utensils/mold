# Library CLI parity

## Goal

Give scripts and terminal users the same single-host Library organization
authority already exposed by the server: browse existing prints, inspect and
preview media, edit titles/favorites/tags, manage collection membership and
collection metadata, and move live prints into the recoverable trash.

`mold library` always targets the server selected by `MOLD_HOST` and
`MOLD_API_KEY`. It has no direct-filesystem fallback. The existing
`mold trash` family remains the authority for inspecting, restoring, sweeping,
and emptying that host's trash.

## Command surface

```text
mold library list [--query TEXT] [--tag TAG] [--collection NAME-OR-SLUG]
                  [--favorite] [--format FORMAT] [--limit N] [--offset N]
                  [--json]
mold library show FILENAME [--json | --preview]
mold library grid [--host URL] [--local]

mold library title FILENAME TEXT
mold library title FILENAME --clear
mold library favorite FILENAME...
mold library unfavorite FILENAME...

mold library tag list [--json]
mold library tag add FILENAME... --tag TAG...
mold library tag remove FILENAME... --tag TAG...
mold library tag rename OLD NEW
mold library tag delete TAG [--yes]

mold library collection list [--json]
mold library collection show NAME-OR-SLUG [--json]
mold library collection create NAME [--description TEXT]
mold library collection update NAME-OR-SLUG
                               [--name TEXT]
                               [--description TEXT | --clear-description]
                               [--cover FILENAME | --clear-cover]
                               [--hidden | --visible]
mold library collection delete NAME-OR-SLUG [--yes]
mold library collection add NAME-OR-SLUG FILENAME...
mold library collection remove NAME-OR-SLUG FILENAME...

mold library trash FILENAME...
```

Collection references resolve in this order: exact host-local id, exact slug,
then case-insensitive exact display name. Ambiguous or missing references fail
without mutation and print the available collection names.

## Behavior and safety

1. Listing is client-filtered from `GET /api/gallery` so it works with every
   server that already returns enriched Library rows. Multiple `--tag` filters
   use AND semantics. `--collection` is resolved before filtering. Filtering
   precedes offset/limit, the stable order is timestamp-descending then
   filename-ascending, the default limit is 50, and the maximum is 1,000.
   Human and JSON output return the identical selected page.
2. `show --preview` fetches the original still. Videos use the server's animated
   preview when available, then its thumbnail. Preview output is emitted only
   to a TTY and reuses the CLI's existing viuer/Ghostty renderer; builds without
   `preview` give the existing actionable feature warning.
3. `grid` launches the existing Mold TUI directly in Library. The TUI already
   probes Kitty, Sixel, and iTerm2-compatible protocols before raw mode and has
   a text-safe fallback. Its strict launch options set the initial workspace to
   Library and turn an unreachable explicit `--host` or `MOLD_HOST` into an
   error instead of silently switching to local files. The interactive grid is
   intentionally the existing merged-host TUI Library; `--local` explicitly
   requests its local-only path. Builds without `tui` fail with a rebuild
   instruction.
4. Every mutation first reads server capabilities. `gallery.organize=false`
   fails with an upgrade-or-enable-metadata diagnostic. When
   `gallery.bulk_mutations=true`, multi-print tag/favorite edits use replay-safe
   `POST /api/gallery/mutations` with a fresh UUID; otherwise they fall back to
   `POST /api/gallery/organize`. Title remains a single-print patch. Collection
   CRUD/membership use the existing collection endpoints. Inputs run through
   shared Rust validators before HTTP.
5. Recoverable removal is `mold library trash` and requires an advertised,
   enabled `gallery.trash`; an older host is never allowed to reinterpret it as
   a hard delete. Existing `mold trash empty` remains the only CLI permanent
   deletion in this slice. Targeted permanent deletion is deferred until the
   server offers a race-free, trashed-only endpoint. Global tag deletion and
   collection deletion confirm because they remove shared organization state;
   deleting a collection never deletes its prints.

Machine-readable output is pure: `--json` conflicts with `--preview`, prompts
and diagnostics stay on stderr, and JSON stdout contains no ANSI sequences.

## Implementation slices

1. Add missing `MoldClient` helpers for collection detail and replay-safe
   gallery mutation, and path-segment encode still/preview/thumbnail filenames.
2. Add Clap enums, help/examples, parsers, validation, filtering, tables, JSON,
   confirmation, and command dispatch in a focused `commands/library.rs`.
3. Extract the existing terminal preview entry point for reuse, and add a TUI
   launch option that starts on Library without duplicating its grid.
4. Add parser/unit tests plus black-box WireMock coverage for request method,
   path, payload, filtering, collection resolution, preview fallback selection,
   capability fallback, strict grid host behavior, confirmations, partial
   multi-file errors, and error propagation. Include filenames containing
   spaces, `#`, `%`, and Unicode.

## Acceptance criteria

- Every organization mutation currently available through the server has an
  ordinary CLI route for existing prints.
- Human and JSON listings select the same rows in the same order.
- No mutating command silently falls back from an unreachable host to local
  files, and no destructive command skips its documented confirmation.
- Still and video preview selection is deterministic and testable without a
  graphics-capable CI terminal; the existing renderer remains the only inline
  rendering implementation.
- JSON stdout is parseable and contains no terminal control bytes, including
  when preview flags or server errors are involved.
- `mold library grid` lands on Library and retains the current TUI's protocol
  detection, multi-host browsing, keyboard navigation, and recoverable delete.
- Default-feature and shipping-feature builds compile; focused CLI/core/TUI
  tests, formatting, and Clippy pass.
