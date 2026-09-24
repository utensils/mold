# Sync All to This Mac

## Intent

Give a Mac user one visible Library toolbar action, **Sync All to This Mac**, to copy every reachable remote Library print and collection into the embedded local Library. The action ignores the current search, shelf, machine filter, and selection. It is additive: it never deletes or overwrites local work. Repeating it should skip copies already present and fill gaps.

## Product flow

1. Place a compact toolbar button beside the existing Library controls, with a download-to-Mac symbol and a tooltip that names images, clips, 3D prints, and collections. A click starts immediately; the existing progress bar and Stop control show the work and its end-of-run summary.
2. Disable it while a save is running or there are no remote machines. If This Mac's engine is stopped, show the current actionable explanation.
3. At start, fetch fresh complete gallery and collection lists from each remote host, regardless of UI filters. Report unreachable hosts without discarding the work from reachable ones.
4. Create missing local collections, including empty ones; resolve memberships by collection name/slug on the destination, never by a source host's ID. Leave existing local collection names and unrelated members intact.
5. Copy every supported gallery media kind (still images, animated images, MP4 clips, GLB 3D prints, WAV audio where present) with original bytes, recipe, timestamp and synthetic flag. The server import route already accepts this set; fix its incomplete refusal text. Download/upload to temporary files in bounded chunks so multi-gigabyte clips do not fill app memory.
6. Preserve user-owned title, favourite and tags on newly copied prints; never overwrite organization on a print already local. Keep a pending organization journal from immediately before import until these mutations succeed, so a relaunch or transient mutation failure can finish filing a copy without changing an unrelated existing local print. Use a stable per-source collision name that fits the filesystem limit. Compare both bytes and the full recipe (including newer, unknown metadata fields), timestamp and synthetic flag before treating a same-named local file as present. Persist a source-to-local copy record keyed by source host and media version, then verify its destination against the fresh local listing so later syncs avoid downloading unchanged media.
7. Reuse the existing idempotent import, bounded concurrency, progress, cancellation, event suppression, and compact error report. Serialize batches with large clips and check available staging space before transfer. Refresh the local Library once at the end. Scope existing selection-based Save to This Mac to its present behavior.

## Verification

- Test an unfiltered run across multiple remote hosts with pictures, clip, GLB, and a collection containing them plus an empty collection; test rerun skips without redownloading unchanged media, same-byte prints with different recipes stay distinct, long collision names fit the filesystem, local-only rows stay untouched, and an unreachable host is reported.
- Test file-backed framing and authenticated download failure; rely on existing server format-validation coverage and verify its refusal text matches the accepted set.
- Run native macOS tests/build and relevant Rust server tests. Exercise the toolbar in the app if feasible, obtain independent subagent review of the final diff, address findings, refresh Understand Anything graph, then open one PR.
