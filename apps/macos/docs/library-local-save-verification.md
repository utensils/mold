# Native Library local save verification

## Automated checks

- `make -C apps/macos test`: 944 MoldClient tests, 7 MoldStyle tests, and 788 app tests passed after the metadata and collision fixes.
- `make -C apps/macos lint`: architecture, color, size, type size, and accessibility checks passed.
- `git diff --check`: passed.

## Disposable macOS UAT

Built the app with the existing embedded engine archive on an isolated `MOLD_HOME`. A separate Mold 0.29.0 server with its own temporary home held two PNG pictures. The app’s This Mac engine answered as Mold 0.30.0.

1. The Machine menu listed All Machines, This Mac, and Remote. Filtering to Remote showed two pictures and put a visible Remote token in search.
2. Command-A selected the two visible pictures. Right-click offered both “Save 2 Copies…” and the distinct “Save 2 to This Mac’s Library.” The latter reported “Saved 2 of 2 pictures to This Mac’s Library.”
3. Filtering to This Mac showed two pictures. SHA-256 hashes matched their remote originals byte for byte. SQLite gallery rows matched for filename, model, prompt, seed, steps, guidance, dimensions, and `metadata_synthetic`.
4. Filtering back to Remote and choosing “Move to Trash on Remote” left the remote live gallery empty with two rows in its recoverable trash. The local Library still showed both pictures and retained their files and hashes.

5. On a clean build, the Machine toolbar displayed “All Machines” and “Remote” by name. A third remote PNG with prompt “a copper fox in moonlight”, model `flux-dev:q8`, seed `4242`, 28 steps, guidance `3.5`, and a 2023 timestamp saved into a fresh This Mac gallery. Its SHA-256, file modification time, and those recipe fields matched the remote row. Saving it a second time left one local gallery row and one local file.

The native import descriptor test also pins an original recipe with a nonzero seed and confirms its timestamp and media bytes travel unchanged.

Final review found that the typed Swift recipe omits newer Rust fields. The import now uses the source listing's raw metadata JSON, and fetches that listing immediately before saving. A transport-backed regression test verifies preservation of batch and true-CFG fields, an unknown nested field, and `metadata_synthetic` through an optimistic favorite edit. A separate native test covers a name collision followed by a successful second import in the same selection.
