- **`mold library download`.** Save prints' original files from any host:
  `mold library download NAME` (a unique prefix of the name works),
  `-o DIR/` or `-o FILE` or `-o -` for stdout, `--latest N` for the newest
  prints, and `--trash` for trashed ones. Files stream to `<name>.part` and are
  renamed when complete, existing files are never overwritten without
  `--force`, and a progress bar shows on a terminal. `mold library list` ends
  with a hint on a terminal. Agents no longer need to decode MCP base64 to hand
  over a file.
