- **`mold library download`.** Save prints' original files from any host:
  `mold library download NAME` (a unique prefix of the name works),
  `-o DIR/` or `-o FILE` or `-o -` for stdout, `--latest N` for the newest
  prints, and `--trash` for trashed ones. Files stream to a new hidden partial file and are moved
  into place when complete, existing files (even ones that appear during the
  download) are never overwritten without `--force`, and a progress bar shows on a terminal. `mold library list` ends
  with a hint on a terminal. Agents no longer need to decode MCP base64 to hand
  over a file.
