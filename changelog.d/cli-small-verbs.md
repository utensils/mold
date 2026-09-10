- **`mold library source-media` recovers what a print was made from.** The
  serving host is the only authority on the conditioning media it retained, so
  the new verb asks it, prints one sentence per availability state, and
  downloads a named member to a file or to stdout. A print that predates
  retention reads as exactly that, never as damage.
- **`mold trash delete <FILENAME>...` removes named prints permanently.**
  The per-file counterpart of `mold trash empty`: it works on live and trashed
  prints alike, and confirms unless `--yes`.
- **`mold jobs amend <ID> --script edited.toml` edits a sequence's stages.**
  Amend replaces the whole stage list, so it takes an edited `mold.chain.v1`
  script rather than per-stage flags; `--fps`, `--seed`, `--steps`,
  `--guidance`, `--strength`, `--motion-tail` and `--audio`/`--no-audio`
  override the script's chain block, `--dry-run` shows what would be sent, and
  a change to the model, size or container is refused by name. The command
  reports how many leading stages kept their rendered clips.
- **`mold run --no-save` keeps one render out of the Library.** The host still
  publishes the print and moves it straight to trash, so `mold trash restore`
  recovers it until retention sweeps it.
- **Shell completion stopped drifting.** The zsh wrapper's file-path flag list
  is now generated from the clap tree instead of a hand-kept list that had
  fallen thirteen flags behind (`--video`, `--audio-file`, `--extend`,
  `--first-frame`, …) while wrongly claiming `--control-model`, which now
  completes ControlNet adapter names. `mold library export --format` offers its
  containers, `--profile` offers the profiles in `mold.db`, and
  `mold skill show` offers the bundle's files.
- **The agent skill and the docs are checked for omissions, not just errors.**
  Every user-facing command must now appear in a tested example, which added
  `mold run --script`, `mold chain validate`, the full `library`, `jobs`,
  `queue` and `trash` verb sets and a positive expand/remix example to the
  skill. Every `bash` example in `README.md` and the website is parsed against
  the real CLI, which found `--negative` (the flag is `--negative-prompt`) and
  a repeated `--lora-scale` that the CLI refuses.
