- **Removed scene-by-scene clip authoring from the apps.** The desktop
  `Simple | Scenes` strip and its timeline, the web and iPhone
  `One shot | Sequence` output, the TUI chain composer, and the Discord
  `/sequence` command are gone. Making a clip is now one flow: a prompt, the
  model controls, and the length slider — including long clips, which the host
  still renders as chained clips stitched into one video exactly as before
  ([#1614](https://github.com/utensils/mold/issues/1614)).
- **Sequence prints stay in your library.** Existing scene-authored clips keep
  their thumbnails, provenance, downloads, and exports; **Use these settings
  again** now restores a plain one-shot clip built from the first scene's
  prompt instead of reopening a timeline.
- **Scripted sequences are unchanged.** `mold run --script shot.toml`, repeated
  `--prompt`, `--frames-per-clip`, `mold chain validate`, `mold jobs`, and the
  `/api/chain-jobs` endpoints all still work for anyone driving sequences from
  the CLI or the API.
