- **Windows nightlies finish even while `main` stays busy.** Windows CLI and
  desktop artifacts now build in a dedicated non-cancellable workflow: the
  active build completes while GitHub keeps only the newest pending commit.
  Rolling publication preserves the shared checksums, tagged releases retain
  their existing immutable build, and the final standalone desktop executable
  is explicitly re-signed after Tauri finishes bundling it.
