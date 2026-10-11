- **Desktop Reuse settings reattaches input media again.** Reusing a print whose
  host never pinned its inputs (anything made before durable source retention,
  or a role the archive did not keep) left a bytes-less descriptor behind —
  "Reattach first frame …" — because the retained-media path switched off the
  local stash and gallery-by-name restore. Retained authority is now the first
  answer and that fallback the second, matching the phone app, and Library no
  longer toasts "Reattach it" for a source Create is about to restore.
