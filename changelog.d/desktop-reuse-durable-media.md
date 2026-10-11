- **Desktop Reuse settings reattaches input media again.** Reusing a print whose
  host never pinned its inputs (anything made before durable source retention,
  or a role the archive did not keep) left a bytes-less descriptor behind —
  "Reattach first frame …" — because the retained-media path switched off the
  local stash and gallery-by-name restore. Retained authority is now the first
  answer and that fallback the second, matching the phone app, and Library no
  longer toasts "Reattach it" for a source Create is about to restore.
- **Web Reuse settings no longer blocks on an unretained print.** An
  `unavailable` inventory became a blocking "Reattach it before developing"
  error on Create even after the Library had restored the source from the
  local stash or gallery; it now defers to that fallback, which also runs for
  an archive that retained other roles but not the source, and the identity
  photo is looked back up in the stash as Create's own reuse path already did.
- **Reuse settings never strands Generate.** A reused print whose every copy
  sits on an unreachable machine no longer holds the desktop composer on
  "Restoring the original input files…", and a failed input restore now
  clears as soon as the input is attached by hand on desktop, web and the
  phone, instead of only after a New print or Reset.
