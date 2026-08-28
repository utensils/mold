- **One durable-print vocabulary on web, desktop, and iPhone.** A queued print's
  stage, hold, cancellation, failure prose, and "outcome unknown" copy now come
  from one shared policy, so the three surfaces no longer disagree ("Rendering"
  and "Accepted" are gone; every surface says "Developing" and "Queued"). A
  print the host reported complete without publishing a file is shown as a
  failure instead of a blank result the web retried forever and the iPhone
  silently skipped for Photos auto-save; a print whose server instance was
  replaced or whose record the host no longer has settles as "Outcome unknown"
  rather than lingering as interrupted work. Clients also stop re-reading the
  queue on `gallery_added` and `job_ended` hints, since every settlement is
  followed by the server's commit hint.
