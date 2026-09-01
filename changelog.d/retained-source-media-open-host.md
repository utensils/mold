- **Reuse settings no longer demands an API key from an open host.** On a
  server started without `MOLD_API_KEY`, every retained source-media lookup
  answered "Connect this machine with an API key" for prints it never looked
  at, so restoring a print's original source never worked on a default host
  and a plain text-to-image print raised the error on every surface. The
  routes now ask whether the caller is authorized — open on a keyless host,
  exactly like the rest of the API — clients ask only about prints that
  recorded conditioning media, a host that genuinely refuses the probe still
  gets the API-key disclosure, and the desktop Lightbox's primary Reuse button
  attaches retained source media the way its right-click item already did.
