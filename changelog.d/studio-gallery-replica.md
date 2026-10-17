- **Libraries follow only what changed.** The web Library, the desktop app
  (every connected machine and this device) and the phone now keep a per-host
  gallery replica that reads the server's change feed, so a refresh or a live
  update downloads only the prints that changed instead of the whole gallery.
  Gallery events from every machine trigger those pulls, the web and desktop
  polls stand down while a machine's live stream is up, an unchanged Library
  does no work at all (and the phone no longer rewrites its saved Library),
  and older servers keep the full listing they use today. Bulk actions on the
  web Library no longer slow down with library size.
