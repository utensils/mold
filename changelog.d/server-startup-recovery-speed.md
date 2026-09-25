- **The server answers in well under a second on a large gallery.** Startup
  gallery recovery read the multi-megabyte archive checkpoint one byte per
  system call, parsed it four times, and loaded the whole authority twice
  before binding — about 7.4 s on a 3,716-print gallery on an external disk,
  now about 0.3 s. A gallery whose files were copied or restored (so their
  inodes moved) was also re-hashed in full on every boot, because the
  re-verified file facts were never saved; they are now committed once, so the
  next start is stat-only again. Each recovery step logs its `elapsed_ms` at
  debug level.
