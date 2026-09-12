- **A render's fixed server overhead is gone from the client's wall clock.** A
  measured request timeline spent ~13 s of a 16.6 s `flux2-klein:q8` render on
  work that had nothing to do with the picture. The RAM/RSS sampler — on the
  1 Hz telemetry tick, on both memory watchdogs, and four times per job — built
  a fresh `sysinfo::System` with a process table on every call, which walks all
  of `/proc`; it now keeps one memory-only system and reads
  `/proc/self/statm`. The ordinary generation's memory watchdog polled a flag
  behind a one-second sleep, so finishing a render waited out the interval;
  it stops on a channel, and its heartbeat only speaks up when RSS has actually
  moved. NVML was re-initialized on every telemetry tick and every hot-cache
  admission; one handle now serves the process and is replaced only when the
  driver invalidates it.
- **PNG encoding is fast by default.** Saving a 1024² still at zlib level 6 was
  measured at ~1.0 s with the GPU idle. mold now encodes with fdeflate's
  PNG-tuned ultra-fast deflate: on a 512² photograph 1.7 ms against 59.0 ms.
  The file gets bigger — 6 % on that photograph, and 6–11 % across the campaign's
  measured prints (a 512² SD1.5 still 316,759 → 335,155 B, a 1024² SDXL still
  1,479,585 → 1,636,155 B). PNG is lossless under both settings, so no pixel
  changes and the raw-pixel hash is unmoved; `MOLD_PNG_ENCODING=balanced`
  restores the smaller files. The
  post-generation `malloc_trim(0)` (another ~0.8 s) also moved after the print
  is saved and the completion is sent.
- **The CLI no longer carries a finished render back as base64.** A streaming
  completion encoded the whole picture into the SSE frame, which the client
  then decoded — while the identical bytes sat in the host's gallery. Where a
  server advertises the new `gallery.persists_outputs` capability and the print
  is being saved, `mold run` asks for `X-Mold-SSE-Payload: metadata-only` and
  fetches the file instead. Older servers, hosts with the output directory
  disabled, and clips, audio and meshes (whose completions carry a thumbnail or
  poster the gallery route does not serve) keep the inline payload exactly as
  before. The client's SSE reader also stopped rescanning its whole buffer on
  every chunk, and stopped turning a multi-byte character split across a chunk
  boundary into replacement characters.
- **Publishing a print no longer costs more as the library grows.** The gallery
  archive authority rewrote its whole index three times per commit — a
  write-ahead copy, the checkpoint, and a backup — so saving one picture into a
  library of ten thousand meant tens of megabytes of serialization and I/O. It
  now appends a delta describing only what changed, and folds the log back into
  a checkpoint every 256 mutations. Measured on a 10,000-print index, a commit
  went from 60.5 ms to 4.4 ms. An existing store is read once and upgraded in
  place, and a crash mid-append drops the torn record whole rather than the
  prints before it.
