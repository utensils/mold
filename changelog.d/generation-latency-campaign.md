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
  PNG-tuned ultra-fast deflate: on a 512² photograph 1.7 ms against 59.0 ms,
  for a 6 % larger file. PNG is lossless under both settings, so no pixel
  changes; `MOLD_PNG_ENCODING=balanced` restores the smaller files. The
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
