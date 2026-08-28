- **MiniMax H3 first-frame (FL2VA) prints render again on the durable queue.**
  Since the single durable admission path landed, every H3 job carrying a
  `source_image` was admitted and then blocked forever as a phantom VRAM
  shortage ("required < headroom"): the admission identity hashed the hydrated
  request while the scheduler resolved the scrubbed row, and the row's missing
  first frame read as a text-only render. The identity is now over the
  persisted form (one scrub authority in `mold-core`), the resolver reads the
  queue-media projection for the first frame, the worker revalidates its own
  hydrated copy and reads the first frame off its media projection at every
  fence rather than trusting the payload-free row, and a refused plan names the
  conjunct that refused it in the log
  ([#1427](https://github.com/utensils/mold/issues/1427)).
