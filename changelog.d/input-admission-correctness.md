- **Reject invalid face-identity requests before downloads.** Forced-local
  generation now refuses an unqualified PuLID model before resolution, repair,
  or pull; server admission is pinned to return before durable or download queues
  ([#1305](https://github.com/utensils/mold/issues/1305)).
- **Honor phone-photo orientation throughout MiniMax H3.** Reference images and
  FL2VA endpoints now share the EXIF- and ICC-aware bounded decoder, while CLI,
  TUI, and server descriptors report the same upright dimensions the model sees
  ([#1431](https://github.com/utensils/mold/issues/1431)).
