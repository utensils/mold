### Fixed

- Prevent native Source Media details from repeatedly scanning the entire library, keep source lists stable during refresh, and avoid reloading originals for unrelated print updates.
- Open legacy retained images, video and audio even when older metadata does not identify their type, with bounded content inspection and temporary-file cleanup.
- Keep macOS navigation and sync responsive by saving drafts and comparing/hashing sync files off the UI thread, while preserving final-quit draft durability and source-access revocation.
