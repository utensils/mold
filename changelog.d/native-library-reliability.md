### Added

- Native macOS nightly builds include bounded responsiveness diagnostics and sync timing signposts, with a read-only capture script for investigating slowdowns on an existing running app.

### Fixed

- Native Library video playback preserves playback intent through streaming stalls and resolves the final video frame only when needed; iOS plays complete locally cached originals and can reopen, share and save them offline.
- macOS viewer Delete moves the displayed print to Trash, and selected Library thumbnails have a stronger outline and checkmark.
- Native Source Media uses a resizable Library-style thumbnail grid and filters, includes hidden collections, and avoids repeated empty-source checks and unrelated preview refreshes.
- macOS Library counts are cached across sync progress redraws. Native macOS and iOS cache preferences now include Unlimited.
