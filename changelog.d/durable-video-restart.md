### Fixed
- Failed video generations retain their exact inputs in the durable queue, allowing safe movement to another machine instead of losing recovery media. Failed and interrupted sequence parents remain visible after reconnect for explicit recovery.

### Added
- Failed or paused unlinked video sequences can start a separate attempt using the selected machine's current model recipe through revision-fenced restart exports and `mold jobs restart`. Exact Resume continues to preserve the original frozen recipe.
