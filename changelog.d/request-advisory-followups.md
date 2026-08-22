- **Browser surfaces show accepted-request advisories.** Web and desktop now
  raise warning notifications for `x-mold-request-warning`, while iPhone keeps
  each advisory in a dismissible inline banner; advisory prose containing
  semicolons stays intact.
- **Literal hash-prefixed tags stay editable.** Library tag normalization now
  matches the server, so `#blue` remains distinct from `blue` when displayed,
  filtered, added, or removed.
- **TUI size guidance uses warning styling.** Accepted off-recipe custom sizes
  now render in the amber warning slot instead of the red error slot.
