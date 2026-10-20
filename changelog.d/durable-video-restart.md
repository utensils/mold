- **Recover failed videos.** Failed video generations retain their exact inputs
  in the durable queue for movement to another machine. Failed and interrupted
  sequence parents remain visible after reconnect for explicit recovery.
- **Restart video sequences.** Failed or paused unlinked sequences can start a
  separate attempt using the selected machine's current model recipe through
  revision-fenced restart exports and `mold jobs restart`. Resume preserves the
  original frozen recipe.
