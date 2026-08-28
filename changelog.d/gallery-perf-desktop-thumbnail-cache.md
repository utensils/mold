- **Desktop Library thumbnails persist on disk.** The app keeps every tile it
  has shown in a content-addressed cache under its app data folder
  (`thumbnail-cache/v1`, bounded to 512 MB / 20 000 files, least-recently-used
  eviction), keyed by host, filename, and the print's content version — never
  by API key — so a cold launch paints the grid from local files without
  asking any machine for a thumbnail it already holds, and scrolling finds the
  tiles around the viewport pre-warmed. Tiles are served through a native
  `mold-thumb://` protocol, so the webview decodes them off the main thread and
  holds no blobs for them. With the built-in engine Off, this device's tiles
  are now real thumbnails rendered in-app (video posters decode only the first
  frame) instead of every full-resolution print. Print a deleted forever
  leaves the cache; a trashed one keeps its tile.
