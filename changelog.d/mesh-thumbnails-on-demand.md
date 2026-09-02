### Fixed

- Mesh prints mirrored onto a machine now show their own poster tile instead of the generic wireframe-cube placeholder. The poster is derived from the stored `.glb` wherever it is missing: `GET /api/gallery/thumbnail/:filename` renders it on demand and caches it, `PUT /api/gallery/import` writes it before announcing the print, and the desktop app renders it in-process while its embedded server is off.
- A mesh whose geometry genuinely cannot be read still answers the placeholder, but tags it apart from a real tile so a client that cached one revalidates into the poster instead of holding a wireframe cube for the life of the file. The desktop's on-disk thumbnail cache no longer stores a placeholder as a print's durable tile.
- A generated mesh writes its poster before the gallery announces it, so a client that loads the tile the moment the print appears is no longer answered a placeholder it would then cache.
