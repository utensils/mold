- **Faster gallery thumbnails on every surface.** `GET /api/gallery/thumbnail/:name`
  now takes `?size=256|512` and `?fmt=png|jpeg` so retina displays get a sharp
  512 px JPEG tile (about a quarter the bytes of the PNG; prints with
  transparency stay PNG), while the default 256 px PNG keeps its exact path and
  ETag for older clients. Video posters decode only the first frame instead of
  the whole clip, startup warmup renders on a bounded thread pool newest-first
  instead of one core serially, orphaned tiles of purged or re-rendered prints
  are swept after warmup, and the Library listing walks a new
  `(output_dir, recency)` index instead of sorting the directory in a temp
  B-tree on every poll.
