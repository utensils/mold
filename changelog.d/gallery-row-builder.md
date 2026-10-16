- **Faster single-print reads.** `GET /api/gallery?filename=` now reads only
  that print's row, organization, and assets instead of building the whole
  library and filtering it, and gallery archive lookups (including Library
  mirror transfer offers) share the cached archive index instead of copying
  it on every read. Responses are unchanged.
