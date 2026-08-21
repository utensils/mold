- **File prints under tags and collections at creation.** `GenerateRequest`
  and the HTTP chain body carry additive `tags` and `collection`, so a print
  can arrive already organized instead of being filed afterwards. On the CLI
  that is `mold run --tag <TAG>` (repeatable, up to 20) and
  `--collection <NAME>`, which resolves by slug and creates the collection
  when absent, so one name means one collection across a fleet. A titled run
  also tags the print with its title slug and says so
  (`filing under tag "smurf-village"`); `--no-auto-tag` or
  `mold config set generate.auto_tag_title false` turns that off. Filing is
  seeded onto the gallery row once, at creation — organization is yours
  afterwards, and a reconcile never resurrects a tag you removed. Nothing
  about filing can fail a render: a host with `MOLD_DB_DISABLE=1`, or a
  collection deleted between listing and Generate, drops the filing and
  reports it on `x-mold-request-warning`.
- **Sequences can be titled.** The HTTP chain body accepts `title`, applied to
  the stitched print exactly like a one-shot's — embedded in its metadata,
  seeded into its gallery row, and folded into its filename as `~slug`.
