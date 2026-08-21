- **File prints under tags and collections at creation.** `GenerateRequest`
  and the HTTP chain body carry additive `tags` and `collection`, so a print
  can arrive already organized instead of being filed afterwards. A titled
  print also picks up its own title slug as a tag — every surface shows that
  chip before you generate, and every surface can turn it off. Filing is
  seeded onto the gallery row once, at creation: organization is yours
  afterwards, and a reconcile never resurrects a tag you removed. Batch
  siblings and prepared variations inherit the filing of the print they came
  from, and a sequence files only the stitched print it lands in the Library.
  Nothing about filing can fail a render — a host with `MOLD_DB_DISABLE=1`, or
  a collection deleted between listing and Generate, drops the filing and
  reports it on `x-mold-request-warning`.
- **Sequences can be titled.** The HTTP chain body accepts `title`, applied to
  the stitched print exactly like a one-shot's — embedded in its metadata,
  seeded into its gallery row, and folded into its filename as `~slug`. On
  iPhone, Sequence output therefore keeps the Title field instead of replacing
  it with a note.
- **`mold run` files from the command line.** `--tag <TAG>` is repeatable, up
  to 20 tags, and `--collection <NAME>` resolves by slug and creates the
  collection when absent, so one name means one collection across a fleet. A
  titled run also tags the print with its title slug and says so
  (`filing under tag "smurf-village"`); `--no-auto-tag` or
  `mold config set generate.auto_tag_title false` turns that off.
- **Desktop, web, and iPhone Create file a print before it develops.** A
  capability-gated **File under** group — the Create inspector between the
  essentials and Advanced on desktop, the controls region (or the controls
  sheet on phones) on web, two 44pt rows under Title on iPhone — carries a
  dashed `{slug} · from title` ghost chip derived from the print title
  (removable, and the removal sticks), typed tags with autocomplete over every
  connected machine's tag counts, and a collection row that pre-selects —
  never creates — the collection whose slug matches the title, with a picker
  for the fleet's collections and an inline **New collection…** that only
  records the name. A mono line previews the filename the print will land as,
  `~title-slug` included. The choice rides one shots, every Batch N sibling,
  every prepared variation, and a sequence's stitched print, and **Reuse
  settings** restores what a print was actually filed under. The group hides
  entirely on a machine whose `capabilities.gallery.organize` is not true; on
  iPhone under **Auto** or **Most capable**, any reachable machine that can
  file is enough. Settings ▸ Library gains **Tag new prints with their title**
  (on by default); turning it off changes nothing about prints you already
  made.
- **The TUI files a print while you make it.** A **File under** section joins
  the Create Advanced accordion, last, after the generation parameters:
  **Title**, comma-separated **Tags**, and one **Collection** by name, each
  edited in a popup that keeps invalid entry on screen with its reason instead
  of closing and discarding it. All three are absent until touched, so an
  untouched form sends exactly the request it always did. A titled print shows
  the tag it will pick up from its own title (`auto: smurf-village`) on the
  Tags row before you generate, and Settings ▸ Library gains **Tag by title**
  to turn that off — the same `generate.auto_tag_title` preference `mold run`
  reads. Filing is per-print intent: it is not remembered across sessions, is
  not a per-model default, and only **Reset to model defaults** clears it. A
  forced-local render files itself the same way and folds the title's slug
  into the saved filename, exactly as a served one does.
- **Downloads are named after the print.** Desktop Save / Export and web Save /
  Download now suggest `{title-slug}__{model}__s{seed}.{ext}` — `mold-core`'s
  own download grammar — instead of the bare title slug or the raw gallery
  filename, so two takes of one print no longer collide in a Downloads folder.
  The file in the gallery is never renamed.
- **The terminal now shows what a host adjusted or dropped.** Advisories about
  an accepted request — a lip-dub render retimed to its reference clip, or a
  filing a host could not apply — ride the `x-mold-request-warning` header, but
  no client read it. `mold run` and `mold chain` now print each one, and the TUI
  shows them on the Create view and in the Timeline. `GenerateResponse` and
  `ChainResponse` gain an additive `request_warnings` list for API consumers.
