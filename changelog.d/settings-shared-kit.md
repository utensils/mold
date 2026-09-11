- **Settings is one page on every screen.** Web Settings is now a jump-nav page with a
  search field, the same sixteen lexicon sections as the desktop app (Look, Defaults for
  new images, Write more for me, Machines, Styles & disk, Style licences, My images &
  trash, Phone pairing, Speed & memory, Accounts & tokens, Cloud GPUs, Per-style defaults,
  Profiles, Advanced, Updates & about), and `?section=` deep links. Every engine key the
  server exposes has a plain-words row — RunPod and Lambda keys live under **Cloud GPUs**,
  logging under **Updates & about** — and a machine with per-style overrides shows one
  collapsed row per style instead of eight rows each, so a server with 150 configuration
  rows is a few screens, not thirty. Rows save as you change them; the duplicate GPU list
  is gone. Web and desktop render the same shared schema, rows, controls and theme cards
  ([#1698](https://github.com/utensils/mold/issues/1698)).
