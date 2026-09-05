- **Prints remember how long they took.** The render time is embedded in every
  new print's metadata and read back for older ones from the gallery, so the
  canvas caption, the Recent tab, and the Lightbox's new **Took** line say
  `4.0s` (or `1m 12s`) instead of nothing — and never `0.0s` when a print does
  not know ([#1597](https://github.com/utensils/mold/issues/1597)).
- **Styles shows each style's typical speed.** A new **Speed** column reads
  `~20s` off the median of your most recent prints with that style; a style
  you have not timed yet shows nothing rather than a guess.
