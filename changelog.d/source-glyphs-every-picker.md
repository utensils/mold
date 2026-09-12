- **Every style picker now shows the source glyph.** The Hugging Face, Civitai,
  and local-file marks used to appear only in the desktop app's own style
  picker. The shared style list now draws one by default for web and the
  phone too, and the Ready to use and Browse more rows, the style details
  panel, and the phone's style cards carry the same mark.
- **Shape tiles no longer wrap onto a second row in a narrow rail.** Five
  aspect-ratio tiles plus their gaps needed more width than web's 320px
  Create rail had to give them, so a 9:16 tile fell to its own line. The
  tiles now sit in a grid that shrinks each one to fit instead.
- **A meter's track is visible again wherever it sits on a panel.** Its
  track color matched a panel's own background on the machine card, the
  host detail page, the activity strip, the cold-start guide, the downloads
  list, and the phone's queue card, so a render's progress meter read as
  empty space until the fill caught up. The track now uses a color that
  contrasts on every panel and surface tone.
