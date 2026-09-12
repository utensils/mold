- **Every style picker now shows the source glyph.** The Hugging Face, Civitai,
  and local-file marks used to appear only in the desktop app's own style
  chip and model rows. The shared style menu now draws one by default for
  web and the phone too, and the Installed and Discover model rows, the
  model detail drawer, and the phone's catalog cards carry the same mark.
- **Shape tiles no longer wrap onto a second row in a narrow rail.** Five
  aspect-ratio tiles plus their gaps needed more width than web's 320px
  Create rail had to give them, so a 9:16 tile fell to its own line. The
  tiles now sit in a grid that shrinks each one to fit instead.
- **The machine card's progress meter is visible again.** Its track color
  matched the card's own background, so a render's progress bar read as
  empty space until the fill caught up. The track now uses a color that
  contrasts on every panel and surface tone.
