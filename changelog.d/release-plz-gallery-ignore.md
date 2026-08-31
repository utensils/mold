- Fixed the release automation, which had been unable to compute versions since
  the gallery expansion: seven `website/public/gallery/*.webp` posters were both
  tracked and matched by the repository's `*.webp` rule, and release-plz refuses
  to run at all in that state.
