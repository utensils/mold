- **Mold Studio Companion, the native iPhone and iPad app.** A new SwiftUI
  app (`apps/ios`, `io.utensils.mold.companion`, "Mold Studio" on the Home
  Screen) that sits beside the existing iPhone app and matches the macOS
  Mold Studio: pair with a machine by scanning the Mac's Pair a Phone… code
  (or by address or Bonjour), browse every machine's prints as one Library
  with favourites, tags, collections and Recently Deleted, and generate
  stills, clips and 3-D objects with each model's own controls. Every screen
  is audited from the smallest text size to the largest accessibility size,
  in light and dark, on iPhone and iPad
  ([#1775](https://github.com/utensils/mold/pull/1775)).
- **Guard: tracked files that `.gitignore` also matches now fail CI.** A test
  fixture from #1767 had silently stopped every release-plz run; the new
  check fails the pull request instead.
