- **The desktop app builds and runs on Windows.** Mold Studio is now a
  three-platform desktop app: `scripts\windows.ps1` is the Windows peer of the
  Nix devshell's `desktop-*` commands (`doctor`, `setup`, `dev`, `ui`, `check`,
  `test`, `build`), a `tauri.windows.conf.json` bundles a decorated window into
  an NSIS installer, and a `windows-latest` CI job runs clippy and the tests on
  every relevant pull request and builds the installer on `main`. Windows gets
  native click-to-route toast notifications, and Ctrl+A now drives Library's
  Select All (it was recognised only as ⌘A, so it did nothing off macOS).
  Two capabilities are honestly absent for now: generated AAC audio tracks
  (`fdk-aac` does not compile with MSVC — video still renders and muxes), and
  in-app updates, which stay macOS-only.
- **Gallery filenames can no longer address a second location on Windows.**
  The desktop reveal-in-folder command carried its own weaker traversal check
  that missed `\` and `:`; every gallery path now shares one rule, which also
  refuses a drive-relative `C:name.png` — a form `Path::join` resolves against
  that drive's working directory rather than the gallery
  ([#1305](https://github.com/utensils/mold/issues/1305)).
- **Publishing a print, a chain output, or a batch record is durable on
  Windows.** The four copies of the post-rename directory fsync were each
  `#[cfg(unix)]` with a silent no-op twin, so on Windows none of them ran and a
  renamed entry could be lost while the file's own bytes survived. They now
  share one implementation that flushes the directory on both platforms.
- **A Windows clone no longer fails every formatting gate.** Git's default
  `core.autocrlf` checked the tree out as CRLF, which prettier rejects for
  every file and which makes a `#!/usr/bin/env bash` shebang invalid. A
  `.gitattributes` pins LF in the working tree on all platforms.
