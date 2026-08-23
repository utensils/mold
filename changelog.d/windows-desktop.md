- **The desktop app builds and runs on Windows.** Mold Studio is now a
  three-platform desktop app: `scripts\windows.ps1` is the Windows peer of the
  Nix devshell's `desktop-*` commands (`doctor`, `setup`, `dev`, `ui`, `check`,
  `test`, `build`), a `tauri.windows.conf.json` bundles a decorated window into
  an NSIS installer, and a `windows-latest` CI job runs clippy and the tests on
  every relevant pull request and builds the installer on `main`. Windows gets
  native click-to-route toast notifications, a Window menu whose items all do
  something, and Ctrl+A now drives Library's Select All (it was recognised only
  as ⌘A, so it did nothing off macOS — **Linux gains this too, and Meta+A stops
  triggering it there**). Two capabilities are honestly absent for
  now: generated AAC audio tracks (`fdk-aac` does not compile with MSVC — video
  still renders), and in-app updates, which stay macOS-only.
- **The embedded engine starts on Windows.** Three separate defects stopped it
  before it could serve a request, and each of them reported something other
  than what was wrong. Gallery lock probing classified contention by
  `ErrorKind::WouldBlock`, which is only the unix spelling — Windows reports
  `ERROR_LOCK_VIOLATION`, so the probe read its own expected answer as a fatal
  error and refused to start. Startup recovery flushed a directory by opening
  it read-only, which Windows answers with a bare "Access is denied". And the
  scheduler's artifact-identity check used two nightly-only APIs. All three are
  fixed, and the engine now boots, authenticates, and shuts down cleanly.
- **Publishing a print, a chain output, or a batch record is durable on
  Windows.** Five copies of the post-rename directory fsync each assumed a unix
  handle: four were `#[cfg(unix)]` with a silent no-op twin, so nothing ran,
  and the fifth failed outright. They now share one implementation that opens
  the directory the way Windows requires and flushes it on both platforms.
- **Gallery filenames can no longer address a second location on Windows.**
  The desktop reveal-in-folder command carried its own weaker traversal check
  that missed `\` and `:`; every gallery path now shares one rule, which also
  refuses a drive-relative `C:name.png` — a form `Path::join` resolves against
  that drive's working directory rather than the gallery.
- **A `mold serve` on Windows shuts down gracefully.** With no SIGTERM there,
  the graceful path — and with it the queue journal's retention fence — was
  reachable only through `POST /api/shutdown`, so closing the console window or
  shutting the machine down discarded every retained queue row. Ctrl-C, console
  close, and system shutdown now all take the same path SIGTERM takes.
- **A Windows clone no longer fails every formatting gate.** Git's default
  `core.autocrlf` checked the tree out as CRLF, which prettier rejects for
  every file and which makes a `#!/usr/bin/env bash` shebang invalid. A
  `.gitattributes` pins LF in the working tree on all platforms.
- **`scripts\windows.ps1` gives the same answers in both PowerShell editions.**
  A `.ps1` runs under whichever edition the user's shell is, and the helper's
  architecture probe did not survive that: under Windows PowerShell 5.1
  `RuntimeInformation::OSArchitecture` reports **X64 on an ARM64 machine**
  (pwsh 7 reports Arm64 on the same box), and on some .NET Framework builds the
  property is missing entirely, which under `Set-StrictMode` crashed the script
  outright. Since that probe gates whether `cuda` enters the build recipe, the
  quiet wrong answer was the worse half. Architecture now comes from WMI, the
  x64 question is answered by the Rust host triple that actually decides what
  cargo builds, and a new test refuses to let the two editions disagree.
- **Windows downloads now ship from rolling and tagged releases.** The x64
  NSIS desktop installer and the separate CPU/remote-client CLI zip share one
  pinned self-signed Authenticode identity; releases include its public
  certificate, SHA-256 coverage, explicit trust instructions, and Windows-logo
  download cards alongside macOS. CI fails closed when the retained PFX or
  either signature is missing. Public-trust signing remains future work.
