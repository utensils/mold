- **Windows release artifacts publish again.** Authenticode verification no
  longer imports the self-signed certificate into the user Root store to force
  a `Valid` status. That import needs interactive trust confirmation, and CI
  runs PowerShell without `-NonInteractive`, so it blocked on a dialog nobody
  could see until the job's six-hour ceiling — stalling every Windows installer
  and, because publication waits on the Windows build, the Linux, container,
  and AUR assets with them. The check now pins the signing thumbprint against
  each signature, so tampered, unsigned, and wrongly signed artifacts are still
  refused
  ([#1379](https://github.com/utensils/mold/pull/1379)).
