# Native iPhone landscape video UAT — 2026-10-09

Device: iPhone 17 Pro Simulator, iOS 26.5. A loopback-only machine serves a
30-second, 640×360 H.264 test-pattern clip with visible timecode. No generation
or real machine mutation is involved. This is Simulator evidence; a physical
phone was not tested.

## Acceptance

- [x] A real clip decodes and plays with AVKit's Pause control available.
- [x] Both landscape orientations use the entire window for the player, hiding
  gallery/status bars and preserving the video's proportions.
- [x] The Close target is at least 44×44 points, stays inside the display, and
  does not intersect any hittable native player button.
- [x] Rotating upright restores gallery actions; rotating again and tapping
  Close returns to All Prints.
- [x] Dark appearance passes the complete targeted UAT.
- [x] Light appearance passes the complete targeted UAT.
- [x] Independent sub-agent review approves the final implementation and both
  dark landscape screenshots, with no unresolved findings.

## Verification and evidence

The new orientation-policy unit test failed to compile before its implementation.
The full native unit suite passed 218 tests in 36 suites. After the final Close
placement change, all six `PlaybackAudioTests` and the targeted
`LandscapePlaybackTests/testClipFillsBothLandscapeOrientationsAndRestoresPortrait`
passed again (7 tests, zero failures). The final binary then passed the identical
light UAT (1 test, zero failures). Both full-device landscape captures and
portrait restoration were visually inspected in both appearances.

`make -C apps/ios lint`, `scripts/tests/ci-routing-contract.sh`,
`scripts/tests/ios-native-ci-scope.py` and `git diff --check` passed.

- [Dark landscape left](uat/landscape-video/dark-left.png)
- [Dark landscape right](uat/landscape-video/dark-right.png)

Local result bundles are retained under
`/Volumes/ExternalStorage/mold-ios-landscape/`: `final-dark-unit.xcresult`
(full unit suite), `reviewed-dark.xcresult` (final playback unit tests and UAT),
and `reviewed-light.xcresult` (final light UAT).

Earlier diagnostic bundles remain alongside them. Review caught and corrected
an incorrectly routed square fixture and a Close/AirPlay overlap. Full-device
screenshots replace cropped application screenshots. One earlier Simulator
runner terminated with signal TERM during launch; the isolated completed runs
are the acceptance evidence, not the restarted zero-test suite output.

The test explicitly enables autoplay/repeat for its process, so saved preferences
and clip completion cannot turn a rotation assertion into a timing failure.
