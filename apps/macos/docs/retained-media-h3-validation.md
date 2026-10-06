# Retained media and H3 packaging validation

Validated on Apple Silicon with macOS 26 and Xcode 26 on 2026-10-05.

- MoldClient: all 1,178 existing/shared tests pass, including draft-scope and aspect-geometry regressions. New preview transport tests also pass for declared and chunked oversized responses through both thumbnail and older-server original routes.
- Native macOS: remote-only application and app test bundle build successfully with strict concurrency. Native architecture lint passes. The app-hosted test suite runs in the macOS native workflow.
- Native iOS: the complete Companion application builds for the generic iOS Simulator destination after sharing the aspect geometry.
- Shipping engine: locked FFI `shipping-metal` Cargo check passes; its H3 runtime capability test passes. Static shipping-feature, engine-freshness, workflow parsing, and Candle identity contracts pass.
- Server: bounded thumbnail renderer tests and protected-route authentication routing test pass. The new endpoint adds direct authentication and path-traversal regressions.
- Synthetic native UAT: a separately identified disposable app used four synthetic image references served over HTTP. Restoring the locator fetched all four private thumbnails; all four previews were visible. Editing the prompt and switching from square to 11:20 preserved the references and the enabled Generate button. The shape menu visibly shows landscape, square, and portrait boxes at their actual proportions. Every generation/mutation request was refused by the fixture; no GPU generation ran. The existing installed app was left running, and the disposable preference/domain data was restored afterward.

Independent implementation review found and corrected late persistence after media edits, origin identity changes during retained inventory/recovery, and an overly large older-server preview streaming ceiling. Regressions cover those boundaries, successful descriptor recovery, and legacy adoption's initial media baseline.

This verifies packaging and client behavior. It is not a new H3 numerical or hardware generation qualification. The screenshot's exact historical remote engine warning was not reproducible; remote placement messages now carry the target machine name and cannot be replaced by a late answer from another target.
