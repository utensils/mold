# Retained media and H3 packaging validation

Validated on Apple Silicon with macOS 26 and Xcode 26 on 2026-10-05.

- MoldClient: all 1,178 existing/shared tests pass, including draft-scope and aspect-geometry regressions. New preview transport tests also pass for declared and chunked oversized responses through both thumbnail and older-server original routes.
- Native macOS: remote-only application and app test bundle build successfully with strict concurrency. Hosted native architecture lint, 912 app tests, and 1,180 shared-client tests pass in [the native workflow](https://github.com/utensils/mold/actions/runs/37403855215).
- Native iOS: the complete Companion application and test bundles build for the generic iOS Simulator destination, including the retained source-alias ambiguity regression.
- Shipping engine: locked FFI `shipping-metal` Cargo check and H3 runtime capability test pass locally and in the hosted native workflow. Static shipping-feature, engine-freshness, workflow parsing, and Candle identity contracts pass.
- Server: bounded thumbnail renderer tests and protected-route authentication routing test pass. The new endpoint adds direct authentication and path-traversal regressions.
- Synthetic native UAT: a separately identified disposable app used four synthetic image references served over HTTP. Restoring the locator fetched all four private thumbnails; all four previews were visible. Editing the prompt and switching from square to 11:20 preserved the references and the enabled Generate button. The shape menu visibly shows landscape, square, and portrait boxes at their actual proportions. Every generation/mutation request was refused by the fixture; no GPU generation ran. The existing installed app was left running, and the disposable preference/domain data was restored afterward.

Independent implementation review found and corrected late persistence after media edits, origin identity changes during retained inventory/recovery, and an overly large older-server preview streaming ceiling. Regressions cover those boundaries, successful descriptor recovery, and legacy adoption's initial media baseline.

This verifies packaging and client behavior. It is not a new H3 numerical or hardware generation qualification. The screenshot's exact historical remote engine warning was not reproducible; remote placement messages now carry the target machine name and cannot be replaced by a late answer from another target.

## All-model host transfer audit

The queue's request-media extraction contract covers every accepted attachment field independently of model family, including typed image/named-image/video/audio/mesh references. Existing archive transfer verifies member bytes, digests, role/position/sink, output identity and recipe before binding a destination-owned archive. Save All covers picture, animated image, video, audio and mesh outputs; selected Save Locally remains the existing picture-only action.

The audit found a chain exception: `stage_source:<index>` was retained but rejected by transfer validation. Canonical authored-stage roles now transfer with their stage positions, including later stages. Single-render settings reuse restores only stage 0; complete authored-sequence reconstruction remains outside this action's existing semantics. Conflicting source-image and stage-0 authority is rejected. Matting derivatives remain durable without being applied twice.

The 12-test server transfer suite passes, including a real round-trip covering all 16 standard media roles plus two chain stages, deleting the original output and pins before reading every destination member. The 14-test retained hydration suite and focused all-field extraction and archive-reopen contracts also pass. The hosted native suite includes a Save All video regression with four ordered references, removing the original host before destination reuse. Model weights and machine-local adapters are separate dependencies and are explicitly refused by portable queue transfer rather than silently stripped.
