# Native macOS generation and Library parity

## Confirmed failures

- The source-image request builder always forwards draft strength (initially 0.75), even when the recipe advertises no strength control. MiniMax H3 requires 1 and rejects the request shown in the report.
- Audio starts disabled, and visiting a recipe without audio clears it. New video drafts should enable audio wherever the recipe supports it, while explicit choices and restored settings remain authoritative.
- Library offers settings reuse but no image-as-source action. The full-size viewer does not attach the shared context menu.

## Implementation

1. Reconcile supported strength from the advertised recipe and apply it at the shared request boundary used by batches, placement, and chains; keep authored strength available when switching back to a capable recipe.
2. Track audio preference separately from recipe availability. Default to on for capable video, preserve explicit off across model changes and restoration, enforce output format and video-only compatibility, and keep unsupported requests free of audio flags.
3. Extend the single Library action plan with Use as Source Image and Add as Reference. Gate on live raster selection and destination capability, use authenticated media from the holding host, preserve the current generation draft, and fence asynchronous loading against changed selection/draft. Attach the same menu to the viewer and check generated-result actions.
4. Add regression tests for request values, capability transitions, restore, action eligibility and routing. Run package/app tests, native architecture lint, and rendered native UAT with disposable state and a controlled backend where appropriate.
5. Refresh architecture graph, update changelog and documentation, obtain final Claude Code review, resolve findings, open PR, inspect reviews and exact-head CI, merge and verify main.

## Review and acceptance

A separate subagent reviews this plan before implementation. UAT must demonstrate library and viewer right-click actions, attached source/reference images, audio defaults and off behavior, and a captured H3 source request with strength 1. Server capacity refusal is distinct from request correctness; do not claim a GPU render without evidence.

### Plan peer review and audit refinements

The independent plan reviewer accepted the design, requiring an explicit old-host
fallback for missing strength capability, parked audio preference separate from
availability, and async fencing across all attachment targets. Implementation
review added two requirements from current server and shared-client source:

- H3 requires strength 1 even without a source; omitting it defaults to 0.75.
- H3 audio is always included, so it must not offer an invalid off/video-only
  control. Optional audio respects both recipe support and the model row's
  checkpoint-asset availability, matching the existing shared client.

Library attachment also fences navigation away and live-row changes, supports a
Library-first launch without visiting Generate, and excludes audio-only files.
