# Fuzzy model search across UI surfaces

## Plan

1. Add pure, dependency-free matching/ranking in Studio and MoldClient, with the same behavioral fixtures. Test before implementation: separator/accent folding, tokens across fields, compact IDs, abbreviations, insertion/deletion/substitution/transposition, exact-match priority, stable ties, numeric variants and unrelated-query refusal.
2. Wire the shared StyleMenu (web Create, Tauri desktop Create, Tauri iOS/Android Create); keep selection, capability gates, keyboard navigation and machine routing intact. Make its search available even for short lists.
3. Wire Installed and curated Discover on web, Tauri desktop/mobile, native macOS and native iOS. Add search to native iOS Installed, which currently has no search field. Preserve explicit kind/source/family filters and user-selected table sorts.
4. Apply matching to installed model command-palette rows, per-style settings, add-on look selectors, supported upscaler/ControlNet selectors and the native Mac source-media model chooser. Keep asynchronous epoch fencing and installation authority unchanged.
5. Add a bounded server-side remote catalog fallback: when the original query returns no results and no provider errors, retrieve a broader prefix candidate set, fuzzy-filter/rank it, and paginate that same bounded set. Retain source/family/kind/NSFW/credential filters. Never broaden authentication failures or outages. Provider coverage remains bounded by their indexes and returned candidates, not an exhaustive fuzzy index of the internet.
6. Document the behavior, run targeted regressions and relevant builds, exercise rendered model search, then obtain an independent sub-agent review. Address findings, publish a conventional commit/PR, verify checks on the final head and merge.

## Surface inventory

| Surface | Generate | Installed / curated discovery | Remote discovery |
| --- | --- | --- | --- |
| Web | `studio/components/StyleMenu.vue` via `CreateStylePicker.vue` | `ModelsPage.vue`, `useCatalog.ts` | server catalog proxy |
| Tauri desktop | shared StyleMenu via `ModelPicker.vue` | `InstalledTab.vue`, `CatalogTab.vue` | server catalog proxy |
| Tauri mobile (iOS + Android) | shared StyleMenu via mobile Style sheet | `MobileCatalogView.vue` | server catalog proxy |
| Native macOS | `Generate/ModelPicker.swift` | `ModelsPane+Grouping.swift`, `Model.matchesDiscovery` | server catalog proxy |
| Native iOS | `Generate/GenerateMenus.swift` | `InstalledModels.swift`, `Model.matchesDiscovery` | server catalog proxy |
| Web / Tauri command palette | installed model commands | existing debounced catalog search | server catalog proxy |

## Matching contract

All query tokens must match some searchable field. Names, display titles, family and repository identity rank above description/summary. Case, accents and punctuation do not affect matching. Direct matches beat compact-ID matches, prefixes, substrings, small spelling errors and bounded ordered abbreviations. Numeric variant tokens require direct matches. Empty queries preserve the original list. Ties preserve original order. Search changes only visibility/order, never model identity, installation state or eligibility. Existing explicit management sorts remain authoritative.

## Validation and review

- Full frontend gate: Studio 2,629 tests, web 2,256 tests, desktop/mobile/UI 7,721 tests; architecture and web/desktop production builds passed. Mobile production build and dead-code/production-graph checks passed separately.
- Catalog: all 123 tests and Clippy passed through Nix. Standalone desktop and Mac FFI lockfiles were refreshed with Cargo for the existing Unicode-normalization dependency.
- Real Chromium component UAT at 1100×850 and 390×844: fuzzy query `flxu q4`, keyboard selection, unrelated-query refusal, clear-search recovery, and upscaler selected-option preservation passed without console errors. This verifies the shipped shared components in an isolated harness, not full native application UAT.
- Native Mac architecture/accessibility lint passed. Local Swift builds and iOS helper lint are blocked by the machine's unaccepted Xcode license; native CI must pass before merge.
- Independent sub-agent review identified missing mobile Generate upscaler and browser ControlNet/reference-adapter coverage, plus native cross-family relevance ordering. All were corrected. Reviewer reran 72 web and 168 desktop selector tests and reported no outstanding actionable findings.
- One pre-existing web media test was corrected to mock the server's public-access confirmation rather than expecting a keyless remote URL without a ticket read. Production media policy was unchanged.
