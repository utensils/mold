### Fixed
- Keep reused native Mac references attached across prompt, shape and seed edits, retries and repeated submissions; show bounded retained image previews and verify retained recipes again after relaunch.
- Include the reviewed Metal MiniMax H3 runtime in native Mac shipping builds and prevent stale placement responses from another machine appearing in Generate.
### Changed
- Show proportional aspect-ratio outlines in the native Mac shape menu and readable explanations when Generate is unavailable.
- Fixed host transfer of retained chain stage images; all stage inputs stay in the destination archive, and first-stage settings reuse restores its source picture without silently substituting later-stage inputs.
