# Held queue model recovery

The reported HAL job is held because its reviewed H3 Turbo adapter is missing. Its batch child has no missing-model error code. Native recovery currently depends on that code, and iOS additionally hides recovery from the ordinary queue card. Preserve the saved explanation, failure details, and cross-host transfer while exposing the existing download-and-retry workflow.

## Reviewed contract

- Add optional `model_download: { model: string }` to queue listing and detail rows. The model is the exact row model. Emit only for held, retryable jobs whose known local manifest has missing downloadable components at the same effective paths acquisition will repair. Inspect file presence only, once per distinct model per listing; do not hash weights, read retained media, query the network, or classify formatted error prose.
- Do not suggest a model download for an unrelated missing file when the recipe is complete, an unknown model, or a missing custom override that the manifest cannot repair. Preserve existing error, diagnostic, and retryability fields.
- Clients prefer the hint. Current server manifest inspection returns an explicit null when acquisition cannot repair the recipe, preventing a stale legacy error from reviving the action. Absent hints retain the existing typed batch-error fallback for older servers and uninspected `hf:`/`cv:` catalog recipes. A present null, malformed or mismatched hint fails closed. Labels say **Download and Retry**, since successful acquisition resumes the job; do not conceal that behavior behind a download-only label.
- Reuse the existing acquisition, license-consent, exact-ticket settlement, and retry orchestration. Fence the host route, credentials, instance, job, model, and durable authority at each mutation. Failed/cancelled downloads must leave the job held. Do not clear the saved error merely because downloading starts.
- Expose recovery on native queue cards/details and applicable web/Tauri queue, host, and activity surfaces. Preserve Move to and Failure Details. Keep action buttons outside row buttons and readable at accessibility text sizes. Authoritative hints must not introduce periodic detail/batch probes for every visible row.

The related native iOS model-list repair layout places active download progress and cancellation below full-width model information, with accessible cancellation and adaptive text sizing. Verify this alongside queue recovery with synthetic fixtures.

## Ownership and peer review

Server owns Rust wire/projection/acquisition verification and API documentation. Native owns shared Swift decoding/resolution and iOS/macOS presentation. Vue owns shared recovery controls plus web/Tauri integration. Root owns integration, user-facing documentation, PR publication, deployment, and cleanup. Work is consolidated into existing PR #1849 alongside the separately hardware-qualified LTX decode fix.

All three peers researched the existing recovery workflow before implementation. Native review requires malformed hints to fail closed and actual diagnostic copy to remain visible. Vue review requires credential-specific recovery state, no nested buttons, and no added per-row polling. Server review requires effective-path equality with the manifest acquisition destination, rather than assuming every missing file is repairable by downloading a model.

## Verification and release

Write failing regressions for the actual missing H3 companion, complete/unknown/custom-override recipes, row/detail parity, deduplicated model inspection, hint validation and legacy fallback, inline actions with retained diagnostics/transfer, and exact-host recovery. Reuse existing license, cancellation, companion-ticket, and retry tests rather than duplicating the entire recovery suite.

Use harmless local fixtures for visual and interaction UAT, including ordinary and accessibility text sizes. At most one simulator is active. No production model download or generation is authorized by this implementation task. Pending TestFlight uploads for #1850 were cancelled before upload so the final consolidated client build can include this fix; ordinary CI continues. Require independent review and green CI on the final commit before merging. Deploy server changes to both hosts when existing jobs and the completed GPU qualification permit a safe activation, then remove task build artifacts while retaining evidence.

## Final independent review

A fresh reviewer inspected the complete PR after implementation. Review found
and resolved a native legacy-to-current hint validation gap and an omitted
server negative result that could revive stale legacy recovery. Actual shell
UAT also caught nested host-row action semantics and contradictory retry copy;
regressions cover each correction. Final review is tied to the published commit
before merge, alongside its CI results.
