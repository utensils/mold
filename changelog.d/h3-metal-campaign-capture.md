Added Metal campaign instrumentation for MiniMax H3: an opt-in
`minimax_h3::campaign_capture` that records machine-readable memory and
phase-budget rows plus an allocation-free budget sidecar for exact prepared
requests, a required native-allocation ceiling in the campaign Metal memory
guard (`MOLD_H3_METAL_CAMPAIGN=1`, `MOLD_H3_METAL_CAMPAIGN_CEILING_MB`), a
budget-only pre-flight refusal path
(`MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY=1`), macOS process-memory probes in the
H3 runtime observer, and the `h3_metal_campaign_watch` external supervisor
dev-bin. Production behavior is unchanged: the capture is inactive without
the campaign variables.
