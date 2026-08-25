- **A MiniMax H3 checkpoint mold cannot run now says so on an H3 build too.**
  On a binary compiled with the H3 engine, submitting a generation for one of
  the pinned identities that has no loader — the `official-bf16` qualification
  references and the two `comfy-pruned-nvfp4` tags — was answered with the
  private ingress boundary's "accepts only its supported compact task
  partition" (HTTP 422) instead of the sentence its own `/api/models` row
  publishes. The row and the refusal now agree again: those identities are
  refused with `MINIMAX_H3_RUNTIME_UNAVAILABLE` over HTTP 501 and the reason
  naming the missing weight-layout runtime, on every build, exactly as they
  already were on builds without the engine. Downloading, verifying,
  inventorying, repairing, and removing them is unchanged, and the reviewed
  compact partition — Ref2VA included — keeps its existing answers
  ([#1354](https://github.com/utensils/mold/issues/1354)).
