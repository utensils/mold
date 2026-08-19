- **Prompt expansion routes to a machine that actually has the expander.**
  Under Auto or Most capable, Create used to send the rewrite to whichever
  machine the print was routed to and fail with a 422 when only a peer had the
  expansion model. Desktop and web now follow the generation route unless that
  machine reports it lacks the model, then re-rank the eligible machines that
  have it with the same ranking the generation router uses — the print itself
  still goes where it was routed, and a pinned machine is never left. When no
  eligible machine has it, Create offers the pull and names the machine
  (web gains that offer for the first time). Remix follows the same route —
  it runs on the same model
  ([#1162](https://github.com/utensils/mold/issues/1162)).
- **`/api/capabilities.expand` names the expansion model.** The additive
  `model` field reports the manifest model local expansion resolves, so clients
  stop hard-coding `qwen3-expand` when they offer to pull it
  ([#1162](https://github.com/utensils/mold/issues/1162)).
