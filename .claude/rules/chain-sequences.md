---
paths:
  - "crates/mold-core/src/chain*.rs"
  - "crates/mold-server/src/chain_*.rs"
  - "crates/mold-server/src/routes_chain*.rs"
  - "crates/mold-cli/src/commands/chain*.rs"
  - "crates/mold-cli/src/commands/run.rs"
  - "crates/mold-cli/src/commands/jobs.rs"
  - "crates/mold-db/src/chain_jobs.rs"
  - "crates/mold-inference/src/chain/**"
  - "studio/lib/chain*.ts"
  - "desktop/src/lib/chain*.ts"
  - "web/src/lib/chain*.ts"
  - "crates/mold-discord/src/commands/**"
---

# Scripted sequences and chain jobs (CLI + API only)

Moved from the root CLAUDE.md; loaded only when working under the paths above.

Scene-by-scene authoring is **retired from every interactive surface**. There is no
`Simple | Scenes` strip, no `One shot | Sequence` output, no timeline, and no
Discord `/sequence`. Making a clip in a GUI is one flow: a prompt,
the model controls, and the length slider. A sequence is now something you script.

- `mold run --script shot.toml` — canonical TOML, schema `mold.chain.v1`. Per-stage `prompt` / `frames` / `transition`.
- `mold chain validate shot.toml` or `mold run --script ... --dry-run` to inspect without submitting. `mold chain` has exactly one subcommand; listing is `mold jobs list`.
- Sugar: `mold run <model> --prompt "..." --prompt "..." --frames-per-clip 97` (uniform smooth only).
- Transitions: `smooth` (default, motion-tail morph), `cut` (fresh latent), `fade` (cut + RGB crossfade).
- Per-stage source images: `source_image_path` (relative to script file) or `source_image_b64`. Resolved by `mold_core::chain_toml::read_script_resolving_paths`.
- **An auto-chained one-shot is not an authored sequence, and after the retirement it is the ONLY chain a GUI creates.** `ChainRequest.ephemeral` (additive, absent means authored) is the distinction, set on the ONE creation path: `mold run --frames 200`, or a length slider past the checkpoint's clip size, renders a long video as a chain because the model cannot do it in one pass, so its job is absent from authored `GET /api/chain-jobs` listings and emits no `chain_job_queued`, while `/api/activity` still exposes the live job on every client. Ephemeral means hidden from sequence history and swept only after settlement; it does **not** mean disposable while active. Graceful shutdown parks it as `paused`, preserves its manifest, source media, completed clips, and tail cache, and allows explicit resume after restart. Its final print carries `chain_job_id: None`. It DOES publish its print with full per-clip provenance. Stage seeds are recorded either way — they describe how the pixels were made, not who authored the split.
- **A one-shot may never silently become a context-free chain.** `mold_core::chain::text_only_auto_chain_refusal` is the server authority: a wan tier whose `source_image` contract is `Unsupported`, or any legacy `ltx-video` tier, hands nothing across a clip boundary, so identical prompt-and-seed stages repeat. Text-only Wan is refused above its clip size at the CLI, in every app, and at `POST /api/chain-jobs`. Legacy LTX-Video takes the other honest route: the CLI and the apps keep it as ONE denoise up to its 257-frame engine ceiling, while server admission rejects a manually submitted `ephemeral` multi-stage request. The Wan and LTX-Video surface-parity fixtures pin the family-specific sentences. SCRIPTED sequences still compose independent clips because there that is what the author asked for.
- **Random seed authority:** an omitted chain seed is materialized once by `ChainRequest::materialize_seed` before dispatch or durable manifest creation; explicit `0` is fixed. The database request JSON must match the manifest’s resolved request, and stage seeds derive from that persisted base and authored offsets. Resume/retake of existing manifests keeps their recorded authority; never reroll at a stage boundary.
- **A remote sequence is a DURABLE CHAIN JOB.** `POST /api/chain-jobs` creates one and `GET /api/chain-jobs/{id}/events` streams its stage progress; `mold run --script` takes that route and hydrates the stitched print from the host's gallery. The synchronous `POST /api/generate/chain` and SSE `POST /api/generate/chain/stream` shims are DELETED — they ran a chain as a hidden ephemeral job and deleted its artifacts after answering, so a dropped connection lost work that could not be resumed, retaken, or reattached. `POST /api/generate/chain/validate` survives and is the only thing left in `routes_chain.rs`, which is read-only planning. A `--script` run therefore leaves a job `mold jobs list` can find, and the CLI does not receive the host-side thumbnail or GIF preview inline. The whole endpoint family — create, events, resume, retake, amend, cancel, delete, gc, stage media, and `/api/capabilities/chain-limits` — stays supported; the apps simply no longer author against it, and read only the ephemeral jobs they create themselves.
- **A print made by a sequence is provenance, not a door back.** `metadata.chain` and `metadata.chain_job_id` still ride the stitched print and still render in the Library. Reuse on such a print degrades to a plain one-shot restore built from the FIRST stage's prompt — never `metadata.prompt`, which for a sequence is every stage newline-joined.
