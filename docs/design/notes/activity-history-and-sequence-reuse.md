# Activity is present tense — job visibility + sequence reuse from the Library

Design note, 2026-07-28. Follow-up to #565 (shared sequence kit), #566 (chain
lifecycle events), #568 (desktop unified Create), the in-flight web unified-Create
landing, and the in-flight iOS landing. **Plan only** — no product code, spec, or
changelog changes are made by this note.

Two problems, one root cause.

1. Desktop Create's activity strip accumulates every settled sequence row —
   `COMPLETED ltx-2.3-22b-distilled:fp8 · 5 clips · 5/5 · plato [Watch][Edit][Delete]`
   forever, plus `FAILED` rows, permanently wedged between the canvas and the
   composer. The web strip lands with the same behavior. It is a log, not a
   workspace.
2. A finished sequence has no way home. Its print sits in the Library carrying
   full per-clip provenance (`metadata.chain` since #564) that nothing reads,
   so "make that again, but clip 3 is different" means retyping five prompts.

The root cause is the same in both: **the strip is holding results because
nothing else does.** The canvas drops a completed sequence on the floor (the
watched-sequence develop state only renders while `state` is `running |
queued`), and the Library treats a sequence print as an anonymous video. So the
only durable trace of a sequence is a row in a strip that was designed for
present-tense work.

---

## The rule

> **Activity is present tense. History is what happened. Results live where
> results live.**

- The **Create activity strip** shows work that is happening, plus work that
  went wrong and still wants a decision. Nothing else. Ever.
- The **canvas** holds the result — a finished sequence lands there like a
  finished print does, with its actions on the caption.
- The **Library** holds every print, including sequence prints, and is where a
  sequence's settings are reused or its job re-entered.
- **Library ▸ History** grows a third tab, **Sequences**, which is the one
  place the durable server-side job list is enumerated, maintained, and
  cleaned up.

This keeps the five-workspace IA intact (§04: do not re-expand it), keeps the
spec's restraint budget (§10), and gives G2's "the Create activity strip renders
this client's own work" an honest boundary: this client's own work *in flight*.

---

# Part A — where settled work lives

## A.0 What the archived mockup actually shows

Worth recording, because it settles the argument. `docs/design/mold-studio-proposed-ui.html`
(the activity/queue strip, ~line 231) is a **single row**: the `Activity` kicker,
one shimmering running job with prompt + percent + a 4 px bar, and queued pills
each with a `✕`. There are no settled rows, no "+N more", no collapse chevron,
and no maintenance buttons. The prototype never contemplated settled work in the
strip at all. Today's desktop strip is strictly a superset that nobody designed.

The "+N more" recollection is worth honoring anyway — as an *overflow count*,
not an expandable pile. That is the digest chip in A.2.

## A.1 Lifespans and their homes

A job has three lifespans, and today all three are crammed into one surface.

| Lifespan | What it is | Home | Actions there |
| --- | --- | --- | --- |
| **In flight** | queued / running prints and sequences | Create activity strip; sidebar **Now developing** | cancel; watch (sequence) |
| **Needs a decision** | failed print; failed / interrupted sequence | Create activity strip, capped and expiring | resume, dismiss, read the error |
| **Settled** | completed / cancelled; failed once it has aged out | canvas (the one you just watched), **Library** (the print), **Library ▸ History ▸ Sequences** (the job record) | open, edit sequence, reuse settings, delete, clear inactive, clean up disk |

Two things follow that are worth stating plainly:

- **A completed sequence is a print.** It is a row in `/api/gallery` with
  `metadata.chain_job_id` and `metadata.chain`. It does not need a second
  representation in Create to be findable.
- **A settled sequence job is a server object, not a notification.** It holds
  cached clips, it is resumable, it is editable, and deleting it costs disk. That
  is a maintenance surface, and maintenance does not belong in the composer.

## A.2 Create activity strip — precise rules

Applies identically to `desktop/src/components/create/ActivityStrip.vue` and
`web/src/components/create/ActivityStrip.vue` (post-landing, which per impl-web
takes `jobs: Job[]` plus `sequences?: ActivityJobVM[]` and emits
`sequence-action`, `clear-inactive`, `cleanup-disk`).

**Rows rendered:**

1. **Active rows** — every `queued` / `running` print and sequence, ordered by
   the existing `mergeActivity` (active first, running before queued, then
   recency). Unchanged behavior, plus prints now sort correctly against
   sequences (see A.5 on `createdAtMs`).
2. **Attention rows** — a settled job that went wrong and has a next move:
   - print with `status === "error"`
   - sequence with `state === "failed"` or `state === "interrupted"`

   Nothing else. `completed` and `cancelled` never render a row: completion is
   not news the composer needs to carry, and a cancellation was the user's own
   decision one second ago.
3. **Digest** — one mono chip at the end of the header row when there is
   settled work the strip is deliberately not showing.

**Attention-row retention:**

- Visible while `now - settledAtMs < SETTLED_VISIBLE_MS` (**5 minutes**).
- At most `MAX_ATTENTION_ROWS` (**2**), newest first. Overflow is counted into
  the digest, never dropped silently.
- Each carries a `✕` **Dismiss** that removes it from the strip immediately.
  Dismiss is **client-side and non-destructive** — the chain job survives on the
  server and stays in History ▸ Sequences. This is the load-bearing distinction
  from `Clear inactive`, which is a server `DELETE`.
- Dismissal is **session-only, not persisted**. It does not need to be: the
  5-minute age rule is computed from server truth
  (`ChainJobSummary.updated_at_unix_ms`), so an app restart never resurrects a
  three-day-old failure. Persisting a dismissal set would be a second retention
  mechanism that can disagree with the first.
- `settledAtMs` for a sequence is `updated_at_unix_ms` (server). For a print it
  is a client stamp written when `status` flips to `error` (there is no server
  record of a failed print — see A.6).

**Digest chip:**

- Rendered only when `settledSequences > 0 || hiddenAttention > 0`.
- Label, mono 9.5 px, `--ink-3` (the `.ms-activity__maint` treatment), failed
  segment in `--stop`:
  - `4 settled sequences`
  - `1 failed · 4 settled sequences`
  - `3 failed` (when no settled sequences remain)
- Click → `/library?panel=history&tab=sequences`. One hop, `⌘[` comes back.
- Counts settled **sequences** only. Settled prints are not counted, because
  every settled print is already a Library row and the last three are already in
  **Now developing**; a count would be double bookkeeping.
- `title="Show settled sequences in History"`.

**Maintenance buttons move out.** `Clear inactive` and `Clean up disk` leave the
strip entirely and land in the History ▸ Sequences footer (A.4). They are
destructive, host-scoped, rarely used, and their presence in the composer is
what made the strip read as a control panel.

**Visibility gate** becomes `active.length || attention.length || digest !== null`.
A user with 40 settled sequences and nothing running sees a single 9.5 px chip,
not 40 rows. A user with nothing at all sees no strip, as today.

**Sequence action labels** stay as they are except one: `watch` renders as
**Watch** while the job is `queued`/`running` and **Open** once it is settled
(§11 voice — watch is present tense; a settled job is opened). Same action id on
the wire.

## A.3 Sidebar "Now developing" — keep the name, close the G14 hole

**Keep "Now developing".** The strip is already called Activity; naming the
sidebar section Activity too would give one product two words for two different
things that share a name. And §11 fixes *develop/developing* as the vocabulary
for generating. Renaming it costs vocabulary and buys nothing the digest chip
doesn't already deliver.

What changes:

- **It gains sequences.** Today `NavRail.railJobs` reads `generation.jobs`
  only, so a running sequence is invisible in the sidebar — a G14 hole the
  unified-Create landing left open. It should render live sequence entries from
  the same shared merge, with the clip counter (`clip 3/5 · developing…`) as its
  status line.
- **It does not gain settled sequences.** The existing `.slice(-3)` window of
  recently finished prints stays prints-only. The rail's own comment calls it "a
  working queue, not a full history"; adding settled sequences there would
  rebuild the pile in a second place, one route away from the one we just
  cleaned up. Settled sequences have two homes already (the print in Library, the
  job in History).
- Empty copy stays `nothing developing`.

## A.4 Library ▸ History ▸ Sequences (new third tab)

`desktop/src/components/library/HistoryDrawer.vue` already is "the past" —
**Runs** (gallery-backed) and **Prompts** (`prompt_history`-backed), opened at
`?panel=history`, reachable from the command palette. A durable sequence job is
the past *with actions attached*. It belongs here, not in a new workspace and
not in the status popover.

- **Tab label: Sequences.** Not "Jobs" — §11 has no "job" in the user-facing
  vocabulary, and the only durable job records that exist are sequences.
- **Source:** `useChainJobsStore.allJobs` (already per-host, already newest-first)
  → `sequenceToVM` → `mergeActivity`, so active jobs sort to the top and the tab
  is never lying about what is running.
- **Row** (shared component, A.7): state chip · model display name · `N clips ·
  M/N` · host label · relative time · error text when present.
- **Actions:** `sequenceActions(state)` unchanged — `Open`/`Watch`, `Edit`,
  `Resume`, `Delete` — plus **Show print** when a gallery row on that host
  carries `metadata.chain_job_id === job.id`. `Edit` and `Open` close the drawer
  and push `/create` through the sequence handoff (Part B, B.5); this is exactly
  the pattern `useRun` / `usePrompt` already use in this drawer.
- **Host scoping:** reuse the drawer's existing `HostFilterChips` bound to
  `gallery.filter`. `chains.clearInactive(hostId?)` already accepts an optional
  host, so the chip scopes the footer action for free.
- **Footer:** `Clear inactive` (two-press confirm, mirroring the Prompts tab's
  `Clear…`, label spelling out scope and count: `Delete 6 inactive sequences on
  plato?`) and `Clean up disk`. Both toast their outcome as they do today.
- **Render cap:** `HISTORY_JOBS_RENDER_CAP = 200`, newest first, with a mono
  footnote `showing 200 of 431` when clipped. No virtualization; a user at 431
  jobs is being told to run Clear inactive, which is right there.
- **Empty state:** `No sequences yet.` with exactly one primary action, `Go to
  Create` (§11 empty-state rule).
- **Freshness:** `chainJobs.syncPolling()` only polls while something is active,
  so the drawer must `void chains.fetchAll()` once on open (and on tab switch to
  Sequences) or a settled list goes stale.
- **URL:** the drawer's `tab` ref becomes URL-synced via `?tab=runs|prompts|sequences`
  so the digest chip can deep-link. Default stays `runs` on desktop.

**Web** has no History drawer at all today. It gets `?panel=history` with the
same `DrawerPanel`, initially carrying only the Sequences tab (the tab bar
renders only when more than one tab exists, so it reads as a titled drawer, not a
lonely tab). Runs/Prompts parity on web stays a separate backlog item — it is not
what this change is about, and blocking on it would leave web's digest chip
pointing nowhere.

> **Gotcha (previously bitten):** `@ui` `DrawerPanel`/`ModalPanel`/`SheetPanel`
> are `position:absolute; inset:0` — they render inside their owning frame per
> spec §05. On a scrolling web page they land off-screen unless wrapped in a
> `fixed inset-0` viewport host. Web's drawer mount must include that wrapper.

## A.5 The canvas holds the result

This is the piece that makes removing settled rows non-regressive, and it is
required, not optional. Without it, "edit the sequence I just made" goes from one
click to three.

Today `watchedSequence` in `GenerateView.vue` returns `null` unless
`state === "running" || state === "queued"`, so at the instant a sequence
finishes the canvas falls back to the empty state and the only surviving trace is
the strip row we are deleting.

New canvas state — **settled watched sequence**:

- When the watched job settles, the canvas shows the finished video (poster
  frame, playable) with the standard caption line: `model · N clips · seed · time
  · host`.
- Caption actions: **Edit sequence** and **Show in library**.
- Resolution of the video: after finalize the store already bumps
  `chains.finalizedTick` and Create refetches the gallery; find the row whose
  `metadata.chain_job_id === jobId` on that host (shared predicate, B.5). If no
  row resolves within one refetch, fall back to the last stage preview with the
  caption actions still present — never a blank canvas.
- On failure of the watched job, the canvas shows the existing
  `GenerateErrorNotice` treatment with `Resume` — matching spec §10's "a failed
  print is still an object you can inspect and retry".
- Completion toast: `Sequence ready — saved to Library` (verb→noun rule).

**Related correctness fix required by this work:** `ActivityJobVM.createdAtMs`
for prints is currently `job.clientId`, a 1,2,3… counter, not a timestamp. It is
harmless today only because every print VM is filtered back out after the merge.
The moment a failed print keeps a row, it sorts against sequence
`created_at_unix_ms` values (~1.7e12) and always loses. `Job` needs a real
`createdAtMs` (wall clock at submit) and `settledAtMs` (stamped on settle).

## A.6 Failed jobs — how prominent, where, how long

| | Failed print | Failed / interrupted sequence |
| --- | --- | --- |
| Server record | none (`generation.jobs` is session state; web persists 10 in localStorage) | durable job row, resumable |
| Strip | attention row, 5 min or until dismissed, error text inline | attention row, 5 min or until dismissed, error text inline, `Resume` |
| After it ages out | gone — and that is honest, it was never recorded | still in History ▸ Sequences with `Resume` / `Edit` / `Delete` |
| Canvas | existing `GenerateErrorNotice` for the active job | new: error notice + `Resume` (A.5) |
| Sidebar | last-3 window shows it in `--stop` (existing behavior) | not shown (A.3) |

Note this **adds** failed-print rows to desktop, which shows none today
(`generation.pending` already excludes `error`). That is deliberate: web's strip
has had dismissible error rows all along, with a test asserting "keeps failed
jobs visible with the server error until dismissed", and desktop losing failures
into a toast is the quieter half of the same bug. Cross-surface parity, and it is
the honest counterweight to removing 40 completed rows.

## A.7 Retention constants (single source, `studio/lib/activity.ts`)

| Constant | Value | Meaning |
| --- | --- | --- |
| `SETTLED_VISIBLE_MS` | `5 * 60_000` | how long a settled-but-wrong job keeps a strip row |
| `MAX_ATTENTION_ROWS` | `2` | strip rows for wrong-but-settled work; overflow → digest |
| `HISTORY_JOBS_RENDER_CAP` | `200` | rows rendered in History ▸ Sequences |
| `RAIL_SETTLED_KEEP` | `3` | existing sidebar last-finished window (unchanged) |
| `PRINT_PRUNE_KEEP` | `12` | existing `generation.prune` cap (unchanged) |

## A.8 Mobile (iPhone) — brief

The iPhone Create queue already renders `generation.pending` only, so it has no
pile-up to fix. It has no `useChainJobsStore` (it must not import desktop-primary
state) and no cross-host chain listing, so **History ▸ Sequences is out of scope
on iPhone** and the digest chip is omitted.

What mobile gets:

- The same attention-row treatment for a failed print / failed sequence in its
  queue section, dismissible, using the shared `partitionActivity`.
- Its door to settled sequences is the **Library viewer**, via Part B — which is
  where an iPhone user is looking for a finished video anyway.
- Its single `sequenceJob` card keeps its current recovery behavior (exact
  `instanceId` match before reattaching) untouched.

## A.9 Rejected alternatives

- **Collapsible settled section in the strip, collapsed by default** — still a
  pile, just folded; the count is the only part anyone reads, and it keeps
  maintenance in the composer.
- **Auto-dismiss settled rows after N seconds with no digest** — silently
  destroys the only pointer to a resumable failed job.
- **Sixth workspace / "Jobs" destination** — §04 says do not re-expand the IA.
- **Job history in the StatusPopover** — it answers "how is the engine", not
  "what did I make"; a 256 px popover anchored above a 32 px trigger is the wrong
  container for an actionable list.
- **Rename "Now developing" → "Activity"** — two surfaces named Activity, and it
  contradicts §11's develop/developing vocabulary.
- **Settled sequences in the sidebar last-3 window** — rebuilds the pile one
  route away.
- **Persisting dismissals to localStorage** — a second retention mechanism that
  can disagree with the age rule; the age rule already survives restart because it
  reads server timestamps.
- **Server-side job retention/TTL** — right idea, wrong PR; `POST /api/chain-jobs/gc`
  already exists and `Clear inactive` is the user-facing control. Note as follow-up.
- **Making the strip a real shared queue with reorder** — that is G2's *Machines*
  half, already resolved there via `PATCH /api/queue/:id`.

---

# Part B — reuse a sequence's settings from the Library

## B.1 One button, sequence-aware

A print's lightbox/viewer keeps **exactly one** reuse entry point, whose behavior
follows the print:

- `metadata.chain` **absent** → today's behavior, unchanged: `Reuse settings`
  loads One shot with that row's model/size/steps/guidance/seed/prompt.
- `metadata.chain` **present** → `Reuse settings` loads a **fresh sequence
  draft**: Output = Sequence, clips from `metadata.chain.stages`, shared params
  from the row's metadata. Not a second button, not a submenu.
- `metadata.chain_job_id` **also present** and the origin host resolves → a
  second, distinct action appears: **Edit sequence**, which re-enters the
  edit-in-place session on the original job with its cached clips.

The distinction users must feel: **Reuse settings makes a new sequence. Edit
sequence continues the old one** (cached clips preserved, `Update sequence`
re-renders only from the earliest changed clip).

This also fixes a small existing wart: `synthetic_generate_request` records a
multi-clip sequence's `metadata.prompt` as the clips **newline-joined**, so
today's single-print reuse on a sequence print dumps five prompts into the One
shot box. Sequence-aware reuse never surfaces that join.

## B.2 What maps, and what is honestly lost

`ChainStageMetadata { prompt, frames, transition, fade_frames, seed }` vs
`SequenceClipForm { id, prompt, frames, transition, fadeFrames, negativePrompt, sourceImage }`.

| Field | Source | Rule |
| --- | --- | --- |
| `prompt` | `chain.stages[i].prompt` | direct |
| `frames` | `chain.stages[i].frames` | direct, then clamped (below) |
| `transition` | `chain.stages[i].transition` | direct; clip 0 coerced to `smooth` by the form, as always |
| `fadeFrames` | `chain.stages[i].fade_frames` | direct, else `DEFAULT_FADE_FRAMES` |
| `id` | — | fresh client uuid per clip |
| `negativePrompt` | `metadata.negative_prompt` | **clip 1 only.** `synthetic_generate_request` records `first.negative_prompt` — clip 1's. Spraying it across every clip would invent authorship. Clips 2..n get `""`. |
| `sourceImage` | `metadata.source_image_sha256` | **desktop only, clip 1 only.** The recorded sha is clip 1's opening frame; desktop's Tauri source stash (`ipc.sourceStashGet`) can restore it when still present. Web and iPhone leave it `null`. |
| per-clip seed | `chain.stages[i].seed` | **not applied.** The rail carries no per-clip seed (the wire carries `seed_offset`), and deriving an offset from an effective u64 seed is guesswork. The shared seed comes from `metadata.seed` exactly as single-print reuse does. |
| shared params | row metadata | `applyMetadataToForm` unchanged — model, width/height, steps, guidance, fps, seed, strength, format |
| `motion_tail_frames` | `chain.motion_tail_frames` | **validation only**, not applied — the live tail is `sequenceMotionTailFrames(selectedEntry)`, a property of the model. Used to clamp (below). |

**Frames clamp.** Clip duration must stay strictly greater than the active motion
tail (an existing invariant). If the model's current tail differs from the
recorded one, any clip with `frames <= motionTail` is raised to the model's
minimum and the surface says so once: `Clip durations raised to fit
flux… — the model's motion tail changed.` Never silently submit an invalid draft;
never silently resize without saying it.

**Lossiness is disclosed, once, quietly.** After a sequence reuse the composer
shows a single mono line beneath the rail:
`reused 5 clips · negatives and clip sources aren't recorded in prints`
— only listing the parts that were actually lossy for *this* row. This is the
§11 rule (errors and caveats say what happened, no apology) and it is what keeps
"Reuse settings" from over-claiming reproducibility.

## B.3 Edit sequence — existence and host resolution

**Origin host.** The chain job lives only on the machine that produced it.

| Surface | Row wrapper | Host id |
| --- | --- | --- |
| desktop | `MergedPrint` | `gallery.hostFor(entry.sourceKey)?.id` |
| web | `HostGalleryImage` | `entry.hostId` (`ORIGIN_HOST_ID` → origin host) |
| iPhone | `GalleryPrint` | `print.hostId` / `print.target` |

Rules:

- Resolve **only** from the entry's own origin bucket. A merged print may appear
  in `availableOn` on three hosts; the other two hold auto-saved *copies*, not the
  producer. Never probe siblings.
- Desktop's `hostFor("local")` returns `null` unless the local server is
  `ready` — an unresolved host means **no Edit action**, only Reuse.
- **Never probe eagerly.** No per-row or on-open existence check across N hosts.
- **Positive-knowledge-only hiding:** if that host's chain listing is already
  loaded (`chains.byHost[hostId]`) and does not contain the id, hide **Edit
  sequence**. If the listing is not loaded, show it — absence of evidence is not
  evidence of absence.
- **Check on click**, once: `chains.fetchDetail(hostId, jobId)`.
  - `404` → the job was deleted or GC'd. **Auto-fall back to the Reuse path**
    and toast `That sequence job is gone from plato. Reused its settings
    instead.` The click is never a dead end (the same rule the prepared-expansion
    invariant states: never leave an enabled control as a text-only dead end).
  - network / 5xx → `Can't reach plato:7680. Check the host in Machines.`
    (§11 server-error copy). Do **not** silently downgrade to Reuse — the job
    probably still exists and a silent downgrade would throw away cached clips.
  - `200` → hand straight into the **existing** `editSequence` flow:
    `normalizeServerChainScript(detail.script)` → `chainScriptToClips` →
    `draft.loadFromJob({ jobId, hostId, baseline, completedStages })`. That path
    is lossless (real negatives, real source images, real cached-stage count) and
    already tested; Library must not grow a parallel one.
  - `detail.script == null` → existing toast `This job carries no editable
    script.`, then offer Reuse.
- **Web needs typed status.** `web/src/api.ts requireJson` throws a plain `Error`
  with the status in the message string; the 404 branch must not be implemented
  by substring-matching. Web gets the studio `ApiError` (or a thin typed wrapper)
  on the chain-job calls.

## B.4 Legacy rows (pre-#564)

- **No `chain`, no `chain_job_id`** → the print offers only today's single-print
  `Reuse settings`, loading One shot with the recorded prompt (which, for a
  multi-clip sequence, is the newline join — all we ever recorded). No synthesized
  one-clip sequence, no fabricated stages. Missing metadata stays omitted rather
  than guessed.
- **`metadata_synthetic` rows** (scanned from disk without sidecars) behave the
  same.
- `chain_job_id` and `chain` are written together by `stitched_output_metadata`,
  so "job id without stages" is not a real state; if it is ever seen, treat it as
  legacy (Reuse only, no Edit) rather than trusting half a record.
- Ephemeral chain outputs (`routes_chain.rs` passes `chain_job_id: None`) carry
  `chain` but no job id → **Reuse only**. Correct: there is no durable job to
  edit.

## B.5 Where the code goes

**New shared module `studio/lib/sequenceReuse.ts`** — pure, injected deps, no
store imports, no Tauri:

```ts
export function chainMetadataToClips(chain: ChainOutputMetadata,
  opts: { negativePromptForFirstClip?: string | null }): SequenceClipForm[];

export interface SequenceReusePlan {
  clips: SequenceClipForm[];
  recordedMotionTailFrames: number;
  lossy: { negatives: boolean; clipSources: boolean; perClipSeeds: boolean };
}
export function planSequenceReuse(metadata: OutputMetadataLike): SequenceReusePlan | null;

export function clampClipsToMotionTail(clips: SequenceClipForm[],
  motionTailFrames: number, minFrames: number): { clips: SequenceClipForm[]; raised: number };

export type SequenceEditAvailability = "available" | "absent" | "unknown-host";
export function sequenceEditAvailability(args: {
  chainJobId: string | null | undefined;
  hostId: string | null;
  knownJobIds: readonly string[] | null;   // null = listing not loaded
}): SequenceEditAvailability;

/** job → print, for the canvas result and History's "Show print". */
export function isPrintOfChainJob(metadata: OutputMetadataLike, jobId: string): boolean;
```

`OutputMetadataLike` is a structural minimum (`{ chain?, chain_job_id?,
negative_prompt? }`) so it accepts all three surfaces' divergent
`OutputMetadata` interfaces without unifying them (that unification is a real
but separate cleanup).

**Chain provenance TS types land once** in `studio/lib/api/chainTypes.ts`
(`ChainStageMetadata`, `ChainOutputMetadata`), mirroring `crates/mold-core/src/chain.rs`,
and both `desktop/src/lib/api/types.ts` and `web/src/types.ts` add
`chain_job_id?: string | null` and `chain?: ChainOutputMetadata | null` to
`OutputMetadata` by importing from there. These fields already cross the wire
today; only the TS types are missing them.

**Per-surface handoff** (the route hop is surface-specific and stays that way):

- **Desktop** — `composer.ts` grows a one-shot sequence slot beside the existing
  prefill: `pendingSequence: SequenceHandoff | null` with `setSequence()` /
  `takeSequence()`, where
  `SequenceHandoff = { kind: "reuse"; metadata } | { kind: "edit"; hostId; jobId }`.
  `GenerateView.applyPrefill()` grows a sibling `applySequenceHandoff()` that runs
  `draft.stopEditing()` + `draft.setOutput("sequence", …)` + clips for `reuse`, or
  the existing `editSequence(payload)` for `edit`.
- **Web** — mutates the shared `useGenerateForm` state and the `@studio` draft
  store directly from `LibraryPage.vue`, then `router.push({ name: "create" })`.
  No handoff store needed.
- **iPhone** — `MobileApp.reusePrint()` already switches `selectedHostId` to the
  print's host before applying metadata; the sequence branch extends that
  (host-switch, refresh models, then clips + `tab = "generate"`). Keychain-only
  keys, snapshot record unchanged. **Reuse only on iPhone in this pass**; Edit
  sequence follows once mobile has a chain-detail fetch on the recovery route.

## B.6 Rejected alternatives

- **Two buttons always (`Reuse settings` + `Reuse as sequence`)** — the print
  already knows what it is; asking the user is a shrug in button form.
- **Deriving `seed_offset` from the recorded per-stage effective seed** —
  u64 arithmetic against a server-side derivation we do not own; a wrong offset
  silently produces a different sequence that claims to be a reuse.
- **Synthesizing a one-clip sequence for legacy rows** — invents provenance
  that was never recorded.
- **Eagerly probing every sequence print's job on Library load** — N rows × N
  hosts of chatter to decide whether to render one button.
- **Hiding `Edit sequence` whenever the listing hasn't loaded** — absence of
  evidence rendered as evidence of absence; the click-time check is cheap.
- **Falling back to Reuse on an unreachable host** — silently throws away cached
  clips that almost certainly still exist.
- **Probing sibling hosts from `availableOn`** — those are auto-saved copies, not
  the producer; a hit there would edit an unrelated job.
- **A parallel Library-side clip loader instead of reusing `editSequence`** — the
  `ChainScript` path is lossless and already tested; a second one would drift.
- **Unifying all three `OutputMetadata` interfaces into `studio/` in this PR** —
  correct, and large enough to bury the feature.

---

# Spec edits needed

`docs/design/mold-studio-spec.html` — small, surgical, in the spec's voice.

1. **§06 "Generate"** — append to the activity-strip sentence: *"The strip is
   present tense: it mirrors the running job and any queued jobs (cancellable),
   holds a failed job briefly so it can be resumed or dismissed, and collapses
   everything else into a count that opens History."*
2. **§06 "Make a sequence"** — amend the final sentences. Current text says
   sequences "render in the *same* activity strip as prints (queued/running/settled,
   watch, cancel, resume)". Change `settled` out of that list, and add: *"A settled
   sequence leaves the strip: its video lands in the canvas with **Edit sequence**
   and **Show in library**, its print is in the Library carrying every clip's
   prompt, and its job record is in **Library ▸ History ▸ Sequences**, which is
   also where **Clear inactive** and **Clean up disk** live. From a sequence print,
   **Reuse settings** starts a fresh sequence from the recorded clips; **Edit
   sequence** re-enters the original job with its cached clips."*
3. **G2 resolution** — replace "with watch/cancel/resume/edit per row" with
   "…renders this client's own work *while it is in flight*, plus a failed job
   briefly; settled work resolves to Library ▸ History ▸ Sequences (G15)."
4. **G14 resolution** — append: "settled sequence jobs live in Library ▸ History
   ▸ Sequences, and a sequence print reloads its clips into the rail from
   `metadata.chain`."
5. **New gap G15 — "Activity is present tense."** *Settled jobs accumulate in the
   Create activity strip instead of resolving to a result and a history.* **Done
   when** the Create strip shows only in-flight work plus a capped, expiring
   attention row, a finished sequence lands in the canvas with its actions, and
   the durable job list with its maintenance actions lives in Library ▸ History ▸
   Sequences on desktop and web. **Resolution:** *(fill on landing.)*
6. **§11 copy table** — add rows:
   - Activity digest — `4 settled sequences` · `1 failed · 4 settled sequences`
   - Sequence toast — `Sequence ready — saved to Library`
   - Gone job — `That sequence job is gone from plato. Reused its settings instead.`
   - Lossy reuse — `reused 5 clips · negatives and clip sources aren't recorded in prints`
   - Clear inactive — `Delete 6 inactive sequences on plato?`

`docs/design/README.md` needs no change. Note in the spec's §06 that the archived
prototype's activity strip was always live-only (A.0) — it is provenance for the
change, not a contradiction of it.

---

# Implementation plan, file by file

### `studio/` (shared, no shell imports)

| File | Change |
| --- | --- |
| `studio/lib/activity.ts` | Add `settledAtMs: number \| null` to both VM variants. Export `isSettled`, `needsAttention`, `SETTLED_VISIBLE_MS`, `MAX_ATTENTION_ROWS`, `partitionActivity(rows, { nowMs, dismissed, settledVisibleMs, maxAttentionRows })` → `{ active, attention, settledSequences, hiddenAttention }`, and `activityDigestLabel(partition)`. `sequenceToVM` sets `settledAtMs = updated_at_unix_ms` when settled. Keep `mergeActivity` as-is (partition consumes its output). |
| `studio/lib/api/chainTypes.ts` | Add `ChainStageMetadata`, `ChainOutputMetadata` mirroring `crates/mold-core/src/chain.rs`. |
| `studio/lib/sequenceReuse.ts` | **New.** `chainMetadataToClips`, `planSequenceReuse`, `clampClipsToMotionTail`, `sequenceEditAvailability`, `isPrintOfChainJob`. |
| `studio/lib/sequenceForm.ts` | No change (`chainScriptToClips` stays the lossless `ChainScript` path used by Edit). |

### `ui/`

| File | Change |
| --- | --- |
| `ui/components/SequenceJobRow.vue` | **New.** Presentational, props-in/emits-out (`vm: ActivityJobVM & { kind: "sequence" }`, `dense?: boolean`, `actions`, `modelLabel`), emits `action`. Same precedent as `SeamPill`/`ClipRail`/`ClipPill`. Rendered by the strip's attention rows, History ▸ Sequences, and web's drawer. |
| `ui/components/ProgressBar.vue`, `DrawerPanel.vue`, `EmptyStateBlock.vue` | No change; reused. |

### `desktop/`

| File | Change |
| --- | --- |
| `desktop/src/lib/generationJob.ts` | Add `createdAtMs` (wall clock at submit) and `settledAtMs` to `Job`. |
| `desktop/src/stores/generation.ts` | Stamp both; stamp `settledAtMs` where `status` flips to `complete`/`error`. `prune` unchanged. |
| `desktop/src/components/create/ActivityStrip.vue` | The main edit. `printVMs.createdAtMs = job.createdAtMs`; build VMs from **all** jobs (not just `pending`); run `partitionActivity`; render active rows + attention rows (via `SequenceJobRow` / a print equivalent) + digest chip; local `dismissed: Set<string>`; **remove** `Clear inactive` / `Clean up disk`; digest routes to `/library?panel=history&tab=sequences`; `Watch`→`Open` label rule. |
| `desktop/src/views/GenerateView.vue` | New settled-watched-sequence canvas state + caption actions (`Edit sequence`, `Show in library`); resolve the print via `isPrintOfChainJob`; failed-watched-job notice with `Resume`; completion toast; consume `composer.takeSequence()` in a new `applySequenceHandoff()`. |
| `desktop/src/components/shell/NavRail.vue` | `railJobs` gains live sequence entries via the shared merge (clip counter as status line); settled window stays prints-only at 3. |
| `desktop/src/components/library/HistoryDrawer.vue` | Third tab **Sequences**; `tab` synced to `?tab=`; `chains.fetchAll()` on open/tab-switch; rows via `SequenceJobRow`; `Show print` action; footer `Clear inactive` (host-scoped, two-press) + `Clean up disk`; render cap + footnote; empty state. |
| `desktop/src/views/LibraryView.vue` | Lightbox + tile menu: sequence-aware `Reuse settings`; `Edit sequence` gated by `sequenceEditAvailability`; 404 auto-fallback; pass origin host id through. |
| `desktop/src/components/gallery/Lightbox.vue` | New emits `reuse-sequence`, `edit-sequence`; new props `canEditSequence`, `isSequence`. |
| `desktop/src/stores/composer.ts` | `pendingSequence` + `setSequence()` / `takeSequence()`. |
| `desktop/src/lib/api/types.ts` | `OutputMetadata` gains `chain_job_id`, `chain` (imported from `@studio`). |
| `desktop/src/components/shell/StatusPopover.vue` | **No change** (deliberate). |

### `web/`

| File | Change |
| --- | --- |
| `web/src/components/create/ActivityStrip.vue` | Same partition/digest/attention treatment; keeps its existing per-row `dismiss` for prints and extends it to sequences; drops `clear-inactive` / `cleanup-disk` emits. |
| `web/src/pages/CreatePage.vue` | Stop passing the maintenance handlers; add the settled-sequence canvas result + caption actions; digest routes to `/library?panel=history&tab=sequences`. |
| `web/src/composables/useGenerateStream.ts` | Add `settledAt` to `Job`; keep `SETTLED_HISTORY_CAP`/`AUTO_REMOVE_DONE_MS` for successful prints. |
| `web/src/composables/useChainJobs.ts` (landing) | Expose the per-host listing to the new drawer; add a `fetchAll()`-equivalent the drawer can call on open. |
| `web/src/components/library/HistoryDrawer.vue` | **New.** `DrawerPanel` in a `fixed inset-0` host, Sequences tab only, tab bar hidden at one tab; same rows/footer/cap/empty state as desktop. |
| `web/src/pages/LibraryPage.vue` | `?panel=history` open state + toolbar button; sequence-aware `onReuse`; `Edit sequence` with the same availability + 404 rules. |
| `web/src/components/gallery/Lightbox.vue` | New `reuse-sequence` / `edit-sequence` emits. |
| `web/src/api.ts` | Typed status on chain-job errors (adopt studio `ApiError`); per-host `listChainJobs` / `deleteChainJob` (currently origin-hardwired). |
| `web/src/types.ts` | `OutputMetadata` gains `chain_job_id`, `chain`. |
| `web/src/lib/sequenceParams.ts` | Reuse path applies shared params via the landing's `applySharedToForm`. |

### `desktop/src/mobile/` (iPhone)

| File | Change |
| --- | --- |
| `MobileApp.vue` | Attention-row treatment in the queue section via `partitionActivity`; `reusePrint()` gains the sequence branch (host switch → models refresh → clips + shared params → `tab = "generate"`). No History ▸ Sequences, no digest chip. |
| `desktop/src/mobile/reuse.ts` | `applyMobileGalleryMetadata` gains a sequence result (`{ clips, lossy }`) so the model-substitution logic stays in one place. |
| `MobileGalleryViewer.vue` | `Use as prompt` becomes sequence-aware for a sequence print (label stays; behavior loads the clip rail). |

### Rust / server

No changes required. `chain_job_id` and `chain` already reach `GET /api/gallery`;
`GET /api/chain-jobs/:id` already 404s for a missing job. (Unrelated: the
`POST /api/chain-jobs/:id/amend` route registration is tracked by the in-flight
server work, not this note.)

### Docs to update on landing

`CHANGELOG.md` `[Unreleased]`, `README.md` if it describes the strip,
`CLAUDE.md`/`AGENTS.md` (activity/history invariant + sequence-reuse invariant),
`crates/mold-cli/src/skill/SKILL.md`, `desktop/docs/`, `apps/mobile/README.md`,
`website/` Library/Create pages, and the spec edits above.

---

# TDD test list

Failing test first, in this order.

**`studio/lib/activity.test.ts`** (extend)

1. `partitionActivity` puts queued/running rows in `active` and never in `attention`.
2. A `completed` sequence appears in neither `active` nor `attention`, and counts once in `settledSequences`.
3. A `cancelled` sequence never produces an attention row.
4. A `failed` sequence settled 1 minute ago produces an attention row; settled 6 minutes ago does not, and counts into `settledSequences`.
5. An `interrupted` sequence is treated as attention (resumable).
6. A failed print produces an attention row keyed distinctly from sequences.
7. `maxAttentionRows: 2` keeps the two newest and reports `hiddenAttention: 1`.
8. `dismissed` keys drop rows from `attention` without changing `settledSequences`.
9. `activityDigestLabel` → `null` / `4 settled sequences` / `1 failed · 4 settled sequences` / `3 failed`.
10. `sequenceToVM` sets `settledAtMs` from `updated_at_unix_ms` only for settled states, `null` while active.
11. Regression: a print VM with a real millisecond `createdAtMs` sorts by recency against sequence rows (guards the `clientId`-as-timestamp bug).

**`studio/lib/sequenceReuse.test.ts`** (new)

12. `chainMetadataToClips` maps prompt/frames/transition/fade for a 3-stage chain, fresh unique ids.
13. Clip 0's transition is coerced to `smooth` regardless of what was recorded.
14. Missing `fade_frames` → `DEFAULT_FADE_FRAMES`.
15. `negativePromptForFirstClip` lands on clip 1 only; clips 2..n are `""`.
16. Per-stage `seed` values are ignored (no per-clip seed leaks into the form) and `lossy.perClipSeeds` is true when they were present.
17. `planSequenceReuse` returns `null` for metadata with no `chain` (legacy row).
18. `planSequenceReuse` reports `lossy.negatives` / `lossy.clipSources` accurately for a row that had neither.
19. `clampClipsToMotionTail` raises only clips at/below the tail and reports `raised`.
20. `sequenceEditAvailability`: `absent` when the listing is loaded and lacks the id; `available` when the listing is not loaded (`knownJobIds: null`); `unknown-host` when `hostId` is null; `absent` when `chainJobId` is null.
21. `isPrintOfChainJob` matches on `chain_job_id` and rejects a same-filename row without it.

**`desktop/src/components/create/ActivityStrip.test.ts`** (extend; existing cases mostly stay)

22. Two completed sequences and nothing running → no `activity-sequence` rows, one digest chip reading `2 settled sequences`.
23. A failed sequence 1 minute old → one row with its error and `Resume`; `✕` dismiss removes it **without** calling `chains.remove` (non-destructive).
24. A failed sequence 6 minutes old → no row; counted in the digest.
25. Three failed jobs → two rows and `1 failed · …` in the digest.
26. Digest click pushes `/library?panel=history&tab=sequences`.
27. `activity-clear-inactive` and `activity-cleanup` are **gone** from the strip (the existing "strip-level Clear inactive" test moves to the drawer suite).
28. A settled sequence's watch button is labelled `Open`; a running one's is `Watch`.
29. Strip is hidden when everything is settled and there is no digest (all sequences cleared).

**`desktop/src/components/library/HistoryDrawer.test.ts`** (extend)

30. Sequences tab lists jobs from every host, active first.
31. `?tab=sequences` opens the drawer on that tab; switching tabs updates the query.
32. Opening the drawer triggers one `chains.fetchAll()`.
33. Row actions route to the owning host (`resume`/`delete` targeting the right `hostId`).
34. `Show print` appears only when a gallery row on that host carries the matching `chain_job_id`, and opens it in the Library.
35. `Clear inactive` scopes to the chip-selected host, two-press confirm, toasts the outcome.
36. 431 jobs → 200 rows and `showing 200 of 431`.
37. Empty state renders with one primary action.

**`desktop/src/views/GenerateView.test.ts`** / `LibraryView.test.ts`

38. A watched sequence that completes renders the canvas result with `Edit sequence` + `Show in library` (not the empty state).
39. A watched sequence that fails renders the error notice with `Resume`.
40. `composer.takeSequence({ kind: "reuse" })` populates the clip rail and switches Output to Sequence, with no `draft.editing` session.
41. `composer.takeSequence({ kind: "edit" })` runs the existing `editSequence` path and sets `draft.editing`.
42. Lightbox on a sequence print shows `Edit sequence`; on a legacy print it does not.
43. `Edit sequence` on a 404 job falls back to Reuse and toasts `That sequence job is gone from …`.
44. `Edit sequence` on an unreachable host toasts the Machines copy and does **not** fall back.
45. Reuse on a print whose model's motion tail grew raises clip frames and says so once.

**`desktop/src/components/shell/NavRail.test.ts`** (extend)

46. A running sequence appears in **Now developing** with its clip counter.
47. A settled sequence does **not** appear in the rail.

**`web/src/components/create/ActivityStrip.test.ts`** / `CreatePage.test.ts` / new `HistoryDrawer.test.ts`

48. Mirror of 22–26 and 29 on web.
49. The existing "keeps failed jobs visible with the server error until dismissed" case still passes (print path unchanged).
50. Web drawer renders inside a `fixed inset-0` host (guards the off-screen `@ui` panel bug).
51. Web sequence reuse mirrors 40, and `Edit sequence` mirrors 43/44 through the typed-status client.

**`desktop/src/mobile/MobileApp.test.ts`**

52. A failed mobile job renders a dismissible attention row.
53. `reusePrint` on a sequence print switches to the print's host, loads the clip rail, and lands on the Create tab.

**`ui/components/SequenceJobRow.test.ts`** (new)

54. Renders state/model/clip-count/host; emits `action` with the action id; `dense` variant drops the progress region.

---

# PR structure

**Two PRs.** They touch overlapping files (`ActivityStrip.vue`, `GenerateView.vue`,
`LibraryView.vue`) but answer different questions, and reviewing "what does the
strip show" together with "how does a print become a clip rail" would bury both.

### PR 1 — `fix(desktop,web): activity is present tense; settled sequences move to History`

Part A in full. `studio/lib/activity.ts` partition + digest, `Job` timestamps,
both strips, the settled-sequence canvas result, NavRail sequences, desktop's
History ▸ Sequences tab, web's History drawer, `ui/components/SequenceJobRow.vue`,
and the minimal `isPrintOfChainJob` predicate (needed by the canvas result).
Spec §06/G2/G14/G15/§11 edits. Tests 1–11, 21, 22–39, 46–50, 54.

Roughly: 2 new files, ~14 edited, ~45 tests. The largest single piece is web's
new drawer; the riskiest is the `GenerateView` canvas state.

### PR 2 — `feat(desktop,web,ios): reuse a sequence's settings from the Library`

Part B in full. `studio/lib/sequenceReuse.ts`, chain provenance TS types on all
three surfaces, lightbox/viewer actions, the composer sequence handoff, web's
typed chain-job status, and the iPhone reuse branch. Tests 12–20, 40–45, 51–53.

Roughly: 1 new file, ~12 edited, ~20 tests.

**Order matters:** PR 1 first. PR 2's `Edit sequence` from the Library and PR 1's
canvas `Edit sequence` land on the same `applySequenceHandoff()` seam, and PR 1
establishes the `?panel=history&tab=sequences` route contract that PR 2's
"job is gone" fallback story assumes.

Splitting either PR by surface (desktop then web) is possible but not
recommended: the shared `studio` predicate is the whole point, and landing
desktop-only would leave web's strip piling up against a spec that says it
shouldn't.

---

# Risks and gotchas

- **`@ui` overlay panels are `position:absolute; inset:0`.** Web's new drawer
  needs a `fixed inset-0` viewport host or it renders off-screen. Previously bitten.
- **`printVMs.createdAtMs = job.clientId`** is a counter, not a timestamp. It
  must be fixed *before* print rows survive the merge, or every failed print sorts
  to the bottom forever.
- **Chain polling stops when nothing is active** (`syncPolling`), so any surface
  that lists settled sequences must fetch on open.
- **`Clear inactive` is a server `DELETE`, dismiss is not.** Do not let the two
  converge in the UI or in a shared helper; the confirm copy must name the host and
  the count.
- **`reactive()`-wrap** any object the new dismissal/partition state mutates from
  SSE or closure callbacks in Vue stores — raw-object mutation bypasses proxy
  traps and freezes the UI.
- **Three divergent `OutputMetadata` interfaces** get one more additive field
  each. Unifying them into `studio/` is the right cleanup and explicitly not this
  PR; the structural `OutputMetadataLike` keeps the new helper honest meanwhile.
- **Server-side job retention** (a TTL or a cap on `chain_jobs`) is the real
  long-term answer to a 431-job list. Out of scope; `Clear inactive` and
  `POST /api/chain-jobs/gc` are the controls until then.
