/*
 * One-shot handoff of a durable sequence from elsewhere in the app into
 * Create. Web has no store layer for this, so the slot is a module singleton
 * (same pattern as useGenerateStream / useChainJobs): the Library ▸ History
 * drawer sets it, Create takes it on arrival, and taking empties it so a
 * back-nav can never replay the handoff.
 *
 * - `edit` re-enters the original job in place, preserving its cached clips.
 * - `inspect` loads and watches a job without granting amend authority.
 * - `reuse` starts a FRESH draft from a print's recorded per-clip provenance
 *   (`metadata.chain`) — a new sequence, no edit session, nothing cached.
 *
 * The reuse variant travels as metadata rather than as finished clips because
 * clip durations must be clamped against the motion tail of the model that is
 * selected in CREATE, which the Library has no view of.
 */
import { ref, type Ref } from "vue";
import type { OutputMetadata } from "../types";

export type SequenceHandoff =
  | { kind: "edit"; hostId: string; jobId: string }
  | { kind: "inspect"; hostId: string; jobId: string }
  | { kind: "reuse"; metadata: OutputMetadata };

const pending = ref<SequenceHandoff | null>(null);

export function setSequenceHandoff(handoff: SequenceHandoff): void {
  pending.value = handoff;
}

export function takeSequenceHandoff(): SequenceHandoff | null {
  const handoff = pending.value;
  pending.value = null;
  return handoff;
}

/** Reactive read for the consuming view's watcher. Never mutate directly. */
export function pendingSequenceHandoff(): Ref<SequenceHandoff | null> {
  return pending;
}
