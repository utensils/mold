/*
 * One-shot handoff of a durable sequence from elsewhere in the app into
 * Create. Web has no store layer for this, so the slot is a module singleton
 * (same pattern as useGenerateStream / useChainJobs): the Library ▸ History
 * drawer sets it, Create takes it on arrival, and taking empties it so a
 * back-nav can never replay the handoff.
 *
 * `edit` re-enters the original job in place, preserving its cached clips.
 * The Library's settings-reuse variant lands on this same slot in a follow-up.
 */
import { ref, type Ref } from "vue";

export interface SequenceHandoff {
  kind: "edit";
  hostId: string;
  jobId: string;
}

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
