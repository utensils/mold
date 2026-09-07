/**
 * Plain-English queue vocabulary (README §02): Being made · Waiting ·
 * Finished · Needs a download first. Every sentence a first-timer can act on,
 * with the mono truth beside it. Pure functions over the shared queue row so
 * the sidebar rail, the Queue view, and the status bar cannot disagree.
 */
import { activeWorkPhaseLabel } from "@studio/api/activity";
import { queueWaitLabel, resolveQueueWait, type QueueStatus } from "@studio/lib/queuePosition";
import { formatEta } from "./format";
import { isCancelledError, type Job } from "./generationJob";
import { modelDisplayNameForId, type DisplayableModel } from "./models";
import type { QueueRow } from "../composables/useQueueActivity";

/**
 * What the host says about a row beyond the job itself. Absent means its
 * queue was never read, never that there is nothing to say — which is why
 * every field degrades to the job's own evidence rather than to a default.
 */
export interface QueueRowContext {
  /** The row's live queue listing, when its host has been read. */
  wait?: QueueStatus | null;
  /** Seconds until the host's own predicted finish, when it predicts one. */
  etaSeconds?: number | null;
  /** Whether this row's host has stopped dispatching altogether. */
  queuePaused?: boolean;
}

/** The row's headline: the words the print was made from, else its style's
 * plain name (a catalog id resolves through the fleet's installed names). */
export function rowTitle(row: QueueRow, models: readonly DisplayableModel[] = []): string {
  const name = (id: string) => modelDisplayNameForId(id, models);
  switch (row.kind) {
    case "print":
      return row.print.prompt.trim() || name(row.print.model);
    case "shared":
      return row.shared.model ? name(row.shared.model) : row.shared.kind;
  }
}

/** One waiting row's sentence, from everything the host said about it. */
function waitStatus(job: Job, context?: QueueRowContext): string {
  const live = context?.wait ?? null;
  if (live?.blockedReason === "preparing") {
    const fraction = live.preparation?.fraction;
    return fraction == null
      ? "Getting a style ready"
      : `Getting a style ready · ${Math.round(fraction * 100)}%`;
  }
  const wait = resolveQueueWait({
    state: live?.state,
    position: live?.position ?? job.queuePosition,
    blockedReason: live?.blockedReason,
    preparation: live?.preparation,
    explicitlyPaused: live?.explicitlyPaused,
  });
  switch (wait.kind) {
    // One row someone paused and a whole queue parked by a restart both wear
    // `state: "paused"`, and saying "after restart" for the first made pausing
    // one job read as stopping everything.
    case "paused":
      return wait.explicit ? "Paused" : "Paused after restart";
    case "held":
      return "Held";
    case "next":
      return "Waiting — next up";
    default:
      return `Waiting — ${queueWaitLabel(wait).toLowerCase()}`;
  }
}

function printStatus(job: Job, context?: QueueRowContext): string {
  if (job.cancelling) return "Stopping…";
  switch (job.status) {
    case "denoising":
      // A paused queue holds dispatch; the print on the GPU keeps going, so
      // its own line never says "paused" — the waiting rows and the header do.
      return context?.etaSeconds == null
        ? `Adding detail — pass ${job.step} of ${job.total}`
        : `Adding detail — about ${formatEta(context.etaSeconds)} left`;
    case "finishing":
      return "Finishing up";
    case "loading":
      return job.stage ? `Getting ready — ${job.stage.toLowerCase()}` : "Getting ready";
    case "queued":
      return waitStatus(job, context);
    case "complete":
      return "Finished — saved to My images";
    case "error":
      if (job.outcomeUnknown) return "Outcome unknown — check My images";
      // A held print is parked, not failed: the host will take it again.
      if (job.retryable) {
        return job.holdCode === "MODEL_NOT_FOUND"
          ? "Needs a download first"
          : `Held — ${job.holdError ?? job.error ?? "waiting on the machine"}`;
      }
      return isCancelledError(job.error) ? "Stopped" : `Failed — ${job.error ?? "no reason given"}`;
  }
  return "";
}

/** One sentence of status, present tense. */
export function rowStatusLine(row: QueueRow, context?: QueueRowContext): string {
  switch (row.kind) {
    case "print":
      return printStatus(row.print, context);
    case "shared":
      return activeWorkPhaseLabel(row.shared);
  }
}

/**
 * How far a running row has got, 0 to 1, or `null` when nothing measures it.
 *
 * The meter and the caption must read the SAME counter. `rowStatusLine` gives
 * a shared row its sentence through `activeWorkPhaseLabel`, which reads the
 * host's `current`/`total`; a meter that understood only a print row's
 * denoise step had no answer for a shared row and drew a hard-coded stub
 * beside a caption that said "Generating PBR views · 11/15".
 *
 * A shared row's counter is the LOCAL one — steps inside the stage the host
 * has named, not a fraction of the whole render. That is the same figure the
 * caption states, so the meter agrees with the words beside it; neither
 * claims to know how much of the job is left, because the host never says.
 */
export function rowProgressFraction(row: QueueRow): number | null {
  switch (row.kind) {
    case "print":
      return row.print.status === "denoising" && row.print.total > 0
        ? clampFraction(row.print.step / row.print.total)
        : null;
    case "shared": {
      const { current, total } = row.shared;
      return current != null && total != null && total > 0 ? clampFraction(current / total) : null;
    }
  }
}

function clampFraction(value: number): number | null {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : null;
}

/** The same sentence in the rail's tighter idiom, where a dash reads as a gap. */
export function railStatusLine(row: QueueRow, context?: QueueRowContext): string {
  return rowStatusLine(row, context).replace(" — ", " · ");
}

/** "image 2 of 4" for a print made as one of a batch, else nothing. */
export function batchPositionLabel(row: QueueRow, jobs: readonly Job[]): string | null {
  if (row.kind !== "print") return null;
  const batch = jobs.filter((job) => job.batchId === row.print.batchId);
  if (batch.length < 2) return null;
  const at = batch.findIndex((job) => job.clientId === row.print.clientId);
  return at < 0 ? null : `image ${at + 1} of ${batch.length}`;
}

/**
 * Prints made since midnight. Counted from the gallery rather than from the
 * session so relaunching the app does not reset the day's tally.
 */
export function madeTodayCount(
  prints: readonly { item: { timestamp: number } }[],
  now: Date = new Date(),
): number {
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  const from = midnight.getTime() / 1000;
  return prints.filter((print) => print.item.timestamp >= from).length;
}

/** The mono glyph for a picture that does not exist yet: its place in line,
 * ⠂ while being made, ✓ when done, ↓ while a style downloads, ! on failure. */
export function rowGlyph(row: QueueRow): string {
  if (row.kind === "print") {
    const job = row.print;
    if (job.status === "complete") return "✓";
    if (job.status === "error" && job.retryable)
      return job.holdCode === "MODEL_NOT_FOUND" ? "↓" : "·";
    if (job.status === "error") return job.outcomeUnknown ? "?" : "!";
    if (job.status === "queued") {
      return job.queuePosition !== null && job.queuePosition >= 0
        ? String(job.queuePosition + 1)
        : "·";
    }
    if (job.status === "loading" && job.stage?.toLowerCase().includes("download")) return "↓";
    return "⠂";
  }
  return row.shared.kind === "download" ? "↓" : "⠂";
}

/** The state colour class for the row's status line and glyph. */
export function rowTone(row: QueueRow): string {
  if (row.kind === "print") {
    const job = row.print;
    if (job.status === "complete") return "text-state-done";
    if (job.status === "error" && job.retryable) return "text-state-blocked";
    if (job.status === "error")
      return job.outcomeUnknown ? "text-state-waiting" : "text-state-failed";
    if (job.status === "queued") return "text-state-waiting";
    return "text-state-active";
  }
  return row.shared.kind === "download" ? "text-state-blocked" : "text-state-active";
}

/** The status bar's queue clause: "1 image being made · 3 waiting". */
export function queueSentence(active: number, waiting: number, paused: boolean): string {
  if (paused) return `queue paused · ${waiting} waiting`;
  if (active === 0 && waiting === 0) return "nothing waiting";
  const making = active === 1 ? "1 image being made" : `${active} images being made`;
  if (active === 0) return `${waiting} waiting`;
  return waiting === 0 ? making : `${making} · ${waiting} waiting`;
}
