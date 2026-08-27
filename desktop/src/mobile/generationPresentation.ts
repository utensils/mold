/**
 * iPhone adapter over the shared durable-child presentation. The phone's
 * `Job` is desktop's, so the mapping is desktop's too; what the phone adds is
 * its persisted pre-admission cancel tap, which is surface state the policy
 * never reads. The `complete` arm stays in `MobileApp.vue`, where the
 * byte-free presentation stub it needs lives.
 */

import {
  presentGenerationChild,
  type GenerationChildPresentation,
} from "@studio/lib/generationPresentation";
import { applyDurablePresentation } from "../lib/durableGenerationPresentation";
import type { Job } from "../lib/generationJob";
import type { MobileDurableGenerationRecovery } from "./mobileGenerationRecovery";

export function presentMobileDurableChild(
  recovery: MobileDurableGenerationRecovery,
  childIndex: number,
  hostLabel: string | null,
  now = Date.now(),
): GenerationChildPresentation {
  return presentGenerationChild({ tracker: recovery.tracker, childIndex, hostLabel, now });
}

export function applyMobileDurablePresentation(
  job: Job,
  p: GenerationChildPresentation,
  opts: { cancelRequested: boolean },
): void {
  applyDurablePresentation(job, p);
  if (p.kind === "waiting" || p.kind === "held" || p.kind === "running") {
    job.cancelling = opts.cancelRequested;
  }
}

export interface MobileDurableHold {
  error: string | null;
  code: string | null;
  retryable: boolean;
}

/** The hold a row is parked on, or `null` for every other arm. */
export function mobileDurableHeld(p: GenerationChildPresentation): MobileDurableHold | null {
  return p.kind === "held" ? { error: p.error, code: p.code, retryable: p.retryable } : null;
}
