import { shallowRef } from "vue";
import type { ApiTarget } from "@studio/api/client";
import type {
  RetainedSourceMediaInventory,
  RetainedSourceMediaMetadataLike,
} from "@studio/api/gallerySourceMedia";

export interface RetainedSourceReuseIntent {
  filename: string;
  origin: ApiTarget;
  inventory: RetainedSourceMediaInventory;
  metadata?: RetainedSourceMediaMetadataLike;
}

export const retainedSourceReuseState =
  shallowRef<RetainedSourceReuseIntent | null>(null);
let version = 0;

export function beginRetainedSourceReuseIntent(): number {
  version += 1;
  retainedSourceReuseState.value = null;
  return version;
}

export function setRetainedSourceReuseIntent(
  intent: RetainedSourceReuseIntent | null,
): void {
  version += 1;
  retainedSourceReuseState.value = intent;
}

export function setRetainedSourceReuseIntentIfCurrent(
  expectedVersion: number,
  intent: RetainedSourceReuseIntent,
): boolean {
  if (expectedVersion !== version) return false;
  retainedSourceReuseState.value = intent;
  return true;
}

export function retainedSourceReuseSnapshot(): {
  version: number;
  intent: RetainedSourceReuseIntent;
} | null {
  return retainedSourceReuseState.value
    ? { version, intent: retainedSourceReuseState.value }
    : null;
}

export function retainedSourceReuseIsCurrent(expectedVersion: number): boolean {
  return expectedVersion === version;
}

export function retainedSourceReuseIntent(): RetainedSourceReuseIntent | null {
  return retainedSourceReuseState.value;
}

export function clearRetainedSourceReuseIntent(): void {
  version += 1;
  retainedSourceReuseState.value = null;
}
