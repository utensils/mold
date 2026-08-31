import type { ApiTarget } from "@studio/api/client";
import type { RetainedSourceMediaInventory } from "@studio/api/gallerySourceMedia";

export interface RetainedSourceReuseIntent {
  filename: string;
  origin: ApiTarget;
  inventory: RetainedSourceMediaInventory;
}

let current: RetainedSourceReuseIntent | null = null;
let version = 0;

export function beginRetainedSourceReuseIntent(): number {
  version += 1;
  current = null;
  return version;
}

export function setRetainedSourceReuseIntent(
  intent: RetainedSourceReuseIntent | null,
): void {
  version += 1;
  current = intent;
}

export function setRetainedSourceReuseIntentIfCurrent(
  expectedVersion: number,
  intent: RetainedSourceReuseIntent,
): boolean {
  if (expectedVersion !== version) return false;
  current = intent;
  return true;
}

export function retainedSourceReuseSnapshot(): {
  version: number;
  intent: RetainedSourceReuseIntent;
} | null {
  return current ? { version, intent: current } : null;
}

export function retainedSourceReuseIsCurrent(expectedVersion: number): boolean {
  return expectedVersion === version;
}

export function retainedSourceReuseIntent(): RetainedSourceReuseIntent | null {
  return current;
}

export function clearRetainedSourceReuseIntent(): void {
  version += 1;
  current = null;
}
