import type { ApiTarget } from "../api/client";
import type { RetainedSourceMediaInventory } from "../api/gallerySourceMedia";

const aborted = () =>
  new DOMException("Source no longer visible", "AbortError");
type Pending = {
  controller: AbortController;
  refs: number;
  started: boolean;
  load: (signal: AbortSignal) => Promise<Blob>;
  resolve: (blob: Blob) => void;
  reject: (error: unknown) => void;
  promise: Promise<Blob>;
};
/** Private, memory-only resources. Keys contain the complete owner authority. */
export class SourcePreviewCache {
  private ready = new Map<string, { blob: Blob; at: number }>();
  private pending = new Map<string, Pending>();
  private running = 0;
  bytes = 0;
  constructor(
    private budget = 16 * 1024 * 1024,
    private concurrency = 4,
  ) {}
  acquire(key: string, load: (signal: AbortSignal) => Promise<Blob>) {
    const cached = this.ready.get(key);
    if (cached && Date.now() - cached.at < 60_000) {
      this.ready.delete(key);
      this.ready.set(key, cached);
      return { promise: Promise.resolve(cached.blob), release() {} };
    }
    if (cached) this.remove(key);
    let entry = this.pending.get(key);
    if (!entry) {
      let resolve!: Pending["resolve"], reject!: Pending["reject"];
      const promise = new Promise<Blob>((yes, no) => {
        resolve = yes;
        reject = no;
      });
      entry = {
        controller: new AbortController(),
        refs: 0,
        started: false,
        load,
        resolve,
        reject,
        promise,
      };
      this.pending.set(key, entry);
    }
    entry.refs++;
    this.pump();
    const held = entry;
    let released = false;
    return {
      promise: held.promise,
      release: () => {
        if (released) return;
        released = true;
        if (--held.refs === 0 && this.pending.get(key) === held) {
          this.pending.delete(key);
          held.controller.abort();
          held.reject(aborted());
          this.pump();
        }
      },
    };
  }
  private remove(key: string) {
    const old = this.ready.get(key);
    if (old) this.bytes -= old.blob.size;
    this.ready.delete(key);
  }
  invalidate(prefix: string) {
    for (const key of this.ready.keys())
      if (key.startsWith(prefix)) this.remove(key);
    for (const [key, entry] of this.pending)
      if (key.startsWith(prefix)) {
        this.pending.delete(key);
        entry.controller.abort();
        entry.reject(aborted());
      }
    this.pump();
  }
  private pump() {
    for (const [key, entry] of this.pending) {
      if (this.running >= this.concurrency) break;
      if (entry.started) continue;
      entry.started = true;
      this.running++;
      void Promise.resolve()
        .then(() => {
          entry.controller.signal.throwIfAborted();
          return entry.load(entry.controller.signal);
        })
        .then(
          (blob) => {
            if (
              entry.controller.signal.aborted ||
              this.pending.get(key) !== entry
            )
              return;
            if (blob.size <= this.budget) {
              while (
                (this.bytes + blob.size > this.budget ||
                  this.ready.size >= 512) &&
                this.ready.size
              )
                this.remove(this.ready.keys().next().value!);
              this.ready.set(key, { blob, at: Date.now() });
              this.bytes += blob.size;
            }
            entry.resolve(blob);
          },
          (error) => entry.reject(error),
        )
        .finally(() => {
          if (this.pending.get(key) === entry) this.pending.delete(key);
          this.running--;
          this.pump();
        });
    }
  }
}
export const sourcePreviews = new SourcePreviewCache();
const inventoryCache = new Map<
  string,
  { value: RetainedSourceMediaInventory; at: number }
>();
const invalidators = new Map<string, Set<(refresh: boolean) => void>>();
export function observeSourceInvalidation(
  key: string,
  listener: (refresh: boolean) => void,
) {
  let listeners = invalidators.get(key);
  if (!listeners) invalidators.set(key, (listeners = new Set()));
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (!listeners.size) invalidators.delete(key);
  };
}
export function invalidateSourceOwner(key: string, refresh = false) {
  inventoryCache.delete(key);
  sourcePreviews.invalidate(`${key}\n`);
  for (const listener of invalidators.get(key) ?? []) listener(refresh);
}
export function cachedSourceInventory(key: string) {
  const cached = inventoryCache.get(key);
  if (!cached || Date.now() - cached.at >= 60_000) {
    inventoryCache.delete(key);
    return;
  }
  return cached.value;
}
export function cacheSourceInventory(
  key: string,
  value: RetainedSourceMediaInventory,
) {
  if (
    value.availability !== "available" &&
    value.availability !== "unavailable_legacy"
  )
    return;
  inventoryCache.delete(key);
  inventoryCache.set(key, { value, at: Date.now() });
  // Inventory size is bounded separately from original/media bytes.
  let members = 0;
  for (const entry of inventoryCache.values())
    members += entry.value.members.length;
  while (inventoryCache.size > 512 || members > 10_000) {
    const oldest = inventoryCache.keys().next().value!;
    members -= inventoryCache.get(oldest)!.value.members.length;
    inventoryCache.delete(oldest);
  }
}
export function clearSourceResources() {
  for (const key of new Set([...inventoryCache.keys(), ...invalidators.keys()]))
    invalidateSourceOwner(key);
  sourcePreviews.invalidate("");
}

// Exact-route event hints never merge authorities or invalidate ordinary polling.
type SourceEvent = { target: ApiTarget; filename: string };
const sourceEvents = new Set<(event: SourceEvent) => void>();
export function observeSourceMediaEvents(
  listener: (event: SourceEvent) => void,
) {
  sourceEvents.add(listener);
  return () => {
    sourceEvents.delete(listener);
  };
}
export function sourceMediaGalleryEvent(
  target: ApiTarget,
  event: { type?: unknown; filename?: unknown },
) {
  if (
    ![
      "gallery_added",
      "gallery_updated",
      "gallery_removed",
      "gallery_trashed",
      "gallery_restored",
    ].includes(String(event.type)) ||
    typeof event.filename !== "string"
  )
    return;
  for (const key of new Set([
    ...inventoryCache.keys(),
    ...invalidators.keys(),
  ])) {
    let identity: unknown[];
    try {
      identity = JSON.parse(key) as unknown[];
    } catch {
      continue;
    }
    if (!Array.isArray(identity)) continue;
    if (
      identity[1] === event.filename &&
      identity[2] === target.baseUrl &&
      (identity[3] ?? null) === (target.apiKey ?? null)
    )
      invalidateSourceOwner(key);
  }
  for (const listener of sourceEvents)
    listener({ target, filename: event.filename });
}
export function sourceEventMatches(
  owner: { filename: string; target: ApiTarget },
  event: SourceEvent,
) {
  return (
    owner.filename === event.filename &&
    owner.target.baseUrl === event.target.baseUrl &&
    (owner.target.apiKey ?? null) === (event.target.apiKey ?? null)
  );
}
