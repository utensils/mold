import { ref, type Ref } from "vue";
import {
  cancelDownload,
  downloadsStreamUrl,
  fetchDownloads,
  looksLikeCatalogId,
  postCatalogDownload,
  postDownload,
} from "../api";
import type {
  DownloadEventWire,
  DownloadJobWire,
  DownloadsListingWire,
} from "../types";

export interface DownloadsState {
  active: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
}

export function newDownloadsState(): DownloadsState {
  return { active: null, queued: [], history: [] };
}

const HISTORY_CAP = 20;

/**
 * Pure reducer — applied to a plain `DownloadsState` so it can be unit-tested
 * without a Vue runtime.
 */
export function applyDownloadEvent(
  state: DownloadsState,
  event: DownloadEventWire,
): void {
  switch (event.type) {
    case "snapshot": {
      // Full-state replace. Server emits this as the first frame to every
      // subscriber so navigation / reconnect doesn't need a separate
      // `fetchDownloads()` call to rehydrate. Any per-job event the
      // server emits afterward is a delta on top of this snapshot.
      //
      // `active` may arrive as `undefined` when the server has nothing
      // running — `DownloadsListing.active` is `Option<DownloadJob>` and
      // serde's `skip_serializing_if = Option::is_none` strips the field.
      // Coalesce so downstream consumers don't have to handle both null
      // and undefined.
      state.active = event.listing.active ?? null;
      state.queued = [...event.listing.queued];
      state.history = [...event.listing.history];
      return;
    }
    case "enqueued": {
      state.queued.push({
        id: event.id,
        model: event.model,
        status: "queued",
        files_done: 0,
        files_total: 0,
        bytes_done: 0,
        bytes_total: 0,
        current_file: null,
        started_at: null,
        completed_at: null,
        error: null,
      });
      return;
    }
    case "dequeued": {
      const idx = state.queued.findIndex((j) => j.id === event.id);
      if (idx >= 0) state.queued.splice(idx, 1);
      return;
    }
    case "started": {
      const idx = state.queued.findIndex((j) => j.id === event.id);
      const from =
        idx >= 0
          ? state.queued.splice(idx, 1)[0]
          : {
              id: event.id,
              model: "",
              status: "queued" as const,
              files_done: 0,
              files_total: 0,
              bytes_done: 0,
              bytes_total: 0,
              current_file: null,
              started_at: null,
              completed_at: null,
              error: null,
            };
      state.active = {
        ...from,
        status: "active",
        files_total: event.files_total,
        bytes_total: event.bytes_total,
        started_at: Date.now(),
      };
      return;
    }
    case "progress": {
      if (state.active?.id !== event.id) return;
      state.active.files_done = event.files_done;
      state.active.bytes_done = event.bytes_done;
      state.active.current_file = event.current_file ?? null;
      return;
    }
    case "file_done": {
      if (state.active?.id !== event.id) return;
      state.active.files_done += 1;
      return;
    }
    case "job_done": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const completed: DownloadJobWire = {
        ...active,
        status: "completed",
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(completed);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
    case "job_failed": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const failed: DownloadJobWire = {
        ...active,
        status: "failed",
        error: event.error,
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(failed);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
    case "job_cancelled": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const cancelled: DownloadJobWire = {
        ...active,
        status: "cancelled",
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(cancelled);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
    case "catalog_ready": {
      // Pure notification — no per-job state to mutate. The reducer
      // ignores it so the active/queued/history shape stays consistent
      // with what the server's `GET /api/downloads` would return for the
      // same instant. Side-effect (refresh model list) lives in the
      // event-source consumer above.
      return;
    }
  }
}

/**
 * Client-side ETA math — server only emits raw counters.
 * history = sliding window of {ts, bytes} samples (last ~10 s).
 */
export function computeEtaSeconds(
  history: Array<{ ts: number; bytes: number }>,
  bytesTotal: number,
): number | null {
  if (history.length < 2) return null;
  const first = history[0];
  const last = history[history.length - 1];
  const deltaBytes = last.bytes - first.bytes;
  const deltaMs = last.ts - first.ts;
  if (deltaMs <= 0 || deltaBytes <= 0) return null;
  const ratePerSec = (deltaBytes * 1000) / deltaMs;
  const remaining = Math.max(0, bytesTotal - last.bytes);
  const eta = remaining / ratePerSec;
  return Number.isFinite(eta) ? Math.round(eta) : null;
}

// ── Vue runtime singleton ────────────────────────────────────────────────────

export interface UseDownloads {
  active: Ref<DownloadJobWire | null>;
  queued: Ref<DownloadJobWire[]>;
  history: Ref<DownloadJobWire[]>;
  ratesByJob: Ref<Record<string, Array<{ ts: number; bytes: number }>>>;
  enqueue: (model: string) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  /// Force-fetch the current downloads listing. Use after any caller
  /// triggers an action that's expected to alter the queue (e.g. catalog
  /// downloads) but the relevant SSE event might be lagged or missed.
  refresh: () => Promise<void>;
  connected: Ref<boolean>;
  close: () => void;
}

type Listener = () => void;
const completionListeners = new Set<Listener>();

export function onDownloadComplete(cb: Listener): () => void {
  completionListeners.add(cb);
  return () => {
    completionListeners.delete(cb);
  };
}

let singleton: UseDownloads | null = null;

export function useDownloads(): UseDownloads {
  if (singleton) return singleton;
  singleton = buildSingleton();
  return singleton;
}

/// Test-only escape hatch — resets the module-level singleton so each test
/// gets a fresh `EventSource` / state. Not exported through the public
/// surface; consumers must import the underscore-prefixed name explicitly.
export function __resetUseDownloadsForTest(): void {
  singleton?.close();
  singleton = null;
}

function buildSingleton(): UseDownloads {
  const active = ref<DownloadJobWire | null>(null);
  const queued = ref<DownloadJobWire[]>([]);
  const history = ref<DownloadJobWire[]>([]);
  const ratesByJob = ref<Record<string, Array<{ ts: number; bytes: number }>>>(
    {},
  );
  const connected = ref(false);
  let es: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  function state(): DownloadsState {
    return {
      active: active.value,
      queued: queued.value,
      history: history.value,
    };
  }

  function writeBack(next: DownloadsState) {
    active.value = next.active;
    queued.value = [...next.queued];
    history.value = [...next.history];
  }

  function applyListing(listing: DownloadsListingWire) {
    active.value = listing.active ?? null;
    queued.value = [...listing.queued];
    history.value = [...listing.history];
  }

  function onEvent(raw: string) {
    let evt: DownloadEventWire;
    try {
      evt = JSON.parse(raw) as DownloadEventWire;
    } catch {
      return;
    }
    const snap = state();
    applyDownloadEvent(snap, evt);
    writeBack(snap);

    // Maintain rate sample window for the active job. Mutate the array
    // in place rather than spreading the whole `ratesByJob` object —
    // pre-fix, every progress tick allocated a fresh object containing
    // every job ever rate-tracked, scaling O(N) with cumulative session
    // length and dragging the main thread after long sessions.
    if (evt.type === "progress" && active.value && active.value.id === evt.id) {
      const id = evt.id;
      const samples = ratesByJob.value[id] ?? [];
      const now = Date.now();
      samples.push({ ts: now, bytes: evt.bytes_done });
      // Drop samples older than 10 s.
      while (samples.length > 0 && now - samples[0].ts > 10_000)
        samples.shift();
      // Direct property assignment — the parent ref's reactivity is
      // triggered by reading `ratesByJob.value[id]` consumers anyway,
      // and we want to avoid O(N) copies. If a consumer needs full-
      // object reactivity, the rate-tracker gets exposed via a
      // dedicated computed.
      ratesByJob.value[id] = samples;
    }

    // Drop the rate window for jobs that just terminated. Without this
    // `ratesByJob` was a slow memory leak — every download ever started
    // kept its rate buckets indefinitely, and the in-place mutation we
    // do above doesn't help unbounded key growth.
    if (
      evt.type === "job_done" ||
      evt.type === "job_failed" ||
      evt.type === "job_cancelled"
    ) {
      delete ratesByJob.value[evt.id];
    }
    // Snapshot replaces the entire view of the queue — any job not in
    // the new listing's active/queued is gone, so its rate window is
    // dead weight too. Rebuild against the surviving id set.
    if (evt.type === "snapshot") {
      const live = new Set<string>();
      if (evt.listing.active) live.add(evt.listing.active.id);
      for (const j of evt.listing.queued) live.add(j.id);
      for (const id of Object.keys(ratesByJob.value)) {
        if (!live.has(id)) delete ratesByJob.value[id];
      }
    }

    if (evt.type === "job_done" || evt.type === "catalog_ready") {
      // `job_done` covers single-pull manifest fetches. `catalog_ready`
      // covers the multi-job catalog case where the model only becomes
      // usable once the primary AND every companion are on disk —
      // refreshing earlier just shows an incomplete model.
      for (const cb of completionListeners) cb();
    }
  }

  function connect() {
    if (closed) return;
    try {
      es = new EventSource(downloadsStreamUrl());
    } catch {
      scheduleReconnect();
      return;
    }
    es.onopen = () => {
      connected.value = true;
    };
    es.onmessage = (ev) => onEvent(ev.data);
    // The server emits named events ("download"); fall back to default too.
    es.addEventListener("download", (ev) =>
      onEvent((ev as MessageEvent).data as string),
    );
    es.onerror = () => {
      connected.value = false;
      es?.close();
      es = null;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (closed) return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => {
      void fetchDownloads()
        .then(applyListing)
        .catch(() => undefined);
      connect();
    }, 2000);
  }

  // Boot: initial snapshot then subscribe.
  void fetchDownloads()
    .then(applyListing)
    .catch(() => undefined);
  connect();

  /// Force-fetch the current `/api/downloads` listing and apply it. The
  /// SSE stream also keeps state fresh, but `refresh()` is the click-time
  /// guarantee — user actions must produce a visible result without
  /// depending on SSE event delivery (which can lag, especially right
  /// after a reconnect or when the page is in a background tab).
  async function refresh(): Promise<void> {
    try {
      const listing = await fetchDownloads();
      applyListing(listing);
    } catch {
      /* network blip — SSE will catch us up if it stays connected */
    }
  }

  async function enqueue(model: string): Promise<void> {
    // Catalog rows (`cv:` / `hf:`) carry their canonical id in `job.model`,
    // but `/api/downloads` only validates against the manifest registry —
    // so retrying a failed catalog download by re-POSTing that id 400s.
    // Route catalog-shaped ids through `/api/catalog/:id/download`, which
    // owns the recipe-payload + companion-pull flow.
    if (looksLikeCatalogId(model)) {
      await postCatalogDownload(model);
    } else {
      await postDownload(model);
    }
    // Belt-and-suspenders against SSE lag. `applyListing` is idempotent
    // with the SSE-driven mutations so the worst case is a no-op write.
    void refresh();
  }

  async function cancel(id: string): Promise<void> {
    await cancelDownload(id);
    void refresh();
  }

  function close() {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    es?.close();
    es = null;
  }

  return {
    active,
    queued,
    history,
    ratesByJob,
    enqueue,
    cancel,
    refresh,
    connected,
    close,
  };
}
