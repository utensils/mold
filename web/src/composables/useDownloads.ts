import { ref, type Ref } from "vue";
import {
  cancelDownload,
  downloadsStreamUrl,
  fetchDownloads,
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

    // Maintain rate sample window for the active job.
    if (evt.type === "progress" && active.value && active.value.id === evt.id) {
      const id = evt.id;
      const samples = ratesByJob.value[id] ?? [];
      const now = Date.now();
      samples.push({ ts: now, bytes: evt.bytes_done });
      // Drop samples older than 10 s.
      while (samples.length > 0 && now - samples[0].ts > 10_000)
        samples.shift();
      ratesByJob.value = { ...ratesByJob.value, [id]: samples };
    }

    if (evt.type === "job_done") {
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

  async function enqueue(model: string): Promise<void> {
    await postDownload(model);
  }

  async function cancel(id: string): Promise<void> {
    await cancelDownload(id);
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
    connected,
    close,
  };
}
