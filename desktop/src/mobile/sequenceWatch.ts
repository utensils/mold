/**
 * iPhone durable-sequence watcher.
 *
 * The old mobile Create polled `/api/chain-jobs/:id` every 2 seconds, which
 * cost a request per clip-second and still lagged the seam-by-seam progress
 * the queue now renders. This module makes SSE the primary transport — the
 * same `/api/chain-jobs/:id/events` stream desktop and web reduce through
 * `applyChainJobEvent` — with two iOS-specific safety nets:
 *
 *  - **Poll fallback.** A persistent stream error (captive portal, proxy that
 *    buffers `text/event-stream`, server restart) drops to a 5s snapshot poll
 *    instead of leaving a live job frozen. A stream that later opens cleanly
 *    retires the poll.
 *  - **Wake re-fetch.** iOS suspends sockets when the app backgrounds and does
 *    NOT always error the fetch; on `visibilitychange → visible` we force a
 *    snapshot so the queue is truthful the instant the user looks at it.
 *
 * Every request rides the FROZEN `ApiTarget` handed in at watch time — the
 * caller snapshots the route (host id, base URL, Keychain key, instance id)
 * once and this module never re-resolves it.
 */
import {
  applyChainJobEvent,
  emptyChainJobLive,
  type ChainJobLive,
} from "@studio/lib/chainJobEvents";
import type { ChainJobDetail, ChainJobEvent } from "@studio/lib/api/chainTypes";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";

const DEFAULT_POLL_INTERVAL_MS = 5_000;

/** The slice of `document` this watcher needs; injectable for tests. */
export interface SequenceWatchVisibility {
  readonly hidden: boolean;
  addEventListener(type: "visibilitychange", listener: () => void): void;
  removeEventListener(type: "visibilitychange", listener: () => void): void;
}

export interface SequenceWatchOptions {
  /** Frozen route target — never re-resolved for the life of the watch. */
  target: ApiTarget;
  jobId: string;
  onUpdate: (live: ChainJobLive) => void;
  onError: (error: unknown) => void;
  pollIntervalMs?: number;
  stream?: typeof sseStream;
  fetchDetail?: (target: ApiTarget, jobId: string) => Promise<ChainJobDetail>;
  visibility?: SequenceWatchVisibility | null;
}

export interface SequenceWatchHandle {
  /** Latest reduced view; the caller mirrors it into reactive state. */
  readonly live: ChainJobLive;
  /** True while the snapshot poll is standing in for a broken stream. */
  readonly polling: boolean;
  /** Force a snapshot fetch (wake, post-action confirmation). */
  refresh(): Promise<void>;
  stop(): void;
}

function defaultFetchDetail(target: ApiTarget, jobId: string): Promise<ChainJobDetail> {
  return apiJsonTo<ChainJobDetail>(target, `/api/chain-jobs/${encodeURIComponent(jobId)}`);
}

function defaultVisibility(): SequenceWatchVisibility | null {
  return typeof document === "undefined" ? null : (document as SequenceWatchVisibility);
}

export function watchChainJob(options: SequenceWatchOptions): SequenceWatchHandle {
  const stream = options.stream ?? sseStream;
  const fetchDetail = options.fetchDetail ?? defaultFetchDetail;
  const visibility = options.visibility === undefined ? defaultVisibility() : options.visibility;
  const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const eventsPath = `/api/chain-jobs/${encodeURIComponent(options.jobId)}/events`;

  let live = emptyChainJobLive();
  let stopped = false;
  let abort: AbortController | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let streamOpen = false;

  function push(next: ChainJobLive): void {
    live = next;
    options.onUpdate(live);
  }

  async function refresh(): Promise<void> {
    if (stopped) return;
    try {
      const job = await fetchDetail(options.target, options.jobId);
      if (stopped) return;
      push(applyChainJobEvent(live, { type: "snapshot", job }));
    } catch (error) {
      if (!stopped) options.onError(error);
    }
  }

  function startPoll(): void {
    if (stopped || pollTimer) return;
    pollTimer = setInterval(() => {
      void refresh();
      // A healed network should win back the cheap transport, so every tick
      // also re-attempts the stream; `onOpen` then retires this timer.
      if (!streamOpen) openStream();
    }, pollIntervalMs);
  }

  function stopPoll(): void {
    if (!pollTimer) return;
    clearInterval(pollTimer);
    pollTimer = null;
  }

  function openStream(): void {
    if (stopped) return;
    abort?.abort();
    streamOpen = false;
    const controller = new AbortController();
    abort = controller;
    void stream(eventsPath, {
      signal: controller.signal,
      // Retry is OFF on purpose: we want the FIRST unrecoverable error to
      // reach onClose so the poll can take over, rather than letting
      // fetchEventSource back off silently behind a frozen row.
      retry: false,
      target: options.target,
      onOpen: () => {
        // A healthy stream is authoritative again — retire the fallback.
        if (stopped || controller.signal.aborted) return;
        streamOpen = true;
        stopPoll();
      },
      onEvent: (_event, data) => {
        if (stopped || controller.signal.aborted) return;
        try {
          push(applyChainJobEvent(live, JSON.parse(data) as ChainJobEvent));
        } catch {
          // A malformed frame is skipped; the next snapshot re-syncs.
        }
      },
      onClose: (error) => {
        if (stopped || controller.signal.aborted) return;
        streamOpen = false;
        if (error) options.onError(error);
        // Whether the stream errored or simply ended early, the job may still
        // be running — re-sync now and let the poll keep the row honest until
        // a later stream attempt opens.
        void refresh();
        startPoll();
      },
    });
  }

  function onVisibilityChange(): void {
    if (stopped || visibility?.hidden) return;
    // iOS can suspend the socket without erroring it; re-open and re-sync.
    void refresh();
    if (pollTimer) openStream();
  }

  visibility?.addEventListener("visibilitychange", onVisibilityChange);
  void refresh();
  openStream();

  return {
    get live() {
      return live;
    },
    get polling() {
      return pollTimer !== null;
    },
    refresh,
    stop() {
      if (stopped) return;
      stopped = true;
      streamOpen = false;
      stopPoll();
      abort?.abort();
      abort = null;
      visibility?.removeEventListener("visibilitychange", onVisibilityChange);
    },
  };
}
