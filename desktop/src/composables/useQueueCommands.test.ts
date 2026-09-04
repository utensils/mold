/**
 * Space is the shell's one un-modified chord, so pause and resume have to be
 * the same key twice. `paused` is read off the display host's queue snapshot,
 * and `jobs.pause` only writes it back onto a snapshot that already exists —
 * so on a launch that had never opened Machines, the first Space paused the
 * queue for real, `paused` stayed false, the rail's button stayed hidden, and
 * a second Space paused again. The queue was stopped with no way back.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve({})),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));

import { useQueueCommands, type QueueCommands } from "./useQueueCommands";
import { useConnectionStore } from "../stores/connection";
import { useJobsStore } from "../stores/jobs";

function commands(): QueueCommands {
  let api!: QueueCommands;
  mount(
    defineComponent({
      setup() {
        api = useQueueCommands();
        return () => null;
      },
    }),
  );
  return api;
}

function readyLocalHost() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
}

/** What `refreshHost` would have written, had anything ever asked for it. */
function snapshot(paused: boolean, canPause = true) {
  return {
    hostId: "local",
    entries: [],
    paused,
    caps: { canPause, canCancelAll: true, canReorder: false },
    gpuOrdinals: [],
    error: null,
  } as never;
}

describe("useQueueCommands — Space on the display host's queue", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    readyLocalHost();
  });

  it("reads the queue before deciding, so the second Space resumes", async () => {
    const jobs = useJobsStore();
    const pause = vi.spyOn(jobs, "pause").mockImplementation(async () => {
      jobs.queues["local"] = snapshot(true);
    });
    const resume = vi.spyOn(jobs, "resume").mockResolvedValue();
    // Nothing has read this host's queue yet — the launch never opened Machines.
    const refresh = vi.spyOn(jobs, "refreshHost").mockImplementation(async () => {
      jobs.queues["local"] ??= snapshot(false);
    });

    const api = commands();
    await api.togglePause();
    expect(refresh).toHaveBeenCalled();
    expect(pause).toHaveBeenCalledWith("local");

    await api.togglePause();
    expect(resume).toHaveBeenCalledWith("local");
    expect(pause).toHaveBeenCalledTimes(1);
  });

  it("does nothing on a host that does not advertise queue pause", async () => {
    const jobs = useJobsStore();
    const pause = vi.spyOn(jobs, "pause").mockResolvedValue();
    vi.spyOn(jobs, "refreshHost").mockImplementation(async () => {
      jobs.queues["local"] = snapshot(false, false);
    });

    const api = commands();
    await api.togglePause();

    expect(pause).not.toHaveBeenCalled();
    expect(api.canPause.value).toBe(false);
  });
});
