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

import { __resetQueueCommandState, useQueueCommands, type QueueCommands } from "./useQueueCommands";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
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
    __resetQueueCommandState();
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
    // The snapshot has already been read — the shell polls every host's queue
    // from launch — and it says this machine cannot pause. Space must cost
    // nothing at all: no request, and no swallowed key (see shortcuts.ts).
    jobs.queues["local"] = snapshot(false, false);
    const refresh = vi.spyOn(jobs, "refreshHost").mockResolvedValue(undefined as never);

    const api = commands();
    await api.togglePause();

    expect(refresh).not.toHaveBeenCalled();
    expect(pause).not.toHaveBeenCalled();
    expect(api.canPause.value).toBe(false);
  });
});

/**
 * The rail's active card is fleet-wide — it shows whichever machine started
 * most recently — while its pause button was bound to the DISPLAY host. On
 * two machines that meant the card showed B's render and paused A.
 */
describe("useQueueCommands — pausing the row's own machine", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    __resetQueueCommandState();
    readyLocalHost();
  });

  function twoHosts() {
    useHostsStore().extras.push({
      id: "plato",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: "plato-instance",
    });
    const jobs = useJobsStore();
    jobs.queues["local"] = { ...(snapshot(false) as object), hostId: "local" } as never;
    jobs.queues["plato"] = { ...(snapshot(false) as object), hostId: "plato" } as never;
    return jobs;
  }

  it("pauses the machine the row runs on, not the one pinned for display", async () => {
    const jobs = twoHosts();
    const pause = vi.spyOn(jobs, "pause").mockResolvedValue();
    vi.spyOn(jobs, "refreshHost").mockResolvedValue(undefined as never);

    const api = commands();
    expect(useHostsStore().all.map((h) => h.id)).toContain("plato");
    expect(api.canPauseFor("plato")).toBe(true);

    await api.togglePauseFor("plato");
    expect(pause).toHaveBeenCalledWith("plato");
    expect(pause).not.toHaveBeenCalledWith("local");
  });

  it("reads a row's own machine, whatever kind of row it is", () => {
    twoHosts();
    const api = commands();
    expect(api.hostIdFor({ kind: "print", print: { hostId: "plato" } } as never)).toBe("plato");
    expect(api.hostIdFor({ kind: "sequence", sequence: { hostId: "plato" } } as never)).toBe(
      "plato",
    );
    expect(api.hostIdFor({ kind: "shared", shared: { hostId: "plato" } } as never)).toBe("plato");
  });

  it("reports each machine's own paused state", () => {
    const jobs = twoHosts();
    jobs.queues["plato"] = { ...(snapshot(true) as object), hostId: "plato" } as never;
    const api = commands();
    expect(api.pausedFor("plato")).toBe(true);
    expect(api.pausedFor("local")).toBe(false);
    expect(api.pausedFor(null)).toBe(false);
  });
});

/**
 * Stop everything cancels every live print on every machine. It is the widest
 * destructive action in the app and it used to be one unconfirmed click, while
 * the narrower per-host Cancel all already armed itself.
 */
describe("useQueueCommands — Stop everything asks first", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    __resetQueueCommandState();
    readyLocalHost();
  });

  it("opens the confirm instead of acting, and acts only on confirm", async () => {
    const jobs = useJobsStore();
    jobs.queues["local"] = snapshot(false);
    const cancelAll = vi.spyOn(jobs, "cancelAll").mockResolvedValue(undefined as never);

    const api = commands();
    expect(api.stopEverythingOpen.value).toBe(false);

    api.askStopEverything();
    expect(api.stopEverythingOpen.value).toBe(true);
    expect(cancelAll).not.toHaveBeenCalled();

    await api.confirmStopEverything();
    expect(cancelAll).toHaveBeenCalledWith("local");
    expect(api.stopEverythingOpen.value).toBe(false);
    expect(api.stopEverythingBusy.value).toBe(false);
  });

  it("closes without acting when the dialog is dismissed", async () => {
    const jobs = useJobsStore();
    jobs.queues["local"] = snapshot(false);
    const cancelAll = vi.spyOn(jobs, "cancelAll").mockResolvedValue(undefined as never);

    const api = commands();
    api.askStopEverything();
    api.cancelStopEverything();
    expect(api.stopEverythingOpen.value).toBe(false);
    expect(cancelAll).not.toHaveBeenCalled();
  });

  it("shares one dialog across every surface that offers the action", () => {
    const rail = commands();
    const view = commands();
    rail.askStopEverything();
    expect(view.stopEverythingOpen.value).toBe(true);
    view.cancelStopEverything();
    expect(rail.stopEverythingOpen.value).toBe(false);
  });

  it("counts the pictures and the machines the confirm is about to stop", () => {
    const jobs = useJobsStore();
    jobs.queues["local"] = snapshot(false);
    const api = commands();
    expect(api.stopEverythingSummary.value).toBe(
      "Stops 0 pictures on 1 machine. Anything part-finished is lost.",
    );
  });
});

/**
 * `useQueueCommands` is instantiated per SURFACE and per ROW (QueueRowMenu is
 * rendered once per row), so an in-flight cancel guard held per instance let
 * the Queue view's Stop button stay armed while the same row's ⋯ ▸ Stop had a
 * request in the air.
 */
describe("useQueueCommands — the in-flight cancel guard is shared", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    __resetQueueCommandState();
    readyLocalHost();
  });

  it("disarms the same row in every instance while one cancel is in flight", () => {
    const rail = commands();
    const rowMenu = commands();
    const row = {
      kind: "shared",
      shared: { kind: "generation", key: "local:generation:1", can_cancel: true, stale: false },
    } as never;

    expect(rowMenu.canCancel(row)).toBe(true);
    rail.cancellingShared.value = ["local:generation:1"];
    expect(rowMenu.canCancel(row)).toBe(false);
  });
});

/**
 * The Queue view's explainer says "Drag a row to reorder". Nothing in that
 * view was draggable — reordering existed only behind the row's ⋯ menu — so
 * the sentence instructed an interaction the view did not implement.
 */
describe("useQueueCommands — dragging a waiting row", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    readyLocalHost();
  });

  function queued(hostId: string, ids: string[], canReorder = true) {
    useJobsStore().queues[hostId] = {
      hostId,
      entries: ids.map((id) => ({ id, state: "queued" })),
      paused: false,
      caps: { canPause: true, canCancelAll: true, canReorder },
      gpuOrdinals: [],
      error: null,
    } as never;
  }

  const printRow = (id: string, hostId = "local") =>
    ({
      key: `print:${id}`,
      createdAtMs: 1,
      kind: "print",
      print: { clientId: 1, id, hostId, status: "queued" },
    }) as never;

  it("moves the dragged row into the slot of the row it was dropped on", async () => {
    queued("local", ["a", "b", "c"]);
    const reorderQueued = vi.spyOn(useJobsStore(), "reorderQueued").mockResolvedValue(true);

    const api = commands();
    await api.dropOn(printRow("c"), printRow("a"));
    expect(reorderQueued).toHaveBeenCalledWith("local", "c", 0);
  });

  it("refuses a drop on a row from another machine, and on itself", async () => {
    queued("local", ["a", "b"]);
    queued("plato", ["x"]);
    const reorderQueued = vi.spyOn(useJobsStore(), "reorderQueued").mockResolvedValue(true);

    const api = commands();
    await api.dropOn(printRow("a"), printRow("x", "plato"));
    await api.dropOn(printRow("a"), printRow("a"));
    expect(reorderQueued).not.toHaveBeenCalled();
  });

  it("offers the drag only where the host says the queue can be reordered", () => {
    queued("local", ["a", "b"], false);
    const api = commands();
    expect(api.canReorder(printRow("a"))).toBe(false);

    queued("local", ["a", "b"], true);
    expect(api.canReorder(printRow("a"))).toBe(true);
    // A row with no server id (still connecting) is never draggable.
    expect(api.canReorder(printRow(""))).toBe(false);
  });
});
