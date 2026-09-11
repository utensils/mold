import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import type { HostRouting } from "./useHostRouting";

const push = vi.fn(async () => undefined);
vi.mock("vue-router", () => ({ useRouter: () => ({ push }) }));

const findQueueEntryById = vi.fn();
vi.mock("@studio/api/queuePlan", () => ({
  findQueueEntryById: (...args: unknown[]) => findQueueEntryById(...args),
}));

const toast = vi.fn();
vi.mock("../lib/toasts", () => ({
  toast: (...args: unknown[]) => toast(...args),
}));

const setGenerationHandoff = vi.fn();
vi.mock("./useGenerationHandoff", () => ({
  setGenerationHandoff: (...args: unknown[]) => setGenerationHandoff(...args),
}));

import { useOpenLiveWork } from "./useOpenLiveWork";

function routing(): HostRouting {
  return {
    hosts: ref([{ id: "plato", url: "http://plato:7680", apiKey: null }]),
  } as unknown as HostRouting;
}

function row(over: Partial<FleetActiveWork> = {}): FleetActiveWork {
  return {
    id: "chain-1",
    kind: "generation",
    phase: "running",
    created_at_unix_ms: 0,
    updated_at_unix_ms: 0,
    can_cancel: true,
    key: "plato:generation:chain-1",
    hostId: "plato",
    hostLabel: "plato",
    routeUrl: "http://plato:7680",
    instanceId: "i-1",
    stale: false,
    hostError: null,
    ...over,
  } as FleetActiveWork;
}

describe("useOpenLiveWork", () => {
  beforeEach(() => {
    push.mockReset();
    push.mockImplementation(async () => undefined);
    findQueueEntryById.mockReset();
    toast.mockReset();
    setGenerationHandoff.mockReset();
  });

  // The four things only the browser decides — its machine list, its
  // pinned-seed handoff, its downloads popover, its own toast — are what
  // the shared driver cannot test for it.
  it("looks the machine up by id and hands Create a pinned-seed handoff", async () => {
    findQueueEntryById.mockResolvedValue({
      id: "job-1",
      state: "running",
      metadata: { prompt: "a cat", model: "ltx2" },
    });
    await useOpenLiveWork(routing())(row({ id: "job-1", execution: null }));
    expect(findQueueEntryById).toHaveBeenCalledWith(
      { baseUrl: "http://plato:7680", apiKey: null },
      "job-1",
    );
    expect(setGenerationHandoff).toHaveBeenCalledWith(
      expect.objectContaining({
        seedPinned: true,
        queueSelection: expect.objectContaining({
          hostId: "plato",
          jobId: "job-1",
        }),
      }),
    );
  });

  it("opens downloads as the shell's popover, not a page", async () => {
    const opened = vi.fn();
    window.addEventListener("mold:open-downloads", opened, { once: true });
    await useOpenLiveWork(routing())(row({ kind: "download", id: "dl-1" }));
    expect(opened).toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
  });

  it("says so in a toast when the machine is unknown", async () => {
    await useOpenLiveWork(routing())(row({ hostId: "ghost", execution: null }));
    expect(toast).toHaveBeenCalledWith("error", expect.any(String));
  });

  // The auto-chain regression: a long video the host split and stitched is
  // `kind: "generation"` with `execution: "chain"`. Its id lives in the
  // chain-job space, so the ordinary generation reattach searches /api/queue
  // for an id that is not there and dead-ends on "cannot restore settings".
  // It must be recognised BEFORE the generation arm.
  it("sends a chain row to its machine instead of failing a queue lookup", async () => {
    await useOpenLiveWork(routing())(row({ execution: "chain" }));
    expect(findQueueEntryById).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
    expect(push).toHaveBeenCalledWith("/machines/plato");
  });

  // An older host still labels a chain row with its own kind.
  it("sends a legacy sequence row to its machine", async () => {
    await useOpenLiveWork(routing())(row({ kind: "sequence" }));
    expect(findQueueEntryById).not.toHaveBeenCalled();
    expect(push).toHaveBeenCalledWith("/machines/plato");
  });

  it("still reattaches an ordinary generation through the queue", async () => {
    findQueueEntryById.mockResolvedValue({
      id: "job-1",
      state: "running",
      metadata: { prompt: "a cat", model: "ltx2" },
    });
    await useOpenLiveWork(routing())(row({ id: "job-1", execution: null }));
    expect(findQueueEntryById).toHaveBeenCalled();
    expect(push).toHaveBeenCalledWith("/create");
  });
});
