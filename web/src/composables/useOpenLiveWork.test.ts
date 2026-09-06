import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import type { HostRouting } from "./useHostRouting";

const push = vi.fn();
vi.mock("vue-router", () => ({ useRouter: () => ({ push }) }));

const findQueueEntryById = vi.fn();
vi.mock("@studio/api/queuePlan", () => ({
  findQueueEntryById: (...args: unknown[]) => findQueueEntryById(...args),
}));

const toast = vi.fn();
vi.mock("../lib/toasts", () => ({
  toast: (...args: unknown[]) => toast(...args),
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
    findQueueEntryById.mockReset();
    toast.mockReset();
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
