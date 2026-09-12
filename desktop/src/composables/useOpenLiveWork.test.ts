import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetActiveWork } from "@studio/api/activity";

/*
 * The shared driver (`studio/composables/useOpenLiveWork`) decides; this pins
 * the four things only the APP supplies — its machine list, its composer
 * handoff, its toast, and where a chain row and a download row go.
 */

const push = vi.fn(async () => undefined);
vi.mock("vue-router", () => ({ useRouter: () => ({ push }) }));

const findQueueEntryById = vi.fn();
vi.mock("@studio/api/queuePlan", () => ({
  findQueueEntryById: (...args: unknown[]) => findQueueEntryById(...args),
}));

const composerSet = vi.fn();
const toastPush = vi.fn();
const hosts = { all: [{ id: "plato", baseUrl: "http://plato:7680", apiKey: "k" }] };
vi.mock("../stores/hosts", () => ({ useHostsStore: () => hosts }));
vi.mock("../stores/composer", () => ({ useComposerStore: () => ({ set: composerSet }) }));
vi.mock("../stores/toasts", () => ({ useToastStore: () => ({ push: toastPush }) }));

import { useOpenLiveWork } from "./useOpenLiveWork";

function row(over: Partial<FleetActiveWork> = {}): FleetActiveWork {
  return {
    id: "job-1",
    kind: "generation",
    phase: "running",
    created_at_unix_ms: 0,
    updated_at_unix_ms: 0,
    can_cancel: true,
    key: "plato:generation:job-1",
    hostId: "plato",
    hostLabel: "plato",
    routeUrl: "http://plato:7680",
    instanceId: "i-1",
    stale: false,
    hostError: null,
    ...over,
  } as FleetActiveWork;
}

describe("useOpenLiveWork (desktop adapter)", () => {
  beforeEach(() => {
    push.mockReset();
    push.mockImplementation(async () => undefined);
    findQueueEntryById.mockReset();
    composerSet.mockReset();
    toastPush.mockReset();
  });

  it("looks the machine up in the hosts store, key included, and hands the composer the entry", async () => {
    findQueueEntryById.mockResolvedValue({
      id: "job-1",
      state: "running",
      metadata: { prompt: "a cat", model: "ltx2" },
    });
    await useOpenLiveWork()(row({ execution: null }));
    expect(findQueueEntryById).toHaveBeenCalledWith(
      { baseUrl: "http://plato:7680", apiKey: "k" },
      "job-1",
    );
    expect(composerSet).toHaveBeenCalledWith(
      expect.objectContaining({
        queueSelection: expect.objectContaining({ hostId: "plato", jobId: "job-1" }),
      }),
    );
    expect(push).toHaveBeenCalledWith("/create");
  });

  it("sends a chain row to the Queue and a download row to Styles", async () => {
    await useOpenLiveWork()(row({ execution: "chain" }));
    expect(push).toHaveBeenCalledWith("/queue");
    push.mockClear();
    await useOpenLiveWork()(row({ kind: "download", id: "dl-1" }));
    expect(push).toHaveBeenCalledWith("/models");
  });

  it("toasts, in the app's own shelf, when the machine is unknown", async () => {
    await useOpenLiveWork()(row({ hostId: "ghost", execution: null }));
    expect(toastPush).toHaveBeenCalledWith(expect.any(String), "error");
  });
});
