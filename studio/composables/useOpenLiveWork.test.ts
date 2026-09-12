import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetActiveWork } from "../api/activity";

const findQueueEntryById = vi.fn();
vi.mock("../api/queuePlan", () => ({
  findQueueEntryById: (...args: unknown[]) => findQueueEntryById(...args),
}));

import {
  LIVE_WORK_HOST_GONE,
  LIVE_WORK_NO_SETTINGS,
  openLiveWorkWith,
  type LiveWorkSurface,
} from "./useOpenLiveWork";

const go = vi.fn();
const fail = vi.fn();
const restore = vi.fn();
const openDownloads = vi.fn(() => null as string | null);

function surface(
  over: Partial<LiveWorkSurface<Record<string, unknown>>> = {},
): LiveWorkSurface<Record<string, unknown>> {
  return {
    targetFor: (hostId) =>
      hostId === "plato"
        ? { baseUrl: "http://plato:7680", apiKey: null }
        : null,
    go,
    fail,
    restore,
    chainDestination: (row) => `/machines/${row.hostId}`,
    openDownloads,
    ...over,
  };
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

describe("openLiveWorkWith", () => {
  beforeEach(() => {
    go.mockReset();
    fail.mockReset();
    restore.mockReset();
    findQueueEntryById.mockReset();
    openDownloads.mockReset().mockReturnValue(null);
  });

  // The auto-chain regression: a long video the host split and stitched is
  // `kind: "generation"` with `execution: "chain"`. Its id lives in the
  // chain-job space, so the ordinary generation reattach searches /api/queue
  // for an id that is not there and dead-ends on "cannot restore settings".
  // It must be recognised BEFORE the generation arm.
  it("sends a chain row where its surface says, without a queue lookup", async () => {
    await openLiveWorkWith(surface())(row({ execution: "chain" }));
    expect(findQueueEntryById).not.toHaveBeenCalled();
    expect(fail).not.toHaveBeenCalled();
    expect(go).toHaveBeenCalledWith("/machines/plato");
  });

  it("lets each surface pick that destination for itself", async () => {
    await openLiveWorkWith(surface({ chainDestination: () => "/queue" }))(
      row({ execution: "chain" }),
    );
    expect(go).toHaveBeenCalledWith("/queue");
  });

  // An older host still labels a chain row with its own kind.
  it("sends a legacy sequence row the same way", async () => {
    await openLiveWorkWith(surface())(row({ kind: "sequence" }));
    expect(findQueueEntryById).not.toHaveBeenCalled();
    expect(go).toHaveBeenCalledWith("/machines/plato");
  });

  it("reattaches an ordinary generation through the queue", async () => {
    findQueueEntryById.mockResolvedValue({
      id: "job-1",
      state: "running",
      metadata: { prompt: "a cat", model: "ltx2" },
    });
    await openLiveWorkWith(surface())(row({ id: "job-1", execution: null }));
    expect(findQueueEntryById).toHaveBeenCalledWith(
      { baseUrl: "http://plato:7680", apiKey: null },
      "job-1",
    );
    expect(restore).toHaveBeenCalledWith(
      expect.objectContaining({ jobId: "job-1", running: true }),
      "plato",
    );
    expect(go).toHaveBeenCalledWith("/create");
  });

  it("says the machine has gone rather than asking it anything", async () => {
    await openLiveWorkWith(surface())(
      row({ id: "job-1", execution: null, hostId: "gone" }),
    );
    expect(findQueueEntryById).not.toHaveBeenCalled();
    expect(fail).toHaveBeenCalledWith(LIVE_WORK_HOST_GONE);
    expect(go).not.toHaveBeenCalled();
  });

  it("says so when the machine hands back no settings", async () => {
    findQueueEntryById.mockResolvedValue(null);
    await openLiveWorkWith(surface())(row({ id: "job-1", execution: null }));
    expect(fail).toHaveBeenCalledWith(LIVE_WORK_NO_SETTINGS);
    expect(go).not.toHaveBeenCalled();
  });

  it("reports a failed lookup in the surface's own words", async () => {
    findQueueEntryById.mockRejectedValue(new Error("connection refused"));
    await openLiveWorkWith(surface())(row({ id: "job-1", execution: null }));
    expect(fail).toHaveBeenCalledWith("connection refused");
  });

  /*
   * A 3-D stage is admitted as an ordinary generation, so it reaches here
   * looking like any other print. New image cannot resume a durable workflow.
   */
  it("routes a 3-D workflow stage to the workflow, not to New image", async () => {
    findQueueEntryById.mockResolvedValue({
      id: "job-1",
      state: "running",
      metadata: {
        prompt: "a cat",
        mesh_workflow: {
          job_id: "wf-9",
          mode: "text_to_3d",
          role: "final_glb",
          stage_index: 2,
        },
      },
    });
    await openLiveWorkWith(surface())(row({ id: "job-1", execution: null }));
    expect(restore).not.toHaveBeenCalled();
    expect(go).toHaveBeenCalledWith(
      expect.objectContaining({
        query: expect.objectContaining({ workflow: "wf-9", host: "plato" }),
      }),
    );
  });

  it("lets the surface answer a download row without navigating", async () => {
    await openLiveWorkWith(surface())(row({ kind: "download" }));
    expect(openDownloads).toHaveBeenCalled();
    expect(go).not.toHaveBeenCalled();
  });

  it("navigates for a surface whose downloads live on a page", async () => {
    openDownloads.mockReturnValue("/models");
    await openLiveWorkWith(surface())(row({ kind: "download" }));
    expect(go).toHaveBeenCalledWith("/models");
  });

  it("falls back to the machine for anything else", async () => {
    await openLiveWorkWith(surface())(
      row({ kind: "upscale" } as Partial<FleetActiveWork>),
    );
    expect(go).toHaveBeenCalledWith("/machines/plato");
  });
});
