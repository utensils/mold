import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { ApiTarget } from "../lib/api/client";
import type { ChainRequest } from "../lib/api/types";

const { apiJson, apiJsonTo, sseStream } = vi.hoisted(() => ({
  apiJson: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJson,
  apiJsonTo,
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));

import { useChainJobsStore } from "./chainJobs";

const target: ApiTarget = { baseUrl: "http://hal9000:7680", apiKey: "remote-key" };
const request = {
  schema: "mold.chain.v1",
  chain: {},
  stage: [],
} as unknown as ChainRequest;

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiJson.mockResolvedValue({ jobs: [] });
  apiJsonTo.mockImplementation((_target: ApiTarget, path: string) =>
    Promise.resolve(path === "/api/chain-jobs" ? { job_id: "remote-chain-1", jobs: [] } : {}),
  );
});

describe("chain jobs on an explicit host", () => {
  it("creates, refreshes, and watches a chain through the same remote target", async () => {
    await useChainJobsStore().create(request, target);

    expect(apiJson).not.toHaveBeenCalled();
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/chain-jobs",
      expect.objectContaining({ method: "POST" }),
    );
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/chain-jobs");
    expect(sseStream).toHaveBeenCalledWith(
      "/api/chain-jobs/remote-chain-1/events",
      expect.objectContaining({ target }),
    );
  });

  it("stops a previous host watch before creating on a different host", async () => {
    const store = useChainJobsStore();
    store.watch("local-chain", { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" });
    const previousAbort = store.abort!;
    apiJsonTo.mockImplementation((_target: ApiTarget, path: string, init?: RequestInit) => {
      if (path === "/api/chain-jobs" && init?.method === "POST") {
        expect(previousAbort.signal.aborted).toBe(true);
        return Promise.resolve({ job_id: "remote-chain-1" });
      }
      return Promise.resolve({ jobs: [] });
    });

    await store.create(request, target);

    expect(previousAbort.signal.aborted).toBe(true);
    expect(store.target).toEqual(target);
  });
});
