import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/ipc", () => ({
  ipc: {
    runpodOverview: vi.fn().mockResolvedValue({
      configured: true,
      credentialSource: "keychain",
      account: null,
      pods: [],
      gpus: [],
      datacenters: [],
      networkVolumes: [],
    }),
    runpodCreate: vi.fn().mockRejectedValue(new Error("GPU is unavailable for Pod creation")),
  },
}));

import { useRunPodStore } from "./runpod";

describe("RunPod operation errors", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("keeps launch errors available across background refreshes until dismissed", async () => {
    const store = useRunPodStore();
    await expect(store.create({} as never)).rejects.toThrow("GPU is unavailable");
    expect(store.operationError).toContain("GPU is unavailable");

    await store.load();
    expect(store.operationError).toContain("GPU is unavailable");

    store.clearOperationError();
    expect(store.operationError).toBeNull();
  });
});
