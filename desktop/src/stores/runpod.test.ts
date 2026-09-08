import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const runpodOverview = vi.fn().mockResolvedValue({
  configured: true,
  credentialSource: "app",
  account: null,
  pods: [],
  gpus: [],
  datacenters: [],
  networkVolumes: [],
});
const runpodDelete = vi.fn();
const runpodNetworkVolumeCreate = vi.fn();
const runpodNetworkVolumeUpdate = vi.fn();
const runpodNetworkVolumeDelete = vi.fn();
const disconnectHost = vi.fn();
const forgetRemoteHost = vi.fn();
const appSettingsGet = vi.fn();
const defaultExtraHosts = [
  {
    id: "pod-123-7680-proxy-runpod-net",
    url: "https://pod-123-7680.proxy.runpod.net",
  },
  { id: "hal9000-7680", url: "http://hal9000:7680" },
];
const extraHosts = [...defaultExtraHosts];

vi.mock("../lib/ipc", () => ({
  ipc: {
    runpodOverview: (...args: unknown[]) => runpodOverview(...args),
    runpodCreate: vi.fn().mockRejectedValue(new Error("GPU is unavailable for Pod creation")),
    runpodDelete: (...args: unknown[]) => runpodDelete(...args),
    runpodNetworkVolumeCreate: (...args: unknown[]) => runpodNetworkVolumeCreate(...args),
    runpodNetworkVolumeUpdate: (...args: unknown[]) => runpodNetworkVolumeUpdate(...args),
    runpodNetworkVolumeDelete: (...args: unknown[]) => runpodNetworkVolumeDelete(...args),
    forgetRemoteHost: (...args: unknown[]) => forgetRemoteHost(...args),
    appSettingsGet: (...args: unknown[]) => appSettingsGet(...args),
  },
}));

vi.mock("./hosts", () => ({
  useHostsStore: () => ({ extras: extraHosts, disconnect: disconnectHost }),
}));

import { useRunPodStore } from "./runpod";

describe("RunPod operation errors", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    runpodOverview.mockClear();
    runpodDelete.mockReset().mockResolvedValue(undefined);
    runpodNetworkVolumeCreate.mockReset().mockResolvedValue({ id: "volume-created" });
    runpodNetworkVolumeUpdate.mockReset().mockResolvedValue({ id: "volume-updated" });
    runpodNetworkVolumeDelete.mockReset().mockResolvedValue(undefined);
    disconnectHost.mockReset().mockResolvedValue(undefined);
    forgetRemoteHost.mockReset().mockResolvedValue([]);
    appSettingsGet.mockReset().mockResolvedValue({ savedHosts: [] });
    extraHosts.splice(0, extraHosts.length, ...defaultExtraHosts);
  });

  it("keeps launch errors available across background refreshes until dismissed", async () => {
    const store = useRunPodStore();
    await expect(store.create({} as never)).rejects.toThrow("GPU is unavailable");
    expect(store.operationError).toContain("GPU is unavailable");

    await store.load();
    expect(store.operationError).toContain("GPU is unavailable");

    store.clearOperationError();
    expect(store.operationError).toBeNull();
  });

  it("keeps volume mutations busy through refresh and returns their IPC result", async () => {
    const store = useRunPodStore();

    const creating = store.createNetworkVolume({ name: "models" } as never);

    expect(store.mutating).toBe("volume:create");
    expect(store.operationError).toBeNull();
    await expect(creating).resolves.toEqual({ id: "volume-created" });
    expect(store.mutating).toBeNull();
    expect(runpodOverview).toHaveBeenCalledOnce();

    const updating = store.updateNetworkVolume({ id: "volume-created" } as never);

    expect(store.mutating).toBe("volume:update:volume-created");
    await expect(updating).resolves.toEqual({ id: "volume-updated" });
    expect(store.mutating).toBeNull();
    expect(runpodOverview).toHaveBeenCalledTimes(2);
  });

  it("records volume mutation failures and always clears the busy state", async () => {
    runpodNetworkVolumeDelete.mockRejectedValueOnce(new Error("volume is attached"));
    const store = useRunPodStore();
    store.operationError = "old failure";

    const deleting = store.deleteNetworkVolume("volume-123");

    expect(store.mutating).toBe("volume:delete:volume-123");
    expect(store.operationError).toBeNull();
    await expect(deleting).rejects.toThrow("volume is attached");
    expect(store.operationError).toContain("volume is attached");
    expect(store.mutating).toBeNull();
    expect(runpodOverview).not.toHaveBeenCalled();
  });

  it("disconnects the deleted pod's Mold host so its download queue is retired", async () => {
    const store = useRunPodStore();

    await store.act("delete", "pod-123");

    expect(runpodDelete).toHaveBeenCalledWith("pod-123");
    expect(disconnectHost).toHaveBeenCalledOnce();
    expect(disconnectHost).toHaveBeenCalledWith("pod-123-7680-proxy-runpod-net");
    expect(forgetRemoteHost).toHaveBeenCalledOnce();
    expect(forgetRemoteHost).toHaveBeenCalledWith("pod-123-7680-proxy-runpod-net");
  });

  it("keeps the host connected when RunPod deletion fails", async () => {
    runpodDelete.mockRejectedValueOnce(new Error("RunPod deletion failed"));
    const store = useRunPodStore();

    await expect(store.act("delete", "pod-123")).rejects.toThrow("RunPod deletion failed");

    expect(disconnectHost).not.toHaveBeenCalled();
    expect(forgetRemoteHost).not.toHaveBeenCalled();
  });

  it("still forgets the host when ordinary disconnect persistence fails", async () => {
    disconnectHost.mockRejectedValueOnce(new Error("settings unavailable"));
    const store = useRunPodStore();

    await expect(store.act("delete", "pod-123")).resolves.toBeUndefined();

    expect(runpodOverview).toHaveBeenCalledOnce();
    expect(forgetRemoteHost).toHaveBeenCalledWith("pod-123-7680-proxy-runpod-net");
    expect(store.operationError).toBeNull();
  });

  it("refreshes after deletion and warns when the host cannot be fully forgotten", async () => {
    forgetRemoteHost.mockRejectedValueOnce(new Error("settings unavailable"));
    const store = useRunPodStore();

    await expect(store.act("delete", "pod-123")).resolves.toBeUndefined();

    expect(runpodOverview).toHaveBeenCalledOnce();
    expect(store.operationError).toContain("was deleted");
    expect(store.operationError).toContain("may reappear after restart");
  });

  it("forgets a matching saved RunPod host even when it is already disconnected", async () => {
    extraHosts.splice(0);
    appSettingsGet.mockResolvedValueOnce({
      savedHosts: [
        {
          id: "saved-pod-alias",
          url: "http://pod-123-7680.proxy.runpod.net:80/api",
        },
        { id: "hal9000-7680", url: "http://hal9000:7680" },
      ],
    });
    const store = useRunPodStore();

    await store.act("delete", "pod-123");

    expect(disconnectHost).not.toHaveBeenCalled();
    expect(forgetRemoteHost).toHaveBeenCalledOnce();
    expect(forgetRemoteHost).toHaveBeenCalledWith("saved-pod-alias");
  });

  it("forgets multiple matching aliases sequentially", async () => {
    extraHosts.splice(0);
    appSettingsGet.mockResolvedValueOnce({
      savedHosts: [
        { id: "pod-alias-a", url: "https://pod-123-7680.proxy.runpod.net" },
        { id: "pod-alias-b", url: "http://pod-123-7680.proxy.runpod.net:80/api" },
      ],
    });
    let active = 0;
    let maxActive = 0;
    forgetRemoteHost.mockImplementation(async () => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await Promise.resolve();
      active -= 1;
      return [];
    });
    const store = useRunPodStore();

    await store.act("delete", "pod-123");

    expect(forgetRemoteHost.mock.calls.map(([id]) => id)).toEqual(["pod-alias-a", "pod-alias-b"]);
    expect(maxActive).toBe(1);
  });
});
