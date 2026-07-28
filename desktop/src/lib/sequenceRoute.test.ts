import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("./ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));

import { routeForModel } from "./sequenceRoute";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import type { ModelEntry } from "./api/types";

function model(family: string, name = "test-model"): Pick<ModelEntry, "name" | "family"> {
  return { name, family };
}

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

function addCudaRemote(id = "hal9000-7680") {
  const hosts = useHostsStore();
  hosts.extras.push({
    id,
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "remote-key",
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.telemetry[id] = {
    queueDepth: 0,
    queueCapacity: 4,
    version: "1.0.0",
    gpuInfo: { name: "NVIDIA L40S", vram_total_mb: 48000, vram_used_mb: 0, backend: "cuda" },
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  readyLocal();
  useAppPrefsStore().settings = { generateTargetHost: null } as never;
});

describe("routeForModel", () => {
  it("routes an ordinary video model through Auto (the local primary)", () => {
    const route = routeForModel(model("ltx-video"));
    expect(route?.hostId).toBe("local");
  });

  it("escalates Auto to Most capable for CUDA-only families", () => {
    addCudaRemote();
    for (const family of ["ltx2", "ltx-2"]) {
      const route = routeForModel(model(family));
      expect(route?.hostId).toBe("hal9000-7680");
      expect(route?.target.apiKey).toBe("remote-key");
    }
  });

  it("lets a sticky host pick beat the CUDA escalation", () => {
    addCudaRemote();
    useAppPrefsStore().settings = { generateTargetHost: "local" } as never;
    expect(routeForModel(model("ltx2"))?.hostId).toBe("local");
  });

  it("treats a ghost sticky pick as Auto instead of wedging the submit", () => {
    addCudaRemote();
    useAppPrefsStore().settings = { generateTargetHost: "ghost-7680" } as never;
    // Auto = least busy; the remote reports queue depth 0 while the local
    // primary's is unknown. What matters: the ghost id resolves, not null.
    expect(routeForModel(model("ltx-video"))?.hostId).toBe("hal9000-7680");
  });

  it("resolves null for a sticky pick that is connected but not ready", () => {
    addCudaRemote();
    const hosts = useHostsStore();
    hosts.extras[0]!.status = "error";
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    expect(routeForModel(model("ltx-video"))).toBeNull();
  });
});
