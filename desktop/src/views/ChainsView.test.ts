import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { ModelEntry } from "../lib/api/types";

const { apiJson, apiJsonTo, sseStream } = vi.hoisted(() => ({
  apiJson: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJson,
  apiJsonTo,
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" }),
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));

import ChainsView from "./ChainsView.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";

const videoModel: ModelEntry = {
  name: "ltx-video-0.9.6:bf16",
  family: "ltx-video",
  downloaded: true,
  is_loaded: false,
  hf_repo: "Lightricks/LTX-Video",
  size_gb: 12,
  default_width: 768,
  default_height: 512,
  default_steps: 30,
  default_guidance: 3,
  description: "",
};

function installRemoteVideoHost() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";

  const hosts = useHostsStore();
  hosts.initialized = true;
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "remote-key",
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.telemetry["hal9000-7680"] = {
    queueDepth: 0,
    queueCapacity: 8,
    version: "0.17.1",
    modelsLoaded: [],
    gpuInfo: {
      backend: "cuda",
      name: "NVIDIA RTX 5090",
      vram_total_mb: 32_768,
      vram_used_mb: 0,
    },
    instanceId: "hal9000",
    hostname: "hal9000",
  };
  useHostModelsStore().byHost["hal9000-7680"] = {
    entries: [videoModel],
    fetchedAt: Date.now(),
    error: null,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiJson.mockImplementation((path: string) => {
    if (path === "/api/models") return Promise.resolve([]);
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path === "/api/resources") {
      return Promise.resolve({ hostname: "local", gpus: [{ backend: "metal" }], system_ram: {} });
    }
    return Promise.resolve({});
  });
  apiJsonTo.mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/models") return Promise.resolve([]);
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path.startsWith("/api/capabilities/chain-limits")) {
      return Promise.resolve({
        max_stages: 8,
        max_total_frames: 777,
        fade_frames_max: 32,
        supports_audio: false,
      });
    }
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    return Promise.resolve({ job_id: "remote-chain-1" });
  });
});

describe("ChainsView multi-host video generation", () => {
  it("shows a video model installed only on a connected remote host", async () => {
    installRemoteVideoHost();

    const wrapper = mount(ChainsView, { shallow: true });
    await flushPromises();

    expect(wrapper.text()).not.toContain("No video models yet");
    expect(wrapper.get("select").text()).toContain("ltx-video-0.9.6:bf16");
  });
});
