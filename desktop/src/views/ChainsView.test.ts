import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { ModelEntry } from "../lib/api/types";
import chainsViewSource from "./ChainsView.vue?raw";

const { apiJson, apiJsonTo, sseStream, routerPush } = vi.hoisted(() => ({
  apiJson: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn().mockResolvedValue(undefined),
  routerPush: vi.fn(),
}));

vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));
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
import { useModelStore } from "../stores/models";
import { useAppPrefsStore } from "../stores/appPrefs";

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
  it("marks Sequence active in the composer mode switch and returns to Create on Single", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, { global: { stubs: { DevelopCanvas: true } } });
    await flushPromises();
    const segments = wrapper.get("[data-test='composer-mode']").findAll("button[role='radio']");
    expect(segments.map((b) => b.text())).toEqual(["Single", "Sequence"]);
    expect(segments[1]!.attributes("aria-checked")).toBe("true");
    await segments[0]!.trigger("click");
    expect(routerPush).toHaveBeenCalledWith("/create");
  });

  it("closes the File tools menu on outside pointerdown and on Escape", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
      attachTo: document.body,
    });
    await flushPromises();

    await wrapper.get("[data-test='file-tools-toggle']").trigger("click");
    expect(wrapper.find("[data-test='file-tools-menu']").exists()).toBe(true);
    document.body.dispatchEvent(new MouseEvent("pointerdown", { bubbles: true }));
    await flushPromises();
    expect(wrapper.find("[data-test='file-tools-menu']").exists()).toBe(false);

    await wrapper.get("[data-test='file-tools-toggle']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='file-tools-menu']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("closes the File tools menu after picking an action", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
      attachTo: document.body,
    });
    await flushPromises();

    await wrapper.get("[data-test='file-tools-toggle']").trigger("click");
    const viewToml = wrapper
      .get("[data-test='file-tools-menu']")
      .findAll("button")
      .find((b) => b.text().includes("View as TOML"));
    await viewToml!.trigger("click");
    expect(wrapper.find("[data-test='file-tools-menu']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("lists chain-incapable video checkpoints disabled with a reason", async () => {
    installRemoteVideoHost();
    const hostModels = useHostModelsStore();
    vi.spyOn(hostModels, "refresh").mockResolvedValue();
    hostModels.byHost["hal9000-7680"]!.entries.push({
      ...videoModel,
      name: "ltx-2.3-22b-dev:fp8",
      family: "ltx2",
      hf_repo: "Lightricks/LTX-2.3",
    });
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
      attachTo: document.body,
    });
    await flushPromises();

    // The compatible ltx-video checkpoint is auto-selected, not the dev one.
    expect(wrapper.get("[data-test='selected-model-name']").text()).toContain("ltx-video");

    await wrapper.get(".ms-model__button").trigger("click");
    await flushPromises();
    const options = wrapper.findAll(".ms-model__option");
    const dev = options.find((o) => o.text().includes("ltx-2.3-22b-dev:fp8"));
    expect(dev).toBeTruthy();
    expect(dev!.attributes("disabled")).toBeDefined();
    expect(dev!.get("[data-test='model-disabled-reason']").text()).toContain("Two-stage");
    wrapper.unmount();
  });

  it("keeps fixed rows rigid so a long jobs list can't crush the filmstrip", () => {
    // Flex children of the h-full overflow-y-auto column shrink to fit the
    // container before it scrolls — every fixed row needs shrink-0 or a long
    // jobs list compresses the clip cards into overlapping slivers.
    const classesFor = (testId: string) =>
      chainsViewSource
        .match(new RegExp(`<[^>]*data-test="${testId}"[^>]*>`, "s"))?.[0]
        ?.match(/class="([^"]*)"/s)?.[1] ?? "";
    expect(classesFor("chain-header")).toContain("shrink-0");
    expect(classesFor("chain-filmstrip")).toContain("shrink-0");
    expect(classesFor("chain-footer")).toContain("shrink-0");
    expect(classesFor("toml-panel")).toContain("shrink-0");
  });

  it("keeps the Create header anatomy: title, summary, mode switch, and host chip", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, { global: { stubs: { DevelopCanvas: true } } });
    await flushPromises();
    const header = wrapper.get("[data-test='chain-header']");
    expect(header.text()).toContain("Untitled sequence");
    expect(header.find("[data-test='composer-mode']").exists()).toBe(true);
    expect(header.get("[data-test='chain-summary']").text()).toMatch(/\d+ clips/);
    expect(header.find("[data-test='host-chip']").exists()).toBe(true);
    wrapper.unmount();
  });

  it("routes chain limits to the sticky generation host over auto least-busy", async () => {
    installRemoteVideoHost();
    // Local also has the model and is less busy, so Auto would pick local —
    // only the sticky pick explains routing to hal9000.
    const hosts = useHostsStore();
    hosts.telemetry["hal9000-7680"]!.queueDepth = 5;
    useHostModelsStore().byHost.local = {
      entries: [videoModel],
      fetchedAt: Date.now(),
      error: null,
    };
    hosts.telemetry.local = {
      queueDepth: 0,
      queueCapacity: 8,
      version: "0.17.1",
      modelsLoaded: [],
      gpuInfo: {
        backend: "metal",
        name: "Apple Metal GPU",
        vram_total_mb: 32_768,
        vram_used_mb: 0,
      },
      instanceId: "local",
      hostname: "local",
    };
    useModelStore().all = [videoModel];
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;

    mount(ChainsView, { shallow: true });
    await flushPromises();

    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      expect.stringContaining("/api/capabilities/chain-limits"),
    );
  });

  it("edits the sequence through the highlighted TOML panel", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
      attachTo: document.body,
    });
    await flushPromises();

    await wrapper.get("[data-test='file-tools-toggle']").trigger("click");
    const viewToml = wrapper
      .get("[data-test='file-tools-menu']")
      .findAll("button")
      .find((b) => b.text().includes("View as TOML"));
    await viewToml!.trigger("click");

    const editor = wrapper.get<HTMLTextAreaElement>("[data-test='toml-editor']");
    expect(editor.element.value).toContain("[chain]");
    // The highlight layer mirrors the draft with token spans.
    expect(wrapper.get(".ms-toml__hl").html()).toContain('class="tk-table"');

    // A third stage applied through the editor lands in the filmstrip form.
    const withThirdStage = `${editor.element.value}\n[[stage]]\nprompt = "the end"\nframes = 25\ntransition = "cut"\n`;
    await editor.setValue(withThirdStage);
    await wrapper.get("[data-test='toml-apply']").trigger("click");
    const state = wrapper.vm as unknown as { form: { stages: { prompt: string }[] } };
    expect(state.form.stages).toHaveLength(3);
    expect(state.form.stages[2]!.prompt).toBe("the end");
    expect(wrapper.find("[data-test='toml-error']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("surfaces a parse error inline instead of clobbering the sequence", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
      attachTo: document.body,
    });
    await flushPromises();

    await wrapper.get("[data-test='file-tools-toggle']").trigger("click");
    const viewToml = wrapper
      .get("[data-test='file-tools-menu']")
      .findAll("button")
      .find((b) => b.text().includes("View as TOML"));
    await viewToml!.trigger("click");

    const editor = wrapper.get<HTMLTextAreaElement>("[data-test='toml-editor']");
    await editor.setValue("not = toml [");
    await wrapper.get("[data-test='toml-apply']").trigger("click");
    expect(wrapper.get("[data-test='toml-error']").text()).toContain("Couldn't parse");
    const state = wrapper.vm as unknown as { form: { stages: unknown[] } };
    expect(state.form.stages).toHaveLength(2);
    wrapper.unmount();
  });

  it("starts with two guided clips and requires every prompt", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
    });
    await flushPromises();

    expect(wrapper.findAllComponents({ name: "StageCard" })).toHaveLength(2);
    expect(wrapper.get("[data-test='generate-sequence']").text()).toBe("Generate sequence");
    expect(
      (wrapper.get("[data-test='generate-sequence']").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("never auto-selects a non-chain LTX development checkpoint", async () => {
    installRemoteVideoHost();
    useHostModelsStore().byHost["hal9000-7680"]!.entries = [
      { ...videoModel, name: "ltx-2.3-22b-dev:fp8", family: "ltx2" },
      videoModel,
    ];

    const wrapper = mount(ChainsView, { global: { stubs: { DevelopCanvas: true } } });
    await flushPromises();

    expect(wrapper.get("[data-test='selected-model-name']").text()).not.toContain("22b-dev");
    expect(wrapper.get("[data-test='selected-model-name']").text()).toContain("ltx-video");
  });

  it("refuses to render when the sticky host isn't ready instead of rerouting", async () => {
    installRemoteVideoHost();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const hosts = useHostsStore();
    hosts.extras[0]!.status = "error";

    const wrapper = mount(ChainsView, { global: { stubs: { DevelopCanvas: true } } });
    await flushPromises();
    const state = wrapper.vm as unknown as {
      form: { stages: { prompt: string }[] };
      render: () => Promise<void>;
    };
    state.form.stages.forEach((stage, index) => (stage.prompt = `clip ${index}`));
    apiJson.mockClear();
    apiJsonTo.mockClear();

    await state.render();

    expect(apiJson).not.toHaveBeenCalledWith("/api/chain-jobs", expect.anything());
    expect(apiJsonTo).not.toHaveBeenCalledWith(
      expect.anything(),
      "/api/chain-jobs",
      expect.anything(),
    );
    const { useToastStore } = await import("../stores/toasts");
    expect(useToastStore().items.some((t) => t.message.includes("isn't reachable"))).toBe(true);
    wrapper.unmount();
  });

  it("does not enable generation for an imported unavailable model", async () => {
    installRemoteVideoHost();
    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
    });
    await flushPromises();

    const state = wrapper.vm as unknown as {
      form: { model: string; stages: { prompt: string }[] };
    };
    state.form.model = "ltx-2.3-22b-dev:fp8";
    state.form.stages.forEach((stage, index) => {
      stage.prompt = `clip ${index + 1}`;
    });
    await wrapper.vm.$nextTick();

    expect(
      (wrapper.get("[data-test='generate-sequence']").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("keeps local LTX-2 available when any local GPU is CUDA", async () => {
    const ltx2 = {
      ...videoModel,
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx-2",
    };
    const conn = useConnectionStore();
    conn.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    conn.status = "ready";
    apiJson.mockImplementation((path: string) => {
      if (path === "/api/models") return Promise.resolve([ltx2]);
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      if (path === "/api/resources") {
        return Promise.resolve({
          hostname: "local",
          gpus: [{ backend: "metal" }, { backend: "cuda" }],
          system_ram: {},
        });
      }
      return Promise.resolve({});
    });

    const wrapper = mount(ChainsView, {
      global: { stubs: { DevelopCanvas: true } },
    });
    await flushPromises();

    await wrapper.get(".ms-model__button").trigger("click");
    const option = wrapper
      .findAll(".ms-model__option")
      .find((candidate) => candidate.text().includes(ltx2.name));
    expect(option).toBeDefined();
    if (!option) throw new Error("expected local LTX-2 model option");
    expect(option.attributes("disabled")).toBeUndefined();
    expect(option.text()).not.toContain("CUDA only");
  });

  it("shows a video model installed only on a connected remote host", async () => {
    installRemoteVideoHost();

    const wrapper = mount(ChainsView, { global: { stubs: { DevelopCanvas: true } } });
    await flushPromises();

    expect(wrapper.text()).not.toContain("No video models yet");
    expect(wrapper.get("[data-test='selected-model-name']").text()).toContain(
      "ltx-video-0.9.6:bf16",
    );
  });

  it("uses the remote CUDA host when LTX-2 is also installed on local Metal", async () => {
    const ltx2 = { ...videoModel, name: "ltx-2-19b-distilled:fp8", family: "ltx-2" };
    installRemoteVideoHost();
    useHostModelsStore().byHost["hal9000-7680"]!.entries = [ltx2];
    useHostModelsStore().byHost.local = {
      entries: [ltx2],
      fetchedAt: Date.now(),
      error: null,
    };
    useHostsStore().telemetry.local = {
      queueDepth: 0,
      queueCapacity: 8,
      version: "0.17.1",
      modelsLoaded: [],
      gpuInfo: {
        backend: "metal",
        name: "Apple Metal GPU",
        vram_total_mb: 32_768,
        vram_used_mb: 0,
      },
      instanceId: "local",
      hostname: "local",
    };
    useModelStore().all = [ltx2];

    mount(ChainsView, { shallow: true });
    await flushPromises();

    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      expect.stringContaining("/api/capabilities/chain-limits"),
    );
  });
});
