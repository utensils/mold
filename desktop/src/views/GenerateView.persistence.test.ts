import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useGenerationStore } from "../stores/generation";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

enableAutoUnmount(afterEach);

const model: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 768,
  default_height: 1152,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

function mountView() {
  // The composer textarea lives in ComposerCard and the model picker in
  // InspectorPanel → ModelPicker — keep all three real so the persisted form
  // + picker close-on outside-click still resolve through the view's DOM.
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      stubs: {
        ComposerCard: false,
        InspectorPanel: false,
        ModelPicker: false,
        ActionBlocker: false,
      },
    },
  });
}

describe("GenerateView form persistence", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson.mockReset();
    apiJson.mockResolvedValue([]);
    apiJsonTo.mockReset();
    apiJsonTo.mockResolvedValue([]);
  });
  afterEach(() => (document.body.innerHTML = ""));

  it("retains the prompt and model across unmount and remount", async () => {
    const first = mountView();
    const store = useGenerateFormStore();
    store.form.prompt = "a lighthouse at dusk";
    store.form.model = "flux-dev:q8";
    await flushPromises();
    first.unmount();

    // A fresh mount (navigating back) reads the same store-backed form.
    const second = mountView();
    await flushPromises();
    expect(second.get("textarea").element.value).toBe("a lighthouse at dusk");
    expect(useGenerateFormStore().form.model).toBe("flux-dev:q8");
  });

  it("auto-select does not clobber a chosen model on remount", async () => {
    const models = useModelStore();
    // Two flux models installed; the user picked the second.
    models.all = [model, { ...model, name: "flux-schnell:q8" }] as ModelEntry[];
    const store = useGenerateFormStore();
    store.form.model = "flux-schnell:q8";
    store.form.family = "flux";

    mountView();
    await flushPromises();

    // The immediate auto-select watch must respect the existing choice.
    expect(store.form.model).toBe("flux-schnell:q8");
  });

  it("closes the model picker when the user clicks elsewhere", async () => {
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
    apiJson.mockResolvedValue([model]);
    useModelStore().all = [model];

    const wrapper = mountView();
    await flushPromises();
    await wrapper.get('[data-test="selected-model-name"]').trigger("click");
    expect(wrapper.find('[data-test="model-option-name"]').exists()).toBe(true);

    await wrapper.get("textarea").trigger("pointerdown");
    await flushPromises();
    expect(wrapper.find('[data-test="model-option-name"]').exists()).toBe(false);
  });

  it("keeps Generate enabled at wan-only 4n+1 frame counts", async () => {
    const wan = {
      ...model,
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      default_frames: 81,
      default_fps: 16,
      frame_step: 4,
      recommended_dimensions: [
        { width: 832, height: 480 },
        { width: 480, height: 832 },
      ],
      dimension_alignment: 16,
      max_pixels: 1280 * 720,
    } as ModelEntry;
    useModelStore().all = [wan];
    const form = useGenerateFormStore().form;
    form.model = wan.name;
    form.family = wan.family;
    form.prompt = "a sailboat crossing the bay";
    form.width = 832;
    form.height = 480;
    form.frames = 45; // valid 4n+1, invalid under the old hard-coded 8n+1 gate
    form.fps = 16;

    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.get('[data-test="generate-button"]').attributes("disabled")).toBeUndefined();
  });

  it("shows the exact LTX blocker and enables a valid conditioned request", async () => {
    const ltx = {
      ...model,
      name: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
      default_fps: 24,
      dimension_alignment: 32,
    } as ModelEntry;
    useModelStore().all = [ltx];
    const form = useGenerateFormStore().form;
    form.model = ltx.name;
    form.family = ltx.family;
    form.prompt = "";
    form.width = 768;
    form.height = 512;
    form.frames = 97;
    form.fps = 24;
    form.sourceVideo = { filename: "guide.mp4", base64: "" };
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");

    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.get('[data-test="action-blocker"]').text()).toContain(
      "Source video cannot be empty.",
    );
    expect(wrapper.get('[data-test="generate-button"]').attributes("disabled")).toBeDefined();
    await wrapper.get('[data-test="generate-button"]').trigger("click");
    expect(submit).not.toHaveBeenCalled();

    form.sourceVideo = { filename: "guide.mp4", base64: "video-bytes" };
    await flushPromises();

    expect(wrapper.find('[data-test="action-blocker"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="generate-button"]').attributes("disabled")).toBeUndefined();
  });

  it("renders the workbench and auto-selects a model installed only on a remote host", async () => {
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
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(target.baseUrl.includes("hal9000") ? [model] : []),
    );

    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.find('[data-test="generate-layout"]').exists()).toBe(true);
    expect(useGenerateFormStore().form.model).toBe("flux-dev:q8");
    expect(useHostModelsStore().hostsFor("flux-dev:q8")).toEqual(["hal9000-7680"]);
  });

  it("shows starter cards only after every connected host reports no models", async () => {
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;

    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.findComponent({ name: "StarterCards" }).exists()).toBe(true);
    expect(wrapper.find('[data-test="generate-layout"]').exists()).toBe(false);
  });
});
