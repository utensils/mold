import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
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
    global: { stubs: { ComposerCard: false, InspectorPanel: false, ModelPicker: false } },
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
