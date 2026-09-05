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
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";

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
  // The composer textarea lives in ComposerCard and the style picker on its
  // chip (ComposerCard's `style` slot → StylePicker → ModelPicker) — keep all
  // four real so the persisted form and the picker's close-on-outside-click
  // still resolve through the view's DOM.
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      stubs: {
        ComposerCard: false,
        InspectorPanel: false,
        StylePicker: false,
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

  it("disables an empty required prompt without showing an obvious blocker", async () => {
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.model = model.name;
    form.family = model.family;
    form.prompt = "";

    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.get('[data-test="generate-button"]').attributes("disabled")).toBeDefined();
    expect(wrapper.find('[data-test="action-blocker"]').exists()).toBe(false);
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

/*
 * Each section keeps the style it was last used with, across a restart. A
 * fresh launch used to open on FLUX or the first style installed whatever the
 * person had been using; now it opens on the style — and so the section —
 * they left, once the machine that has it has reported in.
 */
describe("GenerateView opens on the style last used", () => {
  const still = model;
  const clip = {
    ...model,
    name: "ltx-video",
    family: "ltx-video",
    default_width: 1024,
    default_height: 576,
  } as ModelEntry;

  beforeEach(() => {
    setActivePinia(createPinia());
    apiJson.mockReset();
    apiJson.mockResolvedValue([]);
    apiJsonTo.mockReset();
    apiJsonTo.mockResolvedValue([]);
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
  });
  afterEach(() => (document.body.innerHTML = ""));

  it("lands on the style and section left last, not the first installed", async () => {
    useLastUsedStylesStore().remember("still", still.name);
    useLastUsedStylesStore().remember("clip", clip.name);
    useModelStore().all = [still, clip];
    useHostModelsStore().byHost.local = {
      entries: [still, clip],
      fetchedAt: Date.now(),
      error: null,
    };

    mountView();
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe(clip.name);
  });

  it("falls back to the usual pick when nothing was remembered", async () => {
    useModelStore().all = [clip, still];
    useHostModelsStore().byHost.local = {
      entries: [clip, still],
      fetchedAt: Date.now(),
      error: null,
    };
    mountView();
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe(still.name);
  });

  it("waits for a machine still reporting before giving up on the remembered style", async () => {
    // The remembered clip style lives on plato, which answers after this
    // device. Picking FLUX from the first list to arrive would lock the form
    // before plato's inventory could restore what the person was using.
    useLastUsedStylesStore().remember("clip", clip.name);
    useModelStore().all = [still];
    useHostModelsStore().byHost.local = { entries: [still], fetchedAt: Date.now(), error: null };
    useHostsStore().extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    // plato's inventory is in flight until the test lets it land.
    let landPlato: (entries: ModelEntry[]) => void = () => {};
    apiJsonTo.mockImplementation((target: unknown, path: unknown) => {
      if (
        path === "/api/models" &&
        (target as { baseUrl?: string }).baseUrl === "http://plato:7680"
      ) {
        return new Promise<ModelEntry[]>((resolve) => (landPlato = resolve));
      }
      return Promise.resolve([]);
    });

    mountView();
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe("");

    landPlato([clip]);
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe(clip.name);
  });

  it("waits for a machine still reconnecting at launch, which the fetched count ignores", async () => {
    // Boot reconnect leaves a remembered machine `connecting` while this
    // device's inventory has already landed; `allReadyHostsFetched` counts
    // only READY machines, so it read settled and the fallback locked the
    // form before the machine holding the style could answer.
    useLastUsedStylesStore().remember("clip", clip.name);
    useModelStore().all = [still];
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [still], fetchedAt: Date.now(), error: null };
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "connecting",
      error: null,
      instanceId: null,
    });
    apiJsonTo.mockImplementation((_target: unknown, path: unknown) =>
      path === "/api/models" ? new Promise<never>(() => {}) : Promise.resolve([]),
    );

    mountView();
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe("");

    hosts.extras[0]!.status = "ready";
    hostModels.byHost["plato-7680"] = { entries: [clip], fetchedAt: Date.now(), error: null };
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe(clip.name);
  });

  it("settles for the usual pick once every machine has reported without it", async () => {
    useLastUsedStylesStore().remember("clip", "wan22-ti2v-5b:dmd");
    useModelStore().all = [still];
    useHostModelsStore().byHost.local = { entries: [still], fetchedAt: Date.now(), error: null };
    mountView();
    await flushPromises();
    expect(useGenerateFormStore().form.model).toBe(still.name);
  });

  it("remembers the style as it changes, under the section the style belongs to", async () => {
    useModelStore().all = [still, clip];
    useHostModelsStore().byHost.local = {
      entries: [still, clip],
      fetchedAt: Date.now(),
      error: null,
    };
    mountView();
    await flushPromises();
    const memory = useLastUsedStylesStore();
    expect(memory.bySection.still).toBe(still.name);

    const form = useGenerateFormStore().form;
    form.model = clip.name;
    form.family = clip.family;
    await flushPromises();
    expect(memory.bySection.clip).toBe(clip.name);
    expect(memory.bySection.still).toBe(still.name);
    expect(memory.lastSection).toBe("clip");
  });
});
