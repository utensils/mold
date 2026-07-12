import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

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
  return mount(GenerateView, { shallow: true, attachTo: document.body });
}

describe("GenerateView form persistence", () => {
  beforeEach(() => setActivePinia(createPinia()));
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
});
