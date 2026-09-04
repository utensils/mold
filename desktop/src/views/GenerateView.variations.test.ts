/**
 * "Make 4 variations" is the same picture, made again as a batch of four —
 * nothing about the words, the style, the size or the seed policy changes. It
 * is reachable from the canvas caption, from ⌥↩, and from the command palette,
 * and all three are the one action on the UI store.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

import GenerateView from "./GenerateView.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useUiStore } from "../stores/ui";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

const sdxlModel: ModelEntry = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.0,
} as ModelEntry;

beforeEach(() => {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = [sdxlModel];
  useHostModelsStore().byHost.local = {
    entries: [sdxlModel],
    fetchedAt: Date.now(),
    error: null,
  };
});
afterEach(() => (document.body.innerHTML = ""));

async function mountWithAPrint() {
  mount(GenerateView, { shallow: true, attachTo: document.body });
  await flushPromises();
  const form = useGenerateFormStore().form;
  form.prompt = "a brass teapot on a rainy windowsill";
  form.model = sdxlModel.name;
  form.family = sdxlModel.family;
  form.batchSize = 1;
  form.seed = "4821";
  await flushPromises();
  return form;
}

describe("GenerateView — Make 4 variations", () => {
  it("re-submits the print at batch four, leaving every other setting alone", async () => {
    const form = await mountWithAPrint();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().makeVariations();
    await flushPromises();

    expect(form.batchSize).toBe(4);
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit.mock.calls[0]![1]).toBe(4);
    const request = submit.mock.calls[0]![0];
    expect(request.prompt).toBe("a brass teapot on a rainy windowsill");
    expect(request.model).toBe(sdxlModel.name);
    expect(form.seed).toBe("4821");
  });

  it("stays out of the way in clip mode, where Generate does not mean a batch", async () => {
    const form = await mountWithAPrint();
    useSequenceDraftStore().output = "sequence";
    await flushPromises();
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    useUiStore().makeVariations();
    await flushPromises();

    expect(form.batchSize).toBe(1);
    expect(submit).not.toHaveBeenCalled();
  });
});
