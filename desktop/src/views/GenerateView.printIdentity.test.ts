/**
 * A print's identity — the typed title and its "File under" filing — belongs
 * to the print, not to the parameters. Only ⌘N ("new print") clears it, so
 * loading a template applies that template's settings WITHOUT renaming or
 * re-filing the print in progress, and without clobbering the
 * `fileUnderAutoTag` mirror of Settings ▸ Library.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

// happy-dom v20 does not expose `localStorage`; templates live there.
installMemoryLocalStorage();

import GenerateView from "./GenerateView.vue";
import InspectorPanel from "../components/create/InspectorPanel.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { newGenerateForm } from "../lib/generateForm";
import { saveGenerationTemplate } from "../lib/generationTemplates";
import { addTag, emptyFileUnderState, pickCollection } from "@studio/lib/fileUnder";
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

const sdxl: ModelEntry = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7.0,
} as ModelEntry;

/** A template saved from a DIFFERENT print — its own name and filing included,
 * because `stripTemplateForm` only strips media. */
function templateFromAnotherPrint() {
  const saved = newGenerateForm();
  saved.model = sdxl.name;
  saved.family = sdxl.family;
  saved.prompt = "a river at dawn";
  saved.steps = 12;
  saved.scheduler = "ddim";
  saved.title = "River studies";
  saved.fileUnderAutoTag = false;
  saved.fileUnder = addTag(emptyFileUnderState(), "dawn");
  saved.fileUnderMatch = { id: "c9", name: "River studies", slug: "river-studies" };
  return saveGenerationTemplate("River preset", saved);
}

async function loadTemplateIntoView() {
  const template = templateFromAnotherPrint();
  const wrapper = mount(GenerateView, { shallow: true, attachTo: document.body });
  await flushPromises();
  // The mount's own `models.fetch()` resolves to the mocked empty list, which
  // would leave the view on its starter cards instead of the workbench.
  useModelStore().all = [sdxl];
  await flushPromises();

  const form = useGenerateFormStore().form;
  form.model = sdxl.name;
  form.family = sdxl.family;
  form.prompt = "a smurf village";
  form.title = "Smurf Village";
  form.fileUnderAutoTag = true;
  form.fileUnder = pickCollection(addTag(emptyFileUnderState(), "blue"), {
    name: "Smurf studies",
  });
  form.fileUnderMatch = { id: "c1", name: "Smurf Village", slug: "smurf-village" };

  // Starting points are a tab in the inspector now, not a floating popover:
  // the panel loads the template and the view answers on `load-template`.
  wrapper.findComponent(InspectorPanel).vm.$emit("load-template", template);
  await flushPromises();
  return form;
}

describe("GenerateView — loading a template keeps the print's identity", () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [sdxl];
  });
  afterEach(() => {
    document.body.innerHTML = "";
    localStorage.clear();
  });

  it("applies the template's parameters", async () => {
    const form = await loadTemplateIntoView();
    expect(form.prompt).toBe("a river at dawn");
    expect(form.steps).toBe(12);
    expect(form.scheduler).toBe("ddim");
  });

  it("does not rename the print in progress", async () => {
    const form = await loadTemplateIntoView();
    expect(form.title).toBe("Smurf Village");
  });

  it("does not re-file the print in progress", async () => {
    const form = await loadTemplateIntoView();
    expect(form.fileUnder.manualTags).toEqual(["blue"]);
    expect(form.fileUnder.picked).toEqual({ name: "Smurf studies" });
    expect(form.fileUnder.pickedExplicitly).toBe(true);
    expect(form.fileUnderMatch).toEqual({
      id: "c1",
      name: "Smurf Village",
      slug: "smurf-village",
    });
  });

  it("does not clobber the Settings ▸ Library auto-tag mirror", async () => {
    const form = await loadTemplateIntoView();
    expect(form.fileUnderAutoTag).toBe(true);
  });
});
