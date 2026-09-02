/**
 * Expand and Remix on a recipe that IGNORES the prompt.
 *
 * Hunyuan3D has no text encoder anywhere in the family, so a rewritten prompt
 * changes nothing about the render and the host answers such a transform with
 * exactly ONE result — the guide's image-preparation advice — instead of the
 * requested variations. Every entry point on this view answers with the same
 * sentence: the composer controls render disabled with it, the programmatic
 * paths (⌘E from the composer, Menu ▸ Expand Prompt, Remix) toast it without
 * sending a request, and the missing-expander PULL offer never appears for a
 * request that is never made.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    saveMediaBytes: vi.fn(),
    revealSavedMedia: vi.fn(),
  },
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));
const expandPrompt = vi.fn();
const remixPrompt = vi.fn();
vi.mock("../lib/api/expand", () => ({ expandPrompt: (...a: unknown[]) => expandPrompt(...a) }));
vi.mock("../lib/api/remix", () => ({ remixPrompt: (...a: unknown[]) => remixPrompt(...a) }));

import GenerateView from "./GenerateView.vue";
import ComposerCard from "../components/create/ComposerCard.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useHostModelsStore } from "../stores/hostModels";
import { useGenerateFormStore } from "../stores/generateForm";
import { useToastStore } from "../stores/toasts";
import { useUiStore } from "../stores/ui";
import { applyModelDefaults } from "../lib/generateForm";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import { PROMPT_IGNORED_TRANSFORM_REASON } from "@studio/lib/promptTransform";
import { IGNORED_PROMPT_GUIDANCE } from "@studio/lib/promptRequirement";
import type { ModelEntry } from "../lib/api/types";

enableAutoUnmount(afterEach);

const meshModel = {
  name: "hunyuan3d-mini-turbo:fp16",
  family: "hunyuan3d",
  downloaded: true,
  default_steps: 5,
  default_guidance: 5,
  source_image: "required",
  generation_profile: {
    schema_version: 1,
    profile_id: "hunyuan3d.mini",
    profile_hash: "hash",
    default_recipe_id: "default",
    recipes: [hunyuan3dRecipe()],
  },
} as unknown as ModelEntry;

const rasterModel = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_steps: 30,
  default_guidance: 7,
  generation_profile: {
    schema_version: 1,
    profile_id: "sdxl.base",
    profile_hash: "hash",
    default_recipe_id: "default",
    recipes: [sdxlRecipe()],
  },
} as unknown as ModelEntry;

function primeHost(entries: ModelEntry[]) {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = entries;
  useHostModelsStore().byHost.local = { entries, fetchedAt: Date.now(), error: null };
}

/** The composer and its transform controls render for real; everything else
 *  on this very large view stays stubbed. */
function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: {
      stubs: {
        SequenceComposer: true,
        ComposerCard: false,
        ExpandControl: false,
        StyleChips: true,
        EstimateBadge: true,
        Icon: true,
        Keycap: true,
        ActionBlocker: true,
      },
    },
  });
}

function selectModel(entry: ModelEntry, prompt: string) {
  const form = useGenerateFormStore().form;
  form.model = entry.name;
  form.family = entry.family;
  applyModelDefaults(form, entry);
  form.sourceImage = "c291cmNl";
  form.prompt = prompt;
  return form;
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiJson.mockReset().mockImplementation(() => Promise.resolve([]));
  apiJsonTo.mockReset().mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    return Promise.resolve([]);
  });
  expandPrompt.mockReset().mockResolvedValue({ expanded: ["rewritten"] });
  remixPrompt.mockReset().mockResolvedValue({
    variants: [{ prompt: "rewritten", dimensions: [] }],
    source_prompt: "a chair",
    source_kind: "direct",
  });
  window.localStorage?.clear?.();
});
afterEach(() => {
  document.body.innerHTML = "";
});

describe("GenerateView — a recipe that ignores the prompt", () => {
  it("disables Expand and Remix with the reason on the tooltip and a visible hint", async () => {
    primeHost([meshModel]);
    selectModel(meshModel, "a chair");
    const wrapper = mountView();
    await flushPromises();

    const expand = wrapper.get('[data-test="expand-action"]');
    const remix = wrapper.get('[data-test="remix-action"]');
    expect(expand.attributes("disabled")).toBeDefined();
    expect(remix.attributes("disabled")).toBeDefined();
    expect(expand.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(remix.attributes("title")).toBe(PROMPT_IGNORED_TRANSFORM_REASON);
    expect(wrapper.get('[data-test="transform-blocked-hint"]').text()).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("answers the expand intent with the reason instead of a request", async () => {
    primeHost([meshModel]);
    selectModel(meshModel, "a chair");
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ComposerCard).vm.$emit("expand");
    await flushPromises();
    expect(expandPrompt).not.toHaveBeenCalled();
    expect(useToastStore().items.map((toast) => toast.message)).toContain(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("answers the remix intent with the reason instead of a request", async () => {
    primeHost([meshModel]);
    selectModel(meshModel, "a chair");
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ComposerCard).vm.$emit("remix");
    await flushPromises();
    expect(remixPrompt).not.toHaveBeenCalled();
    expect(useToastStore().items.map((toast) => toast.message)).toContain(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("refuses Menu ▸ Expand Prompt with the same sentence", async () => {
    primeHost([meshModel]);
    selectModel(meshModel, "a chair");
    mountView();
    await flushPromises();

    useUiStore().expandTick++;
    await flushPromises();
    expect(expandPrompt).not.toHaveBeenCalled();
    expect(useToastStore().items.map((toast) => toast.message)).toContain(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
  });

  it("never offers to pull the expander for a transform it will not run", async () => {
    primeHost([meshModel]);
    const hosts = useHostsStore();
    hosts.capabilities.local = {
      expand: { configured: true, model_present: false, model: "qwen3:4b" },
    } as unknown as (typeof hosts.capabilities)[string];
    selectModel(meshModel, "a chair");
    const wrapper = mountView();
    await flushPromises();

    wrapper.findComponent(ComposerCard).vm.$emit("expand");
    await flushPromises();
    expect(expandPrompt).not.toHaveBeenCalled();
    expect(wrapper.find('[data-test="prepared-expansion-batch"]').exists()).toBe(false);
    expect(wrapper.html()).not.toContain("isn't installed on");
  });

  it("leaves both transforms working for a recipe that reads the prompt", async () => {
    primeHost([rasterModel]);
    selectModel(rasterModel, "a chair");
    const wrapper = mountView();
    await flushPromises();

    const expand = wrapper.get('[data-test="expand-action"]');
    expect(expand.attributes("disabled")).toBeUndefined();
    expect(wrapper.find('[data-test="transform-blocked-hint"]').exists()).toBe(false);
    wrapper.findComponent(ComposerCard).vm.$emit("expand");
    await flushPromises();
    expect(expandPrompt).toHaveBeenCalledTimes(1);
  });

  it("explains the source image on the empty canvas instead of the optional-prompt wording", async () => {
    primeHost([meshModel]);
    selectModel(meshModel, "");
    const wrapper = mountView();
    await flushPromises();
    const empty = wrapper.findComponent({ name: "EmptyStateBlock" });
    expect(empty.exists()).toBe(true);
    expect(empty.props("guidance")).toBe(IGNORED_PROMPT_GUIDANCE);
    expect(String(empty.props("guidance"))).not.toContain("animates");
  });
});
