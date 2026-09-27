/**
 * Qwen Image 2.1's `reference_images.canvas: last-reference`, as the desktop
 * Create view applies it: while the canvas intent is still the model default,
 * the canvas follows the LAST reference's aspect at the recipe's default area,
 * rounded half-to-even onto its 32 px grid exactly like the CLI and the engine.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useHostModelsStore } from "../stores/hostModels";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import type { ModelEntry } from "../lib/api/types";
import { qwenImage21Recipe } from "@studio/lib/generationProfile.testFixtures";

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
vi.mock("../lib/ipc", () => ({ ipc: { sourceStashGet: vi.fn() } }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));

/** A PNG signature + IHDR declaring `width`×`height`, as raw base64. */
function pngHeader(width: number, height: number): string {
  const bytes = [
    0x89,
    0x50,
    0x4e,
    0x47,
    0x0d,
    0x0a,
    0x1a,
    0x0a,
    0x00,
    0x00,
    0x00,
    0x0d,
    0x49,
    0x48,
    0x44,
    0x52,
    (width >>> 24) & 0xff,
    (width >>> 16) & 0xff,
    (width >>> 8) & 0xff,
    width & 0xff,
    (height >>> 24) & 0xff,
    (height >>> 16) & 0xff,
    (height >>> 8) & 0xff,
    height & 0xff,
  ];
  return btoa(String.fromCharCode(...bytes));
}

const qwen21 = {
  name: "qwen-image-2.1:bf16",
  family: "qwen-image21",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 40,
  default_guidance: 1,
  generation_profile: {
    schema_version: 1,
    profile_id: "qwen-image21",
    profile_hash: "test",
    default_recipe_id: "default",
    recipes: [qwenImage21Recipe()],
  },
} as unknown as ModelEntry;

describe("GenerateView last-reference canvas (Qwen Image 2.1)", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useModelStore().all = [qwen21];
    useHostModelsStore().byHost.local = { entries: [qwen21], fetchedAt: Date.now(), error: null };
  });
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("sizes the canvas from the last reference, half-to-even on the grid", async () => {
    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = useGenerateFormStore().form;
    form.model = qwen21.name;
    form.family = "qwen-image21";
    form.width = 1024;
    form.height = 1024;
    await flushPromises();

    form.imageAttachments = [pngHeader(1024, 1024), pngHeader(1600, 900)];
    await flushPromises();
    expect([form.width, form.height]).toEqual([1376, 768]);

    // 4225x4096 lands on an exact half cell: upstream rounds it to even.
    form.imageAttachments = [pngHeader(4225, 4096)];
    await flushPromises();
    expect([form.width, form.height]).toEqual([1024, 1024]);
  });
});
