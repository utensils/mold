/**
 * Reuse settings from the Library: retained source-media authority is the
 * first answer, never the only one. A print the host never pinned (everything
 * made before durable retention, or a role it did not retain) still restores
 * through the local stash and gallery-by-name lookup, exactly as the phone
 * does — the retained path must not switch that fallback off.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { useComposerStore } from "../stores/composer";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
}));
vi.mock("@studio/api/generationPlacement", async (importOriginal) => {
  const original = await importOriginal<typeof import("@studio/api/generationPlacement")>();
  const planned = () =>
    Promise.resolve({
      version: 1,
      authoritative: true,
      state_version: 1,
      plan_version: 1,
      outcome: "planned",
      candidate: {
        device_id: "cuda:0",
        execution_fingerprint: "test",
        predicted_start_after_ms: 0,
        predicted_completion_after_ms: 100,
        setup_ms: 0,
        setup_kind: "warm",
        estimate_confidence: "high",
      },
    });
  return {
    ...original,
    previewGenerationPlacement: planned,
    previewChainPlacement: planned,
  };
});
const { sourceStashGet } = vi.hoisted(() => ({ sourceStashGet: vi.fn() }));
vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
  apiFetchTo: vi.fn(async () => ({ json: async () => ({ instance_id: "fixture" }) })),
}));
vi.mock("@studio/api/gallerySourceMedia", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/gallerySourceMedia")>()),
  relayRetainedSourceMedia: vi.fn(async () => ({ source_image: await sourceStashGet() })),
}));
vi.mock("../lib/ipc", () => ({ ipc: { sourceStashGet } }));
enableAutoUnmount(afterEach);

vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));

const model: ModelEntry = {
  name: "sd15:fp16",
  family: "sd15",
  downloaded: true,
  default_width: 512,
  default_height: 512,
  default_steps: 20,
  default_guidance: 7,
} as ModelEntry;

const metadata = {
  prompt: "a camera",
  model: model.name,
  seed: 42,
  steps: 20,
  guidance: 7,
  width: 512,
  height: 512,
  version: "test",
  source_image_sha256: "a".repeat(64),
  source_image_name: "camera.png",
} as OutputMetadata;
const PNG_1X1 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const origin = { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };

function reuseFromLibrary() {
  const composer = useComposerStore();
  const version = composer.beginRetainedSourceReuse({
    metadata,
    print: {
      filename: "print.png",
      metadata,
      hostId: null,
      hostLabel: "This device",
      target: origin,
      settledAtMs: 1_700_000_000_000,
    },
  });
  return { composer, version };
}

describe("GenerateView Reuse settings media fallback", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    apiJsonTo.mockReset();
    apiJsonTo.mockResolvedValue([model]);
    sourceStashGet.mockReset();
    useModelStore().all = [model];
  });

  it("restores a pre-retention print's source from the local stash", async () => {
    const form = useGenerateFormStore().form;
    sourceStashGet.mockResolvedValue("STASHED");
    const { composer, version } = reuseFromLibrary();
    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    composer.setRetainedSourceIfCurrent(version, {
      filename: "print.png",
      origin,
      inventory: { availability: "unavailable_legacy", members: [] },
      metadata,
    });
    await flushPromises();

    expect(sourceStashGet).toHaveBeenCalledWith("a".repeat(64));
    expect(form.sourceImage).toBe("STASHED");
    expect(form.sourceImageName).toBe("camera.png");
    expect(useToastStore().items.filter((t) => t.kind === "error")).toEqual([]);
  });

  it("falls back for roles the retained archive did not pin", async () => {
    const form = useGenerateFormStore().form;
    sourceStashGet.mockResolvedValue("STASHED");
    const { composer, version } = reuseFromLibrary();
    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    composer.setRetainedSourceIfCurrent(version, {
      filename: "print.png",
      origin,
      inventory: { availability: "available", members: [] },
      metadata,
    });
    await flushPromises();

    expect(form.sourceImage).toBe("STASHED");
  });

  it("restores retained bytes without consulting the fallback", async () => {
    const form = useGenerateFormStore().form;
    sourceStashGet.mockResolvedValue("RETAINED");
    const { composer, version } = reuseFromLibrary();
    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    composer.setRetainedSourceIfCurrent(version, {
      filename: "print.png",
      origin,
      inventory: {
        availability: "available",
        members: [
          { member_id: "source", role: "source_image", display_name: "camera.png", size_bytes: 8 },
        ],
      },
      metadata,
    });
    await flushPromises();

    expect(composer.isRetainedSourceCurrent(version)).toBe(true);
    expect(form.sourceImage).toBe("RETAINED");
    // The relay mock reads the stash once; the fallback would read it again.
    expect(sourceStashGet).toHaveBeenCalledTimes(1);
  });

  // The reported case: an FL2VA clip whose first frame was a gallery still,
  // made before the host pinned source media. Reuse left the bytes-less
  // descriptor behind and Create said "Reattach first frame …" for good.
  it("reattaches an unpinned H3 first frame from the stash", async () => {
    const h3 = {
      ...model,
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      recommended_dimensions: [{ width: 1344, height: 768 }],
    } as ModelEntry;
    apiJsonTo.mockResolvedValue([h3]);
    useModelStore().all = [h3];
    sourceStashGet.mockResolvedValue(PNG_1X1);
    const h3Metadata = {
      prompt: "a giant turtle",
      model: h3.name,
      seed: 7,
      steps: 8,
      guidance: 0,
      width: 1344,
      height: 768,
      frames: 121,
      fps: 24,
      version: "test",
      source_image_sha256: "d".repeat(64),
      source_image_name: "mold-jibmix-flux-fp8-1784397960064.png",
    } as OutputMetadata;
    const composer = useComposerStore();
    const version = composer.beginRetainedSourceReuse({ metadata: h3Metadata });
    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = useGenerateFormStore().form;
    expect(form.h3Authoring?.firstFrame?.data).toBeFalsy();

    composer.setRetainedSourceIfCurrent(version, {
      filename: "turtle.mp4",
      origin,
      inventory: { availability: "unavailable_legacy", members: [] },
      metadata: h3Metadata,
    });
    await flushPromises();

    expect(sourceStashGet).toHaveBeenCalledWith("d".repeat(64));
    expect(form.h3Authoring?.firstFrame?.data).toBe(PNG_1X1);
    expect(form.h3Authoring?.firstFrame?.filename).toBe("mold-jibmix-flux-fp8-1784397960064.png");
  });
});
