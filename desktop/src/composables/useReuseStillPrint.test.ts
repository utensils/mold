import { useAppPrefsStore } from "../stores/appPrefs";
import { applyPrefillToForm, buildRequest, newGenerateForm } from "../lib/generateForm";
import { pendingLegacyVideoSettings } from "@studio/lib/videoLibraryExtensionHandoff";
const legacyApi = vi.hoisted(() => ({ inspect: vi.fn(), capture: vi.fn() }));
vi.mock("@studio/api/legacyVideoSettings", async (original) => ({
  ...(await original<typeof import("@studio/api/legacyVideoSettings")>()),
  inspectLegacyVideoSettings: legacyApi.inspect,
}));
vi.mock("@studio/api/chainJobs", async (original) => ({
  ...(await original<typeof import("@studio/api/chainJobs")>()),
  captureChainOwner: legacyApi.capture,
}));
import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { useReuseStillPrint } from "./useReuseStillPrint";
import { useComposerStore } from "../stores/composer";
import { useGalleryStore, type MergedPrint } from "../stores/gallery";

const { inventory } = vi.hoisted(() => ({ inventory: vi.fn() }));
vi.mock("@studio/api/gallerySourceMedia", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/gallerySourceMedia")>()),
  retainedSourceMediaInventory: inventory,
}));

beforeEach(() => {
  setActivePinia(createPinia());
  inventory.mockReset();
  legacyApi.inspect.mockReset();
  legacyApi.capture.mockReset();
  pendingLegacyVideoSettings.value = null;
});

// Every copy on a machine with no resolvable route: nobody can be asked, so
// Create must be told rather than left waiting on "Restoring the original
// input files…" for an inventory that never comes.
it("declines retained authority when no copy of the print can be asked", async () => {
  const gallery = useGalleryStore();
  vi.spyOn(gallery, "targetOfOrNull").mockReturnValue(null);
  vi.spyOn(gallery, "locationsOf").mockReturnValue([]);
  const composer = useComposerStore();
  const entry = {
    sourceKey: "offline-host",
    hostLabel: "offline",
    item: {
      filename: "print.png",
      timestamp: 1,
      metadata: { prompt: "p", model: "m", seed: 1, source_image_sha256: "a".repeat(64) },
    },
  } as unknown as MergedPrint;

  useReuseStillPrint()(entry);
  await flushPromises();

  expect(inventory).not.toHaveBeenCalled();
  expect(composer.retainedSourceDeclinedVersion).toBe(composer.retainedSourceVersion);
});

it("keeps the current draft when exact legacy archive restoration refuses", async () => {
  const gallery = useGalleryStore();
  vi.spyOn(gallery, "targetOfOrNull").mockReturnValue({ baseUrl: "http://plato", apiKey: null });
  const composer = useComposerStore();
  const before = composer.retainedSourceVersion;
  legacyApi.capture.mockResolvedValue({
    hostId: "plato",
    instanceId: "captured",
    target: { baseUrl: "http://plato" },
  });
  legacyApi.inspect.mockRejectedValue(new Error("Missing exact inputs"));
  const entry = {
    sourceKey: "plato",
    hostLabel: "Plato",
    item: { filename: "clip.mp4", metadata: { model: "m", chain: { stages: [] } } },
  } as unknown as MergedPrint;
  expect(await useReuseStillPrint()(entry)).toBe(false);
  expect(composer.retainedSourceVersion).toBe(before);
  expect(pendingLegacyVideoSettings.value).toBeNull();
});

it("restores automatic legacy conditioning into the normal request without a second generic inventory", async () => {
  const gallery = useGalleryStore();
  const target = { baseUrl: "http://plato", apiKey: null };
  vi.spyOn(gallery, "targetOfOrNull").mockReturnValue(target);
  vi.spyOn(useAppPrefsStore(), "update").mockResolvedValue({} as never);
  const owner = { hostId: "plato", label: "Plato", instanceId: "captured", target };
  legacyApi.capture.mockResolvedValue(owner);
  legacyApi.inspect.mockResolvedValue({
    owner,
    notice: "saved length",
    automaticDuration: { durationMs: "8000", scenesEnabled: false, scenes: [] },
    request: {
      model: "minimax-h3-fl2va:official-bf16",
      width: 768,
      height: 768,
      steps: 9,
      guidance: 0,
      fps: 24,
      seed: "18446744073709551615",
      stages: [{ prompt: " literal\\n ", frames: 107, source_image: "YQ==", seed_offset: "0" }],
    },
  });
  const entry = {
    sourceKey: "plato",
    hostLabel: "Plato",
    item: {
      filename: "clip.mp4",
      timestamp: 1,
      metadata: { model: "minimax-h3-fl2va:official-bf16", prompt: "summary", seed: 1, chain: {} },
    },
  } as unknown as MergedPrint;
  expect(await useReuseStillPrint()(entry)).toBe(true);
  const prefill = useComposerStore().prefill!;
  const form = newGenerateForm();
  applyPrefillToForm(form, prefill, []);
  expect(buildRequest(form)).toMatchObject({
    prompt: " literal\\n ",
    seed: "18446744073709551615",
    source_image: "YQ==",
  });
  expect(inventory).not.toHaveBeenCalled();
});
