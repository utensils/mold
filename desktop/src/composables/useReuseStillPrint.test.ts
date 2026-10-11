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
