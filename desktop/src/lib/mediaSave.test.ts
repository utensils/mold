import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const { revealSavedMedia } = vi.hoisted(() => ({ revealSavedMedia: vi.fn() }));

vi.mock("./ipc", () => ({
  inTauri: () => true,
  ipc: { revealSavedMedia },
}));
vi.mock("./api/client", () => ({ apiFetchTo: vi.fn() }));

import { showSavedMediaToast } from "./mediaSave";
import { useToastStore } from "../stores/toasts";

describe("showSavedMediaToast", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    revealSavedMedia.mockReset();
  });

  it("reports when the saved file can no longer be revealed", async () => {
    revealSavedMedia.mockRejectedValue(new Error("The saved file is no longer on disk."));
    const toasts = useToastStore();
    showSavedMediaToast(toasts, {
      filename: "clip.gif",
      path: "/Volumes/Studio/clip.gif",
      directory: "/Volumes/Studio",
    });

    toasts.runAction(toasts.items[0]!.id);
    await vi.waitFor(() => expect(toasts.items.at(-1)?.kind).toBe("error"));
    expect(toasts.items.at(-1)?.message).toBe("The saved file is no longer on disk.");
  });
});
