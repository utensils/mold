import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { HostGalleryImage } from "../lib/multiHostGallery";
const fetchGallery = vi.hoisted(() => vi.fn());
vi.mock("../lib/multiHostGallery", async (original) => ({
  ...(await original()),
  fetchMergedGallery: fetchGallery,
}));
import { startLibraryUnreadObserver, useLibraryUnread } from "./libraryUnread";

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  vi.useFakeTimers();
  fetchGallery.mockReset();
  fetchGallery.mockResolvedValue({
    rawEntries: [],
    reachableHostIds: ["origin"],
  });
});
afterEach(() => vi.useRealTimers());

describe("shell inventory observation", () => {
  it("pauses its polling while Library owns listing refresh", async () => {
    const store = useLibraryUnread();
    store.libraryReaders = 1;
    const stop = startLibraryUnreadObserver();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(fetchGallery).not.toHaveBeenCalled();
    store.libraryReaders = 0;
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchGallery).toHaveBeenCalledTimes(1);
    stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchGallery).toHaveBeenCalledTimes(1);
  });

  it("persists a viewed group and leaves another arrival unread on remount", () => {
    const store = useLibraryUnread();
    store.observe([], ["origin"]);
    const a: HostGalleryImage = {
      hostId: "origin",
      hostLabel: "Origin",
      filename: "a.png",
      timestamp: 1,
      metadata: {
        prompt: "fixture",
        model: "flux",
        seed: 1,
        steps: 4,
        guidance: 3.5,
        width: 64,
        height: 64,
        version: "fixture",
      },
    };
    const b = {
      ...a,
      filename: "b.png",
      timestamp: 2,
      metadata: { ...a.metadata, seed: 2 },
    };
    store.observe([a, b], ["origin"]);
    store.view([a]);
    setActivePinia(createPinia());
    const restored = useLibraryUnread();
    expect(restored.ledger.isUnread(["origin|a.png"])).toBe(false);
    expect(restored.ledger.isUnread(["origin|b.png"])).toBe(true);
  });
});
