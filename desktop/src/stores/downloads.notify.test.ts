import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useDownloadsStore } from "./downloads";
import { useToastStore } from "./toasts";
import { notifyPullFailed } from "../lib/notify";
import { emptyDownloadsState } from "../lib/downloads";
import type { DownloadJob } from "../lib/api/types";

vi.mock("../lib/api/catalog", () => ({ startCatalogDownload: vi.fn() }));
vi.mock("../lib/notify", () => ({ notifyPulled: vi.fn(), notifyPullFailed: vi.fn() }));

const failed = (overrides: Partial<DownloadJob> = {}): DownloadJob => ({
  id: "a",
  model: "flux2-klein:q4",
  status: "failed",
  files_done: 0,
  files_total: 1,
  bytes_done: 0,
  bytes_total: 100,
  error: "disk full",
  ...overrides,
});

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(notifyPullFailed).mockReset();
});

describe("pull failure notifications (G11)", () => {
  it("toasts the model, error, and fires a native notification for a primary pull", () => {
    const store = useDownloadsStore();
    store.history = [failed()];
    store.onJobFailed("a");

    const toasts = useToastStore();
    const msg = toasts.items.at(-1);
    expect(msg?.kind).toBe("error");
    expect(msg?.message).toContain("Couldn't pull flux2-klein:q4");
    expect(msg?.message).toContain("disk full");
    expect(notifyPullFailed).toHaveBeenCalledWith("flux2-klein:q4", "disk full");
  });

  it("names the host on a remote pull failure", () => {
    const store = useDownloadsStore();
    store.hostStates["hal9000"] = {
      ...emptyDownloadsState(),
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: null },
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
      history: [failed({ id: "x", model: "sdxl:base" })],
    };
    store.onJobFailed("x", "hal9000");

    const toasts = useToastStore();
    expect(toasts.items.at(-1)?.message).toContain("Couldn't pull sdxl:base on hal9000");
  });
});
