/**
 * Save and Copy image on the Create canvas.
 *
 * Every ordinary desktop generation settles through the durable path, which
 * records the FILE the host saved and leaves `result.image` empty. Gating both
 * actions on those inline bytes therefore hid Save from every print the app
 * makes and left Copy image copying an empty string. What the canvas is
 * showing is the host's file, so that is what both actions act on.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

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
vi.mock("../lib/ipc", () => ({ inTauri: () => false, ipc: { saveMediaBytes: vi.fn() } }));

const saveGalleryMedia = vi.fn(() =>
  Promise.resolve({ filename: "out.png", path: "/tmp/out.png", directory: "Pictures" }),
);
vi.mock("../lib/mediaSave", () => ({
  saveGalleryMedia: (...args: unknown[]) => saveGalleryMedia(...(args as [])),
  showSavedMediaToast: vi.fn(),
}));

/** Stands in for the real helper, but runs the byte reader it is handed — the
 *  point of the case is WHERE the canvas gets the bytes from. */
const copyImageBytesToClipboard = vi.fn(
  async (path: string, deps?: { fetchImage?: (path: string) => Promise<Uint8Array> }) => {
    await deps?.fetchImage?.(path);
  },
);
vi.mock("../lib/clipboard", () => ({
  copyImageBytesToClipboard: (
    path: string,
    deps?: { fetchImage?: (path: string) => Promise<Uint8Array> },
  ) => copyImageBytesToClipboard(path, deps),
  copyBase64ImageToClipboard: vi.fn(() => Promise.resolve()),
}));

import GenerateView from "./GenerateView.vue";
import { newJob } from "../lib/generationJob";
import { useConnectionStore } from "../stores/connection";
import { useContextMenuStore } from "../stores/contextMenu";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import type { GenerateRequest, ModelEntry } from "../lib/api/types";

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
  saveGalleryMedia.mockClear();
  copyImageBytesToClipboard.mockClear();
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = [sdxlModel];
  useHostModelsStore().byHost.local = { entries: [sdxlModel], fetchedAt: Date.now(), error: null };
});
afterEach(() => (document.body.innerHTML = ""));

/** A settled print exactly as the durable path leaves it: a filename on the
 *  host, no inline bytes, and the media loaded as a blob URL. */
function finishDurablePrint() {
  const generation = useGenerationStore();
  const job = newJob({
    prompt: "a brass teapot on a rainy windowsill",
    model: sdxlModel.name,
    width: 1024,
    height: 1024,
    steps: 30,
  } as GenerateRequest);
  Object.assign(job, {
    clientId: 1,
    batchId: 1,
    id: "finished-print",
    status: "complete",
    resultUrl: "blob:print-1",
    result: {
      image: "",
      filename: "mold-sdxl-1788329609977.png",
      model: sdxlModel.name,
      format: "png",
      width: 1024,
      height: 1024,
      seed_used: 4821,
      generation_time_ms: 1000,
    },
  });
  generation.jobs.push(job);
  generation.selectedClientId = job.clientId;
  return job;
}

async function mountView(shallow = true) {
  const wrapper = mount(GenerateView, { shallow, attachTo: document.body });
  await flushPromises();
  return wrapper;
}

describe("GenerateView — saving the print on the canvas", () => {
  it("offers Save for a print the host holds as a file", async () => {
    const wrapper = await mountView(false);
    finishDurablePrint();
    await flushPromises();

    expect(document.querySelector("[data-test='canvas-save']")).not.toBeNull();

    await wrapper.get("[data-test='canvas-save']").trigger("click");
    await flushPromises();

    expect(saveGalleryMedia).toHaveBeenCalledTimes(1);
    const [target, filename, outputFilename] = saveGalleryMedia.mock.calls[0]! as unknown as [
      { baseUrl: string; apiKey: string | null },
      string,
      string,
    ];
    expect(target).toMatchObject({ baseUrl: "http://127.0.0.1:7680" });
    expect(filename).toBe("mold-sdxl-1788329609977.png");
    expect(outputFilename).toContain("s4821");
  });

  it("copies the picture the canvas is showing, not the empty inline field", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve({ arrayBuffer: () => Promise.resolve(new Uint8Array([1, 2, 3]).buffer) }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = await mountView();
    finishDurablePrint();
    await flushPromises();

    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const entry = useContextMenuStore().entries.find(
      (candidate) => !("separator" in candidate) && candidate.label === "Copy image",
    );
    expect(entry).toMatchObject({ disabled: false });
    if (!entry || "separator" in entry) throw new Error("no Copy image entry");
    entry.action?.();
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith("blob:print-1");
    expect(copyImageBytesToClipboard).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });
});
