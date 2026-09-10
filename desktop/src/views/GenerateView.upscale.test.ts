/**
 * "Make bigger" follows the media file on the Create canvas. Durable video
 * completions carry an encoded MP4 filename but may omit the old
 * `video_frames` marker, so the filename/format must select the durable
 * Framewise workflow and retain the machine that made the print.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises, mount } from "@vue/test-utils";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

const {
  createFramewiseUpscale,
  findRecoverableFramewiseUpscale,
  getFramewiseUpscale,
  transitionFramewiseUpscale,
  upscaleLibraryImage,
} = vi.hoisted(() => ({
  createFramewiseUpscale: vi.fn(),
  findRecoverableFramewiseUpscale: vi.fn(),
  getFramewiseUpscale: vi.fn(),
  transitionFramewiseUpscale: vi.fn(),
  upscaleLibraryImage: vi.fn(),
}));

vi.mock("@studio/api/videoUpscale", () => ({
  createFramewiseUpscale,
  findRecoverableFramewiseUpscale,
  getFramewiseUpscale,
  transitionFramewiseUpscale,
  upscaleLibraryImage,
}));

const upscaleImage = vi.hoisted(() => vi.fn());
vi.mock("../lib/api/upscale", () => ({ upscaleImage }));
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
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { saveOutputBytes: vi.fn(() => Promise.resolve("saved.png")) },
}));

import GenerateView from "./GenerateView.vue";
import { newJob } from "../lib/generationJob";
import { useConnectionStore } from "../stores/connection";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import type { GenerateRequest, ModelEntry } from "../lib/api/types";

const target = { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
const imageModel = {
  name: "sdxl-base:fp16",
  family: "sdxl",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 30,
  default_guidance: 7,
} as ModelEntry;
const upscaler = {
  name: "real-esrgan-x4plus:fp16",
  family: "real-esrgan",
  downloaded: true,
} as ModelEntry;

const queuedJob = {
  contract_version: 1,
  id: "vup-canvas-1",
  state: "queued",
  source: { kind: "library", filename: "teapot-motion.mp4" },
  model: upscaler.name,
  completed_frames: 0,
  total_frames: 24,
  disclosure: "Framewise upscale",
};
const completedJob = {
  ...queuedJob,
  state: "completed",
  completed_frames: 24,
  output_filename: "teapot-motion-framewise-upscaled.mp4",
};

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

let wrapper: ReturnType<typeof mount> | null = null;

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  localStorage.clear();
  findRecoverableFramewiseUpscale.mockResolvedValue(null);
  createFramewiseUpscale.mockResolvedValue(queuedJob);
  getFramewiseUpscale.mockResolvedValue(completedJob);
  transitionFramewiseUpscale.mockResolvedValue(queuedJob);
  upscaleLibraryImage.mockResolvedValue({
    filename: "teapot-upscaled.png",
    model: upscaler.name,
    scale_factor: 4,
  });

  const connection = useConnectionStore();
  connection.info = { mode: "local", ...target };
  connection.status = "ready";
  const hosts = useHostsStore();
  hosts.initialized = true;
  hosts.capabilities.local = {
    video_upscale: { available: true, gallery_image: true },
  } as never;
  useModelStore().all = [imageModel, upscaler];
  useHostModelsStore().byHost.local = {
    entries: [imageModel, upscaler],
    fetchedAt: Date.now(),
    error: null,
  };
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
});

function finishPrint(
  filename: string,
  format: "png" | "mp4",
  origin: { hostId: string | null; hostLabel: string | null } = {
    hostId: null,
    hostLabel: null,
  },
) {
  const generation = useGenerationStore();
  const job = newJob({
    prompt: "a brass teapot in motion",
    model: imageModel.name,
    width: 1024,
    height: 1024,
    steps: 30,
  } as GenerateRequest);
  Object.assign(job, {
    clientId: 1,
    batchId: 1,
    id: "finished-print",
    status: "complete",
    ...origin,
    resultUrl: format === "mp4" ? "blob:encoded-video" : "blob:encoded-image",
    result: {
      image: "ZW5jb2RlZC1tZWRpYQ==",
      filename,
      model: imageModel.name,
      format,
      width: 1024,
      height: 1024,
      seed_used: 4821,
      generation_time_ms: 1000,
      // Deliberately no video_frames: durable encoded clips do not need the
      // raw-frame fallback marker.
    },
  });
  generation.jobs.push(job);
  generation.selectedClientId = job.clientId;
  return job;
}

async function mountView() {
  wrapper = mount(GenerateView, { shallow: false, attachTo: document.body });
  await flushPromises();
  return wrapper;
}

describe("GenerateView — Make bigger", () => {
  it("routes an encoded MP4 without video_frames through Framewise upscale on its source host", async () => {
    const view = await mountView();
    finishPrint("teapot-motion.mp4", "mp4");
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();

    expect(document.querySelector("[aria-label='Framewise upscale video']")).not.toBeNull();
    expect(findRecoverableFramewiseUpscale).toHaveBeenCalledWith(target, "teapot-motion.mp4");
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(createFramewiseUpscale).toHaveBeenCalledWith(target, "teapot-motion.mp4", upscaler.name);
    expect(getFramewiseUpscale).toHaveBeenCalledWith(target, "vup-canvas-1");
    expect(upscaleLibraryImage).not.toHaveBeenCalled();
    expect(upscaleImage).not.toHaveBeenCalled();
  });

  it("retains a remote print's URL and API key for recovery and submission", async () => {
    const remoteTarget = { baseUrl: "http://plato:7680", apiKey: "plato-secret" };
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: remoteTarget.baseUrl,
      apiKey: remoteTarget.apiKey,
      status: "ready",
      error: null,
      instanceId: "plato-instance",
    });
    hosts.capabilities["plato-7680"] = {
      video_upscale: { available: true, gallery_image: true },
    } as never;
    const view = await mountView();
    finishPrint("remote-motion.mp4", "mp4", {
      hostId: "plato-7680",
      hostLabel: "plato",
    });
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    expect(findRecoverableFramewiseUpscale).toHaveBeenCalledWith(remoteTarget, "remote-motion.mp4");
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(createFramewiseUpscale).toHaveBeenCalledWith(
      remoteTarget,
      "remote-motion.mp4",
      upscaler.name,
    );
    expect(upscaleLibraryImage).not.toHaveBeenCalled();
  });

  it("keeps the opened remote authority frozen across create, poll, and resume", async () => {
    const originalTarget = { baseUrl: "http://plato:7680", apiKey: "original-secret" };
    const hosts = useHostsStore();
    const remote = {
      id: "plato-7680",
      label: "plato",
      url: originalTarget.baseUrl,
      apiKey: originalTarget.apiKey,
      status: "ready" as const,
      error: null,
      instanceId: "plato-instance",
    };
    hosts.extras.push(remote);
    hosts.capabilities[remote.id] = {
      video_upscale: { available: true, gallery_image: true },
    } as never;
    const pausedAfterPoll = {
      ...queuedJob,
      state: "paused",
      completed_frames: 5,
    };
    getFramewiseUpscale.mockResolvedValueOnce(pausedAfterPoll).mockResolvedValueOnce(completedJob);
    transitionFramewiseUpscale.mockResolvedValueOnce(queuedJob);
    const view = await mountView();
    finishPrint("authority-motion.mp4", "mp4", {
      hostId: remote.id,
      hostLabel: remote.label,
    });
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    expect(findRecoverableFramewiseUpscale).toHaveBeenCalledWith(
      originalTarget,
      "authority-motion.mp4",
    );

    remote.url = "http://replacement:7680";
    remote.apiKey = "replacement-secret";
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(createFramewiseUpscale).toHaveBeenCalledWith(
      originalTarget,
      "authority-motion.mp4",
      upscaler.name,
    );
    expect(getFramewiseUpscale).toHaveBeenNthCalledWith(1, originalTarget, "vup-canvas-1");
    const resume = [...document.querySelectorAll("button")].find(
      (button) => button.textContent?.trim() === "Resume",
    ) as HTMLButtonElement;
    expect(resume).toBeDefined();
    resume.click();
    await flushPromises();

    expect(transitionFramewiseUpscale).toHaveBeenCalledWith(
      originalTarget,
      "vup-canvas-1",
      "resume",
    );
    expect(getFramewiseUpscale).toHaveBeenNthCalledWith(2, originalTarget, "vup-canvas-1");
  });

  it("hides Make bigger when the saved host no longer matches the generation authority", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://replacement:7680",
      apiKey: "replacement-secret",
      status: "ready",
      error: null,
      instanceId: "replacement-instance",
    });
    hosts.capabilities["plato-7680"] = {
      video_upscale: { available: true, gallery_image: true },
    } as never;
    const generation = useGenerationStore();
    vi.spyOn(generation, "targetForJob").mockReturnValue({
      baseUrl: "http://plato:7680",
      apiKey: "original-secret",
    });
    await mountView();
    finishPrint("mismatched-authority.mp4", "mp4", {
      hostId: "plato-7680",
      hostLabel: "plato",
    });
    await flushPromises();

    expect(document.querySelector("[data-test='canvas-upscale']")).toBeNull();
    expect(findRecoverableFramewiseUpscale).not.toHaveBeenCalled();
  });

  it("hides Make bigger for a video when its source host reports Framewise upscale unavailable", async () => {
    useHostsStore().capabilities.local = {
      video_upscale: { available: false, gallery_image: true },
    } as never;
    await mountView();
    finishPrint("unavailable.mp4", "mp4");
    await flushPromises();

    expect(document.querySelector("[data-test='canvas-upscale']")).toBeNull();
  });

  it("keeps a PNG on the gallery image upscale endpoint", async () => {
    const view = await mountView();
    finishPrint("teapot.png", "png");
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    expect(document.querySelector("[aria-label='Upscale image']")).not.toBeNull();
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(upscaleLibraryImage).toHaveBeenCalledWith(target, "teapot.png", upscaler.name);
    expect(createFramewiseUpscale).not.toHaveBeenCalled();
  });

  it("recovers a paused Framewise job and resumes it on the source host", async () => {
    const pausedJob = {
      ...queuedJob,
      id: "vup-paused",
      state: "paused",
      completed_frames: 7,
      model: "real-esrgan-x2plus:fp16",
    };
    findRecoverableFramewiseUpscale.mockResolvedValueOnce(pausedJob);
    transitionFramewiseUpscale.mockResolvedValueOnce({ ...pausedJob, state: "queued" });
    const view = await mountView();
    finishPrint("teapot-motion.mp4", "mp4");
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    const resume = [...document.querySelectorAll("button")].find(
      (button) => button.textContent?.trim() === "Resume",
    ) as HTMLButtonElement;
    expect(resume).toBeDefined();
    resume.click();
    await flushPromises();

    expect(transitionFramewiseUpscale).toHaveBeenCalledWith(target, "vup-paused", "resume");
    expect(getFramewiseUpscale).toHaveBeenCalledWith(target, "vup-paused");
    expect(createFramewiseUpscale).not.toHaveBeenCalled();
  });

  it("ignores an obsolete recovery after the dialog is closed and reopened", async () => {
    const staleRecovery = deferred<typeof queuedJob>();
    findRecoverableFramewiseUpscale
      .mockReturnValueOnce(staleRecovery.promise)
      .mockResolvedValueOnce(null);
    const view = await mountView();
    finishPrint("teapot-motion.mp4", "mp4");
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    (document.querySelector("[aria-label='Close']") as HTMLButtonElement).click();
    await flushPromises();
    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    expect(document.querySelector("[data-test='start-upscale']")).not.toBeNull();

    staleRecovery.resolve({ ...queuedJob, id: "obsolete", state: "paused" });
    await flushPromises();

    expect(document.querySelector("[data-test='start-upscale']")).not.toBeNull();
    expect(
      [...document.querySelectorAll("button")].some(
        (button) => button.textContent?.trim() === "Resume",
      ),
    ).toBe(false);
    expect(getFramewiseUpscale).not.toHaveBeenCalledWith(target, "obsolete");
  });

  it("keeps recovery busy and admits only one create after it settles", async () => {
    const recovery = deferred<null>();
    findRecoverableFramewiseUpscale.mockReturnValueOnce(recovery.promise);
    const view = await mountView();
    finishPrint("teapot-motion.mp4", "mp4");
    await flushPromises();

    await view.get("[data-test='canvas-upscale']").trigger("click");
    await flushPromises();
    const start = document.querySelector("[data-test='start-upscale']") as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    start.click();
    start.click();
    expect(createFramewiseUpscale).not.toHaveBeenCalled();

    recovery.resolve(null);
    await flushPromises();
    expect(start.disabled).toBe(false);
    start.click();
    start.click();
    await flushPromises();

    expect(createFramewiseUpscale).toHaveBeenCalledTimes(1);
  });
});
