import { flushPromises, mount, type DOMWrapper, type VueWrapper } from "@vue/test-utils";
import { createPinia, type Pinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { GalleryImage, ModelEntry, ServerStatus } from "../lib/api/types";
import type { GenerateForm } from "../lib/generateForm";

const {
  invoke,
  apiFetchTo,
  apiJsonTo,
  sseStream,
  streamableMediaUrl,
  evictMedia,
  applySourceFitPreprocess,
  expandPrompt,
  startCatalogDownload,
} = vi.hoisted(() => ({
  invoke: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn(),
  streamableMediaUrl: vi.fn(),
  evictMedia: vi.fn(),
  applySourceFitPreprocess: vi.fn(),
  expandPrompt: vi.fn(),
  startCatalogDownload: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));
vi.mock("../lib/sourceFitPreprocess", () => ({ applySourceFitPreprocess }));
vi.mock("../lib/api/expand", () => ({ expandPrompt }));
vi.mock("../lib/api/catalog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/catalog")>()),
  startCatalogDownload,
}));
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl,
  evictMedia,
}));

import MobileApp from "./MobileApp.vue";
import MobileLoraControls from "./MobileLoraControls.vue";
import MobileTemplates from "./MobileTemplates.vue";
import { useMobileDownloadsStore } from "./mobileDownloads";

installMemoryLocalStorage();

const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: "secret" };
const status: ServerStatus = {
  version: "0.18.0",
  models_loaded: [],
  uptime_secs: 60,
  hostname: "studio",
  instance_id: "studio-id",
};
const model: ModelEntry = {
  name: "ltx2:q8",
  family: "ltx2",
  size_gb: 20,
  is_loaded: false,
  hf_repo: "example/ltx2",
  default_steps: 30,
  default_guidance: 3,
  default_width: 768,
  default_height: 512,
  description: "Video model",
  downloaded: true,
};
const print: GalleryImage = {
  filename: "storm clip.mp4",
  timestamp: 1_700_000_000,
  format: "mp4",
  metadata: {
    prompt: "a ship crossing violet lightning",
    negative_prompt: "calm water",
    model: model.name,
    seed: 77,
    steps: 28,
    guidance: 4.25,
    width: 1536,
    height: 1024,
    generation_width: 768,
    generation_height: 512,
    output_format: "mp4",
    scheduler: "ddim",
    frames: 121,
    fps: 30,
  },
};

let wrapper: VueWrapper | null = null;
let objectUrlSequence = 0;
const openStreams: Array<{
  path: string;
  options: {
    body: Record<string, unknown>;
    headers?: Record<string, string>;
    signal: AbortSignal;
    onOpen?: () => void;
    onClose?: (error: Error | null) => void;
    onEvent: (event: string, data: string) => void;
    target?: { baseUrl: string; apiKey: string | null };
  };
  resolve: () => void;
}> = [];

function mountMobileApp(pinia: Pinia = createPinia()): VueWrapper {
  return mount(MobileApp, {
    attachTo: document.body,
    global: { plugins: [pinia] },
  });
}

function fieldControl(label: string): DOMWrapper<Element> {
  const field = wrapper
    ?.findAll("label.field")
    .find((candidate) => candidate.find("span").text() === label);
  if (!field) throw new Error(`Missing ${label} field`);
  return field.find("input, textarea, select");
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(
    "mold.mobile.hosts.v1",
    JSON.stringify([
      {
        id: "studio-id",
        name: "Studio",
        baseUrl: target.baseUrl,
        hostname: "studio",
        version: "0.18.0",
        online: false,
      },
    ]),
  );
  invoke
    .mockReset()
    .mockImplementation((command: string) =>
      Promise.resolve(command === "keychain_get_api_key" ? target.apiKey : null),
    );
  apiJsonTo.mockReset().mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/status") return Promise.resolve(status);
    if (path === "/api/models") return Promise.resolve([model]);
    if (path === "/api/gallery") return Promise.resolve([print]);
    return Promise.reject(new Error(`Unexpected API path: ${path}`));
  });
  apiFetchTo.mockReset().mockResolvedValue({
    blob: () => Promise.resolve(new Blob(["thumbnail"])),
  } as Response);
  openStreams.length = 0;
  sseStream.mockReset().mockImplementation(
    (
      path: string,
      options: {
        body: Record<string, unknown>;
        headers?: Record<string, string>;
        signal: AbortSignal;
        onOpen?: () => void;
        onClose?: (error: Error | null) => void;
        onEvent: (event: string, data: string) => void;
      },
    ) =>
      new Promise<void>((resolve) => {
        openStreams.push({ path, options, resolve });
      }),
  );
  streamableMediaUrl.mockReset().mockResolvedValue("https://studio/media/full-video");
  evictMedia.mockReset();
  applySourceFitPreprocess.mockReset().mockImplementation((input) =>
    Promise.resolve({
      source: input.source,
      mask: input.mask,
      changed: false,
    }),
  );
  expandPrompt.mockReset().mockImplementation((prompt: string, options: { variations?: number }) =>
    Promise.resolve({
      expanded: Array.from(
        { length: options.variations ?? 1 },
        (_, index) => `${prompt} · prepared ${index + 1}`,
      ),
    }),
  );
  startCatalogDownload.mockReset().mockResolvedValue("expansion-job");
  objectUrlSequence = 0;
  URL.createObjectURL = vi.fn(() => `blob:thumbnail-${++objectUrlSequence}`);
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.themeFamily;
});

describe("MobileApp generation queue", () => {
  async function submitPrompt(prompt: string): Promise<void> {
    await fieldControl("Prompt").setValue(prompt);
    await wrapper?.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  it("shows a useful retry state when the selected host cannot load models", async () => {
    apiJsonTo.mockRejectedValue(new Error("Connection refused"));

    wrapper = mountMobileApp();
    await flushPromises();

    expect(fieldControl("Model").attributes("disabled")).toBeDefined();
    expect(fieldControl("Model").text()).toContain("No generation models available");
    expect(wrapper.get("[data-test='mobile-model-error']").text()).toContain(
      "Couldn’t load generation models",
    );
    expect(wrapper.get("[data-test='mobile-model-retry']").text()).toBe("Retry");
    expect(
      wrapper.findAll("label.field").some((field) => field.text().includes("Negative prompt")),
    ).toBe(false);
    expect(wrapper.find("[data-test='mobile-source-disclosure']").exists()).toBe(false);
  });

  it("applies model defaults to the resolution picker and snapshots those dimensions", async () => {
    const imageModel: ModelEntry = {
      ...model,
      name: "flux:image",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
      default_steps: 24,
      default_guidance: 3.5,
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel, model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    expect(wrapper.get("[data-orientation='square']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-aspect='1:1']").attributes("aria-pressed")).toBe("true");
    expect(
      (wrapper.get("[data-test='mobile-resolution-tier']").element as HTMLSelectElement).value,
    ).toBe("1:1 · 1024×1024");

    await fieldControl("Model").setValue(model.name);
    await flushPromises();
    expect(wrapper.get("[data-orientation='landscape']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-aspect='3:2']").attributes("aria-pressed")).toBe("true");
    expect(
      (wrapper.get("[data-test='mobile-resolution-tier']").element as HTMLSelectElement).value,
    ).toBe("3:2 · 768×512");

    await submitPrompt("use the video defaults");
    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.options.body).toMatchObject({
      model: model.name,
      width: 768,
      height: 512,
    });
  });

  it("snapshots form and host before asynchronous source preprocessing", async () => {
    const studioModel: ModelEntry = {
      ...model,
      name: "flux:studio",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    const renderModel: ModelEntry = { ...studioModel, name: "flux:render" };
    const renderTarget = { baseUrl: "http://render.tailnet.ts.net:7680", apiKey: "secret" };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "render-id",
          name: "Render",
          baseUrl: renderTarget.baseUrl,
          hostname: "render",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((apiTarget: { baseUrl: string }, path: string) => {
      const remote = apiTarget.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: remote ? "render" : "studio",
          instance_id: remote ? "render-id" : "studio-id",
        });
      }
      if (path === "/api/models") return Promise.resolve([remote ? renderModel : studioModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path.startsWith("/api/catalog/installed")) return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementationOnce(
      (input: { source: string | null; mask: string | null }) =>
        new Promise((resolve) => {
          finishPreprocess = () =>
            resolve({ source: `fitted:${input.source}`, mask: input.mask, changed: true });
        }),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    const sourceInput = wrapper.get("[data-test='mobile-source-input']");
    Object.defineProperty(sourceInput.element, "files", {
      configurable: true,
      value: [new File(["source"], "source.png", { type: "image/png" })],
    });
    await sourceInput.trigger("change");
    await flushPromises();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await flushPromises();
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-source-preview']").exists()).toBe(true),
    );
    await fieldControl("Prompt").setValue("first prompt");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    await fieldControl("Prompt").setValue("next prompt");
    await wrapper.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();
    expect(fieldControl("Model").element).toHaveProperty("value", renderModel.name);

    finishPreprocess();
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.options.target).toEqual(target);
    expect(openStreams[0]?.options.body).toMatchObject({
      prompt: "first prompt",
      model: studioModel.name,
    });
    expect(openStreams[0]?.options.body.source_image).toMatch(/^fitted:/);
  });

  it("rechecks the iPhone media budget after source preprocessing", async () => {
    const oversizedBase64 = {
      length: 61 * 1024 * 1024,
      endsWith: () => false,
    } as unknown as string;
    applySourceFitPreprocess.mockResolvedValueOnce({
      source: oversizedBase64,
      mask: null,
      changed: true,
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("small source");
    liveForm.sourceImageName = "small.jpg";
    await fieldControl("Prompt").setValue("fit this source safely");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Combined generation media must be 45 MiB or smaller on iPhone",
    );
  });

  it("clears host-local model artifacts and syncs same-name model capabilities", async () => {
    const remoteTarget = { baseUrl: "http://render.tailnet.ts.net:7680", apiKey: "secret" };
    const remoteModel: ModelEntry = {
      ...model,
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        { id: "studio-id", name: "Studio", baseUrl: target.baseUrl, online: false },
        { id: "render-id", name: "Render", baseUrl: remoteTarget.baseUrl, online: false },
      ]),
    );
    apiJsonTo.mockImplementation((apiTarget: { baseUrl: string }, path: string) => {
      const remote = apiTarget.baseUrl === remoteTarget.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: remote ? "render" : "studio",
          instance_id: remote ? "render-id" : "studio-id",
        });
      }
      if (path === "/api/models") return Promise.resolve([remote ? remoteModel : model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path.startsWith("/api/catalog/installed")) return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.loras = [
      { path: "/studio/style.safetensors", name: "Style", scale: 1, trainedWords: [] },
    ];
    liveForm.upscaleModel = "studio-upscaler";
    liveForm.controlModel = "/studio/control.safetensors";
    liveForm.cameraControl = "/studio/camera.safetensors";
    liveForm.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: "studio-source-upscaler",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    };
    const studioTemplateForm = JSON.parse(JSON.stringify(liveForm)) as GenerateForm;

    await wrapper.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();

    expect(liveForm.family).toBe("flux");
    expect(liveForm.loras).toEqual([]);
    expect(liveForm.upscaleModel).toBe("");
    expect(liveForm.controlModel).toBe("");
    expect(liveForm.cameraControl).toBeNull();
    expect(liveForm.sourceFit).toMatchObject({ mode: "upscale-then-fit", upscalerModel: "" });

    // Templates remain portable between hosts, but loading one from its
    // origin host must not resurrect paths that only exist there. A same-name
    // model on the active host also owns the capability family.
    wrapper.getComponent(MobileTemplates).vm.$emit("load", {
      id: "studio-template",
      name: "Studio setup",
      createdAt: 1,
      updatedAt: 1,
      scopeId: "studio-id",
      form: studioTemplateForm,
      mediaReferences: [],
    });
    await flushPromises();

    expect(liveForm.family).toBe("flux");
    expect(liveForm.loras).toEqual([]);
    expect(liveForm.upscaleModel).toBe("");
    expect(liveForm.controlModel).toBe("");
    expect(liveForm.cameraControl).toBeNull();
    expect(liveForm.sourceFit).toMatchObject({ mode: "upscale-then-fit", upscalerModel: "" });
  });

  it("routes long distilled video through chain SSE with metadata-only completion", async () => {
    const chainModel: ModelEntry = {
      ...model,
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([chainModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a continuous flight through clouds");
    await wrapper.get("[data-test='mobile-frames']").setValue("177");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-chain-cue']").text()).toContain("2 chained clips");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.path).toBe("/api/generate/chain/stream");
    expect(openStreams[0]?.options.headers).toEqual({
      "X-Mold-SSE-Payload": "metadata-only",
    });
    expect(openStreams[0]?.options.body).toMatchObject({
      model: chainModel.name,
      prompt: "a continuous flight through clouds",
      total_frames: 177,
      clip_frames: 97,
      motion_tail_frames: 17,
    });
    expect(openStreams[0]?.options.body).not.toHaveProperty("frames");

    openStreams[0]?.resolve();
    await flushPromises();
  });

  it("does not retain Qwen Target validation after switching to text-to-video", async () => {
    const qwen: ModelEntry = {
      ...model,
      name: "qwen-image-edit:bf16",
      family: "qwen-image-edit",
      default_width: 1024,
      default_height: 1024,
    };
    const video: ModelEntry = {
      ...model,
      name: "ltx-video-0.9.8-13b-dev:bf16",
      family: "ltx-video",
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([qwen, video]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a clean product orbit");
    expect(fieldControl("Model").element).toHaveProperty("value", qwen.name);
    expect(wrapper.get("[data-test='mobile-source-validation']").text()).toContain("Target photo");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );

    await fieldControl("Model").setValue(video.name);
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

  it("blocks invalid numeric parameters and oversized custom resolutions", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a precise studio portrait");

    await fieldControl("Steps").setValue("0");
    expect(wrapper.get("[data-test='mobile-basic-parameter-error']").text()).toContain("1 to 100");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
    await fieldControl("Steps").setValue("20");
    await fieldControl("Guidance").setValue("101");
    expect(wrapper.get("[data-test='mobile-basic-parameter-error']").text()).toContain("0 to 100");
    await fieldControl("Guidance").setValue("3");

    await wrapper.get("[data-test='mobile-resolution-custom-toggle']").trigger("click");
    await wrapper.get("input[aria-label='Custom width']").setValue("2000");
    await wrapper.get("input[aria-label='Custom height']").setValue("2000");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-resolution-error']").text()).toContain("1.8 MP");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
  });

  it("keeps auxiliary models out of the Model picker while exposing their tools", async () => {
    const imageModel: ModelEntry = {
      ...model,
      name: "flux:image",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    const upscaler: ModelEntry = {
      ...model,
      name: "real-esrgan-x4plus:fp16",
      family: "upscaler",
      downloaded: false,
    };
    const controlNet: ModelEntry = {
      ...model,
      name: "controlnet-canny-sd15",
      family: "controlnet",
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([upscaler, controlNet, imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();

    expect(
      fieldControl("Model")
        .findAll("option")
        .map((option) => option.text()),
    ).toEqual([imageModel.name]);
    expect(wrapper.get("[data-test='mobile-upscale']").text()).toContain(upscaler.name);
    expect(
      wrapper.findAll("label.field").some((field) => field.text().includes("Negative prompt")),
    ).toBe(false);
  });

  it("prepares Batch for review, then queues edited siblings with provenance", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three variations of a storm");
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe("Develop 3 prints");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "three variations of a storm",
      { variations: 3, modelFamily: model.family },
      target,
    );
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(0);
    expect(wrapper.get("[data-test='mobile-prepared-expansion']").text()).toContain(
      "Review 3 variations",
    );
    const editors = wrapper.findAll(".mobile-prepared-editor");
    await editors[1]?.setValue("an edited middle storm");
    const developPrepared = wrapper.get("[data-test='mobile-develop-prepared']");
    (developPrepared.element as HTMLButtonElement).focus();
    await developPrepared.trigger("click");
    await flushPromises();

    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(3);
    expect(document.activeElement).toBe(fieldControl("Prompt").element);
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop 3 prints (+3 queued)",
    );
    expect(openStreams).toHaveLength(2);
    const firstSeed = openStreams[0]?.options.body.seed as number;
    expect(openStreams.map((stream) => stream.options.body.batch_size)).toEqual([1, 1]);
    expect(openStreams.map((stream) => stream.options.body.seed)).toEqual([
      firstSeed,
      firstSeed + 1,
    ]);
    expect(openStreams.map((stream) => stream.options.body.prompt)).toEqual([
      "three variations of a storm · prepared 1",
      "an edited middle storm",
    ]);
    expect(openStreams.map((stream) => stream.options.body.original_prompt)).toEqual([
      "three variations of a storm",
      "three variations of a storm",
    ]);
    expect(openStreams[0]?.options.body.batch_id).toEqual(expect.any(String));
    expect(openStreams.map((stream) => stream.options.body.batch_index)).toEqual([1, 2]);
    expect(openStreams.map((stream) => stream.options.body.batch_count)).toEqual([3, 3]);
  });

  it("identifies a failed middle prepared sibling by variation and reviewed prompt", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three weather studies");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    const editors = wrapper.findAll(".mobile-prepared-editor");
    await editors[0]!.setValue("clear dawn");
    await editors[1]!.setValue("middle thunderstorm");
    await editors[2]!.setValue("quiet dusk");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("first"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 1,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(openStreams).toHaveLength(3);
    openStreams[1]!.options.onEvent("error", JSON.stringify({ message: "host ran out of memory" }));
    openStreams[1]!.resolve();
    openStreams[2]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("third"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 3,
        generation_time_ms: 700,
        model: model.name,
      }),
    );
    openStreams[2]!.resolve();
    await flushPromises();

    const liveSummary = wrapper.findAll(".sr-only[aria-live='polite']")[1]!.text();
    expect(liveSummary).toContain("Variation 2, “middle thunderstorm”");
    expect(liveSummary).toContain("host ran out of memory");
    expect(wrapper.find("img.result-media").exists()).toBe(true);
  });

  it("composes prepared sibling failures with an unconfirmed cancellation caveat", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three mixed outcomes");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    const editors = wrapper.findAll(".mobile-prepared-editor");
    await editors[0]!.setValue("successful dawn");
    await editors[1]!.setValue("failed middle storm");
    await editors[2]!.setValue("cancelled dusk");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("first"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 1,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(openStreams).toHaveLength(3);
    const cancelRow = wrapper
      .findAll("[data-test='mobile-generation-job']")
      .find((row) => row.text().includes("cancelled dusk"))!;
    await cancelRow.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();
    openStreams[2]!.resolve();
    openStreams[1]!.options.onEvent("error", JSON.stringify({ message: "host ran out of memory" }));
    openStreams[1]!.resolve();
    await flushPromises();

    const liveSummary = wrapper.findAll(".sr-only[aria-live='polite']")[1]!.text();
    expect(liveSummary).toContain("Variation 2, “failed middle storm”");
    expect(liveSummary).toContain("host ran out of memory");
    expect(liveSummary).toContain("Cancellation failed");
    expect(liveSummary).toContain("remote cancellation was not confirmed");
    expect(wrapper.find("img.result-media").exists()).toBe(true);
  });

  it("rejects late or malformed exact-N expansion responses without replacing the form", async () => {
    let resolveExpansion!: (value: { expanded: string[] }) => void;
    expandPrompt.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveExpansion = resolve;
      }),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("original source");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await fieldControl("Prompt").setValue("newer source");
    resolveExpansion({ expanded: ["one", "two", "three"] });
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-expansion-error']").text()).toContain(
      "changed while expansion was running",
    );
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("newer source");

    expandPrompt.mockResolvedValueOnce({ expanded: ["one", "  ", "three"] });
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-expansion-error']").text()).toContain(
      "Prompt 2 was empty",
    );
  });

  it("invalidates quick Batch 1 submission when Undo wins deferred preprocessing", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementationOnce(
      (input: { source: string | null; mask: string | null }) =>
        new Promise((resolve) => {
          finishPreprocess = () =>
            resolve({ source: input.source, mask: input.mask, changed: false });
        }),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await fieldControl("Prompt").setValue("small lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toContain("prepared 1");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    finishPreprocess();
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("small lighthouse");
  });

  it("lets Discard cancel a prepared batch while source preprocessing is pending", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementationOnce(
      (input: { source: string | null; mask: string | null }) =>
        new Promise((resolve) => {
          finishPreprocess = () =>
            resolve({ source: input.source, mask: input.mask, changed: false });
        }),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two wet-plate portraits");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    expect(
      wrapper.get("[data-test='mobile-discard-prepared']").attributes("disabled"),
    ).toBeUndefined();
    await wrapper.get("[data-test='mobile-discard-prepared']").trigger("click");
    finishPreprocess();
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
  });

  it("unlocks preserved prepared work after preprocessing fails and allows retry", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    applySourceFitPreprocess.mockRejectedValueOnce(new Error("upscaler unavailable"));
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two preserved portraits");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "upscaler unavailable",
    );
    const editor = wrapper.findAll(".mobile-prepared-editor")[0]!;
    expect(editor.attributes("disabled")).toBeUndefined();
    expect(
      wrapper.get("[data-test='mobile-develop-prepared']").attributes("disabled"),
    ).toBeUndefined();
    await editor.setValue("edited after preprocessing failed");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(2);
    expect(openStreams[0]!.options.body.prompt).toBe("edited after preprocessing failed");
  });

  it("does not steal focus when delayed prepared submission finishes after the user moves", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementationOnce(
      (input: { source: string | null; mask: string | null }) =>
        new Promise((resolve) => {
          finishPreprocess = () =>
            resolve({ source: input.source, mask: input.mask, changed: false });
        }),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two quiet portraits");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    const developPrepared = wrapper.get("[data-test='mobile-develop-prepared']");
    (developPrepared.element as HTMLButtonElement).focus();
    await developPrepared.trigger("click");
    await flushPromises();
    const outside = document.createElement("button");
    document.body.append(outside);
    outside.focus();
    finishPreprocess();
    await flushPromises();

    expect(document.activeElement).toBe(outside);
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(2);
  });

  it("keeps a missing expansion-model pull inline on the exact selected host", async () => {
    expandPrompt
      .mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      )
      .mockResolvedValueOnce({ expanded: ["a lighthouse after the storm"] });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-pull-expansion']").text()).toContain(
      "Pull expansion model",
    );
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("qwen3-expand:q8");
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("Studio");
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams.find((stream) => stream.path === "/api/downloads/stream");
    expect(downloadStream).toBeDefined();
    expect(downloadStream?.options.target).toEqual(target);
    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith("qwen3-expand:q8", target, false);

    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "expansion-job",
        model: "qwen3-expand:q8",
        position: 0,
      }),
    );
    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({
        type: "started",
        id: "expansion-job",
        files_total: 2,
        bytes_total: 1_000,
      }),
    );
    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({
        type: "progress",
        id: "expansion-job",
        files_done: 1,
        bytes_done: 500,
        current_file: "model.safetensors",
      }),
    );
    await flushPromises();
    expect(wrapper.get("[role='progressbar']").attributes("aria-valuenow")).toBe("50");
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("model.safetensors");

    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
    );
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-retry-expansion']").text()).toBe("Retry expansion");
    await wrapper.get("[data-test='mobile-retry-expansion']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse after the storm",
    );
    expect(wrapper.get("[data-test='mobile-prompt-expand']").text()).toBe("Expand");
    expect(
      wrapper.get("[data-test='mobile-prompt-expand']").attributes("disabled"),
    ).toBeUndefined();
  });

  it.each(["job_done", "job_failed", "job_cancelled"] as const)(
    "retains %s recovery UI when the terminal event beats the returned pull ID",
    async (type) => {
      const pinia = createPinia();
      const downloads = useMobileDownloadsStore(pinia);
      const response = deferred<string | null>();
      const retryResponse = deferred<string | null>();
      startCatalogDownload
        .mockReturnValueOnce(response.promise)
        .mockReturnValueOnce(retryResponse.promise);
      expandPrompt.mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      );
      wrapper = mountMobileApp(pinia);
      await flushPromises();
      await fieldControl("Prompt").setValue("fast cached expansion");
      await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
      await flushPromises();
      downloads.registerConsumer("catalog", [
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          apiKey: "catalog-rotated-key",
          hostname: "studio",
          version: "0.18.0",
          instanceId: "studio-id",
          online: true,
        },
      ]);
      await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
      await flushPromises();
      const frozenStream = openStreams
        .filter((stream) => stream.path === "/api/downloads/stream")
        .at(-1)!;
      frozenStream.options.onEvent(
        "download",
        JSON.stringify({
          type: "snapshot",
          listing: { active_jobs: [], queued: [], history: [] },
        }),
      );
      await flushPromises();
      frozenStream.options.onEvent(
        "download",
        JSON.stringify(
          type === "job_done"
            ? { type, id: "fast-job", model: "qwen3-expand:q8" }
            : type === "job_failed"
              ? { type, id: "fast-job", error: "cached failure" }
              : { type, id: "fast-job" },
        ),
      );

      response.resolve("fast-job");
      await flushPromises();
      expect(
        openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options
          .target?.apiKey,
      ).toBe("catalog-rotated-key");
      if (type === "job_done") {
        expect(wrapper.get("[data-test='mobile-retry-expansion']").text()).toBe("Retry expansion");
      } else {
        expect(wrapper.get("[data-test='mobile-retry-expansion-pull']").text()).toContain(
          "qwen3-expand:q8",
        );
        if (type === "job_failed") {
          expect(wrapper.get(".mobile-expansion-pull").text()).toContain("cached failure");
        }
        await wrapper.get("[data-test='mobile-retry-expansion-pull']").trigger("click");
        await flushPromises();
        const retryStream = openStreams
          .filter((stream) => stream.path === "/api/downloads/stream")
          .at(-1)!;
        expect(retryStream.options.target?.apiKey).toBe(target.apiKey);
        retryStream.options.onEvent(
          "download",
          JSON.stringify({
            type: "snapshot",
            listing: { active_jobs: [], queued: [], history: [] },
          }),
        );
        retryResponse.reject(new Error("retry stopped"));
        await flushPromises();
        expect(
          openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options
            .target?.apiKey,
        ).toBe("catalog-rotated-key");
      }
    },
  );

  it("releases fatal post-open recovery before a late pull ID and reacquires on Retry", async () => {
    const pinia = createPinia();
    const downloads = useMobileDownloadsStore(pinia);
    const lateResponse = deferred<string | null>();
    const retryResponse = deferred<string | null>();
    startCatalogDownload
      .mockReturnValueOnce(lateResponse.promise)
      .mockReturnValueOnce(retryResponse.promise);
    expandPrompt.mockRejectedValueOnce(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await fieldControl("Prompt").setValue("closed stream expansion");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    downloads.registerConsumer("catalog", [
      {
        id: "studio-id",
        name: "Studio",
        baseUrl: target.baseUrl,
        apiKey: "catalog-rotated-key",
        hostname: "studio",
        version: "0.18.0",
        instanceId: "studio-id",
        online: true,
      },
    ]);
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const frozenStream = openStreams
      .filter((stream) => stream.path === "/api/downloads/stream")
      .at(-1)!;
    frozenStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    frozenStream.options.onClose?.(new Error("download stream died"));
    await flushPromises();
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("download stream died");
    expect(
      openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options.target
        ?.apiKey,
    ).toBe("catalog-rotated-key");

    lateResponse.resolve("late-dead-job");
    await flushPromises();
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("download stream died");
    expect(
      openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options.target
        ?.apiKey,
    ).toBe("catalog-rotated-key");

    await wrapper.get("[data-test='mobile-retry-expansion-pull']").trigger("click");
    await flushPromises();
    const retryStream = openStreams
      .filter((stream) => stream.path === "/api/downloads/stream")
      .at(-1)!;
    expect(retryStream.options.target?.apiKey).toBe(target.apiKey);
    retryStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    retryResponse.reject(new Error("retry stopped"));
    await flushPromises();
    expect(
      openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options.target
        ?.apiKey,
    ).toBe("catalog-rotated-key");
  });

  it.each(["edit", "remove"] as const)(
    "keeps prepared work when a %s supersedes a pending missing-model replacement",
    async (action) => {
      const pinia = createPinia();
      const downloads = useMobileDownloadsStore(pinia);
      const release = vi.spyOn(downloads, "releaseFrozenPull");
      wrapper = mountMobileApp(pinia);
      await flushPromises();
      await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
      await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
      await fieldControl("Prompt").setValue("three preserved studies");
      await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
      await flushPromises();

      expandPrompt.mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      );
      await wrapper.get("[data-test='mobile-regenerate-prepared']").trigger("click");
      await flushPromises();
      downloads.registerConsumer("catalog", [
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          apiKey: "catalog-rotated-key",
          hostname: "studio",
          version: "0.18.0",
          instanceId: "studio-id",
          online: true,
        },
      ]);
      await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
      await flushPromises();
      const downloadStream = openStreams
        .filter((stream) => stream.path === "/api/downloads/stream")
        .at(-1)!;
      expect(downloadStream.options.target?.apiKey).toBe(target.apiKey);
      downloadStream.options.onEvent(
        "download",
        JSON.stringify({
          type: "snapshot",
          listing: { active_jobs: [], queued: [], history: [] },
        }),
      );
      await flushPromises();

      if (action === "edit") {
        await wrapper.findAll(".mobile-prepared-editor")[0]!.setValue("my newer reviewed edit");
      } else {
        await wrapper.findAll("[data-test='mobile-prepared-remove']")[2]!.trigger("click");
      }
      await flushPromises();

      expect(release).toHaveBeenCalled();
      expect(
        openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options
          .target?.apiKey,
      ).toBe("catalog-rotated-key");
      expect(wrapper.find(".mobile-expansion-pull").exists()).toBe(false);
      expect(wrapper.get("[data-test='mobile-prepared-expansion']").text()).toContain(
        "Pending replacement cancelled",
      );
      if (action === "edit") {
        expect(
          (wrapper.findAll(".mobile-prepared-editor")[0]!.element as HTMLTextAreaElement).value,
        ).toBe("my newer reviewed edit");
        expect(wrapper.findAll(".mobile-prepared-editor")).toHaveLength(3);
      } else {
        expect(wrapper.findAll(".mobile-prepared-editor")).toHaveLength(2);
      }
      downloadStream.options.onEvent(
        "download",
        JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
      );
      await flushPromises();
      expect(wrapper.find(".mobile-expansion-pull").exists()).toBe(false);
    },
  );

  it("aborts missing-model retry when form identity changes during the pull", async () => {
    expandPrompt.mockRejectedValueOnce(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    const pinia = createPinia();
    const downloads = useMobileDownloadsStore(pinia);
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await fieldControl("Prompt").setValue("frozen lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    downloads.registerConsumer("catalog", [
      {
        id: "studio-id",
        name: "Studio",
        baseUrl: target.baseUrl,
        apiKey: "catalog-rotated-key",
        hostname: "studio",
        version: "0.18.0",
        instanceId: "studio-id",
        online: true,
      },
    ]);
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams
      .filter((stream) => stream.path === "/api/downloads/stream")
      .at(-1)!;
    expect(downloadStream.options.target?.apiKey).toBe(target.apiKey);
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "expansion-job",
        model: "qwen3-expand:q8",
        position: 0,
      }),
    );
    await fieldControl("Prompt").setValue("newer lighthouse");
    expect(
      openStreams.filter((stream) => stream.path === "/api/downloads/stream").at(-1)!.options.target
        ?.apiKey,
    ).toBe("catalog-rotated-key");
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
    );
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledTimes(1);
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("newer lighthouse");
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("inputs changed");
    expect(wrapper.find("[data-test='mobile-retry-expansion']").exists()).toBe(false);
  });

  it("rejects a late recovery expansion when inputs change before commit", async () => {
    let resolveRetry!: (value: { expanded: string[] }) => void;
    expandPrompt
      .mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRetry = resolve;
          }),
      );
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("frozen harbor");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams.find((stream) => stream.path === "/api/downloads/stream")!;
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "expansion-job",
        model: "qwen3-expand:q8",
        position: 0,
      }),
    );
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
    );
    await flushPromises();
    await wrapper.get("[data-test='mobile-retry-expansion']").trigger("click");
    await flushPromises();
    await fieldControl("Prompt").setValue("new harbor input");
    resolveRetry({ expanded: ["late expanded harbor"] });
    await flushPromises();

    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("new harbor input");
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("inputs changed");
  });

  it("restores prompt focus when stale Batch N refresh explicitly becomes Batch 1", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two studies");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-decrement']").trigger("click");
    const refresh = wrapper.get("[data-test='mobile-refresh-prepared']");
    (refresh.element as HTMLButtonElement).focus();
    await refresh.trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(document.activeElement).toBe(fieldControl("Prompt").element);
  });

  it("does not steal outside focus when delayed Batch N to Batch 1 refresh completes", async () => {
    let resolveRefresh!: (value: { expanded: string[] }) => void;
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two delayed studies");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-decrement']").trigger("click");
    expandPrompt.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRefresh = resolve;
        }),
    );
    const refresh = wrapper.get("[data-test='mobile-refresh-prepared']");
    (refresh.element as HTMLButtonElement).focus();
    await refresh.trigger("click");
    const outside = document.createElement("button");
    document.body.append(outside);
    outside.focus();
    resolveRefresh({ expanded: ["one delayed study"] });
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(document.activeElement).toBe(outside);
  });

  it("revokes deferred expansion and preprocessing authority on unmount", async () => {
    let resolveExpansion!: (value: { expanded: string[] }) => void;
    expandPrompt.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveExpansion = resolve;
      }),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("late expansion");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    wrapper.unmount();
    wrapper = null;
    resolveExpansion({ expanded: ["late one", "late two"] });
    await flushPromises();
    expect(openStreams).toHaveLength(0);

    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    let finishPreprocess!: () => void;
    applySourceFitPreprocess.mockImplementationOnce(
      (input: { source: string | null; mask: string | null }) =>
        new Promise((resolve) => {
          finishPreprocess = () =>
            resolve({ source: input.source, mask: input.mask, changed: false });
        }),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("late preprocess");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    wrapper.unmount();
    wrapper = null;
    finishPreprocess();
    await flushPromises();
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
  });

  it("keeps a remounted Generate download consumer safe from the old instance", async () => {
    const pinia = createPinia();
    const downloads = useMobileDownloadsStore(pinia);
    const register = vi.spyOn(downloads, "registerConsumer");
    const unregister = vi.spyOn(downloads, "unregisterConsumer");
    let rejectOldExpansion!: (reason: unknown) => void;
    expandPrompt.mockReturnValueOnce(
      new Promise((_resolve, reject) => {
        rejectOldExpansion = reject;
      }),
    );
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await fieldControl("Prompt").setValue("old expansion");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    wrapper.unmount();
    wrapper = null;
    const oldConsumerId = unregister.mock.calls.at(-1)![0];

    expandPrompt.mockRejectedValueOnce(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await fieldControl("Prompt").setValue("new expansion");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const newConsumerId = register.mock.calls.at(-1)![0];
    expect(newConsumerId).not.toBe("mobile-generate");
    expect(newConsumerId).not.toBe(oldConsumerId);
    const newStream = openStreams.find((stream) => stream.path === "/api/downloads/stream")!;
    const unregisterCount = unregister.mock.calls.length;

    rejectOldExpansion(new Error("local expand model not found, run: mold pull qwen3-expand:q8"));
    await flushPromises();
    expect(unregister.mock.calls).toHaveLength(unregisterCount);
    expect(newStream.options.signal.aborted).toBe(false);
  });

  it("revokes a frozen pull and prevents a deferred retry from acting after unmount", async () => {
    const pinia = createPinia();
    const downloads = useMobileDownloadsStore(pinia);
    const release = vi.spyOn(downloads, "releaseFrozenPull");
    const unregister = vi.spyOn(downloads, "unregisterConsumer");
    let resolveRetry!: (value: { expanded: string[] }) => void;
    expandPrompt
      .mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRetry = resolve;
          }),
      );
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await fieldControl("Prompt").setValue("unmounted retry");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams.find((stream) => stream.path === "/api/downloads/stream")!;
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "expansion-job",
        model: "qwen3-expand:q8",
        position: 0,
      }),
    );
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
    );
    await flushPromises();
    await wrapper.get("[data-test='mobile-retry-expansion']").trigger("click");
    await flushPromises();

    wrapper.unmount();
    wrapper = null;
    const releaseCount = release.mock.calls.length;
    const unregisterCount = unregister.mock.calls.length;
    expect(downloadStream.options.signal.aborted).toBe(true);
    resolveRetry({ expanded: ["late retry result"] });
    await flushPromises();

    expect(release.mock.calls).toHaveLength(releaseCount);
    expect(unregister.mock.calls).toHaveLength(unregisterCount);
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
  });

  it("uses an explicit mobile seed mode and submits the fixed value", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-seed-mode-random']").attributes("aria-pressed")).toBe(
      "true",
    );
    expect(wrapper.find("[data-test='mobile-seed-input']").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-seed-mode-fixed']").trigger("click");
    await fieldControl("Prompt").setValue("repeat this print");
    await wrapper.get("[data-test='mobile-seed-input']").setValue("not-a-number");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
    await wrapper.get("[data-test='mobile-seed-input']").setValue("1234");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams[0]?.options.body).toMatchObject({ seed: 1234 });
    await wrapper.get("[data-test='mobile-seed-mode-random']").trigger("click");
    expect(wrapper.find("[data-test='mobile-seed-input']").exists()).toBe(false);
  });

  it("snapshots multiple prompts, shows their live queue, and cancels only one", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first prompt");
    openStreams[0]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "denoise_step", step: 3, total: 30, elapsed_ms: 10 }),
    );
    await submitPrompt("second prompt");
    openStreams[1]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-2" }),
    );
    await flushPromises();

    expect(openStreams).toHaveLength(2);
    expect(openStreams.map((stream) => stream.options.body.prompt)).toEqual([
      "first prompt",
      "second prompt",
    ]);
    expect(
      openStreams.every(
        (stream) => stream.options.headers?.["X-Mold-SSE-Payload"] === "metadata-only",
      ),
    ).toBe(true);
    expect(openStreams.every((stream) => stream.options.body.model === model.name)).toBe(true);
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop print (+2 queued)",
    );
    expect(wrapper.get("[data-test='mobile-generation-queue']").attributes("aria-live")).toBe(
      undefined,
    );
    expect(wrapper.get(".sr-only[aria-live='polite']").text()).toBe("2 active generations.");

    const rows = wrapper.findAll("[data-test='mobile-generation-job']");
    expect(rows).toHaveLength(2);
    expect(rows[0]?.text()).toContain("first prompt");
    expect(rows[0]?.get("[data-test='mobile-generation-status']").text()).toBe("3/30");
    expect(rows[1]?.text()).toContain("second prompt");
    expect(rows[1]?.get("[data-test='mobile-generation-status']").text()).toBe("QUEUED #1");

    await rows[1]?.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/job-2", { method: "DELETE" });
    expect(openStreams[0]?.options.signal.aborted).toBe(false);
    expect(openStreams[1]?.options.signal.aborted).toBe(true);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(1);
    expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain("first prompt");
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop print (+1 queued)",
    );
    expect(wrapper.get(".sr-only[aria-live='polite']").text()).toBe("1 active generation.");
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation cancelled.",
    );

    const firstSignal = openStreams[0]?.options.signal;
    wrapper.unmount();
    wrapper = null;
    expect(firstSignal?.aborted).toBe(true);
  });

  it("keeps an unconfirmed remote-cancellation warning after the stream settles", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("cancel before the remote queue id arrives");

    await wrapper.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toMatch(
      /remote cancellation was not confirmed/i,
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Cancellation failed",
    );

    openStreams[0]?.resolve();
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toMatch(
      /remote cancellation was not confirmed/i,
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Cancellation failed",
    );
  });

  it("keeps a completed result visible while a queued sibling settles independently", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("finished prompt");
    await submitPrompt("failing prompt");
    openStreams[1]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-failing" }),
    );
    openStreams[0]?.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("generated-image"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 42,
        generation_time_ms: 1_250,
        model: model.name,
      }),
    );
    openStreams[0]?.resolve();
    await flushPromises();

    expect(wrapper.find("img.result-media").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(1);
    expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain("failing prompt");

    openStreams[1]?.options.onEvent("error", JSON.stringify({ message: "host ran out of memory" }));
    openStreams[1]?.resolve();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(wrapper.find("img.result-media").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe(
      "host ran out of memory",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation failed. host ran out of memory",
    );
  });

  it("promotes the last of simultaneous completions before pruning older results", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first simultaneous prompt");
    await submitPrompt("second simultaneous prompt");
    for (const [index, stream] of openStreams.entries()) {
      stream.options.onEvent(
        "complete",
        JSON.stringify({
          image: btoa(`generated-${index}`),
          format: "png",
          width: 768,
          height: 512,
          seed_used: index + 1,
          generation_time_ms: 500,
          model: model.name,
        }),
      );
      stream.resolve();
    }
    await flushPromises();

    expect(wrapper.get("img.result-media").attributes("src")).toBe("blob:thumbnail-2");
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("0.5s · seed 2");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:thumbnail-1");
  });

  it("streams a metadata-only generated video from the host", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("stream this video");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "mp4",
        filename: "generated clip.mp4",
        width: 768,
        height: 512,
        seed_used: 23,
        generation_time_ms: 500,
        model: model.name,
        video_frames: 121,
        video_fps: 30,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/generated%20clip.mp4", {
      target,
      cacheKey: "studio-id",
      allowLegacyBlob: false,
    });
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(wrapper.get("video.result-media").attributes("src")).toBe(
      "https://studio/media/full-video",
    );
  });

  it("does not show an older result when the latest saved result has no filename", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first successful prompt");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("first-result"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 1,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(wrapper.find("img.result-media").exists()).toBe(true);

    await submitPrompt("metadata without a file");
    openStreams[1]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        width: 768,
        height: 512,
        seed_used: 2,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[1]!.resolve();
    await flushPromises();

    expect(wrapper.find(".result-media").exists()).toBe(false);
    expect(wrapper.get(".result-preview-error").text()).toContain("saved result URL");
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "saved result URL",
    );
  });

  it("shows ticket failures and lets the user retry the preview", async () => {
    streamableMediaUrl
      .mockRejectedValueOnce(new Error("The host refused the media ticket."))
      .mockResolvedValueOnce("https://studio/media/retried-image");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("ticketed image");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        filename: "ticketed image.png",
        width: 768,
        height: 512,
        seed_used: 4,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(wrapper.get(".result-preview-error").text()).toContain("refused the media ticket");
    expect(wrapper.find(".result-media").exists()).toBe(false);
    await wrapper.get(".result-preview-error button").trigger("click");
    await flushPromises();

    expect(wrapper.find(".result-preview-error").exists()).toBe(false);
    expect(wrapper.get("img.result-media").attributes("src")).toBe(
      "https://studio/media/retried-image",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Result preview refreshed.",
    );
  });

  it("renews an expired generated-video ticket when playback starts", async () => {
    streamableMediaUrl
      .mockResolvedValueOnce("https://studio/media/video?media_token=old&expires=1")
      .mockResolvedValueOnce("https://studio/media/video?media_token=new&expires=4102444800");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("renew this video");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "mp4",
        filename: "renew this video.mp4",
        width: 768,
        height: 512,
        seed_used: 5,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(wrapper.get("video.result-media").attributes("src")).toContain("media_token=old");

    await wrapper.get("video.result-media").trigger("play");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("video.result-media").attributes("src")).toContain("media_token=new");
  });

  it("bounds automatic media recovery and exposes a manual retry", async () => {
    const unchangedUrl = "https://studio/media/missing?media_token=unchanged&expires=4102444800";
    streamableMediaUrl.mockResolvedValueOnce(unchangedUrl).mockResolvedValueOnce(unchangedUrl);
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("missing generated image");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        filename: "missing.png",
        width: 768,
        height: 512,
        seed_used: 6,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    const originalImage = wrapper.get("img.result-media").element;
    await wrapper.get("img.result-media").trigger("error");
    await flushPromises();
    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("img.result-media").attributes("src")).toBe(unchangedUrl);
    expect(wrapper.get("img.result-media").element).not.toBe(originalImage);

    await wrapper.get("img.result-media").trigger("error");
    await flushPromises();
    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.find("img.result-media").exists()).toBe(false);
    expect(wrapper.get(".result-preview-error").text()).toContain(
      "Couldn’t load this generated print",
    );
    expect(wrapper.get(".result-preview-error button").text()).toBe("Try preview again");
  });

  it("remounts generated video when forced renewal returns the same URL", async () => {
    const unchangedUrl =
      "https://studio/media/missing-video?media_token=unchanged&expires=4102444800";
    streamableMediaUrl.mockResolvedValueOnce(unchangedUrl).mockResolvedValueOnce(unchangedUrl);
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("missing generated video");
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "mp4",
        filename: "missing-video.mp4",
        width: 768,
        height: 512,
        seed_used: 7,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    const originalVideo = wrapper.get("video.result-media").element;
    await wrapper.get("video.result-media").trigger("error");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("video.result-media").attributes("src")).toBe(unchangedUrl);
    expect(wrapper.get("video.result-media").element).not.toBe(originalVideo);

    await wrapper.get("video.result-media").trigger("error");
    await flushPromises();
    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.find("video.result-media").exists()).toBe(false);
    expect(wrapper.get(".result-preview-error").text()).toContain(
      "Couldn’t load this generated print",
    );
  });

  it("shows a completion that wins the cancellation race", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("almost finished prompt");
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-finishing" }),
    );
    apiFetchTo.mockImplementationOnce(async () => {
      openStreams[0]!.options.onEvent(
        "complete",
        JSON.stringify({
          image: btoa("finished-during-cancel"),
          format: "png",
          width: 768,
          height: 512,
          seed_used: 91,
          generation_time_ms: 500,
          model: model.name,
        }),
      );
      openStreams[0]!.resolve();
      return new Response(null, { status: 204 });
    });

    await wrapper.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(wrapper.find("img.result-media").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("0.5s · seed 91");
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation completed.",
    );
  });

  it("preserves a server failure that wins the cancellation race", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("failing during cancel");
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-failing" }),
    );
    apiFetchTo.mockImplementationOnce(async () => {
      openStreams[0]!.options.onEvent(
        "error",
        JSON.stringify({ message: "host ran out of memory" }),
      );
      openStreams[0]!.resolve();
      return new Response(null, { status: 204 });
    });

    await wrapper.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe(
      "host ran out of memory",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation failed. host ran out of memory",
    );
  });

  it("coalesces simultaneous completion refreshes while Gallery is open", async () => {
    const galleryResolvers: Array<() => void> = [];
    let galleryCalls = 0;
    let galleryInFlight = 0;
    let maxGalleryInFlight = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        galleryCalls += 1;
        galleryInFlight += 1;
        maxGalleryInFlight = Math.max(maxGalleryInFlight, galleryInFlight);
        return new Promise<GalleryImage[]>((resolve) => {
          galleryResolvers.push(() => {
            galleryInFlight -= 1;
            resolve([print]);
          });
        });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("first gallery prompt");
    await submitPrompt("second gallery prompt");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(galleryResolvers).toHaveLength(1));

    for (const [index, stream] of openStreams.entries()) {
      stream.options.onEvent(
        "complete",
        JSON.stringify({
          image: btoa(`generated-${index}`),
          format: "png",
          width: 768,
          height: 512,
          seed_used: index + 1,
          generation_time_ms: 500,
          model: model.name,
        }),
      );
      stream.resolve();
    }
    await flushPromises();

    expect(galleryCalls).toBe(1);
    expect(maxGalleryInFlight).toBe(1);
    galleryResolvers[0]!();
    await vi.waitFor(() => expect(galleryResolvers).toHaveLength(2));
    expect(galleryCalls).toBe(2);
    expect(maxGalleryInFlight).toBe(1);

    galleryResolvers[1]!();
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    expect(galleryCalls).toBe(2);
  });

  it("serializes Load older with a completion-triggered Gallery refresh", async () => {
    const prints = Array.from({ length: 41 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    let galleryCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        galleryCalls += 1;
        return Promise.resolve(prints);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("refresh after older page");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("button.gallery-more").exists()).toBe(true));

    const olderThumbnail = { release: null as (() => void) | null };
    apiFetchTo.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          olderThumbnail.release = () =>
            resolve({ blob: () => Promise.resolve(new Blob(["older"])) } as Response);
        }),
    );
    await wrapper.get("button.gallery-more").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("button.gallery-more").attributes("disabled")).toBeDefined(),
    );

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("generated-before-refresh"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 12,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(galleryCalls).toBe(1);

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);

    if (!olderThumbnail.release) throw new Error("Older thumbnail request did not start");
    olderThumbnail.release();
    await flushPromises();
    expect(galleryCalls).toBe(1);

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await vi.waitFor(() => expect(galleryCalls).toBe(2));
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(40));
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
  });

  it("keeps focus stable when the viewer closes before a queued Gallery refresh starts", async () => {
    const prints = Array.from({ length: 41 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    let galleryCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        galleryCalls += 1;
        return Promise.resolve(prints);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("refresh after an early viewer close");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("button.gallery-more").exists()).toBe(true));

    const olderThumbnail = { release: null as (() => void) | null };
    apiFetchTo.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          olderThumbnail.release = () =>
            resolve({ blob: () => Promise.resolve(new Blob(["older"])) } as Response);
        }),
    );
    await wrapper.get("button.gallery-more").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("button.gallery-more").attributes("disabled")).toBeDefined(),
    );

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("generated-before-refresh"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 13,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();
    expect(galleryCalls).toBe(1);

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
    expect(galleryCalls).toBe(1);

    if (!olderThumbnail.release) throw new Error("Older thumbnail request did not start");
    olderThumbnail.release();
    await vi.waitFor(() => expect(galleryCalls).toBe(2));
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(40));
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
  });
});

describe("MobileApp settings", () => {
  it("opens as a focused destination and returns to the unchanged primary tab", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='mobile-open-settings']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-settings']").isVisible()).toBe(true);
    expect(wrapper.find(".mobile-tabs").exists()).toBe(false);
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-settings-back']").element);

    await wrapper.get("[data-test='mobile-settings-back']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-open-settings']").element);
  });

  it("applies and persists family and appearance changes immediately", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-open-settings']").trigger("click");

    await wrapper.get('input[name="mobile-theme-family"][value="safelight"]').setValue(true);
    await wrapper.get('input[name="mobile-theme-appearance"][value="light"]').setValue(true);
    await flushPromises();

    expect(document.documentElement.dataset.themeFamily).toBe("safelight");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(JSON.parse(localStorage.getItem("mold.mobile.settings.v1") ?? "{}")).toEqual({
      theme: "light",
      themeFamily: "safelight",
    });

    await wrapper.get('input[name="mobile-theme-appearance"][value="system"]').setValue(true);
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it("opens host management from Settings", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-open-settings']").trigger("click");
    await wrapper.get(".mobile-settings-manage").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-settings']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-hosts']").attributes("aria-current")).toBe("page");
  });
});

describe("MobileApp gallery", () => {
  it("defers completion refreshes until an open viewer closes", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("complete behind the viewer");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery")).toHaveLength(1);
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("completed behind viewer"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 61,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation completed.",
    );
    expect(wrapper.get("[data-test='gallery-viewer'] .sr-only[aria-live='polite']").text()).toBe(
      "Generation completed.",
    );
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery")).toHaveLength(1);
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith("blob:thumbnail-1");

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await vi.waitFor(() =>
      expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery")).toHaveLength(2),
    );
    await vi.waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:thumbnail-1"));
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
  });

  it("opens media first, then explicitly reuses the prompt and visible settings", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const tile = wrapper.get("[data-test='gallery-item']");
    expect(tile.attributes("aria-label")).toBe("Open storm clip.mp4 from Studio");
    await tile.trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    expect(wrapper.find("[data-test='gallery-viewer-video']").exists()).toBe(true);
    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/storm%20clip.mp4", {
      target,
      cacheKey: "studio-id",
      allowLegacyBlob: false,
    });

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
      "page",
    );
    expect(wrapper.get("#mobile-prompt").element).toHaveProperty("value", print.metadata.prompt);
    expect(fieldControl("Negative prompt").element).toHaveProperty("value", "calm water");
    expect(wrapper.get("[data-orientation='landscape']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-aspect='3:2']").attributes("aria-pressed")).toBe("true");
    expect(
      (wrapper.get("[data-test='mobile-resolution-tier']").element as HTMLSelectElement).value,
    ).toBe("3:2 · 768×512");
    expect(fieldControl("Format").element).toHaveProperty("value", "mp4");
    expect(fieldControl("Frames").element).toHaveProperty("value", "121");
    expect(fieldControl("FPS").element).toHaveProperty("value", "30");
  });

  it("uses a still gallery print as the selected model's source image", async () => {
    const still = { ...print, filename: "source print.png", format: "png" as const };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["source bytes"], { type: "image/png" })),
    } as Response);

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-use-source']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
        "page",
      ),
    );

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/source%20print.png");
    expect(wrapper.get("[data-test='mobile-source-preview']").attributes("alt")).toBe(
      "source print.png",
    );
  });

  it("rejects an oversized gallery source before reading or base64 expansion", async () => {
    const still = { ...print, filename: "huge source.png", format: "png" as const };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    const readBlob = vi.fn(() => Promise.resolve(new Blob(["should not be read"])));
    apiFetchTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(
        path.includes("/api/gallery/image/")
          ? ({
              headers: new Headers({ "content-length": String(46 * 1024 * 1024) }),
              blob: readBlob,
            } as unknown as Response)
          : ({ blob: () => Promise.resolve(new Blob(["thumbnail"])) } as Response),
      ),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-use-source']").trigger("click");
    await flushPromises();

    expect(readBlob).not.toHaveBeenCalled();
    expect(wrapper.get(".gallery-viewer-reuse-error").text()).toContain(
      "Combined generation media must be 45 MiB or smaller on iPhone",
    );
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("keeps the viewer open when the print host's models cannot be loaded", async () => {
    const remoteTarget = { baseUrl: "http://remote.tailnet.ts.net:7680", apiKey: "secret" };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "remote-id",
          name: "Remote",
          baseUrl: remoteTarget.baseUrl,
          hostname: "remote",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/gallery") {
        return Promise.resolve(baseUrl === remoteTarget.baseUrl ? [print] : []);
      }
      if (baseUrl === remoteTarget.baseUrl) return Promise.reject(new Error("offline"));
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    expect(wrapper.get("[role='alert']").text()).toContain("Couldn’t load models from Remote");
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
  });

  it("reloads model ownership after removing the active host before reuse", async () => {
    const remoteTarget = { baseUrl: "http://remote.tailnet.ts.net:7680", apiKey: "secret" };
    const studioModel: ModelEntry = {
      ...model,
      name: "flux:studio-only",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "remote-id",
          name: "Remote",
          baseUrl: remoteTarget.baseUrl,
          hostname: "remote",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: baseUrl === remoteTarget.baseUrl ? "remote" : "studio",
        });
      }
      if (path === "/api/models") {
        return Promise.resolve(baseUrl === remoteTarget.baseUrl ? [model] : [studioModel]);
      }
      if (path === "/api/gallery") {
        return Promise.resolve(
          baseUrl === remoteTarget.baseUrl
            ? [{ ...print, metadata: { ...print.metadata, frames: 97 } }]
            : [],
        );
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await vi.waitFor(() =>
      expect(fieldControl("Model").element).toHaveProperty("value", studioModel.name),
    );

    const hostsTab = wrapper
      .findAll("button.mobile-tab")
      .find((button) => button.text() === "Hosts");
    if (!hostsTab) throw new Error("Missing Hosts tab");
    await hostsTab.trigger("click");
    const studioRow = wrapper
      .findAll(".host-row")
      .find((row) => row.find(".host-name").text() === "Studio");
    if (!studioRow) throw new Error("Missing Studio host row");
    await studioRow.get("[data-test='mobile-host-row']").trigger("click");
    const forget = wrapper.get("[data-test='host-detail-forget']");
    await forget.trigger("click");
    expect(forget.text()).toContain("Forget Studio?");
    await forget.trigger("click");
    await vi.waitFor(() => expect(apiJsonTo).toHaveBeenCalledWith(remoteTarget, "/api/models"));

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(fieldControl("Model").element).toHaveProperty("value", model.name);
    expect(wrapper.get(".status-line").text()).toBe("Prompt settings restored");
    const developButton = wrapper
      .findAll("button")
      .find((button) => button.text() === "Develop print");
    expect(developButton?.attributes("disabled")).toBeUndefined();
  });
});

describe("MobileApp host and catalog coordination", () => {
  const remoteTarget = { baseUrl: "http://render.tailnet.ts.net:7680", apiKey: "secret" };

  function installTwoHosts(): void {
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "render-id",
          name: "Render Box",
          baseUrl: remoteTarget.baseUrl,
          hostname: "render",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
  }

  it("keeps newer host probe results and aborts superseded and unmounted probes", async () => {
    vi.useFakeTimers();
    try {
      installTwoHosts();
      const remoteProbes: Array<{
        resolve: (value: ServerStatus) => void;
        reject: (reason: Error) => void;
        signal: AbortSignal | undefined;
      }> = [];
      apiJsonTo.mockImplementation(
        (requestTarget: unknown, path: string, init?: { signal?: AbortSignal }) => {
          const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
          if (path === "/api/models") return Promise.resolve([model]);
          if (path === "/api/status" && baseUrl === target.baseUrl) return Promise.resolve(status);
          if (path === "/api/status") {
            return new Promise<ServerStatus>((resolve, reject) => {
              remoteProbes.push({ resolve, reject, signal: init?.signal });
            });
          }
          return Promise.reject(new Error(`Unexpected API path: ${path}`));
        },
      );

      wrapper = mountMobileApp();
      await flushPromises();
      expect(remoteProbes).toHaveLength(1);

      await vi.advanceTimersByTimeAsync(10_000);
      expect(remoteProbes).toHaveLength(2);
      expect(remoteProbes[0]?.signal?.aborted).toBe(true);

      remoteProbes[1]?.resolve({ ...status, version: "0.19.0", hostname: "render" });
      await flushPromises();
      remoteProbes[0]?.reject(new Error("stale timeout"));
      await flushPromises();

      await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
      const remoteRow = wrapper
        .findAll(".host-row")
        .find((row) => row.find(".host-name").text() === "Render Box");
      expect(remoteRow?.text()).toContain("v0.19.0");
      expect(remoteRow?.text()).not.toContain("offline");

      await vi.advanceTimersByTimeAsync(10_000);
      expect(remoteProbes).toHaveLength(3);
      const unmountedSignal = remoteProbes[2]?.signal;
      wrapper.unmount();
      wrapper = null;
      expect(unmountedSignal?.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens Catalog on the viewed host without changing the generation host", async () => {
    installTwoHosts();
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: baseUrl === remoteTarget.baseUrl ? "render" : "studio",
        });
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve({ json: () => Promise.resolve([model]) } as Response);
      }
      return Promise.resolve({ blob: () => Promise.resolve(new Blob()) } as Response);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    const remoteRow = wrapper
      .findAll(".host-row")
      .find((row) => row.find(".host-name").text() === "Render Box");
    if (!remoteRow) throw new Error("Missing Render Box host row");
    await remoteRow.get("[data-test='mobile-host-row']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='host-detail-catalog']").trigger("click");
    await flushPromises();

    expect(wrapper.get("select[aria-label='Catalog host']").element).toHaveProperty(
      "value",
      "render-id",
    );
    expect(wrapper.get(".mobile-header .host-chip").text()).toBe("Studio");
    expect(wrapper.get("[data-test='mobile-tab-catalog']").attributes("aria-current")).toBe("page");
  });

  it("keeps the catalog download stream alive off-tab and refreshes Generate models", async () => {
    const pulledModel = { ...model, name: "ltx2:new-download" };
    let downloadFinished = false;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") {
        return Promise.resolve(downloadFinished ? [model, pulledModel] : [model]);
      }
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve({ json: () => Promise.resolve([model]) } as Response);
      }
      return Promise.resolve({ blob: () => Promise.resolve(new Blob()) } as Response);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-catalog']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams.find((stream) => stream.path === "/api/downloads/stream");
    if (!downloadStream) throw new Error("Catalog download stream did not open");
    downloadStream.options.onOpen?.();

    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    expect(downloadStream.options.signal.aborted).toBe(false);
    downloadFinished = true;
    downloadStream.options.onEvent(
      "message",
      JSON.stringify({ type: "job_done", id: "download-1", model: pulledModel.name }),
    );
    await flushPromises();

    const options = fieldControl("Model")
      .findAll("option")
      .map((option) => option.text());
    expect(options).toContain(pulledModel.name);
  });
});
