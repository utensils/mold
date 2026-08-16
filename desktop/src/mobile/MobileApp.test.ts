import { flushPromises, mount, type DOMWrapper, type VueWrapper } from "@vue/test-utils";
import { createPinia, type Pinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { GalleryImage, ModelEntry, ServerStatus } from "../lib/api/types";
import { applyModelDefaults, type GenerateForm } from "../lib/generateForm";

const {
  invoke,
  apiFetchTo,
  apiJsonTo,
  sseStream,
  streamableMediaUrl,
  evictMedia,
  applySourceFitPreprocess,
  expandPrompt,
  remixPrompt,
  startCatalogDownload,
  previewChainPlacement,
  previewGenerationPlacement,
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
  checkBarcodeScannerPermissions,
  requestBarcodeScannerPermissions,
  cancelBarcodeScanner,
  scanPairingQr,
  claimPairingSession,
  getCurrentDeepLinks,
  onOpenDeepLinks,
  unlistenDeepLinks,
} = vi.hoisted(() => ({
  invoke: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn(),
  streamableMediaUrl: vi.fn(),
  evictMedia: vi.fn(),
  applySourceFitPreprocess: vi.fn(),
  expandPrompt: vi.fn(),
  remixPrompt: vi.fn(),
  startCatalogDownload: vi.fn(),
  previewChainPlacement: vi.fn(),
  previewGenerationPlacement: vi.fn(),
  persistGenerationSourceMedia: vi.fn(),
  restoreGenerationSourceMedia: vi.fn(),
  checkBarcodeScannerPermissions: vi.fn(),
  requestBarcodeScannerPermissions: vi.fn(),
  cancelBarcodeScanner: vi.fn(),
  scanPairingQr: vi.fn(),
  claimPairingSession: vi.fn(),
  getCurrentDeepLinks: vi.fn(),
  onOpenDeepLinks: vi.fn(),
  unlistenDeepLinks: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("@tauri-apps/plugin-barcode-scanner", () => ({
  Format: { QRCode: "QRCode" },
  checkPermissions: checkBarcodeScannerPermissions,
  requestPermissions: requestBarcodeScannerPermissions,
  cancel: cancelBarcodeScanner,
  scan: scanPairingQr,
}));
vi.mock("@tauri-apps/plugin-deep-link", () => ({
  getCurrent: getCurrentDeepLinks,
  onOpenUrl: onOpenDeepLinks,
}));
vi.mock("@studio/api/pairing", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/pairing")>()),
  claimPairingSession,
}));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));
vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));
vi.mock("../lib/sourceFitPreprocess", () => ({ applySourceFitPreprocess }));
vi.mock("../lib/api/expand", () => ({ expandPrompt }));
vi.mock("../lib/api/remix", () => ({ remixPrompt }));
vi.mock("../lib/api/catalog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/catalog")>()),
  startCatalogDownload,
}));
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl,
  evictMedia,
}));
vi.mock("@studio/api/generationPlacement", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationPlacement")>()),
  previewChainPlacement,
  previewGenerationPlacement,
}));
vi.mock("@studio/lib/generationSourceMedia", () => ({
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
}));

function plannedPlacement() {
  return {
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
  };
}
// Keep the real reconciliation logic but collapse its re-attach poll interval
// so tests never wait out the production cadence.
vi.mock("./mobileGenerationRecovery", async (importOriginal) => {
  const original = await importOriginal<typeof import("./mobileGenerationRecovery")>();
  return {
    ...original,
    reconcileInterruptedGenerationJobs: (
      jobs: Parameters<typeof original.reconcileInterruptedGenerationJobs>[0],
      options: Parameters<typeof original.reconcileInterruptedGenerationJobs>[1],
    ) => original.reconcileInterruptedGenerationJobs(jobs, { ...options, pollIntervalMs: 0 }),
  };
});

import MobileApp from "./MobileApp.vue";
import MobileImagePickerSheet from "./MobileImagePickerSheet.vue";
import MobileLoraControls from "./MobileLoraControls.vue";
import MobileSourceControls from "./MobileSourceControls.vue";
import MobileTemplates from "./MobileTemplates.vue";
import { useMobileDownloadsStore } from "./mobileDownloads";
import { ApiError } from "../lib/api/client";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";

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
// The kit's natural-language expansion directive for the cinematic chip
// (`styleHint("cinematic")`) — pinned verbatim so the wire contract is explicit.
const cinematicHint =
  "Cinematic look — cinematic film still, cinematic lighting, anamorphic, dramatic mood, subtle film grain";
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

class FakeIntersectionObserver {
  static instances: FakeIntersectionObserver[] = [];
  targets: Element[] = [];

  constructor(private callback: IntersectionObserverCallback) {
    FakeIntersectionObserver.instances.push(this);
  }

  observe(target: Element): void {
    this.targets.push(target);
  }

  disconnect(): void {
    this.targets = [];
  }

  intersect(isIntersecting = true): void {
    this.callback(
      this.targets.map((target) => ({ isIntersecting, target }) as IntersectionObserverEntry),
      this as unknown as IntersectionObserver,
    );
  }
}

function scrollToGallerySentinel(): void {
  for (const observer of FakeIntersectionObserver.instances) observer.intersect();
}

function mountMobileApp(pinia: Pinia = createPinia()): VueWrapper {
  return mount(MobileApp, {
    attachTo: document.body,
    // DevelopCanvas paints on a real 2D context happy-dom doesn't provide.
    global: { plugins: [pinia], stubs: { DevelopCanvas: true } },
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
  FakeIntersectionObserver.instances = [];
  (globalThis as { IntersectionObserver?: unknown }).IntersectionObserver =
    FakeIntersectionObserver;
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
    if (path === "/api/activity")
      return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
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
  remixPrompt.mockReset().mockImplementation((request) =>
    Promise.resolve({
      source_prompt: request.source_prompt,
      ...(request.root_prompt ? { root_prompt: request.root_prompt } : {}),
      source_kind: request.source_kind,
      variants: Array.from({ length: request.variations ?? 3 }, (_, index) => ({
        prompt: `${request.source_prompt} · remix ${index + 1}`,
        dimensions: [request.dimensions[index % request.dimensions.length]],
      })),
    }),
  );
  startCatalogDownload.mockReset().mockResolvedValue("expansion-job");
  previewChainPlacement.mockReset().mockResolvedValue(plannedPlacement());
  previewGenerationPlacement.mockReset().mockResolvedValue(plannedPlacement());
  persistGenerationSourceMedia.mockReset().mockResolvedValue(null);
  restoreGenerationSourceMedia.mockReset().mockResolvedValue(null);
  checkBarcodeScannerPermissions.mockReset().mockResolvedValue("granted");
  requestBarcodeScannerPermissions.mockReset();
  cancelBarcodeScanner.mockReset().mockResolvedValue(undefined);
  scanPairingQr.mockReset();
  claimPairingSession.mockReset();
  getCurrentDeepLinks.mockReset().mockResolvedValue(null);
  unlistenDeepLinks.mockReset();
  onOpenDeepLinks.mockReset().mockResolvedValue(unlistenDeepLinks);
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
  delete (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
});

describe("MobileApp sequence generation", () => {
  it("removes capability-restricted models from the picker before submission", async () => {
    const restrictedModel: ModelEntry = {
      ...model,
      name: "AutoencoderKLMiniMaxH3",
      family: "minimax-h3",
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model, restrictedModel]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          gallery: { can_delete: true },
          model_access: {
            restrictions: [
              {
                code: "minimax_h3_authorization_required",
                family: "minimax-h3",
                message: "MiniMax H3 is not activated.",
                license_url: "https://example.test/license",
                authorization_url: "https://example.test/authorize",
              },
            ],
          },
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();

    const values = fieldControl("Model")
      .findAll("option")
      .map((option) => option.attributes("value"));
    expect(values).toContain(model.name);
    expect(values).not.toContain(restrictedModel.name);
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
  });

  it("shows active work started by another client with host attribution", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: "mobile-host",
          observed_at_unix_ms: 10,
          items: [
            {
              id: "foreign-job",
              kind: "generation",
              phase: "running",
              model: "flux-dev:q4",
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: false,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();
    const activity = wrapper.get("[data-test='shared-live-activity']");
    expect(activity.text()).toContain("Studio");
    expect(activity.text()).toContain("flux-dev:q4");
    expect(localStorage.getItem("mold.mobile.live-activity.v1")).not.toContain(target.apiKey);
  });

  it("queues a durable two-clip sequence on the selected Keychain-authenticated host", async () => {
    const sequenceModel = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      default_steps: 7,
      default_guidance: 1,
    };
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    apiJsonTo.mockImplementation((_target: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.resolve({
          model: sequenceModel.name,
          frames_per_clip_cap: 97,
          frames_per_clip_recommended: 97,
          max_stages: 8,
          max_total_frames: 777,
          fade_frames_max: 32,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "bf16",
          supports_audio: false,
        });
      }
      if (path === "/api/chain-jobs" && init?.method === "POST") {
        return Promise.resolve({ job_id: "sequence-job-1" });
      }
      if (path === "/api/chain-jobs/sequence-job-1") {
        return new Promise(() => {});
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    let finishPreview!: (value: Awaited<ReturnType<typeof previewChainPlacement>>) => void;
    previewChainPlacement.mockReturnValueOnce(
      new Promise((resolve) => {
        finishPreview = resolve;
      }),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    const prompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("A paper boat crosses a moonlit pond");
    await prompts[1]!.setValue("Fireflies gather as the sky brightens");
    const sequenceForm = wrapper.getComponent({ name: "MobileSequenceComposer" }).props("form") as {
      strength: number;
    };
    const tappedStrength = sequenceForm.strength;
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();
    sequenceForm.strength = 0.12;
    await prompts[0]!.setValue("A later edit that belongs to the next submission");
    finishPreview({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "unsupported",
      candidate: null,
    });
    await flushPromises();

    expect(previewChainPlacement).toHaveBeenCalledWith(target, expect.anything());
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/chain-jobs",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("A paper boat crosses a moonlit pond"),
      }),
    );
    const createCall = apiJsonTo.mock.calls.find(
      (call) => call[1] === "/api/chain-jobs" && (call[2] as RequestInit)?.method === "POST",
    );
    const request = JSON.parse(String((createCall?.[2] as RequestInit)?.body));
    expect(request.strength).toBe(tappedStrength);
    expect(request.stages[0].prompt).toBe("A paper boat crosses a moonlit pond");
    const recovery = localStorage.getItem("mold.mobile.sequence-job.v1");
    expect(JSON.parse(recovery ?? "null")).toEqual({
      hostId: "studio-id",
      baseUrl: target.baseUrl,
      instanceId: "studio-id",
      jobId: "sequence-job-1",
    });
    expect(recovery).not.toContain(target.apiKey);
    expect(wrapper.get("[data-test='mobile-sequence-job']").text()).toContain("queued");
  });

  it("refuses recovery when a saved server identity cannot be verified", async () => {
    const sequenceModel = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
    };
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    localStorage.setItem(
      "mold.mobile.sequence-job.v1",
      JSON.stringify({
        hostId: "studio-id",
        baseUrl: target.baseUrl,
        instanceId: "studio-id",
        jobId: "saved-sequence",
      }),
    );
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") {
        return Promise.resolve({ ...status, instance_id: undefined });
      }
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.resolve({
          model: sequenceModel.name,
          frames_per_clip_cap: 97,
          frames_per_clip_recommended: 97,
          max_stages: 8,
          max_total_frames: 777,
          fade_frames_max: 32,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "q8",
          supports_audio: true,
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();

    expect(localStorage.getItem("mold.mobile.sequence-job.v1")).toBeNull();
    expect(wrapper.text()).toContain(
      "The exact machine for this saved sequence is no longer available.",
    );
    expect(apiJsonTo).not.toHaveBeenCalledWith(expect.anything(), "/api/chain-jobs/saved-sequence");
  });
});

describe("MobileApp Output field", () => {
  const sequenceModel: ModelEntry = {
    ...model,
    name: "ltx-video-0.9.8-2b-dev:bf16",
    family: "ltx-video",
    default_steps: 7,
    default_guidance: 1,
    default_width: 704,
    default_height: 480,
    default_frames: 25,
    default_fps: 30,
  };

  function installModels(entries: ModelEntry[]): void {
    apiJsonTo.mockImplementation((_target: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve(entries);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.resolve({
          model: sequenceModel.name,
          frames_per_clip_cap: 97,
          frames_per_clip_recommended: 97,
          max_stages: 8,
          max_total_frames: 777,
          fade_frames_max: 32,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "bf16",
          supports_audio: false,
          supports_sequence: true,
        });
      }
      if (path === "/api/chain-jobs" && init?.method === "POST") {
        return Promise.resolve({ job_id: "sequence-job-1" });
      }
      if (path.startsWith("/api/chain-jobs/")) return new Promise(() => {});
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
  }

  function outputSegment(label: string): DOMWrapper<Element> {
    const button = wrapper
      ?.get("[data-test='mobile-output-mode']")
      .findAll("button")
      .find((candidate) => candidate.text() === label);
    if (!button) throw new Error(`Missing ${label} output segment`);
    return button;
  }

  async function composeTwoClips(): Promise<void> {
    const prompts = wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("a paper boat");
    await prompts[1]!.setValue("fireflies gather");
  }

  it("replaces the Single | Sequence mode pair with an Output field above the model", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    // The old radiogroup pinned above the form is gone for good.
    expect(wrapper.find("[data-test='mobile-create-mode']").exists()).toBe(false);

    const output = wrapper.get("[data-test='mobile-output-mode']");
    const modelField = wrapper
      .findAll("label.field")
      .find((field) => field.find("span").text() === "Model")!;
    expect(
      output.element.compareDocumentPosition(modelField.element) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(output.text()).toContain("One shot");
    expect(output.text()).toContain("Sequence");
    expect(wrapper.find("[data-test='mobile-sequence-composer']").exists()).toBe(false);
  });

  it("keeps the one-shot Develop action outside the scrolling form", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    const action = wrapper.get("[data-test='mobile-create-action']");
    expect(action.find("[data-test='mobile-develop-button']").exists()).toBe(true);
    expect(
      wrapper.get(".mobile-content").find("[data-test='mobile-develop-button']").exists(),
    ).toBe(false);

    await outputSegment("Sequence").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-create-action']").exists()).toBe(false);
  });

  it("keeps a corrective explanation beside a disabled persistent Develop action", async () => {
    installModels([]);
    wrapper = mountMobileApp();
    await flushPromises();

    const action = wrapper.get("[data-test='mobile-create-action']");
    expect(action.get("[data-test='mobile-develop-button']").attributes("disabled")).toBeDefined();
    expect(action.get("[data-test='mobile-develop-blocker']").text()).toContain(
      "Choose an installed model before generating.",
    );
  });

  it("migrates the legacy create-mode key into the shared draft and retires it", async () => {
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-sequence-composer']").exists()).toBe(true);
    expect(outputSegment("Sequence").attributes("aria-checked")).toBe("true");
    expect(localStorage.getItem("mold.mobile.create-mode.v1")).toBeNull();
  });

  it("keeps One shot and Sequence prompts isolated while switching", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a paper boat crosses a moonlit pond");
    await outputSegment("Sequence").trigger("click");
    await flushPromises();

    const clipPrompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    expect((clipPrompts[0]!.element as HTMLTextAreaElement).value).toBe("");

    await clipPrompts[0]!.setValue("a paper boat under fireflies");
    await outputSegment("One shot").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a paper boat crosses a moonlit pond",
    );

    await outputSegment("Sequence").trigger("click");
    await flushPromises();
    expect(
      (
        wrapper.findAll("[data-test='mobile-sequence-clip'] textarea")[0]!
          .element as HTMLTextAreaElement
      ).value,
    ).toBe("a paper boat under fireflies");
  });

  it("narrows the picker to chain-capable models and restores the single pick on the way back", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    expect((fieldControl("Model").element as HTMLSelectElement).value).toBe(model.name);

    await outputSegment("Sequence").trigger("click");
    await flushPromises();

    const picker = fieldControl("Video model");
    const options = picker.findAll("option").map((option) => option.attributes("value"));
    expect(options).toEqual([sequenceModel.name]);
    expect((picker.element as HTMLSelectElement).value).toBe(sequenceModel.name);

    await outputSegment("One shot").trigger("click");
    await flushPromises();
    expect((fieldControl("Model").element as HTMLSelectElement).value).toBe(model.name);
  });

  it("guides to Video + Models in Discover when no chain-capable model is installed", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.reject(new Error("not a chain model"));
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await outputSegment("Sequence").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-sequence-composer']").exists()).toBe(false);
    const empty = wrapper.get("[data-test='mobile-sequence-empty']");
    expect(empty.text()).toContain("Sequences need a video model");

    // Browsing must LAND on the filtered Discover shelf, not just switch tabs.
    await empty.get("[data-test='mobile-sequence-browse']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-catalog']").attributes("aria-current")).toBe("page");
    const media = wrapper
      .get(".mobile-catalog-media")
      .findAll("button")
      .find((button) => button.text() === "Video")!;
    expect(media.attributes("aria-pressed")).toBe("true");
    const kinds = wrapper.get("[data-test='mobile-catalog-kind-chips']");
    const checkpoints = kinds.findAll("button").find((button) => button.text() === "Models")!;
    expect(checkpoints.attributes("aria-pressed")).toBe("true");
  });

  it("reads the selected model's fps into the shared params, not a generic 24", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await outputSegment("Sequence").trigger("click");
    await flushPromises();

    expect(
      (wrapper.get("[data-test='mobile-sequence-fps'] input").element as HTMLInputElement).value,
    ).toBe("30");
    expect(wrapper.get("[data-test='mobile-sequence-duration']").text()).toContain("@ 30fps");
  });

  it("submits the sequence with the form's live shared params, not composer copies", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await outputSegment("Sequence").trigger("click");
    await flushPromises();

    await composeTwoClips();
    // The shared params are the ONLY copy — editing them here is exactly what
    // the outgoing chain request must carry.
    await fieldControl("Steps").setValue(11);
    await fieldControl("Guidance").setValue(2.5);
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const post = apiJsonTo.mock.calls.find(
      (call: unknown[]) => call[1] === "/api/chain-jobs" && (call[2] as RequestInit)?.method,
    );
    const body = JSON.parse((post?.[2] as RequestInit).body as string) as Record<string, unknown>;
    expect(body.model).toBe(sequenceModel.name);
    expect(body.steps).toBe(11);
    expect(body.guidance).toBe(2.5);
    expect(body.width).toBe(sequenceModel.default_width);
    expect(body.height).toBe(sequenceModel.default_height);
    // The model's own fps, not the generic 24-fps form fallback.
    expect(body.fps).toBe(sequenceModel.default_fps);
    // LTX-Video carries no motion tail, so its seams are plain joins.
    expect(body.motion_tail_frames).toBe(0);
    expect(body.stages).toEqual([
      { prompt: "a paper boat", frames: 25, transition: "smooth" },
      { prompt: "fireflies gather", frames: 25, transition: "smooth" },
    ]);

    const recovery = localStorage.getItem("mold.mobile.sequence-job.v1");
    expect(JSON.parse(recovery ?? "null")).toEqual({
      hostId: "studio-id",
      baseUrl: target.baseUrl,
      instanceId: "studio-id",
      jobId: "sequence-job-1",
    });
    expect(recovery).not.toContain(target.apiKey);
  });

  it("watches the durable job over SSE on the frozen route and shows it in the one queue", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await outputSegment("Sequence").trigger("click");
    await flushPromises();
    await composeTwoClips();
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const events = openStreams.find(
      (stream) => stream.path === "/api/chain-jobs/sequence-job-1/events",
    );
    if (!events) throw new Error("The sequence event stream never opened");
    expect((events.options as { target?: unknown }).target).toEqual(target);

    events.options.onEvent(
      "message",
      JSON.stringify({
        type: "snapshot",
        job: {
          id: "sequence-job-1",
          state: "running",
          model: sequenceModel.name,
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1_700_000_000_000,
          updated_at_unix_ms: 1_700_000_000_000,
          error: null,
          stages: [
            { idx: 0, state: "running" },
            { idx: 1, state: "pending" },
          ],
        },
      }),
    );
    events.options.onEvent(
      "message",
      JSON.stringify({ type: "denoise_step", stage_idx: 0, step: 4, total: 8 }),
    );
    await flushPromises();

    const row = wrapper.get("[data-test='mobile-sequence-job']");
    expect(row.text()).toContain("2 clips");
    expect(row.text()).toContain("running");
    expect(row.text()).toContain("clip 1/2");
    expect(row.text()).toContain("50%");
    // Sequences and prints share ONE queue list.
    expect(wrapper.get("[data-test='mobile-generation-queue']").element.contains(row.element)).toBe(
      true,
    );

    await row.get("[data-test='mobile-sequence-cancel']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/chain-jobs/sequence-job-1/cancel", {
      method: "POST",
    });
  });

  it("offers Resume and Dismiss once the durable job settles", async () => {
    installModels([model, sequenceModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await outputSegment("Sequence").trigger("click");
    await flushPromises();
    await composeTwoClips();
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const events = openStreams.find(
      (stream) => stream.path === "/api/chain-jobs/sequence-job-1/events",
    )!;
    events.options.onEvent(
      "message",
      JSON.stringify({
        type: "snapshot",
        job: {
          id: "sequence-job-1",
          state: "interrupted",
          model: sequenceModel.name,
          stage_count: 2,
          current_stage: 1,
          created_at_unix_ms: 1_700_000_000_000,
          updated_at_unix_ms: 1_700_000_000_000,
          error: "CUDA_ERROR_OUT_OF_MEMORY",
          stages: [
            { idx: 0, state: "completed" },
            { idx: 1, state: "failed" },
          ],
        },
      }),
    );
    await flushPromises();

    const row = wrapper.get("[data-test='mobile-sequence-job']");
    expect(row.text()).toContain("needs more GPU memory");
    const failure = row.get("[data-test='mobile-sequence-error-disclosure']");
    expect(failure.attributes("aria-expanded")).toBe("false");
    expect(failure.attributes("aria-label")).toBeUndefined();
    await failure.trigger("click");
    expect(failure.attributes("aria-expanded")).toBe("true");
    expect(failure.classes()).toContain("mobile-sequence-row-error--expanded");
    expect(row.find("[data-test='mobile-sequence-cancel']").exists()).toBe(false);
    expect(row.find("[data-test='mobile-sequence-dismiss']").exists()).toBe(true);
    // The row survives for its actions, but a settled job is not active work.
    expect(wrapper.get("[data-test='mobile-queue-count']").text()).toBe("0 active");
    // A settled job has nothing left to stream.
    expect(events.options.signal.aborted).toBe(true);
    expect(localStorage.getItem("mold.mobile.sequence-job.v1")).toBeNull();

    await row.get("[data-test='mobile-sequence-resume']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/chain-jobs/sequence-job-1/resume", {
      method: "POST",
    });
    // Resuming re-attaches on the SAME frozen route and re-arms recovery.
    const resumed = wrapper.get("[data-test='mobile-sequence-job']");
    expect(resumed.find("[data-test='mobile-sequence-cancel']").exists()).toBe(true);
    expect(JSON.parse(localStorage.getItem("mold.mobile.sequence-job.v1") ?? "null")).toEqual({
      hostId: "studio-id",
      baseUrl: target.baseUrl,
      instanceId: "studio-id",
      jobId: "sequence-job-1",
    });
    const resumedStream = openStreams
      .filter((stream) => stream.path === "/api/chain-jobs/sequence-job-1/events")
      .at(-1)!;
    expect(resumedStream).not.toBe(events);
    resumedStream.options.onEvent(
      "message",
      JSON.stringify({
        type: "snapshot",
        job: {
          id: "sequence-job-1",
          state: "cancelled",
          model: sequenceModel.name,
          stage_count: 2,
          current_stage: 1,
          created_at_unix_ms: 1_700_000_000_000,
          updated_at_unix_ms: 1_700_000_000_100,
          error: null,
          stages: [
            { idx: 0, state: "completed" },
            { idx: 1, state: "pending" },
          ],
        },
      }),
    );
    await flushPromises();

    await wrapper
      .get("[data-test='mobile-sequence-job'] [data-test='mobile-sequence-dismiss']")
      .trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-sequence-job']").exists()).toBe(false);
  });
});

describe("MobileApp generation queue", () => {
  async function submitPrompt(prompt: string): Promise<void> {
    await fieldControl("Prompt").setValue(prompt);
    await wrapper?.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  it("composes the chosen style preset into the outgoing prompt without mutating the textarea", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a red fox in snow");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    // The shared kit's cinematic preset templates the prompt into the look.
    expect(openStreams[0]?.options.body.prompt).toBe(
      "cinematic film still of a red fox in snow, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    // The composition is applied to a draft clone at submit — the live prompt stays the user's words.
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("a red fox in snow");
  });

  it("bakes a styled quick expansion, clears the chip, and restores it on Undo", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a lighthouse");
    await fieldControl("Negative prompt").setValue("text");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    // The chip travels as a natural-language directive, never a prompt suffix.
    expect(expandPrompt).toHaveBeenCalledWith(
      "a lighthouse",
      { variations: 1, modelFamily: model.family, task: "text-to-video", style: cinematicHint },
      target,
    );
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse · prepared 1",
    );
    // Bake-and-clear: the rewrite absorbed the look, so the chip resets — and
    // the curated negative moves into the form, its only remaining home.
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("None");
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe(
      "text, anime, cartoon, graphic, washed out",
    );

    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("a lighthouse");
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("Cinematic");
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe("text");
  });

  it("remixes the original idea into three reviewable variants and applies one without queueing", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse · prepared 1",
    );

    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();

    expect(remixPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        source_prompt: "a lighthouse",
        root_prompt: "a lighthouse",
        source_kind: "original",
        model_family: model.family,
        variations: 3,
        task: "text-to-video",
      }),
      target,
    );
    expect(wrapper.findAll(".mobile-remix-editor")).toHaveLength(3);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(0);

    await wrapper.get("input[aria-label='Select remix 2']").setValue(true);
    await wrapper.get("[data-test='mobile-remix-apply']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse · remix 2",
    );
    expect(openStreams).toHaveLength(0);

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams[0]?.options.target).toEqual(target);
    expect(openStreams[0]?.options.body.prompt_transform).toEqual({
      operation: "remix",
      root_prompt: "a lighthouse",
      source_prompt: "a lighthouse",
      source_kind: "original",
      task: "text-to-video",
      dimensions: expect.any(Array),
    });

    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse · prepared 1",
    );
  });

  it("turns multiple selected remixes into a frozen prepared batch", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a train in the desert");

    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();
    await wrapper.get("input[aria-label='Select remix 1']").setValue(true);
    await wrapper.get("input[aria-label='Select remix 3']").setValue(true);
    await wrapper.get("[data-test='mobile-remix-apply']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-remix-review']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-prepared-expansion']").text()).toContain(
      "Review 2 variations",
    );
    expect(
      wrapper
        .findAll(".mobile-prepared-editor")
        .map((editor) => (editor.element as HTMLTextAreaElement).value),
    ).toEqual(["a train in the desert · remix 1", "a train in the desert · remix 3"]);
    expect(openStreams).toHaveLength(0);

    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(2);
    expect(openStreams.map((stream) => stream.options.body.original_prompt)).toEqual([
      "a train in the desert",
      "a train in the desert",
    ]);
    expect(openStreams.map((stream) => stream.options.body.prompt_transform)).toEqual([
      expect.objectContaining({
        operation: "remix",
        source_prompt: "a train in the desert",
        source_kind: "direct",
        task: "text-to-video",
        dimensions: expect.any(Array),
      }),
      expect.objectContaining({
        operation: "remix",
        source_prompt: "a train in the desert",
        source_kind: "direct",
        task: "text-to-video",
        dimensions: expect.any(Array),
      }),
    ]);
  });

  it("keeps the root original_prompt for a prepared batch remixed from current text", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-remix-options']").trigger("click");
    await wrapper.get("input[name='mobile-remix-source'][value='current']").setValue(true);
    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();
    await wrapper.get("input[aria-label='Select remix 1']").setValue(true);
    await wrapper.get("input[aria-label='Select remix 3']").setValue(true);
    await wrapper.get("[data-test='mobile-remix-apply']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(openStreams.map((stream) => stream.options.body.original_prompt)).toEqual([
      "a lighthouse",
      "a lighthouse",
    ]);
    expect(openStreams[0]?.options.body.prompt_transform).toMatchObject({
      root_prompt: "a lighthouse",
      source_prompt: "a lighthouse · prepared 1",
      source_kind: "current",
    });
  });

  it("carries a finished quick Expand to a newly selected host without stale recovery", async () => {
    const renderTarget = {
      baseUrl: "http://render.tailnet.ts.net:7680",
      apiKey: "render-secret",
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          instanceId: "studio-id",
        },
        {
          id: "render-id",
          name: "Render",
          baseUrl: renderTarget.baseUrl,
          hostname: "render",
          instanceId: "render-id",
        },
      ]),
    );
    invoke.mockImplementation((command: string, args?: { hostId?: string }) =>
      Promise.resolve(
        command === "keychain_get_api_key"
          ? args?.hostId === "render-id"
            ? renderTarget.apiKey
            : target.apiKey
          : null,
      ),
    );
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("portable lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-quick-expansion-stale']").exists()).toBe(false);
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "portable lighthouse · prepared 1",
    );
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams[0]?.options.target).toEqual(renderTarget);
  });

  it("carries an applied single Remix across hosts but still stales semantic edits", async () => {
    const renderTarget = {
      baseUrl: "http://render.tailnet.ts.net:7680",
      apiKey: "render-secret",
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          instanceId: "studio-id",
        },
        {
          id: "render-id",
          name: "Render",
          baseUrl: renderTarget.baseUrl,
          hostname: "render",
          instanceId: "render-id",
        },
      ]),
    );
    invoke.mockImplementation((command: string, args?: { hostId?: string }) =>
      Promise.resolve(
        command === "keychain_get_api_key"
          ? args?.hostId === "render-id"
            ? renderTarget.apiKey
            : target.apiKey
          : null,
      ),
    );
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("portable train");
    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();
    await wrapper.get("input[aria-label='Select remix 1']").setValue(true);
    await wrapper.get("[data-test='mobile-remix-apply']").trigger("click");
    await wrapper.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-quick-expansion-stale']").exists()).toBe(false);
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "portable train · remix 1",
    );

    await fieldControl("Prompt").setValue("user edited the remix");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-quick-expansion-stale']").text()).toContain(
      "Expanded prompt changed",
    );
  });

  it("keeps a reviewed multi-variation batch pinned when host selection changes", async () => {
    const renderTarget = {
      baseUrl: "http://render.tailnet.ts.net:7680",
      apiKey: "render-secret",
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        { id: "studio-id", name: "Studio", baseUrl: target.baseUrl, instanceId: "studio-id" },
        {
          id: "render-id",
          name: "Render",
          baseUrl: renderTarget.baseUrl,
          instanceId: "render-id",
        },
      ]),
    );
    invoke.mockImplementation((command: string, args?: { hostId?: string }) =>
      Promise.resolve(
        command === "keychain_get_api_key"
          ? args?.hostId === "render-id"
            ? renderTarget.apiKey
            : target.apiKey
          : null,
      ),
    );
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two pinned storms");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-prepared-expansion']").text()).toContain(
      "Host selection changed from Studio to Render",
    );
  });

  it("recovers a stale quick expansion with readable current-model actions", async () => {
    const catalogModel: ModelEntry = {
      ...model,
      name: "cv:1759168",
      family: "sdxl",
      display_name: "Juggernaut XL - Ragnarok",
      description: "Juggernaut XL - Ragnarok by RunDiffusion",
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model, catalogModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await fieldControl("Model").setValue(catalogModel.name);
    await flushPromises();

    const stale = wrapper.get("[data-test='mobile-quick-expansion-stale']");
    expect(stale.text()).toContain("Juggernaut XL - Ragnarok");
    expect(stale.text()).not.toContain("cv:1759168");
    expect(stale.find("[data-test='mobile-reexpand-and-develop']").exists()).toBe(true);
    expect(stale.find("[data-test='mobile-develop-expanded-anyway']").exists()).toBe(true);
    const action = wrapper.get("[data-test='mobile-create-action']");
    expect(action.get("[data-test='mobile-develop-button']").attributes("disabled")).toBeDefined();
    expect(action.get("[data-test='mobile-develop-blocker']").text()).toContain(
      "Use a recovery action above.",
    );

    await stale.get("[data-test='mobile-develop-expanded-anyway']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.options.body).toMatchObject({
      model: catalogModel.name,
      prompt: "a lighthouse · prepared 1",
      original_prompt: "a lighthouse",
    });
  });

  it("merges the preset negative when a styled quick expansion clears the chip", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a lighthouse");
    await fieldControl("Negative prompt").setValue("text");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    // Bake-and-clear drops the chip, so the curated negative has to land in the
    // form now — the submit-time merge no longer sees a preset.
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("None");
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe(
      "text, anime, cartoon, graphic, washed out",
    );

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(1);
    // …and exactly once — the cleared chip can't merge it a second time.
    expect(openStreams[0]?.options.body.negative_prompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
  });

  it("keeps the chip frozen on a prepared batch and names style drift as stale work", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three storm studies");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "three storm studies",
      { variations: 3, modelFamily: model.family, task: "text-to-video", style: cinematicHint },
      target,
    );
    // Prepared keeps the chip — it is the frozen-style indicator…
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("Cinematic");

    // …so style drift is specifically named stale work.
    await wrapper.get("[data-test='mobile-style-anime']").trigger("click");
    expect(wrapper.get(".mobile-prepared-stale").text()).toContain(
      "Style changed from Cinematic to Anime.",
    );
    await wrapper.get("[data-test='mobile-style-anime']").trigger("click");
    expect(wrapper.get(".mobile-prepared-stale").text()).toContain(
      "Style Cinematic was removed after these variations were prepared.",
    );
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    expect(wrapper.find(".mobile-prepared-stale").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    // Reviewed prompts ship verbatim — the prepared submit path never suffixes.
    expect(openStreams.map((stream) => stream.options.body.prompt)).toEqual([
      "three storm studies · prepared 1",
      "three storm studies · prepared 2",
      "three storm studies · prepared 3",
    ]);
  });

  it("clears the chip when a prepared pair collapses into the composer", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("storm pair");
    await fieldControl("Negative prompt").setValue("text");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    await wrapper.findAll("[data-test='mobile-prepared-remove']")[0]!.trigger("click");
    await wrapper.get("[data-test='mobile-confirm-collapse']").trigger("click");
    await flushPromises();

    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "storm pair · prepared 2",
    );
    // The surviving reviewed text absorbed the frozen style — same bake-and-clear.
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("None");
    // …so the frozen style's negative moves into the form with it.
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe(
      "text, anime, cartoon, graphic, washed out",
    );

    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    await flushPromises();
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("storm pair");
    // Undo restores the source prompt; the chip stays cleared because only a
    // quick-apply snapshot re-arms it (mirrors desktop).
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("None");
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe("text");
  });

  it("re-requests a recovered expansion pull with the frozen style and still bakes on apply", async () => {
    expandPrompt
      .mockRejectedValueOnce(
        new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
      )
      .mockResolvedValueOnce({ expanded: ["a lighthouse after the storm"] });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("lighthouse");
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams.find((stream) => stream.path === "/api/downloads/stream");
    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({ type: "snapshot", listing: { active_jobs: [], queued: [], history: [] } }),
    );
    await flushPromises();
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
      JSON.stringify({ type: "started", id: "expansion-job", files_total: 2, bytes_total: 1_000 }),
    );
    downloadStream?.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "expansion-job", model: "qwen3-expand:q8" }),
    );
    await flushPromises();

    await wrapper.get("[data-test='mobile-retry-expansion']").trigger("click");
    await flushPromises();

    // The retried request reuses the immutable recovery record's frozen style.
    expect(expandPrompt).toHaveBeenLastCalledWith(
      "lighthouse",
      { variations: 1, modelFamily: model.family, task: "text-to-video", style: cinematicHint },
      target,
    );
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a lighthouse after the storm",
    );
    // The retried apply is still a quick Batch 1 — it bakes and clears too.
    expect(wrapper.get("[data-test='mobile-style-active']").text()).toBe("None");
  });

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
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("1024 × 1024 px");

    await fieldControl("Model").setValue(model.name);
    await flushPromises();
    expect(wrapper.get("[data-orientation='landscape']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-aspect='3:2']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("768 × 512 px");

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
    await wrapper.get("[data-test='mobile-source-add']").trigger("click");
    wrapper
      .getComponent(MobileImagePickerSheet)
      .vm.$emit("pick", { filename: "source.png", base64: btoa("source") });
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
    // An oversized custom size is advisory only — the server is the
    // authority, so Develop stays enabled and the size still submits.
    expect(wrapper.find("[data-test='mobile-resolution-error']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-resolution-warning']").text()).toContain("1.8 MP");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
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
      { variations: 3, modelFamily: model.family, task: "text-to-video" },
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

    expect(previewGenerationPlacement).toHaveBeenCalledWith(
      target,
      expect.objectContaining({ batch_size: 1 }),
      3,
    );
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(3);
    expect(document.activeElement).toBe(fieldControl("Prompt").element);
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop 3 prints (+3 queued)",
    );
    expect(openStreams).toHaveLength(3);
    const firstSeed = openStreams[0]?.options.body.seed as number;
    expect(openStreams.map((stream) => stream.options.body.batch_size)).toEqual([1, 1, 1]);
    expect(openStreams.map((stream) => stream.options.body.seed)).toEqual([
      firstSeed,
      firstSeed + 1,
      firstSeed + 2,
    ]);
    expect(openStreams.map((stream) => stream.options.body.prompt)).toEqual([
      "three variations of a storm · prepared 1",
      "an edited middle storm",
      "three variations of a storm · prepared 3",
    ]);
    expect(openStreams.map((stream) => stream.options.body.original_prompt)).toEqual([
      "three variations of a storm",
      "three variations of a storm",
      "three variations of a storm",
    ]);
    expect(openStreams[0]?.options.body.batch_id).toEqual(expect.any(String));
    expect(openStreams.map((stream) => stream.options.body.batch_index)).toEqual([1, 2, 3]);
    expect(openStreams.map((stream) => stream.options.body.batch_count)).toEqual([3, 3, 3]);
  });

  it.each([
    ["quick Batch 1", 1],
    ["prepared Batch N", 2],
  ])(
    "does not submit %s when placement returns after the same URL/key has a new instance",
    async (_label, count) => {
      wrapper = mountMobileApp();
      await flushPromises();
      await fieldControl("Prompt").setValue("identity-fenced storm");
      if (count === 1) {
        await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
        await flushPromises();
      } else {
        await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
        await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
        await flushPromises();
      }

      const preview = deferred<ReturnType<typeof plannedPlacement>>();
      previewGenerationPlacement.mockReturnValueOnce(preview.promise);
      if (count === 1) {
        await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
      } else {
        await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
      }
      await vi.waitFor(() => expect(previewGenerationPlacement).toHaveBeenCalledTimes(1));
      expect(previewGenerationPlacement).toHaveBeenCalledWith(
        target,
        expect.objectContaining({ batch_size: 1 }),
        count,
      );

      apiJsonTo.mockImplementation((_target: unknown, path: string) => {
        if (path === "/api/status") {
          return Promise.resolve({ ...status, instance_id: "replacement-instance" });
        }
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve([print]);
        return Promise.reject(new Error(`Unexpected API path: ${path}`));
      });
      window.dispatchEvent(new Event("pageshow"));
      await flushPromises();
      preview.resolve(plannedPlacement());
      await flushPromises();

      expect(openStreams).toHaveLength(0);
      expect(wrapper.text()).toContain("connection details changed while checking placement");
    },
  );

  it.each([
    [
      "authoritative infeasible",
      {
        version: 1,
        authoritative: false,
        state_version: 2,
        plan_version: 2,
        outcome: "infeasible",
        candidate: null,
        reason: "insufficient_vram",
      },
    ],
    ["null", null],
    ["malformed", {}],
    [
      "contradictory unsupported",
      {
        version: 1,
        authoritative: true,
        state_version: 2,
        plan_version: 2,
        outcome: "unsupported",
      },
    ],
    ["HTTP 401", new ApiError("unauthorized", 401)],
    ["HTTP 403", new ApiError("forbidden", 403)],
    ["HTTP 426", new ApiError("upgrade", 426)],
    ["HTTP 500", new ApiError("failed", 500)],
  ])(
    "preserves prepared Batch N and opens zero streams for %s placement",
    async (_case, result) => {
      wrapper = mountMobileApp();
      await flushPromises();
      await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
      await fieldControl("Prompt").setValue("preserved storm pair");
      await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
      await flushPromises();
      if (result instanceof Error) {
        previewGenerationPlacement.mockRejectedValueOnce(result);
      } else {
        previewGenerationPlacement.mockResolvedValueOnce(result);
      }

      await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
      await flushPromises();

      expect(openStreams).toHaveLength(0);
      expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(true);
      expect(
        wrapper.get("[data-test='mobile-develop-prepared']").attributes("disabled"),
      ).toBeUndefined();
    },
  );

  it.each([
    [
      "authoritative infeasibility",
      {
        version: 1,
        authoritative: true,
        state_version: 2,
        plan_version: 2,
        outcome: "infeasible",
        candidate: null,
        reason: "model is missing a component",
        missing_components: [
          {
            kind: "vae",
            name: "ae.safetensors",
            present: false,
            repair_model: model.name,
          },
        ],
      },
      "Studio cannot run this print: model is missing a component. Missing components: ae.safetensors. Nothing was queued.",
    ],
    [
      "temporary scheduler failure",
      {
        version: 1,
        authoritative: false,
        state_version: 2,
        plan_version: 2,
        outcome: "temporarily_unavailable",
        candidate: null,
        reason: "scheduler snapshot changed",
      },
      "Studio could not compute a placement plan right now. Reason: scheduler snapshot changed. Try again. Nothing was queued.",
    ],
    [
      "malformed infeasible metadata",
      {
        version: 1,
        authoritative: true,
        state_version: 2,
        plan_version: 2,
        outcome: "infeasible",
        candidate: null,
        reason: "model is missing a component",
        missing_components: [
          {
            kind: "vae",
            name: "",
            present: false,
            repair_model: model.name,
          },
        ],
      },
      "Studio returned an invalid placement response.",
    ],
    ["malformed preview", {}, "Studio returned an invalid placement response."],
  ])("names %s without discarding prepared work", async (_case, result, expected) => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("preserved storm pair");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    previewGenerationPlacement.mockResolvedValueOnce(result);

    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain(expected);
    expect(openStreams).toHaveLength(0);
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(true);
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
    expect(liveSummary).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    expect(wrapper.find("img.result-media").exists()).toBe(true);
  });

  it("keeps a prepared sibling live when cancellation is unconfirmed", async () => {
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
    expect(openStreams[2]!.options.signal.aborted).toBe(false);
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]!.text()).toContain(
      "Cancellation failed",
    );
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
    openStreams[1]!.options.onEvent("error", JSON.stringify({ message: "host ran out of memory" }));
    openStreams[1]!.resolve();
    await flushPromises();

    const liveSummary = wrapper.findAll(".sr-only[aria-live='polite']")[1]!.text();
    expect(liveSummary).toContain("Variation 2, “failed middle storm”");
    expect(liveSummary).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    expect(liveSummary).not.toContain("remote cancellation was not confirmed");
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

  it("refuses a missing-model Remix pull after its frozen dimensions change", async () => {
    remixPrompt.mockRejectedValueOnce(
      new Error("local expand model not found, run: mold pull qwen3-expand:q8"),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("lighthouse in rain");
    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-pull-expansion']").text()).toContain("Pull");

    const styleDimension = wrapper
      .findAll(".mobile-remix-dimensions label")
      .find((label) => label.text().includes("Style"));
    await styleDimension?.get("input").setValue(false);
    await wrapper.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();

    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(wrapper.get(".mobile-expansion-pull").text()).toContain("Remix inputs changed");
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

  it("uses an RFC 4122 consumer id when native randomUUID is unavailable", async () => {
    vi.stubGlobal("crypto", {
      getRandomValues(bytes: Uint8Array) {
        bytes.fill(0);
        return bytes;
      },
    });
    const pinia = createPinia();
    const downloads = useMobileDownloadsStore(pinia);
    const unregister = vi.spyOn(downloads, "unregisterConsumer");

    wrapper = mountMobileApp(pinia);
    await flushPromises();
    wrapper.unmount();
    wrapper = null;

    expect(unregister).toHaveBeenCalledWith("mobile-generate-00000000-0000-4000-8000-000000000000");
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

  it("develops the active print over a live latent preview once one arrives", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("a latent preview print");

    // Before the first preview frame the status line stands alone.
    expect(wrapper.find("[data-test='mobile-develop-bed']").exists()).toBe(false);

    openStreams[0]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "denoise_step", step: 2, total: 8, elapsed_ms: 40 }),
    );
    openStreams[0]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "preview", image: btoa("latent-png"), step: 2, total: 8 }),
    );
    await flushPromises();

    const bed = wrapper.get("[data-test='mobile-develop-bed']");
    // The bed adopts the submitted print's aspect ratio and hands the same
    // ratio to CSS so the 55vh cap can ride the width axis without distortion.
    expect(bed.attributes("style")).toContain("aspect-ratio");
    expect(bed.attributes("style")).toContain("--bed-ar");
    const preview = wrapper.get("[data-test='mobile-develop-preview']");
    // The reducer turns the base64 payload into a blob URL (mocked here).
    expect(preview.attributes("src")).toMatch(/^blob:/);
    // blur(max(2, 14 − 12·p)) at p = 2/8 → 11px.
    expect(preview.attributes("style")).toContain("blur(11px)");
    // The develop grain layers over the preview and thins with progress.
    expect(wrapper.find("develop-canvas-stub").exists()).toBe(true);
    // iPhone already keeps the changing status outside and below the noisy
    // preview, matching the desktop/web placement invariant.
    const summary = wrapper.get("[data-test='mobile-generation-summary']");
    expect(summary.text()).toBe("Developing 2 / 8");
    expect(bed.find("[data-test='mobile-generation-summary']").exists()).toBe(false);
    expect(
      bed.element.compareDocumentPosition(summary.element) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

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

    // Settled jobs drop the bed; the finished print renders instead.
    expect(wrapper.find("[data-test='mobile-develop-bed']").exists()).toBe(false);
    expect(wrapper.find("img.result-media").exists()).toBe(true);
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

  it("counts a queued print's live place in line rather than its submit slot", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      if (path === "/api/queue")
        return Promise.resolve({
          entries: [
            { id: "job-a", model: "m", state: "running", started_at_unix_ms: 1, position: 0 },
            { id: "job-2", model: "m", state: "queued", started_at_unix_ms: 2, position: 1 },
          ],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "settled",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [],
          },
        });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first prompt");
    openStreams[0]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "denoise_step", step: 3, total: 30, elapsed_ms: 10 }),
    );
    await submitPrompt("second prompt");
    // The one-shot SSE frame says 7; the live listing says 1 and wins.
    openStreams[1]?.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 7, id: "job-2" }),
    );
    await flushPromises();
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    const rows = wrapper.findAll("[data-test='mobile-generation-job']");
    expect(rows[1]?.get("[data-test='mobile-generation-status']").text()).toBe("QUEUED #1");
  });

  it("keeps a pre-ID job live when remote cancellation is unconfirmed", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("cancel before the remote queue id arrives");

    await wrapper.get("[data-test='mobile-generation-cancel']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("Queued");
    expect(openStreams[0]?.options.signal.aborted).toBe(false);
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Cancellation failed",
    );

    openStreams[0]?.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("finished after failed cancellation"),
        format: "png",
        width: 768,
        height: 512,
        seed_used: 1,
        generation_time_ms: 500,
        model: model.name,
      }),
    );
    openStreams[0]?.resolve();
    await flushPromises();
    expect(wrapper.find("img.result-media").exists()).toBe(true);
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
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
  });

  it("auto-saves completed stills to Photos from the authenticated host gallery", async () => {
    apiFetchTo.mockResolvedValueOnce({
      blob: () => Promise.resolve(new Blob(["generated-image"], { type: "image/png" })),
    } as Response);
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("save this print");

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        filename: "saved print.png",
        width: 768,
        height: 512,
        seed_used: 42,
        generation_time_ms: 1_250,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/saved%20print.png");
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: btoa("generated-image"),
    });
  });

  it("auto-saves both post-upscale stills but never buffers a completed video", async () => {
    apiFetchTo
      .mockResolvedValueOnce({
        blob: () => Promise.resolve(new Blob(["original-image"], { type: "image/png" })),
      } as Response)
      .mockResolvedValueOnce({
        blob: () => Promise.resolve(new Blob(["upscaled-image"], { type: "image/png" })),
      } as Response);
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("save both versions");

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        original_filename: "saved-original.png",
        filename: "saved-upscaled.png",
        width: 1536,
        height: 1024,
        original_width: 768,
        original_height: 512,
        seed_used: 42,
        generation_time_ms: 1_250,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/saved-original.png");
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/saved-upscaled.png");
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: btoa("original-image"),
    });
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: btoa("upscaled-image"),
    });

    apiFetchTo.mockClear();
    invoke.mockClear();
    await submitPrompt("leave this video remote");
    openStreams[1]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "mp4",
        filename: "saved-video.mp4",
        width: 768,
        height: 512,
        seed_used: 43,
        generation_time_ms: 1_500,
        model: model.name,
        video_frames: 121,
        video_fps: 30,
      }),
    );
    openStreams[1]!.resolve();
    await flushPromises();

    expect(apiFetchTo).not.toHaveBeenCalledWith(target, "/api/gallery/image/saved-video.mp4");
    expect(invoke).not.toHaveBeenCalledWith("save_image_to_photos", expect.anything());
  });

  it("does not fetch or save completed stills when Photos auto-save is off", async () => {
    localStorage.setItem(
      "mold.mobile.settings.v1",
      JSON.stringify({
        theme: "system",
        themeFamily: "safelight",
        autoSavePhotos: false,
      }),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    apiFetchTo.mockClear();
    invoke.mockClear();
    await submitPrompt("keep this remote");

    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "png",
        filename: "remote only.png",
        width: 768,
        height: 512,
        seed_used: 42,
        generation_time_ms: 1_250,
        model: model.name,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    expect(apiFetchTo).not.toHaveBeenCalledWith(target, "/api/gallery/image/remote%20only.png");
    expect(invoke).not.toHaveBeenCalledWith("save_image_to_photos", expect.anything());
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
    await wrapper
      .findAll(".result-preview-error button")
      .find((button) => button.text() === "Try preview again")!
      .trigger("click");
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
    expect(
      wrapper
        .findAll(".result-preview-error button")
        .some((button) => button.text() === "Try preview again"),
    ).toBe(true);
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

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
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

  it("auto-loads older prints on scroll and serializes with a completion refresh", async () => {
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
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(true),
    );
    expect(wrapper.find("button.gallery-more").exists()).toBe(false);

    const olderThumbnail = { release: null as (() => void) | null };
    apiFetchTo.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          olderThumbnail.release = () =>
            resolve({ blob: () => Promise.resolve(new Blob(["older"])) } as Response);
        }),
    );
    scrollToGallerySentinel();
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='mobile-gallery-sentinel']").text()).toBe(
        "Loading older prints…",
      ),
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

  it("continues auto-loading when a failed page leaves the sentinel visible", async () => {
    const prints = Array.from({ length: 81 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(prints);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    let thumbnailCall = 0;
    apiFetchTo.mockImplementation(() => {
      thumbnailCall += 1;
      if (thumbnailCall > 40 && thumbnailCall <= 80) {
        return Promise.reject(new Error("thumbnail unavailable"));
      }
      return Promise.resolve({
        blob: () => Promise.resolve(new Blob([`thumbnail-${thumbnailCall}`])),
      } as Response);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(true),
    );

    scrollToGallerySentinel();

    await vi.waitFor(() => expect(thumbnailCall).toBe(81));
    expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(41);
    expect(wrapper.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(false);
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
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(true),
    );

    const olderThumbnail = { release: null as (() => void) | null };
    apiFetchTo.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          olderThumbnail.release = () =>
            resolve({ blob: () => Promise.resolve(new Blob(["older"])) } as Response);
        }),
    );
    scrollToGallerySentinel();
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='mobile-gallery-sentinel']").text()).toBe(
        "Loading older prints…",
      ),
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

describe("MobileApp wan source conditioning", () => {
  const wanModel: ModelEntry = {
    name: "wan22-i2v-a14b",
    family: "wan",
    size_gb: 30,
    is_loaded: false,
    hf_repo: "example/wan",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 832,
    default_height: 480,
    description: "Wan video model",
    downloaded: true,
  };

  function serveWan(entry: ModelEntry, gallery: GalleryImage[] = []): void {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([entry]);
      if (path === "/api/gallery") return Promise.resolve(gallery);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      if (path.startsWith("/api/catalog/installed")) return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
  }

  async function pickInto(title: string, filename: string, base64: string): Promise<void> {
    const sheet = wrapper
      ?.findAllComponents(MobileImagePickerSheet)
      .find((candidate) => candidate.props("title") === title);
    if (!sheet) throw new Error(`Missing ${title} picker`);
    sheet.vm.$emit("pick", { filename, base64 });
    await flushPromises();
  }

  it("shows the H3 first-frame blocker before prompt entry and clears it after picking", async () => {
    const h3Model: ModelEntry = {
      ...wanModel,
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      hf_repo: "Comfy-Org/MiniMax-H3",
      default_steps: 21,
      default_guidance: 0,
      default_width: 1344,
      default_height: 768,
      default_frames: 124,
      default_fps: 24,
      source_image: "required",
    };
    serveWan(h3Model);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-h3-authoring-error']").text()).toContain(
      "requires a first frame",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );

    // The boundary wells live in the primary Create stack now — no Advanced
    // sheet required.
    await wrapper.get("[data-test='source-gallery']").trigger("click");
    await pickInto(
      "First frame",
      "opening.png",
      "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC",
    );
    expect(wrapper.find("[data-test='mobile-h3-authoring-error']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
  });

  it("keeps the well and offers no end frame when the host advertises no contract", async () => {
    serveWan(wanModel);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-add']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-end-frame-add']").exists()).toBe(false);
    await fieldControl("Prompt").setValue("a lantern drifting downriver");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

  it("hides the source well for an advertised text-to-video checkpoint", async () => {
    serveWan({ ...wanModel, name: "wan22-t2v-a14b", source_image: "unsupported" });
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-source-add']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-end-frame-add']").exists()).toBe(false);
    await fieldControl("Prompt").setValue("a lantern drifting downriver");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

  it("holds Develop when a hidden well still holds a rejected source image", async () => {
    // The well is gone on an advertised text-to-video checkpoint, so it can no
    // longer report anything: only the app-level gate stands between a stale
    // source image and a request admission would refuse.
    serveWan({ ...wanModel, name: "wan22-t2v-a14b", source_image: "unsupported" });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lantern drifting downriver");

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("stale opening");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-source-conditioning-gate']").text()).toContain(
      "text-to-video only",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
  });

  it("holds Develop with a visible reason until a required source is attached", async () => {
    serveWan({ ...wanModel, source_image: "required" });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lantern drifting downriver");

    expect(wrapper.get("[data-test='mobile-source-conditioning-gate']").text()).toContain(
      "image-to-video only",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );
    // The collapsed disclosure has to say the section is not optional, and it
    // opens itself so the well is one tap away.
    const disclosure = wrapper.get("[data-test='mobile-source-disclosure']");
    expect(disclosure.get("summary small").text()).toBe("Required");
    expect(disclosure.attributes("open")).toBeDefined();

    await pickInto("Source image", "opening.png", btoa("opening"));
    expect(wrapper.find("[data-test='mobile-source-conditioning-gate']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

  it("refuses an end-frame-only draft and queues the pair as two keyframes", async () => {
    serveWan({ ...wanModel, source_image: "optional" });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lantern drifting downriver");
    await fieldControl("Frames").setValue("97");
    await flushPromises();

    await pickInto("End frame", "closing.png", btoa("closing"));
    expect(wrapper.get("[data-test='mobile-source-conditioning-gate']").text()).toContain(
      "needs a first frame",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).toHaveProperty(
      "disabled",
    );

    await pickInto("Source image", "opening.png", btoa("opening"));
    expect(wrapper.get("[data-test='mobile-source-disclosure']").get("summary small").text()).toBe(
      "opening.png · end frame",
    );
    // The closing index follows the frame count held at submit time, not the
    // one that was current when the end frame was attached.
    await fieldControl("Frames").setValue("81");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.path).toBe("/api/generate/stream");
    expect(openStreams[0]?.options.body).toMatchObject({
      keyframes: [
        { frame: 0, image: btoa("opening"), name: "opening.png" },
        { frame: 80, image: btoa("closing"), name: "closing.png" },
      ],
    });
    // The engine refuses `source_image` + `keyframes` together ("not both");
    // the pair travels only as keyframes.
    expect(openStreams[0]?.options.body.source_image).toBeFalsy();
  });

  it("sends no keyframes for a lone source image", async () => {
    serveWan({ ...wanModel, source_image: "optional" });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a lantern drifting downriver");
    await fieldControl("Frames").setValue("81");
    await pickInto("Source image", "opening.png", btoa("opening"));
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.options.body.source_image).toBe(btoa("opening"));
    expect(openStreams[0]?.options.body).not.toHaveProperty("keyframes");
  });

  it("says the reused end frame cannot be restored from saved metadata", async () => {
    const flfPrint: GalleryImage = {
      filename: "river.mp4",
      timestamp: 1_700_000_100,
      format: "mp4",
      metadata: {
        prompt: "a lantern drifting downriver",
        model: wanModel.name,
        seed: 11,
        steps: 20,
        guidance: 3.5,
        width: 832,
        height: 480,
        frames: 81,
        output_format: "mp4",
        keyframes: [
          { frame: 0, name: "opening.png", sha256: "aa" },
          { frame: 80, name: "closing.png", sha256: "bb" },
        ],
      },
    };
    serveWan({ ...wanModel, source_image: "optional" }, [flfPrint]);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    const summary = wrapper.get("[data-test='mobile-generation-summary']").text();
    // No source provenance in a first/last print — both endpoints are named.
    expect(summary).toContain(
      "The end frame (closing.png) and the first frame (opening.png) can't be restored",
    );
    expect(summary).toContain("Prompt settings restored");
    expect(wrapper.find("[data-test='mobile-generation-error']").exists()).toBe(true);
  });
});

describe("MobileApp transport error copy", () => {
  it("humanizes a dead connection when saving a host", async () => {
    apiJsonTo.mockImplementation((apiTarget: { baseUrl: string }, path: string) => {
      if (apiTarget.baseUrl.includes("render.local")) {
        return Promise.reject(new TypeError("Load failed"));
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await fieldControl("Address or MagicDNS name").setValue("render.local");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(wrapper.text()).toContain(
      "Couldn’t reach render.local. Check the connection and try again.",
    );
    expect(wrapper.text()).not.toContain("Load failed");
  });

  it("keeps an empty-bodied auth failure visible with API-key copy", async () => {
    const { ApiError } = await import("../lib/api/client");
    apiJsonTo.mockImplementation((apiTarget: { baseUrl: string }, path: string) => {
      if (apiTarget.baseUrl.includes("render.local")) {
        return Promise.reject(new ApiError("", 401));
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await fieldControl("Address or MagicDNS name").setValue("render.local");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(wrapper.text()).toContain(
      "render.local didn’t accept the API key. Update it in Machines and try again.",
    );
  });

  it("keeps background model-load failures out of the generation status line", async () => {
    apiJsonTo.mockRejectedValue(new TypeError("Load failed"));
    wrapper = mountMobileApp();
    await flushPromises();

    const banner = wrapper.get("[data-test='mobile-model-error']");
    expect(banner.text()).toContain("Couldn’t load generation models from Studio.");
    expect(banner.text()).toContain("Couldn’t reach Studio. Check the connection and try again.");
    expect(banner.text()).not.toContain("Load failed");
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("Ready");
  });

  it("reloads models automatically when the app returns to the foreground", async () => {
    let hostReachable = false;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (!hostReachable) return Promise.reject(new TypeError("Load failed"));
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-model-error']").exists()).toBe(true);

    hostReachable = true;
    document.dispatchEvent(new Event("visibilitychange"));
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-model-error']").exists()).toBe(false);
    expect(fieldControl("Model").element).toHaveProperty("value", model.name);
  });
});

describe("MobileApp foreground resume", () => {
  const resumedPrint: GalleryImage = {
    filename: "resumed print.png",
    timestamp: 1_700_000_100,
    format: "png",
    metadata: {
      prompt: "a ship crossing violet lightning",
      model: model.name,
      seed: 77,
      steps: 28,
      guidance: 4,
      width: 768,
      height: 512,
    },
  };

  it("restores the native WKWebView frame on launch and after a picker resume", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      value: {},
      configurable: true,
    });
    wrapper = mountMobileApp();
    await flushPromises();
    expect(invoke).toHaveBeenCalledWith("restore_mobile_viewport");

    invoke.mockClear();
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();
    expect(invoke).toHaveBeenCalledWith("restore_mobile_viewport");
  });

  async function submitSeededPrompt(prompt: string, seed: number): Promise<void> {
    const liveForm = wrapper!.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.seed = String(seed);
    // Match resumedPrint's recorded metadata — the reconciler's gallery join
    // requires dims and steps to agree, exactly as a real host records them.
    liveForm.width = 768;
    liveForm.height = 512;
    liveForm.steps = 28;
    await fieldControl("Prompt").setValue(prompt);
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  function killStream(index = 0, message = "Load failed"): void {
    openStreams[index]!.options.onClose?.(new TypeError(message) as Error);
    openStreams[index]!.resolve();
  }

  it("renders the finished print when resuming after the stream died mid-generation", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([resumedPrint]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitSeededPrompt("a ship crossing violet lightning", 77);
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-9" }),
    );
    // iOS suspension killed the socket; the print finished server-side.
    killStream();
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("seed 77");
    expect(wrapper.text()).not.toContain("Load failed");
    expect(wrapper.get("img.result-media").attributes("src")).toBe(
      "https://studio/media/full-video",
    );
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation completed.",
    );
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
  });

  it("clears a zombie queued job and explains the interruption without transport jargon", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "job-9",
              model: model.name,
              state: "queued",
              position: 0,
              started_at_unix_ms: 0,
            },
          ],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitSeededPrompt("a print the suspension orphaned", 41);
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-9" }),
    );
    killStream(0, "The network connection was lost.");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/job-9", { method: "DELETE" });
    const summary = wrapper.get("[data-test='mobile-generation-summary']").text();
    expect(summary).toBe(
      "The connection dropped while this print waited in Studio’s queue. Develop again to requeue it.",
    );
    expect(summary).not.toContain("network connection was lost");
    expect(
      wrapper.get("[data-test='mobile-generation-summary']").find("[role='alert']").exists(),
    ).toBe(true);
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toContain(
      "Generation failed.",
    );
  });

  it("re-attaches to a print still developing on the host and completes it", async () => {
    let queueCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/queue") {
        queueCalls += 1;
        return Promise.resolve({
          entries:
            queueCalls <= 2
              ? [{ id: "job-9", model: model.name, state: "running", position: 0 }]
              : [],
        });
      }
      if (path === "/api/gallery") return Promise.resolve([resumedPrint]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitSeededPrompt("a ship crossing violet lightning", 77);
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-9" }),
    );
    killStream();
    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));

    expect(queueCalls).toBeGreaterThanOrEqual(3);
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("seed 77");
    expect(wrapper.text()).not.toContain("Load failed");
  });

  it("renews the promoted result's expired media ticket when returning to the foreground", async () => {
    streamableMediaUrl
      .mockResolvedValueOnce("https://studio/media/video?media_token=old&expires=1")
      .mockResolvedValueOnce("https://studio/media/video?media_token=new&expires=4102444800");
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("renew on resume");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: "",
        format: "mp4",
        filename: "renew on resume.mp4",
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

    document.dispatchEvent(new Event("visibilitychange"));
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("video.result-media").attributes("src")).toContain("media_token=new");
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
      autoSavePhotos: true,
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

describe("MobileApp create settings reset", () => {
  it("restores the model defaults from Create without discarding the prompt", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a ship crossing violet lightning");
    await fieldControl("Negative prompt").setValue("calm water");
    await fieldControl("Steps").setValue("12");
    await flushPromises();

    await wrapper.get("[data-test='mobile-settings-reset']").trigger("click");
    await flushPromises();

    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a ship crossing violet lightning",
    );
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe("");
    // The selected model's defaults, not the bare form defaults.
    expect((fieldControl("Steps").element as HTMLInputElement).value).toBe("30");
    expect(wrapper.get("[data-test='mobile-settings-reset']").attributes("aria-label")).toBe(
      "Reset settings to model defaults",
    );
  });

  it("badges and resets LTX-2 guidance overrides from the Advanced sheet", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");
    await wrapper.get("[data-test='mobile-ltx2-stg-scale']").setValue("1.5");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-advanced-trigger-count']").text()).toBe("1");
    expect(wrapper.get("[data-test='mobile-advanced-count']").text()).toBe("1");

    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-advanced-trigger-count']").exists()).toBe(false);
    expect(
      (wrapper.get("[data-test='mobile-ltx2-stg-scale']").element as HTMLInputElement).value,
    ).toBe("");

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.guidanceOverrides.stgScale = 1.5;
    liveForm.family = "flux";
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-advanced-trigger-count']").exists()).toBe(false);
  });

  it("keeps the visible source across an H3 round trip via the shared bridge", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = "QUJD";
    liveForm.sourceImageName = "pic.png";
    await flushPromises();

    const h3: ModelEntry = {
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      downloaded: true,
      source_image: "required",
      default_frames: 124,
      default_fps: 24,
      default_steps: 30,
      default_guidance: 7,
      default_width: 1344,
      default_height: 768,
    } as ModelEntry;
    applyModelDefaults(liveForm, h3);
    expect(liveForm.h3Authoring?.firstFrame?.data).toBe("QUJD");
    expect(liveForm.sourceImage).toBeNull();

    applyModelDefaults(liveForm, {
      ...h3,
      name: "flux:q8",
      family: "flux",
      source_image: null,
    } as ModelEntry);
    expect(liveForm.sourceImage).toBe("QUJD");
    expect(liveForm.sourceImageName).toBe("pic.png");
  });

  it("Advanced reset preserves staged source media — it lives in the primary form", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = "SRC";
    liveForm.sourceImageName = "pic.png";
    await flushPromises(); // the new-source watcher re-derives sourceFit
    liveForm.strength = 0.4;
    liveForm.sourceFit = { mode: "crop-fill" };
    liveForm.maskImage = "MASK";
    liveForm.controlImage = "CTRL";
    liveForm.controlModel = "canny";
    liveForm.controlScale = 0.8;
    liveForm.imageAttachments = ["ATT"];
    liveForm.negativePrompt = "blurry";
    await flushPromises();

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");
    await flushPromises();

    expect(liveForm.sourceImage).toBe("SRC");
    expect(liveForm.sourceImageName).toBe("pic.png");
    expect(liveForm.strength).toBe(0.4);
    expect(liveForm.sourceFit).toEqual({ mode: "crop-fill" });
    expect(liveForm.maskImage).toBe("MASK");
    expect(liveForm.controlImage).toBe("CTRL");
    expect(liveForm.controlModel).toBe("canny");
    expect(liveForm.controlScale).toBe(0.8);
    expect(liveForm.imageAttachments).toEqual(["ATT"]);
    expect(liveForm.negativePrompt).toBe("");
  });
});

describe("MobileApp primary navigation", () => {
  it("starts each tab at the top instead of carrying another tab's scroll position", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();

    const content = wrapper.get(".mobile-content").element as HTMLElement;
    content.scrollTop = 420;

    await wrapper.get("[data-test='mobile-tab-catalog']").trigger("click");
    await flushPromises();

    expect(content.scrollTop).toBe(0);
  });
});

describe("MobileApp gallery", () => {
  it("loads reachable hosts without waiting for a host already known to be offline", async () => {
    const offlineTarget = { baseUrl: "http://halcyon.tailnet.ts.net:7680", apiKey: "secret" };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Plato",
          baseUrl: target.baseUrl,
          hostname: "plato",
          version: "0.18.0",
          online: false,
        },
        {
          id: "halcyon-id",
          name: "Halcyon",
          baseUrl: offlineTarget.baseUrl,
          hostname: "halcyon",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (baseUrl === offlineTarget.baseUrl) {
        if (path === "/api/status") return Promise.reject(new Error("offline"));
        if (path === "/api/gallery") return new Promise(() => {});
      }
      if (path === "/api/status") return Promise.resolve({ ...status, hostname: "plato" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    expect(wrapper.find(".empty-state").exists()).toBe(false);
    expect(wrapper.text()).toContain("1 host unavailable");
  });

  it("refreshes the Library when a previously offline host recovers", async () => {
    const recoveredTarget = { baseUrl: "http://halcyon.tailnet.ts.net:7680", apiKey: "secret" };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Plato",
          baseUrl: target.baseUrl,
          hostname: "plato",
          version: "0.18.0",
          online: false,
        },
        {
          id: "halcyon-id",
          name: "Halcyon",
          baseUrl: recoveredTarget.baseUrl,
          hostname: "halcyon",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    const recoveredProbe = deferred<ServerStatus>();
    let halcyonProbeCount = 0;
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (baseUrl === recoveredTarget.baseUrl && path === "/api/status") {
        halcyonProbeCount += 1;
        return halcyonProbeCount === 1
          ? Promise.reject(new Error("offline"))
          : recoveredProbe.promise;
      }
      if (path === "/api/status") return Promise.resolve({ ...status, hostname: "plato" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/gallery") {
        return Promise.resolve(baseUrl === recoveredTarget.baseUrl ? [print] : []);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='gallery-item']").exists()).toBe(false);

    window.dispatchEvent(new Event("pageshow"));
    await vi.waitFor(() => expect(halcyonProbeCount).toBe(2));
    recoveredProbe.resolve({ ...status, hostname: "halcyon", instance_id: "halcyon-id" });

    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    expect(wrapper.text()).not.toContain("host unavailable");
  });

  it("keeps the native image context menu and enters multi-select from Select", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const contextMenu = new Event("contextmenu", { bubbles: true, cancelable: true });
    wrapper.get("[data-test='gallery-item'] img").element.dispatchEvent(contextMenu);
    expect(contextMenu.defaultPrevented).toBe(false);
    expect(wrapper.find("[data-test='mobile-gallery-actions']").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
    expect(wrapper.get("[data-test='gallery-item']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-test='mobile-gallery-selection-indicator']").text()).toBe("✓");
  });

  it("deletes one selected print from every host that contains a copy", async () => {
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        return Promise.resolve([{ ...print, timestamp: print.timestamp + (render ? 1 : 0) }]);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockImplementation((_route, _path, init?: RequestInit) =>
      Promise.resolve(
        init?.method === "DELETE"
          ? new Response(null, { status: 204 })
          : ({ blob: () => Promise.resolve(new Blob(["thumbnail"])) } as Response),
      ),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));

    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.findAll("[data-test='gallery-item']")[0]!.trigger("click");
    const deleteButton = () =>
      wrapper!.get("[data-test='mobile-gallery-actions']").find("button.danger");
    await deleteButton().trigger("click");
    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain(
      "Delete 1 everywhere?",
    );
    expect(deleteButton().text()).toBe("Confirm");
    await deleteButton().trigger("click");
    await flushPromises();

    const deletes = apiFetchTo.mock.calls.filter(([, , init]) => init?.method === "DELETE");
    expect(deletes).toHaveLength(2);
    expect(deletes.map(([, path]) => path)).toEqual([
      "/api/gallery/image/storm%20clip.mp4",
      "/api/gallery/image/storm%20clip.mp4",
    ]);
    expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(0);
  });

  it("does not hide or delete a chained legacy copy outside the representative window", async () => {
    const renderTarget = { baseUrl: "http://render.tailnet.ts.net:7680", apiKey: "secret" };
    const archiveTarget = { baseUrl: "http://archive.tailnet.ts.net:7680", apiKey: "secret" };
    const routes = [
      { id: "studio-id", name: "Studio", baseUrl: target.baseUrl },
      { id: "render-id", name: "Render", baseUrl: renderTarget.baseUrl },
      { id: "archive-id", name: "Archive", baseUrl: archiveTarget.baseUrl },
    ];
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify(routes.map((host) => ({ ...host, online: false }))),
    );
    const copies = new Map([
      [target.baseUrl, { filename: "newest.mp4", timestamp: 6_000 }],
      [renderTarget.baseUrl, { filename: "middle.mp4", timestamp: 3_000 }],
      [archiveTarget.baseUrl, { filename: "oldest.mp4", timestamp: 0 }],
    ]);
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string) => {
      const host = routes.find((candidate) => candidate.baseUrl === route.baseUrl)!;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: host.name.toLowerCase(),
          instance_id: host.id,
        });
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        return Promise.resolve([
          {
            ...print,
            ...copies.get(route.baseUrl),
            size_bytes: 4_096,
            metadata: { ...print.metadata, seed: 42, model: "flux-dev:q8" },
          },
        ]);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockImplementation((_route, _path, init?: RequestInit) =>
      Promise.resolve(
        init?.method === "DELETE"
          ? new Response(null, { status: 204 })
          : ({ blob: () => Promise.resolve(new Blob(["thumbnail"])) } as Response),
      ),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(2));

    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.findAll("[data-test='gallery-item']")[1]!.trigger("click");
    const deleteButton = () =>
      wrapper!.get("[data-test='mobile-gallery-actions']").find("button.danger");
    await deleteButton().trigger("click");
    await deleteButton().trigger("click");
    await flushPromises();

    const deletePaths = apiFetchTo.mock.calls
      .filter(([, , init]) => init?.method === "DELETE")
      .map(([, path]) => path);
    expect(deletePaths).toEqual(["/api/gallery/image/oldest.mp4"]);
    expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(1);
  });

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
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("768 × 512 px");
    expect(fieldControl("Format").element).toHaveProperty("value", "mp4");
    expect(fieldControl("Frames").element).toHaveProperty("value", "121");
    expect(fieldControl("FPS").element).toHaveProperty("value", "30");
  });

  it("restores the original source image attributes and crop when reusing a print", async () => {
    const sourcePrint: GalleryImage = {
      ...print,
      filename: "portrait-result.png",
      format: "png",
      metadata: {
        ...print.metadata,
        output_format: "png",
        source_image_name: "portrait.jpg",
        source_image_sha256: "a".repeat(64),
        source_fit: { mode: "crop-fill", alignX: "right", alignY: "top" },
      },
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([sourcePrint]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    restoreGenerationSourceMedia.mockResolvedValue({
      draftId: "source-portrait",
      base64: btoa("original portrait"),
      filename: "portrait.jpg",
      kind: "upload",
      width: 1600,
      height: 900,
      mime: "image/jpeg",
      sourceFit: { mode: "crop-fill", alignX: "right", alignY: "top" },
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(restoreGenerationSourceMedia).toHaveBeenCalledWith("a".repeat(64));
    expect(wrapper.getComponent(MobileSourceControls).props("form")).toMatchObject({
      sourceImage: btoa("original portrait"),
      sourceImageName: "portrait.jpg",
      sourceImageWidth: 1600,
      sourceImageHeight: 900,
      sourceFit: { mode: "crop-fill", alignX: "right", alignY: "top" },
      width: 768,
      height: 512,
    });
    expect(wrapper.get("[data-test='mobile-source-fit']").element).toHaveProperty(
      "value",
      "crop-fill",
    );
  });

  it("opens the latest generated image in the full-screen print viewer when tapped", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("expand this result");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    openStreams[0]!.options.onEvent(
      "complete",
      JSON.stringify({
        image: btoa("generated image"),
        format: "png",
        filename: "expand-this-result.png",
        width: 768,
        height: 512,
        seed_used: 9,
        generation_time_ms: 500,
        model: model.name,
        metadata: print.metadata,
      }),
    );
    openStreams[0]!.resolve();
    await flushPromises();

    await wrapper.get("[data-test='mobile-generated-result']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    expect(wrapper.get("[data-test='gallery-viewer-image']").attributes("src")).toContain("blob:");
  });

  it("shows New and Upscaled indicators on mobile Library tiles", async () => {
    localStorage.setItem(
      "mold.mobile.library-seen-at.v1",
      JSON.stringify({ "studio-id": print.timestamp - 1 }),
    );
    localStorage.setItem("mold.mobile.library-visited.v1", "true");
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        return Promise.resolve([
          {
            ...print,
            filename: "new-upscaled.png",
            format: "png",
            metadata: {
              ...print.metadata,
              upscale_model: "real-esrgan-x4plus",
              generation_width: 384,
              generation_height: 256,
            },
          },
        ]);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();

    const tile = wrapper.get("[data-test='gallery-item']");
    expect(tile.get("[data-test='new-badge']").text()).toBe("New");
    expect(tile.get("[data-test='upscaled-badge']").text()).toBe("Upscaled");
    expect(JSON.parse(localStorage.getItem("mold.mobile.library-seen-at.v1") ?? "{}")).toEqual({
      "studio-id": print.timestamp,
    });
    expect(localStorage.getItem("mold.mobile.library-seen.v1")).toBeNull();
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

  it("appends a gallery source to the selected Ref2VA partition", async () => {
    const ref2vaModel: ModelEntry = {
      ...model,
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      default_steps: 50,
      default_guidance: 0,
      default_width: 1344,
      default_height: 768,
      default_frames: 124,
      default_fps: 24,
      supports_audio: true,
    };
    const still = { ...print, filename: "ordered subject.png", format: "png" as const };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([ref2vaModel]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockResolvedValue({
      headers: new Headers({ "content-type": "image/png" }),
      blob: () => Promise.resolve(new Blob(["subject bytes"], { type: "image/png" })),
    } as Response);

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-use-source']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='mobile-generation-summary']").text()).toContain(
        "reference 1",
      ),
    );

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='h3-reference-0']").text()).toContain("ordered subject.png");
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
      if (baseUrl === remoteTarget.baseUrl && path === "/api/status") {
        return Promise.resolve({ ...status, hostname: "remote", instance_id: "remote-id" });
      }
      if (baseUrl === remoteTarget.baseUrl) return Promise.reject(new Error("models offline"));
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

  it("reuses a sequence print as a clip rail on the Create tab", async () => {
    // iPhone gets Reuse only: the rail reloads from `metadata.chain`, no edit
    // session, and the composer lands on Create with clip 1's prompt (never
    // the newline join the print records as `metadata.prompt`).
    const sequenceModel: ModelEntry = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      supports_sequence: true,
    } as ModelEntry;
    const sequencePrint: GalleryImage = {
      ...print,
      filename: "sequence.mp4",
      metadata: {
        ...print.metadata,
        model: sequenceModel.name,
        prompt: "a harbour at dawn\nthe boats leave",
        chain_job_id: "job-9",
        chain: {
          stage_count: 2,
          motion_tail_frames: 0,
          stages: [
            { prompt: "a harbour at dawn", frames: 25, transition: "smooth" },
            { prompt: "the boats leave", frames: 33, transition: "cut" },
          ],
        },
      },
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.resolve({
          model: sequenceModel.name,
          frames_per_clip_cap: 97,
          frames_per_clip_recommended: 97,
          max_stages: 8,
          max_total_frames: 777,
          fade_frames_max: 32,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "bf16",
          supports_audio: false,
        });
      }
      if (path === "/api/gallery") return Promise.resolve([sequencePrint]);
      if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    const pinia = createPinia();
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    const draft = useSequenceDraftStore(pinia);
    expect(draft.output).toBe("sequence");
    expect(draft.clips.map((clip) => clip.prompt)).toEqual([
      "a harbour at dawn",
      "the boats leave",
    ]);
    expect(draft.editing).toBeNull();
    expect(wrapper.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
      "page",
    );
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
  });

  it("keeps an H3 sequence snapshot in the viewer with an explicit refusal", async () => {
    const sequenceModel: ModelEntry = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      supports_sequence: true,
    } as ModelEntry;
    const h3Sequence: GalleryImage = {
      ...print,
      filename: "invalid-h3-sequence.mp4",
      metadata: {
        ...print.metadata,
        model: "minimax-h3-fl2va:official-bf16",
        chain_job_id: "job-h3",
        chain: {
          stage_count: 2,
          motion_tail_frames: 0,
          stages: [
            { prompt: "opening", frames: 25, transition: "smooth" },
            { prompt: "closing", frames: 25, transition: "cut" },
          ],
        },
      },
    };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/gallery") return Promise.resolve([h3Sequence]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    const pinia = createPinia();
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(wrapper.get(".gallery-viewer-reuse-error").text()).toContain(
      "cannot render a clip sequence",
    );
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    expect(useSequenceDraftStore(pinia).output).toBe("single");
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
      .find((button) => button.text() === "Machines");
    if (!hostsTab) throw new Error("Missing Machines tab");
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

  function pairingPayload(overrides: Record<string, unknown> = {}): string {
    return JSON.stringify({
      type: "mold.mobile-pairing",
      version: 1,
      base_url: "http://pair.local:7680",
      token: "one-time-token",
      expires_at: Math.floor(Date.now() / 1000) + 120,
      instance_id: "pair-id",
      name: "Pair Host",
      ...overrides,
    });
  }

  function pairingUrl(overrides: Record<string, unknown> = {}): string {
    const payload = JSON.parse(pairingPayload(overrides)) as Record<string, string>;
    const url = new URL("mold://pair");
    for (const [key, value] of Object.entries(payload)) {
      if (key !== "type" && value !== null) url.searchParams.set(key, String(value));
    }
    return url.toString();
  }

  async function scanFromMachines(): Promise<void> {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-scan-pairing']").trigger("click");
    await flushPromises();
  }

  it("settles first-run camera permission before opening the pairing scanner", async () => {
    checkBarcodeScannerPermissions.mockResolvedValue("prompt");
    requestBarcodeScannerPermissions.mockResolvedValue("granted");
    scanPairingQr.mockRejectedValue("cancelled");

    await scanFromMachines();

    expect(requestBarcodeScannerPermissions).toHaveBeenCalledOnce();
    expect(scanPairingQr).toHaveBeenCalledOnce();
    expect(scanPairingQr).toHaveBeenCalledWith({
      cameraDirection: "back",
      formats: ["QRCode"],
      windowed: true,
    });
    expect(requestBarcodeScannerPermissions.mock.invocationCallOrder[0]).toBeLessThan(
      scanPairingQr.mock.invocationCallOrder[0]!,
    );
  });

  it("wraps camera mode in cancellable Mold UI and cleans up without an error", async () => {
    const pendingScan = deferred<{ content: string }>();
    scanPairingQr.mockReturnValue(pendingScan.promise);
    cancelBarcodeScanner.mockImplementation(async () => pendingScan.reject("cancelled"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-scan-pairing']").trigger("click");
    await flushPromises();

    const scanner = wrapper.get("[data-test='mobile-pair-scanner']");
    expect(scanner.attributes("role")).toBe("dialog");
    expect(scanner.text()).toContain("Point your camera at the QR code");
    await scanner.get("[data-test='mobile-pair-scanner-cancel']").trigger("click");
    await flushPromises();

    expect(cancelBarcodeScanner).toHaveBeenCalledOnce();
    expect(wrapper.find("[data-test='mobile-pair-scanner']").exists()).toBe(false);
    expect(wrapper.find(".error-text").exists()).toBe(false);
  });

  it("does not open the pairing scanner when camera access is denied", async () => {
    checkBarcodeScannerPermissions.mockResolvedValue("prompt");
    requestBarcodeScannerPermissions.mockResolvedValue("denied");

    await scanFromMachines();

    expect(scanPairingQr).not.toHaveBeenCalled();
    expect(wrapper?.get(".error-text").text()).toContain("Allow camera access in Settings");
  });

  it("renders structured native scanner failures as readable copy", async () => {
    scanPairingQr.mockRejectedValue({ message: "The camera is unavailable." });

    await scanFromMachines();

    expect(wrapper?.get(".error-text").text()).toBe("The camera is unavailable.");
  });

  it("rejects expired pairing codes before contacting the host", async () => {
    scanPairingQr.mockResolvedValue({
      content: pairingPayload({ expires_at: Math.floor(Date.now() / 1000) - 1 }),
    });

    await scanFromMachines();

    expect(claimPairingSession).not.toHaveBeenCalled();
    expect(wrapper?.get(".error-text").text()).toContain("pairing code expired");
    expect(invoke).not.toHaveBeenCalledWith("keychain_set_api_key", expect.anything());
  });

  it("rejects a pairing claim from a different server identity", async () => {
    scanPairingQr.mockResolvedValue({ content: pairingPayload() });
    claimPairingSession.mockResolvedValue({
      api_key: "paired-key",
      instance_id: "wrong-host",
      hostname: "impostor",
    });

    await scanFromMachines();

    expect(claimPairingSession).toHaveBeenCalledWith("http://pair.local:7680", "one-time-token", {
      name: "Mold on iPhone",
      kind: "iphone",
    });
    expect(wrapper?.get(".error-text").text()).toContain("different Mold host");
    expect(invoke).not.toHaveBeenCalledWith("keychain_set_api_key", expect.anything());
  });

  it("stores a verified pairing key through the existing Keychain host path", async () => {
    scanPairingQr.mockResolvedValue({ content: pairingPayload() });
    claimPairingSession.mockResolvedValue({
      api_key: "paired-key",
      instance_id: "pair-id",
      hostname: "pair-host",
    });
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status" && baseUrl === "http://pair.local:7680") {
        return Promise.resolve({ ...status, instance_id: "pair-id", hostname: "pair-host" });
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    await scanFromMachines();
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("keychain_set_api_key", {
        hostId: "pair-local-7680",
        apiKey: "paired-key",
      }),
    );

    const saved = JSON.parse(localStorage.getItem("mold.mobile.hosts.v1") ?? "[]");
    expect(saved).toContainEqual(
      expect.objectContaining({
        id: "pair-local-7680",
        instanceId: "pair-id",
        name: "pair-host",
        baseUrl: "http://pair.local:7680",
      }),
    );
  });

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

  it("claims a cold-launch iOS Camera pairing link through the existing Keychain path", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      value: {},
      configurable: true,
    });
    getCurrentDeepLinks.mockResolvedValue([pairingUrl()]);
    claimPairingSession.mockResolvedValue({
      api_key: "paired-key",
      instance_id: "pair-id",
      hostname: "pair-host",
    });
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status" && baseUrl === "http://pair.local:7680") {
        return Promise.resolve({ ...status, instance_id: "pair-id", hostname: "pair-host" });
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("keychain_set_api_key", {
        hostId: "pair-local-7680",
        apiKey: "paired-key",
      }),
    );

    expect(onOpenDeepLinks).toHaveBeenCalledOnce();
    expect(claimPairingSession).toHaveBeenCalledWith("http://pair.local:7680", "one-time-token", {
      name: "Mold on iPhone",
      kind: "iphone",
    });
  });

  it("stops listening for iOS pairing links when the mobile shell unmounts", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      value: {},
      configurable: true,
    });
    wrapper = mountMobileApp();
    await vi.waitFor(() => expect(onOpenDeepLinks).toHaveBeenCalledOnce());

    wrapper.unmount();
    wrapper = null;

    expect(unlistenDeepLinks).toHaveBeenCalledOnce();
  });
});

describe("MobileApp machines telemetry", () => {
  async function openMachines(): Promise<void> {
    const hostsTab = wrapper!
      .findAll("button.mobile-tab")
      .find((button) => button.text() === "Machines");
    if (!hostsTab) throw new Error("Missing Machines tab");
    await hostsTab.trigger("click");
    await flushPromises();
  }

  it("mirrors VRAM usage and queue depth on an online host card", async () => {
    apiJsonTo.mockReset().mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          gpu_info: {
            name: "RTX 4090",
            vram_total_mb: 24_000,
            vram_used_mb: 9_840,
            backend: "cuda",
          },
          queue_depth: 2,
          queue_capacity: 8,
        } satisfies ServerStatus);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await openMachines();

    const telemetry = wrapper.get("[data-test='mobile-host-telemetry']");
    expect(telemetry.get(".host-telemetry-mem").text()).toBe("9.8 / 24.0 GB");
    expect(telemetry.get(".host-telemetry-queue").text()).toBe("queue 2");
    expect(telemetry.get(".meter").attributes("aria-valuenow")).toBe("41");
  });

  it("aggregates every GPU on a host card", async () => {
    apiJsonTo.mockReset().mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          gpu_info: {
            name: "RTX 3090",
            vram_total_mb: 24_000,
            vram_used_mb: 10_000,
            backend: "cuda",
          },
          gpus: [
            {
              ordinal: 0,
              name: "RTX 3090",
              vram_total_bytes: 24_000_000_000,
              vram_used_bytes: 10_000_000_000,
              state: "generating",
            },
            {
              ordinal: 1,
              name: "NVIDIA B200",
              vram_total_bytes: 80_000_000_000,
              vram_used_bytes: 20_000_000_000,
              state: "idle",
            },
          ],
          queue_depth: 1,
        } satisfies ServerStatus);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await openMachines();

    const telemetry = wrapper.get("[data-test='mobile-host-telemetry']");
    expect(telemetry.get(".host-telemetry-mem").text()).toBe("30.0 / 104.0 GB");
    expect(telemetry.get(".meter").attributes("aria-valuenow")).toBe("29");
  });

  it("shows dashes and a zero queue when the host omits GPU info", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await openMachines();

    const telemetry = wrapper.get("[data-test='mobile-host-telemetry']");
    expect(telemetry.get(".host-telemetry-mem").text()).toBe("—");
    expect(telemetry.get(".host-telemetry-queue").text()).toBe("queue 0");
  });

  it("disconnects without forgetting and reconnects only on request", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await openMachines();
    await wrapper.get("[data-test='mobile-host-row']").trigger("click");
    await wrapper.get("[data-test='host-detail-disconnect']").trigger("click");
    await flushPromises();

    let saved = JSON.parse(localStorage.getItem("mold.mobile.hosts.v1") ?? "[]");
    expect(saved[0]).toMatchObject({ connected: false });
    expect(wrapper.get("[data-test='mobile-host-detail'] .host-chip").text()).toBe("disconnected");
    expect(wrapper.find("[data-test='host-detail-reconnect']").exists()).toBe(true);

    await wrapper.get("[data-test='host-detail-reconnect']").trigger("click");
    await flushPromises();
    saved = JSON.parse(localStorage.getItem("mold.mobile.hosts.v1") ?? "[]");
    expect(saved[0]).toMatchObject({ connected: true });
  });
});
