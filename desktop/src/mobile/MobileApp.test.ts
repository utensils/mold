import { flushPromises, mount, type DOMWrapper, type VueWrapper } from "@vue/test-utils";
import { createPinia, type Pinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { GalleryImage, ModelEntry, ServerStatus } from "../lib/api/types";
import { applyModelDefaults, newGenerateForm, type GenerateForm } from "../lib/generateForm";
import { saveGenerationTemplate } from "../lib/generationTemplates";
import { MOBILE_GENERATION_TEMPLATES_STORAGE_KEY } from "./mobileTemplateStorage";
import {
  MOBILE_DURABLE_GENERATIONS_KEY,
  createMobileDurableGenerationRecovery,
  reduceMobileDurableGenerationRecovery,
} from "./mobileGenerationRecovery";
import {
  loadCachedGallery,
  storeCachedGallery,
  storeCachedGalleryMedia,
  storeCachedHostPresentation,
} from "./galleryCache";
import { clearSessionScrollForTests, sessionScrollPosition } from "@studio/lib/libraryOrganization";
import { thumbnailTier } from "@studio/lib/thumbnailPersistentCache";

const {
  invoke,
  apiFetchTo,
  apiJsonTo,
  sseStream,
  streamableMediaUrl,
  evictMedia,
  applyH3BoundaryFit,
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
  isNativeAndroidRuntime,
  isNativeIOSRuntime,
} = vi.hoisted(() => ({
  invoke: vi.fn(),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  sseStream: vi.fn(),
  streamableMediaUrl: vi.fn(),
  evictMedia: vi.fn(),
  applyH3BoundaryFit: vi.fn(),
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
  isNativeAndroidRuntime: vi.fn(),
  isNativeIOSRuntime: vi.fn(),
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
vi.mock("../lib/sourceFitPreprocess", () => ({ applyH3BoundaryFit, applySourceFitPreprocess }));
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
vi.mock("./platform", () => ({ isNativeAndroidRuntime, isNativeIOSRuntime }));

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
vi.mock("../lib/generationRecovery", async (importOriginal) => {
  const original = await importOriginal<typeof import("../lib/generationRecovery")>();
  return {
    ...original,
    reconcileInterruptedGenerationJobs: (
      jobs: Parameters<typeof original.reconcileInterruptedGenerationJobs>[0],
      options: Parameters<typeof original.reconcileInterruptedGenerationJobs>[1],
    ) => original.reconcileInterruptedGenerationJobs(jobs, { ...options, pollIntervalMs: 0 }),
  };
});

import MobileApp from "./MobileApp.vue";
import IdentityPhotoWell from "@studio/components/IdentityPhotoWell.vue";
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
  // Newer than the submission under test: gallery recovery is age-bounded.
  // A getter, so a late-running test still reads a stamp after its own submit.
  get timestamp() {
    return Math.floor(Date.now() / 1000) + 5;
  },
  format: "mp4",
  media_version: "mobile-test-v1",
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

function cachedThumbnailRef(item: GalleryImage, hostId = "studio-instance") {
  return {
    hostId,
    filename: item.filename,
    mediaVersion: item.media_version!,
    tier: thumbnailTier(),
  };
}

function cachedThumbnailBlob(label = "cached thumbnail"): Blob {
  return new Blob([Uint8Array.of(0x89, 0x50, 0x4e, 0x47), label], { type: "image/png" });
}

let durableJobSequence = 0;

/** The host's answer to `POST /api/generation-batches`: one accepted child per
 * submitted request. Every print is admitted here — there is no attached
 * generation stream left to fall back to. */
function durableBatchResponse(
  init: RequestInit | undefined,
  overrides: { instanceId?: string; state?: string; filenames?: readonly string[] } = {},
): Record<string, unknown> {
  const body = JSON.parse(String(init?.body)) as {
    client_batch_id: string;
    requests: unknown[];
  };
  const now = Date.now();
  const state = overrides.state ?? durableAdmission.state;
  const filenames = overrides.filenames ?? durableAdmission.filenames;
  return {
    id: `batch-${body.client_batch_id}`,
    client_batch_id: body.client_batch_id,
    instance_id: overrides.instanceId ?? status.instance_id,
    durable: true,
    children: body.requests.map((_request, index) => ({
      index: index + 1,
      job_id: `durable-job-${++durableJobSequence}`,
      state,
      created_at_ms: now,
      updated_at_ms: now + 1,
      ...(state === "complete"
        ? {
            completed_at_ms: now + 2,
            result: { filename: filenames[index] ?? filenames[0] ?? "saved print.png" },
          }
        : {}),
    })),
  };
}

/**
 * How the shared host answers admission. A print has no held stream any more,
 * so a test that needs a finished print says so here before it submits — the
 * completion is metadata-only and its media comes from the host gallery.
 */
let durableAdmission: { state: string; filenames: readonly string[] } = {
  state: "accepted",
  filenames: [],
};

/**
 * What a current machine answers for anything a test does not handle itself.
 * Every print now goes through durable admission, so a test that only cares
 * about the request it submits still needs the machine to advertise the queue
 * and accept the batch.
 */
function durableApiFallback(
  path: string,
  init?: RequestInit,
  callTarget?: unknown,
): Promise<unknown> {
  // Scoped to the primary machine on purpose: a test that probes a SECOND
  // host's reachability still gets the rejection it is pinning.
  if (
    path === "/api/capabilities" &&
    (callTarget as { baseUrl?: string } | undefined)?.baseUrl === target.baseUrl
  ) {
    return Promise.resolve(durableQueueCapabilities);
  }
  if (path === "/api/generation-batches" && init?.method === "POST") {
    return Promise.resolve(durableBatchResponse(init));
  }
  // An auto-chained sequence is created as a durable chain job before its own
  // event stream opens.
  if (path === "/api/chain-jobs" && init?.method === "POST") {
    return Promise.resolve({ job_id: "chain-job-1" });
  }
  if (path === "/api/generation-batches/status") {
    return Promise.resolve({
      instance_id: status.instance_id,
      batches: [],
      missing: { client_batch_ids: [], batch_ids: [] },
    });
  }
  return Promise.reject(new Error(`Unexpected API path: ${path}`));
}

/** Finish every print this submission admits, naming its saved gallery file. */
function admitCompletedPrints(...filenames: string[]): void {
  durableAdmission = { state: "complete", filenames };
}

/** Every request body the phone actually admitted, in submission order. A
 * print carries no held stream any more, so this is where its wire shape is
 * read. */
function admittedRequests(): Array<Record<string, unknown>> {
  return admittedBatches().flatMap((body) => body.requests as Array<Record<string, unknown>>);
}

/** The exact authenticated machine each admission was posted to. */
function admissionTargets(): unknown[] {
  return admissionCalls().map((call) => call[0]);
}

/** The admission bodies themselves, for tests that assert chunking. */
function admittedBatches(): Array<{ client_batch_id: string; requests: unknown[] }> {
  return admissionCalls().map((call) => JSON.parse(String((call[2] as RequestInit).body)));
}

/** Every `POST /api/generation-batches` the phone actually made. */
function admissionCalls(): unknown[][] {
  return (apiJsonTo.mock.calls as unknown[][]).filter(
    (call) =>
      call[1] === "/api/generation-batches" &&
      (call[2] as RequestInit | undefined)?.method === "POST",
  );
}

/**
 * What a current machine advertises. The batch chunk limit's presence IS the
 * durable-generation contract, and the media block is what lets a print
 * carrying a source image, an identity photo, or H3 references be queued at
 * all — without it every media-bearing print is refused by name.
 */
/**
 * An image checkpoint. A durable print takes its output format from the
 * REQUEST — there is no SSE completion frame to carry one — so a test that
 * asserts the still tile must develop on a still model rather than the
 * shared video fixture.
 */
const stillModel: ModelEntry = {
  ...model,
  name: "flux-dev:q8",
  family: "flux",
  description: "Image model",
};

/**
 * Hold every durable admission so a test can settle its prints at a moment of
 * its own choosing — the durable stand-in for driving an SSE completion frame.
 */
function heldAdmissions(): {
  count: () => number;
  settleAt: (
    index: number,
    options?: {
      state?: string;
      filename?: string | null;
      error?: string;
      children?: ReadonlyArray<{ state?: string; filename?: string | null; error?: string }>;
    },
  ) => void;
  settle: (state?: string, filenames?: readonly string[]) => void;
} {
  const pending: Array<{ init: RequestInit; resolve: (value: unknown) => void }> = [];
  const base = apiJsonTo.getMockImplementation()!;
  apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
    if (path === "/api/generation-batches" && init?.method === "POST") {
      return new Promise((resolve) => pending.push({ init, resolve }));
    }
    return base(callTarget, path, init);
  });
  /** Answer ONE held admission, in submission order. */
  function settleAt(
    index: number,
    options: {
      state?: string;
      filename?: string | null;
      error?: string;
      children?: ReadonlyArray<{ state?: string; filename?: string | null; error?: string }>;
    } = {},
  ): void {
    const entry = pending[index];
    if (!entry) throw new Error(`No held admission at ${index}`);
    pending[index] = undefined as never;
    const parsed = JSON.parse(String(entry.init.body)) as {
      client_batch_id: string;
      requests: unknown[];
    };
    const state = options.state ?? "complete";
    const now = Date.now();
    entry.resolve({
      id: `batch-${parsed.client_batch_id}`,
      client_batch_id: parsed.client_batch_id,
      instance_id: status.instance_id,
      durable: true,
      children: parsed.requests.map((_request, offset) => {
        const child = options.children?.[offset] ?? options;
        const childState = child.state ?? state;
        return {
          index: offset + 1,
          job_id: `durable-job-${index + 1}-${offset + 1}`,
          state: childState,
          created_at_ms: now,
          updated_at_ms: now + 1,
          ...(child.error === undefined ? {} : { error: child.error }),
          ...(childState === "complete" || childState === "failed" || childState === "cancelled"
            ? { completed_at_ms: now + 2 }
            : {}),
          ...(childState === "complete"
            ? {
                result:
                  child.filename === null ? {} : { filename: child.filename ?? "saved print.png" },
              }
            : {}),
        };
      }),
    });
  }

  return {
    count: () => pending.length,
    settleAt,
    settle(state = "complete", filenames: readonly string[] = []) {
      for (let index = 0; index < pending.length; index += 1) {
        if (!pending[index]) continue;
        settleAt(index, {
          state,
          ...(filenames[index] === undefined ? {} : { filename: filenames[index]! }),
        });
      }
    },
  };
}

/** Serve one still checkpoint, leaving every other path on the default mock. */
function serveStillModel(): void {
  const base = apiJsonTo.getMockImplementation()!;
  apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
    if (path === "/api/models") return Promise.resolve([stillModel]);
    return base(callTarget, path, init);
  });
}

const durableQueueCapabilities = {
  video_upscale: { available: true },
  queue: { heterogeneous_batch_max_outputs: 64 },
  durable_media: {
    protocol_version: 2,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    private_h3: true,
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
    onOpen?: (response?: Response) => void;
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

function mobileTouch(type: string, x: number, y: number, ended = false): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  const point = { identifier: 17, clientX: x, clientY: y };
  Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
  Object.defineProperty(event, "changedTouches", { value: [point] });
  return event;
}

async function swipeMobileContent(fromX: number, toX: number, y = 180): Promise<void> {
  const content = wrapper!.get(".mobile-content").element;
  content.dispatchEvent(mobileTouch("touchstart", fromX, y));
  content.dispatchEvent(mobileTouch("touchmove", toX, y));
  content.dispatchEvent(mobileTouch("touchend", toX, y, true));
  await flushPromises();
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
  clearSessionScrollForTests();
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
  durableJobSequence = 0;
  durableAdmission = { state: "accepted", filenames: [] };
  apiJsonTo
    .mockReset()
    .mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/generation-batches" && init?.method === "POST") {
        return Promise.resolve(durableBatchResponse(init));
      }
      if (path === "/api/generation-batches/status") {
        return Promise.resolve({
          instance_id: status.instance_id,
          batches: [],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
  apiFetchTo
    .mockReset()
    .mockImplementation(async (requestTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/chain-jobs" && init?.method === "POST") {
        const body = await apiJsonTo(requestTarget, path, init);
        return new Response(JSON.stringify(body));
      }
      return new Response(cachedThumbnailBlob("network thumbnail"), {
        status: 200,
        headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
      });
    });
  openStreams.length = 0;
  sseStream.mockReset().mockImplementation(
    (
      path: string,
      options: {
        body: Record<string, unknown>;
        headers?: Record<string, string>;
        signal: AbortSignal;
        onOpen?: (response?: Response) => void;
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
  applyH3BoundaryFit.mockReset().mockImplementation((state) => Promise.resolve(state));
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
  isNativeAndroidRuntime.mockReset().mockReturnValue(false);
  isNativeIOSRuntime.mockReset().mockReturnValue(false);
  objectUrlSequence = 0;
  URL.createObjectURL = vi.fn(() => `blob:thumbnail-${++objectUrlSequence}`);
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.themeFamily;
  delete (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  Reflect.deleteProperty(globalThis, "indexedDB");
  Reflect.deleteProperty(document, "elementsFromPoint");
});

describe("MobileApp sequence generation", () => {
  /** A machine serving one sequence-capable model, with Create already on
   *  Sequence output — the only shape that still opens a placement probe. */
  function serveSequence(): void {
    const sequenceModel = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      default_steps: 7,
      default_guidance: 1,
    };
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
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
      if (path === "/api/chain-jobs/sequence-job-1") return new Promise(() => {});
      return durableApiFallback(path, init, callTarget);
    });
  }

  async function fillSequenceClips(): Promise<void> {
    const prompts = wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("A responsive placement check");
    await prompts[1]!.setValue("A second clip so the rail is valid");
    await flushPromises();
  }

  it("enters Wan on its default duration after an H3 model", async () => {
    const h3: ModelEntry = {
      ...model,
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      source_image: "required",
      default_frames: 124,
      default_fps: 24,
    };
    const wan: ModelEntry = {
      ...model,
      name: "wan22-ti2v-5b:turbo",
      family: "wan",
      default_frames: 121,
      default_fps: 24,
      frame_step: 4,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([h3, wan]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(liveForm.frames).toBe(124);

    await fieldControl("Model").setValue(wan.name);
    await flushPromises();

    expect(liveForm.frames).toBe(121);
    expect(liveForm.fps).toBe(24);
    expect(wrapper.text()).not.toContain("Frames must be 4n+1");
  });

  it("retires dormant original-prompt provenance when a new prompt is authored", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    const form = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    form.originalPrompt = "the source for an earlier generated print";

    await fieldControl("Prompt").setValue("a completely new print");

    expect(form.originalPrompt).toBeNull();
  });

  it("preserves quick-expansion provenance when the rewrite is hand-edited", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    const form = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    await fieldControl("Prompt").setValue("a hand-edited lighthouse rewrite");
    await flushPromises();

    expect(form.originalPrompt).toBe("a lighthouse");
    expect(wrapper.find("[data-test='mobile-quick-expansion-stale']").exists()).toBe(true);
  });

  it("lets the user cancel a sequence placement preview before anything is queued", async () => {
    // Only a SEQUENCE is planned before it is created; a print is admitted
    // durably and never opens a placement probe.
    const preview = deferred<ReturnType<typeof plannedPlacement>>();
    previewChainPlacement.mockReturnValueOnce(preview.promise);
    serveSequence();
    wrapper = mountMobileApp();
    await flushPromises();
    await fillSequenceClips();

    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));

    const button = wrapper.get("[data-test='mobile-generate-sequence']");
    expect(button.text()).toBe("Cancel · Preparing sequence…");
    expect(button.attributes("disabled")).toBeUndefined();
    await button.trigger("click");
    expect(previewChainPlacement).toHaveBeenCalledTimes(1);

    const previewOptions = previewChainPlacement.mock.calls[0]?.[3];
    expect(previewOptions?.signal?.aborted).toBe(true);

    preview.resolve(plannedPlacement());
    await flushPromises();
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/chain-jobs")).toHaveLength(0);
    expect(button.text()).toContain("Generate sequence");
  });

  it("releases iOS background execution exactly once after placement cancellation settles", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-placement-cancel");
      }
      return Promise.resolve(null);
    });
    const preview = deferred<ReturnType<typeof plannedPlacement>>();
    previewChainPlacement.mockReturnValueOnce(preview.promise);
    serveSequence();
    wrapper = mountMobileApp();
    await flushPromises();
    await fillSequenceClips();

    const button = wrapper.get("[data-test='mobile-generate-sequence']");
    await button.trigger("click");
    await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));
    await button.trigger("click");
    preview.resolve(plannedPlacement());

    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
        token: "mobile-background-placement-cancel",
      }),
    );
    expect(
      invoke.mock.calls.filter(([command]) => command === "end_mobile_background_task"),
    ).toHaveLength(1);
  });

  it("holds iOS background execution through sequence placement and creation", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-sequence");
      }
      return Promise.resolve(null);
    });
    const preview = deferred<ReturnType<typeof plannedPlacement>>();
    previewChainPlacement.mockReturnValueOnce(preview.promise);
    serveSequence();
    wrapper = mountMobileApp();
    await flushPromises();
    await fillSequenceClips();

    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("begin_mobile_background_task", {
        name: "Preparing remote sequence",
      }),
    );
    expect(invoke).not.toHaveBeenCalledWith("end_mobile_background_task", expect.anything());

    preview.resolve(plannedPlacement());
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
        token: "mobile-background-sequence",
      }),
    );
  });

  it("holds iOS background execution across a print's durable admission", async () => {
    // A print no longer waits on a placement probe, so the assertion the
    // phone still owes is that the background task spans admission itself.
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-generation");
      }
      return Promise.resolve(null);
    });
    const admission = deferred<unknown>();
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") return admission.promise;
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a background-safe admission");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("begin_mobile_background_task", {
        name: "Preparing remote generation",
      }),
    );
    expect(invoke).not.toHaveBeenCalledWith("end_mobile_background_task", expect.anything());

    admission.resolve(
      durableBatchResponse({ body: JSON.stringify({ client_batch_id: "b", requests: [{}] }) }),
    );
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
        token: "mobile-background-generation",
      }),
    );
  });

  it("releases iOS background execution when sequence placement fails", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-placement-failure");
      }
      return Promise.resolve(null);
    });
    serveSequence();
    previewChainPlacement.mockRejectedValueOnce(new Error("placement unavailable"));
    wrapper = mountMobileApp();
    await flushPromises();
    await fillSequenceClips();

    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");

    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
        token: "mobile-background-placement-failure",
      }),
    );
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/chain-jobs")).toHaveLength(0);
  });

  it("cancels source preparation and releases Generate when prompt work takes authority", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    const preprocess = deferred<{ source: string | null; mask: string | null; changed: boolean }>();
    applySourceFitPreprocess.mockReturnValueOnce(preprocess.promise);
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await fieldControl("Prompt").setValue("a source-preparing prompt");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await vi.waitFor(() => expect(applySourceFitPreprocess).toHaveBeenCalledTimes(1));
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    preprocess.resolve({ source: btoa("source"), mask: null, changed: false });
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect(
      wrapper.get("[data-test='mobile-develop-button']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("releases Generate when invalidated source preparation rejects", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-stale-source");
      }
      return Promise.resolve(null);
    });
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    const preprocess = deferred<{ source: string | null; mask: string | null; changed: boolean }>();
    applySourceFitPreprocess.mockReturnValueOnce(preprocess.promise);
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await fieldControl("Prompt").setValue("a source-preparing prompt");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await vi.waitFor(() => expect(applySourceFitPreprocess).toHaveBeenCalledTimes(1));
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    preprocess.reject(new Error("stale source failure"));
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect(wrapper.text()).not.toContain("stale source failure");
    expect(
      wrapper.get("[data-test='mobile-develop-button']").attributes("disabled"),
    ).toBeUndefined();
    expect(
      invoke.mock.calls.filter(([command]) => command === "end_mobile_background_task"),
    ).toEqual([["end_mobile_background_task", { token: "mobile-background-stale-source" }]]);
  });

  it("removes capability-restricted models from the picker before submission", async () => {
    const restrictedModel: ModelEntry = {
      ...model,
      name: "AutoencoderKLMiniMaxH3",
      family: "minimax-h3",
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
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
      return durableApiFallback(path, init, callTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
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
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();
    const activity = wrapper.get("[data-test='shared-live-activity']");
    expect(activity.text()).toContain("Studio");
    expect(activity.text()).toContain("flux-dev:q4");
    expect(localStorage.getItem("mold.mobile.live-activity.v1")).not.toContain(target.apiKey);
  });

  it("interleaves local and recovered work newest-first", async () => {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/generation-batches" && init?.method === "POST") {
        return Promise.resolve(durableBatchResponse(init));
      }
      if (path === "/api/generation-batches/status") {
        return Promise.resolve({
          instance_id: status.instance_id,
          batches: [],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: Date.now(),
          items: [
            {
              id: "newer-running",
              kind: "generation",
              phase: "running",
              model: "newer developing print",
              created_at_unix_ms: Date.now() + 60_000,
              updated_at_unix_ms: Date.now() + 60_000,
              can_cancel: false,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("older queued print");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    const text = wrapper.get(".mobile-generation-queue-list").text();
    expect(text.indexOf("newer developing print")).toBeLessThan(text.indexOf("older queued print"));
  });

  it("swipes a queued item from another client to exact-host pause and resume", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status")
        return Promise.resolve({ ...status, queue_paused: false, queue_capacity: 2 });
      if (path === "/api/capabilities")
        return Promise.resolve({
          queue: {
            can_pause: true,
            can_pause_job: true,
            heterogeneous_batch_max_outputs: 8,
          },
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/queue?limit=2") return Promise.resolve({ entries: [] });
      if (path === "/api/activity")
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: 10,
          items: [
            {
              id: "foreign-queued",
              kind: "generation",
              phase: "queued",
              model: model.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: true,
            },
          ],
        });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const row = wrapper.get(".live-activity-row");
    const touch = (type: string, x: number, ended = false) => {
      const event = new Event(type, { bubbles: true, cancelable: true });
      const point = { clientX: x, clientY: 100 };
      Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
      Object.defineProperty(event, "changedTouches", { value: [point] });
      return event;
    };
    row.element.dispatchEvent(touch("touchstart", 260));
    row.element.dispatchEvent(touch("touchmove", 160));
    row.element.dispatchEvent(touch("touchend", 160, true));
    await flushPromises();

    expect(row.classes()).toContain("live-activity-row--actions-open");
    await row.get("[data-test='mobile-fleet-queue-control']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/foreign-queued/pause", {
      method: "POST",
    });
  });

  it("resumes a restart-paused auto-chain through its durable chain authority", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: 10,
          items: [
            {
              id: "restart-paused-auto-chain",
              kind: "generation",
              execution: "chain",
              phase: "paused",
              model: model.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: true,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    const row = wrapper.get(".live-activity-row");
    const touch = (type: string, x: number, ended = false) => {
      const event = new Event(type, { bubbles: true, cancelable: true });
      const point = { clientX: x, clientY: 100 };
      Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
      Object.defineProperty(event, "changedTouches", { value: [point] });
      return event;
    };
    row.element.dispatchEvent(touch("touchstart", 260));
    row.element.dispatchEvent(touch("touchmove", 160));
    row.element.dispatchEvent(touch("touchend", 160, true));
    await flushPromises();

    const control = row.get("[data-test='mobile-fleet-queue-control']");
    expect(control.text()).toBe("Resume");
    await control.trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(
      target,
      "/api/chain-jobs/restart-paused-auto-chain/resume",
      { method: "POST" },
    );
  });

  it("cancels a running auto-chain through its durable chain authority", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: 10,
          items: [
            {
              id: "running-auto-chain",
              kind: "generation",
              execution: "chain",
              phase: "running",
              model: model.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: true,
            },
          ],
        });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    const row = wrapper.get(".live-activity-row");
    const touch = (type: string, x: number, ended = false) => {
      const event = new Event(type, { bubbles: true, cancelable: true });
      const point = { clientX: x, clientY: 100 };
      Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
      Object.defineProperty(event, "changedTouches", { value: [point] });
      return event;
    };
    row.element.dispatchEvent(touch("touchstart", 260));
    row.element.dispatchEvent(touch("touchmove", 160));
    row.element.dispatchEvent(touch("touchend", 160, true));
    await flushPromises();

    const control = row.get("[data-test='mobile-fleet-queue-control']");
    expect(control.text()).toBe("Cancel");
    await control.trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/chain-jobs/running-auto-chain/cancel", {
      method: "POST",
    });
  });

  it("loads a tapped server-owned generation into Create like desktop", async () => {
    const pagedStatus = { ...status, queue_capacity: 3 };
    const metadata = {
      prompt: "a lighthouse beyond the red dunes",
      title: "Queue lighthouse study",
      negative_prompt: "fog",
      model: model.name,
      seed: 0,
      steps: 18,
      guidance: 2.5,
      width: 1024,
      height: 576,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(pagedStatus);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: 10,
          items: [
            {
              id: "foreign-job",
              kind: "generation",
              phase: "running",
              model: model.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: true,
            },
          ],
        });
      }
      if (path === "/api/queue?limit=3") {
        return Promise.resolve({
          entries: [
            {
              id: "foreign-job",
              model: model.name,
              state: "running",
              started_at_unix_ms: 1,
              position: 0,
              metadata,
              seed_pinned: false,
            },
          ],
          live_only_entries: [],
          page: { limit: 3, offset: 0, returned: 1 },
          plan: null,
        });
      }
      if (path === "/api/queue/foreign-job/preview") {
        return Promise.resolve({
          preview_image: "UFJFVklFVw==",
          step: 7,
          total: 18,
          stage: "Denoising",
          updated_at_ms: 1,
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();

    const row = wrapper.get("[data-test='live-activity-select-studio-id:generation:foreign-job']");
    expect(row.element.tagName).toBe("BUTTON");
    await row.trigger("click");
    await flushPromises();

    expect(fieldControl("Prompt").element).toHaveProperty(
      "value",
      "a lighthouse beyond the red dunes",
    );
    expect(fieldControl("Steps").element).toHaveProperty("value", "18");
    expect(fieldControl("Guidance").element).toHaveProperty("value", "2.5");
    expect(fieldControl("Negative prompt").element).toHaveProperty("value", "fog");
    expect(fieldControl("Title").element).toHaveProperty("value", "Queue lighthouse study");
    expect(wrapper.text()).toContain("New seed for every print.");
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe("Developing 7 / 18");
    expect(wrapper.get("[data-test='mobile-develop-preview']").attributes("src")).toBe(
      "data:image/png;base64,UFJFVklFVw==",
    );
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/queue?limit=3");
    expect(apiJsonTo.mock.calls.some(([, path]) => path === "/api/queue")).toBe(false);
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/queue/foreign-job/preview",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("routes a tapped server-owned auto-chain generation through its durable chain", async () => {
    const pinia = createPinia();
    const priorModel: ModelEntry = {
      ...model,
      name: "flux-dev:q8",
      family: "flux",
      default_guidance: 4,
    };
    const sequenceModel: ModelEntry = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([priorModel, sequenceModel]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: status.instance_id,
          observed_at_unix_ms: 10,
          items: [
            {
              id: "foreign-auto-chain",
              kind: "generation",
              execution: "chain",
              phase: "running",
              model: sequenceModel.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: true,
            },
          ],
        });
      }
      if (path === "/api/queue") return Promise.resolve({ entries: [], plan: null });
      if (path === "/api/chain-jobs/foreign-auto-chain") {
        return Promise.resolve({
          id: "foreign-auto-chain",
          state: "running",
          model: sequenceModel.name,
          stage_count: 2,
          current_stage: 0,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          error: null,
          ephemeral: true,
          stages: [],
          script: {
            schema: "mold.chain.v1",
            chain: {
              model: sequenceModel.name,
              width: 1024,
              height: 576,
              fps: 24,
              steps: 18,
              guidance: 2.5,
            },
            stages: [
              { prompt: "A lighthouse wakes", frames: 25, transition: "smooth" },
              { prompt: "The beam crosses red dunes", frames: 33, transition: "smooth" },
            ],
          },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    await flushPromises();

    const autoChainRow = wrapper.get(
      "[data-test='live-activity-select-studio-id:generation:foreign-auto-chain']",
    );
    await autoChainRow.trigger("click");
    await flushPromises();

    const draft = useSequenceDraftStore(pinia);
    expect(draft.output).toBe("sequence");
    expect(draft.clips.map((clip) => clip.prompt)).toEqual([
      "A lighthouse wakes",
      "The beam crosses red dunes",
    ]);
    expect(wrapper.find("[data-test='mobile-sequence-composer']").exists()).toBe(true);
    expect(fieldControl("Video model").element).toHaveProperty("value", sequenceModel.name);
    // The fixed-CFG sentence is the host's own control note now. This fixture
    // advertises no generation profile, so the phone renders nothing rather
    // than composing copy for a value it did not choose — see
    // MobileSharedParams.test.ts for the note-bearing cases.
    expect(wrapper.text()).not.toContain("Distilled recipe fixes CFG");
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/chain-jobs/foreign-auto-chain");
    expect(apiJsonTo.mock.calls.some(([, path]) => path === "/api/queue?limit=3")).toBe(false);
  });

  it("refuses to restore a stale queue row after the server instance changes", async () => {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: "replaced-studio-id",
          observed_at_unix_ms: 10,
          items: [
            {
              id: "colliding-job-id",
              kind: "generation",
              phase: "running",
              model: model.name,
              created_at_unix_ms: 1,
              updated_at_unix_ms: 9,
              can_cancel: false,
            },
          ],
        });
      }
      if (path === "/api/queue") {
        return Promise.reject(new Error("queue must not be read from the replacement instance"));
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();
    const queueReads = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/queue").length;

    await wrapper
      .get("[data-test='live-activity-select-studio-id:generation:colliding-job-id']")
      .trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "different Mold server instance",
    );
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/queue")).toHaveLength(
      queueReads,
    );
  });

  it("queues a durable two-clip sequence on the selected Keychain-authenticated host", async () => {
    const sequenceModel = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      default_steps: 7,
      default_guidance: 1,
    };
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-sequence");
      }
      return Promise.resolve(null);
    });
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockImplementation(
      async (requestTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/chain-jobs" && init?.method === "POST") {
          const body = await apiJsonTo(requestTarget, path, init);
          return new Response(JSON.stringify(body), {
            headers: {
              "x-mold-request-warning":
                "Reference timing was adjusted; the sequence still rendered.",
            },
          });
        }
        return { blob: () => Promise.resolve(new Blob(["thumbnail"])) } as Response;
      },
    );

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
    expect(invoke).toHaveBeenCalledWith("begin_mobile_background_task", {
      name: "Preparing remote sequence",
    });
    expect(invoke).not.toHaveBeenCalledWith("end_mobile_background_task", expect.anything());
    sequenceForm.strength = 0.12;
    await prompts[0]!.setValue("A later edit that belongs to the next submission");
    finishPreview({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "unsupported",
      candidate: null,
    });
    await flushPromises();
    expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
      token: "mobile-background-sequence",
    });

    expect(previewChainPlacement).toHaveBeenCalledWith(
      target,
      expect.anything(),
      1,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
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
    const advisory = wrapper.get("[data-test='mobile-request-advisories']");
    expect(advisory.text()).toContain(
      "Reference timing was adjusted; the sequence still rendered.",
    );
    await advisory.get("[data-test='mobile-request-advisories-dismiss']").trigger("click");
    expect(wrapper.find("[data-test='mobile-request-advisories']").exists()).toBe(false);
  });

  it("parks a retained opening image when the checkpoint reads no source image", async () => {
    // The well is hidden for an advertised text-to-video checkpoint, so the
    // request must not ship the image the user can no longer see or remove.
    const sequenceModel = {
      ...model,
      name: "wan22-t2v-a14b:q5",
      family: "wan",
      default_steps: 20,
      default_guidance: 5,
      default_frames: 81,
      default_fps: 16,
      source_image: "unsupported",
      supports_sequence: true,
    };
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.resolve({
          model: sequenceModel.name,
          frames_per_clip_cap: 81,
          frames_per_clip_recommended: 81,
          max_stages: 8,
          max_total_frames: 648,
          fade_frames_max: 32,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "q5",
          supports_audio: false,
          supports_sequence: true,
        });
      }
      if (path === "/api/chain-jobs" && init?.method === "POST") {
        return Promise.resolve({ job_id: "sequence-job-2" });
      }
      if (path === "/api/chain-jobs/sequence-job-2") {
        return new Promise(() => {});
      }
      return durableApiFallback(path, init, callTarget);
    });
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "unsupported",
      candidate: null,
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: btoa("stale opening") };
    const prompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("A paper boat crosses a moonlit pond");
    await prompts[1]!.setValue("Fireflies gather as the sky brightens");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-sequence-source-disclosure']").exists()).toBe(false);
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const createCall = apiJsonTo.mock.calls.find(
      (call) => call[1] === "/api/chain-jobs" && (call[2] as RequestInit)?.method === "POST",
    );
    expect(createCall).toBeDefined();
    const request = JSON.parse(String((createCall?.[2] as RequestInit)?.body));
    expect(request.stages[0]).not.toHaveProperty("source_image");
    expect(String((createCall?.[2] as RequestInit)?.body)).not.toContain(btoa("stale opening"));
  });

  it("cancels the exact sequence when cancellation arrives before its id", async () => {
    const sequenceModel = {
      ...model,
      name: "ltx-video-0.9.8-2b-distilled:bf16",
      family: "ltx-video",
      default_steps: 7,
      default_guidance: 1,
    };
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "begin_mobile_background_task") {
        return Promise.resolve("mobile-background-sequence-cancel");
      }
      return Promise.resolve(null);
    });
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    let finishCreate!: (value: { job_id: string }) => void;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
        return new Promise((resolve) => (finishCreate = resolve));
      }
      return durableApiFallback(path, init, callTarget);
    });
    previewChainPlacement.mockResolvedValue({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "unsupported",
      candidate: null,
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const prompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("opening");
    await prompts[1]!.setValue("ending");

    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await vi.waitFor(() => expect(finishCreate).toBeTypeOf("function"));
    const create = apiJsonTo.mock.calls.find(
      (call) => call[1] === "/api/chain-jobs" && (call[2] as RequestInit)?.method === "POST",
    );
    expect((create?.[2] as RequestInit).signal).toBeUndefined();
    const operationId = new Headers((create?.[2] as RequestInit).headers).get(
      "x-mold-operation-id",
    );
    expect(operationId).toMatch(/^[0-9a-f-]{36}$/);
    const cancel = wrapper.get("[data-test='mobile-generate-sequence']");
    expect(cancel.text()).toContain("Cancel");
    await cancel.trigger("click");
    await vi.waitFor(() =>
      expect(apiFetchTo).toHaveBeenCalledWith(
        target,
        `/api/chain-jobs/${operationId}/operations/${operationId}/cancel`,
        { method: "POST", keepalive: true },
      ),
    );
    finishCreate({ job_id: "late-sequence" });
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/chain-jobs/late-sequence/cancel", {
      method: "POST",
    });
    expect(invoke).toHaveBeenCalledWith("end_mobile_background_task", {
      token: "mobile-background-sequence-cancel",
    });
    expect(localStorage.getItem("mold.mobile.sequence-job.v1")).toBeNull();
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") {
        return Promise.resolve({ ...status, instance_id: undefined });
      }
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
      return durableApiFallback(path, init, callTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve(entries);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
      return durableApiFallback(path, init, callTarget);
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
      "Choose a model before generating.",
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      if (path.startsWith("/api/capabilities/chain-limits")) {
        return Promise.reject(new Error("not a chain model"));
      }
      return durableApiFallback(path, init, callTarget);
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

    await row.get("[data-test='swipe-action-cancel']").trigger("click");
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
    expect(row.find("[data-test='swipe-action-cancel']").exists()).toBe(false);
    expect(row.find("[data-test='swipe-action-dismiss']").exists()).toBe(true);
    // The row survives for its actions, but a settled job is not active work.
    expect(wrapper.get("[data-test='mobile-queue-count']").text()).toBe("0 active");
    // A settled job has nothing left to stream.
    expect(events.options.signal.aborted).toBe(true);
    expect(localStorage.getItem("mold.mobile.sequence-job.v1")).toBeNull();

    await row.get("[data-test='swipe-action-resume']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/chain-jobs/sequence-job-1/resume", {
      method: "POST",
    });
    // Resuming re-attaches on the SAME frozen route and re-arms recovery.
    const resumed = wrapper.get("[data-test='mobile-sequence-job']");
    expect(resumed.find("[data-test='swipe-action-cancel']").exists()).toBe(true);
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
      .get("[data-test='mobile-sequence-job'] [data-test='swipe-action-dismiss']")
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

  it("submits a pinned canonical-v2 request without showing or calling placement", async () => {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "canonical-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "canonical-job",
              state: "accepted",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("queue before preparation");

    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    expect(previewChainPlacement).not.toHaveBeenCalled();
    expect(
      apiJsonTo.mock.calls.filter(([, path]) => path === "/api/generation-batches"),
    ).toHaveLength(1);
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
    expect(wrapper.text()).not.toContain("Checking placement");
    expect(wrapper.text()).toContain("Queued");
  });

  it("uses the Machines lifecycle authority for a locally submitted print", async () => {
    let activityItems: Array<Record<string, unknown>> = [];
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({ events: { available: true }, ...durableQueueCapabilities });
      }
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: "studio-id",
          observed_at_unix_ms: Date.now(),
          items: activityItems,
        });
      }
      if (path === "/api/queue") {
        return Promise.resolve({
          entries: [
            {
              id: "lifecycle-job",
              model: model.name,
              state: "queued",
              started_at_unix_ms: 1,
              position: 0,
            },
          ],
          plan: null,
        });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "lifecycle-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "lifecycle-job",
              state: "queued",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("show the real lifecycle");
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-status']").text()).toBe("NEXT UP");

    activityItems = [
      {
        id: "lifecycle-job",
        kind: "generation",
        phase: "preparing",
        model: model.name,
        created_at_unix_ms: 10,
        updated_at_unix_ms: 12,
        preparation_progress: {
          component: "Opening MiniMax H3 checkpoints",
          bytes_done: 0,
          bytes_total: 0,
        },
        can_cancel: true,
      },
    ];
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-status']").text()).toBe(
      "PREPARING · OPENING MINIMAX H3 CHECKPOINTS",
    );
    expect(wrapper.get(".mobile-generation-job").classes()).toContain(
      "mobile-generation-job--detailed-status",
    );

    activityItems = [
      {
        id: "lifecycle-job",
        kind: "generation",
        phase: "running",
        model: model.name,
        stage: "Encoding video",
        current: 15,
        total: 20,
        created_at_unix_ms: 10,
        updated_at_unix_ms: 12,
        can_cancel: true,
      },
    ];
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-status']").text()).toBe(
      "ENCODING VIDEO · 15/20",
    );
    expect(wrapper.get(".mobile-generation-job").classes()).toContain(
      "mobile-generation-job--detailed-status",
    );
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe(
      "Encoding video · 15/20",
    );
    expect(wrapper.get("[data-test='mobile-queue-count']").text()).toBe("1 active");
  });

  it("confirms and tracks the exact model download when a pinned v2 host is missing it", async () => {
    let installed = false;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve(installed ? [model] : []);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "downloaded-model-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "downloaded-model-job",
              state: "accepted",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    const pullResponse = deferred<string | null>();
    startCatalogDownload.mockReturnValue(pullResponse.promise);

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    applyModelDefaults(liveForm, model);
    await submitPrompt("download before admission");

    const sheet = wrapper.get("[data-test='mobile-missing-model-sheet']");
    expect(document.body.textContent).toContain(`${model.name} isn't installed on Studio`);
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    expect(startCatalogDownload).not.toHaveBeenCalled();

    await sheet.get("[data-test='mobile-pull-expansion']").trigger("click");
    await flushPromises();
    const downloadStream = openStreams
      .filter((stream) => stream.path === "/api/downloads/stream")
      .at(-1)!;
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: { active_jobs: [], queued: [], history: [] },
      }),
    );
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith(model.name, target, false);
    expect(sheet.text()).toContain(`Starting ${model.name} on Studio`);
    pullResponse.resolve("model-job");
    await flushPromises();

    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "enqueued", id: "model-job", model: model.name, position: 0 }),
    );
    await flushPromises();

    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "other-job", model: model.name }),
    );
    await flushPromises();
    expect(sheet.text()).not.toContain("Done");

    installed = true;
    downloadStream.options.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "model-job", model: model.name }),
    );
    await flushPromises();
    expect(
      apiJsonTo.mock.calls.filter(([, path]) => path === "/api/models").length,
    ).toBeGreaterThan(1);
    await vi.waitFor(() =>
      expect(
        wrapper
          ?.get("[data-test='mobile-missing-model-sheet']")
          .get("[data-test='mobile-retry-expansion']")
          .text(),
      ).toBe("Done"),
    );
    expect(
      apiJsonTo.mock.calls.filter(([, path]) => path === "/api/models").length,
    ).toBeGreaterThan(1);

    await wrapper
      .get("[data-test='mobile-missing-model-sheet']")
      .get("[data-test='mobile-retry-expansion']")
      .trigger("click");
    await submitPrompt("download before admission");
    expect(
      apiJsonTo.mock.calls.filter(([, path]) => path === "/api/generation-batches"),
    ).toHaveLength(1);
    expect(wrapper.get("[data-test='mobile-missing-model-sheet']").classes()).not.toContain(
      "is-open",
    );
  });

  it("offers the same missing-model recovery for a restored sequence", async () => {
    const restoredSequenceModel: ModelEntry = {
      ...model,
      name: "ltx-video-0.9.8-2b-dev:bf16",
      family: "ltx-video",
      default_frames: 25,
      default_fps: 30,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });
    const pinia = createPinia();
    wrapper = mountMobileApp(pinia);
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    applyModelDefaults(liveForm, restoredSequenceModel);
    const draft = useSequenceDraftStore(pinia);
    draft.output = "sequence";
    draft.ensureClips(restoredSequenceModel.default_frames ?? 97);
    draft.clips[0]!.prompt = "opening shot";
    draft.clips[1]!.prompt = "closing shot";
    await flushPromises();

    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-missing-model-sheet']").classes()).toContain("is-open");
    expect(document.body.textContent).toContain(
      `${restoredSequenceModel.name} isn't installed on Studio`,
    );
    expect(previewChainPlacement).not.toHaveBeenCalled();
  });

  it("keeps accepting durable prints while earlier admissions are still awaiting responses", async () => {
    const pendingAdmissions = new Map<
      string,
      (batch: {
        id: string;
        client_batch_id: string;
        instance_id: string;
        durable: true;
        children: Array<{
          index: number;
          job_id: string;
          state: "queued";
          created_at_ms: number;
          updated_at_ms: number;
        }>;
      }) => void
    >();
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return new Promise((resolve) => pendingAdmissions.set(clientBatchId, resolve));
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("first durable print");
    await submitPrompt("second durable print");

    expect(pendingAdmissions.size).toBe(2);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(2);
    expect(openStreams.filter((stream) => stream.path === "/api/events")).toHaveLength(1);
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);

    let index = 0;
    for (const [clientBatchId, resolve] of pendingAdmissions) {
      index += 1;
      resolve({
        id: `batch-${index}`,
        client_batch_id: clientBatchId,
        instance_id: "studio-id",
        durable: true,
        children: [
          {
            index: 1,
            job_id: `job-${index}`,
            state: "queued",
            created_at_ms: 10,
            updated_at_ms: 11,
          },
        ],
      });
    }
    await flushPromises();
  });

  it.each(["QuotaExceededError", "SecurityError"])(
    "admits once through the durable endpoint and warns when recovery storage raises %s",
    async (name) => {
      apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve([print]);
        if (path === "/api/capabilities") {
          return Promise.resolve({
            events: { available: true },
            ...durableQueueCapabilities,
          });
        }
        if (path === "/api/activity") {
          return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
        }
        if (path === "/api/generation-batches" && init?.method === "POST") {
          const clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
          return Promise.resolve({
            id: "storage-failure-batch",
            client_batch_id: clientBatchId,
            instance_id: "studio-id",
            durable: true,
            children: [
              {
                index: 1,
                job_id: "storage-failure-job",
                state: "queued",
                created_at_ms: 1,
                updated_at_ms: 1,
              },
            ],
          });
        }
        return durableApiFallback(path, init, callTarget);
      });

      wrapper = mountMobileApp();
      await flushPromises();
      const storage = localStorage;
      vi.stubGlobal("localStorage", {
        get length() {
          return storage.length;
        },
        clear: () => storage.clear(),
        getItem: (key: string) => storage.getItem(key),
        key: (index: number) => storage.key(index),
        removeItem: (key: string) => storage.removeItem(key),
        setItem: (key: string, value: string) => {
          if (key === MOBILE_DURABLE_GENERATIONS_KEY) {
            throw Object.assign(new Error("storage unavailable"), { name });
          }
          storage.setItem(key, value);
        },
      });

      await submitPrompt("storage must not veto this print");

      const durablePosts = apiJsonTo.mock.calls.filter(
        ([, path, init]) => path === "/api/generation-batches" && init?.method === "POST",
      );
      expect(durablePosts).toHaveLength(1);
      expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(
        0,
      );
      expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain("QUEUED");
      expect(wrapper.get("[data-test='mobile-request-advisories']").text()).toContain(
        "Recovery storage is unavailable",
      );
    },
  );

  it("posts supported source media to the durable batch lifecycle without persisting or streaming it", async () => {
    const imageModel: ModelEntry = {
      ...model,
      name: "flux-dev:fp8",
      family: "flux",
      source_image: "optional",
    };
    let admittedBody: Record<string, unknown> | null = null;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
          durable_media: {
            protocol_version: 2,
            encrypted_at_rest: true,
            generate_request_media: true,
            identity: true,
            private_h3: false,
          },
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        admittedBody = JSON.parse(String(init.body)) as Record<string, unknown>;
        const clientBatchId = admittedBody.client_batch_id as string;
        return Promise.resolve({
          id: "media-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "media-job",
              state: "queued",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("PRIVATE-DURABLE-SOURCE");
    liveForm.sourceImageName = "source.png";
    await submitPrompt("durable source print");
    await flushPromises();

    expect(admittedBody).not.toBeNull();
    expect((admittedBody!.requests as Array<Record<string, unknown>>)[0]).toMatchObject({
      source_image: btoa("PRIVATE-DURABLE-SOURCE"),
    });
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
    expect(localStorage.getItem("mold.mobile.durable-generations.v1") ?? "").not.toContain(
      btoa("PRIVATE-DURABLE-SOURCE"),
    );
  });

  it.each([
    ["commit-then-500", new ApiError("response lost after commit", 500)],
    ["disconnect", new TypeError("response lost after commit")],
  ])(
    "recovers an ambiguous durable %s POST by client UUID without retrying or streaming",
    async (_case, failure) => {
      let clientBatchId = "";
      const recoveredBatch = () => ({
        id: "recovered-batch",
        client_batch_id: clientBatchId,
        instance_id: "studio-id",
        durable: true,
        children: [
          {
            index: 1,
            job_id: "recovered-job",
            state: "queued",
            created_at_ms: 10,
            updated_at_ms: 11,
          },
        ],
      });
      apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve([print]);
        if (path === "/api/capabilities") {
          return Promise.resolve({
            events: { available: true },
            ...durableQueueCapabilities,
          });
        }
        if (path === "/api/activity") {
          return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
        }
        if (path === "/api/generation-batches" && init?.method === "POST") {
          clientBatchId = JSON.parse(String(init.body)).client_batch_id;
          return Promise.reject(failure);
        }
        if (path === `/api/generation-batches/by-client/${clientBatchId}`) {
          return Promise.resolve(recoveredBatch());
        }
        if (path === "/api/generation-batches/status" && init?.method === "POST") {
          expect(JSON.parse(String(init.body))).toEqual({ client_batch_ids: [clientBatchId] });
          return Promise.resolve({
            instance_id: "studio-id",
            batches: [recoveredBatch()],
            missing: { client_batch_ids: [], batch_ids: [] },
          });
        }
        return durableApiFallback(path, init, callTarget);
      });

      wrapper = mountMobileApp();
      await flushPromises();
      await submitPrompt("ambiguous durable print");
      await vi.waitFor(() =>
        expect(
          apiJsonTo.mock.calls.filter(([, path]) =>
            String(path).startsWith("/api/generation-batches/by-client/"),
          ),
        ).toHaveLength(1),
      );

      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/generation-batches"),
      ).toHaveLength(1);
      expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(
        0,
      );
      expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain(
        "ambiguous durable print",
      );
    },
  );

  it("hydrates a relaunch-recovered print from the host's queue row on selection", async () => {
    // A recovery persisted by a previous launch: byte-free presentation only,
    // no in-memory request. Selecting it must restore the print's REAL
    // settings from the host, never the "Recovered print" stub.
    const relaunchBatch = {
      id: "relaunch-batch-id",
      client_batch_id: "relaunch-batch",
      instance_id: "studio-id",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "relaunch-job",
          state: "running",
          created_at_ms: 100,
          updated_at_ms: 101,
        },
      ],
    };
    const recovery = reduceMobileDurableGenerationRecovery(
      createMobileDurableGenerationRecovery({
        hostId: "studio-id",
        expectedInstanceId: "studio-id",
        clientBatchId: "relaunch-batch",
        requests: [
          {
            prompt: "lost with the old process",
            model: model.name,
            width: 768,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: 7,
            batch_size: 1,
            output_format: "png",
          },
        ],
        submittedAtMs: 100,
      }),
      { type: "batch_snapshot", batch: relaunchBatch as never },
    );
    localStorage.setItem(MOBILE_DURABLE_GENERATIONS_KEY, JSON.stringify([recovery]));
    const realMetadata = {
      prompt: "the real prompt from the host",
      negative_prompt: "no fog",
      model: model.name,
      width: 768,
      height: 512,
      steps: 4,
      guidance: 3.5,
      seed: 7,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/capabilities") {
        return Promise.resolve({ events: { available: true }, ...durableQueueCapabilities });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches/status" && init?.method === "POST") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [relaunchBatch],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      if (path.startsWith("/api/queue")) {
        return Promise.resolve({
          entries: [
            {
              id: "relaunch-job",
              state: "running",
              model: model.name,
              prompt: realMetadata.prompt,
              started_at_unix_ms: 100,
              position: 0,
              metadata: realMetadata,
            },
          ],
          plan: null,
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();
    const content = wrapper.get(".mobile-content").element as HTMLElement;
    content.scrollTop = 640;
    const row = wrapper.get("[data-test='mobile-generation-job']");
    const rowButton = row.get(".mobile-generation-job");
    (rowButton.element as HTMLElement).focus();
    await rowButton.trigger("keydown", { key: "Enter" });
    await flushPromises();
    await flushPromises();

    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "the real prompt from the host",
    );
    expect(wrapper.get("[data-test='mobile-generation-job']").text()).not.toContain(
      "Recovered print",
    );
    expect(content.scrollTop).toBe(0);
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-create-heading']").element);
  });

  it("persists one pre-admission cancel tap until the exact server job id is reconciled", async () => {
    const admission = deferred<Record<string, unknown>>();
    const firstRead = deferred<Record<string, unknown>>();
    let clientBatchId = "";
    const batch = (state: "queued" | "cancelled") => ({
      id: "mobile-pre-id-batch",
      client_batch_id: clientBatchId,
      instance_id: "studio-id",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "mobile-server-job-id",
          state,
          created_at_ms: 10,
          updated_at_ms: state === "queued" ? 11 : 12,
          ...(state === "cancelled" ? { completed_at_ms: 12 } : {}),
        },
      ],
    });
    let statusReads = 0;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id;
        return admission.promise;
      }
      if (path === "/api/generation-batches/status") {
        statusReads += 1;
        if (statusReads === 1) return firstRead.promise;
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [batch("cancelled")],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("cancel before admission returns");
    await wrapper.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    expect(
      JSON.parse(localStorage.getItem("mold.mobile.durable-generations.v1") ?? "[]")[0],
    ).toMatchObject({ cancelRequestedChildIndexes: [1] });
    firstRead.resolve({
      instance_id: "studio-id",
      batches: [batch("queued")],
      missing: { client_batch_ids: [], batch_ids: [] },
    });

    await vi.waitFor(() =>
      expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/mobile-server-job-id", {
        method: "DELETE",
      }),
    );
    await vi.waitFor(() =>
      expect(wrapper!.find("[data-test='mobile-generation-queue']").exists()).toBe(false),
    );

    admission.resolve(batch("queued"));
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
  });

  it("retires a durable tracker when the event authority reports a replacement instance", async () => {
    let clientBatchId = "";
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id;
        return Promise.resolve({
          id: "mismatch-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "mismatch-job",
              state: "queued",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("old instance print");
    const events = openStreams.find((stream) => stream.path === "/api/events")!;
    events.options.onEvent("authority", JSON.stringify({ instance_id: "replacement-instance" }));
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-generation-error']").text()).toContain(
      "original server instance changed",
    );
    expect(JSON.parse(localStorage.getItem("mold.mobile.durable-generations.v1") ?? "[]")).toEqual(
      [],
    );
    expect(apiFetchTo).not.toHaveBeenCalledWith(
      expect.anything(),
      "/api/queue/mismatch-job",
      expect.anything(),
    );
  });

  it("keeps a hold's error and Retry through a same-instance resync", async () => {
    let clientBatchId = "";
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({ events: { available: true }, ...durableQueueCapabilities });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "gap-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [
            {
              index: 1,
              job_id: "gap-job",
              state: "held",
              retryable: true,
              error: "model is not downloaded",
              error_code: "UNKNOWN_MODEL",
              created_at_ms: 10,
              updated_at_ms: 11,
            },
          ],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("held across a gap");
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(true);

    // The gap's bulk read is answered by the fallback's EMPTY response, so the
    // row stays under a resync overlay — the hold it covers is still the
    // host's word.
    openStreams
      .find((stream) => stream.path === "/api/events")!
      .options.onEvent(
        "resync_required",
        JSON.stringify({ instance_id: "studio-id", missed_events: 1 }),
      );
    await flushPromises();
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Re-syncing with host",
    );
    expect(wrapper.get("[data-test='mobile-generation-held-error']").text()).toBe(
      "model is not downloaded",
    );
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(true);
  });

  it("drops a hold's error and Retry once the child is developing", async () => {
    let clientBatchId = "";
    let state = "held";
    const child = () => ({
      index: 1,
      job_id: "resumed-job",
      state,
      retryable: state === "held",
      error: state === "held" ? "model is not downloaded" : undefined,
      error_code: state === "held" ? "UNKNOWN_MODEL" : undefined,
      created_at_ms: 10,
      updated_at_ms: state === "held" ? 11 : 12,
    });
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({ events: { available: true }, ...durableQueueCapabilities });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "resumed-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [child()],
        });
      }
      if (path === "/api/generation-batches/status" && init?.method === "POST") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [
            {
              id: "resumed-batch",
              client_batch_id: clientBatchId,
              instance_id: "studio-id",
              durable: true,
              children: [child()],
            },
          ],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("held then resumed");
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(true);

    // The retry moved the child straight to running before any snapshot
    // showed it queued: the hold is over, so its error and Retry go with it.
    state = "running";
    openStreams
      .find((stream) => stream.path === "/api/events")!
      .options.onEvent("event", JSON.stringify({ type: "job_started", id: "resumed-job" }));
    await flushPromises();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-generation-held-error']").exists()).toBe(false);
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(false);
  });

  it("retries only an explicitly retryable held child on its exact authenticated host", async () => {
    let clientBatchId = "";
    const heldBatch = () => ({
      id: "retry-batch",
      client_batch_id: clientBatchId,
      instance_id: "studio-id",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "retry/job",
          state: "held",
          retryable: true,
          error: "artifact digest mismatch",
          created_at_ms: 10,
          updated_at_ms: 11,
        },
      ],
    });
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve(heldBatch());
      }
      if (path === "/api/generation-batches/status" && init?.method === "POST") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [heldBatch()],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("retry held print");

    const retry = wrapper.get("[data-test='swipe-action-retry']");
    expect(retry.text()).toBe("Retry");
    expect(retry.classes()).toContain("swipe-row__action");
    expect(wrapper.get("[data-test='mobile-generation-held-error']").text()).toBe(
      "artifact digest mismatch",
    );
    const confirmation = deferred<Response>();
    apiFetchTo.mockImplementation((_target: unknown, path: string) =>
      path === "/api/queue/retry%2Fjob/retry"
        ? confirmation.promise
        : Promise.resolve(new Response(null, { status: 204 })),
    );
    const retryClick = retry.trigger("click");
    await flushPromises();
    // While the retry is in flight the row offers no second Retry action.
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(false);
    openStreams
      .find((stream) => stream.path === "/api/events")!
      .options.onEvent("event", JSON.stringify({ type: "generation_states_committed" }));
    await flushPromises();
    expect(wrapper.find("[data-test='swipe-action-retry']").exists()).toBe(false);
    confirmation.resolve(new Response(null, { status: 202 }));
    await retryClick;
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/retry%2Fjob/retry", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: expect.any(String),
    });
    const retryRequest = vi
      .mocked(apiFetchTo)
      .mock.calls.find(([, path]) => path === "/api/queue/retry%2Fjob/retry")!;
    expect(JSON.parse(String(retryRequest[2]?.body))).toEqual({
      instance_id: "studio-id",
      batch_id: "retry-batch",
      client_batch_id: expect.any(String),
      job_id: "retry/job",
    });
  });

  it("keeps a bulk-reconciled completion when the older admission response arrives later", async () => {
    const admission = deferred<Record<string, unknown>>();
    let clientBatchId = "";
    const batch = (state: "queued" | "complete") => ({
      id: "racing-batch",
      client_batch_id: clientBatchId,
      instance_id: "studio-id",
      durable: true,
      children: [
        {
          index: 1,
          job_id: "racing-job",
          state,
          created_at_ms: 10,
          updated_at_ms: state === "complete" ? 13 : 11,
          ...(state === "complete"
            ? { completed_at_ms: 13, result: { filename: "racing.png" } }
            : {}),
        },
      ],
    });
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id;
        return admission.promise;
      }
      if (path === "/api/generation-batches/status") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [batch("complete")],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    streamableMediaUrl.mockResolvedValue("https://studio/exact-host/racing.png");

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("racing durable print");
    openStreams.find((stream) => stream.path === "/api/events")!.options.onOpen?.();
    await vi.waitFor(() =>
      expect(wrapper!.find("[data-test='mobile-generation-queue']").exists()).toBe(false),
    );
    const photoCalls = invoke.mock.calls.filter(([command]) => command === "save_image_to_photos");

    admission.resolve(batch("queued"));
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(
      invoke.mock.calls.filter(([command]) => command === "save_image_to_photos"),
    ).toHaveLength(photoCalls.length);
  });

  it("keeps an incapable telemetry-routed target failure visible after reconciliation", async () => {
    let clientBatchId = "";
    const child = (state: "accepted" | "failed") => ({
      index: 1,
      job_id: "incapable-job",
      state,
      created_at_ms: 10,
      updated_at_ms: state === "accepted" ? 11 : 12,
      ...(state === "failed"
        ? {
            completed_at_ms: 12,
            error: "Studio cannot run this model on its available device",
          }
        : {}),
    });
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        clientBatchId = JSON.parse(String(init.body)).client_batch_id as string;
        return Promise.resolve({
          id: "incapable-batch",
          client_batch_id: clientBatchId,
          instance_id: "studio-id",
          durable: true,
          children: [child("accepted")],
        });
      }
      if (path === "/api/generation-batches/status") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [
            {
              id: "incapable-batch",
              client_batch_id: clientBatchId,
              instance_id: "studio-id",
              durable: true,
              children: [child("failed")],
            },
          ],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("frozen target failure");
    openStreams.find((stream) => stream.path === "/api/events")!.options.onOpen?.();

    await vi.waitFor(() =>
      expect(wrapper!.get("[data-test='mobile-generation-summary']").text()).toContain(
        "Studio cannot run this model on its available device",
      ),
    );
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
  });

  it("chunks a prepared Batch N at the durable host limit", async () => {
    const admittedChunks: Array<Array<Record<string, unknown>>> = [];
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          ...durableQueueCapabilities,
          events: { available: true },
          queue: { heterogeneous_batch_max_outputs: 1 },
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const body = JSON.parse(String(init.body)) as {
          client_batch_id: string;
          requests: Array<Record<string, unknown>>;
        };
        admittedChunks.push(body.requests);
        return Promise.resolve({
          id: "batch-n",
          client_batch_id: body.client_batch_id,
          instance_id: "studio-id",
          durable: true,
          children: body.requests.map((_request, offset) => ({
            index: offset + 1,
            job_id: `batch-n-job-${offset + 1}`,
            state: "queued",
            created_at_ms: 10,
            updated_at_ms: 11,
          })),
        });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two durable variations");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(admittedChunks).toHaveLength(2);
    expect(admittedChunks.flat().map((request) => request.batch_size)).toEqual([1, 1]);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(2);
    expect(openStreams.filter((stream) => stream.path === "/api/events")).toHaveLength(1);
    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
  });

  it("admits a capable media-free print durably and settles it from one host event stream", async () => {
    let phase: "queued" | "running" | "complete" = "queued";
    let admittedClientBatchId = "";
    const durableBatch = () => ({
      id: "batch-1",
      client_batch_id: "client-from-request",
      instance_id: "studio-id",
      durable: true as const,
      children: [
        {
          index: 1,
          job_id: "durable-job-1",
          state: phase,
          created_at_ms: 10,
          updated_at_ms: phase === "queued" ? 11 : phase === "running" ? 12 : 13,
          ...(phase === "complete"
            ? {
                completed_at_ms: 13,
                result: { filename: "durable.png" },
              }
            : {}),
        },
      ],
    });
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          ...durableQueueCapabilities,
        });
      }
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const clientBatchId = JSON.parse(String(init.body)).client_batch_id;
        admittedClientBatchId = clientBatchId;
        return Promise.resolve({ ...durableBatch(), client_batch_id: clientBatchId });
      }
      if (path === "/api/generation-batches/status") {
        return Promise.resolve({
          instance_id: "studio-id",
          batches: [{ ...durableBatch(), client_batch_id: admittedClientBatchId }],
          missing: { client_batch_ids: [], batch_ids: [] },
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["durable-image"], { type: "image/png" })),
    } as Response);
    streamableMediaUrl.mockResolvedValue("https://studio/exact-host/durable.png");

    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("durable print");

    expect(openStreams.filter((stream) => stream.path === "/api/generate/stream")).toHaveLength(0);
    const hostEvents = openStreams.filter((stream) => stream.path === "/api/events");
    expect(hostEvents).toHaveLength(1);
    expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain("durable print");
    const persisted = localStorage.getItem("mold.mobile.durable-generations.v1") ?? "";
    expect(persisted).not.toContain("durable print");
    expect(persisted).not.toContain(target.apiKey);
    expect(persisted).not.toContain(target.baseUrl);

    phase = "running";
    hostEvents[0]!.options.onEvent(
      "event",
      JSON.stringify({ type: "job_started", id: "durable-job-1", model: model.name }),
    );
    await vi.waitFor(() =>
      expect(wrapper!.get("[data-test='mobile-queue-count']").text()).toBe("1 active"),
    );

    phase = "complete";
    hostEvents[0]!.options.onEvent(
      "event",
      JSON.stringify({ type: "job_state_committed", id: "durable-job-1" }),
    );
    await vi.waitFor(() =>
      expect(wrapper!.find("[data-test='mobile-generation-queue']").exists()).toBe(false),
    );
    expect(streamableMediaUrl).toHaveBeenCalledWith(
      "/api/gallery/image/durable.png",
      expect.objectContaining({ target }),
    );
    expect(invoke).toHaveBeenCalledWith(
      "save_image_to_photos",
      expect.objectContaining({ dataB64: expect.any(String) }),
    );
    expect(JSON.parse(localStorage.getItem("mold.mobile.durable-generations.v1") ?? "[]")).toEqual(
      [],
    );

    const photoCalls = invoke.mock.calls.filter(([command]) => command === "save_image_to_photos");
    hostEvents[0]!.options.onEvent(
      "event",
      JSON.stringify({ type: "gallery_added", filename: "durable.png" }),
    );
    await flushPromises();
    expect(
      invoke.mock.calls.filter(([command]) => command === "save_image_to_photos"),
    ).toHaveLength(photoCalls.length);
  });

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
    expect(admittedRequests()[0]?.prompt).toBe(
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
    expect(admissionTargets()[0]).toEqual(target);
    expect(admittedRequests()[0]?.prompt_transform).toEqual({
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
    expect(admittedRequests()).toHaveLength(2);
    expect(admittedRequests().map((request) => request.original_prompt)).toEqual([
      "a train in the desert",
      "a train in the desert",
    ]);
    expect(admittedRequests().map((request) => request.prompt_transform)).toEqual([
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

    expect(admittedRequests().map((request) => request.original_prompt)).toEqual([
      "a lighthouse",
      "a lighthouse",
    ]);
    expect(admittedRequests()[0]?.prompt_transform).toMatchObject({
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, route);
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
    expect(admissionTargets()[0]).toEqual(renderTarget);
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, route);
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, route);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two pinned storms");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model, catalogModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]).toMatchObject({
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
    expect(admittedRequests()[0]?.negative_prompt).toBe(
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    expect(admittedRequests().map((request) => request.prompt)).toEqual([
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel, model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    expect(wrapper.get("[data-shape='1:1']").attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("1024 × 1024 px");

    await fieldControl("Model").setValue(model.name);
    await flushPromises();
    expect(wrapper.get("[data-shape='3:2']").attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("768 × 512 px");

    await submitPrompt("use the video defaults");
    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]).toMatchObject({
      model: model.name,
      width: 768,
      height: 512,
    });
  });

  it("snapshots form and host before asynchronous source preprocessing", async () => {
    const studioModel: ModelEntry = {
      ...model,
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      source_image: "required",
      default_frames: 81,
      default_fps: 16,
      frame_step: 4,
    };
    const renderModel: ModelEntry = {
      ...studioModel,
      name: "wan22-i2v-a14b:q8",
      default_frames: 73,
    };
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
    apiJsonTo.mockImplementation(
      (apiTarget: { baseUrl: string }, path: string, init?: RequestInit) => {
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
        return durableApiFallback(path, init, apiTarget);
      },
    );

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
    await wrapper.get("[data-test='mobile-source-gallery']").trigger("click");
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

    expect(admittedRequests()).toHaveLength(1);
    expect(admissionTargets()[0]).toEqual(target);
    expect(admittedRequests()[0]).toMatchObject({
      prompt: "first prompt",
      model: studioModel.name,
      frames: 81,
    });
    expect(admittedRequests()[0]?.source_image).toMatch(/^fitted:/);
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
      "Combined generation media must be 45 MiB or smaller on this phone",
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
    apiJsonTo.mockImplementation(
      (apiTarget: { baseUrl: string }, path: string, init?: RequestInit) => {
        const remote = apiTarget.baseUrl === remoteTarget.baseUrl;
        // This machine's capability probe fails: the labels below pin that.
        if (path === "/api/capabilities") {
          return Promise.reject(new Error("capabilities unavailable"));
        }
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
        return durableApiFallback(path, init, apiTarget);
      },
    );

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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([chainModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("a continuous flight through clouds");
    await wrapper.get("[data-test='mobile-frames']").setValue("177");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-chain-cue']").text()).toContain("2 chained clips");

    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    // A sequence is CREATED as a durable chain job and followed on its own
    // event stream; the auto-expand body rides the create.
    const create = apiFetchTo.mock.calls.find((call) => call[1] === "/api/chain-jobs");
    expect(create).toBeTruthy();
    const body = JSON.parse(String((create![2] as RequestInit).body)) as Record<string, unknown>;
    expect(body).toMatchObject({
      ephemeral: true,
      model: chainModel.name,
      prompt: "a continuous flight through clouds",
      total_frames: 177,
      clip_frames: 97,
      motion_tail_frames: 17,
    });
    expect(body).not.toHaveProperty("frames");
    expect(openStreams).toHaveLength(1);
    expect(openStreams[0]?.path).toBe("/api/chain-jobs/chain-job-1/events");

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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([qwen, video]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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

  it("fits only Qwen edit's Target before submitting ordered references", async () => {
    const qwen: ModelEntry = {
      ...model,
      name: "qwen-image-edit:bf16",
      family: "qwen-image-edit",
      default_width: 1024,
      default_height: 1024,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([qwen]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    applySourceFitPreprocess.mockResolvedValueOnce({
      source: "FITTED_TARGET",
      mask: null,
      changed: true,
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.imageAttachments = ["TARGET", "REFERENCE"];
    liveForm.sourceFit = { mode: "crop-fill" };
    liveForm.width = 1328;
    liveForm.height = 1328;
    await fieldControl("Prompt").setValue("change the target only");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(applySourceFitPreprocess).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "TARGET",
        mask: null,
        policy: { mode: "crop-fill" },
        target: { width: 1024, height: 1024 },
      }),
      expect.any(Object),
    );
    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]?.edit_images).toEqual(["FITTED_TARGET", "REFERENCE"]);
    expect(liveForm.imageAttachments).toEqual(["TARGET", "REFERENCE"]);
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
    // This legacy model's synthesized profile leaves custom sizes to server
    // admission, so the exact model wiring must not invent a local blocker.
    expect(wrapper.find("[data-test='mobile-resolution-error']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-resolution-warning']").exists()).toBe(false);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([upscaler, controlNet, imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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

  it("queues Batch N directly without implicitly expanding the prompt", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three variations of a storm");
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe("Develop 3 prints");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(expandPrompt).not.toHaveBeenCalled();
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    // Batch N is admitted as N ordered singleton children in one durable
    // batch; nothing is previewed for placement and no stream is opened.
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    // Only the machine-wide lifecycle stream is open; no per-print stream is.
    expect(openStreams.filter((stream) => stream.path !== "/api/events")).toHaveLength(0);
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(3);
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop 3 prints (+3 queued)",
    );

    const requests = admittedRequests();
    expect(requests).toHaveLength(3);
    const firstSeed = requests[0]?.seed as number;
    expect(requests.map((request) => request.batch_size)).toEqual([1, 1, 1]);
    expect(requests.map((request) => request.seed)).toEqual([
      firstSeed,
      firstSeed + 1,
      firstSeed + 2,
    ]);
    expect(requests.map((request) => request.prompt)).toEqual([
      "three variations of a storm",
      "three variations of a storm",
      "three variations of a storm",
    ]);
    expect(requests.map((request) => request.expand)).toEqual([undefined, undefined, undefined]);
    expect(requests.map((request) => request.original_prompt)).toEqual([
      undefined,
      undefined,
      undefined,
    ]);
    expect(requests.map((request) => request.batch_id)).toEqual([undefined, undefined, undefined]);
  });

  it("prepares Batch N only from the explicit Expand action", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three variations of a storm");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
  });

  it("does not submit a print when the same URL and key report a new instance", async () => {
    // A print no longer waits on a placement probe, but source preprocessing
    // still runs against the frozen machine — that is the window in which the
    // machine can be replaced under the submission.
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    const preprocess = deferred<{ source: string | null; mask: string | null; changed: boolean }>();
    applySourceFitPreprocess.mockReturnValueOnce(preprocess.promise);
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await fieldControl("Prompt").setValue("identity-fenced storm");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") {
        return Promise.resolve({ ...status, instance_id: "replacement-instance" });
      }
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();
    preprocess.resolve({ source: btoa("fitted"), mask: null, changed: true });
    await flushPromises();

    expect(admittedRequests()).toHaveLength(0);
    expect(wrapper.text()).toContain("connection details changed");
  });

  it("preserves prepared Batch N and queues nothing when the machine refuses the print", async () => {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      // A machine that advertises no durable queue cannot take the print, and
      // there is no attached stream left to fall back to.
      if (path === "/api/capabilities") return Promise.resolve({});
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("preserved storm pair");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("does not advertise the durable generation queue");
    expect(wrapper.text()).toContain("Nothing was queued.");
    expect(admittedRequests()).toHaveLength(0);
    expect(openStreams).toHaveLength(0);
    // The reviewed set survives, and Develop stays live so it can be retried.
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(true);
    expect(
      wrapper.get("[data-test='mobile-develop-prepared']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("identifies a failed middle prepared sibling by variation and reviewed prompt", async () => {
    serveStillModel();
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three weather studies");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    const editors = wrapper.findAll(".mobile-prepared-editor");
    await editors[0]!.setValue("clear dawn");
    await editors[1]!.setValue("middle thunderstorm");
    await editors[2]!.setValue("quiet dusk");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    await flushPromises();
    expect(admittedRequests()).toHaveLength(3);
    // One batch, three children, three different outcomes.
    admissions.settleAt(0, {
      children: [
        { state: "complete", filename: "first.png" },
        { state: "failed", error: "host ran out of memory" },
        { state: "complete", filename: "third.png" },
      ],
    });
    await flushPromises();
    await flushPromises();

    // The partial failure names the one-based variation and its reviewed
    // prompt wherever it is announced.
    const announced = wrapper
      .findAll(".sr-only[aria-live='polite']")
      .map((region) => region.text())
      .join(" ");
    expect(announced).toContain("Variation 2, “middle thunderstorm”");
    expect(announced).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));
  });

  it("keeps a prepared sibling live when cancellation is unconfirmed", async () => {
    serveStillModel();
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three mixed outcomes");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    const editors = wrapper.findAll(".mobile-prepared-editor");
    await editors[0]!.setValue("successful dawn");
    await editors[1]!.setValue("failed middle storm");
    await editors[2]!.setValue("cancelled dusk");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    await flushPromises();
    expect(admittedRequests()).toHaveLength(3);

    const cancelRow = wrapper
      .findAll("[data-test='mobile-generation-job']")
      .find((row) => row.text().includes("cancelled dusk"))!;
    await cancelRow.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    // The machine never confirmed the cancel, and it settles the batch anyway:
    // the failure still names its own variation and the successful sibling is
    // preserved.
    admissions.settleAt(0, {
      children: [
        { state: "complete", filename: "first.png" },
        { state: "failed", error: "host ran out of memory" },
        { state: "complete", filename: "third.png" },
      ],
    });
    await flushPromises();
    await flushPromises();

    const announced = wrapper
      .findAll(".sr-only[aria-live='polite']")
      .map((region) => region.text())
      .join(" ");
    expect(announced).toContain("Variation 2, “failed middle storm”");
    expect(announced).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await fieldControl("Prompt").setValue("newer source");
    resolveExpansion({ expanded: ["one", "two", "three"] });
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-expansion-error']").text()).toContain(
      "changed while expansion was running",
    );
    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe("newer source");

    expandPrompt.mockResolvedValueOnce({ expanded: ["one", "  ", "three"] });
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-prepared-expansion']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-expansion-error']").text()).toContain(
      "Prompt 2 was empty",
    );
  });

  it("invalidates quick Batch 1 submission when Undo wins deferred preprocessing", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
    applySourceFitPreprocess.mockRejectedValueOnce(new Error("upscaler unavailable"));
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = btoa("source");
    liveForm.sourceImageName = "source.png";
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two preserved portraits");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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

    expect(admittedRequests()).toHaveLength(2);
    expect(admittedRequests()[0]!.prompt).toBe("edited after preprocessing failed");
  });

  it("does not steal focus when delayed prepared submission finishes after the user moves", async () => {
    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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

  it("keeps exact-count failures out of the missing-model pull flow", async () => {
    expandPrompt.mockRejectedValueOnce(
      new Error(
        "expected exactly 2 distinct non-empty prompts, but the expansion backend returned 1. " +
          "The model may need re-downloading: mold pull qwen3-expand",
      ),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("lighthouse");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-pull-expansion']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-expansion-error']").text()).toContain(
      "Expansion failed on Studio: expected exactly 2 distinct non-empty prompts",
    );
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
      await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    wrapper.unmount();
    wrapper = null;
    resolveExpansion({ expanded: ["late one", "late two"] });
    await flushPromises();
    expect(openStreams).toHaveLength(0);

    const imageModel: ModelEntry = { ...model, name: "flux:image", family: "flux" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([imageModel]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
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

    expect(admittedRequests()[0]).toMatchObject({ seed: 1234 });
    await wrapper.get("[data-test='mobile-seed-mode-random']").trigger("click");
    expect(wrapper.find("[data-test='mobile-seed-input']").exists()).toBe(false);
  });

  it("develops a selected running print over its live latent preview", async () => {
    // A print holds no stream, so its latent comes from the machine's own
    // `GET /api/queue/:id/preview` once the row is selected.
    serveStillModel();
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/queue/durable-job-1-1/preview") {
        return Promise.resolve({
          preview_image: btoa("latent-png"),
          step: 2,
          total: 8,
          stage: "Denoising",
          updated_at_ms: 1,
        });
      }
      return base(callTarget, path, init);
    });
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("a latent preview print");

    // Before a preview arrives the status line stands alone.
    expect(wrapper.find("[data-test='mobile-develop-bed']").exists()).toBe(false);

    admissions.settleAt(0, { state: "running" });
    await flushPromises();
    await wrapper
      .get("[data-test='mobile-generation-job'] .mobile-generation-job")
      .trigger("click");
    await flushPromises();
    await flushPromises();

    const bed = wrapper.get("[data-test='mobile-develop-bed']");
    // The bed adopts the submitted print's aspect ratio and hands the same
    // ratio to CSS so the 55vh cap can ride the width axis without distortion.
    expect(bed.attributes("style")).toContain("aspect-ratio");
    expect(bed.attributes("style")).toContain("--bed-ar");
    const preview = wrapper.get("[data-test='mobile-develop-preview']");
    // The queue preview is rendered straight from its base64 payload; there
    // is no session blob for a print the machine owns.
    expect(preview.attributes("src")).toBe(`data:image/png;base64,${btoa("latent-png")}`);
    // blur(max(2, 14 − 12·p)) at p = 2/8 → 11px.
    expect(preview.attributes("style")).toContain("blur(11px)");
    // The develop grain layers over the preview and thins with progress.
    expect(wrapper.find("develop-canvas-stub").exists()).toBe(true);
    // The changing status stays outside and below the noisy preview, matching
    // the desktop/web placement invariant.
    const summary = wrapper.get("[data-test='mobile-generation-summary']");
    expect(bed.find("[data-test='mobile-generation-summary']").exists()).toBe(false);
    expect(
      bed.element.compareDocumentPosition(summary.element) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("snapshots multiple prompts, shows their live queue, and cancels only one", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first prompt");
    await submitPrompt("second prompt");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(2);
    expect(admittedRequests().map((request) => request.prompt)).toEqual([
      "first prompt",
      "second prompt",
    ]);
    expect(admittedRequests().every((request) => request.model === model.name)).toBe(true);
    expect(wrapper.get("[data-test='mobile-develop-button']").text()).toBe(
      "Develop print (+2 queued)",
    );

    const rows = wrapper.findAll("[data-test='mobile-generation-job']");
    expect(rows).toHaveLength(2);
    expect(rows.map((row) => row.text()).join(" ")).toContain("first prompt");
    expect(rows.map((row) => row.text()).join(" ")).toContain("second prompt");

    // Each print is its own durable child, so cancelling one names that
    // child's own queue id and leaves the other alone.
    const second = rows.find((row) => row.text().includes("second prompt"));
    await second?.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/durable-job-2", {
      method: "DELETE",
    });
    // The machine owns the outcome: the tap is a request, so the cancelled
    // print reads as cancelling until its child settles, and the OTHER print
    // is untouched.
    const after = wrapper.findAll("[data-test='mobile-generation-job']");
    const cancelled = after.find((row) => row.text().includes("second prompt"));
    const untouched = after.find((row) => row.text().includes("first prompt"));
    expect(cancelled?.text()).toContain("Cancelling");
    expect(untouched?.text()).not.toContain("Cancelling");
  });

  it("reveals exact-host pause, resume, and cancel actions with a natural queue swipe", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status")
        return Promise.resolve({ ...status, queue_paused: false, queue_capacity: 2 });
      if (path === "/api/capabilities")
        return Promise.resolve({
          queue: {
            can_pause: true,
            can_pause_job: true,
            durable_queue: true,
            heterogeneous_batch_max_outputs: 8,
          },
        });
      if (path === "/api/generation-batches" && init?.method === "POST") {
        return Promise.resolve(durableBatchResponse(init, { state: "accepted" }));
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
      if (path === "/api/queue?limit=2") return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("pause this queued print");
    await flushPromises();

    const row = wrapper.get("[data-test='mobile-generation-job']");
    const control = row.get("[data-test='swipe-action-queue-pause']");
    expect(control.text()).toBe("Pause");
    expect(row.get("[data-test='swipe-action-cancel']").text()).toBe("Cancel");

    await control.trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/durable-job-1/pause", {
      method: "POST",
    });
    expect(row.get("[data-test='mobile-generation-status']").text()).toBe("PAUSED");
    const resume = row.get("[data-test='swipe-action-queue-resume']");
    expect(resume.text()).toBe("Resume");

    await resume.trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/queue/durable-job-1/resume", {
      method: "POST",
    });
    expect(row.get("[data-test='mobile-generation-status']").text()).toBe("QUEUED");
    expect(row.get("[data-test='swipe-action-queue-pause']").text()).toBe("Pause");
  });

  it("counts a queued print's live place in line rather than its submit slot", async () => {
    // A print carries no held stream, so its place in line comes from the
    // machine's own queue listing keyed on the durable child's job id.
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      if (path === "/api/queue")
        return Promise.resolve({
          entries: [
            {
              id: "durable-job-1",
              model: "m",
              state: "running",
              started_at_unix_ms: 1,
              position: 0,
            },
            {
              id: "durable-job-2",
              model: "m",
              state: "queued",
              started_at_unix_ms: 2,
              position: 1,
            },
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
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first prompt");
    await submitPrompt("second prompt");
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    const second = wrapper
      .findAll("[data-test='mobile-generation-job']")
      .find((row) => row.text().includes("second prompt"));
    expect(second?.get("[data-test='mobile-generation-status']").text()).toBe("QUEUED #1");
  });

  it("keeps a transient empty execution plan in the queue on a busy host", async () => {
    // Reported from a MiniMax H3 render on a 1-GPU host: while the running job
    // owned the resources needed to resolve the next execution plan, the
    // waiting row briefly rendered `No usable device` before becoming `Next
    // up`. The coordinator keeps this state queued because the active render
    // can release exactly what the next planning pass needs.
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      if (path === "/api/queue")
        return Promise.resolve({
          entries: [
            {
              id: "durable-job-1",
              model: "m",
              state: "running",
              started_at_unix_ms: 1,
              position: 0,
            },
            {
              id: "durable-job-2",
              model: "m",
              state: "queued",
              started_at_unix_ms: 2,
              position: 1,
            },
          ],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "settled",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "w2",
                parent_id: "durable-job-2",
                work_kind: "generation",
                priority_class: "normal",
                queue_rank: 1,
                bypass_count: 0,
                estimate_confidence: "low",
                blocked_reason: "no_schedulable_device",
              },
            ],
          },
        });
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first prompt");
    await submitPrompt("second prompt");
    await flushPromises();
    window.dispatchEvent(new Event("pageshow"));
    await flushPromises();

    const second = wrapper
      .findAll("[data-test='mobile-generation-job']")
      .find((row) => row.text().includes("second prompt"));
    // This is queue bookkeeping for every model family, so it must fall
    // through to the authoritative position rather than name the planner.
    expect(second?.get("[data-test='mobile-generation-status']").text()).toBe("QUEUED #1");
    // The active/queued header is not asserted here: a durable print's
    // liveness comes from its own batch lifecycle rather than this listing,
    // and `activityCountLabel` is pinned directly in studio.
  });

  it("keeps a pre-ID job live when remote cancellation is unconfirmed", async () => {
    // Cancelled BEFORE the machine answers with a job id: the tap is recorded,
    // the print stays live, and a completion that lands anyway still shows.
    serveStillModel();
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("cancel before the remote queue id arrives");

    await wrapper.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();
    // The tap's status read is answered by the fallback's EMPTY bulk response
    // (neither found nor missing), which the reducer files as an incomplete
    // response the next snapshot repairs — the row says so instead of
    // pretending the machine confirmed anything.
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toBe(
      "Re-syncing with host",
    );

    admissions.settleAt(0, { filename: "finished-anyway.png" });
    await flushPromises();
    await flushPromises();
    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));
  });

  it("keeps a completed result visible while a queued sibling settles independently", async () => {
    serveStillModel();
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("finished prompt");
    await submitPrompt("failing prompt");
    admissions.settleAt(0, { filename: "finished.png" });
    await flushPromises();
    await flushPromises();

    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));
    expect(wrapper.findAll("[data-test='mobile-generation-job']")).toHaveLength(1);
    expect(wrapper.get("[data-test='mobile-generation-job']").text()).toContain("failing prompt");

    admissions.settleAt(1, { state: "failed", error: "host ran out of memory" });
    await flushPromises();
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
    admitCompletedPrints("saved print.png");
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("save this print");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/saved%20print.png");
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: btoa("generated-image"),
    });
  });

  it("auto-saves both post-upscale stills to Photos", async () => {
    serveStillModel();
    apiFetchTo
      .mockResolvedValueOnce({
        blob: () => Promise.resolve(new Blob(["original-image"], { type: "image/png" })),
      } as Response)
      .mockResolvedValueOnce({
        blob: () => Promise.resolve(new Blob(["upscaled-image"], { type: "image/png" })),
      } as Response);
    admitCompletedPrints("saved-original.png", "saved-upscaled.png");
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("save both versions");
    await flushPromises();

    await vi.waitFor(() =>
      expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/saved-original.png"),
    );
    expect(invoke).toHaveBeenCalledWith("save_image_to_photos", {
      dataB64: btoa("original-image"),
    });
  });

  it("never buffers a completed video for Photos", async () => {
    // The shared fixture is a video checkpoint, so this print settles as mp4.
    admitCompletedPrints("saved-video.mp4");
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("leave this video remote");
    await flushPromises();
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
    admitCompletedPrints("remote only.png");
    wrapper = mountMobileApp();
    await flushPromises();
    apiFetchTo.mockClear();
    invoke.mockClear();
    await submitPrompt("keep this remote");

    await flushPromises();

    expect(apiFetchTo).not.toHaveBeenCalledWith(target, "/api/gallery/image/remote%20only.png");
    expect(invoke).not.toHaveBeenCalledWith("save_image_to_photos", expect.anything());
  });

  it("promotes the last of simultaneous completions before pruning older results", async () => {
    serveStillModel();
    // Resolve each print's media by its own filename so the promoted result
    // is identifiable — a durable print has no encoded payload to tell apart.
    streamableMediaUrl.mockImplementation((path: string) =>
      Promise.resolve(`https://studio/media/${String(path).split("/").pop()}`),
    );
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first simultaneous prompt");
    await submitPrompt("second simultaneous prompt");
    admissions.settle("complete", ["first.png", "second.png"]);
    await flushPromises();
    await flushPromises();

    await vi.waitFor(() =>
      expect(wrapper!.get("img.result-media").attributes("src")).toBe(
        "https://studio/media/second.png",
      ),
    );
  });

  it("streams a metadata-only generated video from the host", async () => {
    admitCompletedPrints("generated clip.mp4");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("stream this video");
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

  it("fails a completion that published no file instead of showing a stub", async () => {
    serveStillModel();
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("metadata without a file");
    admissions.settleAt(0, { filename: null });
    await flushPromises();
    await flushPromises();

    // The machine settles a child on what reached its gallery, so a
    // completion naming no file is a failure the user sees — never a stub
    // Photos auto-save silently gets nothing from.
    expect(wrapper.find(".result-media").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "published no file",
    );
  });

  it("shows ticket failures and lets the user retry the preview", async () => {
    serveStillModel();
    streamableMediaUrl
      .mockRejectedValueOnce(new Error("The host refused the media ticket."))
      .mockResolvedValueOnce("https://studio/media/retried-image");
    admitCompletedPrints("ticketed image.png");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("ticketed image");
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
    admitCompletedPrints("renew this video.mp4");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("renew this video");
    await flushPromises();
    expect(wrapper.get("video.result-media").attributes("src")).toContain("media_token=old");

    await wrapper.get("video.result-media").trigger("play");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("video.result-media").attributes("src")).toContain("media_token=new");
  });

  it("bounds automatic media recovery and exposes a manual retry", async () => {
    serveStillModel();
    const unchangedUrl = "https://studio/media/missing?media_token=unchanged&expires=4102444800";
    streamableMediaUrl.mockResolvedValueOnce(unchangedUrl).mockResolvedValueOnce(unchangedUrl);
    admitCompletedPrints("missing.png");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("missing generated image");
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

  it("backs off transient generated-video failures before remounting the media element", async () => {
    vi.useFakeTimers();
    const unchangedUrl =
      "https://studio/media/missing-video?media_token=unchanged&expires=4102444800";
    streamableMediaUrl.mockResolvedValueOnce(unchangedUrl).mockResolvedValueOnce(unchangedUrl);
    admitCompletedPrints("missing-video.mp4");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("missing generated video");
    await flushPromises();

    const originalVideo = wrapper.get("video.result-media").element;
    expect(wrapper.get("video.result-media").attributes("preload")).toBe("metadata");
    await wrapper.get("video.result-media").trigger("error");
    await flushPromises();

    expect(streamableMediaUrl).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(249);
    expect(streamableMediaUrl).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await flushPromises();
    expect(streamableMediaUrl).toHaveBeenCalledTimes(2);
    expect(wrapper.get("video.result-media").attributes("src")).toBe(unchangedUrl);
    expect(wrapper.get("video.result-media").element).not.toBe(originalVideo);
  });

  it("bounds delayed generated-video recovery and exposes a manual retry", async () => {
    vi.useFakeTimers();
    const unchangedUrl =
      "https://studio/media/missing-video?media_token=unchanged&expires=4102444800";
    streamableMediaUrl.mockResolvedValue(unchangedUrl);
    admitCompletedPrints("missing-video.mp4");
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("persistently missing generated video");
    await flushPromises();

    for (const delay of [250, 750, 1_500]) {
      await wrapper.get("video.result-media").trigger("error");
      await vi.advanceTimersByTimeAsync(delay);
      await flushPromises();
    }
    expect(streamableMediaUrl).toHaveBeenCalledTimes(4);

    await wrapper.get("video.result-media").trigger("error");
    await flushPromises();
    expect(wrapper.find("video.result-media").exists()).toBe(false);
    expect(wrapper.get(".result-preview-error").text()).toContain(
      "Couldn’t load this generated print",
    );
    expect(
      wrapper
        .findAll(".result-preview-error button")
        .some((button) => button.text() === "Try preview again"),
    ).toBe(true);
  });

  it("replaces a stale video retry when a newer generated result fails", async () => {
    vi.useFakeTimers();
    streamableMediaUrl.mockResolvedValue(
      "https://studio/media/video?media_token=renewed&expires=4102444800",
    );
    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();

    await submitPrompt("first video");
    admissions.settleAt(0, { filename: "first-video.mp4" });
    await flushPromises();
    await vi.waitFor(() => expect(wrapper!.find("video.result-media").exists()).toBe(true));
    await wrapper.get("video.result-media").trigger("error");

    await submitPrompt("second video");
    admissions.settleAt(1, { filename: "second-video.mp4" });
    await flushPromises();
    await vi.waitFor(() => expect(wrapper!.find("video.result-media").exists()).toBe(true));
    const secondVideo = wrapper.get("video.result-media").element;
    await wrapper.get("video.result-media").trigger("error");

    const before = streamableMediaUrl.mock.calls.length;
    await vi.advanceTimersByTimeAsync(250);
    await flushPromises();
    // The retry belongs to the NEWER print: the stale one is replaced, not
    // renewed alongside it.
    expect(streamableMediaUrl.mock.calls.length).toBe(before + 1);
    expect(wrapper.get("video.result-media").element).not.toBe(secondVideo);
  });

  it("shows a completion that wins the cancellation race", async () => {
    // The race is now against ADMISSION: the tap lands while the POST is in
    // flight and the machine answers with a finished child.
    serveStillModel();
    const admission = deferred<unknown>();
    const base = apiJsonTo.getMockImplementation()!;
    let body: RequestInit | undefined;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/generation-batches" && init?.method === "POST") {
        body = init;
        return admission.promise;
      }
      return base(callTarget, path, init);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("almost finished prompt");
    apiFetchTo.mockImplementationOnce(async () => new Response(null, { status: 204 }));

    await wrapper.get("[data-test='swipe-action-cancel']").trigger("click");
    admission.resolve(durableBatchResponse(body, { state: "complete" }));
    await flushPromises();
    await flushPromises();

    await vi.waitFor(() => expect(wrapper!.find("img.result-media").exists()).toBe(true));
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
    expect(wrapper.findAll(".sr-only[aria-live='polite']")[1]?.text()).toBe(
      "Generation completed.",
    );
  });

  it("preserves a server failure that wins the cancellation race", async () => {
    const admission = deferred<unknown>();
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/generation-batches" && init?.method === "POST") {
        const parsed = JSON.parse(String(init.body)) as { client_batch_id: string };
        admission.resolve({
          id: `batch-${parsed.client_batch_id}`,
          client_batch_id: parsed.client_batch_id,
          instance_id: status.instance_id,
          durable: true,
          children: [
            {
              index: 1,
              job_id: "durable-job-1",
              state: "failed",
              error: "host ran out of memory",
              created_at_ms: 1,
              updated_at_ms: 2,
              completed_at_ms: 3,
            },
          ],
        });
        return admission.promise;
      }
      return base(callTarget, path, init);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("failing during cancel");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
  });

  it("coalesces simultaneous completion refreshes while Gallery is open", async () => {
    const galleryResolvers: Array<() => void> = [];
    let galleryCalls = 0;
    let galleryInFlight = 0;
    let maxGalleryInFlight = 0;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
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
      return durableApiFallback(path, init, callTarget);
    });

    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("first gallery prompt");
    await submitPrompt("second gallery prompt");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(galleryResolvers).toHaveLength(1));

    admissions.settle("complete", ["first.png", "second.png"]);
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

  it("indexes every print while windowing tiles and defers refresh while the viewer is open", async () => {
    const prints = Array.from({ length: 41 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    let galleryCalls = 0;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        galleryCalls += 1;
        return Promise.resolve(prints);
      }
      return durableApiFallback(path, init, callTarget);
    });

    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("refresh after older page");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(
        wrapper?.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total"),
      ).toBe("41"),
    );
    expect(wrapper.find("button.gallery-more").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='gallery-item']").length).toBeLessThanOrEqual(40);

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);

    admissions.settle();
    await flushPromises();
    expect(galleryCalls).toBe(1);

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await vi.waitFor(() => expect(galleryCalls).toBe(2));
    await vi.waitFor(() =>
      expect(
        wrapper?.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total"),
      ).toBe("41"),
    );
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
  });

  it("keeps thumbnail failures local to the bounded visible window", async () => {
    const prints = Array.from({ length: 81 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(prints);
      return durableApiFallback(path, init, callTarget);
    });
    let thumbnailCall = 0;
    apiFetchTo.mockImplementation(() => {
      thumbnailCall += 1;
      if (thumbnailCall > 20 && thumbnailCall <= 30) {
        return Promise.reject(new Error("thumbnail unavailable"));
      }
      return Promise.resolve({
        blob: () => Promise.resolve(new Blob([`thumbnail-${thumbnailCall}`])),
      } as Response);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(thumbnailCall).toBe(40));
    expect(wrapper.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total")).toBe(
      "81",
    );
    expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);
    expect(wrapper.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(false);
  });

  it("keeps a ten-thousand-print Library to a bounded mounted tile window", async () => {
    const prints = Array.from({ length: 10_000 }, (_, index) => ({
      ...print,
      filename: `large-library-${index}.png`,
      timestamp: print.timestamp - index,
    }));
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(prints);
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(
        wrapper?.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total"),
      ).toBe("10000"),
    );

    expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);
    expect(
      Number(wrapper.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-mounted")),
    ).toBeLessThanOrEqual(40);
  });

  it("does not let a stalled thumbnail block every older print", async () => {
    vi.useFakeTimers();
    try {
      const prints = Array.from({ length: 81 }, (_, index) => ({
        ...print,
        filename: `stalled-pagination-print-${index}.mp4`,
        timestamp: print.timestamp - index,
      }));
      apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve(prints);
        return durableApiFallback(path, init, callTarget);
      });
      let thumbnailCall = 0;
      let stalledSignal: AbortSignal | undefined;
      const requestedPaths: string[] = [];
      apiFetchTo.mockImplementation((_target, path, init) => {
        thumbnailCall += 1;
        requestedPaths.push(path);
        if (thumbnailCall === 1) {
          stalledSignal = init?.signal;
          // Model a WebView transport that ignores AbortSignal as well as
          // never resolving. The page deadline must still advance the grid.
          return new Promise(() => {});
        }
        return Promise.resolve(
          new Response(cachedThumbnailBlob(`thumbnail-${thumbnailCall}`), {
            status: 200,
            headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
          }),
        );
      });

      wrapper = mountMobileApp();
      await vi.advanceTimersByTimeAsync(0);
      await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
      await vi.advanceTimersByTimeAsync(0);
      expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);

      await vi.advanceTimersByTimeAsync(50);

      expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);
      await vi.waitFor(() =>
        expect(wrapper?.findAll("[data-test='gallery-thumbnail-pending']")).toHaveLength(1),
      );
      expect(wrapper.text()).not.toContain("Loading older prints…");

      await vi.advanceTimersByTimeAsync(5_020);

      expect(stalledSignal?.aborted).toBe(true);
      expect(requestedPaths).toContain(
        "/api/gallery/thumbnail/stalled-pagination-print-39.mp4?v=mobile-test-v1&size=256&fmt=jpeg",
      );
      expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);
      expect(wrapper.findAll("[data-test='gallery-thumbnail-pending']")).toHaveLength(1);
      expect(wrapper.find("[data-test='mobile-gallery-sentinel']").exists()).toBe(false);
      expect(wrapper.text()).not.toContain("Loading older prints…");
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries a failed thumbnail without blocking later prints", async () => {
    vi.useFakeTimers();
    try {
      const prints = Array.from({ length: 41 }, (_, index) => ({
        ...print,
        filename: `retry-pagination-print-${index}.mp4`,
        timestamp: print.timestamp - index,
      }));
      apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve(prints);
        return durableApiFallback(path, init, callTarget);
      });
      const attempts = new Map<string, number>();
      apiFetchTo.mockImplementation((_target, path) => {
        const attempt = (attempts.get(path) ?? 0) + 1;
        attempts.set(path, attempt);
        if (path.includes("retry-pagination-print-39.mp4") && attempt === 1) {
          return Promise.reject(new Error("temporary thumbnail failure"));
        }
        return Promise.resolve(
          new Response(cachedThumbnailBlob(`${path}-${attempt}`), {
            status: 200,
            headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
          }),
        );
      });

      wrapper = mountMobileApp();
      await vi.advanceTimersByTimeAsync(0);
      await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(20);

      expect(wrapper.findAll("[data-test='gallery-item']")).toHaveLength(40);
      await vi.waitFor(() =>
        expect(wrapper?.findAll("[data-test='gallery-thumbnail-pending']")).toHaveLength(1),
      );

      await vi.advanceTimersByTimeAsync(4_000);

      expect(wrapper.find("[data-test='gallery-thumbnail-pending']").exists()).toBe(false);
      expect(
        attempts.get(
          "/api/gallery/thumbnail/retry-pagination-print-39.mp4?v=mobile-test-v1&size=256&fmt=jpeg",
        ),
      ).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("aborts pending thumbnail work when the mobile app unmounts", async () => {
    const prints = Array.from({ length: 41 }, (_, index) => ({
      ...print,
      filename: `unmount-pagination-print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(prints);
      return durableApiFallback(path, init, callTarget);
    });
    const lateThumbnail = deferred<Response>();
    let pendingSignal: AbortSignal | undefined;
    apiFetchTo.mockImplementation((_target, path, init) => {
      if (path.includes("unmount-pagination-print-0.mp4")) {
        pendingSignal = init?.signal;
        return lateThumbnail.promise;
      }
      return Promise.resolve({ blob: () => Promise.resolve(new Blob([path])) } as Response);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(pendingSignal).toBeDefined());
    const objectUrlsBeforeUnmount = vi.mocked(URL.createObjectURL).mock.calls.length;
    const timeoutSpy = vi.spyOn(globalThis, "setTimeout");

    wrapper.unmount();
    wrapper = null;
    expect(pendingSignal?.aborted).toBe(true);
    timeoutSpy.mockClear();
    lateThumbnail.resolve({ blob: () => Promise.resolve(new Blob(["late"])) } as Response);
    await flushPromises();

    expect(URL.createObjectURL).toHaveBeenCalledTimes(objectUrlsBeforeUnmount);
    expect(
      timeoutSpy.mock.calls.filter(([, delay]) => typeof delay === "number" && delay >= 2_000),
    ).toHaveLength(0);
    timeoutSpy.mockRestore();
  });

  it("keeps focus stable when the viewer closes before a queued Gallery refresh starts", async () => {
    const prints = Array.from({ length: 41 }, (_, index) => ({
      ...print,
      filename: `print-${index}.mp4`,
      timestamp: print.timestamp - index,
    }));
    let galleryCalls = 0;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") {
        galleryCalls += 1;
        return Promise.resolve(prints);
      }
      return durableApiFallback(path, init, callTarget);
    });

    const admissions = heldAdmissions();
    wrapper = mountMobileApp();
    await flushPromises();
    await submitPrompt("refresh after an early viewer close");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(
        wrapper?.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total"),
      ).toBe("41"),
    );

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();

    admissions.settle();
    await flushPromises();
    expect(galleryCalls).toBe(1);

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await vi.waitFor(() => expect(galleryCalls).toBe(2));

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([entry]);
      if (path === "/api/gallery") return Promise.resolve(gallery);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      if (path.startsWith("/api/catalog/installed")) return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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

  it("keeps opaque H3 catalog rows out of the generation picker", async () => {
    serveWan({
      ...wanModel,
      name: "hf:MiniMaxAI/MiniMax-H3",
      family: "minimax-h3",
      hf_repo: "MiniMaxAI/MiniMax-H3",
      source_image: "required",
    });
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-model-empty']").text()).toContain(
      "No downloaded generation model is available",
    );
    expect(wrapper.find("[data-test='mobile-h3-authoring-error']").exists()).toBe(false);
  });

  it("snaps a stale steps value onto the recipe's fixed control instead of blocking Develop", async () => {
    // The reviewed compact H3 tags pin steps at their tier count. Gallery
    // reuse restores a print saved before that pin by writing straight into
    // the live form (`applyMetadataToForm`), leaving the model alone and
    // only moving `steps` — so the model-pick reconcile never runs. The
    // submit-time snap cannot rescue it either: `stepsError` disables
    // Develop and `basicParametersValid` returns `generate()` early, both
    // first. The live form has to re-assert the fixed value itself.
    const fixedStepsProfile = {
      schema_version: 1,
      profile_id: "minimax-h3.minimax-h3-fl2va",
      profile_hash: "h3-compact-hash",
      default_recipe_id: "default",
      recipes: [
        {
          id: "default",
          label: "Default",
          request_selector: {},
          defaults: { width: 1344, height: 768, steps: 21, guidance: 0, frames: 124, fps: 24 },
          resolution: {
            domain: "buckets" as const,
            alignment: 32,
            min_width: 64,
            min_height: 64,
            max_pixels: 1344 * 768,
            off_bucket: "reject" as const,
            aspect_groups: [
              {
                id: "7:4",
                label: "7:4",
                presets: [
                  { id: "1344x768", width: 1344, height: 768, tier: "recommended" as const },
                ],
              },
            ],
          },
          steps: { default: 21, min: 21, max: 21, step: 1, mode: "fixed" as const },
          guidance: { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed" as const },
          temporal: {
            frames: { default: 124, min: 124, max: 124, step: 17, mode: "fixed" as const },
            frame_offset: 5,
            fps: { mode: "fixed" as const, value: 24 },
          },
          capabilities: {
            guidance: {
              adjustable: false,
              supports_negative_prompt: false,
              fixed_scale: 0,
            },
            negative_prompt: { mode: "hidden" as const, required: false },
            supports_lora: false,
            supports_controlnet: false,
            supports_sequence: false,
            supports_extend: false,
            supports_audio: true,
            source_video: { mode: "hidden" as const, required: false },
            mask: { mode: "hidden" as const, required: false },
            keyframes: { mode: "hidden" as const, required: false },
            audio: { mode: "hidden" as const, required: false },
            lora: { mode: "hidden" as const, max_count: 0 },
            controlnet: { mode: "hidden" as const, max_count: 0 },
            output: {
              default_format: "mp4" as const,
              formats: ["mp4" as const],
              audio_requires_mp4: true,
            },
            wan_recipe: {
              mode: "hidden" as const,
              supports_distill_strength: false,
              supports_first_last_frame: false,
            },
            schedulers: [],
          },
          provenance: [],
        },
      ],
    };
    const h3Model = {
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
      generation_profile: fixedStepsProfile,
    } as unknown as ModelEntry;
    serveWan(h3Model);
    wrapper = mountMobileApp();
    await flushPromises();

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(liveForm.steps).toBe(21);

    // Exactly what reuse does for a print saved at the old flexible ladder:
    // a direct write, same model, no reconcile.
    liveForm.steps = 30;
    await flushPromises();

    expect(liveForm.steps).toBe(21);
    await fieldControl("Prompt").setValue("a ship crossing violet lightning");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes()).not.toHaveProperty(
      "disabled",
    );
  });

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

  it("uses the selected H3 model profile and queues its conditioned render", async () => {
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
      recommended_dimensions: [
        { width: 1344, height: 768 },
        { width: 1024, height: 768 },
        { width: 768, height: 768 },
        { width: 768, height: 1024 },
      ],
      dimension_alignment: 32,
      max_pixels: 1344 * 768,
    };
    serveWan(h3Model);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-orientation]").exists()).toBe(false);
    expect(
      wrapper
        .findAll("[data-test='mobile-resolution-shape'] button")
        .map((button) => button.text()),
    ).toEqual(["1:1", "4:3", "3:4", "16:9"]);
    expect(wrapper.get("[data-shape='16:9']").attributes("aria-checked")).toBe("true");

    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.h3Authoring = {
      firstFrame: {
        filename: "opening.png",
        mimeType: "image/png",
        width: 1344,
        height: 768,
        data: "QUJD",
      },
      lastFrame: null,
      references: [],
    };
    await fieldControl("Prompt").setValue("a pickup crossing a desert at dusk");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await vi.waitFor(() => expect(admittedRequests()).toHaveLength(1));

    expect(admittedRequests()[0]).toMatchObject({
      model: h3Model.name,
      width: 1344,
      height: 768,
      frames: 124,
      fps: 24,
      source_image: "QUJD",
    });
  });

  it("keeps the well and offers no end frame when the host advertises no contract", async () => {
    serveWan(wanModel);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-end-frame-well']").exists()).toBe(false);
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
    expect(wrapper.find("[data-test='mobile-source-well']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-end-frame-well']").exists()).toBe(false);
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

    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]).toMatchObject({
      keyframes: [
        { frame: 0, image: btoa("opening"), name: "opening.png" },
        { frame: 80, image: btoa("closing"), name: "closing.png" },
      ],
    });
    // The engine refuses `source_image` + `keyframes` together ("not both");
    // the pair travels only as keyframes.
    expect(admittedRequests()[0]?.source_image).toBeFalsy();
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

    // 81 frames on wan auto-chains, so this print is created as a chain job
    // rather than admitted through the durable batch.
    const create = apiFetchTo.mock.calls.find((call) => call[1] === "/api/chain-jobs");
    const body = JSON.parse(String((create![2] as RequestInit).body)) as Record<string, unknown>;
    expect(body.source_image).toBe(btoa("opening"));
    expect(body).not.toHaveProperty("keyframes");
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
    apiJsonTo.mockImplementation(
      (apiTarget: { baseUrl: string }, path: string, init?: RequestInit) => {
        if (apiTarget.baseUrl.includes("render.local")) {
          return Promise.reject(new TypeError("Load failed"));
        }
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve([]);
        return durableApiFallback(path, init, apiTarget);
      },
    );
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
    apiJsonTo.mockImplementation(
      (apiTarget: { baseUrl: string }, path: string, init?: RequestInit) => {
        if (apiTarget.baseUrl.includes("render.local")) {
          return Promise.reject(new ApiError("", 401));
        }
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/gallery") return Promise.resolve([]);
        return durableApiFallback(path, init, apiTarget);
      },
    );
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (!hostReachable) return Promise.reject(new TypeError("Load failed"));
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
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
    // Newer than the submission under test: gallery recovery is age-bounded.
    get timestamp() {
      return Math.floor(Date.now() / 1000) + 5;
    },
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

  it("restores the native WKWebView frame after the software keyboard dismisses", async () => {
    vi.useFakeTimers();
    const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => undefined);
    try {
      Object.defineProperty(window, "__TAURI_INTERNALS__", {
        value: {},
        configurable: true,
      });
      wrapper = mountMobileApp();
      await flushPromises();
      invoke.mockClear();

      const prompt = fieldControl("Prompt").element as HTMLTextAreaElement;
      prompt.focus();
      prompt.blur();
      await vi.advanceTimersByTimeAsync(0);
      expect(invoke).toHaveBeenCalledTimes(1);
      expect(invoke).toHaveBeenLastCalledWith("restore_mobile_viewport");
      expect(scrollTo).toHaveBeenLastCalledWith(0, 0);

      await vi.advanceTimersByTimeAsync(400);
      expect(invoke).toHaveBeenCalledTimes(2);
      expect(invoke).toHaveBeenLastCalledWith("restore_mobile_viewport");
      expect(scrollTo).toHaveBeenCalledTimes(2);
    } finally {
      scrollTo.mockRestore();
      vi.useRealTimers();
    }
  });

  it("resets only the keyboard-shifted document layer and preserves Create scrolling", async () => {
    const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => undefined);
    try {
      Object.defineProperty(window, "__TAURI_INTERNALS__", {
        value: {},
        configurable: true,
      });
      wrapper = mountMobileApp();
      await flushPromises();
      scrollTo.mockClear();

      const content = wrapper.get(".mobile-content").element as HTMLElement;
      content.scrollTop = 420;
      const prompt = fieldControl("Prompt").element as HTMLTextAreaElement;
      prompt.focus();
      prompt.blur();
      await Promise.resolve();

      expect(scrollTo).toHaveBeenCalledWith(0, 0);
      expect(content.scrollTop).toBe(420);
    } finally {
      scrollTo.mockRestore();
    }
  });

  it("scrolls the prompt field into view when its keyboard opens", async () => {
    vi.useFakeTimers();
    try {
      wrapper = mountMobileApp();
      await flushPromises();

      const prompt = fieldControl("Prompt").element as HTMLTextAreaElement;
      const promptField = prompt.closest<HTMLElement>(".field")!;
      const scrollIntoView = vi.fn();
      Object.defineProperty(promptField, "scrollIntoView", {
        configurable: true,
        value: scrollIntoView,
      });

      prompt.focus();
      await vi.advanceTimersByTimeAsync(0);
      expect(scrollIntoView).toHaveBeenCalledWith({ block: "center", inline: "nearest" });

      await vi.advanceTimersByTimeAsync(400);
      expect(scrollIntoView).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("tracks the visual viewport so sheets clear the keyboard and the header stays put", async () => {
    const originalViewport = Object.getOwnPropertyDescriptor(window, "visualViewport");
    const visualViewport = new EventTarget() as EventTarget & { pageTop: number; height: number };
    visualViewport.pageTop = 58;
    visualViewport.height = 510;
    Object.defineProperty(window, "visualViewport", {
      value: visualViewport,
      configurable: true,
    });
    try {
      wrapper = mountMobileApp();
      await flushPromises();
      expect(
        document.documentElement.style.getPropertyValue("--mobile-visual-viewport-page-top"),
      ).toBe("58px");
      expect(
        document.documentElement.style.getPropertyValue("--mobile-visual-viewport-height"),
      ).toBe("510px");

      visualViewport.pageTop = 0;
      visualViewport.height = 844;
      visualViewport.dispatchEvent(new Event("resize"));
      expect(
        document.documentElement.style.getPropertyValue("--mobile-visual-viewport-page-top"),
      ).toBe("0px");
      expect(
        document.documentElement.style.getPropertyValue("--mobile-visual-viewport-height"),
      ).toBe("844px");
    } finally {
      wrapper?.unmount();
      wrapper = null;
      if (originalViewport) Object.defineProperty(window, "visualViewport", originalViewport);
      else Reflect.deleteProperty(window, "visualViewport");
    }
  });

  it("reanchors the viewport when focus moves between keyboard editors", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      value: {},
      configurable: true,
    });
    wrapper = mountMobileApp();
    await flushPromises();
    invoke.mockClear();

    const prompt = fieldControl("Prompt").element as HTMLTextAreaElement;
    const negativePrompt = fieldControl("Negative prompt").element as HTMLTextAreaElement;
    prompt.focus();
    negativePrompt.focus();
    await Promise.resolve();

    expect(invoke).toHaveBeenCalledTimes(1);
    expect(invoke).toHaveBeenLastCalledWith("restore_mobile_viewport");
  });

  it("replaces a dismissal restore with active-keyboard re-anchoring on refocus", async () => {
    vi.useFakeTimers();
    try {
      Object.defineProperty(window, "__TAURI_INTERNALS__", {
        value: {},
        configurable: true,
      });
      wrapper = mountMobileApp();
      await flushPromises();
      invoke.mockClear();

      const prompt = fieldControl("Prompt").element as HTMLTextAreaElement;
      prompt.focus();
      prompt.blur();
      await vi.advanceTimersByTimeAsync(0);
      expect(invoke).toHaveBeenCalledTimes(1);

      prompt.focus();
      await vi.advanceTimersByTimeAsync(400);
      expect(invoke).toHaveBeenCalledTimes(3);
      expect(invoke).toHaveBeenLastCalledWith("restore_mobile_viewport");
    } finally {
      vi.useRealTimers();
    }
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

  it("renders the finished print the machine saved while the app was away", async () => {
    // A print holds no socket that can die: it is admitted durably and its
    // media is fetched from the machine's own gallery.
    admitCompletedPrints("resumed print.png");
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([stillModel]);
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/gallery") return Promise.resolve([resumedPrint]);
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await submitSeededPrompt("a ship crossing violet lightning", 77);
    await flushPromises();

    await vi.waitFor(() =>
      expect(wrapper!.get("[data-test='mobile-generation-summary']").text()).toContain("seed 77"),
    );
    expect(wrapper.text()).not.toContain("Load failed");
    expect(wrapper.get("img.result-media").attributes("src")).toBe(
      "https://studio/media/full-video",
    );
    expect(wrapper.find("[data-test='mobile-generation-queue']").exists()).toBe(false);
  });

  it("detaches without cancelling accepted work when the app closes", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await submitSeededPrompt("a queue that survives app closure", 42);
    openStreams[0]!.options.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-close" }),
    );
    const signal = openStreams[0]!.options.signal;

    wrapper.unmount();
    wrapper = null;
    await flushPromises();

    expect(signal.aborted).toBe(true);
    expect(apiFetchTo).not.toHaveBeenCalledWith(target, "/api/queue/job-close", {
      method: "DELETE",
    });
  });

  it("renews the promoted result's expired media ticket when returning to the foreground", async () => {
    streamableMediaUrl
      .mockResolvedValueOnce("https://studio/media/video?media_token=old&expires=1")
      .mockResolvedValueOnce("https://studio/media/video?media_token=new&expires=4102444800");
    admitCompletedPrints("renew on resume.mp4");
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("renew on resume");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
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
      autoTagTitle: true,
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
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    expect(wrapper.get("[data-test='mobile-batch-value']").attributes("value")).toBe("2");
    expect(
      wrapper
        .get("[data-test='mobile-batch-control']")
        .element.closest("[data-test='mobile-advanced-sheet']"),
    ).toBeNull();
    await wrapper.get("[data-test='mobile-settings-reset']").trigger("click");
    await flushPromises();

    expect((fieldControl("Prompt").element as HTMLTextAreaElement).value).toBe(
      "a ship crossing violet lightning",
    );
    expect((fieldControl("Negative prompt").element as HTMLInputElement).value).toBe("");
    // The selected model's defaults, not the bare form defaults.
    expect((fieldControl("Steps").element as HTMLInputElement).value).toBe("30");
    expect(wrapper.get("[data-test='mobile-batch-value']").attributes("value")).toBe("1");
    expect(wrapper.get("[data-test='mobile-settings-reset']").attributes("aria-label")).toBe(
      "Reset settings to model defaults",
    );
    expect(wrapper.get("[data-test='mobile-settings-reset']").element.parentElement).toBe(
      wrapper.get(".mobile-create-head").element,
    );

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");
    expect(wrapper.get("[data-test='mobile-batch-value']").attributes("value")).toBe("2");

    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.enableAudio = true;
    await flushPromises();
    await wrapper.get("[data-test='mobile-settings-reset']").trigger("click");
    expect(draft.enableAudio).toBe(false);
  });

  it("discards the sequence opening image on the primary Reset", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    // Read the live form while One shot still mounts the LoRA controls; the
    // sequence bench renders a different subtree over the same form object.
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;

    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    liveForm.strength = 0.4;
    await flushPromises();

    await wrapper.get("[data-test='mobile-settings-reset']").trigger("click");
    await flushPromises();

    expect(draft.openingImage).toBeNull();
    expect(liveForm.strength).not.toBe(0.4);
  });

  it("badges and resets LTX-2 guidance overrides from the Advanced sheet", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    expect(wrapper.get("[data-test='mobile-advanced-reset']").element.closest("header")).toBe(
      wrapper.get(".mobile-advanced-sheet-head").element,
    );
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

  it("does not badge the general Batch setting as Advanced", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    const batch = wrapper.get("[data-test='mobile-batch-value']");
    await batch.setValue("30");
    await batch.trigger("change");
    await flushPromises();

    expect(batch.attributes("value")).toBe("30");
    expect(wrapper.find("[data-test='mobile-advanced-trigger-count']").exists()).toBe(false);

    // The reported Hal9000 batch restored this actual Advanced field; it is
    // the source of the pictured `1`, not the adjacent Batch value.
    await fieldControl("Negative prompt").setValue("anime, cartoon, graphic, washed out");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-advanced-trigger-count']").text()).toBe("1");

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    expect(wrapper.get("[data-test='mobile-advanced-count']").text()).toBe("1");
  });

  it("keeps generated audio in the primary Create settings", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    const control = wrapper.get("[data-test='mobile-generate-audio-control']");
    expect(control.element.closest(".mobile-advanced-sheet")).toBeNull();
    await control.get("input").setValue(true);
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(liveForm.enableAudio).toBe(true);

    await wrapper.get("[data-test='mobile-open-advanced']").trigger("click");
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");
    expect(liveForm.enableAudio).toBe(true);
  });

  it("keeps LTX-2.5 audio visible and disabled for a video-only checkpoint", async () => {
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/models") {
        return Promise.resolve([
          {
            ...model,
            name: "ltx-2.5-22b-distilled:q4",
            supports_audio: false,
          },
        ]);
      }
      return base(callTarget, path, init);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    const control = wrapper.get("[data-test='mobile-generate-audio-control']");
    expect(control.get("input").attributes()).toHaveProperty("disabled");
    expect(wrapper.text()).toContain("Audio assets are not included with this checkpoint");
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
    liveForm.enableAudio = true;
    liveForm.cameraControl = "dolly-in";
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
    expect(liveForm.enableAudio).toBe(true);
    expect(liveForm.cameraControl).toBeNull();
  });
});

describe("MobileApp primary navigation", () => {
  it("swipes left and right through the four major destinations", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await swipeMobileContent(280, 100);
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    await swipeMobileContent(280, 100);
    expect(wrapper.get("[data-test='mobile-tab-catalog']").attributes("aria-current")).toBe("page");
    await swipeMobileContent(100, 280);
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
  });

  it("ignores left swipes in Settings and returns to the same underlying tab", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-open-settings']").trigger("click");
    await swipeMobileContent(280, 100);
    expect(wrapper.find("[data-test='mobile-settings-back']").exists()).toBe(true);

    await wrapper.get("[data-test='mobile-settings-back']").trigger("click");
    expect(wrapper.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
      "page",
    );
  });

  it("swipes right out of Machine Detail without stealing swipe-row gestures", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-host-row']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-host-detail']").exists()).toBe(true);

    await swipeMobileContent(90, 270);
    expect(wrapper.find("[data-test='mobile-host-detail']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-hosts']").attributes("aria-current")).toBe("page");
  });

  it("opens Machine Detail at the top and restores the Machines list scroll", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    const content = wrapper.get(".mobile-content").element as HTMLElement;
    content.scrollTop = 360;

    await wrapper.get("[data-test='mobile-host-row']").trigger("click");
    await flushPromises();
    expect(content.scrollTop).toBe(0);

    await wrapper.get("[data-test='host-detail-back']").trigger("click");
    await flushPromises();
    expect(content.scrollTop).toBe(360);
  });

  it("does not switch destinations when a horizontal control owns the gesture", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    const modelPicker = fieldControl("Model").element;
    modelPicker.dispatchEvent(mobileTouch("touchstart", 280, 180));
    modelPicker.dispatchEvent(mobileTouch("touchmove", 100, 180));
    modelPicker.dispatchEvent(mobileTouch("touchend", 100, 180, true));
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
      "page",
    );
  });

  it("pulls Models and Machines to refresh their remote snapshots", async () => {
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "discover_mold_hosts") return Promise.resolve([]);
      return Promise.resolve(null);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const content = wrapper.get(".mobile-content").element;

    await wrapper.get("[data-test='mobile-tab-catalog']").trigger("click");
    await flushPromises();
    const catalogBefore = apiJsonTo.mock.calls.filter(([, path]) =>
      String(path).startsWith("/api/catalog/search"),
    ).length;
    content.dispatchEvent(mobileTouch("touchstart", 120, 100));
    content.dispatchEvent(mobileTouch("touchmove", 120, 230));
    content.dispatchEvent(mobileTouch("touchend", 120, 230, true));
    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => String(path).startsWith("/api/catalog/search"))
          .length,
      ).toBeGreaterThan(catalogBefore),
    );

    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await flushPromises();
    const probesBefore = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length;
    content.dispatchEvent(mobileTouch("touchstart", 120, 100));
    content.dispatchEvent(mobileTouch("touchmove", 120, 230));
    content.dispatchEvent(mobileTouch("touchend", 120, 230, true));
    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length,
      ).toBeGreaterThan(probesBefore),
    );
    expect(invoke).toHaveBeenCalledWith("discover_mold_hosts", { timeoutMs: 2500 });

    await wrapper.get("[data-test='mobile-host-row']").trigger("click");
    await flushPromises();
    const detailModelsBefore = apiJsonTo.mock.calls.filter(
      ([, path]) => path === "/api/models",
    ).length;
    content.dispatchEvent(mobileTouch("touchstart", 120, 100));
    content.dispatchEvent(mobileTouch("touchmove", 120, 230));
    content.dispatchEvent(mobileTouch("touchend", 120, 230, true));
    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/models").length,
      ).toBeGreaterThan(detailModelsBefore),
    );
  });

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

  it("restores the Library scroll position when returning during the session", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();

    const content = wrapper.get(".mobile-content").element as HTMLElement;
    content.scrollTop = 420;
    await wrapper.get("[data-test='mobile-tab-catalog']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    await vi.waitFor(() => expect(content.scrollTop).toBe(420));
  });

  it("restores a deep Library position against the virtual grid extent", async () => {
    const manyPrints = Array.from({ length: 81 }, (_, index) => ({
      ...print,
      filename: `deep-scroll-${index}.png`,
      timestamp: print.timestamp - index,
    }));
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(manyPrints);
      if (path === "/api/activity") {
        return Promise.resolve({
          instance_id: "mobile-host",
          observed_at_unix_ms: 1,
          items: [],
        });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const content = wrapper.get(".mobile-content").element as HTMLElement;
    let scrollTop = 0;
    Object.defineProperty(content, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        const grid = wrapper!.find("[data-test='mobile-gallery-grid']");
        const logicalExtent = grid.exists()
          ? Number(grid.attributes("data-gallery-total")) * 10
          : 0;
        scrollTop = Math.min(value, logicalExtent);
      },
    });

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    await vi.waitFor(() =>
      expect(
        wrapper!.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-total"),
      ).toBe("81"),
    );
    await flushPromises();
    content.scrollTop = 700;
    await wrapper.get("[data-test='mobile-tab-catalog']").trigger("click");
    await flushPromises();
    expect(scrollTop).toBe(0);
    expect(sessionScrollPosition("mobile:library").top).toBe(700);
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(scrollTop).toBe(700));
  });
});

describe("MobileApp gallery", () => {
  it("hydrates a live gallery replacement when its cached keys are unchanged", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: true,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    const liveGallery = deferred<GalleryImage[]>();
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status")
        return Promise.resolve({ ...status, instance_id: "studio-instance" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(null);
      if (path === "/api/gallery") return liveGallery.promise;
      if (path === "/api/activity")
        return Promise.resolve({
          instance_id: "studio-instance",
          observed_at_unix_ms: 1,
          items: [],
        });
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => {
      const tile = wrapper!.get("[data-test='gallery-item']");
      expect(tile.find(".mobile-media-placeholder").exists()).toBe(false);
      expect(tile.get("img").attributes("src")).toContain("blob:");
    });
    expect(wrapper.text()).not.toContain("Loading prints…");
    const cachedUrl = wrapper.get("[data-test='gallery-item'] img").attributes("src");
    const objectUrlsBeforeLive = vi.mocked(URL.createObjectURL).mock.calls.length;

    liveGallery.resolve([print]);
    await flushPromises();

    await vi.waitFor(() => {
      const tile = wrapper!.get("[data-test='gallery-item']");
      expect(tile.find(".mobile-media-placeholder").exists()).toBe(false);
      expect(tile.get("img").attributes("src")).toBe(cachedUrl);
    });
    expect(URL.createObjectURL).toHaveBeenCalledTimes(objectUrlsBeforeLive);
  });

  it("reloads a cached tile when the live content version changes", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: true,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob("old"));
    const liveGallery = deferred<GalleryImage[]>();
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status")
        return Promise.resolve({ ...status, instance_id: "studio-instance" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(null);
      if (path === "/api/gallery") return liveGallery.promise;
      if (path === "/api/activity")
        return Promise.resolve({
          instance_id: "studio-instance",
          observed_at_unix_ms: 1,
          items: [],
        });
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-item'] img").attributes("src")).toContain("blob:"),
    );
    const cachedUrl = wrapper.get("[data-test='gallery-item'] img").attributes("src");

    liveGallery.resolve([{ ...print, media_version: "mobile-test-v2" }]);
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-item'] img").attributes("src")).not.toBe(cachedUrl),
    );
  });

  it("reuses a cached print immediately while its host is offline", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedHostPresentation({
      hostId: "studio-instance",
      updatedAt: 1,
      instanceId: "studio-instance",
      serverVersion: status.version,
      models: [model],
      capabilities: null,
    });
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    apiJsonTo.mockRejectedValue(new Error("offline"));
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer']").exists()).toBe(false),
    );
    expect(wrapper.get("#mobile-prompt").element).toHaveProperty(
      "value",
      "a ship crossing violet lightning",
    );
    const reusedForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(reusedForm).toMatchObject({ seed: "77", width: 768, height: 512, steps: 28 });
  });

  it("closes during a pending reuse and ignores its late host response", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    const lateStatus = deferred<ServerStatus>();
    const lateModels = deferred<ModelEntry[]>();
    let statusCalls = 0;
    let modelCalls = 0;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") {
        statusCalls += 1;
        return statusCalls === 1 ? Promise.reject(new Error("offline")) : lateStatus.promise;
      }
      if (path === "/api/models") {
        modelCalls += 1;
        return modelCalls === 1 ? Promise.reject(new Error("offline")) : lateModels.promise;
      }
      if (path === "/api/capabilities") return Promise.resolve(null);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-viewer-reuse']").text()).toBe("Loading settings…"),
    );

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    lateStatus.resolve({ ...status, instance_id: "studio-instance" });
    lateModels.resolve([model]);
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    expect(wrapper.text()).not.toContain(print.metadata.prompt);
  });

  it("closes during Use as source and ignores its late media response", async () => {
    const still: GalleryImage = { ...print, filename: "source.png", format: "png" };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    const lateMedia = deferred<Response>();
    apiFetchTo.mockReturnValue(lateMedia.promise);

    await wrapper.get("[data-test='gallery-viewer-use-source']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-viewer-use-source']").text()).toBe(
        "Loading source…",
      ),
    );
    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    lateMedia.resolve({
      headers: new Headers(),
      blob: () => Promise.resolve(new Blob(["late source"], { type: "image/png" })),
    } as Response);
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(liveForm.sourceImage).toBeNull();
  });

  it("keeps a superseding gallery selection while a Ref2VA source hash finishes", async () => {
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
    const first = { ...print, filename: "first subject.png", format: "png" as const };
    const second = { ...print, filename: "second subject.png", format: "png" as const };
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([ref2vaModel]);
      if (path === "/api/gallery") return Promise.resolve([first, second]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });
    apiFetchTo.mockImplementation(() =>
      Promise.resolve(
        new Response(cachedThumbnailBlob("subject bytes"), {
          status: 200,
          headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
        }),
      ),
    );
    const lateDigest = deferred<ArrayBuffer>();
    const digest = vi
      .spyOn(globalThis.crypto.subtle, "digest")
      .mockReturnValueOnce(lateDigest.promise);

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(2));
    await wrapper.findAll("[data-test='gallery-item']")[0]!.trigger("click");
    await wrapper.get("[data-test='gallery-viewer-use-source']").trigger("click");
    await vi.waitFor(() => expect(digest).toHaveBeenCalledTimes(1));

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    await wrapper.findAll("[data-test='gallery-item']")[1]!.trigger("click");
    expect(wrapper.get("[data-test='gallery-viewer']").text()).toContain("second subject.png");

    lateDigest.resolve(new Uint8Array(32).buffer);
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-current")).toBe("page");
    expect(wrapper.get("[data-test='gallery-viewer']").text()).toContain("second subject.png");
    expect(liveForm.h3Authoring?.references).toEqual([]);
    expect(wrapper.text()).not.toContain("Added gallery print as reference");
  });

  it("keeps retained verified models authoritative while the host reconnects", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    const statusCalls = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length;
    apiJsonTo.mockImplementation((_callTarget: unknown, path: string, _init?: RequestInit) => {
      if (path === "/api/status") return Promise.reject(new Error("offline"));
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return Promise.reject(new Error("offline"));
    });
    window.dispatchEvent(new Event("pageshow"));
    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length,
      ).toBeGreaterThan(statusCalls),
    );

    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer']").exists()).toBe(false),
    );

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Prompt settings restored",
    );
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    expect(wrapper.get("[data-test='mobile-host-health']").text()).toBe("reconnecting…");
  });

  it("finishes Reuse from a media-bearing draft without self-cancelling", async () => {
    const conditioned = {
      ...print,
      metadata: {
        ...print.metadata,
        source_image_name: "restored.png",
        source_image_sha256: "a".repeat(64),
      },
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([conditioned]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
    restoreGenerationSourceMedia.mockResolvedValue({
      kind: "upload",
      filename: "restored.png",
      base64: "RESTORED_SOURCE",
      width: 512,
      height: 512,
    });
    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceImage = "OLD_SOURCE";
    liveForm.identityImage = { filename: "old-face.png", base64: "OLD_FACE" };
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");

    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer']").exists()).toBe(false),
    );
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "Prompt settings restored",
    );
    expect(liveForm.prompt).toBe(print.metadata.prompt);
    expect(liveForm.sourceImage).toBe("RESTORED_SOURCE");
    expect(liveForm.sourceImageName).toBe("restored.png");
  });

  it("times out a reuse even when the transport ignores AbortSignal", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    apiJsonTo.mockRejectedValue(new Error("offline"));
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    apiJsonTo.mockImplementation(() => new Promise(() => {}));
    const realSetTimeout = globalThis.setTimeout;
    const timeout = vi.spyOn(globalThis, "setTimeout").mockImplementation((handler, delay) => {
      if (delay === 9_000) {
        queueMicrotask(() => (handler as () => void)());
        return 1 as unknown as ReturnType<typeof setTimeout>;
      }
      return realSetTimeout(handler, delay);
    });

    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-viewer-reuse']").text()).toBe("Reuse settings"),
    );
    timeout.mockRestore();

    expect(wrapper.get("[role='alert']").text()).toContain(
      "Loading saved settings from Studio timed out. Try again.",
    );
  });

  it("finishes cached reuse when source restoration ignores AbortSignal", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    const conditioned = {
      ...print,
      metadata: { ...print.metadata, source_image_name: "missing-source.png" },
    };
    await storeCachedGallery("studio-instance", [conditioned]);
    await storeCachedHostPresentation({
      hostId: "studio-instance",
      updatedAt: Date.now(),
      instanceId: "studio-instance",
      serverVersion: status.version,
      models: [model],
      capabilities: null,
    });
    await storeCachedGalleryMedia(cachedThumbnailRef(conditioned), cachedThumbnailBlob());
    apiJsonTo.mockRejectedValue(new Error("offline"));
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    apiFetchTo.mockImplementation(() => new Promise(() => {}));
    const realSetTimeout = globalThis.setTimeout;
    const timeout = vi.spyOn(globalThis, "setTimeout").mockImplementation((handler, delay) => {
      if (delay === 9_000) {
        queueMicrotask(() => (handler as () => void)());
        return 1 as unknown as ReturnType<typeof setTimeout>;
      }
      return realSetTimeout(handler, delay);
    });

    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer']").exists()).toBe(false),
    );
    timeout.mockRestore();

    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "The original source image is unavailable. Reattach it before developing.",
    );
  });

  it("does not trust cached model data from a different known server version", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          version: "2.0.0",
          online: false,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedHostPresentation({
      hostId: "studio-instance",
      updatedAt: 1,
      instanceId: "studio-instance",
      serverVersion: "1.0.0",
      models: [model],
      capabilities: null,
    });
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    apiJsonTo.mockRejectedValue(new Error("offline"));
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer-reuse']").attributes("aria-busy")).toBe(
        "false",
      ),
    );

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    expect(wrapper.get("[role='alert']").text()).toContain("Couldn’t load models from Studio");
  });

  it("renders instance-scoped cached gallery metadata and thumbnails while its host is offline", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    await storeCachedGallery("studio-instance", [print]);
    await storeCachedGalleryMedia(cachedThumbnailRef(print), cachedThumbnailBlob());
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status" || path === "/api/gallery") {
        return Promise.reject(new Error("offline"));
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-item'] img").attributes("src")).toContain("blob:"),
    );
    expect(wrapper.text()).toContain("Showing saved Library");
  });

  it("keeps saved Library tag filters visible when its host is offline", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "studio-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    localStorage.setItem(
      "mold.mobile.gallery-capabilities.v1",
      JSON.stringify({
        "url-derived-id": {
          instanceId: "studio-instance",
          gallery: { organize: true, trash: { enabled: true, retention_days: 30 } },
        },
      }),
    );
    localStorage.setItem(
      "mold.mobile.gallery-tags.v1",
      JSON.stringify({
        "url-derived-id": {
          instanceId: "studio-instance",
          tags: [
            { name: "Barbie", count: 6 },
            { name: "moon", count: 6 },
          ],
        },
      }),
    );
    // Older saved rows may not carry organization fields. Their known tag
    // inventory must still survive the loading -> saved-Library transition.
    await storeCachedGallery("studio-instance", [print]);
    apiJsonTo.mockRejectedValue(new Error("offline"));
    apiFetchTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.text()).toContain("Showing saved Library"));

    expect(wrapper.find("[data-test='mobile-library-chip-favorites']").exists()).toBe(true);
    expect(
      wrapper.findAll("[data-test='mobile-library-chip-tag']").map((chip) => chip.text()),
    ).toEqual(["Barbie6", "moon6"]);
    const filters = wrapper.get("[data-test='mobile-library-chips']").element;
    const error = wrapper.get("[data-test='mobile-library-gallery-error']").element;
    const grid = wrapper.get("[data-test='mobile-gallery-pinch-surface']").element;
    expect(filters.compareDocumentPosition(error) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(error.compareDocumentPosition(grid) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("does not restore saved tags across a host instance boundary", async () => {
    localStorage.setItem(
      "mold.mobile.gallery-capabilities.v1",
      JSON.stringify({
        "studio-id": {
          instanceId: "mobile-host",
          gallery: { organize: true },
        },
      }),
    );
    localStorage.setItem(
      "mold.mobile.gallery-tags.v1",
      JSON.stringify({
        "studio-id": {
          instanceId: "replaced-instance",
          tags: [{ name: "Wrong server", count: 9 }],
        },
      }),
    );
    apiJsonTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    expect(wrapper.find("[data-test='mobile-library-chip-tag']").exists()).toBe(false);
  });

  it("ignores malformed saved tag inventories", async () => {
    localStorage.setItem(
      "mold.mobile.gallery-capabilities.v1",
      JSON.stringify({
        "studio-id": {
          instanceId: "mobile-host",
          gallery: { organize: true },
        },
      }),
    );
    localStorage.setItem(
      "mold.mobile.gallery-tags.v1",
      JSON.stringify({
        "studio-id": {
          instanceId: "mobile-host",
          tags: [{ name: null, count: 2 }],
        },
      }),
    );
    apiJsonTo.mockRejectedValue(new Error("offline"));

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    expect(wrapper.find("[data-test='mobile-library-chip-tag']").exists()).toBe(false);
  });

  it("does not restore an old instance cache from a gallery response that resolves after replacement", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: new IDBFactory(),
    });
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "url-derived-id",
          instanceId: "old-instance",
          name: "Studio",
          baseUrl: target.baseUrl,
          online: false,
        },
      ]),
    );
    await storeCachedGallery("old-instance", [print]);
    const galleryResponse = deferred<GalleryImage[]>();
    let replacement = false;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          instance_id: replacement ? "replacement-instance" : "old-instance",
        });
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return galleryResponse.promise;
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() =>
      expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/gallery", expect.anything()),
    );
    const statusCalls = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length;
    replacement = true;
    window.dispatchEvent(new Event("pageshow"));
    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status").length,
      ).toBeGreaterThan(statusCalls),
    );
    await vi.waitFor(async () => expect(await loadCachedGallery("old-instance")).toEqual([]));
    galleryResponse.resolve([print]);
    await flushPromises();

    expect(await loadCachedGallery("old-instance")).toEqual([]);
  });

  it("reuses promptless print settings without reviving stale original prompt text", async () => {
    const promptless: GalleryImage = {
      ...print,
      metadata: {
        ...print.metadata,
        prompt: "",
        original_prompt: "a previous prompt that did not render this print",
      },
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([promptless]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='gallery-viewer-reuse']").text()).toBe("Reuse settings");
    expect(wrapper.text()).toContain("No prompt was used for this print.");
    expect(wrapper.text()).not.toContain("a previous prompt that did not render this print");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();
    expect(wrapper.get("#mobile-prompt").element).toHaveProperty("value", "");
    expect(fieldControl("Negative prompt").element).toHaveProperty("value", "calm water");
  });

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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (baseUrl === offlineTarget.baseUrl) {
        if (path === "/api/status") return Promise.reject(new Error("offline"));
        if (path === "/api/gallery") return new Promise(() => {});
      }
      if (path === "/api/status") return Promise.resolve({ ...status, hostname: "plato" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return durableApiFallback(path, init, requestTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");

    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    expect(wrapper.find(".empty-state").exists()).toBe(false);
    expect(wrapper.text()).toContain("1 host unavailable");
  });

  it("settles a host whose capabilities endpoint stalls instead of leaving the Library loading", async () => {
    vi.useFakeTimers();
    try {
      apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") return Promise.resolve(status);
        if (path === "/api/models") return Promise.resolve([model]);
        // The capabilities read never settles: the host deadline alone must
        // resolve this host so the multi-host refresh can finish.
        if (path === "/api/capabilities") return new Promise(() => {});
        if (path === "/api/gallery") return Promise.resolve([print]);
        if (path === "/api/activity")
          return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
        return durableApiFallback(path, init, callTarget);
      });

      wrapper = mountMobileApp();
      await vi.advanceTimersByTimeAsync(0);
      await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
      await vi.advanceTimersByTimeAsync(9_000);

      expect(wrapper.find("[data-test='gallery-item']").exists()).toBe(true);
    } finally {
      vi.useRealTimers();
    }
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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (baseUrl === recoveredTarget.baseUrl && path === "/api/status") {
        halcyonProbeCount += 1;
        return halcyonProbeCount === 1
          ? Promise.reject(new Error("offline"))
          : recoveredProbe.promise;
      }
      if (path === "/api/status") return Promise.resolve({ ...status, hostname: "plato" });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") {
        return Promise.resolve(baseUrl === recoveredTarget.baseUrl ? [print] : []);
      }
      return durableApiFallback(path, init, requestTarget);
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

  it("keeps the native image context menu and enters multi-select from its native Select action", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const contextMenu = new Event("contextmenu", { bubbles: true, cancelable: true });
    wrapper.get("[data-test='gallery-item'] img").element.dispatchEvent(contextMenu);
    expect(contextMenu.defaultPrevented).toBe(false);
    expect(wrapper.find("[data-test='mobile-gallery-actions']").exists()).toBe(false);
    expect(invoke).toHaveBeenCalledWith("extend_gallery_context_menu", {
      upscaleLabel: "Framewise upscale…",
    });

    window.dispatchEvent(new CustomEvent("mold:native-gallery-select"));
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
    expect(wrapper.get("[data-test='gallery-item']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.get("[data-test='mobile-gallery-selection-indicator']").text()).toBe("✓");
  });

  it("opens Framewise upscale from mobile Library selection and the native iOS menu", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    const upscaler: ModelEntry = {
      ...model,
      name: "real-esrgan-x4plus:fp16",
      family: "upscaler",
      description: "4x video upscaler",
      downloaded: false,
    };
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/models") return Promise.resolve([model, upscaler]);
      if (path === "/api/video-upscale-jobs") return Promise.resolve([]);
      return base(callTarget, path, init);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    const selectionUpscale = wrapper.get("[data-test='mobile-gallery-upscale']");
    expect(selectionUpscale.text()).toBe("Upscale");
    await selectionUpscale.trigger("click");
    await flushPromises();
    expect(document.querySelector("[data-test='upscale-dialog']")?.getAttribute("aria-label")).toBe(
      "Framewise upscale video",
    );
    expect(document.querySelector("[data-test='upscale-model']")?.textContent).toContain(
      "downloads on first use",
    );

    (document.querySelector(".upscale-dialog__close") as HTMLButtonElement).click();
    await flushPromises();
    const contextMenu = new Event("contextmenu", { bubbles: true, cancelable: true });
    wrapper.get("[data-test='gallery-item'] img").element.dispatchEvent(contextMenu);
    expect(invoke).toHaveBeenCalledWith("extend_gallery_context_menu", {
      upscaleLabel: "Framewise upscale…",
    });
    window.dispatchEvent(new CustomEvent("mold:native-gallery-upscale"));
    await flushPromises();
    expect(document.querySelector("[data-test='upscale-dialog']")?.getAttribute("aria-label")).toBe(
      "Framewise upscale video",
    );
  });

  it("shows a failed mobile Framewise submission inside the open upscale sheet", async () => {
    const upscaler: ModelEntry = {
      ...model,
      name: "real-esrgan-x4plus:fp16",
      family: "upscaler",
      description: "4x video upscaler",
      downloaded: false,
    };
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/models") return Promise.resolve([model, upscaler]);
      if (path === "/api/video-upscale-jobs" && init?.method === "POST") {
        return Promise.reject(new Error("Framewise upscale is unavailable on this machine"));
      }
      if (path === "/api/video-upscale-jobs") return Promise.resolve([]);
      return base(callTarget, path, init);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-upscale']").trigger("click");
    await flushPromises();
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(document.querySelector("[data-test='upscale-dialog']")).not.toBeNull();
    expect(document.querySelector("[data-test='upscale-error']")?.textContent).toContain(
      "Framewise upscale is unavailable on this machine",
    );
  });

  it("uses the selected Library image as the older-host upscale fallback source", async () => {
    serveStillModel();
    const imagePrint: GalleryImage = {
      ...print,
      filename: "selected-upscale.png",
      format: "png",
      metadata: { ...print.metadata, model: stillModel.name, output_format: "png" },
    };
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/gallery") return Promise.resolve([imagePrint]);
      return base(callTarget, path, init);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='mobile-gallery-upscale']").trigger("click");
    await flushPromises();
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(
      apiFetchTo.mock.calls.some(([, path]) => String(path).includes("selected-upscale.png")),
    ).toBe(true);
    expect(wrapper.getComponent(MobileSourceControls).props("form")).toMatchObject({
      sourceImageName: "selected-upscale.png",
    });
  });

  it("announces an older-host image upscale fallback failure after closing the dialog", async () => {
    serveStillModel();
    const imagePrint: GalleryImage = {
      ...print,
      filename: "unavailable-upscale.png",
      format: "png",
      metadata: { ...print.metadata, model: stillModel.name, output_format: "png" },
    };
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/gallery") return Promise.resolve([imagePrint]);
      return base(callTarget, path, init);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='mobile-gallery-upscale']").trigger("click");
    await flushPromises();
    apiFetchTo.mockRejectedValueOnce(new Error("source unavailable"));
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(document.querySelector("[data-test='upscale-dialog']")).toBeNull();
    expect(wrapper.text()).toContain("source unavailable");
  });

  it("moves mobile viewer upscale above the native dialog top layer", async () => {
    const upscaler: ModelEntry = {
      ...model,
      name: "real-esrgan-x4plus:fp16",
      family: "upscaler",
      description: "4x video upscaler",
      downloaded: false,
    };
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/models") return Promise.resolve([model, upscaler]);
      if (path === "/api/video-upscale-jobs") return Promise.resolve([]);
      return base(callTarget, path, init);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-upscale']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(document.querySelector("[data-test='upscale-dialog']")?.getAttribute("aria-label")).toBe(
      "Framewise upscale video",
    );
  });

  it("keeps a tapped Library tile selected after iOS dispatches its delayed click", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    const tile = wrapper.get("[data-test='gallery-item']");
    await tile.trigger("pointerdown", {
      pointerId: 40,
      pointerType: "touch",
      isPrimary: true,
      clientX: 20,
      clientY: 240,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 40,
        pointerType: "touch",
        isPrimary: true,
      }),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tile.trigger("click", { detail: 1 });

    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
    expect(tile.attributes("aria-pressed")).toBe("true");

    // Two more taps complete before WKWebView dispatches either compatibility
    // click. Both delayed clicks must be consumed without changing the two
    // pointerdown selections.
    await tile.trigger("pointerdown", {
      pointerId: 41,
      pointerType: "touch",
      isPrimary: true,
      clientX: 20,
      clientY: 240,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 41,
        pointerType: "touch",
        isPrimary: true,
      }),
    );
    await tile.trigger("pointerdown", {
      pointerId: 42,
      pointerType: "touch",
      isPrimary: true,
      clientX: 20,
      clientY: 240,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 42,
        pointerType: "touch",
        isPrimary: true,
      }),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tile.trigger("click", { detail: 1 });
    await tile.trigger("click", { detail: 1 });

    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
    expect(tile.attributes("aria-pressed")).toBe("true");
  });

  it("backs the native iOS image context menu with image data instead of a blob URL", async () => {
    isNativeIOSRuntime.mockReturnValue(true);
    apiFetchTo.mockResolvedValue(
      new Response(cachedThumbnailBlob("native context menu"), {
        status: 200,
        headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
      }),
    );

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await vi.waitFor(() => expect(apiFetchTo).toHaveBeenCalled());

    await vi.waitFor(() =>
      expect(wrapper?.get("[data-test='gallery-item'] img").attributes("src")).toMatch(
        /^data:image\/png;base64,/,
      ),
    );
    const image = wrapper.get("[data-test='gallery-item'] img");
    expect(URL.createObjectURL).not.toHaveBeenCalled();

    const contextMenu = new Event("contextmenu", { bubbles: true, cancelable: true });
    image.element.dispatchEvent(contextMenu);
    expect(contextMenu.defaultPrevented).toBe(false);
  });

  it("drag-selects and drag-deselects every Library tile crossed in Select mode", async () => {
    const prints = [
      print,
      { ...print, filename: "second.png", timestamp: print.timestamp - 1 },
      { ...print, filename: "third.png", timestamp: print.timestamp - 2 },
    ];
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve(prints);
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(3));
    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    const tiles = wrapper.findAll("[data-test='gallery-item']");
    const toolbar = wrapper.get("[data-test='mobile-gallery-actions']").element;
    Object.defineProperty(document, "elementsFromPoint", {
      configurable: true,
      value: vi.fn((x: number) => [
        toolbar,
        (x < 150 ? tiles[0] : x < 250 ? tiles[1] : tiles[2])!.element,
      ]),
    });

    await tiles[0]!.trigger("pointerdown", {
      pointerId: 41,
      pointerType: "touch",
      isPrimary: true,
      clientX: 20,
      clientY: 240,
    });
    window.dispatchEvent(
      new PointerEvent("pointermove", {
        pointerId: 41,
        pointerType: "touch",
        isPrimary: true,
        // One fast event crosses both remaining columns. The sticky toolbar
        // is deliberately first in the hit stack to cover edge auto-scroll.
        clientX: 320,
        clientY: 240,
      }),
    );
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 41,
        pointerType: "touch",
        isPrimary: true,
      }),
    );
    await wrapper.vm.$nextTick();

    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("3 selected");
    expect(tiles.map((tile) => tile.attributes("aria-pressed"))).toEqual(["true", "true", "true"]);

    await tiles[0]!.trigger("pointerdown", {
      pointerId: 42,
      pointerType: "touch",
      isPrimary: true,
      clientX: 20,
      clientY: 240,
    });
    window.dispatchEvent(
      new PointerEvent("pointermove", {
        pointerId: 42,
        pointerType: "touch",
        isPrimary: true,
        clientX: 200,
        clientY: 240,
      }),
    );
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 42,
        pointerType: "touch",
        isPrimary: true,
      }),
    );
    await wrapper.vm.$nextTick();

    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
    expect(tiles.map((tile) => tile.attributes("aria-pressed"))).toEqual([
      "false",
      "false",
      "true",
    ]);
  });

  it("leaves a vertical Select-mode swipe to native Library scrolling", async () => {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='mobile-gallery-select']").trigger("click");
    const tile = wrapper.get("[data-test='gallery-item']");

    await tile.trigger("pointerdown", {
      pointerId: 43,
      pointerType: "touch",
      isPrimary: true,
      clientX: 40,
      clientY: 300,
    });
    const verticalMove = new PointerEvent("pointermove", {
      pointerId: 43,
      pointerType: "touch",
      isPrimary: true,
      clientX: 43,
      clientY: 360,
      cancelable: true,
    });
    window.dispatchEvent(verticalMove);
    window.dispatchEvent(new PointerEvent("pointercancel", { pointerId: 43 }));
    await wrapper.vm.$nextTick();

    expect(verticalMove.defaultPrevented).toBe(false);
    expect(tile.attributes("aria-pressed")).toBe("false");
    expect(wrapper.get("[data-test='mobile-gallery-actions']").text()).toContain("0 selected");
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      // This machine's capability probe fails: the labels below pin that.
      if (path === "/api/capabilities") {
        return Promise.reject(new Error("capabilities unavailable"));
      }
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
      return durableApiFallback(path, init, route);
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const host = routes.find((candidate) => candidate.baseUrl === route.baseUrl)!;
      // This machine's capability probe fails: the labels below pin that.
      if (path === "/api/capabilities") {
        return Promise.reject(new Error("capabilities unavailable"));
      }
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
      return durableApiFallback(path, init, route);
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
    // Hold admission so the print settles while the viewer is already open.
    const admissions = heldAdmissions();
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

    admissions.settle();
    await flushPromises();
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
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith("blob:thumbnail-1");
    expect(wrapper.get("[data-test='gallery-item'] img").attributes("src")).toBe(
      "blob:thumbnail-1",
    );
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-tab-gallery']").element);
  });

  it("opens media first, then explicitly reuses the prompt and visible settings", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const tile = wrapper.get("[data-test='gallery-item']");
    expect(tile.attributes("aria-label")).toBe("Open a ship crossing violet lightning from Studio");
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
    expect(wrapper.get("[data-shape='3:2']").attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='mobile-resolution-tier-dims']").text()).toBe("768 × 512 px");
    expect(fieldControl("Format").element).toHaveProperty("value", "mp4");
    expect(fieldControl("Frames").element).toHaveProperty("value", "121");
    expect(fieldControl("FPS").element).toHaveProperty("value", "30");
  });

  it("keeps generation notches after reusing an ordinary nightly LTX output", async () => {
    const nightlyPrint: GalleryImage = {
      ...print,
      filename: "nightly-ltx.mp4",
      metadata: {
        prompt: "a live nightly print",
        model: "ltx-2.3-22b-distilled:fp8",
        seed: 44,
        steps: 8,
        guidance: 1,
        width: 768,
        height: 768,
        frames: 217,
        fps: 24,
        pipeline: "distilled",
        output_mode: "one-shot",
        version: "0.23.3 (b3e803c 2026-08-21)",
      },
    };
    const nightlyModel = {
      ...model,
      name: "ltx-2.3-22b-distilled:fp8",
      default_steps: 8,
      default_guidance: 1,
      max_frames: 481,
      frame_step: 8,
      frame_offset: 1,
      default_fps: 24,
    };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([nightlyModel]);
      if (path === "/api/gallery") return Promise.resolve([nightlyPrint]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    const duration = wrapper.findAll(".video-duration")[0]!;
    expect(duration.get("[data-test='video-duration-detail']").text()).toContain("3 generations");
    expect(duration.findAll(".ms-slider__mark b").map((mark) => mark.text())).toEqual([
      "1×",
      "2×",
      "3×",
      "4×",
      "5×",
      "6×",
    ]);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([sourcePrint]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, callTarget);
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
    serveStillModel();
    admitCompletedPrints("expand-this-result.png");
    // The viewer opens the host's own gallery entry for the print, so the
    // listing has to hold the still this admission saved.
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/gallery") {
        return Promise.resolve([
          {
            filename: "expand-this-result.png",
            timestamp: Math.floor(Date.now() / 1000) + 5,
            format: "png",
            metadata: { prompt: "expand this result", model: stillModel.name },
          },
        ]);
      }
      return base(callTarget, path, init);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await fieldControl("Prompt").setValue("expand this result");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    await flushPromises();

    // A durable print's media is fetched from the host gallery, so the tile
    // appears a tick after the completion lands.
    await vi.waitFor(() =>
      expect(wrapper!.find("[data-test='mobile-generated-result']").exists()).toBe(true),
    );
    await wrapper.get("[data-test='mobile-generated-result']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    // A durable print carries no encoded payload, so the viewer resolves its
    // media from the host through a ticketed URL rather than a session blob.
    expect(wrapper.get("[data-test='gallery-viewer-image']").attributes("src")).toBe(
      "https://studio/media/full-video",
    );

    await wrapper.get("[data-test='gallery-viewer-upscale']").trigger("click");
    await flushPromises();
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(
      apiFetchTo.mock.calls.some(([, path]) => String(path).includes("expand-this-result.png")),
    ).toBe(true);
    expect(wrapper.getComponent(MobileSourceControls).props("form")).toMatchObject({
      sourceImageName: "expand-this-result.png",
    });
  });

  it("shows New and Upscaled indicators on mobile Library tiles", async () => {
    localStorage.setItem(
      "mold.mobile.library-seen-at.v1",
      JSON.stringify({ "studio-id": print.timestamp - 1 }),
    );
    localStorage.setItem("mold.mobile.library-visited.v1", "true");
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
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
      return durableApiFallback(path, init, callTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["source bytes"], { type: "image/png" })),
    } as Response);

    wrapper = mountMobileApp();
    await flushPromises();
    const liveForm = wrapper.getComponent(MobileLoraControls).props("form") as GenerateForm;
    liveForm.sourceFit = { mode: "lanczos-resize" };
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

    expect(apiFetchTo).toHaveBeenCalledWith(
      target,
      "/api/gallery/image/source%20print.png",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(wrapper.get("[data-test='mobile-source-preview']").attributes("alt")).toBe(
      "source print.png",
    );
    expect(liveForm.sourceFit).toEqual({ mode: "crop-fill" });
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([ref2vaModel]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockImplementation(() =>
      Promise.resolve(
        new Response(cachedThumbnailBlob("subject bytes"), {
          status: 200,
          headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
        }),
      ),
    );

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

    // Ref2VA references live in the primary Create stack, not behind Advanced.
    expect(wrapper.find("[data-test='mobile-h3-authoring']").exists()).toBe(true);
    expect(wrapper.get("[data-test='h3-reference-0']").text()).toContain("ordered subject.png");
  });

  it("rejects an oversized gallery source before reading or base64 expansion", async () => {
    const still = { ...print, filename: "huge source.png", format: "png" as const };
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([still]);
      return durableApiFallback(path, init, callTarget);
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
      "Combined generation media must be 45 MiB or smaller on this phone",
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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
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
      return durableApiFallback(path, init, requestTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
      return durableApiFallback(path, init, callTarget);
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/gallery") return Promise.resolve([h3Sequence]);
      return durableApiFallback(path, init, callTarget);
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
    localStorage.setItem("mold.mobile.generate-target.v1", "studio-id");
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      // This machine's capability probe fails: the labels below pin that.
      if (path === "/api/capabilities") {
        return Promise.reject(new Error("capabilities unavailable"));
      }
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          instance_id: baseUrl === remoteTarget.baseUrl ? "remote-instance" : "studio-instance",
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
      return durableApiFallback(path, init, requestTarget);
    });

    wrapper = mountMobileApp();
    await vi.waitFor(
      () => expect(fieldControl("Model").element).toHaveProperty("value", studioModel.name),
      { timeout: 5_000 },
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
  }, 15_000);
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

  it("offers secure Android pairing and nearby discovery", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "discover_mold_hosts") {
        return Promise.resolve([
          {
            name: "Render Box",
            host: "192.168.1.50",
            port: 7680,
            authRequired: true,
          },
        ]);
      }
      return Promise.resolve(null);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");

    expect(wrapper.find("[data-test='mobile-scan-pairing']").exists()).toBe(true);
    await wrapper.get("[data-test='mobile-discover-hosts']").trigger("click");
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith("discover_mold_hosts", { timeoutMs: 2500 });
    expect(wrapper.get("[data-test='mobile-discovered-host']").text()).toContain("Render Box");
    expect(wrapper.get("[data-test='mobile-discovered-host']").text()).toContain(
      "192.168.1.50:7680",
    );
    await wrapper.get("[data-test='mobile-discovered-host'] button").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-discovered-key-prompt']").exists()).toBe(true);
    expect(apiJsonTo).not.toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://192.168.1.50:7680" }),
      "/api/status",
    );

    await wrapper.get("[data-test='mobile-discovered-api-key']").setValue("peer-secret");
    await wrapper.get("[data-test='mobile-discovered-key-prompt']").trigger("submit");
    await flushPromises();
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://192.168.1.50:7680", apiKey: "peer-secret" },
      "/api/status",
    );
  });

  it("names Google Play instead of TestFlight in Android settings", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-open-settings']").trigger("click");

    expect(wrapper.get("[data-test='mobile-update-channel']").text()).toBe("Google Play");
  });

  it("claims Android pairing codes with an Android client identity", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "scan_android_pairing_code") {
        return Promise.resolve({ content: pairingPayload() });
      }
      return Promise.resolve(null);
    });
    claimPairingSession.mockResolvedValue({
      api_key: "paired-key",
      instance_id: "wrong-host",
      hostname: "impostor",
    });

    await scanFromMachines();

    expect(claimPairingSession).toHaveBeenCalledWith("http://pair.local:7680", "one-time-token", {
      name: "Mold on Android",
      kind: "android",
    });
    expect(scanPairingQr).not.toHaveBeenCalled();
  });

  it("cancels Android pairing through the native session and settles the pending scan", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    const pendingScan = deferred<{ content: string }>();
    invoke.mockImplementation((command: string) => {
      if (command === "scan_android_pairing_code") return pendingScan.promise;
      if (command === "cancel_android_pairing_scan") {
        pendingScan.reject("cancelled");
        return Promise.resolve(null);
      }
      return Promise.resolve(null);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-scan-pairing']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-pair-scanner-cancel']").trigger("click");
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith("cancel_android_pairing_scan");
    expect(cancelBarcodeScanner).not.toHaveBeenCalled();
    expect(wrapper.find("[data-test='mobile-pair-scanner']").exists()).toBe(false);
    expect(wrapper.find(".error-text").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-scan-pairing']").attributes("disabled")).toBeUndefined();
  });

  it("cancels the native Android pairing session when the app unmounts", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    const pendingScan = deferred<{ content: string }>();
    invoke.mockImplementation((command: string) => {
      if (command === "scan_android_pairing_code") return pendingScan.promise;
      if (command === "cancel_android_pairing_scan") {
        pendingScan.reject("cancelled");
        return Promise.resolve(null);
      }
      return Promise.resolve(null);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-scan-pairing']").trigger("click");
    await flushPromises();

    wrapper.unmount();
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith("cancel_android_pairing_scan");
    expect(cancelBarcodeScanner).not.toHaveBeenCalled();
  });

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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status" && baseUrl === "http://pair.local:7680") {
        return Promise.resolve({ ...status, instance_id: "pair-id", hostname: "pair-host" });
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return durableApiFallback(path, init, requestTarget);
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

  it("stores and connects a QR-paired Android host after the native camera closes", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    invoke.mockImplementation((command: string) => {
      if (command === "scan_android_pairing_code") {
        return Promise.resolve({ content: pairingPayload() });
      }
      return Promise.resolve(null);
    });
    claimPairingSession.mockResolvedValue({
      api_key: "paired-key",
      instance_id: "pair-id",
      hostname: "pair-host",
    });
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status" && baseUrl === "http://pair.local:7680") {
        return Promise.resolve({ ...status, instance_id: "pair-id", hostname: "pair-host" });
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return durableApiFallback(path, init, requestTarget);
    });

    await scanFromMachines();
    await vi.waitFor(() =>
      expect(invoke).toHaveBeenCalledWith("keychain_set_api_key", {
        hostId: "pair-local-7680",
        apiKey: "paired-key",
      }),
    );

    expect(wrapper?.find("[data-test='mobile-pair-scanner']").exists()).toBe(false);
    expect(wrapper?.find(".error-text").exists()).toBe(false);
    expect(wrapper?.get("[data-test='mobile-tab-generate']").attributes("aria-current")).toBe(
      "page",
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
          if (path === "/api/status" && baseUrl === target.baseUrl)
            return Promise.resolve({ ...status, instance_id: "studio-instance" });
          if (path === "/api/status") {
            return new Promise<ServerStatus>((resolve, reject) => {
              remoteProbes.push({ resolve, reject, signal: init?.signal });
            });
          }
          return durableApiFallback(path, init, requestTarget);
        },
      );

      wrapper = mountMobileApp();
      await flushPromises();
      expect(remoteProbes).toHaveLength(1);

      await vi.advanceTimersByTimeAsync(10_000);
      expect(remoteProbes).toHaveLength(2);
      expect(remoteProbes[0]?.signal?.aborted).toBe(true);

      remoteProbes[1]?.resolve({
        ...status,
        instance_id: "render-instance",
        version: "0.19.0",
        hostname: "render",
      });
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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      // This machine's capability probe fails: the labels below pin that.
      if (path === "/api/capabilities") {
        return Promise.reject(new Error("capabilities unavailable"));
      }
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          instance_id: baseUrl === remoteTarget.baseUrl ? "render-instance" : "studio-instance",
          hostname: baseUrl === remoteTarget.baseUrl ? "render" : "studio",
        });
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      return durableApiFallback(path, init, requestTarget);
    });
    apiFetchTo.mockImplementation((_callTarget: unknown, path: string, _init?: RequestInit) => {
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
    // Browsing another machine's catalog leaves the generation target alone;
    // with two machines reachable that target is the Auto policy, which is
    // what the header chip names.
    expect(wrapper.get(".mobile-header .host-chip").text()).toBe("Auto");
    expect(localStorage.getItem("mold.mobile.generate-target.v1")).toBeNull();
    expect(wrapper.get("[data-test='mobile-tab-catalog']").attributes("aria-current")).toBe("page");
  });

  it("keeps the catalog download stream alive off-tab and refreshes Generate models", async () => {
    const pulledModel = { ...model, name: "ltx2:new-download" };
    let downloadFinished = false;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") {
        return Promise.resolve(downloadFinished ? [model, pulledModel] : [model]);
      }
      if (path === "/api/catalog/families") return Promise.resolve({ families: [] });
      if (path.startsWith("/api/catalog/search")) return Promise.resolve({ entries: [] });
      return durableApiFallback(path, init, callTarget);
    });
    apiFetchTo.mockImplementation((_callTarget: unknown, path: string, _init?: RequestInit) => {
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
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status" && baseUrl === "http://pair.local:7680") {
        return Promise.resolve({ ...status, instance_id: "pair-id", hostname: "pair-host" });
      }
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/gallery") return Promise.resolve([print]);
      return durableApiFallback(path, init, requestTarget);
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
    apiJsonTo
      .mockReset()
      .mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
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
        return durableApiFallback(path, init, callTarget);
      });

    wrapper = mountMobileApp();
    await flushPromises();
    await openMachines();

    const telemetry = wrapper.get("[data-test='mobile-host-telemetry']");
    expect(telemetry.get(".host-telemetry-mem").text()).toBe("9.8 / 24.0 GB");
    expect(telemetry.get(".host-telemetry-queue").text()).toBe("queue 2");
    expect(telemetry.get(".meter").attributes("aria-valuenow")).toBe("41");
  });

  it("keeps last-good telemetry and capabilities through repeated probe failures, then recovers", async () => {
    vi.useFakeTimers();
    let failing = false;
    let currentStatus: ServerStatus = {
      ...status,
      gpu_info: {
        name: "RTX 4090",
        vram_total_mb: 24_000,
        vram_used_mb: 9_840,
        backend: "cuda",
      },
      queue_depth: 2,
      queue_capacity: 8,
    };
    apiJsonTo
      .mockReset()
      .mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") {
          return failing
            ? Promise.reject(new Error("status timeout"))
            : Promise.resolve(currentStatus);
        }
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/capabilities") {
          return Promise.resolve({ gallery: { can_delete: true, organize: true } });
        }
        if (path === "/api/gallery") return Promise.resolve([print]);
        if (path === "/api/activity") {
          return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
        }
        return durableApiFallback(path, init, callTarget);
      });

    wrapper = mountMobileApp();
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-file-under']").exists()).toBe(true);
    await openMachines();

    failing = true;
    await vi.advanceTimersByTimeAsync(10_000);
    await flushPromises();
    let health = wrapper.get("[data-test='mobile-host-health']");
    expect(health.text()).toBe("reconnecting…");
    expect(wrapper.get("[data-test='mobile-host-telemetry'] .host-telemetry-mem").text()).toBe(
      "9.8 / 24.0 GB",
    );
    expect(wrapper.get(".status-dot").classes()).toContain("is-reconnecting");

    await vi.advanceTimersByTimeAsync(10_000);
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-host-health']").text()).toBe("reconnecting…");
    expect(wrapper.find("[data-test='mobile-host-telemetry']").exists()).toBe(true);

    currentStatus = {
      ...currentStatus,
      version: "0.19.0",
      queue_depth: 4,
      gpu_info: { ...currentStatus.gpu_info!, vram_used_mb: 12_000 },
    };
    failing = false;
    await vi.advanceTimersByTimeAsync(10_000);
    await flushPromises();
    health = wrapper.get("[data-test='mobile-host-health']");
    expect(health.text()).toBe("v0.19.0");
    expect(wrapper.get("[data-test='mobile-host-telemetry'] .host-telemetry-queue").text()).toBe(
      "queue 4",
    );

    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    expect(wrapper.find("[data-test='mobile-file-under']").exists()).toBe(true);
  });

  it("fences a replacement instance and retires its old telemetry and capabilities", async () => {
    vi.useFakeTimers();
    let replacement = false;
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          instanceId: "studio-id",
          online: false,
        },
      ]),
    );
    apiJsonTo
      .mockReset()
      .mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
        if (path === "/api/status") {
          return Promise.resolve({
            ...status,
            instance_id: replacement ? "replacement-id" : "studio-id",
            gpu_info: {
              name: "RTX 4090",
              vram_total_mb: 24_000,
              vram_used_mb: 9_840,
              backend: "cuda",
            },
            queue_depth: 2,
          } satisfies ServerStatus);
        }
        if (path === "/api/models") return Promise.resolve([model]);
        if (path === "/api/capabilities") {
          return Promise.resolve({ gallery: { can_delete: true, organize: true } });
        }
        if (path === "/api/gallery") return Promise.resolve([print]);
        if (path === "/api/activity") {
          return Promise.resolve({ instance_id: "studio-id", observed_at_unix_ms: 1, items: [] });
        }
        return durableApiFallback(path, init, callTarget);
      });

    wrapper = mountMobileApp();
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-file-under']").exists()).toBe(true);

    replacement = true;
    await vi.advanceTimersByTimeAsync(10_000);
    await flushPromises();
    await openMachines();

    expect(wrapper.get("[data-test='mobile-host-health']").text()).toBe("identity changed");
    expect(wrapper.find("[data-test='mobile-host-telemetry']").exists()).toBe(false);
    expect(wrapper.get(".status-dot").classes()).toContain("is-error");

    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    expect(wrapper.find("[data-test='mobile-file-under']").exists()).toBe(false);
    expect(wrapper.get(".mobile-header .host-chip").text()).toBe("Studio · identity changed");
  });

  it("aggregates every GPU on a host card", async () => {
    apiJsonTo
      .mockReset()
      .mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
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
        return durableApiFallback(path, init, callTarget);
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

describe("mobile Library pinch-to-resize", () => {
  async function openLibrary(): Promise<VueWrapper> {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    return wrapper;
  }

  /**
   * A touch is primary only while no other finger is already down, which is
   * what makes the first finger — and only the first — able to start a drag.
   */
  let touchesDown = 0;

  function touchDown(target: Element, pointerId: number, clientX: number, clientY = 0): void {
    target.dispatchEvent(
      new PointerEvent("pointerdown", {
        pointerId,
        pointerType: "touch",
        isPrimary: touchesDown === 0,
        clientX,
        clientY,
        bubbles: true,
        cancelable: true,
      }),
    );
    touchesDown += 1;
  }

  function touchMove(pointerId: number, clientX: number, clientY = 0): void {
    window.dispatchEvent(
      new PointerEvent("pointermove", {
        pointerId,
        pointerType: "touch",
        clientX,
        clientY,
        cancelable: true,
      }),
    );
  }

  function touchUp(pointerId: number): void {
    window.dispatchEvent(new PointerEvent("pointerup", { pointerId, pointerType: "touch" }));
    touchesDown = Math.max(0, touchesDown - 1);
  }

  beforeEach(() => {
    touchesDown = 0;
  });

  async function pinch(app: VueWrapper, from: number, to: number): Promise<void> {
    const grid = app.get("[data-test='mobile-gallery-grid']").element;
    touchDown(grid, 71, 0);
    touchDown(grid, 72, from);
    touchMove(72, to);
    touchUp(72);
    touchUp(71);
    await app.vm.$nextTick();
  }

  function columnsOf(app: VueWrapper): string | undefined {
    return app.get("[data-test='mobile-gallery-grid']").attributes("data-gallery-columns");
  }

  it("renders the saved three-across grid before any gesture", async () => {
    const app = await openLibrary();
    const grid = app.get("[data-test='mobile-gallery-grid']");

    expect(grid.attributes("data-gallery-columns")).toBe("3");
    expect(grid.attributes("style")).toContain("--mobile-gallery-columns: 3");
    expect(app.get(".mobile-library-heading .section-note").text()).toContain("Pinch to resize");
  });

  it("spreading two fingers enlarges the thumbnails and persists the choice", async () => {
    const app = await openLibrary();

    await pinch(app, 200, 340);

    const grid = app.get("[data-test='mobile-gallery-grid']");
    expect(grid.attributes("data-gallery-columns")).toBe("2");
    expect(grid.attributes("style")).toContain("--mobile-gallery-columns: 2");
    expect(localStorage.getItem("mold.mobile.galleryColumns.v1")).toBe("2");
    expect(app.get("[data-test='mobile-gallery-zoom-status']").text()).toContain("2 across");
  });

  it("pinching two fingers together shrinks the thumbnails", async () => {
    const app = await openLibrary();

    await pinch(app, 300, 210);

    expect(columnsOf(app)).toBe("4");
    expect(localStorage.getItem("mold.mobile.galleryColumns.v1")).toBe("4");
  });

  it("resizes from the unused print area instead of requiring both fingers over tiles", async () => {
    const app = await openLibrary();
    const surface = app.get("[data-test='mobile-gallery-pinch-surface']").element;

    touchDown(surface, 73, 20, 700);
    touchDown(surface, 74, 220, 700);
    touchMove(74, 360, 700);
    touchUp(74);
    touchUp(73);
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("2");
  });

  it("restores the persisted size on the next visit", async () => {
    localStorage.setItem("mold.mobile.galleryColumns.v1", "5");

    const app = await openLibrary();

    expect(columnsOf(app)).toBe("5");
  });

  it("a one-finger drag never resizes the grid", async () => {
    const app = await openLibrary();

    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 80, 20, 20);
    touchMove(80, 320);
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("3");
  });

  it("a pinch in select mode resizes and unpaints the tile it started on", async () => {
    const app = await openLibrary();
    await app.get("[data-test='mobile-gallery-select']").trigger("click");

    // The first finger remains pending until movement proves drag intent.
    await app.get("[data-test='gallery-item']").trigger("pointerdown", {
      pointerId: 91,
      pointerType: "touch",
      isPrimary: true,
      clientX: 0,
      clientY: 0,
    });
    expect(app.get("[data-test='mobile-gallery-actions']").text()).toContain("0 selected");

    // The second finger makes it a pinch without changing the selection.
    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 92, 200);
    touchMove(91, 0, 900);
    touchMove(92, 340);
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("2");
    expect(app.get("[data-test='mobile-gallery-actions']").text()).toContain("0 selected");
  });

  it("a pinch leaves an existing selection exactly as the user left it", async () => {
    const app = await openLibrary();
    await app.get("[data-test='mobile-gallery-select']").trigger("click");
    const tile = app.get("[data-test='gallery-item']");
    await tile.trigger("click", { detail: 1 });
    expect(app.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");

    // Starting the pinch on an already-selected tile must not deselect it.
    await tile.trigger("pointerdown", {
      pointerId: 93,
      pointerType: "touch",
      isPrimary: true,
      clientX: 0,
      clientY: 0,
    });
    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 94, 200);
    touchMove(94, 340);
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("2");
    expect(app.get("[data-test='mobile-gallery-actions']").text()).toContain("1 selected");
  });

  it("ignores a mouse pointer so desktop-class input never starts a pinch", async () => {
    const app = await openLibrary();
    const grid = app.get("[data-test='mobile-gallery-grid']").element;

    for (const pointerId of [95, 96]) {
      grid.dispatchEvent(
        new PointerEvent("pointerdown", {
          pointerId,
          pointerType: "mouse",
          clientX: pointerId * 2,
          clientY: 0,
          bubbles: true,
        }),
      );
    }
    window.dispatchEvent(
      new PointerEvent("pointermove", { pointerId: 96, pointerType: "mouse", clientX: 900 }),
    );
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("3");
  });

  it("a pinch never opens the print WKWebView delivers a delayed click for", async () => {
    const app = await openLibrary();
    const tile = app.get("[data-test='gallery-item']");

    // The resting finger lands on a tile; the other one does the spreading.
    touchDown(tile.element, 71, 0);
    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 72, 200);
    touchMove(72, 340);
    touchUp(72);
    touchUp(71);
    await app.vm.$nextTick();
    expect(columnsOf(app)).toBe("2");

    // WKWebView now dispatches the compatibility click for the resting finger.
    await tile.trigger("click", { detail: 1 });

    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(false);

    // A later, deliberate tap still opens the print.
    await tile.trigger("click", { detail: 1 });
    await flushPromises();
    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("a deliberate tap after a pinch opens the print on the first try", async () => {
    // WebKit usually synthesizes NO compatibility click once a second touch
    // lands, so the claim above is provisional. A fresh touch sequence must
    // discard it, or every pinch would cost the user a dead tap.
    const app = await openLibrary();
    const tile = app.get("[data-test='gallery-item']");

    touchDown(tile.element, 71, 0);
    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 72, 200);
    touchMove(72, 340);
    touchUp(72);
    touchUp(71);
    await app.vm.$nextTick();
    expect(columnsOf(app)).toBe("2");

    // No compatibility click ever arrived; the user simply taps a print.
    touchDown(tile.element, 73, 10, 10);
    touchUp(73);
    await tile.trigger("click", { detail: 1 });
    await flushPromises();

    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("pumping one finger cannot stockpile swallowed taps", async () => {
    const app = await openLibrary();
    const tile = app.get("[data-test='gallery-item']");
    const grid = app.get("[data-test='mobile-gallery-grid']").element;

    touchDown(tile.element, 71, 0);
    for (const id of [72, 73, 74]) {
      touchDown(grid, id, 200);
      touchMove(id, 210);
      touchUp(id);
    }
    touchUp(71);
    await app.vm.$nextTick();

    // At most one click is ever owed, so the second tap must land.
    await tile.trigger("click", { detail: 1 });
    await tile.trigger("click", { detail: 1 });
    await flushPromises();
    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("a pointerup from elsewhere in the app never claims a tap", async () => {
    const app = await openLibrary();
    const tile = app.get("[data-test='gallery-item']");
    const grid = app.get("[data-test='mobile-gallery-grid']").element;

    touchDown(tile.element, 71, 0);
    touchDown(grid, 72, 200);
    touchMove(72, 340);
    // A finger the gesture never tracked lifts somewhere else entirely.
    touchUp(60);
    touchUp(72);
    touchUp(71);
    await app.vm.$nextTick();

    await tile.trigger("click", { detail: 1 });
    await tile.trigger("click", { detail: 1 });
    await flushPromises();
    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("a cancelled pinch claims no tap at all", async () => {
    const app = await openLibrary();
    const tile = app.get("[data-test='gallery-item']");

    touchDown(tile.element, 71, 0);
    touchDown(app.get("[data-test='mobile-gallery-grid']").element, 72, 200);
    touchMove(72, 340);
    for (const pointerId of [72, 71]) {
      window.dispatchEvent(new PointerEvent("pointercancel", { pointerId, pointerType: "touch" }));
    }
    await app.vm.$nextTick();

    await tile.trigger("click", { detail: 1 });
    await flushPromises();
    expect(app.find("[data-test='gallery-viewer']").exists()).toBe(true);
  });

  it("a finger stranded by a suspend cannot make the next touch resize", async () => {
    const app = await openLibrary();
    const grid = app.get("[data-test='mobile-gallery-grid']").element;

    // The app is backgrounded mid-touch, so this pointerup never arrives.
    touchDown(grid, 71, 0);
    document.dispatchEvent(new Event("visibilitychange"));
    await flushPromises();

    // A single finger scrolling the grid must not read as a second pinch point.
    touchDown(grid, 72, 200);
    touchMove(72, 340);
    await app.vm.$nextTick();

    expect(columnsOf(app)).toBe("3");
  });
});

describe("MobileApp automatic generation routing", () => {
  const renderTarget = {
    baseUrl: "http://render.tailnet.ts.net:7680",
    apiKey: "render-secret",
  };

  function twoHosts(): void {
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
  }

  /** `/api/status`, `/api/models` and friends for a two-machine fleet. */
  function fleetApi(options: {
    studioGpu?: Record<string, unknown>;
    renderGpu?: Record<string, unknown>;
    studioModels?: ModelEntry[];
    renderModels?: ModelEntry[];
  }): void {
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
          gpu_info: render ? options.renderGpu : options.studioGpu,
        });
      if (path === "/api/models")
        return Promise.resolve(
          render ? (options.renderModels ?? [model]) : (options.studioModels ?? [model]),
        );
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, route);
    });
  }

  /** A `planned` preview whose predicted completion decides Auto. */
  function plannedIn(completionMs: number) {
    const preview = plannedPlacement();
    preview.candidate.predicted_completion_after_ms = completionMs;
    return preview;
  }

  function hostOptions(): string[] {
    return wrapper!
      .get("[data-test='mobile-generate-host']")
      .findAll("option")
      .map((option) => option.attributes("value") ?? "");
  }

  async function develop(prompt = "a routed lighthouse"): Promise<void> {
    await fieldControl("Prompt").setValue(prompt);
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  it("keeps one machine on today's behaviour with no automatic options", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    // A single saved machine renders no Host field at all, and nothing
    // promises a choice that does not exist.
    expect(wrapper.find("[data-test='mobile-generate-host']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-routing-hint']").exists()).toBe(false);
    await develop();
    expect(admissionTargets()[0]).toEqual(target);
  });

  it("offers Auto and Most capable once two machines are reachable", async () => {
    twoHosts();
    fleetApi({});
    wrapper = mountMobileApp();
    await flushPromises();

    expect(hostOptions()).toEqual(["auto", "capable", "studio-id", "render-id"]);
    expect(
      (wrapper.get("[data-test='mobile-generate-host']").element as HTMLSelectElement).value,
    ).toBe("auto");
    expect(wrapper.get("[data-test='mobile-routing-hint']").text()).toContain("least busy");

    await wrapper.get("[data-test='mobile-generate-host']").setValue("capable");
    await flushPromises();
    expect(localStorage.getItem("mold.mobile.generate-target.v1")).toBe("capable");
    expect(wrapper.get("[data-test='mobile-routing-hint']").text()).toContain("strongest GPU");
  });

  it("offers every reachable gallery even when generation is pinned to one host", async () => {
    twoHosts();
    fleetApi({});
    localStorage.setItem("mold.mobile.generate-target.v1", "studio-id");
    wrapper = mountMobileApp();
    await flushPromises();

    const sources = wrapper.getComponent(MobileSourceControls).props("gallerySources") as Array<{
      id: string;
      label: string;
      target: { baseUrl: string };
    }>;
    expect(sources.map(({ id, label, target: route }) => [id, label, route.baseUrl])).toEqual([
      ["studio-id", "Studio", target.baseUrl],
      ["render-id", "Render", renderTarget.baseUrl],
    ]);
    expect(wrapper.getComponent(MobileSourceControls).props("target")).toMatchObject({
      baseUrl: target.baseUrl,
    });
  });

  it("hides the automatic options again while only one machine answers", async () => {
    twoHosts();
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      if (route.baseUrl === renderTarget.baseUrl) return Promise.reject(new Error("offline"));
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, route);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    expect(hostOptions()).toEqual(["studio-id", "render-id"]);
    expect(wrapper.find("[data-test='mobile-routing-hint']").exists()).toBe(false);
    await develop();
    expect(admissionTargets()[0]).toEqual(target);
  });

  it("Auto asks every candidate and freezes the soonest plan's machine", async () => {
    twoHosts();
    fleetApi({});
    previewGenerationPlacement.mockImplementation((probe: { baseUrl: string }) =>
      Promise.resolve(plannedIn(probe.baseUrl === renderTarget.baseUrl ? 100 : 9_000)),
    );
    wrapper = mountMobileApp();
    await flushPromises();

    await develop();

    // A print is ranked from each machine's captured queue/GPU snapshot, so
    // Auto opens no placement probe at all.
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    // The frozen route carries the winner's URL and its Keychain key.
    expect(admissionTargets()[0]).toEqual(renderTarget);
  });

  it("Most capable prefers the CUDA machine even when it plans later", async () => {
    twoHosts();
    fleetApi({
      studioGpu: {
        name: "Apple M3 Max",
        vram_total_mb: 128_000,
        vram_used_mb: 0,
        backend: "metal",
      },
      renderGpu: {
        name: "NVIDIA GeForce RTX 4090",
        vram_total_mb: 24_000,
        vram_used_mb: 0,
        backend: "cuda",
      },
    });
    previewGenerationPlacement.mockImplementation((probe: { baseUrl: string }) =>
      Promise.resolve(plannedIn(probe.baseUrl === renderTarget.baseUrl ? 9_000 : 100)),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-generate-host']").setValue("capable");
    await flushPromises();

    await develop();
    expect(admissionTargets()[0]).toEqual(renderTarget);
  });

  it("routes only to machines that already have the model", async () => {
    const renderOnly: ModelEntry = { ...model, name: "z-image-turbo:q6", family: "zimage" };
    twoHosts();
    fleetApi({ studioModels: [], renderModels: [renderOnly] });
    wrapper = mountMobileApp();
    await flushPromises();

    // The union picker offers the model even though the browsed machine
    // lacks it, tagged with the machine that has it.
    const modelOptions = fieldControl("Model")
      .findAll("option")
      .map((option) => option.text());
    expect(modelOptions.some((label) => label.includes("Render"))).toBe(true);

    await develop();
    expect(admissionTargets()[0]).toEqual(renderTarget);
  });
});

describe("MobileApp automatic sequence routing", () => {
  const renderTarget = {
    baseUrl: "http://render.tailnet.ts.net:7680",
    apiKey: "render-secret",
  };
  const sequenceModel: ModelEntry = {
    ...model,
    name: "ltx-video-0.9.8-2b-distilled:bf16",
    family: "ltx-video",
    default_steps: 7,
    default_guidance: 1,
  };

  it("freezes the fan-out winner as the sequence's recovery host", async () => {
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
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
      if (path === "/api/chain-jobs" && init?.method === "POST")
        return Promise.resolve({ job_id: "sequence-job-1" });
      if (path === "/api/chain-jobs/sequence-job-1") return new Promise(() => {});
      return durableApiFallback(path, init, route);
    });
    previewChainPlacement.mockImplementation((probe: { baseUrl: string }) => {
      const preview = plannedPlacement();
      preview.candidate.predicted_completion_after_ms =
        probe.baseUrl === renderTarget.baseUrl ? 100 : 9_000;
      return Promise.resolve(preview);
    });

    wrapper = mountMobileApp();
    await flushPromises();
    const prompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("A paper boat crosses a moonlit pond");
    await prompts[1]!.setValue("Fireflies gather as the sky brightens");
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const created = apiJsonTo.mock.calls.filter(
      (call: unknown[]) =>
        call[1] === "/api/chain-jobs" && (call[2] as RequestInit)?.method === "POST",
    );
    expect(created).toHaveLength(1);
    expect(created[0]![0]).toEqual(renderTarget);
    const recovery = localStorage.getItem("mold.mobile.sequence-job.v1");
    expect(JSON.parse(recovery ?? "null")).toEqual({
      hostId: "render-id",
      baseUrl: renderTarget.baseUrl,
      instanceId: "render-id",
      jobId: "sequence-job-1",
    });
    expect(recovery).not.toContain(renderTarget.apiKey);
  });
});

describe("MobileApp routing target consistency", () => {
  const renderTarget = {
    baseUrl: "http://render.tailnet.ts.net:7680",
    apiKey: "render-secret",
  };

  function twoHosts(): void {
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, route);
    });
  }

  async function develop(): Promise<void> {
    await fieldControl("Prompt").setValue("a consistent lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  it("keeps a pinned target and the browsed machine in step", async () => {
    twoHosts();
    wrapper = mountMobileApp();
    await flushPromises();

    // Pin Studio in Create, then use Render for generations from Machines.
    await wrapper.get("[data-test='mobile-generate-host']").setValue("studio-id");
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-hosts']").trigger("click");
    const renderRow = wrapper
      .findAll(".host-row")
      .find((row) => row.find(".host-name").text() === "Render");
    if (!renderRow) throw new Error("Missing Render host row");
    const useForGenerations = renderRow
      .findAll("button")
      .find((button) => button.text() === "Use host");
    if (!useForGenerations) throw new Error("Missing use-for-generations action");
    await useForGenerations.trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-generate']").trigger("click");
    await flushPromises();

    // The picker must never show one machine while work goes to another.
    expect(
      (wrapper.get("[data-test='mobile-generate-host']").element as HTMLSelectElement).value,
    ).toBe("render-id");
    expect(localStorage.getItem("mold.mobile.generate-target.v1")).toBe("render-id");
    await develop();
    expect(admissionTargets()[0]).toEqual(renderTarget);
  });
});

describe("MobileApp Library organization", () => {
  const nowSecs = Math.floor(Date.now() / 1000);
  const organizedCapabilities = {
    gallery: {
      can_delete: true,
      organize: true,
      trash: { enabled: true, retention_days: 30 },
    },
  };

  function libraryPrint(
    filename: string,
    timestamp: number,
    extra: Record<string, unknown> = {},
  ): GalleryImage {
    return {
      filename,
      timestamp,
      format: "png",
      metadata: {
        prompt: `prompt for ${filename}`,
        model: model.name,
        seed: 7,
        steps: 4,
        guidance: 3.5,
        width: 512,
        height: 512,
      },
      ...extra,
    } as GalleryImage;
  }

  const favoritePrint = libraryPrint("fav.png", nowSecs + 4, {
    favorite: true,
    tags: ["Blue"],
    collections: ["c1"],
  });
  const plainPrint = libraryPrint("plain.png", nowSecs + 3);
  const trashedPrint = libraryPrint("gone.png", nowSecs + 2, {
    trashed_at: nowSecs - 86_400,
    purge_at: nowSecs + 3 * 86_400,
  });
  let libraryCollectionHidden = false;

  function installLibraryApi(): void {
    libraryCollectionHidden = false;
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(organizedCapabilities);
      if (path === "/api/gallery") return Promise.resolve([favoritePrint, plainPrint]);
      if (path === "/api/gallery?view=trash") return Promise.resolve([trashedPrint]);
      if (path.startsWith("/api/gallery/collections/") && init?.method === "PATCH") {
        const body = JSON.parse(String(init.body)) as { hidden?: boolean };
        libraryCollectionHidden = body.hidden === true;
        return Promise.resolve({
          id: "c1",
          name: "Portraits",
          slug: "portraits",
          count: 1,
          hidden: libraryCollectionHidden,
        });
      }
      if (path === "/api/gallery/collections") {
        return Promise.resolve([
          {
            id: "c1",
            name: "Portraits",
            slug: "portraits",
            description: null,
            cover_filename: "cover.png",
            count: 1,
            created_at: 1,
            updated_at: 1,
            hidden: libraryCollectionHidden,
          },
        ]);
      }
      if (path === "/api/gallery/tags") return Promise.resolve([{ name: "Blue", count: 1 }]);
      if (path === "/api/gallery/trash" && init?.method === "DELETE") {
        return Promise.resolve({ purged: 1 });
      }
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
    // Thumbnails read blobs; organization mutations read empty JSON bodies.
    apiFetchTo.mockImplementation((_callTarget, _path, init) =>
      Promise.resolve(
        init?.method
          ? new Response(null, { status: 204 })
          : new Response(cachedThumbnailBlob(), {
              status: 200,
              headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
            }),
      ),
    );
  }

  async function openLibrary(): Promise<void> {
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await flushPromises();
  }

  function organizeCalls(path: string): Array<[unknown, string, RequestInit | undefined]> {
    return apiFetchTo.mock.calls.filter(([, calledPath]) => calledPath === path) as Array<
      [unknown, string, RequestInit | undefined]
    >;
  }

  function libraryTouch(type: string, clientY?: number): Event {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, "touches", {
      configurable: true,
      value: clientY === undefined ? [] : [{ identifier: 7, clientX: 120, clientY }],
    });
    Object.defineProperty(event, "changedTouches", {
      configurable: true,
      value: clientY === undefined ? [] : [{ identifier: 7, clientX: 120, clientY }],
    });
    return event;
  }

  it("refreshes the Library after a natural pull from the top", async () => {
    installLibraryApi();
    await openLibrary();
    const content = wrapper!.get(".mobile-content").element;
    const before = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery").length;

    content.dispatchEvent(libraryTouch("touchstart", 100));
    const move = libraryTouch("touchmove", 230);
    content.dispatchEvent(move);
    await flushPromises();
    expect(move.defaultPrevented).toBe(true);
    expect(wrapper!.get("[data-test='mobile-view-pull']").text()).toContain("Release to refresh");
    content.dispatchEvent(libraryTouch("touchend"));

    await vi.waitFor(() =>
      expect(
        apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery").length,
      ).toBeGreaterThan(before),
    );
  });

  it("resets a cancelled Library pull without refreshing", async () => {
    installLibraryApi();
    await openLibrary();
    const content = wrapper!.get(".mobile-content").element;
    const before = apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery").length;

    content.dispatchEvent(libraryTouch("touchstart", 100));
    content.dispatchEvent(libraryTouch("touchmove", 230));
    content.dispatchEvent(libraryTouch("touchcancel"));
    await flushPromises();

    expect(wrapper!.get("[data-test='mobile-view-pull']").text()).not.toContain(
      "Release to refresh",
    );
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/gallery")).toHaveLength(before);
  });

  it("hides every organization affordance when no host advertises it", async () => {
    await openLibrary();

    expect(wrapper?.find("[data-test='mobile-library-scope']").exists()).toBe(false);
    expect(wrapper?.find("[data-test='mobile-library-chips']").exists()).toBe(false);

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    expect(wrapper?.find("[data-test='mobile-gallery-favorite']").exists()).toBe(false);
    expect(wrapper?.find("[data-test='mobile-gallery-tag']").exists()).toBe(false);
    expect(wrapper?.find("[data-test='mobile-gallery-collect']").exists()).toBe(false);

    // Legacy hosts keep today's hard-delete wording.
    await wrapper?.get("[data-test='gallery-item']").trigger("click");
    await wrapper?.get("[data-test='mobile-gallery-delete']").trigger("click");
    expect(wrapper?.get("[data-test='mobile-gallery-actions']").text()).toContain(
      "Delete 1 everywhere?",
    );
  });

  it("renders the scope row with counts and lazily loads the trash", async () => {
    installLibraryApi();
    await openLibrary();

    const scope = wrapper!.get("[data-test='mobile-library-scope']");
    expect(scope.get("[data-test='mobile-library-scope-prints']").text()).toContain("Prints");
    expect(scope.get("[data-test='mobile-library-scope-prints']").text()).toContain("2");
    expect(scope.get("[data-test='mobile-library-scope-collections']").text()).toContain("1");
    expect(apiJsonTo).not.toHaveBeenCalledWith(
      target,
      "/api/gallery?view=trash",
      expect.anything(),
    );

    await scope.get("[data-test='mobile-library-scope-trash']").trigger("click");
    await flushPromises();
    await vi.waitFor(() =>
      expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/gallery?view=trash", expect.anything()),
    );
    await vi.waitFor(() => expect(wrapper?.find("[data-test='purge-chip']").exists()).toBe(true));

    expect(wrapper?.get("[data-test='purge-chip']").text()).toBe("Purges in 3 d");
    const banner = wrapper!.get("[data-test='mobile-library-trash-banner']");
    expect(banner.text()).toContain("Prints stay in the trash");
    expect(banner.text()).toContain("30 d");
    expect(scope.get("[data-test='mobile-library-scope-trash']").text()).toContain("1");
  });

  it("filters the grid with the Favorites chip and marks favorite tiles", async () => {
    installLibraryApi();
    await openLibrary();

    expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(2);
    expect(wrapper?.findAll("[data-test='favorite-badge']")).toHaveLength(1);

    await wrapper?.get("[data-test='mobile-library-chip-favorites']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));
    expect(wrapper?.get("[data-test='gallery-item']").attributes("aria-label")).toContain(
      "fav.png",
    );

    // The host's tags ride the chip row (single host: no host chips).
    expect(wrapper?.find("[data-test='mobile-library-chip-tag']").text()).toContain("Blue");
    expect(wrapper?.find("[data-test='mobile-library-chip-host']").exists()).toBe(false);
  });

  it("lists collections as cards and drills into a member grid", async () => {
    installLibraryApi();
    await openLibrary();

    const gridThumbnail = wrapper!.get("[data-test='gallery-item'] img").attributes("src");

    await wrapper?.get("[data-test='mobile-library-scope-collections']").trigger("click");
    await flushPromises();

    const card = wrapper!.get("[data-test='mobile-collection-portraits']");
    expect(card.text()).toContain("Portraits");
    expect(card.text()).toContain("1");
    await vi.waitFor(() => expect(card.find(".mobile-collection-cover img").exists()).toBe(true));
    expect(card.get(".mobile-collection-cover img").attributes("src")).not.toBe(gridThumbnail);
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(gridThumbnail);

    vi.useFakeTimers();
    await card.get(".mobile-collection-cover img").trigger("error");
    expect(card.find(".mobile-collection-cover img").exists()).toBe(false);
    expect(card.findComponent({ name: "MobileMediaPlaceholder" }).exists()).toBe(true);
    const coverFetchCount = () =>
      apiFetchTo.mock.calls.filter(([, path]) => String(path).includes("cover.png")).length;
    const callsBeforeHidingShelf = coverFetchCount();
    await wrapper?.get("[data-test='mobile-tab-catalog']").trigger("click");
    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    expect(coverFetchCount()).toBe(callsBeforeHidingShelf);
    await wrapper?.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    const restoredCard = wrapper!.get("[data-test='mobile-collection-portraits']");
    await vi.waitFor(() =>
      expect(restoredCard.find(".mobile-collection-cover img").exists()).toBe(true),
    );
    vi.useRealTimers();

    await card.get("[data-test='mobile-collection-open']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));
    expect(wrapper?.get("[data-test='mobile-collection-drillin']").text()).toContain("Portraits");

    await wrapper?.get("[data-test='mobile-collection-back']").trigger("click");
    await flushPromises();
    expect(wrapper?.find("[data-test='mobile-collection-list']").exists()).toBe(true);
  });

  it("falls back to a readable collection member when its explicit cover is unavailable", async () => {
    installLibraryApi();
    apiFetchTo.mockImplementation((_callTarget: unknown, path: string, _init?: RequestInit) => {
      if (path.includes("cover.png")) return Promise.reject(new Error("cover host offline"));
      return Promise.resolve(
        new Response(cachedThumbnailBlob("member thumbnail"), {
          status: 200,
          headers: { "x-mold-thumbnail-rendition": `${thumbnailTier()}-jpg` },
        }),
      );
    });
    await openLibrary();

    await wrapper?.get("[data-test='mobile-library-scope-collections']").trigger("click");
    await flushPromises();

    const card = wrapper!.get("[data-test='mobile-collection-portraits']");
    await vi.waitFor(() => expect(card.find(".mobile-collection-cover img").exists()).toBe(true));
    expect(
      apiFetchTo.mock.calls.some(([, path]) => String(path).includes(favoritePrint.filename)),
    ).toBe(true);
  });

  it("offers Hide from Library from a collection card on iPhone and Android", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-library-scope-collections']").trigger("click");
    await flushPromises();
    const card = wrapper!.get("[data-test='mobile-collection-portraits']");
    await card.get("[data-test='mobile-collection-menu']").trigger("click");
    expect(card.get("[data-test='mobile-collection-hidden']").text()).toBe("Hide from Library");
    await card.get("[data-test='mobile-collection-hidden']").trigger("click");
    await flushPromises();

    const patch = apiJsonTo.mock.calls.find(
      ([, path, init]) => path === "/api/gallery/collections/c1" && init?.method === "PATCH",
    );
    expect(JSON.parse(String(patch?.[2]?.body))).toEqual({ hidden: true });
    await wrapper?.get("[data-test='mobile-library-scope-prints']").trigger("click");
    await flushPromises();
    expect(wrapper?.get("[data-test='mobile-library-scope-prints']").text()).toContain("1");
    expect(wrapper?.find("[data-test='mobile-library-chip-tag']").exists()).toBe(false);
  });

  it("fans a bulk favorite out through /api/gallery/organize", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    const tiles = wrapper!.findAll("[data-test='gallery-item']");
    const plainTile = tiles.find((tile) => tile.attributes("aria-label")?.includes("plain.png"));
    await plainTile!.trigger("click");
    await wrapper?.get("[data-test='mobile-gallery-favorite']").trigger("click");
    await flushPromises();

    const calls = organizeCalls("/api/gallery/organize");
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0]?.[2]?.body))).toEqual({
      filenames: ["plain.png"],
      favorite: true,
    });
    expect(wrapper?.findAll("[data-test='favorite-badge']")).toHaveLength(0); // select mode hides badges
    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    expect(wrapper?.findAll("[data-test='favorite-badge']")).toHaveLength(2);
  });

  it("adds a tag to the selection from the tag sheet", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper?.get("[data-test='gallery-item']").trigger("click");
    await wrapper?.get("[data-test='mobile-gallery-tag']").trigger("click");

    await wrapper?.get("[data-test='mobile-tag-sheet-input']").setValue("Grain");
    await wrapper?.get("[data-test='mobile-tag-sheet-add']").trigger("submit");
    await flushPromises();

    const calls = organizeCalls("/api/gallery/organize");
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0]?.[2]?.body))).toEqual({
      filenames: ["fav.png"],
      add_tags: ["Grain"],
    });
  });

  it("drops an unfavorited print from the grid while the Favorites filter is on", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-library-chip-favorites']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper?.get("[data-test='gallery-item']").trigger("click");
    // Every selected print is a favorite, so ♥ unfavorites — the print no
    // longer matches the active filter and must leave the grid immediately.
    await wrapper?.get("[data-test='mobile-gallery-favorite']").trigger("click");
    await flushPromises();

    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(0));
  });

  it("drops a print from the grid when its active-filter tag is removed", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-library-chip-tag']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper?.get("[data-test='gallery-item']").trigger("click");
    await wrapper?.get("[data-test='mobile-gallery-tag']").trigger("click");
    await wrapper?.get("[data-test='mobile-tag-sheet-remove']").trigger("click");
    await flushPromises();

    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(0));
  });

  it("keeps an incomplete trash snapshot retry-eligible instead of authoritative", async () => {
    installLibraryApi();
    const base = apiJsonTo.getMockImplementation()!;
    let trashReads = 0;
    apiJsonTo.mockImplementation(
      (requestTarget: unknown, path: string, init?: RequestInit): Promise<unknown> => {
        if (path === "/api/gallery?view=trash") {
          trashReads += 1;
          return trashReads === 1
            ? Promise.reject(new Error("trash listing failed"))
            : Promise.resolve([trashedPrint]);
        }
        return base(requestTarget, path, init) as Promise<unknown>;
      },
    );
    await openLibrary();

    await wrapper?.get("[data-test='mobile-library-scope-trash']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(trashReads).toBe(1));
    expect(wrapper?.text()).toContain("unavailable");

    // The failed pass never becomes the authoritative snapshot: re-entering
    // the Trash scope refetches instead of trusting the incomplete read.
    await wrapper?.get("[data-test='mobile-library-scope-prints']").trigger("click");
    await flushPromises();
    await wrapper?.get("[data-test='mobile-library-scope-trash']").trigger("click");
    await vi.waitFor(() => expect(trashReads).toBe(2));
    await vi.waitFor(() => expect(wrapper?.find("[data-test='purge-chip']").exists()).toBe(true));
  });

  it("removes a disconnected machine's tag chips from the merged Library", async () => {
    const platoTarget = { baseUrl: "http://plato.tailnet.ts.net:7680", apiKey: "secret" };
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
          id: "plato-id",
          name: "Plato",
          baseUrl: platoTarget.baseUrl,
          hostname: "plato",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string, init?: RequestInit) => {
      const fromPlato = (requestTarget as { baseUrl: string }).baseUrl === platoTarget.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve(
          fromPlato ? { ...status, hostname: "plato", instance_id: "plato-id" } : status,
        );
      }
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") return Promise.resolve(organizedCapabilities);
      if (path === "/api/gallery")
        return Promise.resolve(fromPlato ? [plainPrint] : [favoritePrint]);
      if (path === "/api/gallery/collections") return Promise.resolve([]);
      if (path === "/api/gallery/tags") {
        return Promise.resolve(
          fromPlato ? [{ name: "Haunt", count: 1 }] : [{ name: "Blue", count: 1 }],
        );
      }
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, requestTarget);
    });
    apiFetchTo.mockImplementation(() =>
      Promise.resolve({
        status: 204,
        blob: () => Promise.resolve(new Blob(["thumbnail"])),
        text: () => Promise.resolve(""),
      } as unknown as Response),
    );
    await openLibrary();
    const chipNames = () =>
      wrapper!.findAll("[data-test='mobile-library-chip-tag']").map((chip) => chip.text());
    await vi.waitFor(() => expect(chipNames().join(" ")).toContain("Haunt"));

    // Disconnect Plato from Machines, then return to the Library.
    const machinesTab = wrapper!
      .findAll("button.mobile-tab")
      .find((button) => button.text() === "Machines");
    await machinesTab!.trigger("click");
    await flushPromises();
    const platoRow = wrapper!
      .findAll("[data-test='mobile-host-row']")
      .find((row) => row.text().includes("Plato"));
    await platoRow!.trigger("click");
    await flushPromises();
    await wrapper!.get("[data-test='host-detail-disconnect']").trigger("click");
    await flushPromises();
    await wrapper!.get("[data-test='mobile-tab-gallery']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const chips = chipNames().join(" ");
    expect(chips).toContain("Blue");
    expect(chips).not.toContain("Haunt");
  });

  it("moves a selection to the trash behind the two-tap confirm", async () => {
    installLibraryApi();
    await openLibrary();

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper?.get("[data-test='gallery-item']").trigger("click");

    const deleteButton = wrapper!.get("[data-test='mobile-gallery-delete']");
    expect(deleteButton.text()).toBe("Trash");
    await deleteButton.trigger("click");
    expect(wrapper?.get("[data-test='mobile-gallery-actions']").text()).toContain(
      "Move 1 to trash?",
    );
    expect(organizeCalls("/api/gallery/trash")).toHaveLength(0);

    await deleteButton.trigger("click");
    await flushPromises();
    const calls = organizeCalls("/api/gallery/trash");
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0]?.[2]?.body))).toEqual({ filenames: ["fav.png"] });
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(1));
  });

  async function openTrashScope(): Promise<void> {
    await wrapper?.get("[data-test='mobile-library-scope-trash']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='purge-chip']").exists()).toBe(true));
  }

  it("restores a trashed selection back into the Library", async () => {
    installLibraryApi();
    await openLibrary();
    await openTrashScope();

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    await wrapper?.get("[data-test='gallery-item']").trigger("click");

    // Restore is the primary action.
    await wrapper?.get("[data-test='mobile-gallery-restore']").trigger("click");
    await flushPromises();
    const restores = organizeCalls("/api/gallery/trash/restore");
    expect(restores).toHaveLength(1);
    expect(JSON.parse(String(restores[0]?.[2]?.body))).toEqual({ filenames: ["gone.png"] });
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-library-empty']").exists()).toBe(true),
    );

    // The restored print rejoined the live Library locally.
    await wrapper?.get("[data-test='mobile-library-scope-prints']").trigger("click");
    await flushPromises();
    await vi.waitFor(() => expect(wrapper?.findAll("[data-test='gallery-item']")).toHaveLength(3));
  });

  it("deletes forever from the Trash scope behind the two-tap confirm", async () => {
    installLibraryApi();
    await openLibrary();
    await openTrashScope();

    await wrapper?.get("[data-test='mobile-gallery-select']").trigger("click");
    const tiles = wrapper!.findAll("[data-test='gallery-item']");
    const trashedTile = tiles.find((tile) => tile.attributes("aria-label")?.includes("gone.png"));
    await trashedTile!.trigger("click");

    const deleteButton = wrapper!.get("[data-test='mobile-gallery-delete']");
    expect(deleteButton.text()).toBe("Delete forever");
    await deleteButton.trigger("click");
    expect(wrapper?.get("[data-test='mobile-gallery-actions']").text()).toContain(
      "Delete 1 forever?",
    );
    expect(organizeCalls("/api/gallery/image/gone.png?permanent=true")).toHaveLength(0);
    await deleteButton.trigger("click");
    await flushPromises();
    const forever = organizeCalls("/api/gallery/image/gone.png?permanent=true");
    expect(forever).toHaveLength(1);
    expect(forever[0]?.[2]?.method).toBe("DELETE");
  });

  it("empties the trash from the header behind a two-step confirm", async () => {
    installLibraryApi();
    await openLibrary();
    await openTrashScope();

    const emptyCalls = () =>
      apiJsonTo.mock.calls.filter(
        ([, path, init]) =>
          path === "/api/gallery/trash" && (init as RequestInit | undefined)?.method === "DELETE",
      );

    const button = wrapper!.get("[data-test='mobile-library-empty-trash']");
    await button.trigger("click");
    expect(emptyCalls()).toHaveLength(0);
    expect(wrapper?.get("[data-test='mobile-library-empty-prompt']").text()).toContain(
      "Delete everything in the trash forever?",
    );

    await button.trigger("click");
    await flushPromises();
    expect(emptyCalls()).toHaveLength(1);
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-library-empty']").exists()).toBe(true),
    );
  });
});

describe("MobileApp Create title", () => {
  it("carries a trimmed title on the outgoing request and omits it when blank", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Title").setValue("  Grain test 01 ");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(1);
    expect(admittedRequests()[0]?.title).toBe("Grain test 01");

    await fieldControl("Title").setValue("   ");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(admittedRequests()).toHaveLength(2);
    expect(openStreams[1]?.options.body.title).toBeUndefined();
  });

  it("keeps the Title field for Sequence output and stamps the stitched print", async () => {
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
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model, sequenceModel]);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
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
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Title").setValue("Storm study");
    const sequenceSegment = wrapper
      .get("[data-test='mobile-output-mode']")
      .findAll("button")
      .find((candidate) => candidate.text() === "Sequence");
    await sequenceSegment!.trigger("click");
    await flushPromises();

    // The chain wire carries `title` for the STITCHED print, so the field
    // stays visible and the old "sequences have no title" note is retired.
    expect(
      wrapper.findAll("label.field").some((field) => field.find("span").text() === "Title"),
    ).toBe(true);
    expect(wrapper.find("[data-test='mobile-sequence-title-note']").exists()).toBe(false);

    const prompts = wrapper.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("a paper boat");
    await prompts[1]!.setValue("fireflies gather");
    await wrapper.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const post = apiJsonTo.mock.calls.find(
      ([, path, init]) =>
        path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeTruthy();
    expect(JSON.parse(String((post![2] as RequestInit).body)).title).toBe("Storm study");
  });

  it("refuses an over-long title with an inline error instead of dropping it", async () => {
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Title").setValue("x".repeat(121));
    expect(wrapper.get("[data-test='mobile-create-title-error']").text()).toBe(
      "Titles are at most 120 characters.",
    );
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(0);
  });
});

describe("MobileApp Create File under", () => {
  const filingCapabilities = {
    gallery: {
      can_delete: true,
      organize: true,
      trash: { enabled: true, retention_days: 30 },
    },
  };
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

  function installFilingApi(organize = true): void {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model, sequenceModel]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          ...durableQueueCapabilities,
          ...(organize ? filingCapabilities : { gallery: { can_delete: true, organize: false } }),
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/gallery/collections") {
        return Promise.resolve([
          {
            id: "c1",
            name: "Smurfs",
            slug: "smurfs",
            description: null,
            cover_filename: null,
            count: 12,
            created_at: 1,
            updated_at: 1,
          },
        ]);
      }
      if (path === "/api/gallery/tags") return Promise.resolve([{ name: "#kodak", count: 4 }]);
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
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, callTarget);
    });
  }

  async function openCreateWithFiling(organize = true): Promise<void> {
    installFilingApi(organize);
    wrapper = mountMobileApp();
    await flushPromises();
    if (organize) {
      await vi.waitFor(() =>
        expect(wrapper?.find("[data-test='mobile-file-under']").exists()).toBe(true),
      );
    }
  }

  async function chooseSequenceOutput(): Promise<void> {
    const sequenceSegment = wrapper!
      .get("[data-test='mobile-output-mode']")
      .findAll("button")
      .find((candidate) => candidate.text() === "Sequence");
    await sequenceSegment!.trigger("click");
    await flushPromises();
  }

  function chainBody(): Record<string, unknown> {
    const post = apiJsonTo.mock.calls.find(
      ([, path, init]) =>
        path === "/api/chain-jobs" && (init as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeTruthy();
    return JSON.parse(String((post![2] as RequestInit).body)) as Record<string, unknown>;
  }

  it("hides File under and files nothing while no reachable machine can organize", async () => {
    await openCreateWithFiling(false);

    expect(wrapper!.find("[data-test='mobile-file-under']").exists()).toBe(false);

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    // The title still rides; only the filing is withheld from a machine that
    // would reject it.
    expect(admittedRequests()[0]?.title).toBe("Smurfs");
    expect(admittedRequests()[0]?.tags).toBeUndefined();
    expect(admittedRequests()[0]?.collection).toBeUndefined();
  });

  it("files a one-shot print under the ghost tag and the title-matched collection", async () => {
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-ghost']").text()).toContain("smurfs");
    expect(wrapper!.find("[data-test='mobile-file-under-collection-match']").exists()).toBe(true);

    await fieldControl("Prompt").setValue("a village of small blue characters");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()[0]?.tags).toEqual(["smurfs"]);
    // Always by name: the routed machine resolves or creates it by slug.
    expect(admittedRequests()[0]?.collection).toEqual({ name: "Smurfs" });
  });

  it("carries a tag typed in the sheet and honours a removed ghost chip", async () => {
    await openCreateWithFiling();
    await fieldControl("Title").setValue("Smurfs");

    await wrapper!.get("[data-test='mobile-file-under-add-tag']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-tag-input']").setValue("#kodak");
    await wrapper!.get("[data-test='mobile-file-under-tag-add']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-tag-sheet-done']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-ghost-remove']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-collection-clear']").trigger("click");

    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()[0]?.tags).toEqual(["kodak"]);
    expect(admittedRequests()[0]?.collection).toBeUndefined();
  });

  it("drops the ghost tag when the Settings auto-tag preference is off", async () => {
    localStorage.setItem(
      "mold.mobile.settings.v1",
      JSON.stringify({ theme: "system", themeFamily: "safelight", autoTagTitle: false }),
    );
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);

    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()[0]?.tags).toBeUndefined();
    // The collection match is a separate decision and still applies.
    expect(admittedRequests()[0]?.collection).toEqual({ name: "Smurfs" });
  });

  it("gives every prepared Batch N sibling the same filing", async () => {
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    await wrapper!.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper!.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three variations of a storm");
    await wrapper!.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper!.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(3);
    for (const request of admittedRequests()) {
      expect(request.title).toBe("Smurfs");
      expect(request.tags).toEqual(["smurfs"]);
      expect(request.collection).toEqual({ name: "Smurfs" });
    }
  });

  it("files a sequence's stitched print from the same draft, title included", async () => {
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    await chooseSequenceOutput();

    // The chain wire carries `title`/`tags`/`collection` now, so the Title
    // field stays visible and the old "sequences have no title" note is gone.
    expect(
      wrapper!.findAll("label.field").some((field) => field.find("span").text() === "Title"),
    ).toBe(true);
    expect(wrapper!.find("[data-test='mobile-sequence-title-note']").exists()).toBe(false);
    expect(wrapper!.find("[data-test='mobile-file-under']").exists()).toBe(true);

    const prompts = wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("a paper boat");
    await prompts[1]!.setValue("fireflies gather");
    await wrapper!.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const body = chainBody();
    expect(body.title).toBe("Smurfs");
    expect(body.tags).toEqual(["smurfs"]);
    expect(body.collection).toEqual({ name: "Smurfs" });
    // Filing describes the stitched print, never an intermediate clip.
    for (const stage of body.stages as Array<Record<string, unknown>>) {
      expect(stage).not.toHaveProperty("tags");
      expect(stage).not.toHaveProperty("collection");
    }
  });

  it("restores the filing a print actually landed with on Use as prompt", async () => {
    const filed: GalleryImage = {
      filename: "filed.png",
      timestamp: Math.floor(Date.now() / 1000) + 6,
      format: "png",
      tags: ["Blue"],
      metadata: {
        prompt: "a village of small blue characters",
        model: model.name,
        seed: 7,
        steps: 4,
        guidance: 3.5,
        width: 512,
        height: 512,
        title: "Smurfs",
        // The print landed WITHOUT its own title tag: that is an opt-out, and
        // re-offering the ghost chip would quietly re-file the reuse.
        tags: ["Blue"],
        collection: "River studies",
      },
    } as GalleryImage;
    installFilingApi();
    const withPrint = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target: unknown, path: string, init?: RequestInit) =>
      path === "/api/gallery" ? Promise.resolve([filed]) : withPrint(target, path, init),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect((fieldControl("Title").element as HTMLInputElement).value).toBe("Smurfs");
    expect(wrapper.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-file-under-tag']").text()).toContain("Blue");
    expect(wrapper.get("[data-test='mobile-file-under-collection']").text()).toContain(
      "River studies",
    );
    // An explicit pick outranks the title match, so nothing claims a match.
    expect(wrapper.find("[data-test='mobile-file-under-collection-match']").exists()).toBe(false);
  });

  it("re-derives the ghost chip against a Library rename rather than the stamp", async () => {
    const renamed: GalleryImage = {
      filename: "renamed.png",
      timestamp: Math.floor(Date.now() / 1000) + 6,
      format: "png",
      // The Library rename wins over the metadata stamp for the title, so the
      // ghost opt-out has to be judged against the name this reuse carries.
      title: "Harbour lights",
      metadata: {
        prompt: "a village of small blue characters",
        model: model.name,
        seed: 7,
        steps: 4,
        guidance: 3.5,
        width: 512,
        height: 512,
        title: "Smurfs",
        tags: ["smurfs", "Blue"],
      },
    } as GalleryImage;
    installFilingApi();
    const withPrint = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target: unknown, path: string, init?: RequestInit) =>
      path === "/api/gallery" ? Promise.resolve([renamed]) : withPrint(target, path, init),
    );
    wrapper = mountMobileApp();
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect((fieldControl("Title").element as HTMLInputElement).value).toBe("Harbour lights");
    // Never invent `harbour-lights`: the print was not filed under it.
    expect(wrapper.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);
    // And never drop `smurfs`, which the print really does carry.
    expect(
      wrapper.findAll("[data-test='mobile-file-under-tag']").map((chip) => chip.text()),
    ).toEqual([expect.stringContaining("smurfs"), expect.stringContaining("Blue")]);
  });

  it("leaves the print's name and filing alone when a template is loaded", async () => {
    // A template snapshots the whole form, so it also carries whatever title,
    // filing, and auto-tag mirror were live when it was saved. None of those
    // are generation settings: loading a template must not rename the print
    // in progress, re-file it, or silently switch the ghost chip off.
    const stale = newGenerateForm();
    stale.model = model.name;
    stale.family = model.family;
    stale.prompt = "a template prompt";
    stale.title = "Someone else's name";
    stale.fileUnderAutoTag = false;
    stale.fileUnder = { ...stale.fileUnder, manualTags: ["stale"] };
    saveGenerationTemplate("Storm", stale, MOBILE_GENERATION_TEMPLATES_STORAGE_KEY, "studio-id");
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    await wrapper!.get("[data-test='mobile-template-disclosure']").trigger("click");
    await wrapper!.get("[data-test='mobile-template-load']").trigger("click");
    await flushPromises();

    expect(fieldControl("Prompt").element).toHaveProperty("value", "a template prompt");
    expect((fieldControl("Title").element as HTMLInputElement).value).toBe("Smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-ghost']").text()).toContain("smurfs");
    expect(wrapper!.findAll("[data-test='mobile-file-under-tag']")).toHaveLength(0);
  });

  it("keeps the print's name and filing across both Reset controls", async () => {
    // Neither Reset restores a MODEL default here: the print's name and the
    // Library filing it carries are the user's, not the checkpoint's.
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    await wrapper!.get("[data-test='mobile-file-under-add-tag']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-tag-input']").setValue("blue");
    await wrapper!.get("[data-test='mobile-file-under-tag-add']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-tag-sheet-done']").trigger("click");

    await wrapper!.get("[data-test='mobile-settings-reset']").trigger("click");
    await flushPromises();

    expect((fieldControl("Title").element as HTMLInputElement).value).toBe("Smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-ghost']").text()).toContain("smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-tag']").text()).toContain("blue");

    await wrapper!.get("[data-test='mobile-open-advanced']").trigger("click");
    await wrapper!.get("[data-test='mobile-advanced-reset']").trigger("click");
    await flushPromises();

    expect((fieldControl("Title").element as HTMLInputElement).value).toBe("Smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-ghost']").text()).toContain("smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-tag']").text()).toContain("blue");
  });

  it("freezes the title with the filing it derived, not the one typed mid-flight", async () => {
    // Source fitting and the placement fan-out can run for minutes. A title
    // edited inside that window must not reach the wire on its own: its ghost
    // tag and its collection match were derived from the OLD name, so shipping
    // the new one alone files the print under a name nothing else agrees with.
    const fitting = deferred<{ source: string; mask: string | null; changed: boolean }>();
    applySourceFitPreprocess.mockImplementationOnce(() => fitting.promise);
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a village of small blue characters");
    const form = wrapper!.getComponent(MobileLoraControls).props("form") as GenerateForm;
    form.sourceImage = "c3JjCg==";
    await flushPromises();

    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(0);

    // The edit lands while the source is still being prepared.
    await fieldControl("Title").setValue("Harbour lights");
    fitting.resolve({ source: "c3JjCg==", mask: null, changed: false });
    await flushPromises();

    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]?.title).toBe("Smurfs");
    expect(admittedRequests()[0]?.tags).toEqual(["smurfs"]);
    expect(admittedRequests()[0]?.collection).toEqual({ name: "Smurfs" });
  });

  it("refuses an over-long title at the tap instead of blaming the source", async () => {
    await openCreateWithFiling();

    await fieldControl("Title").setValue("x".repeat(121));
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(openStreams).toHaveLength(0);
    expect(wrapper!.get("[data-test='mobile-create-title-error']").text()).toBe(
      "Titles are at most 120 characters.",
    );
    // The old throw surfaced from inside source preparation.
    expect(wrapper!.text()).not.toContain("Couldn’t prepare the source image");
  });

  it("never mirrors the derived collection match onto the live form", async () => {
    // `cloneGenerateForm(form)` is the prepared/quick "inputs changed while the
    // source was being prepared" fence. The title match is derived from a
    // listing that can land at any moment — a host reconnecting, a first
    // capability read — so mirroring it onto the form would refuse a
    // submission nothing the user touched had changed. Only the submission
    // clone carries it, written at the tap by `applyFileUnderPolicy`.
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");
    expect(wrapper!.find("[data-test='mobile-file-under-collection-match']").exists()).toBe(true);

    const form = wrapper!.getComponent(MobileLoraControls).props("form") as GenerateForm;
    expect(form.fileUnderMatch).toBeNull();

    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()[0]?.collection).toEqual({ name: "Smurfs" });
    expect(form.fileUnderMatch).toBeNull();
  });

  it("hides the group when only a machine that cannot run the model can file", async () => {
    // Auto routes by model. A peer that advertises `gallery.organize` but does
    // not hold the selected checkpoint is never a candidate, so it must not
    // qualify a group whose print lands on the machine that CAN run it and
    // then silently drops the filing.
    const filer = {
      id: "filer-id",
      name: "Filer",
      baseUrl: "http://filer:7680",
      hostname: "filer",
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          online: false,
        },
        { ...filer, online: false },
      ]),
    );
    apiJsonTo.mockImplementation((probe: unknown, path: string, init?: RequestInit) => {
      const isFiler = (probe as { baseUrl?: string } | undefined)?.baseUrl === filer.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve(
          isFiler ? { ...status, hostname: "filer", instance_id: "filer-id" } : status,
        );
      }
      // Only Studio holds the model; only the Filer can organize.
      if (path === "/api/models") return Promise.resolve(isFiler ? [] : [model]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          ...durableQueueCapabilities,
          gallery: { can_delete: true, organize: isFiler },
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/gallery/collections") return Promise.resolve([]);
      if (path === "/api/gallery/tags") return Promise.resolve([]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, target);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-file-under']").exists()).toBe(false);

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(1);
    expect(admittedRequests()[0]?.tags).toBeUndefined();
    expect(admittedRequests()[0]?.collection).toBeUndefined();
  });

  // ── The machine an automatic policy actually picked ───────────────────────
  // The group's gate reads the CANDIDATE set, so any machine that could file
  // qualifies it. Auto then picks exactly ONE out of that set on model and
  // capacity grounds. A winner that cannot organize must not be sent fields it
  // would quietly ignore.
  const filerTarget = { baseUrl: "http://render.tailnet.ts.net:7680", apiKey: "render-secret" };

  function twoMachines(): void {
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
          baseUrl: filerTarget.baseUrl,
          hostname: "render",
          instanceId: "render-id",
        },
      ]),
    );
    invoke.mockImplementation((command: string, args?: { hostId?: string }) =>
      Promise.resolve(
        command === "keychain_get_api_key"
          ? args?.hostId === "render-id"
            ? filerTarget.apiKey
            : target.apiKey
          : null,
      ),
    );
  }

  /** Both machines hold the model; only Studio can organize. */
  let autoWinnerBaseUrl: string | null = null;

  function fleetFilingApi(): void {
    apiJsonTo.mockImplementation((probe: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = probe.baseUrl === filerTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
          // Auto ranks a print from each machine's own captured queue depth.
          queue_depth: autoWinnerBaseUrl === null ? 0 : probe.baseUrl === autoWinnerBaseUrl ? 0 : 9,
        });
      if (path === "/api/models") return Promise.resolve([model]);
      if (path === "/api/capabilities") {
        return Promise.resolve({
          ...durableQueueCapabilities,
          ...(render ? { gallery: { can_delete: true, organize: false } } : filingCapabilities),
        });
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/gallery/collections") {
        return Promise.resolve(
          render
            ? []
            : [
                {
                  id: "c1",
                  name: "Smurfs",
                  slug: "smurfs",
                  description: null,
                  cover_filename: null,
                  count: 12,
                  created_at: 1,
                  updated_at: 1,
                },
              ],
        );
      }
      if (path === "/api/gallery/tags") return Promise.resolve([]);
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, probe);
    });
  }

  /**
   * Steer Auto onto one machine. A print is ranked from each machine's own
   * captured queue depth — it opens no placement probe — so the loser is
   * simply the busier machine.
   */
  function autoWinner(baseUrl: string): void {
    autoWinnerBaseUrl = baseUrl;
    const plan = (probe: { baseUrl: string }) => {
      const preview = plannedPlacement();
      preview.candidate.predicted_completion_after_ms = probe.baseUrl === baseUrl ? 100 : 9_000;
      return Promise.resolve(preview);
    };
    previewGenerationPlacement.mockImplementation(plan);
    previewChainPlacement.mockImplementation(plan);
  }

  async function openFleetCreate(): Promise<void> {
    twoMachines();
    fleetFilingApi();
    wrapper = mountMobileApp();
    await flushPromises();
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-file-under']").exists()).toBe(true),
    );
  }

  it("drops the filing and names the machine when Auto lands on one that can't organize", async () => {
    autoWinner(filerTarget.baseUrl);
    await openFleetCreate();

    await fieldControl("Title").setValue("Smurfs");
    // The group is offered because Studio, a candidate, can file.
    expect(wrapper!.get("[data-test='mobile-file-under-ghost']").text()).toContain("smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    // Routing is a model/capacity decision and is NOT narrowed by filing: the
    // incapable machine still won, and the print still develops on it.
    expect(admittedRequests()).toHaveLength(1);
    expect(admissionTargets()[0]).toEqual(filerTarget);
    expect(admittedRequests()[0]?.tags).toBeUndefined();
    expect(admittedRequests()[0]?.collection).toBeUndefined();
    // The title is not filing — it rides regardless.
    expect(admittedRequests()[0]?.title).toBe("Smurfs");

    const banner = wrapper!.get("[data-test='mobile-file-under-dropped']");
    expect(banner.text()).toContain("Render");
    expect(banner.text()).toContain("The print still develops.");
  });

  it("keeps the filing when Auto lands on a machine that can organize", async () => {
    autoWinner(target.baseUrl);
    await openFleetCreate();

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admissionTargets()[0]).toEqual(target);
    expect(admittedRequests()[0]?.tags).toEqual(["smurfs"]);
    expect(admittedRequests()[0]?.collection).toEqual({ name: "Smurfs" });
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(false);
  });

  it("gives every prepared Batch N sibling the same dropped outcome, once", async () => {
    // Prepared work never re-routes: it freezes the BROWSED machine at
    // preparation and develops there, which under an automatic policy need not
    // be a machine the candidate-set gate spoke for. Browsing Render while
    // Studio keeps the group visible is exactly that seam.
    localStorage.setItem("mold.mobile.selected-host.v1", "render-id");
    await openFleetCreate();

    await fieldControl("Title").setValue("Smurfs");
    await wrapper!.get("[data-test='mobile-batch-increment']").trigger("click");
    await wrapper!.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("three variations of a storm");
    await wrapper!.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    await wrapper!.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(3);
    for (const admissionTarget of admissionTargets()) {
      expect(admissionTarget).toEqual(filerTarget);
    }
    for (const request of admittedRequests()) {
      expect(request.title).toBe("Smurfs");
      expect(request.tags).toBeUndefined();
      expect(request.collection).toBeUndefined();
    }
    // One outcome for the whole batch, reported once.
    expect(wrapper!.findAll("[data-test='mobile-file-under-dropped']")).toHaveLength(1);
    expect(wrapper!.get("[data-test='mobile-file-under-dropped']").text()).toContain("Render");
  });

  it("leaves the pinned path alone — the group hides and nothing is dropped", async () => {
    await openFleetCreate();
    await wrapper!.get("[data-test='mobile-generate-host']").setValue("render-id");
    await flushPromises();

    // Pinned to the machine that cannot file: the group is simply not offered,
    // so there is nothing to drop and nothing to explain.
    expect(wrapper!.find("[data-test='mobile-file-under']").exists()).toBe(false);

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()[0]?.tags).toBeUndefined();
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(false);
  });

  it("keeps the dropped notice as a persistent inline banner, never a toast", async () => {
    autoWinner(filerTarget.baseUrl);
    await openFleetCreate();

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    const banner = wrapper!.get("[data-test='mobile-file-under-dropped']");
    expect(banner.attributes("role")).toBe("alert");
    // Inline in the Create stack, not floating chrome.
    expect(banner.element.closest(".mobile-content")).not.toBeNull();

    // Nothing retires it on a timer: it is still there after the queue settles.
    await flushPromises();
    await flushPromises();
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(true);

    // Only an explicit 44pt action does.
    await wrapper!.get("[data-test='mobile-file-under-dropped-dismiss']").trigger("click");
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(false);
  });

  it("supersedes the notice when the next print files successfully", async () => {
    autoWinner(filerTarget.baseUrl);
    await openFleetCreate();

    await fieldControl("Title").setValue("Smurfs");
    await fieldControl("Prompt").setValue("a lighthouse");
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(true);

    // Auto's ranking is captured at poll time, so pin the next print to the
    // machine that CAN file rather than waiting for a re-rank.
    await wrapper!.get("[data-test='mobile-generate-host']").setValue("studio-id");
    await flushPromises();
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(2);
    expect(admittedRequests()[1]?.tags).toEqual(["smurfs"]);
    expect(wrapper!.find("[data-test='mobile-file-under-dropped']").exists()).toBe(false);
  });

  it("drops the stitched print's filing when the sequence machine can't organize", async () => {
    // A sequence freezes the browsed machine unless the fan-out replaces it,
    // and its filing rides the chain body — the same silent-loss seam, one
    // endpoint over.
    localStorage.setItem("mold.mobile.selected-host.v1", "render-id");
    twoMachines();
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
    apiJsonTo.mockImplementation((probe: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = probe.baseUrl === filerTarget.baseUrl;
      if (path === "/api/status")
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      if (path === "/api/models") return Promise.resolve([model, sequenceModel]);
      if (path === "/api/capabilities") {
        return Promise.resolve(
          render ? { gallery: { can_delete: true, organize: false } } : filingCapabilities,
        );
      }
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/gallery/collections") return Promise.resolve([]);
      if (path === "/api/gallery/tags") return Promise.resolve([]);
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
      if (path === "/api/activity")
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      return durableApiFallback(path, init, target);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='mobile-file-under']").exists()).toBe(true),
    );
    autoWinner(filerTarget.baseUrl);

    await fieldControl("Title").setValue("Smurfs");
    await chooseSequenceOutput();
    const prompts = wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea");
    await prompts[0]!.setValue("a paper boat");
    await prompts[1]!.setValue("fireflies gather");
    await wrapper!.get("[data-test='mobile-generate-sequence']").trigger("click");
    await flushPromises();

    const body = chainBody();
    // The timeline still renders; only the filing is withheld, and said.
    expect(body.title).toBe("Smurfs");
    expect(body.tags).toBeUndefined();
    expect(body.collection).toBeUndefined();
    expect(wrapper!.get("[data-test='mobile-file-under-dropped']").text()).toContain("Render");
  });

  it("previews the creation-time filename with the title slug", async () => {
    await openCreateWithFiling();

    await fieldControl("Title").setValue("Smurfs");

    expect(wrapper!.get("[data-test='mobile-file-under-filename']").text()).toMatch(
      /files as mold-ltx2-q8-\d+~smurfs\./,
    );
  });
});

describe("MobileApp identity photo", () => {
  /** A 1×1 PNG the shared header pre-checks accept. */
  const PNG_1X1 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

  const identityModel: ModelEntry = {
    name: "flux-dev:q8",
    family: "flux",
    size_gb: 12,
    is_loaded: false,
    hf_repo: "example/flux",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "Identity-qualified checkpoint",
    downloaded: true,
    supports_identity: true,
  };
  const plainModel: ModelEntry = {
    ...identityModel,
    name: "flux-schnell:q8",
    description: "No identity adapter",
    supports_identity: false,
  };

  function serveIdentity(models: ModelEntry[], gallery: GalleryImage[] = []): void {
    apiJsonTo.mockImplementation((callTarget: unknown, path: string, init?: RequestInit) => {
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve(models);
      if (path === "/api/gallery") return Promise.resolve(gallery);
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      if (path.startsWith("/api/catalog/installed")) return Promise.resolve([]);
      return durableApiFallback(path, init, callTarget);
    });
  }

  function photoFile(name = "ada.png"): File {
    const binary = atob(PNG_1X1);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new File([bytes], name, { type: "image/png" });
  }

  function well() {
    const found = wrapper?.findComponent(IdentityPhotoWell);
    if (!found?.exists()) throw new Error("Identity well is not mounted");
    return found;
  }

  async function attachPhoto(name = "ada.png"): Promise<void> {
    well().vm.$emit("file", photoFile(name));
    await flushPromises();
  }

  async function develop(prompt = "a portrait in warm light"): Promise<void> {
    await fieldControl("Prompt").setValue(prompt);
    await wrapper!.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
  }

  it("uses Android's native photo picker and stages the returned bytes", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    serveIdentity([identityModel]);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "pick_identity_photo") {
        return Promise.resolve({
          cancelled: false,
          filename: "ada.png",
          mimeType: "image/png",
          sizeBytes: 68,
          dataB64: PNG_1X1,
        });
      }
      return Promise.resolve(null);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='identity-well']").trigger("click");
    expect(wrapper.get("[data-test='mobile-identity-picker']").classes()).toContain("is-open");
    expect(well().props("touchTargetSize")).toBe(48);
    expect(wrapper.get("[data-test='mobile-identity-picker']").attributes("style")).toContain(
      "--mobile-sheet-touch-target: 48px",
    );
    await wrapper.get("[data-test='mobile-identity-pick-library']").trigger("click");
    await flushPromises();

    expect(invoke).toHaveBeenCalledWith("pick_identity_photo", { source: "library" });
    expect(well().props("image")).toBe(PNG_1X1);
    expect(well().props("filename")).toBe("ada.png");
  });

  it("closes Android's identity source sheet on the system back gesture", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='identity-well']").trigger("click");
    window.dispatchEvent(new PopStateEvent("popstate"));
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-identity-picker']").classes()).not.toContain("is-open");
    expect(invoke).not.toHaveBeenCalledWith("pick_identity_photo", expect.anything());
  });

  it("keeps Android semantic tile activation and viewer history in sync", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    const { frames: _frames, fps: _fps, ...stillMetadata } = print.metadata;
    const androidPrint: GalleryImage = {
      ...print,
      filename: "storm.png",
      format: "png",
      metadata: {
        ...stillMetadata,
        model: identityModel.name,
        output_format: "png",
      },
    };
    const historyBack = vi.spyOn(window.history, "back").mockImplementation(() => {});
    serveIdentity([identityModel], [androidPrint]);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    const semanticTile = wrapper.get("[data-test='gallery-item']");
    vi.spyOn(semanticTile.element, "getBoundingClientRect").mockReturnValue({
      left: 100,
      right: 200,
      top: 100,
      bottom: 200,
      width: 100,
      height: 100,
      x: 100,
      y: 100,
      toJSON: () => ({}),
    });
    await semanticTile.trigger("click", { detail: 0, clientX: 0, clientY: 0 });
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);

    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper?.find("[data-test='gallery-viewer']").exists()).toBe(false),
    );
    expect(historyBack).toHaveBeenCalledTimes(1);

    // The real browser removes the marker when history.back() lands.
    window.history.replaceState(null, "");
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click", {
      detail: 0,
      clientX: 0,
      clientY: 0,
    });
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);

    window.dispatchEvent(new PopStateEvent("popstate"));
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(historyBack).toHaveBeenCalledTimes(1);
    window.history.replaceState(null, "");
    historyBack.mockRestore();
  });

  it("keeps an oversized Android pick inline and never stages its bytes", async () => {
    isNativeAndroidRuntime.mockReturnValue(true);
    serveIdentity([identityModel]);
    invoke.mockImplementation((command: string) => {
      if (command === "keychain_get_api_key") return Promise.resolve(target.apiKey);
      if (command === "pick_identity_photo") {
        return Promise.resolve({
          cancelled: false,
          filename: "huge.jpg",
          mimeType: "image/jpeg",
          sizeBytes: 16 * 1024 * 1024 + 1,
          dataB64: PNG_1X1,
        });
      }
      return Promise.resolve(null);
    });
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='identity-well']").trigger("click");
    await wrapper.get("[data-test='mobile-identity-pick-camera']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain("16 MiB");
    expect(well().props("image")).toBeNull();
  });

  it("mounts the well and the two knobs only for a checkpoint that advertises support", async () => {
    serveIdentity([identityModel, plainModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-identity-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-identity-section']").exists()).toBe(true);

    await fieldControl("Model").setValue(plainModel.name);
    await flushPromises();

    // Absent, not disabled: a control for a capability this checkpoint does
    // not have would be a dead end.
    expect(wrapper.find("[data-test='mobile-identity-well']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-identity-section']").exists()).toBe(false);
  });

  it("parks a staged photo across a capability-losing model switch instead of blocking Develop", async () => {
    serveIdentity([identityModel, plainModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    expect(well().props("image")).toBe(PNG_1X1);

    await fieldControl("Model").setValue(plainModel.name);
    await fieldControl("Prompt").setValue("a parked identity print");
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-identity-well']").exists()).toBe(false);
    // Parking is not a refusal: the partition simply does not travel.
    expect(
      wrapper.get("[data-test='mobile-develop-button']").attributes("disabled"),
    ).toBeUndefined();

    await develop("a parked identity print");
    expect(admittedRequests()).toHaveLength(1);
    const parked = admittedRequests()[0]!;
    expect(parked.id_image).toBeUndefined();
    expect(parked.id_image_name).toBeUndefined();

    // Selecting a qualified checkpoint again brings the photo back untouched.
    await fieldControl("Model").setValue(identityModel.name);
    await flushPromises();
    expect(well().props("image")).toBe(PNG_1X1);
    expect(well().props("filename")).toBe("ada.png");
  });

  it("ships the four request fields, with the photo never fitted to the canvas", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await wrapper.get("[data-test='mobile-identity-weight']").setValue("0.75");
    await wrapper.get("[data-test='mobile-identity-start-step']").setValue("3");
    await flushPromises();

    await develop();

    const body = admittedRequests()[0]! as Record<string, unknown>;
    expect(body.id_image).toBe(PNG_1X1);
    expect(body.id_image_name).toBe("ada.png");
    expect(body.id_weight).toBe(0.75);
    expect(body.id_start_step).toBe(3);
    // A face reference is not a composition input: no source-fit provenance,
    // and the bytes are exactly what was picked.
    expect(body.source_fit).toBeUndefined();
    // The photo is kept under the digest of what shipped so Use as prompt can
    // look it back up; metadata records the digest, never the face.
    expect(persistGenerationSourceMedia).toHaveBeenCalledWith(
      PNG_1X1,
      expect.objectContaining({ filename: "ada.png" }),
    );
  });

  it("leaves the knobs absent until touched so the server's defaults apply", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();

    await develop();

    const body = admittedRequests()[0]! as Record<string, unknown>;
    expect(body.id_image).toBe(PNG_1X1);
    expect(body.id_weight).toBeUndefined();
    expect(body.id_start_step).toBeUndefined();
  });

  it("counts only the two knobs in the Advanced badge", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await flushPromises();

    // The photo well is primary-form media, exactly like the source wells.
    expect(wrapper.find("[data-test='mobile-advanced-trigger-count']").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-identity-weight']").setValue("1.4");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-advanced-trigger-count']").text()).toBe("1");

    await wrapper.get("[data-test='mobile-identity-start-step']").setValue("2");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-advanced-trigger-count']").text()).toBe("2");

    // Reset clears the knobs and keeps the attached face where the user put it.
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-advanced-trigger-count']").exists()).toBe(false);
    expect(well().props("image")).toBe(PNG_1X1);
  });

  it("refuses an unsupported photo inline and never stages it", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    well().vm.$emit("file", new File(["nope"], "face.gif", { type: "image/gif" }));
    await flushPromises();

    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain(
      "PNG or JPEG",
    );
    expect(well().props("image")).toBeNull();
  });

  it("names the refusal inline and blocks Develop when a knob has no photo", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-identity-weight']").setValue("1.4");
    await fieldControl("Prompt").setValue("a portrait with no face attached");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-identity-error']").text()).toContain(
      "Attach an identity photo",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes("disabled")).toBe("");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(0);
  });

  it("carries the identity partition onto every prepared Batch N sibling", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await wrapper.get("[data-test='mobile-identity-weight']").setValue("0.6");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two portrait studies");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    const editors = wrapper.findAll(".mobile-prepared-editor");
    expect(editors).toHaveLength(2);
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();

    expect(admittedRequests()).toHaveLength(2);
    for (const request of admittedRequests()) {
      expect(request.id_image).toBe(PNG_1X1);
      expect(request.id_weight).toBe(0.6);
    }
  });

  it("stales reviewed work when the identity photo changes, like any conditioning media", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    await fieldControl("Prompt").setValue("a portrait at dusk");
    await wrapper.get("[data-test='mobile-prompt-remix']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-remix-review']").text()).not.toContain(
      "Conditioning media changed",
    );

    await attachPhoto();

    expect(wrapper.get("[data-test='mobile-remix-review']").text()).toContain(
      "Conditioning media changed after this remix was prepared.",
    );
  });

  it("shows identity provenance in the Library viewer's Info sheet", async () => {
    const identityPrint: GalleryImage = {
      ...print,
      filename: "portrait.png",
      format: "png",
      metadata: {
        ...print.metadata,
        model: identityModel.name,
        id_image_name: "ada.png",
        id_image_sha256: "b".repeat(64),
        id_weight: 0.8,
        id_start_step: 2,
      },
    };
    serveIdentity([identityModel], [identityPrint]);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-info']").trigger("click");
    await flushPromises();

    const facts = wrapper.get("[data-test='gallery-viewer-identity']").text();
    expect(facts).toContain(`ada.png · ${"b".repeat(12)}`);
    expect(facts).toContain("0.8 · from step 2");
  });

  it("restores the knobs and re-attaches the photo on Use as prompt", async () => {
    const identityPrint: GalleryImage = {
      ...print,
      filename: "portrait.png",
      format: "png",
      metadata: {
        ...print.metadata,
        model: identityModel.name,
        id_image_name: "ada.png",
        id_image_sha256: "c".repeat(64),
        id_weight: 0.9,
        id_start_step: 1,
      },
    };
    serveIdentity([identityModel], [identityPrint]);
    restoreGenerationSourceMedia.mockImplementation((sha256: string | null | undefined) =>
      Promise.resolve(sha256 === "c".repeat(64) ? { base64: PNG_1X1, filename: "ada.png" } : null),
    );
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(well().props("image")).toBe(PNG_1X1);
    expect(well().props("filename")).toBe("ada.png");
    expect(
      (wrapper.get("[data-test='mobile-identity-weight']").element as HTMLInputElement).value,
    ).toBe("0.9");
    expect(
      (wrapper.get("[data-test='mobile-identity-start-step']").element as HTMLInputElement).value,
    ).toBe("1");
  });

  it("discloses a stash miss inline rather than rendering a different face", async () => {
    const identityPrint: GalleryImage = {
      ...print,
      filename: "portrait.png",
      format: "png",
      metadata: {
        ...print.metadata,
        model: identityModel.name,
        id_image_name: "ada.png",
        id_image_sha256: "d".repeat(64),
        id_weight: 0.9,
      },
    };
    serveIdentity([identityModel], [identityPrint]);
    restoreGenerationSourceMedia.mockResolvedValue(null);
    wrapper = mountMobileApp();
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    // Persistent inline copy, never a toast — and the well says the same thing
    // beside the control the user has to correct.
    expect(wrapper.get("[data-test='mobile-generation-summary']").text()).toContain(
      "not on this device",
    );
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain(
      "not on this device",
    );
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes("disabled")).toBe("");
  });

  it("blocks the prepared set's own Develop when the photo is cleared under a live knob", async () => {
    // The reviewed card has its own Develop, which never consults the
    // composer's blocker. Without the identity reason travelling with the
    // reviewed work, `buildRequest` would silently drop the partition and
    // every variation would render without the face.
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await wrapper.get("[data-test='mobile-identity-weight']").setValue("0.6");
    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    await fieldControl("Prompt").setValue("two portrait studies");
    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();
    const preparedDevelop = wrapper.get("[data-test='mobile-develop-prepared']");
    expect(preparedDevelop.attributes("disabled")).toBeUndefined();

    well().vm.$emit("clear");
    await flushPromises();

    const card = wrapper.get("[data-test='mobile-prepared-expansion']");
    expect(card.text()).toContain("Attach an identity photo");
    expect(wrapper.get("[data-test='mobile-develop-prepared']").attributes("disabled")).toBe("");
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(0);

    // Reattaching the same face makes the reviewed set submittable again.
    await attachPhoto();
    expect(
      wrapper.get("[data-test='mobile-develop-prepared']").attributes("disabled"),
    ).toBeUndefined();
    await wrapper.get("[data-test='mobile-develop-prepared']").trigger("click");
    await flushPromises();
    expect(admittedRequests()).toHaveLength(2);
    for (const request of admittedRequests()) {
      expect(request.id_image).toBe(PNG_1X1);
    }
  });

  it("never routes an identity print to a co-owner that does not advertise support", async () => {
    // The picker row is the deduplicated fleet union under Auto, so the
    // machine it came from is not necessarily the machine that runs the
    // print. Only the owners that advertise identity themselves may be asked
    // for a plan — even when the other one would answer sooner.
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
    apiJsonTo.mockImplementation((route: { baseUrl: string }, path: string, init?: RequestInit) => {
      const render = route.baseUrl === renderTarget.baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: render ? "render" : "studio",
          instance_id: render ? "render-id" : "studio-id",
        });
      }
      if (path === "/api/models") {
        // Both machines hold the model; only Studio links the adapter.
        return Promise.resolve([
          render ? { ...identityModel, supports_identity: false } : identityModel,
        ]);
      }
      if (path === "/api/capabilities") return Promise.resolve(durableQueueCapabilities);
      if (path === "/api/gallery") return Promise.resolve([]);
      if (path === "/api/activity") {
        return Promise.resolve({ instance_id: "mobile-host", observed_at_unix_ms: 1, items: [] });
      }
      return durableApiFallback(path, init, route);
    });
    // Render would win on speed alone.
    previewGenerationPlacement.mockImplementation((probe: { baseUrl: string }) => {
      const preview = plannedPlacement();
      preview.candidate.predicted_completion_after_ms =
        probe.baseUrl === renderTarget.baseUrl ? 10 : 9_000;
      return Promise.resolve(preview);
    });
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await develop("a routed portrait");

    // Render would win on speed, but only an owner that advertises identity
    // itself is a candidate at all — so the print freezes onto Studio.
    expect(admittedRequests()).toHaveLength(1);
    expect(admissionTargets()[0]).toEqual(target);
    expect(admittedRequests()[0]?.id_image).toBe(PNG_1X1);
  });

  it("refuses an oversized photo without ever reading it", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();

    const arrayBuffer = vi.fn(() => Promise.resolve(new ArrayBuffer(0)));
    const huge = {
      name: "huge.png",
      type: "image/png",
      size: 17 * 1024 * 1024,
      arrayBuffer,
    } as unknown as File;
    well().vm.$emit("file", huge);
    await flushPromises();

    expect(arrayBuffer).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain(
      "16 MiB or smaller",
    );
    expect(well().props("image")).toBeNull();
  });

  it("keeps a fractional start step and names the whole-number rule", async () => {
    serveIdentity([identityModel]);
    wrapper = mountMobileApp();
    await flushPromises();
    await attachPhoto();
    await fieldControl("Prompt").setValue("a portrait at 2.9 steps");
    await wrapper.get("[data-test='mobile-identity-start-step']").setValue("2.9");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-identity-error']").text()).toContain("whole number");
    expect(wrapper.get("[data-test='mobile-develop-button']").attributes("disabled")).toBe("");
    await wrapper.get("[data-test='mobile-develop-button']").trigger("click");
    await flushPromises();
    expect(openStreams).toHaveLength(0);
  });
});
