/**
 * A finished 3-D print on the Create canvas. A mesh completion carries binary
 * glTF in `image` and a rendered PNG in `mesh_poster`, so every raster arm
 * would draw glTF bytes as a picture — the mesh probe runs first, the viewer
 * gets an object URL (never a tens-of-megabytes data: URL in the DOM), the
 * poster is its still, and the caption is the shared mesh stats line.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    saveMediaBytes: vi.fn(),
    revealSavedMedia: vi.fn(),
  },
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));
vi.mock("../lib/api/history", () => ({ fetchHistory: vi.fn(() => Promise.resolve([])) }));
const applySourceFitPreprocess = vi.fn();
vi.mock("../lib/sourceFitPreprocess", () => ({
  applySourceFitPreprocess: (...args: unknown[]) => applySourceFitPreprocess(...args),
  applyH3BoundaryFit: vi.fn(),
}));

import GenerateView from "./GenerateView.vue";
import MeshViewer from "@studio/components/MeshViewer.vue";
import { useConnectionStore } from "../stores/connection";
import { useContextMenuStore } from "../stores/contextMenu";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { newJob, useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useGenerateFormStore } from "../stores/generateForm";
import { useUiStore } from "../stores/ui";
import { applyModelDefaults } from "../lib/generateForm";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { CompleteEvent, ModelEntry } from "../lib/api/types";

enableAutoUnmount(afterEach);

const meshModel: ModelEntry = {
  name: "hunyuan3d-mini-turbo:fp16",
  family: "hunyuan3d",
  downloaded: true,
  default_steps: 5,
  default_guidance: 5,
  source_image: "required",
} as ModelEntry;

/** The same checkpoint, carrying the advertised v1 recipe — the authority for
 * the zero canvas, the ignored prompt, and the mesh controls. */
const profiledMeshModel: ModelEntry = {
  ...meshModel,
  generation_profile: {
    schema_version: 1,
    profile_id: "hunyuan3d.mini",
    profile_hash: "hash",
    default_recipe_id: "default",
    recipes: [hunyuan3dRecipe()],
  },
} as unknown as ModelEntry;

/**
 * A REAL PNG header (signature + IHDR 1170 × 2532, base64), so the
 * source-resolution watcher actually decodes it and reaches the canvas write
 * it must not make. An arbitrary string decodes to nothing and lets the
 * watcher return early, which is how the canvas guard used to pass vacuously.
 */
const PNG_1170x2532 = "iVBORw0KGgoAAAANSUhEUgAABJIAAAnk";

function meshCompletion(overrides: Partial<CompleteEvent> = {}): CompleteEvent {
  return {
    image: "R0xURg==",
    format: "glb",
    width: 512,
    height: 512,
    seed_used: 7,
    generation_time_ms: 4_000,
    model: meshModel.name,
    filename: "mold-hunyuan3d.glb",
    mesh_vertices: 24_576,
    mesh_faces: 49_152,
    mesh_poster: "UE9TVEVS",
    mesh_bounds_min: [-0.5, -0.4, -0.3],
    mesh_bounds_max: [0.5, 0.4, 0.3],
    ...overrides,
  } as CompleteEvent;
}

function primeMeshJob(result: CompleteEvent = meshCompletion()) {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
  useModelStore().all = [profiledMeshModel];
  const job = newJob({
    prompt: "",
    model: meshModel.name,
    width: 0,
    height: 0,
    steps: 5,
  });
  job.clientId = 1;
  job.status = "complete";
  job.result = result;
  const generation = useGenerationStore();
  generation.jobs = [job];
  generation.selectedClientId = 1;
  return job;
}

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: { stubs: { SequenceComposer: true, ComposerCard: true } },
  });
}

let createObjectURL: ReturnType<typeof vi.fn>;
let revokeObjectURL: ReturnType<typeof vi.fn>;

beforeEach(() => {
  setActivePinia(createPinia());
  apiJson
    .mockReset()
    .mockImplementation((path: unknown) =>
      Promise.resolve(path === "/api/models" ? [profiledMeshModel] : []),
    );
  apiJsonTo.mockReset().mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/models") return Promise.resolve([profiledMeshModel]);
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    return Promise.resolve([]);
  });
  applySourceFitPreprocess
    .mockReset()
    .mockImplementation((input: { source: string; mask: string | null }) =>
      Promise.resolve({ source: input.source, mask: input.mask, changed: false }),
    );
  createObjectURL = vi.fn(() => "blob:mesh-1");
  revokeObjectURL = vi.fn();
  vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
  window.localStorage?.clear?.();
});
afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

describe("GenerateView — mesh results", () => {
  it("mounts the mesh viewer on the GLB with the poster as its still", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();

    const viewer = wrapper.findComponent(MeshViewer);
    expect(viewer.exists()).toBe(true);
    expect(viewer.props("src")).toBe("blob:mesh-1");
    expect(viewer.props("poster")).toBe("data:image/png;base64,UE9TVEVS");
    expect(viewer.props("autoRotate")).toBe(true);
    expect(viewer.props("expandable")).toBe(true);
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect((createObjectURL.mock.calls[0]?.[0] as Blob).type).toBe("model/gltf-binary");
  });

  it("never draws glTF bytes through the still, video or audio arms", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();
    const frame = wrapper.get("[data-test='preview-frame']");
    expect(frame.find("[data-test='preview-audio']").exists()).toBe(false);
    expect(frame.find("video").exists()).toBe(false);
    expect(frame.find("img").exists()).toBe(false);
  });

  it("frames the canvas on the poster instead of the recipe's zero canvas", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();
    // `0 / 0` is invalid CSS: both frame rules would be dropped and the
    // viewer, which is absolutely positioned, would have no box to fill.
    const style = wrapper.get("[data-test='preview-frame']").attributes("style") ?? "";
    expect(style).toContain("aspect-ratio: 512 / 512");
    expect(style).not.toContain("NaN");
  });

  it("captions the print with the shared mesh stats line", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();
    const caption = wrapper.get("[data-test='generation-edge-code']").text();
    expect(caption).toContain("49,152 tris · 24,576 verts · 1.00×0.80×0.60");
    // The poster's pixel size is not the print's size — it must not read as one.
    expect(caption).not.toContain("512×512");
  });

  it("refuses Copy image and Use as source for a mesh print", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();
    await wrapper.get("[data-test='preview-frame']").trigger("contextmenu");
    const entries = useContextMenuStore().entries.filter((entry) => !("separator" in entry));
    const labelled = (label: string) =>
      entries.find((entry) => !("separator" in entry) && entry.label === label);
    expect(labelled("Copy image")).toMatchObject({ disabled: true });
    expect(labelled("Use as source")).toMatchObject({ disabled: true });
    // The save entry names what it saves: binary glTF, not an image.
    expect(labelled("Save mesh")).toMatchObject({ disabled: false });
    expect(labelled("Save image")).toBeUndefined();
  });

  it("releases the object URL when the canvas moves on", async () => {
    primeMeshJob();
    const wrapper = mountView();
    await flushPromises();
    wrapper.unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mesh-1");
  });

  it("does not block Generate on the zero canvas a mesh recipe advertises", async () => {
    primeMeshJob();
    useHostModelsStore().byHost.local = {
      entries: [profiledMeshModel],
      fetchedAt: Date.now(),
      error: null,
    };
    const form = useGenerateFormStore().form;
    form.model = profiledMeshModel.name;
    form.family = "hunyuan3d";
    applyModelDefaults(form, profiledMeshModel);
    form.prompt = "";
    const wrapper = mountView();
    await flushPromises();
    // A decodable source runs the resolution watcher for real; the canvasless
    // recipe's zero canvas must survive it, or Generate is blocked on a size
    // the request never reads.
    form.sourceImage = PNG_1170x2532;
    form.sourceImageName = "armchair.png";
    await flushPromises();
    expect(form.sourceImageWidth).toBe(1170);
    expect(form.width).toBe(0);
    expect(form.height).toBe(0);
    const composer = wrapper.findComponent({ name: "ComposerCard" });
    expect(composer.props("disabledReason")).toBeNull();
    expect(composer.props("disabled")).toBe(false);
  });

  it("submits the source at its native size instead of fitting it to a 0 × 0 canvas", async () => {
    primeMeshJob();
    useHostModelsStore().byHost.local = {
      entries: [profiledMeshModel],
      fetchedAt: Date.now(),
      error: null,
    };
    const submitBatch = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });
    mountView();
    await flushPromises();
    const form = useGenerateFormStore().form;
    form.model = profiledMeshModel.name;
    form.family = "hunyuan3d";
    applyModelDefaults(form, profiledMeshModel);
    form.sourceImage = PNG_1170x2532;
    form.sourceImageName = "armchair.png";
    form.prompt = "";
    await flushPromises();

    useUiStore().generateTick++;
    await flushPromises();
    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
    expect(submitBatch).toHaveBeenCalledTimes(1);
    expect(submitBatch.mock.calls[0]![0]).toMatchObject({
      source_image: PNG_1170x2532,
      width: 0,
      height: 0,
    });
  });

  it("leaves a raster print on the ordinary still arm", async () => {
    primeMeshJob({
      image: "aW1hZ2U=",
      format: "png",
      width: 1024,
      height: 1024,
      seed_used: 3,
      generation_time_ms: 1_000,
      model: "sdxl-base:fp16",
      filename: "mold-sdxl.png",
    } as CompleteEvent);
    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.findComponent(MeshViewer).exists()).toBe(false);
    expect(createObjectURL).not.toHaveBeenCalled();
  });
});
