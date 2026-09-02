import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { createPinia, setActivePinia } from "pinia";
import LibraryPage from "./LibraryPage.vue";
import { __testing__ as formTesting } from "../composables/useGenerateForm";
import { clearSessionScrollForTests } from "@studio/lib/libraryOrganization";
import type { GalleryImage, OutputFormat } from "../types";

/** A 3-D print is its own kind of print, so the Library's kind filter has to
 * name it — it is neither an image nor a clip. */

const { listGalleryMock, fetchModelsMock } = vi.hoisted(() => ({
  listGalleryMock: vi.fn(),
  fetchModelsMock: vi.fn(),
}));

vi.mock("@studio/api/galleryOrganization", () => ({
  patchGalleryImage: vi.fn(async () => null),
  organizeGallery: vi.fn(async () => undefined),
  createCollection: vi.fn(async () => ({})),
  updateCollection: vi.fn(async () => ({})),
  deleteCollection: vi.fn(async () => undefined),
  setCollectionItems: vi.fn(async () => null),
  trashMany: vi.fn(async () => undefined),
  restoreTrashed: vi.fn(async () => undefined),
  deleteGalleryImageForever: vi.fn(async () => undefined),
  emptyTrash: vi.fn(async () => ({ purged: 0 })),
  listCollections: vi.fn(async () => []),
  listTags: vi.fn(async () => []),
}));

vi.mock("../api", () => ({
  listGallery: listGalleryMock,
  fetchModels: fetchModelsMock,
  deleteGalleryImage: vi.fn(async () => undefined),
  imageUrl: (f: string) => `/api/gallery/image/${f}`,
  thumbnailUrl: (f: string) => `/api/gallery/thumbnail/${f}`,
}));

vi.mock("../components/machines/hostClient", () => ({
  hostDeleteGalleryImage: vi.fn(async () => undefined),
  hostGallery: vi.fn(async () => []),
  hostCapabilities: vi.fn(async () => ({})),
  hostApiTarget: (host: { url: string; apiKey?: string }) => ({
    baseUrl: host.url,
    apiKey: host.apiKey ?? null,
  }),
}));

vi.mock("../lib/galleryMedia", () => ({ fetchGalleryBlob: vi.fn() }));

vi.mock("../lib/multiHostGallery", async () => {
  const actual = await vi.importActual<
    typeof import("../lib/multiHostGallery")
  >("../lib/multiHostGallery");
  return {
    ...actual,
    fetchMergedGallery: async () => {
      const list = (await listGalleryMock()) as object[];
      const tag = (e: object) => ({
        hostId: "origin",
        hostLabel: "this server",
        ...e,
      });
      return {
        entries: list.map(tag),
        rawEntries: list.map(tag),
        reachableHostIds: ["origin"],
        unreachableHostIds: [],
        remoteHostCount: 0,
      };
    },
  };
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

function makeEntry(filename: string, format: OutputFormat): GalleryImage {
  return {
    filename,
    timestamp: 1,
    format,
    metadata: {
      prompt: `prompt for ${filename}`,
      model: "flux-dev:fp16",
      seed: 1,
      steps: 20,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      version: "test",
    },
  };
}

const GalleryGridStub = defineComponent({
  name: "GalleryGrid",
  props: {
    entries: { type: Array, required: true },
    loading: Boolean,
    thumbnailSize: Number,
    selectMode: Boolean,
    selection: { type: Object, default: undefined },
    fresh: { type: Object, default: undefined },
  },
  emits: ["open", "toggle-select", "drag-select", "context-menu"],
  template: `<div data-test="grid"><span data-test="grid-names">{{ entries.map((e) => e.filename).join(',') }}</span></div>`,
});

async function mounted() {
  const wrapper = mount(LibraryPage, {
    global: {
      stubs: {
        GalleryGrid: GalleryGridStub,
        GalleryFeed: { template: "<div data-test='feed' />" },
        Lightbox: { template: "<div />" },
        Transition: false,
      },
    },
  });
  await flushPromises();
  return wrapper;
}

/** The kind filter's rendered segments, in render order. */
function kindSegments(wrapper: Awaited<ReturnType<typeof mounted>>) {
  return wrapper.get("[data-test='gallery-filter']").findAll("[role='radio']");
}

async function pickKind(
  wrapper: Awaited<ReturnType<typeof mounted>>,
  label: string,
) {
  const segment = kindSegments(wrapper).find(
    (candidate) => candidate.text() === label,
  );
  if (!segment) throw new Error(`no “${label}” segment on the kind filter`);
  await segment.trigger("click");
  await flushPromises();
}

beforeEach(() => {
  setActivePinia(createPinia());
  clearSessionScrollForTests();
  localStorage.clear();
  formTesting.resetForTest();
  listGalleryMock
    .mockReset()
    .mockResolvedValue([
      makeEntry("still.png", "png"),
      makeEntry("clip.mp4", "mp4"),
      makeEntry("chair.glb", "glb"),
    ]);
  fetchModelsMock.mockReset().mockResolvedValue([]);
});

describe("LibraryPage 3-D kind filter", () => {
  it("offers a 3D segment beside Images, Video and Audio", async () => {
    const wrapper = await mounted();
    expect(kindSegments(wrapper).map((segment) => segment.text())).toEqual([
      "All",
      "Images",
      "Video",
      "Audio",
      "3D",
    ]);
  });

  it("narrows the grid to mesh prints", async () => {
    const wrapper = await mounted();
    await pickKind(wrapper, "3D");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("chair.glb");
  });

  it("keeps a mesh print out of the Images bucket", async () => {
    const wrapper = await mounted();
    await pickKind(wrapper, "Images");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("still.png");
  });
});
