import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import GalleryPage from "./GalleryPage.vue";
import {
  requestConfirm,
  resetNotifications,
  runToastAction,
  useNotifications,
} from "../lib/toasts";
import {
  __testing__ as formTesting,
  useGenerateForm,
} from "../composables/useGenerateForm";
import type { GalleryImage } from "../types";

const { listGalleryMock, deleteMock } = vi.hoisted(() => ({
  listGalleryMock: vi.fn(),
  deleteMock: vi.fn(),
}));
const { pushMock, replaceMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  replaceMock: vi.fn(),
}));

vi.mock("../api", () => ({
  listGallery: listGalleryMock,
  deleteGalleryImage: deleteMock,
  imageUrl: (f: string) => `/api/gallery/image/${f}`,
  thumbnailUrl: (f: string) => `/api/gallery/thumbnail/${f}`,
}));

// The gallery now merges every host; in tests the origin's list (listGalleryMock)
// is the single source, tagged as the origin host.
vi.mock("../lib/multiHostGallery", () => ({
  fetchMergedGallery: async () => {
    const list = (await listGalleryMock()) as unknown[];
    return {
      entries: list.map((e) => ({
        ...(e as object),
        hostId: "origin",
        hostLabel: "this server",
      })),
      reachableHostIds: ["origin"],
      unreachableHostIds: [],
      remoteHostCount: 0,
    };
  },
}));

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ push: pushMock, replace: replaceMock }),
}));

// Keep the real toast store + undoable helpers; only auto-accept confirms.
vi.mock("../lib/toasts", async () => {
  const actual =
    await vi.importActual<typeof import("../lib/toasts")>("../lib/toasts");
  return { ...actual, requestConfirm: vi.fn(async () => true) };
});

function makeEntry(
  filename: string,
  prompt: string,
  model: string,
  seed: number,
): GalleryImage {
  return {
    filename,
    timestamp: seed,
    format: "png",
    metadata: {
      prompt,
      model,
      seed,
      steps: 20,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      version: "test",
    },
  };
}

const cat = makeEntry("cat.png", "a wandering cat", "flux-dev:fp16", 1);
const dog = makeEntry("dog.png", "a running dog", "sdxl:fp16", 2);

const GalleryGridStub = defineComponent({
  name: "GalleryGrid",
  props: {
    entries: { type: Array, required: true },
    loading: Boolean,
    selectMode: Boolean,
    selection: { type: Object, default: undefined },
    fresh: { type: Object, default: undefined },
  },
  emits: ["open", "toggle-select", "drag-select"],
  template: `<div data-test="grid">
    <button data-test="grid-open" @click="$emit('open', entries[0])">open</button>
    <span data-test="grid-count">{{ entries.length }}</span>
  </div>`,
});

const LightboxStub = defineComponent({
  name: "Lightbox",
  props: {
    item: { type: Object, default: null },
    index: Number,
    total: Number,
    hasPrev: Boolean,
    hasNext: Boolean,
    muted: Boolean,
  },
  emits: ["close", "prev", "next", "reuse", "use-source", "upscale", "delete"],
  template: `<div v-if="item" data-test="lightbox">
    <button data-test="lb-reuse" @click="$emit('reuse', item)">reuse</button>
    <button data-test="lb-delete" @click="$emit('delete', item)">delete</button>
  </div>`,
});

function mountPage() {
  return mount(GalleryPage, {
    global: {
      stubs: {
        GalleryGrid: GalleryGridStub,
        GalleryFeed: { template: "<div data-test='feed' />" },
        Lightbox: LightboxStub,
        Transition: false,
      },
    },
  });
}

async function mounted() {
  const wrapper = mountPage();
  await flushPromises();
  return wrapper;
}

describe("GalleryPage", () => {
  beforeEach(() => {
    localStorage.clear();
    resetNotifications();
    formTesting.resetForTest();
    listGalleryMock.mockReset().mockResolvedValue([cat, dog]);
    deleteMock.mockReset().mockResolvedValue(undefined);
    pushMock.mockReset();
    replaceMock.mockReset();
    vi.mocked(requestConfirm).mockReset().mockResolvedValue(true);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("says 'this server' — not 'all hosts' — when no remotes are connected", async () => {
    const wrapper = await mounted();
    const count = wrapper.get("[data-test='gallery-count']").text();
    expect(count).toContain("this server");
    expect(count).not.toContain("all hosts");
  });

  it("renders every print in the grid by default", async () => {
    const wrapper = await mounted();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("2");
  });

  it("narrows the grid as the user searches", async () => {
    const wrapper = await mounted();
    await wrapper.find("[data-test='gallery-search']").setValue("dog");
    await flushPromises();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");
  });

  it("shows the search-empty state with a clear action", async () => {
    const wrapper = await mounted();
    await wrapper.find("[data-test='gallery-search']").setValue("zzz-nope");
    await flushPromises();
    expect(wrapper.find("[data-test='grid']").exists()).toBe(false);
    expect(wrapper.text()).toContain("No prints match");

    await wrapper.find("[data-test='clear-search']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("2");
  });

  it("shows the video-empty prompt when filtering video with none present", async () => {
    const wrapper = await mounted();
    const videoSeg = wrapper
      .findAll(".ms-seg__btn")
      .find((b) => b.text() === "Video");
    expect(videoSeg).toBeTruthy();
    await videoSeg!.trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("No video clips yet");
    expect(wrapper.text()).toContain(
      "Generate with an LTX Video model to see clips here.",
    );
  });

  it("reuse writes the form and routes to Create", async () => {
    const wrapper = await mounted();
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-reuse']").trigger("click");
    await flushPromises();

    expect(useGenerateForm().state.value.prompt).toBe("a wandering cat");
    expect(pushMock).toHaveBeenCalledWith({ name: "create" });
  });

  it("single delete removes optimistically and commits only after the window", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();

    // Optimistic: gone from the list, but no DELETE yet.
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");
    expect(deleteMock).not.toHaveBeenCalled();

    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(deleteMock).toHaveBeenCalledWith("cat.png");
  });

  it("single delete can be undone before the window elapses", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");

    const toastId = useNotifications().toasts[0]!.id;
    runToastAction(toastId); // Undo
    await flushPromises();

    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(deleteMock).not.toHaveBeenCalled();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("2");
  });

  it("bulk delete-all still honors the typed confirm and deletes every print", async () => {
    const wrapper = await mounted();
    await wrapper.find("[data-test='gallery-select']").trigger("click");
    await wrapper.vm.$nextTick();

    const deleteAll = wrapper
      .findAll(".gal__bar-danger")
      .find((b) => b.text() === "Delete all");
    expect(deleteAll).toBeTruthy();
    await deleteAll!.trigger("click");
    await flushPromises();

    expect(vi.mocked(requestConfirm).mock.calls[0]![0]).toMatchObject({
      typedPhrase: "delete",
    });
    expect(deleteMock).toHaveBeenCalledWith("cat.png");
    expect(deleteMock).toHaveBeenCalledWith("dog.png");
  });

  it("badges prints that arrive on a later refresh as fresh", async () => {
    const wrapper = await mounted();
    const grid = wrapper.findComponent(GalleryGridStub);
    expect((grid.props("fresh") as Set<string>).size).toBe(0);

    const fresh = makeEntry("new.png", "a fresh print", "flux-dev:fp16", 3);
    listGalleryMock.mockResolvedValue([fresh, cat, dog]);
    await wrapper.find("[aria-label='Refresh gallery']").trigger("click");
    await flushPromises();

    expect((grid.props("fresh") as Set<string>).has("new.png")).toBe(true);
  });
});
