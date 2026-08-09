import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { createPinia, setActivePinia } from "pinia";
import LibraryPage from "./LibraryPage.vue";
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

const { listGalleryMock, deleteMock, fetchModelsMock } = vi.hoisted(() => ({
  listGalleryMock: vi.fn(),
  deleteMock: vi.fn(),
  fetchModelsMock: vi.fn(),
}));
const { hostDeleteMock, fetchBlobMock } = vi.hoisted(() => ({
  hostDeleteMock: vi.fn(),
  fetchBlobMock: vi.fn(),
}));
const { pushMock, replaceMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  replaceMock: vi.fn(),
}));

vi.mock("../api", () => ({
  listGallery: listGalleryMock,
  fetchModels: fetchModelsMock,
  deleteGalleryImage: deleteMock,
  imageUrl: (f: string) => `/api/gallery/image/${f}`,
  thumbnailUrl: (f: string) => `/api/gallery/thumbnail/${f}`,
}));

vi.mock("../components/machines/hostClient", () => ({
  hostDeleteGalleryImage: hostDeleteMock,
  hostGallery: vi.fn(),
}));

vi.mock("../lib/galleryMedia", () => ({
  fetchGalleryBlob: fetchBlobMock,
}));

// The gallery now merges every host; in tests the origin's list (listGalleryMock)
// is the single source. Entries default to the origin but may carry their own
// host tag so collisions across machines can be exercised.
vi.mock("../lib/multiHostGallery", async () => {
  const actual = await vi.importActual<
    typeof import("../lib/multiHostGallery")
  >("../lib/multiHostGallery");
  return {
    ...actual,
    fetchMergedGallery: async () => {
      const list = (await listGalleryMock()) as object[];
      return {
        entries: list.map((e) => ({
          hostId: "origin",
          hostLabel: "this server",
          ...e,
        })),
        rawEntries: list.map((e) => ({
          hostId: "origin",
          hostLabel: "this server",
          ...e,
        })),
        reachableHostIds: ["origin"],
        unreachableHostIds: [],
        remoteHostCount: 0,
      };
    },
  };
});

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
    thumbnailSize: Number,
    selectMode: Boolean,
    selection: { type: Object, default: undefined },
    fresh: { type: Object, default: undefined },
  },
  emits: ["open", "toggle-select", "drag-select", "context-menu"],
  template: `<div data-test="grid">
    <button data-test="grid-open" @click="$emit('open', entries[0])">open</button>
    <button data-test="grid-open-last" @click="$emit('open', entries[entries.length - 1])">open last</button>
    <button data-test="grid-context" @click="$emit('context-menu', { item: entries[0], x: 24, y: 32 })">context</button>
    <span data-test="grid-count">{{ entries.length }}</span>
    <span data-test="grid-keys">{{ entries.map((e) => e.hostId + '|' + e.filename).join(',') }}</span>
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
    <button data-test="lb-source" @click="$emit('use-source', item)">source</button>
    <span data-test="lb-key">{{ item.hostId }}|{{ item.filename }}</span>
  </div>`,
});

function mountPage() {
  return mount(LibraryPage, {
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

beforeEach(() => setActivePinia(createPinia()));

describe("LibraryPage", () => {
  beforeEach(() => {
    localStorage.clear();
    resetNotifications();
    formTesting.resetForTest();
    listGalleryMock.mockReset().mockResolvedValue([cat, dog]);
    fetchModelsMock.mockReset().mockResolvedValue([
      {
        name: "flux-dev:fp16",
        family: "flux",
        default_width: 1024,
        default_height: 1024,
        default_steps: 20,
        default_guidance: 3.5,
      },
      {
        name: "sdxl:fp16",
        family: "sdxl",
        default_width: 1024,
        default_height: 1024,
        default_steps: 30,
        default_guidance: 7.5,
      },
    ]);
    deleteMock.mockReset().mockResolvedValue(undefined);
    hostDeleteMock.mockReset().mockResolvedValue(undefined);
    fetchBlobMock.mockReset().mockResolvedValue(new Blob(["bytes"]));
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

  it("restores, applies, and persists the toolbar thumbnail-size slider", async () => {
    localStorage.setItem("mold.gallery.thumbnailSize.v1", "280");
    const wrapper = await mounted();
    const slider = wrapper.get<HTMLInputElement>(
      'input[aria-label="Thumbnail size"]',
    );

    expect(slider.element.value).toBe("280");
    expect(wrapper.getComponent(GalleryGridStub).props("thumbnailSize")).toBe(
      280,
    );

    await slider.setValue("320");
    expect(wrapper.getComponent(GalleryGridStub).props("thumbnailSize")).toBe(
      320,
    );
    expect(localStorage.getItem("mold.gallery.thumbnailSize.v1")).toBe("320");
  });

  it("filters prints by their owning host", async () => {
    listGalleryMock.mockResolvedValue([
      cat,
      { ...dog, hostId: "studio", hostLabel: "Studio" },
    ]);
    const wrapper = await mounted();
    const studio = wrapper
      .findAll("[data-test='gallery-host-filter']")
      .find((button) => button.text() === "Studio");
    expect(studio).toBeTruthy();
    await studio!.trigger("click");
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("1");
    expect(wrapper.get("[data-test='grid-keys']").text()).toContain(
      "studio|dog.png",
    );
  });

  it("opens a print action menu from a tile context click", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-context']").trigger("click");
    expect(wrapper.get("[data-test='gallery-context-menu']").text()).toContain(
      "Reuse settings",
    );
    expect(wrapper.get("[data-test='gallery-context-menu']").text()).toContain(
      "Use as source",
    );
  });

  it("appends Library source images to Ref2VA's dedicated ordered references", async () => {
    const form = useGenerateForm();
    form.state.value.model = "minimax-h3-ref2va:comfy-pruned-int8";
    form.state.value.modelFamily = "minimax-h3";
    const wrapper = await mounted();

    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-source']").trigger("click");
    await flushPromises();

    expect(form.state.value.imageAttachments).toEqual([]);
    expect(form.state.value.h3Authoring?.references).toHaveLength(1);
    expect(
      form.state.value.h3Authoring?.references[0]?.reference,
    ).toMatchObject({
      kind: "image",
      media: { authority: "inline", data: "Ynl0ZXM=" },
      provenance: { name: "dog.png" },
      mime_type: "image/png",
      width: 1024,
      height: 1024,
    });
    expect(pushMock).toHaveBeenCalledWith({ name: "create" });
  });

  it("refreshes the gallery while the page remains visible", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    expect(listGalleryMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listGalleryMock).toHaveBeenCalledTimes(2);
    wrapper.unmount();
  });

  it("does not start its refresh timer after unmounting during initial load", async () => {
    vi.useFakeTimers();
    let resolveGallery!: (entries: GalleryImage[]) => void;
    listGalleryMock.mockReturnValueOnce(
      new Promise((resolve) => (resolveGallery = resolve)),
    );
    const wrapper = mountPage();
    wrapper.unmount();
    resolveGallery([cat, dog]);
    await flushPromises();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(listGalleryMock).toHaveBeenCalledTimes(1);
  });

  it("does not overlap periodic gallery refreshes", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    let resolveRefresh!: (entries: GalleryImage[]) => void;
    listGalleryMock.mockReturnValueOnce(
      new Promise((resolve) => (resolveRefresh = resolve)),
    );
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listGalleryMock).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listGalleryMock).toHaveBeenCalledTimes(2);
    resolveRefresh([cat, dog]);
    await flushPromises();
    wrapper.unmount();
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
    await wrapper.find("[data-test='grid-open-last']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-reuse']").trigger("click");
    await flushPromises();

    expect(useGenerateForm().state.value.prompt).toBe("a wandering cat");
    expect(pushMock).toHaveBeenCalledWith({ name: "create" });
  });

  it("reuse restores the selected model's family defaults", async () => {
    const wrapper = await mounted();
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-reuse']").trigger("click");
    await flushPromises();

    expect(useGenerateForm().state.value.model).toBe("sdxl:fp16");
    expect(useGenerateForm().state.value.modelFamily).toBe("sdxl");
  });

  it("single delete removes optimistically and commits only after the window", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();

    await wrapper.find("[data-test='grid-open-last']").trigger("click");
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

    expect((grid.props("fresh") as Set<string>).has("origin|new.png")).toBe(
      true,
    );
  });
});

describe("LibraryPage multi-host identity", () => {
  const STUDIO = {
    id: "studio-7680",
    name: "studio",
    url: "http://studio:7680",
    apiKey: "studio-key",
  };
  // Same filename on two machines — mold names prints model+seed+timestamp,
  // so this is the common case once a second host is connected.
  const mine = {
    ...makeEntry("twin.png", "a twin print", "flux-dev:fp16", 5),
    hostId: "origin",
    hostLabel: "this server",
  };
  const theirs = {
    ...makeEntry("twin.png", "a twin print", "flux-dev:fp16", 6),
    hostId: "studio-7680",
    hostLabel: "studio",
  };

  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem("mold.web.hosts.v1", JSON.stringify([STUDIO]));
    resetNotifications();
    formTesting.resetForTest();
    listGalleryMock.mockReset().mockResolvedValue([theirs, mine]);
    deleteMock.mockReset().mockResolvedValue(undefined);
    hostDeleteMock.mockReset().mockResolvedValue(undefined);
    fetchBlobMock.mockReset().mockResolvedValue(new Blob(["bytes"]));
    pushMock.mockReset();
    replaceMock.mockReset();
    vi.mocked(requestConfirm).mockReset().mockResolvedValue(true);
  });

  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  it("renders one logical twin and deletes it from every host", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");

    // entries[0] is the studio copy.
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("No prints yet");

    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(hostDeleteMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680" }),
      "twin.png",
    );
    expect(deleteMock).toHaveBeenCalledWith("twin.png");
  });

  it("keeps each physical twin visible in its owning host filter", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();

    const studio = wrapper
      .findAll("[data-test='gallery-host-filter']")
      .find((button) => button.text() === "studio");
    expect(studio).toBeTruthy();
    await studio!.trigger("click");
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");
    expect(wrapper.find("[data-test='grid-keys']").text()).toBe(
      "studio-7680|twin.png",
    );

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("No prints yet");

    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(deleteMock).toHaveBeenCalledWith("twin.png");
    expect(hostDeleteMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680" }),
      "twin.png",
    );
  });

  it("deletes everywhere from a non-representative host-filter copy", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    const local = wrapper
      .findAll("[data-test='gallery-host-filter']")
      .find((button) => button.text() === "this server");
    await local!.trigger("click");
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("No prints yet");
    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(deleteMock).toHaveBeenCalledWith("twin.png");
    expect(hostDeleteMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680" }),
      "twin.png",
    );
  });

  it("uses representative groups when deleting a chained host-filter copy", async () => {
    vi.useFakeTimers();
    const archive = {
      id: "archive-7680",
      name: "archive",
      url: "http://archive:7680",
      apiKey: "archive-key",
    };
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([STUDIO, archive]),
    );
    const identified = (
      hostId: string,
      hostLabel: string,
      filename: string,
      timestamp: number,
    ) => ({
      ...makeEntry(filename, "shared", "flux-dev:q8", 42),
      hostId,
      hostLabel,
      timestamp,
      size_bytes: 4_096,
    });
    listGalleryMock.mockResolvedValue([
      identified("origin", "this server", "newest.png", 6_000),
      identified("studio-7680", "studio", "middle.png", 3_000),
      identified("archive-7680", "archive", "oldest.png", 0),
    ]);
    const wrapper = mountPage();
    await flushPromises();
    const studio = wrapper
      .findAll("[data-test='gallery-host-filter']")
      .find((button) => button.text() === "studio");
    await studio!.trigger("click");
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();
    vi.advanceTimersByTime(6_000);
    await flushPromises();

    expect(deleteMock).toHaveBeenCalledWith("newest.png");
    expect(hostDeleteMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680" }),
      "middle.png",
    );
    expect(hostDeleteMock).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "archive-7680" }),
      "oldest.png",
    );
    expect(wrapper.find("[data-test='grid-keys']").text()).toBe(
      "archive-7680|oldest.png",
    );
  });

  it("bulk delete routes each selected twin to its own host", async () => {
    const wrapper = mountPage();
    await flushPromises();
    await wrapper.find("[data-test='gallery-select']").trigger("click");
    await wrapper.vm.$nextTick();

    const deleteAll = wrapper
      .findAll(".gal__bar-danger")
      .find((b) => b.text() === "Delete all");
    await deleteAll!.trigger("click");
    await flushPromises();

    expect(deleteMock).toHaveBeenCalledTimes(1);
    expect(deleteMock).toHaveBeenCalledWith("twin.png");
    expect(hostDeleteMock).toHaveBeenCalledTimes(1);
    expect(hostDeleteMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680" }),
      "twin.png",
    );
    expect(wrapper.text()).toContain("No prints yet");
  });

  it("fetches Use as source from the host that owns the print", async () => {
    const wrapper = mountPage();
    await flushPromises();

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='lb-key']").text()).toBe(
      "studio-7680|twin.png",
    );

    await wrapper.find("[data-test='lb-source']").trigger("click");
    await flushPromises();

    expect(fetchBlobMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "studio-7680", apiKey: "studio-key" }),
      "twin.png",
    );
    expect(pushMock).toHaveBeenCalledWith({ name: "create" });
  });

  it("fetches a local print's source bytes from the origin", async () => {
    const wrapper = mountPage();
    await flushPromises();

    const local = wrapper
      .findAll("[data-test='gallery-host-filter']")
      .find((button) => button.text() === "this server");
    expect(local).toBeTruthy();
    await local!.trigger("click");
    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-source']").trigger("click");
    await flushPromises();

    expect(fetchBlobMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: "origin" }),
      "twin.png",
    );
  });

  it("never falls back to the origin for a print whose host is gone", async () => {
    // The host was forgotten between the merge and the action. Deleting or
    // sourcing against the origin would hit the same-named local twin.
    localStorage.removeItem("mold.web.hosts.v1");
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-source']").trigger("click");
    await flushPromises();
    expect(fetchBlobMock).not.toHaveBeenCalled();

    await wrapper.find("[data-test='grid-open']").trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.find("[data-test='lb-delete']").trigger("click");
    await flushPromises();
    vi.advanceTimersByTime(6000);
    await flushPromises();

    // Delete-everywhere still removes the reachable origin copy, but it must
    // not reroute the forgotten host's concrete copy to that origin.
    expect(deleteMock).toHaveBeenCalledWith("twin.png");
    expect(hostDeleteMock).not.toHaveBeenCalled();
    // Only the failed remote copy returns.
    expect(wrapper.find("[data-test='grid-count']").text()).toBe("1");
    expect(wrapper.find("[data-test='grid-keys']").text()).toBe(
      "studio-7680|twin.png",
    );
  });
});
