/**
 * Library organization (V3 "Shelf"): the Prints | Collections | Trash scope
 * control and its URL sync, the filter-chip row, the collections shelf and
 * drill-in, trash restore / delete-forever / empty-trash with PLAIN confirms
 * (never a typed phrase), the bulk bar's organization actions, and the
 * capability gate that hides all of it on an older host.
 */
import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { createPinia, setActivePinia } from "pinia";
import LibraryPage from "./LibraryPage.vue";
import {
  requestConfirm,
  requestText,
  resetNotifications,
  runToastAction,
  useNotifications,
} from "../lib/toasts";
import { __testing__ as formTesting } from "../composables/useGenerateForm";
import type { Collection, GalleryImage } from "../types";

const {
  listGalleryMock,
  deleteMock,
  fetchModelsMock,
  hostDeleteMock,
  hostCapabilitiesMock,
  hostGalleryMock,
} = vi.hoisted(() => ({
  listGalleryMock: vi.fn(),
  deleteMock: vi.fn(),
  fetchModelsMock: vi.fn(),
  hostDeleteMock: vi.fn(),
  hostCapabilitiesMock: vi.fn(),
  hostGalleryMock: vi.fn(),
}));
const { pushMock, replaceMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  replaceMock: vi.fn(),
}));
const orgApi = vi.hoisted(() => ({
  patchGalleryImage: vi.fn(async () => null),
  organizeGallery: vi.fn(async () => undefined),
  createCollection: vi.fn(async (_t: unknown, body: { name: string }) => ({
    id: `id-${body.name}`,
    name: body.name,
    slug: body.name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    description: null,
    cover_filename: null,
    count: 0,
    created_at: 0,
    updated_at: 0,
  })),
  updateCollection: vi.fn(async () => ({})),
  updateCollectionHidden: vi.fn(async (_target: unknown, id: string, hidden: boolean) => ({
    id,
    hidden,
  })),
  deleteCollection: vi.fn(async () => undefined),
  setCollectionItems: vi.fn(async () => null),
  trashMany: vi.fn(async () => undefined),
  restoreTrashed: vi.fn(async () => undefined),
  deleteGalleryImageForever: vi.fn(async () => undefined),
  emptyTrash: vi.fn(async () => ({ purged: 1 })),
  listCollections: vi.fn(async () => [] as Collection[]),
  listTags: vi.fn(async () => [] as { name: string; count: number }[]),
}));
vi.mock("@studio/api/galleryOrganization", () => orgApi);

vi.mock("../api", () => ({
  listGallery: listGalleryMock,
  fetchModels: fetchModelsMock,
  deleteGalleryImage: deleteMock,
  imageUrl: (f: string) => `/api/gallery/image/${f}`,
  thumbnailUrl: (f: string) => `/api/gallery/thumbnail/${f}`,
}));
vi.mock("../components/machines/hostClient", () => ({
  hostDeleteGalleryImage: hostDeleteMock,
  hostGallery: hostGalleryMock,
  hostCapabilities: hostCapabilitiesMock,
  hostApiTarget: (host: { url: string; apiKey?: string }) => ({
    baseUrl: host.url,
    apiKey: host.apiKey ?? null,
  }),
}));
vi.mock("../lib/galleryMedia", () => ({
  fetchGalleryBlob: vi.fn(),
  resolveThumbnailSrc: vi.fn(async () => ""),
}));
vi.mock("../lib/multiHostGallery", async () => {
  const actual = await vi.importActual<
    typeof import("../lib/multiHostGallery")
  >("../lib/multiHostGallery");
  return {
    ...actual,
    fetchMergedGallery: async () => {
      const list = (await listGalleryMock()) as object[];
      const raw = list.map((e) => ({
        hostId: "origin",
        hostLabel: "this server",
        ...e,
      }));
      return {
        entries: raw,
        rawEntries: raw,
        reachableHostIds: ["origin"],
        unreachableHostIds: [],
        remoteHostCount: 0,
      };
    },
  };
});
const routeQuery = vi.hoisted(() => ({ value: {} as Record<string, unknown> }));
vi.mock("vue-router", () => ({
  useRoute: () =>
    new Proxy(
      {},
      { get: (_t, key) => (key === "query" ? routeQuery.value : undefined) },
    ),
  useRouter: () => ({ push: pushMock, replace: replaceMock }),
}));
vi.mock("../lib/toasts", async () => {
  const actual =
    await vi.importActual<typeof import("../lib/toasts")>("../lib/toasts");
  return {
    ...actual,
    requestConfirm: vi.fn(async () => true),
    requestText: vi.fn(async () => null),
  };
});

const ORIGIN_TARGET = { baseUrl: window.location.origin, apiKey: null };

function makeEntry(
  filename: string,
  seed: number,
  extra: Partial<GalleryImage> = {},
): GalleryImage {
  return {
    filename,
    timestamp: seed,
    format: "png",
    metadata: {
      prompt: `prompt ${filename}`,
      model: "flux-dev:fp16",
      seed,
      steps: 20,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      version: "test",
    },
    ...extra,
  };
}

function collection(id: string, name: string, extra: Partial<Collection> = {}) {
  return {
    id,
    name,
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    description: null,
    cover_filename: null,
    count: 1,
    created_at: 1,
    updated_at: 1_700_000_000,
    ...extra,
  } satisfies Collection;
}

// Newest first in the grid: smurf, frog, plain.
const smurf = makeEntry("smurf.png", 3, {
  title: "Smurf 04",
  favorite: true,
  tags: ["smurf", "blue"],
  collections: ["c-smurfs"],
});
const frog = makeEntry("frog.png", 2, { tags: ["green"] });
const plain = makeEntry("plain.png", 1);
const trashedOld = makeEntry("old.png", 4, {
  trashed_at: 1_700_000_000,
  purge_at: Math.floor(Date.now() / 1000) + 3 * 86_400,
});

const GalleryGridStub = defineComponent({
  name: "GalleryGrid",
  props: {
    entries: { type: Array, required: true },
    loading: Boolean,
    thumbnailSize: Number,
    selectMode: Boolean,
    selection: { type: Object, default: undefined },
    fresh: { type: Object, default: undefined },
    trash: Boolean,
  },
  emits: [
    "open",
    "toggle-select",
    "drag-select",
    "context-menu",
    "restore",
    "delete-forever",
  ],
  template: `<div data-test="grid" :data-trash="trash ? 'true' : 'false'">
    <button data-test="grid-open" @click="$emit('open', entries[0])">open</button>
    <button data-test="grid-context" @click="$emit('context-menu', { item: entries[0], x: 24, y: 32 })">context</button>
    <button data-test="grid-select-first" @click="$emit('toggle-select', { item: entries[0], shift: false, meta: false })">sel</button>
    <button data-test="grid-select-second" @click="$emit('toggle-select', { item: entries[1], shift: false, meta: false })">sel2</button>
    <button data-test="grid-restore" @click="$emit('restore', entries[0])">restore</button>
    <button data-test="grid-delete-forever" @click="$emit('delete-forever', entries[0])">forever</button>
    <span data-test="grid-count">{{ entries.length }}</span>
    <span data-test="grid-names">{{ entries.map((e) => e.filename).join(',') }}</span>
  </div>`,
});

const LightboxStub = defineComponent({
  name: "Lightbox",
  props: {
    item: { type: Object, default: null },
    canOrganize: Boolean,
    canTrash: Boolean,
    inTrash: Boolean,
    collections: { type: Array, default: () => [] },
    tagSuggestions: { type: Array, default: () => [] },
  },
  emits: [
    "close",
    "reuse",
    "delete",
    "rename",
    "favorite",
    "add-tag",
    "remove-tag",
    "set-collection",
    "new-collection",
    "restore",
    "delete-forever",
  ],
  template: `<div v-if="item" data-test="lightbox" :data-can-organize="canOrganize" :data-can-trash="canTrash" :data-in-trash="inTrash">
    <span data-test="lb-title">{{ item.title ?? '' }}</span>
    <span data-test="lb-fav">{{ item.favorite ? 'fav' : 'plain' }}</span>
    <span data-test="lb-tags">{{ (item.tags ?? []).join(',') }}</span>
    <span data-test="lb-collections">{{ collections.map((c) => c.slug + ':' + (c.checked ? 'on' : 'off')).join(',') }}</span>
    <button data-test="lb-rename" @click="$emit('rename', item, 'Renamed')">rename</button>
    <button data-test="lb-favorite" @click="$emit('favorite', item, !item.favorite)">fav</button>
    <button data-test="lb-add-tag" @click="$emit('add-tag', item, 'keep')">tag</button>
    <button data-test="lb-set-collection" @click="$emit('set-collection', item, 'rivers', true)">coll</button>
    <button data-test="lb-delete" @click="$emit('delete', item)">delete</button>
    <button data-test="lb-restore" @click="$emit('restore', item)">restore</button>
    <button data-test="lb-delete-forever" @click="$emit('delete-forever', item)">forever</button>
  </div>`,
});

const mountedPages: Array<{ unmount: () => void }> = [];
function mountPage() {
  const wrapper = mount(LibraryPage, {
    global: {
      stubs: {
        GalleryGrid: GalleryGridStub,
        GalleryFeed: { template: "<div data-test='feed' />" },
        Lightbox: LightboxStub,
        Transition: false,
        RouterLink: { template: "<a><slot /></a>" },
      },
    },
  });
  mountedPages.push(wrapper);
  return wrapper;
}

async function mounted() {
  const wrapper = mountPage();
  await flushPromises();
  return wrapper;
}

const ORGANIZING = {
  gallery: {
    can_delete: true,
    organize: true,
    trash: { enabled: true, retention_days: 30 },
  },
};

beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  routeQuery.value = {};
  resetNotifications();
  formTesting.resetForTest();
  listGalleryMock.mockReset().mockResolvedValue([smurf, frog, plain]);
  fetchModelsMock.mockReset().mockResolvedValue([]);
  deleteMock.mockReset().mockResolvedValue(undefined);
  hostDeleteMock.mockReset().mockResolvedValue(undefined);
  hostCapabilitiesMock.mockReset().mockResolvedValue(ORGANIZING);
  hostGalleryMock.mockReset().mockResolvedValue([trashedOld]);
  for (const fn of Object.values(orgApi)) fn.mockClear();
  orgApi.listCollections.mockResolvedValue([
    collection("c-smurfs", "Smurfs", { cover_filename: "smurf.png" }),
    collection("c-rivers", "Rivers", { count: 0 }),
  ]);
  orgApi.listTags.mockResolvedValue([
    { name: "smurf", count: 1 },
    { name: "blue", count: 1 },
    { name: "green", count: 1 },
  ]);
  pushMock.mockReset();
  replaceMock.mockReset();
  vi.mocked(requestConfirm).mockReset().mockResolvedValue(true);
  vi.mocked(requestText).mockReset().mockResolvedValue(null);
});

afterEach(() => {
  // Pages register document-level keyboard listeners; a page left mounted
  // would answer the next test's shortcuts too.
  for (const page of mountedPages.splice(0)) page.unmount();
  vi.useRealTimers();
});

describe("Library scopes", () => {
  it("renders Prints | Collections | Trash with counts and syncs the scope to the URL", async () => {
    const wrapper = await mounted();
    const scope = wrapper.get("[data-test='library-scope']");
    const labels = scope.findAll(".ms-seg__btn").map((b) => b.text());
    expect(labels).toEqual(["Prints · 3", "Collections · 2", "Trash · 1"]);
    expect(wrapper.get("[data-test='grid']").attributes("data-trash")).toBe(
      "false",
    );

    await scope.findAll(".ms-seg__btn")[2]!.trigger("click");
    await flushPromises();
    expect(replaceMock).toHaveBeenLastCalledWith({
      query: { scope: "trash" },
    });
    expect(wrapper.get("[data-test='grid']").attributes("data-trash")).toBe(
      "true",
    );
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("old.png");
    expect(wrapper.get("[data-test='gallery-count']").text()).toContain(
      "1 print in trash",
    );

    await scope.findAll(".ms-seg__btn")[0]!.trigger("click");
    expect(replaceMock).toHaveBeenLastCalledWith({ query: {} });
  });

  it("enters a scope from the URL", async () => {
    routeQuery.value = { scope: "collections", c: "smurfs" };
    const wrapper = await mounted();
    expect(wrapper.get("[data-test='crumb-here']").text()).toBe("Smurfs");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("smurf.png");
  });

  it("hides the scope control and organization chrome on a host without organize or trash", async () => {
    hostCapabilitiesMock.mockResolvedValue({ gallery: { can_delete: true } });
    hostGalleryMock.mockResolvedValue([]);
    const wrapper = await mounted();
    expect(wrapper.find("[data-test='library-scope']").exists()).toBe(false);
    expect(wrapper.find("[data-test='chip-favorites']").exists()).toBe(false);
    expect(wrapper.find("[data-test='library-chip-row']").exists()).toBe(false);
    // Legacy wording survives for the destructive path.
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    expect(wrapper.get("[data-test='bulk-delete']").text()).toBe(
      "Delete selected",
    );
    expect(wrapper.find("[data-test='bulk-favorite']").exists()).toBe(false);
    await wrapper.get("[data-test='grid-open']").trigger("click");
    expect(
      wrapper.get("[data-test='lightbox']").attributes("data-can-organize"),
    ).toBe("false");
  });
});

describe("Library chip row", () => {
  it("filters by favorites and tags with URL sync, and searches titles and tags", async () => {
    const wrapper = await mounted();
    expect(wrapper.get("[data-test='chip-favorites']").text()).toContain("1");
    await wrapper.get("[data-test='chip-favorites']").trigger("click");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("smurf.png");
    expect(replaceMock).toHaveBeenLastCalledWith({ query: { fav: "1" } });
    await wrapper.get("[data-test='chip-favorites']").trigger("click");

    const green = wrapper
      .findAll("[data-test='chip-tag']")
      .find((chip) => chip.text().startsWith("green"));
    await green!.trigger("click");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("frog.png");
    expect(replaceMock).toHaveBeenLastCalledWith({ query: { tag: "green" } });
    await green!.trigger("click");

    await wrapper.get("[data-test='gallery-search']").setValue("smurf 04");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("smurf.png");
    await wrapper.get("[data-test='gallery-search']").setValue("green");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("frog.png");
  });

  it("offers a clear-filters empty state when chips exclude everything", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='chip-favorites']").trigger("click");
    const green = wrapper
      .findAll("[data-test='chip-tag']")
      .find((chip) => chip.text().startsWith("green"));
    await green!.trigger("click");
    expect(wrapper.text()).toContain("No prints match");
    await wrapper.get("[data-test='clear-filters']").trigger("click");
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("3");
  });
});

describe("Collections scope", () => {
  it("shows the shelf, opens a drill-in with a breadcrumb, and creates a collection on the primary", async () => {
    const wrapper = await mounted();
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[1]!
      .trigger("click");
    await flushPromises();
    const cards = wrapper.findAll("[data-test='collection-card']");
    expect(
      cards.map((c) => c.get("[data-test='collection-name']").text()),
    ).toEqual(["Rivers", "Smurfs"]);
    expect(cards[1]!.get("[data-test='collection-meta']").text()).toBe(
      "1 print · this server",
    );

    await cards[1]!.get("[data-test='collection-open']").trigger("click");
    expect(replaceMock).toHaveBeenLastCalledWith({
      query: { scope: "collections", c: "smurfs" },
    });
    expect(wrapper.get("[data-test='crumb-here']").text()).toBe("Smurfs");
    expect(wrapper.get("[data-test='grid-names']").text()).toBe("smurf.png");
    await wrapper.get("[data-test='crumb-collections']").trigger("click");
    expect(wrapper.findAll("[data-test='collection-card']")).toHaveLength(2);

    vi.mocked(requestText).mockResolvedValueOnce("Film grain");
    await wrapper.get("[data-test='new-collection']").trigger("click");
    await flushPromises();
    expect(orgApi.createCollection).toHaveBeenCalledWith(ORIGIN_TARGET, {
      name: "Film grain",
    });
  });

  it("hides, renames, and deletes a collection with a plain confirm that names it", async () => {
    const wrapper = await mounted();
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[1]!
      .trigger("click");
    await flushPromises();
    const card = wrapper
      .findAll("[data-test='collection-card']")
      .find((c) => c.attributes("data-slug") === "smurfs")!;
    await card.get("[data-test='collection-menu']").trigger("click");
    vi.mocked(requestText).mockResolvedValueOnce("Blue folk");
    await card.get("[data-test='collection-rename']").trigger("click");
    await flushPromises();
    expect(orgApi.updateCollection).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "c-smurfs",
      { name: "Blue folk" },
    );

    await card.get("[data-test='collection-menu']").trigger("click");
    await card.get("[data-test='collection-hidden']").trigger("click");
    await flushPromises();
    expect(orgApi.updateCollectionHidden).toHaveBeenCalledWith(ORIGIN_TARGET, "c-smurfs", true);

    await card.get("[data-test='collection-menu']").trigger("click");
    await card.get("[data-test='collection-delete']").trigger("click");
    await flushPromises();
    const options = vi.mocked(requestConfirm).mock.calls[0]![0];
    expect(options.title).toBe("Delete collection “Smurfs”?");
    expect(options.body).toBe("Its prints stay in the Library.");
    expect(options.typedPhrase).toBeUndefined();
    expect(orgApi.deleteCollection).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "c-smurfs",
    );
  });

  it("shows an inviting empty shelf with one action", async () => {
    orgApi.listCollections.mockResolvedValue([]);
    const wrapper = await mounted();
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[1]!
      .trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("No collections yet");
    expect(wrapper.find("[data-test='empty-new-collection']").exists()).toBe(
      true,
    );
  });
});

describe("Lightbox edits fan out", () => {
  it("renames, favorites, tags, and collects the open print through the studio helpers", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    const lb = wrapper.get("[data-test='lightbox']");
    expect(lb.attributes("data-can-organize")).toBe("true");
    expect(lb.attributes("data-can-trash")).toBe("true");
    expect(wrapper.get("[data-test='lb-collections']").text()).toBe(
      "rivers:off,smurfs:on",
    );

    await wrapper.get("[data-test='lb-rename']").trigger("click");
    expect(orgApi.patchGalleryImage).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "smurf.png",
      { title: "Renamed" },
    );
    // Optimistic: the viewer already shows the new title.
    expect(wrapper.get("[data-test='lb-title']").text()).toBe("Renamed");

    await wrapper.get("[data-test='lb-favorite']").trigger("click");
    expect(orgApi.organizeGallery).toHaveBeenCalledWith(ORIGIN_TARGET, {
      filenames: ["smurf.png"],
      favorite: false,
    });
    expect(wrapper.get("[data-test='lb-fav']").text()).toBe("plain");

    await wrapper.get("[data-test='lb-add-tag']").trigger("click");
    expect(orgApi.organizeGallery).toHaveBeenLastCalledWith(ORIGIN_TARGET, {
      filenames: ["smurf.png"],
      add_tags: ["keep"],
    });

    await wrapper.get("[data-test='lb-set-collection']").trigger("click");
    await flushPromises();
    expect(orgApi.setCollectionItems).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "c-rivers",
      { add: ["smurf.png"], remove: [] },
    );
  });
});

describe("Trash scope", () => {
  async function inTrash() {
    const wrapper = await mounted();
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[2]!
      .trigger("click");
    await flushPromises();
    return wrapper;
  }

  it("shows the retention banner with mono numbers", async () => {
    const wrapper = await inTrash();
    const banner = wrapper.get("[data-test='trash-banner']");
    expect(banner.text()).toContain("Prints stay in the trash");
    expect(banner.find(".gal__mono").text()).toBe("30 d");
    expect(banner.text()).toContain("Change retention");
  });

  it("restores a print and refuses nothing silently", async () => {
    const wrapper = await inTrash();
    await wrapper.get("[data-test='grid-restore']").trigger("click");
    await flushPromises();
    expect(orgApi.restoreTrashed).toHaveBeenCalledWith(ORIGIN_TARGET, [
      "old.png",
    ]);
    expect(useNotifications().toasts.some((t) => /Restored/.test(t.text))).toBe(
      true,
    );
  });

  it("deletes forever behind a plain danger confirm", async () => {
    const wrapper = await inTrash();
    await wrapper.get("[data-test='grid-delete-forever']").trigger("click");
    await flushPromises();
    const options = vi.mocked(requestConfirm).mock.calls[0]![0];
    expect(options).toMatchObject({
      title: "Delete print forever?",
      confirmLabel: "Delete forever",
      danger: true,
    });
    expect(options.typedPhrase).toBeUndefined();
    expect(orgApi.deleteGalleryImageForever).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "old.png",
    );
    expect(wrapper.text()).toContain("No prints in the trash");
  });

  it("empties the trash with a plain confirm naming the count and hosts", async () => {
    const wrapper = await inTrash();
    await wrapper.get("[data-test='empty-trash']").trigger("click");
    await flushPromises();
    const options = vi.mocked(requestConfirm).mock.calls[0]![0];
    expect(options.title).toBe("Empty trash?");
    expect(options.body).toBe(
      "Delete 1 print in the trash on this server forever? This can't be undone.",
    );
    expect(options.typedPhrase).toBeUndefined();
    expect(orgApi.emptyTrash).toHaveBeenCalledWith(ORIGIN_TARGET);
  });

  it("the lightbox in the trash gets Restore / Delete forever", async () => {
    const wrapper = await inTrash();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    expect(
      wrapper.get("[data-test='lightbox']").attributes("data-in-trash"),
    ).toBe("true");
    await wrapper.get("[data-test='lb-restore']").trigger("click");
    await flushPromises();
    expect(orgApi.restoreTrashed).toHaveBeenCalled();
  });
});

describe("Trash-aware deletes in Prints", () => {
  it("single delete on a trash host skips the confirm, offers undo, then lands in Trash", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-delete']").trigger("click");
    await flushPromises();
    expect(requestConfirm).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("2");
    expect(useNotifications().toasts[0]!.text).toBe("Moved to trash");
    expect(deleteMock).not.toHaveBeenCalled();

    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(deleteMock).toHaveBeenCalledWith("smurf.png");
    // Lands in the Trash scope without waiting for the next poll.
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[2]!
      .trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='grid-names']").text()).toBe(
      "old.png,smurf.png",
    );
  });

  it("bulk Trash is undoable and commits through the bulk trash route", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='grid-select-second']").trigger("click");
    expect(wrapper.get("[data-test='bulk-delete']").text()).toBe("Trash");
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    await flushPromises();
    expect(requestConfirm).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("1");

    const toastId = useNotifications().toasts[0]!.id;
    runToastAction(toastId); // Undo
    await flushPromises();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("3");
    expect(orgApi.trashMany).not.toHaveBeenCalled();

    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    vi.advanceTimersByTime(6000);
    await flushPromises();
    expect(orgApi.trashMany).toHaveBeenCalledWith(ORIGIN_TARGET, ["smurf.png"]);
  });

  it("Trash all asks a plain confirm and never a typed phrase", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    expect(wrapper.get("[data-test='bulk-delete-all']").text()).toBe(
      "Trash all",
    );
    await wrapper.get("[data-test='bulk-delete-all']").trigger("click");
    await flushPromises();
    const options = vi.mocked(requestConfirm).mock.calls[0]![0];
    expect(options.title).toBe("Move all 3 prints to the trash?");
    expect(options.typedPhrase).toBeUndefined();
    expect(deleteMock).toHaveBeenCalledTimes(3);
  });
});

describe("Bulk organization bar", () => {
  it("favorites and tags the selection and adds it to a collection", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='grid-select-second']").trigger("click");

    // Mixed selection (one favorite, one not) → Favorite sets all.
    expect(wrapper.get("[data-test='bulk-favorite']").text()).toBe("Favorite");
    await wrapper.get("[data-test='bulk-favorite']").trigger("click");
    expect(orgApi.organizeGallery).toHaveBeenCalledWith(ORIGIN_TARGET, {
      filenames: ["smurf.png", "frog.png"],
      favorite: true,
    });
    await flushPromises();
    expect(wrapper.get("[data-test='bulk-favorite']").text()).toBe(
      "Unfavorite",
    );

    // The bulk popovers teleport to <body>.
    await wrapper.get("[data-test='bulk-tags']").trigger("click");
    await flushPromises();
    const input = document.querySelector<HTMLInputElement>(
      "[data-test='bulk-tags-panel'] [data-test='tag-input']",
    )!;
    expect(input).toBeTruthy();
    input.value = "keep";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises();
    expect(orgApi.organizeGallery).toHaveBeenLastCalledWith(ORIGIN_TARGET, {
      filenames: ["smurf.png", "frog.png"],
      add_tags: ["keep"],
    });

    await wrapper.get("[data-test='bulk-collections']").trigger("click");
    await flushPromises();
    const rows = document.querySelectorAll<HTMLInputElement>(
      "[data-test='bulk-collections-panel'] [data-test='collection-toggle']",
    );
    expect(rows.length).toBe(2);
    rows[0]!.checked = true;
    rows[0]!.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(orgApi.setCollectionItems).toHaveBeenCalledWith(
      ORIGIN_TARGET,
      "c-rivers",
      { add: ["smurf.png", "frog.png"], remove: [] },
    );
  });
});

describe("Keyboard", () => {
  it("F favorites the open print and ⌘⇧N starts a new collection", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "f" }));
    await flushPromises();
    expect(orgApi.organizeGallery).toHaveBeenCalledWith(ORIGIN_TARGET, {
      filenames: ["smurf.png"],
      favorite: false,
    });
    vi.mocked(requestText).mockResolvedValueOnce("Keyboard made");
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "n", metaKey: true, shiftKey: true }),
    );
    await flushPromises();
    expect(orgApi.createCollection).toHaveBeenCalledWith(ORIGIN_TARGET, {
      name: "Keyboard made",
    });
  });

  it("ignores shortcuts while typing in the search field", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    const input = wrapper.get("[data-test='gallery-search']");
    await input.trigger("keydown", { key: "f" });
    expect(orgApi.organizeGallery).not.toHaveBeenCalled();
  });
});

describe("Refresh races (codex review)", () => {
  it("a poll inside the undo window never resurrects pending prints, and undo restores without duplicates", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    // Shift the delete to t = 7 s so the 10 s poll fires inside the 6 s
    // undo window (which then ends at t = 13 s).
    vi.advanceTimersByTime(7000);
    await flushPromises();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='grid-select-second']").trigger("click");
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("1");

    // t = 10 s: the poll refetches the still-live rows from the server.
    // They must stay masked while the trash commit is pending.
    vi.advanceTimersByTime(3000);
    await flushPromises();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("1");

    // Undo restores each print exactly once — never duplicated rows.
    const toastId = useNotifications().toasts[0]!.id;
    runToastAction(toastId);
    await flushPromises();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("3");
    expect(
      wrapper.get("[data-test='grid-names']").text().split(",").sort(),
    ).toEqual(["frog.png", "plain.png", "smurf.png"]);
    expect(orgApi.trashMany).not.toHaveBeenCalled();
  });

  it("commit after a mid-window poll still removes the prints for good", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    vi.advanceTimersByTime(7000);
    await flushPromises();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("2");

    // t = 10 s poll re-reads the still-live rows.
    vi.advanceTimersByTime(3000);
    await flushPromises();
    // t = 13 s: the undo window elapses, the trash commit fires.
    listGalleryMock.mockResolvedValue([frog, plain]);
    vi.advanceTimersByTime(3000);
    await flushPromises();
    expect(orgApi.trashMany).toHaveBeenCalledWith(ORIGIN_TARGET, ["smurf.png"]);
    expect(wrapper.get("[data-test='grid-count']").text()).toBe("2");
    expect(wrapper.get("[data-test='grid-names']").text()).not.toContain(
      "smurf.png",
    );
  });
});

describe("Mixed-capability delete wording", () => {
  it("names the permanent deletions when only some copies could be trashed", async () => {
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        { id: "plato", name: "plato", url: "http://plato:7680" },
      ]),
    );
    hostCapabilitiesMock.mockImplementation(async (host: { id: string }) =>
      host.id === "origin" ? ORGANIZING : { gallery: { can_delete: true } },
    );
    listGalleryMock.mockResolvedValue([
      smurf,
      { ...smurf, hostId: "plato", hostLabel: "plato" },
    ]);
    const wrapper = await mounted();
    await wrapper.get("[data-test='gallery-select']").trigger("click");
    await wrapper.get("[data-test='grid-select-first']").trigger("click");
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    await flushPromises();
    // One copy hard-deletes on plato, so the flow is not reversible.
    expect(requestConfirm).toHaveBeenCalled();
    const texts = useNotifications().toasts.map((t) => t.text);
    expect(texts).toContain(
      "Trashed 1 print; 1 copy on an older machine was deleted permanently.",
    );
    expect(texts.some((t) => t.includes("Trashed print everywhere"))).toBe(
      false,
    );
  });
});

describe("Capability-scoped scope options", () => {
  const ORGANIZE_NO_TRASH = {
    gallery: {
      can_delete: true,
      organize: true,
      trash: { enabled: false, retention_days: 0 },
    },
  };

  it("offers Trash only when a host has trash enabled", async () => {
    hostCapabilitiesMock.mockResolvedValue(ORGANIZE_NO_TRASH);
    hostGalleryMock.mockResolvedValue([]);
    const wrapper = await mounted();
    const labels = wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")
      .map((b) => b.text());
    expect(labels).toEqual(["Prints · 3", "Collections · 2"]);
  });

  it("offers Collections only when a host can organize", async () => {
    hostCapabilitiesMock.mockResolvedValue({
      gallery: {
        can_delete: true,
        trash: { enabled: true, retention_days: 30 },
      },
    });
    const wrapper = await mounted();
    const labels = wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")
      .map((b) => b.text());
    expect(labels).toEqual(["Prints · 3", "Trash · 1"]);
  });

  it("clamps a deep-linked scope no host offers once capabilities arrive", async () => {
    routeQuery.value = { scope: "trash" };
    hostCapabilitiesMock.mockResolvedValue(ORGANIZE_NO_TRASH);
    hostGalleryMock.mockResolvedValue([]);
    const wrapper = await mounted();
    expect(wrapper.get("[data-test='grid']").attributes("data-trash")).toBe(
      "false",
    );
    expect(replaceMock).toHaveBeenLastCalledWith({ query: {} });
  });

  it("keeps a deep-linked scope while capabilities are unknown", async () => {
    routeQuery.value = { scope: "trash" };
    hostCapabilitiesMock.mockRejectedValue(new Error("down"));
    hostGalleryMock.mockResolvedValue([]);
    const wrapper = await mounted();
    // A failed probe is unknown, not incapable — never clamp on it.
    expect(wrapper.get(".gal").attributes("data-scope")).toBe("trash");
  });
});

describe("Pending trash shadows survive failed listings", () => {
  it("keeps a just-trashed shadow until a successful listing confirms it", async () => {
    vi.useFakeTimers();
    const wrapper = mountPage();
    await flushPromises();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-delete']").trigger("click");
    await flushPromises();
    vi.advanceTimersByTime(6000); // undo window elapses → commit → shadow
    await flushPromises();
    // The server no longer lists it live, but the trash listing FAILS —
    // the shadow must survive that refresh instead of vanishing.
    listGalleryMock.mockResolvedValue([frog, plain]);
    hostGalleryMock.mockRejectedValue(new Error("listing down"));
    vi.advanceTimersByTime(4000); // t = 10 s poll
    await flushPromises();
    await wrapper
      .get("[data-test='library-scope']")
      .findAll(".ms-seg__btn")[2]!
      .trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='grid-names']").text()).toContain(
      "smurf.png",
    );

    // A successful listing that OMITS the row still keeps the shadow.
    hostGalleryMock.mockResolvedValue([trashedOld]);
    vi.advanceTimersByTime(10_000);
    await flushPromises();
    expect(wrapper.get("[data-test='grid-names']").text()).toContain(
      "smurf.png",
    );

    // Once a successful listing includes it, the shadow yields to the row.
    hostGalleryMock.mockResolvedValue([
      trashedOld,
      { ...smurf, trashed_at: 1_700_000_100 },
    ]);
    vi.advanceTimersByTime(10_000);
    await flushPromises();
    const names = wrapper.get("[data-test='grid-names']").text().split(",");
    expect(names.filter((n) => n === "smurf.png")).toHaveLength(1);
  });
});
