/**
 * A sequence print's two doors out of the Library: **Reuse settings** (a NEW
 * draft from the recorded clips) and **Edit sequence** (re-entering the
 * durable job with its cached clips). The rules that matter are the ones a
 * regression would silently break — which action shows, and what happens when
 * the click-time existence check comes back 404 vs unreachable.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const { apiFetchTo, localGalleryList } = vi.hoisted(() => ({
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  localGalleryList: vi.fn(),
}));

vi.mock("@tanstack/vue-virtual", () => ({
  useVirtualizer: (options: { value: { count: number } }) => ({
    getTotalSize: () => options.value.count * 188,
    getVirtualItems: () =>
      Array.from({ length: options.value.count }, (_, index) => ({
        index,
        key: index,
        start: index * 188,
      })),
  }),
}));
vi.mock("../lib/gallery/layout", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/gallery/layout")>();
  return {
    ...actual,
    layoutJustifiedRows: (items: GalleryImage[], _w: number, targetHeight: number) =>
      items.map((item) => ({
        height: targetHeight,
        items: [{ item, width: targetHeight, height: targetHeight }],
      })),
  };
});
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo,
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public readonly status: number,
    ) {
      super(message);
      this.name = "ApiError";
    }
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    localGalleryDelete: vi.fn(),
    localGalleryList,
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));

import LibraryView from "./LibraryView.vue";
import { ApiError } from "../lib/api/client";
import { useChainJobsStore } from "../stores/chainJobs";
import { useComposerStore } from "../stores/composer";
import { useConnectionStore } from "../stores/connection";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";
import type { GalleryImage, OutputMetadata } from "../lib/api/types";
import type { ChainJobDetail } from "@studio/lib/api/chainTypes";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

const baseMetadata: OutputMetadata = {
  prompt: "a harbour at dawn\nthe boats leave",
  model: "ltx-2.3-22b-distilled:fp8",
  seed: 7,
  steps: 8,
  guidance: 1,
  width: 1216,
  height: 704,
};

const chainMetadata: OutputMetadata = {
  ...baseMetadata,
  chain_job_id: "job-9",
  chain: {
    stage_count: 2,
    motion_tail_frames: 8,
    stages: [
      { prompt: "a harbour at dawn", frames: 97, transition: "smooth" },
      { prompt: "the boats leave", frames: 65, transition: "cut" },
    ],
  },
};

const sequencePrint: GalleryImage = {
  filename: "sequence.mp4",
  timestamp: 3,
  metadata: chainMetadata,
};
const legacyPrint: GalleryImage = {
  filename: "legacy.png",
  timestamp: 2,
  metadata: { ...baseMetadata, seed: 42 },
};

const stub = { template: "<div />" };
const historyDrawerStub = { name: "HistoryDrawer", props: ["open"], template: "<div />" };

async function mountView(opts: { localOrigin?: boolean; knownJobIds?: string[] } = {}) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
    ],
  });
  await router.push("/library");

  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = null;
  connection.status = "error";

  const hosts = useHostsStore();
  hosts.extras.push({
    id: "plato-7680",
    label: "plato",
    url: "http://plato:7680",
    apiKey: "secret",
    status: "ready",
    error: null,
    instanceId: null,
  });

  const gallery = useGalleryStore();
  const bucket = opts.localOrigin ? "local" : "plato-7680";
  gallery.buckets[bucket] = {
    items: [sequencePrint, legacyPrint],
    loading: false,
    error: null,
    loaded: true,
  };
  localGalleryList.mockResolvedValue(opts.localOrigin ? [sequencePrint, legacyPrint] : []);

  const chains = useChainJobsStore();
  if (opts.knownJobIds) {
    chains.byHost["plato-7680"] = {
      jobs: opts.knownJobIds.map((id) => ({
        id,
        state: "completed" as const,
        model: "ltx-2.3-22b-distilled:fp8",
        stage_count: 2,
        current_stage: 2,
        created_at_unix_ms: 1,
        updated_at_unix_ms: 2,
      })),
      error: null,
    };
  }

  const wrapper = mount(LibraryView, {
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: stub, HostFilterChips: stub, HistoryDrawer: historyDrawerStub },
    },
  });
  await flushPromises();
  return { wrapper, gallery, router, chains };
}

async function menuFor(
  wrapper: Awaited<ReturnType<typeof mountView>>["wrapper"],
  seedLabel: string,
) {
  const tile = wrapper.findAll("button").find((b) => b.text().includes(seedLabel));
  expect(tile, `tile ${seedLabel}`).toBeDefined();
  await tile!.trigger("contextmenu");
  return useContextMenuStore();
}

function entryNamed(menu: ReturnType<typeof useContextMenuStore>, label: string) {
  return menu.entries.find(
    (entry): entry is Exclude<MenuEntry, { separator: true }> =>
      !("separator" in entry) && entry.label === label,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("Library — sequence print actions", () => {
  it("offers Edit sequence on a sequence print but not on a legacy one", async () => {
    const { wrapper } = await mountView();

    expect(entryNamed(await menuFor(wrapper, "S 7"), "Edit sequence")).toBeDefined();
    useContextMenuStore().close();
    expect(entryNamed(await menuFor(wrapper, "S 42"), "Edit sequence")).toBeUndefined();
  });

  it("hides Edit sequence once the host's loaded listing proves the job is gone", async () => {
    // Positive knowledge only: an unloaded listing keeps the action visible.
    const { wrapper } = await mountView({ knownJobIds: ["job-other"] });
    expect(entryNamed(await menuFor(wrapper, "S 7"), "Edit sequence")).toBeUndefined();
  });

  it("reuses a sequence print's clips as a NEW draft on the Create sequence route", async () => {
    const { wrapper, router } = await mountView();
    const menu = await menuFor(wrapper, "S 7");
    menu.activate(entryNamed(menu, "Reuse settings")!);
    await flushPromises();

    const handoff = useComposerStore().pendingSequence;
    expect(handoff?.kind).toBe("reuse");
    expect(handoff).toMatchObject({ kind: "reuse", metadata: { chain_job_id: "job-9" } });
    // Never the newline-joined single-prompt prefill — that join is exactly
    // what sequence-aware reuse exists to avoid.
    expect(useComposerStore().prefill).toBeNull();
    expect(router.currentRoute.value.fullPath).toBe("/create?output=sequence");
  });

  it("keeps single-print reuse untouched for a legacy print", async () => {
    const { wrapper, router } = await mountView();
    const menu = await menuFor(wrapper, "S 42");
    menu.activate(entryNamed(menu, "Reuse settings")!);
    await flushPromises();

    expect(useComposerStore().pendingSequence).toBeNull();
    expect(useComposerStore().prefill).toMatchObject({ metadata: { seed: 42 } });
    expect(router.currentRoute.value.fullPath).toBe("/create");
  });

  it("hands a live job straight into the edit session", async () => {
    const { wrapper, chains, router } = await mountView();
    vi.spyOn(chains, "fetchDetail").mockResolvedValue({
      id: "job-9",
      state: "completed",
      model: "ltx-2.3-22b-distilled:fp8",
      stage_count: 2,
      current_stage: 2,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      stages: [],
    } as unknown as ChainJobDetail);

    const menu = await menuFor(wrapper, "S 7");
    menu.activate(entryNamed(menu, "Edit sequence")!);
    await flushPromises();

    expect(chains.fetchDetail).toHaveBeenCalledWith("plato-7680", "job-9");
    expect(useComposerStore().pendingSequence).toEqual({
      kind: "edit",
      hostId: "plato-7680",
      jobId: "job-9",
    });
    expect(router.currentRoute.value.fullPath).toBe("/create?output=sequence");
  });

  it("falls back to reuse and names the host when the job is gone (404)", async () => {
    const { wrapper, chains, router } = await mountView();
    vi.spyOn(chains, "fetchDetail").mockRejectedValue(new ApiError("gone", 404));

    const menu = await menuFor(wrapper, "S 7");
    menu.activate(entryNamed(menu, "Edit sequence")!);
    await flushPromises();

    expect(useComposerStore().pendingSequence?.kind).toBe("reuse");
    expect(
      useToastStore().items.some(
        (t) => t.message === "That sequence job is gone from plato. Reused its settings instead.",
      ),
    ).toBe(true);
    expect(router.currentRoute.value.fullPath).toBe("/create?output=sequence");
  });

  it("does NOT silently downgrade to reuse when the host is unreachable", async () => {
    // The job almost certainly still exists; quietly starting a fresh
    // sequence would throw away its cached clips.
    const { wrapper, chains, router } = await mountView();
    vi.spyOn(chains, "fetchDetail").mockRejectedValue(new Error("network down"));

    const menu = await menuFor(wrapper, "S 7");
    menu.activate(entryNamed(menu, "Edit sequence")!);
    await flushPromises();

    expect(useComposerStore().pendingSequence).toBeNull();
    expect(
      useToastStore().items.some(
        (t) => t.message === "Can't reach plato. Check the host in Machines." && t.kind === "error",
      ),
    ).toBe(true);
    expect(router.currentRoute.value.path).toBe("/library");
  });

  it("offers reuse but no edit when the origin host does not resolve", async () => {
    // `hostFor("local")` is null while the built-in engine isn't ready, and
    // an unresolved origin must never guess a sibling host's job.
    const { wrapper } = await mountView({ localOrigin: true });
    const menu = await menuFor(wrapper, "S 7");
    expect(entryNamed(menu, "Reuse settings")).toBeDefined();
    expect(entryNamed(menu, "Edit sequence")).toBeUndefined();
  });
});
