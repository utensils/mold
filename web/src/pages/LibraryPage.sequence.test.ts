/**
 * Web mirror of the Library's two sequence doors: primary **Edit sequence**
 * (re-entering the durable job with its cached clips) and explicit
 * **Duplicate as new** (a NEW draft from the print's recorded clips), including the click-time 404 →
 * reuse fallback and the deliberate refusal to downgrade on an unreachable
 * host.
 */
import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { createPinia, setActivePinia } from "pinia";
import LibraryPage from "./LibraryPage.vue";
import { resetNotifications, useNotifications } from "../lib/toasts";
import { __testing__ as formTesting } from "../composables/useGenerateForm";
import {
  __testing__ as chainTesting,
  useChainJobs,
} from "../composables/useChainJobs";
import {
  pendingSequenceHandoff,
  takeSequenceHandoff,
} from "../composables/useSequenceHandoff";
import type { GalleryImage } from "../types";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useGenerateForm } from "../composables/useGenerateForm";

const { listGalleryMock, deleteMock, fetchModelsMock, getChainJobMock } =
  vi.hoisted(() => ({
    listGalleryMock: vi.fn(),
    deleteMock: vi.fn(),
    fetchModelsMock: vi.fn(),
    getChainJobMock: vi.fn(),
  }));
const { pushMock, replaceMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  replaceMock: vi.fn(),
}));

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ApiHttpError: actual.ApiHttpError,
    listGallery: listGalleryMock,
    fetchModels: fetchModelsMock,
    deleteGalleryImage: deleteMock,
    getChainJob: getChainJobMock,
    imageUrl: (f: string) => `/api/gallery/image/${f}`,
    thumbnailUrl: (f: string) => `/api/gallery/thumbnail/${f}`,
  };
});
import { ApiHttpError } from "../api";

vi.mock("../components/machines/hostClient", () => ({
  hostDeleteGalleryImage: vi.fn(),
  hostGallery: vi.fn(async () => []),
  hostCapabilities: vi.fn(async () => ({ gallery: { can_delete: true } })),
  hostApiTarget: (host: { url: string; apiKey?: string }) => ({
    baseUrl: host.url,
    apiKey: host.apiKey ?? null,
  }),
}));
vi.mock("@studio/api/galleryOrganization", () => ({
  listCollections: vi.fn(async () => []),
  listTags: vi.fn(async () => []),
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

const sequencePrint: GalleryImage = {
  filename: "sequence.mp4",
  timestamp: 3,
  format: "mp4",
  metadata: {
    prompt: "a harbour at dawn\nthe boats leave",
    model: "ltx-video",
    seed: 7,
    steps: 25,
    guidance: 3,
    width: 1024,
    height: 576,
    version: "test",
    chain_job_id: "job-9",
    chain: {
      stage_count: 2,
      motion_tail_frames: 0,
      stages: [
        { prompt: "a harbour at dawn", frames: 25, transition: "smooth" },
        { prompt: "the boats leave", frames: 25, transition: "cut" },
      ],
    },
  },
};
const legacyPrint: GalleryImage = {
  filename: "legacy.png",
  timestamp: 2,
  format: "png",
  metadata: {
    prompt: "a wandering cat",
    model: "flux-dev:fp16",
    seed: 1,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 1024,
    version: "test",
  },
};
const oneShotVideoPrint: GalleryImage = {
  filename: "one-shot.mp4",
  timestamp: 1,
  format: "mp4",
  metadata: {
    prompt: "a plane crosses the runway",
    model: "ltx-video",
    seed: 202,
    steps: 31,
    guidance: 4,
    width: 1024,
    height: 576,
    frames: 121,
    fps: 30,
    version: "test",
  },
};

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
    <button data-test="grid-open-legacy" @click="$emit('open', entries[1])">open legacy</button>
    <button data-test="grid-open-video" @click="$emit('open', entries[2])">open video</button>
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
    isSequence: Boolean,
    canEditSequence: Boolean,
  },
  emits: [
    "close",
    "prev",
    "next",
    "reuse",
    "use-source",
    "upscale",
    "delete",
    "edit-sequence",
  ],
  template: `<div v-if="item" data-test="lightbox">
    <button
      data-test="lb-primary"
      @click="canEditSequence ? $emit('edit-sequence', item) : $emit('reuse', item)"
    >primary</button>
    <button data-test="lb-reuse" @click="$emit('reuse', item)">duplicate</button>
    <button
      v-if="canEditSequence"
      data-test="lb-edit-sequence"
      @click="$emit('edit-sequence', item)"
    >edit</button>
    <span data-test="lb-is-sequence">{{ isSequence }}</span>
  </div>`,
});

async function mounted() {
  const wrapper = mount(LibraryPage, {
    global: {
      stubs: {
        GalleryGrid: GalleryGridStub,
        GalleryFeed: { template: "<div data-test='feed' />" },
        HistoryDrawer: { template: "<div />" },
        Lightbox: LightboxStub,
        Transition: false,
      },
    },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  resetNotifications();
  formTesting.resetForTest();
  chainTesting.reset();
  takeSequenceHandoff();
  listGalleryMock
    .mockReset()
    .mockResolvedValue([sequencePrint, legacyPrint, oneShotVideoPrint]);
  fetchModelsMock.mockReset().mockResolvedValue([
    {
      name: "ltx-video",
      family: "ltx-video",
      default_width: 1024,
      default_height: 576,
      default_steps: 25,
      default_guidance: 3,
    },
  ]);
  deleteMock.mockReset().mockResolvedValue(undefined);
  getChainJobMock.mockReset();
  pushMock.mockReset();
  replaceMock.mockReset();
});

describe("LibraryPage — sequence prints", () => {
  it("duplicates a sequence print's clips into a NEW draft", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-reuse']").trigger("click");
    await flushPromises();

    const handoff = pendingSequenceHandoff().value;
    expect(handoff).toMatchObject({
      kind: "reuse",
      metadata: { chain_job_id: "job-9" },
    });
    expect(pushMock).toHaveBeenCalledWith({
      path: "/create",
      query: { output: "sequence" },
    });
  });

  it.each([
    {
      kind: "image",
      selector: "[data-test='grid-open-legacy']",
      expected: legacyPrint.metadata,
    },
    {
      kind: "video",
      selector: "[data-test='grid-open-video']",
      expected: oneShotVideoPrint.metadata,
    },
  ])(
    "restores a normal $kind print into One shot while Sequence is active",
    async ({ selector, expected }) => {
      const draft = useSequenceDraftStore();
      draft.hydrate();
      draft.setOutput(
        "sequence",
        { getPrompt: () => "stale sequence", setPrompt: () => {} },
        25,
      );
      const wrapper = await mounted();
      await wrapper.get(selector).trigger("click");
      await wrapper.get("[data-test='lb-reuse']").trigger("click");
      await flushPromises();

      expect(pendingSequenceHandoff().value).toBeNull();
      expect(draft.output).toBe("single");
      expect(useGenerateForm().state.value.model).toBe(expected.model);
      expect(useGenerateForm().state.value.prompt).toBe(expected.prompt);
      expect(useGenerateForm().state.value.seed).toBe(expected.seed);
      expect(useGenerateForm().state.value.steps).toBe(expected.steps);
      expect(pushMock).toHaveBeenCalledWith({ name: "create" });
    },
  );

  it("offers Edit sequence only on a sequence print", async () => {
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    expect(wrapper.find("[data-test='lb-edit-sequence']").exists()).toBe(true);
    expect(wrapper.get("[data-test='lb-is-sequence']").text()).toBe("true");

    await wrapper.get("[data-test='grid-open-legacy']").trigger("click");
    expect(wrapper.find("[data-test='lb-edit-sequence']").exists()).toBe(false);
  });

  it("hides Edit sequence once the loaded listing proves the job is gone", async () => {
    useChainJobs().state.byHost["origin"] = { jobs: [], error: null };
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    expect(wrapper.find("[data-test='lb-edit-sequence']").exists()).toBe(false);
  });

  it("hands a live job into the edit session", async () => {
    getChainJobMock.mockResolvedValue({ id: "job-9", state: "completed" });
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-primary']").trigger("click");
    await flushPromises();

    expect(getChainJobMock).toHaveBeenCalledWith("job-9", {
      baseUrl: expect.any(String),
    });
    expect(pendingSequenceHandoff().value).toEqual({
      kind: "edit",
      hostId: "origin",
      jobId: "job-9",
    });
    expect(pushMock).toHaveBeenCalledWith({
      path: "/create",
      query: { output: "sequence" },
    });
  });

  it("falls back to reuse and names the host on a 404", async () => {
    getChainJobMock.mockRejectedValue(
      new ApiHttpError("GET /api/chain-jobs/job-9", 404, ""),
    );
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-primary']").trigger("click");
    await flushPromises();

    expect(pendingSequenceHandoff().value?.kind).toBe("reuse");
    expect(
      useNotifications().toasts.some((t) =>
        t.text.startsWith("That sequence job is gone from"),
      ),
    ).toBe(true);
  });

  it("does NOT silently downgrade to reuse when the host is unreachable", async () => {
    getChainJobMock.mockRejectedValue(new Error("network down"));
    const wrapper = await mounted();
    await wrapper.get("[data-test='grid-open']").trigger("click");
    await wrapper.get("[data-test='lb-primary']").trigger("click");
    await flushPromises();

    expect(pendingSequenceHandoff().value).toBeNull();
    expect(
      useNotifications().toasts.some(
        (t) =>
          t.kind === "error" && t.text.endsWith("Check the host in Machines."),
      ),
    ).toBe(true);
    expect(pushMock).not.toHaveBeenCalled();
  });
});
