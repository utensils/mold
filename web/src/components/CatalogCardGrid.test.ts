import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import CatalogCardGrid from "./CatalogCardGrid.vue";
import type { RoutableHost } from "../lib/hostRouting";

const toastMock = vi.fn();
vi.mock("../lib/toasts", () => ({
  toast: (...args: unknown[]) => toastMock(...args),
}));

// Single-host registry: pulls resolve to the serving origin with no picker,
// exactly as this grid behaved before installs became host-targeted.
const mockHosts = ref<RoutableHost[]>([
  {
    id: "origin",
    label: "this server",
    url: "",
    status: "ready",
    queueDepth: 0,
    gpu: null,
  },
]);
vi.mock("../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: mockHosts,
    modelOwnerIds: () => [],
    inventoryKnown: () => true,
  }),
}));

// Stub IntersectionObserver so jsdom/happy-dom mounts don't crash on it.
// The test inspects markup rather than driving observer callbacks.
class FakeIntersectionObserver {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
  takeRecords = vi.fn(() => []);
}

const baseEntry = {
  id: "hf:row-0",
  name: "Row 0",
  family: "flux",
  supported: true,
  installed: false,
  source: "hf",
  source_id: "r0",
  author: null,
  family_role: "foundation",
  sub_family: null,
  modality: "image",
  kind: "checkpoint",
  file_format: "safetensors",
  bundling: "separated",
  size_bytes: 1,
  download_count: 100,
  rating: null,
  likes: 0,
  nsfw: false,
  thumbnail_url: null,
  description: null,
  license: null,
  license_flags: null,
  tags: [],
  companions: [],
  download_recipe: { files: [], needs_token: null },
  created_at: null,
  updated_at: null,
  added_at: 0,
};

let mockState: {
  entries: any;
  visibleEntries: any;
  resultCount: any;
  layout: any;
  total: any;
  loading: any;
  loadingMore: any;
  hasMore: any;
  errorMsg: any;
  providerErrors: any;
  refresh: ReturnType<typeof vi.fn>;
  loadMore: ReturnType<typeof vi.fn>;
  openDetail: ReturnType<typeof vi.fn>;
  startDownload: ReturnType<typeof vi.fn>;
};

vi.mock("../composables/useCatalog", () => ({
  useCatalog: () => mockState,
}));

beforeEach(() => {
  toastMock.mockReset();
  (globalThis as any).IntersectionObserver = FakeIntersectionObserver;
  mockState = {
    entries: ref([baseEntry]),
    visibleEntries: ref([baseEntry]),
    resultCount: ref(1),
    layout: ref<"grid" | "list">("grid"),
    total: ref<number | null>(1),
    loading: ref(false),
    loadingMore: ref(false),
    hasMore: ref(true),
    errorMsg: ref<string | null>(null),
    providerErrors: ref([]),
    refresh: vi.fn(),
    loadMore: vi.fn(),
    openDetail: vi.fn(),
    startDownload: vi.fn(),
  };
});

afterEach(() => {
  vi.restoreAllMocks();
  delete (globalThis as any).IntersectionObserver;
});

describe("CatalogCardGrid result count", () => {
  // The unfiltered feed is checkpoint-heavy, so switching between All and
  // Models leaves the first ~130 cards byte-identical — the filter applied
  // but the visible screen did not move, which read as "the chips do
  // nothing". A count is the one piece of feedback every filter changes.
  it("reports the count the composable resolved", () => {
    mockState.resultCount = ref(854);
    const w = mount(CatalogCardGrid);
    expect(w.find('[data-testid="catalog-result-count"]').text()).toContain(
      "854",
    );
  });

  it("singularises a one-row result", () => {
    mockState.resultCount = ref(1);
    const w = mount(CatalogCardGrid);
    expect(w.find('[data-testid="catalog-result-count"]').text()).toBe(
      "1 result",
    );
  });
});

describe("CatalogCardGrid provider resilience", () => {
  it("keeps healthy rows visible beside a provider warning and retries", async () => {
    mockState.providerErrors = ref([
      { source: "civitai", message: "Civitai is temporarily unavailable." },
    ]);
    const w = mount(CatalogCardGrid);

    expect(w.get("[data-test='catalog-provider-warning']").text()).toContain(
      "Civitai",
    );
    expect(w.findAllComponents({ name: "CatalogCard" })).toHaveLength(1);
    await w.get("[data-test='catalog-retry']").trigger("click");
    expect(mockState.refresh).toHaveBeenCalledOnce();
  });
});

describe("CatalogCardGrid download feedback", () => {
  it("shows a user-visible error when a catalog pull is rejected", async () => {
    mockState.startDownload.mockRejectedValueOnce(
      new Error("not a supported built-in model or LoRA"),
    );
    const w = mount(CatalogCardGrid);

    w.getComponent({ name: "CatalogCard" }).vm.$emit("pull");
    await flushPromises();

    expect(toastMock).toHaveBeenCalledWith(
      "error",
      "not a supported built-in model or LoRA",
    );
  });
});

describe("CatalogCardGrid layout", () => {
  it("renders the existing card grid by default", () => {
    const w = mount(CatalogCardGrid);
    expect(
      w.get("[data-test='catalog-results']").attributes("data-layout"),
    ).toBe("grid");
  });

  it("switches the same catalog cards into the compact list layout", () => {
    mockState.layout = ref("list");
    const w = mount(CatalogCardGrid);
    expect(
      w.get("[data-test='catalog-results']").attributes("data-layout"),
    ).toBe("list");
    expect(
      w.getComponent({ name: "CatalogCard" }).attributes("data-layout"),
    ).toBe("list");
  });
});

describe("CatalogCardGrid client-side filtering", () => {
  // The grid renders the modality-filtered view, not the raw fetched page:
  // rendering `entries` would put image models under the Video chip.
  it("renders visibleEntries rather than every fetched entry", () => {
    const video = { ...baseEntry, id: "hf:row-1", modality: "video" };
    mockState.entries = ref([baseEntry, video]);
    mockState.visibleEntries = ref([video]);
    mockState.resultCount = ref(1);
    const w = mount(CatalogCardGrid);
    expect(w.findAllComponents({ name: "CatalogCard" }).length).toBe(1);
  });

  it("does not send the user to a 'Refresh catalog' control that no longer exists", () => {
    mockState.visibleEntries = ref([]);
    mockState.resultCount = ref(0);
    mockState.hasMore = ref(false);
    const w = mount(CatalogCardGrid);
    expect(w.text()).toContain("No models found.");
    expect(w.text()).not.toContain("Refresh catalog");
  });

  it("shows the empty state when the filter hides every fetched row", () => {
    mockState.entries = ref([baseEntry]);
    mockState.visibleEntries = ref([]);
    mockState.resultCount = ref(0);
    mockState.hasMore = ref(false);
    const w = mount(CatalogCardGrid);
    expect(w.text()).toContain("No models found.");
  });
});

describe("CatalogCardGrid infinite scroll", () => {
  it("renders the load-more sentinel when hasMore is true", () => {
    const w = mount(CatalogCardGrid);
    expect(w.find('[data-testid="catalog-load-more-sentinel"]').exists()).toBe(
      true,
    );
  });

  it("omits the sentinel when hasMore is false", () => {
    mockState.hasMore = ref(false);
    const w = mount(CatalogCardGrid);
    expect(w.find('[data-testid="catalog-load-more-sentinel"]').exists()).toBe(
      false,
    );
  });

  it("shows a loading-more indicator when loadingMore is true", () => {
    mockState.loadingMore = ref(true);
    const w = mount(CatalogCardGrid);
    expect(w.text()).toContain("Loading more");
  });

  it("attaches an IntersectionObserver to the sentinel on mount", () => {
    const observeSpy = vi.fn();
    (globalThis as any).IntersectionObserver = class {
      observe = observeSpy;
      disconnect = vi.fn();
      unobserve = vi.fn();
      takeRecords = () => [];
    };
    mount(CatalogCardGrid);
    expect(observeSpy).toHaveBeenCalledTimes(1);
  });

  it("calls loadMore when the observer reports the sentinel intersecting", () => {
    let storedCallback: ((entries: any[]) => void) | null = null;
    (globalThis as any).IntersectionObserver = class {
      constructor(cb: (entries: any[]) => void) {
        storedCallback = cb;
      }
      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() {
        return [];
      }
    };
    mount(CatalogCardGrid);
    expect(storedCallback).not.toBeNull();
    storedCallback!([{ isIntersecting: true }]);
    expect(mockState.loadMore).toHaveBeenCalled();
  });
});
