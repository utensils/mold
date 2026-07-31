import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { computed, nextTick, ref } from "vue";
import CommandK from "./CommandK.vue";
import PalettePanel from "@ui/components/PalettePanel.vue";
import { theme, themeFamily } from "../../lib/theme";
import {
  useGenerateForm,
  __testing__ as generateFormTesting,
} from "../../composables/useGenerateForm";
import type { PaletteItem } from "@ui/components/types";
import { CATALOG_SEARCH_DEBOUNCE_MS } from "@studio/lib/modelSearch";

/** Loose stand-ins for the wire shapes; the palette only reads a few fields. */
interface FakeCatalogEntry {
  id: string;
  name: string;
  family: string;
  source: string;
  installed: boolean;
  supported: boolean;
}
interface FakeCatalogResponse {
  entries: FakeCatalogEntry[];
  page: number;
  page_size: number;
  total: number;
}
interface FakeInstallTarget {
  host: { id: string; label: string };
  action: string;
}
interface FakeInstallPlan {
  targets: FakeInstallTarget[];
  canInstall: boolean;
  label: string;
}

const EMPTY_CATALOG: FakeCatalogResponse = {
  entries: [],
  page: 1,
  page_size: 12,
  total: 0,
};
const NO_TARGETS: FakeInstallPlan = {
  targets: [],
  canInstall: false,
  label: "Pull",
};

const pushMock = vi.hoisted(() => vi.fn());
const setTargetMock = vi.hoisted(() => vi.fn());
const refreshMock = vi.hoisted(() => vi.fn(async () => {}));
const fetchCatalogSearchMock = vi.hoisted(() =>
  vi.fn<(params: unknown) => Promise<FakeCatalogResponse>>(),
);
const startDownloadOnMock = vi.hoisted(() => vi.fn(async () => {}));
const planForMock = vi.hoisted(() => vi.fn<() => FakeInstallPlan>());
const toastMock = vi.hoisted(() => vi.fn());

// Full-enough model rows so applyModelDefaults can resolve family + defaults.
const MODELS = vi.hoisted(() => [
  {
    name: "flux-dev:q4",
    downloaded: true,
    family: "flux",
    description: "",
    hf_repo: "",
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
  },
  {
    name: "ltx2-distilled",
    downloaded: true,
    family: "ltx2",
    description: "",
    hf_repo: "",
    default_width: 768,
    default_height: 512,
    default_steps: 8,
    default_guidance: 1,
  },
]);

const owners = vi.hoisted(() => ({
  value: { "flux-dev:q4": ["origin"], "ltx2-distilled": ["bender"] } as Record<
    string,
    string[]
  >,
}));
const targetId = vi.hoisted(() => ({ value: "origin" }));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: pushMock }),
}));

vi.mock("../../api", () => ({
  fetchCatalogSearch: fetchCatalogSearchMock,
}));

vi.mock("../../lib/toasts", () => ({ toast: toastMock }));

vi.mock("../../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: ref([
      { id: "origin", label: "this server", status: "ready" },
      { id: "bender", label: "bender", status: "ready" },
    ]),
    installedModels: computed(() => MODELS),
    targetId: computed(() => targetId.value),
    setTarget: setTargetMock,
    modelOwnerIds: (name: string) => owners.value[name] ?? [],
    refresh: refreshMock,
  }),
}));

vi.mock("../../composables/useModelInstallTargets", () => ({
  useModelInstallTargets: () => ({
    planFor: planForMock,
    startDownloadOn: startDownloadOnMock,
    queuedMessage: () => "Download queued on bender",
  }),
}));

function items(wrapper: ReturnType<typeof mount>): PaletteItem[] {
  return wrapper.findComponent(PalettePanel).props("items") as PaletteItem[];
}

async function openPalette() {
  const wrapper = mount(CommandK, { props: { open: false } });
  await wrapper.setProps({ open: true });
  await flushPromises();
  return wrapper;
}

/** Type a query and let the catalog debounce fire.
 *  Advance by exactly the debounce — `runAllTimers` would also chase the
 *  generate form's self-rescheduling persist timer forever. */
async function type(wrapper: ReturnType<typeof mount>, q: string) {
  wrapper.findComponent(PalettePanel).vm.$emit("update:query", q);
  await nextTick();
  await vi.advanceTimersByTimeAsync(CATALOG_SEARCH_DEBOUNCE_MS);
  await flushPromises();
}

describe("CommandK", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    pushMock.mockClear();
    setTargetMock.mockClear();
    toastMock.mockClear();
    startDownloadOnMock.mockClear();
    fetchCatalogSearchMock.mockClear();
    fetchCatalogSearchMock.mockResolvedValue(EMPTY_CATALOG);
    planForMock.mockReturnValue(NO_TARGETS);
    targetId.value = "origin";
    localStorage.clear();
    generateFormTesting.resetForTest();
  });
  afterEach(() => {
    vi.useRealTimers();
    theme.value = "system";
    themeFamily.value = "mold";
  });

  it("offers navigation, action, and theme commands", async () => {
    const wrapper = await openPalette();
    const ids = items(wrapper).map((i) => i.id);
    expect(ids).toContain("go-library");
    expect(ids).toContain("go-machines");
    expect(ids).toContain("go-settings");
    expect(ids).toContain("action-new-print");
    expect(ids).toContain("theme-family-safelight");
  });

  it("lists installed models across the fleet as use commands", async () => {
    const wrapper = await openPalette();
    const rows = items(wrapper);
    expect(rows.map((i) => i.id)).toEqual(
      expect.arrayContaining(["model-flux-dev:q4", "model-ltx2-distilled"]),
    );
    expect(rows.find((i) => i.id === "model-flux-dev:q4")?.label).toBe(
      "Use flux-dev:q4",
    );
  });

  it("names the machine a model would move generation to", async () => {
    const wrapper = await openPalette();
    const rows = items(wrapper);
    // origin is pinned and owns flux, so no move is implied there.
    expect(rows.find((i) => i.id === "model-flux-dev:q4")?.hint).toBe("flux");
    expect(rows.find((i) => i.id === "model-ltx2-distilled")?.hint).toBe(
      "on bender",
    );
  });

  it("repins the generation target when the model lives elsewhere", async () => {
    const form = useGenerateForm();
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-ltx2-distilled");
    await nextTick();

    expect(setTargetMock).toHaveBeenCalledWith("bender");
    expect(form.state.value.model).toBe("ltx2-distilled");
    expect(pushMock).toHaveBeenCalledWith("/create");
  });

  it("leaves the target alone for a model the pinned machine already has", async () => {
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(setTargetMock).not.toHaveBeenCalled();
  });

  it("searches the catalog once the query is long enough", async () => {
    const wrapper = await openPalette();
    await type(wrapper, "q");
    expect(fetchCatalogSearchMock).not.toHaveBeenCalled();

    await type(wrapper, "qwen");
    expect(fetchCatalogSearchMock).toHaveBeenCalledWith({
      q: "qwen",
      kind: "checkpoint",
      page_size: 12,
    });
  });

  it("offers an install row for a catalog model nobody has", async () => {
    fetchCatalogSearchMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/qwen",
          name: "Qwen Image",
          family: "qwen-image",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await type(wrapper, "qwen");

    const row = items(wrapper).find((i) => i.id === "install-hf:org/qwen");
    expect(row?.label).toBe("Install Qwen Image");
    expect(row?.hint).toBe("not installed · hf");
    expect(row?.section).toBe("Install");
  });

  it("drops unsupported catalog entries", async () => {
    fetchCatalogSearchMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/broken",
          name: "Broken",
          family: "x",
          source: "hf",
          installed: false,
          supported: false,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await type(wrapper, "broken");

    expect(items(wrapper).some((i) => i.id.startsWith("install-"))).toBe(false);
  });

  it("never offers to install a model the fleet already has", async () => {
    fetchCatalogSearchMock.mockResolvedValue({
      entries: [
        {
          id: "flux-dev:q4",
          name: "FLUX Dev",
          family: "flux",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await type(wrapper, "flux");

    expect(items(wrapper).some((i) => i.id.startsWith("install-"))).toBe(false);
  });

  it("queues an install on the plan's first target without opening the picker", async () => {
    const target: FakeInstallTarget = {
      host: { id: "bender", label: "bender" },
      action: "install",
    };
    planForMock.mockReturnValue({
      targets: [target],
      canInstall: true,
      label: "Pull",
    });
    fetchCatalogSearchMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/qwen",
          name: "Qwen Image",
          family: "qwen-image",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await type(wrapper, "qwen");
    wrapper.findComponent(PalettePanel).vm.$emit("run", "install-hf:org/qwen");
    await flushPromises();

    expect(startDownloadOnMock).toHaveBeenCalledWith(target, "hf:org/qwen");
    expect(toastMock).toHaveBeenCalledWith(
      "success",
      "Download queued on bender",
    );
  });

  it("reports a failed install instead of closing silently", async () => {
    planForMock.mockReturnValue(NO_TARGETS);
    startDownloadOnMock.mockRejectedValueOnce(new Error("host is gone"));
    fetchCatalogSearchMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/qwen",
          name: "Qwen Image",
          family: "qwen-image",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await type(wrapper, "qwen");
    wrapper.findComponent(PalettePanel).vm.$emit("run", "install-hf:org/qwen");
    await flushPromises();

    expect(toastMock).toHaveBeenCalledWith(
      "error",
      "Couldn't queue Qwen Image: host is gone",
    );
  });

  it("discards a stale catalog response for an abandoned query", async () => {
    let releaseFirst: (v: FakeCatalogResponse) => void = () => {};
    fetchCatalogSearchMock.mockImplementationOnce(
      () => new Promise((resolve) => (releaseFirst = resolve)),
    );
    fetchCatalogSearchMock.mockResolvedValueOnce({
      entries: [
        {
          id: "hf:org/second",
          name: "Second",
          family: "x",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });

    const wrapper = await openPalette();
    await type(wrapper, "first");
    await type(wrapper, "second");
    releaseFirst({
      entries: [
        {
          id: "hf:org/first",
          name: "First",
          family: "x",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    await flushPromises();

    const ids = items(wrapper).map((i) => i.id);
    expect(ids).toContain("install-hf:org/second");
    expect(ids).not.toContain("install-hf:org/first");
  });

  it("marks the palette busy only while a catalog search is in flight", async () => {
    const wrapper = await openPalette();
    const panel = wrapper.findComponent(PalettePanel);
    expect(panel.props("busy")).toBe(false);

    panel.vm.$emit("update:query", "qwen");
    await nextTick();
    expect(panel.props("busy")).toBe(true);

    await vi.advanceTimersByTimeAsync(CATALOG_SEARCH_DEBOUNCE_MS);
    await flushPromises();
    expect(panel.props("busy")).toBe(false);
  });

  it("filters the items by a case-insensitive query", async () => {
    const wrapper = await openPalette();
    await type(wrapper, "safelight");

    expect(items(wrapper).map((i) => i.id)).toEqual(["theme-family-safelight"]);
  });

  it("runs a navigation command and closes", async () => {
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "go-models");
    await nextTick();

    expect(pushMock).toHaveBeenCalledWith("/models");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("closes on Escape after focus leaves the search input", async () => {
    const wrapper = await openPalette();
    const row = wrapper.get("[role='option']");
    (row.element as HTMLElement).focus();

    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await nextTick();

    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("mutates the shared theme refs from theme commands", async () => {
    const wrapper = await openPalette();
    const palette = wrapper.findComponent(PalettePanel);

    palette.vm.$emit("run", "theme-appearance-dark");
    await nextTick();
    expect(theme.value).toBe("dark");

    palette.vm.$emit("run", "theme-family-safelight");
    await nextTick();
    expect(themeFamily.value).toBe("safelight");
  });

  it("selects a model and opens create when a model command runs", async () => {
    const form = useGenerateForm();
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(pushMock).toHaveBeenCalledWith("/create");
  });

  it("applies the model's defaults and clears stale family state on a model command", async () => {
    const form = useGenerateForm();
    // Start in a Qwen-edit configuration carrying two edit images — the shape
    // that must not survive a switch to a FLUX model.
    form.state.value.model = "qwen-image-edit:q4";
    form.state.value.modelFamily = "qwen-image-edit";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "a.png", base64: "A" },
      { kind: "upload", filename: "b.png", base64: "B" },
    ];
    await nextTick();

    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(form.state.value.modelFamily).toBe("flux");
    // FLUX takes a single source image and never emits edit_images.
    expect(form.state.value.imageAttachments.length).toBeLessThanOrEqual(1);
    expect(form.toRequest().edit_images).toBeUndefined();
  });
});
