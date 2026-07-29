import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const fetchHistoryAll = vi
  .fn()
  .mockResolvedValue({ entries: [], supportedHostIds: [], unreachableHostIds: [] });
const clearHistoryOn = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/api/history", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api/history")>();
  return {
    ...actual,
    fetchHistory: vi.fn().mockResolvedValue([]),
    fetchHistoryAll: (...a: unknown[]) => fetchHistoryAll(...a),
    clearHistory: vi.fn().mockResolvedValue(undefined),
    clearHistoryOn: (...a: unknown[]) => clearHistoryOn(...a),
  };
});
vi.mock("../../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiFetch: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiJson: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { localGalleryList: vi.fn().mockResolvedValue({ images: [], target: null }) },
}));

import HistoryDrawer from "./HistoryDrawer.vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore } from "../../stores/hosts";
import { useComposerStore } from "../../stores/composer";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { GalleryImage } from "../../lib/api/types";
import type { ChainJobSummary } from "@studio/lib/api/chainTypes";

const stub = { template: "<div />" };

function run(filename: string, prompt: string, timestamp: number): GalleryImage {
  return {
    filename,
    timestamp,
    metadata: {
      prompt,
      model: "flux2-klein",
      seed: 42,
      steps: 4,
      guidance: 1,
      width: 1024,
      height: 768,
    },
  };
}

let router: Router;

async function mountDrawer({ extra = false, at = "/library" } = {}) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/create", component: stub },
      { path: "/library", component: stub },
    ],
  });
  router.push(at);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://x", apiKey: null };
  conn.status = "ready";
  const gallery = useGalleryStore();
  // Local primary → one bucket, key "local" (the built-in engine's host id).
  gallery.buckets["local"] = {
    items: [
      run("a.png", "a lighthouse at dusk", 1_700_000_100),
      run("b.png", "a stoic owl", 1_700_000_000),
    ],
    loading: false,
    error: null,
    loaded: true,
  };
  if (extra) {
    useHostsStore().extras.push({
      id: "okra-7680",
      label: "okra",
      url: "http://okra:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    gallery.buckets["okra-7680"] = {
      items: [run("o.png", "a remote heron", 1_700_000_050)],
      loading: false,
      error: null,
      loaded: true,
    };
  }
  // Durable sequence actions are network calls; the drawer's own wiring is
  // what these tests exercise, so every store action is a spy from mount.
  const chains = useChainJobsStore();
  vi.spyOn(chains, "fetchAll").mockResolvedValue(undefined);
  vi.spyOn(chains, "resume").mockResolvedValue(undefined);
  vi.spyOn(chains, "remove").mockResolvedValue(undefined);
  vi.spyOn(chains, "watch").mockReturnValue(undefined);
  vi.spyOn(chains, "clearInactive").mockResolvedValue({ cleared: 2, failed: 0 });
  vi.spyOn(chains, "gc").mockResolvedValue({
    swept_ephemeral_jobs: 0,
    pruned_artifact_dirs: 0,
  });

  const wrapper = mount(HistoryDrawer, {
    props: { open: true },
    attachTo: document.body,
    global: { plugins: [pinia, router], stubs: { AuthedMedia: stub } },
  });
  await flushPromises();
  return wrapper;
}

const promptEntry = (prompt: string, hostId: string, hostLabel: string, used_at: number) => ({
  prompt,
  model: "flux2-klein",
  used_at,
  hostId,
  hostLabel,
});

beforeEach(() => {
  vi.clearAllMocks();
  fetchHistoryAll.mockResolvedValue({ entries: [], supportedHostIds: [], unreachableHostIds: [] });
  clearHistoryOn.mockResolvedValue(undefined);
});

describe("HistoryDrawer runs", () => {
  it("starts wider and persists left-edge resizing or a double-click reset", async () => {
    const wrapper = await mountDrawer();
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined);
    expect(wrapper.getComponent(DrawerPanel).props("width")).toBe(620);

    const handle = wrapper.getComponent(PanelResizeHandle);
    expect(handle.props("label")).toBe("Resize history");
    handle.vm.$emit("resize", -80);
    await flushPromises();
    expect(wrapper.getComponent(DrawerPanel).props("width")).toBe(700);
    handle.vm.$emit("commit");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ historyDrawerWidth: 700 });

    handle.vm.$emit("reset");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ historyDrawerWidth: null });
  });

  it("shows past runs with thumbnails and full metadata", async () => {
    const wrapper = await mountDrawer();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("a lighthouse at dusk");
    expect(rows[0]!.text()).toContain("flux2-klein");
    expect(rows[0]!.text()).toContain("1024×768");
    expect(rows[0]!.text()).toContain("S 42");
    expect(rows[0]!.text()).toContain("4 steps");
  });

  it("reuses a run's full settings including the seed", async () => {
    const wrapper = await mountDrawer();
    await wrapper.get("[data-test='run-row']").trigger("click");
    await flushPromises();
    const composer = useComposerStore();
    expect(composer.prefill).toEqual({
      metadata: expect.objectContaining({
        prompt: "a lighthouse at dusk",
        seed: 42,
        width: 1024,
        height: 768,
      }),
    });
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("filters runs by prompt text", async () => {
    const wrapper = await mountDrawer();
    await wrapper.get("input[type='search']").setValue("owl");
    await flushPromises();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("a stoic owl");
  });

  it("loads the prompt log only when that tab is opened", async () => {
    const wrapper = await mountDrawer();
    expect(fetchHistoryAll).not.toHaveBeenCalled();
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(fetchHistoryAll).toHaveBeenCalled();
  });
});

describe("HistoryDrawer multi-host", () => {
  it("shows no host chips with a single source", async () => {
    const wrapper = await mountDrawer();
    expect(wrapper.find("[role='tablist']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='host-badge']")).toHaveLength(0);
  });

  it("shows filter chips and per-row host chips when several hosts are live", async () => {
    const wrapper = await mountDrawer({ extra: true });
    expect(wrapper.find("[role='tablist']").exists()).toBe(true);
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(3);
    const badges = wrapper.findAll("[data-test='host-badge']");
    expect(badges.map((b) => b.text())).toEqual(["This device", "okra", "This device"]);
  });

  it("the chip filter narrows the runs list and hides row chips", async () => {
    const wrapper = await mountDrawer({ extra: true });
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("a remote heron");
    expect(wrapper.findAll("[data-test='host-badge']")).toHaveLength(0);
  });

  it("fans the prompt log out over every ready host and tags rows", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    const targets = fetchHistoryAll.mock.calls[0]![0] as Array<{ hostId: string }>;
    expect(targets.map((t) => t.hostId)).toEqual(["local", "okra-7680"]);
    const rows = wrapper.findAll("[data-test='prompt-row']");
    expect(rows).toHaveLength(2);
    const badges = wrapper.findAll("[data-test='host-badge']");
    expect(badges.map((b) => b.text())).toEqual(["This Mac", "okra"]);
  });

  it("the chip filter narrows the prompt log too", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='prompt-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("remote prompt");
  });

  it("shows the unavailable state only when no host supports history", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [],
      supportedHostIds: [],
      unreachableHostIds: [],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Prompt history isn't available");

    fetchHistoryAll.mockResolvedValue({
      entries: [promptEntry("kept", "local", "This Mac", 1_000)],
      supportedHostIds: ["local"],
      unreachableHostIds: [],
    });
    const other = await mountDrawer({ extra: true });
    await other.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(other.text()).not.toContain("Prompt history isn't available");
  });

  it("clears only the filtered host", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await flushPromises();
    expect(clearHistoryOn).toHaveBeenCalledTimes(1);
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://okra:7680", apiKey: null });
  });

  it("clearing All names every history-capable host and hits them all", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='clear-history']").trigger("click");
    const label = wrapper.get("[data-test='clear-history']").text();
    expect(label).toContain("This device");
    expect(label).toContain("okra");
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await flushPromises();
    expect(clearHistoryOn).toHaveBeenCalledTimes(2);
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://x", apiKey: null });
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://okra:7680", apiKey: null });
  });
});

// ── Sequences: the one place durable sequence jobs are enumerated ──────────

function chainJob(overrides: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "job-1",
    state: "completed",
    model: "ltx-2.3-22b-distilled:fp8",
    stage_count: 5,
    current_stage: 4,
    created_at_unix_ms: 1_700_000_000_000,
    updated_at_unix_ms: 1_700_000_060_000,
    error: null,
    ...overrides,
  };
}

describe("HistoryDrawer sequences", () => {
  it("lists jobs from every host with active work first", async () => {
    const wrapper = await mountDrawer({ extra: true });
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [chainJob()], error: null };
    chains.byHost["okra-7680"] = {
      jobs: [chainJob({ id: "job-live", state: "running", created_at_unix_ms: 1_600_000_000_000 })],
      error: null,
    };
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();

    const rows = wrapper.findAll("[data-test='sequence-job-row']");
    expect(rows).toHaveLength(2);
    // Older, but running — the shared merge floats it.
    expect(rows[0]!.text()).toContain("running");
    expect(rows[0]!.text()).toContain("okra");
    expect(rows[1]!.text()).toContain("completed");
    expect(rows[1]!.text()).toContain("5 clips");
  });

  it("loads a selected sequence into Create without granting edit authority", async () => {
    const wrapper = await mountDrawer();
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [chainJob()], error: null };
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='sequence-job-row']").trigger("click");
    await flushPromises();
    expect(useComposerStore().pendingSequence).toEqual({
      kind: "inspect",
      hostId: "local",
      jobId: "job-1",
    });
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("opens on the tab named in the URL and writes the tab back on switch", async () => {
    const wrapper = await mountDrawer({ at: "/library?panel=history&tab=sequences" });
    expect(wrapper.get("[data-test='tab-sequences']").attributes("aria-pressed")).toBe("true");
    await wrapper.get("[data-test='tab-runs']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.tab).toBe("runs");
  });

  it("fetches the durable listing once when the drawer opens", async () => {
    const wrapper = await mountDrawer({ at: "/library?panel=history&tab=sequences" });
    const chains = useChainJobsStore();
    // Chain polling stops when nothing is active, so a settled list goes
    // stale unless the drawer refetches on open.
    expect(chains.fetchAll).toHaveBeenCalledTimes(1);
    await wrapper.get("[data-test='tab-runs']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();
    expect(chains.fetchAll).toHaveBeenCalledTimes(2);
  });

  it("routes row actions to the owning host", async () => {
    const wrapper = await mountDrawer({ extra: true });
    const chains = useChainJobsStore();
    chains.byHost["okra-7680"] = {
      jobs: [chainJob({ state: "failed", error: "boom" })],
      error: null,
    };
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='seq-resume']").trigger("click");
    expect(chains.resume).toHaveBeenCalledWith("okra-7680", "job-1");
    // Delete drops cached clips on that host — never a one-click verb.
    await wrapper.get("[data-test='seq-delete']").trigger("click");
    expect(chains.remove).not.toHaveBeenCalled();
    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    expect(chains.remove).toHaveBeenCalledWith("okra-7680", "job-1");
  });

  it("shows the print only when that host's gallery carries the job id", async () => {
    const wrapper = await mountDrawer();
    const gallery = useGalleryStore();
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [chainJob(), chainJob({ id: "job-2" })], error: null };
    gallery.buckets["local"]!.items.push({
      filename: "sequence.mp4",
      timestamp: 1_700_000_060,
      metadata: {
        prompt: "five clips",
        model: "ltx-2.3-22b-distilled:fp8",
        seed: 7,
        steps: 8,
        guidance: 1,
        width: 1536,
        height: 640,
        chain_job_id: "job-1",
      },
    });
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();

    const rows = wrapper.findAll("[data-test='sequence-row']");
    expect(rows).toHaveLength(2);
    const withPrint = rows.filter((r) => r.find("[data-test='seq-show-print']").exists());
    expect(withPrint).toHaveLength(1);

    await withPrint[0]!.get("[data-test='seq-show-print']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/library");
    expect(router.currentRoute.value.query.print).toBe("sequence.mp4");
  });

  it("scopes Clear inactive to the chip-selected host behind a two-press confirm", async () => {
    const wrapper = await mountDrawer({ extra: true });
    const gallery = useGalleryStore();
    const chains = useChainJobsStore();
    chains.byHost["okra-7680"] = {
      jobs: [chainJob(), chainJob({ id: "job-2" })],
      error: null,
    };
    gallery.filter = "okra-7680";
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='seq-clear-inactive']").trigger("click");
    expect(chains.clearInactive).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='seq-clear-inactive']").text()).toBe(
      "Delete 2 inactive sequences on okra?",
    );
    await wrapper.get("[data-test='seq-clear-inactive']").trigger("click");
    await flushPromises();
    expect(chains.clearInactive).toHaveBeenCalledWith("okra-7680");
  });

  it("warns before discarding scene playback and edit caches", async () => {
    const wrapper = await mountDrawer({ extra: true });
    const chains = useChainJobsStore();
    chains.byHost.local = { jobs: [chainJob()], error: null };
    chains.byHost["okra-7680"] = {
      jobs: [chainJob({ id: "job-2" })],
      error: null,
    };
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='seq-cleanup-disk']").trigger("click");
    expect(chains.gc).not.toHaveBeenCalled();
    const dialog = document.querySelector("[data-test='confirm-dialog']") as HTMLElement;
    expect(dialog.textContent).toContain(
      "cached scene media used for scene playback and sequence editing",
    );
    expect(dialog.textContent).toContain("Final videos in the Library remain.");

    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    await flushPromises();
    expect(chains.gc).toHaveBeenCalledTimes(2);
  });

  it("caps the rendered list and says how much it is holding back", async () => {
    const wrapper = await mountDrawer();
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: Array.from({ length: 431 }, (_, i) =>
        chainJob({ id: `job-${i}`, created_at_unix_ms: 1_700_000_000_000 + i }),
      ),
      error: null,
    };
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll("[data-test='sequence-row']")).toHaveLength(200);
    expect(wrapper.get("[data-test='sequence-cap-note']").text()).toBe("showing 200 of 431");
  });

  it("offers exactly one action when there are no sequences", async () => {
    const wrapper = await mountDrawer();
    await wrapper.get("[data-test='tab-sequences']").trigger("click");
    await flushPromises();
    const empty = wrapper.get("[data-test='sequences-empty']");
    expect(empty.text()).toContain("No sequences yet");
    const buttons = empty.findAll("button");
    expect(buttons).toHaveLength(1);
    await buttons[0]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/create");
  });
});
