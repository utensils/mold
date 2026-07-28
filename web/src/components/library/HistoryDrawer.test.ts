import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";

const listChainJobs = vi.fn();
const deleteChainJob = vi.fn().mockResolvedValue(undefined);
const gcChainJobs = vi
  .fn()
  .mockResolvedValue({ swept_ephemeral_jobs: 0, pruned_artifact_dirs: 3 });
vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    listChainJobs: (...a: unknown[]) => listChainJobs(...a),
    deleteChainJob: (...a: unknown[]) => deleteChainJob(...a),
    gcChainJobs: (...a: unknown[]) => gcChainJobs(...a),
    cancelChainJob: vi.fn().mockResolvedValue(undefined),
    resumeChainJob: vi.fn().mockResolvedValue(undefined),
    getChainJob: vi.fn().mockResolvedValue({}),
  };
});

import HistoryDrawer from "./HistoryDrawer.vue";
import { __testing__ } from "../../composables/useChainJobs";
import type { ChainJobSummary } from "@studio/lib/api/chainTypes";

const stub = { template: "<div />" };
const routerPush = vi.fn();

const NOW = Date.now();

function job(overrides: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "chain-1",
    state: "completed",
    model: "ltx-2.3-22b-distilled:fp8",
    stage_count: 5,
    current_stage: 4,
    created_at_unix_ms: NOW - 300_000,
    updated_at_unix_ms: NOW - 240_000,
    error: null,
    ...overrides,
  };
}

async function mountDrawer(jobs: ChainJobSummary[]) {
  listChainJobs.mockResolvedValue({ jobs });
  const wrapper = mount(HistoryDrawer, {
    props: { open: true },
    global: {
      stubs: { EmptyStateBlock: false, DrawerPanel: false, Icon: stub },
      mocks: { $router: { push: routerPush } },
    },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  __testing__.reset();
  vi.clearAllMocks();
  localStorage.clear();
});
afterEach(() => (document.body.innerHTML = ""));

describe("web HistoryDrawer", () => {
  // Previously bitten: @ui overlay panels are `position: absolute; inset: 0`,
  // so on a long scrolling page they render off-screen without this host.
  it("renders inside a viewport-fixed host", async () => {
    const wrapper = await mountDrawer([job()]);
    const host = wrapper.get("[data-test='history-drawer-host']");
    expect(host.classes()).toEqual(
      expect.arrayContaining(["fixed", "inset-0"]),
    );
  });

  it("fetches the durable listing when it opens and lists active work first", async () => {
    const wrapper = await mountDrawer([
      job(),
      job({ id: "chain-live", state: "running", created_at_unix_ms: 1 }),
    ]);
    expect(listChainJobs).toHaveBeenCalled();

    const rows = wrapper.findAll("[data-test='sequence-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("running");
    expect(rows[1]!.text()).toContain("completed");
    expect(rows[1]!.text()).toContain("5 clips");
  });

  it("re-enters a job in Create rather than acting on it here", async () => {
    const wrapper = await mountDrawer([job()]);
    await wrapper.get("[data-test='seq-edit']").trigger("click");
    expect(wrapper.emitted("open-sequence")?.[0]).toEqual([
      { hostId: "origin", jobId: "chain-1", edit: true },
    ]);
    // A settled job is opened, not watched (§11 voice).
    expect(wrapper.get("[data-test='seq-watch']").text()).toBe("Open");
  });

  it("holds Clear inactive behind a confirm that names the count", async () => {
    const wrapper = await mountDrawer([job(), job({ id: "chain-2" })]);
    await wrapper.get("[data-test='seq-clear-inactive']").trigger("click");
    expect(deleteChainJob).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='seq-clear-inactive']").text()).toBe(
      "Delete 2 inactive sequences?",
    );
    await wrapper.get("[data-test='seq-clear-inactive']").trigger("click");
    await flushPromises();
    expect(deleteChainJob).toHaveBeenCalledTimes(2);
  });

  it("sweeps disk from the same footer the strip lost", async () => {
    const wrapper = await mountDrawer([job()]);
    await wrapper.get("[data-test='seq-cleanup-disk']").trigger("click");
    await flushPromises();
    expect(gcChainJobs).toHaveBeenCalled();
  });

  it("caps the rendered list and says how much it is holding back", async () => {
    const wrapper = await mountDrawer(
      Array.from({ length: 431 }, (_, i) =>
        job({ id: `chain-${i}`, created_at_unix_ms: NOW - i }),
      ),
    );
    expect(wrapper.findAll("[data-test='sequence-row']")).toHaveLength(200);
    expect(wrapper.get("[data-test='sequence-cap-note']").text()).toBe(
      "showing 200 of 431",
    );
  });

  it("offers exactly one action when there are no sequences", async () => {
    const wrapper = await mountDrawer([]);
    const empty = wrapper.get("[data-test='sequences-empty']");
    expect(empty.text()).toContain("No sequences yet");
    expect(empty.findAll("button")).toHaveLength(1);
    await empty.get("button").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
    expect(routerPush).toHaveBeenCalledWith("/create");
  });
});
