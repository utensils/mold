import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, getActivePinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import QueueView from "./QueueView.vue";
import {
  __resetQueueCommandState,
  useQueueCommands,
  type QueueCommands,
} from "../composables/useQueueCommands";
import { useConnectionStore } from "../stores/connection";
import { isSeparator, useContextMenuStore } from "../stores/contextMenu";
import { useGalleryStore } from "../stores/gallery";
import { useGenerationStore } from "../stores/generation";
import { useHostsStore } from "../stores/hosts";
import { useJobsStore, type HostQueueSnapshot } from "../stores/jobs";

const stub = { template: "<div />" };
let router: Router;

// One app, one Stop-everything dialog: that state is module-scoped and pinia
// does not clear it between tests.
beforeEach(() => __resetQueueCommandState());

async function mountView() {
  router = createRouter({
    history: createMemoryHistory(),
    routes: ["/create", "/queue", "/library"].map((path) => ({ path, component: stub })),
  });
  router.push("/queue");
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  connection.status = "ready";
  useHostsStore().telemetry.local = { queueDepth: 0, queueCapacity: 8, version: null };
  const wrapper = mount(QueueView, {
    global: { plugins: [pinia, router], stubs: { AuthedMedia: stub } },
  });
  await flushPromises();
  return wrapper;
}

function menuLabels() {
  return useContextMenuStore().entries.flatMap((e) => (isSeparator(e) ? [] : [e.label]));
}

/** A handle on the same shared queue commands the view holds — the Stop
 *  everything dialog is module-scoped and rendered once, in the shell. */
function queueCommands(): QueueCommands {
  let api!: QueueCommands;
  mount(
    defineComponent({
      setup() {
        api = useQueueCommands();
        return () => null;
      },
    }),
    { global: { plugins: [getActivePinia() as never, router] } },
  );
  return api;
}

describe("QueueView", () => {
  it("explains an empty queue in the lexicon", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='queue-headline']").text()).toBe("0 being made · 0 waiting");
    expect(wrapper.get("[data-test='queue-empty']").text()).toContain(
      "Describe a picture in New image",
    );
    expect(wrapper.get("[data-test='queue-stop-all']").attributes("disabled")).toBeDefined();
  });

  it("lists every print as a sentence with its style in mono, and opens one from the keyboard", async () => {
    const wrapper = await mountView();
    useGenerationStore().jobs = [
      {
        clientId: 1,
        id: "srv-1",
        model: "flux-dev:q8",
        prompt: "a brass teapot",
        status: "denoising",
        step: 18,
        total: 28,
      } as never,
    ];
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    expect(row.text()).toContain("a brass teapot");
    expect(row.text()).toContain("Adding detail — pass 18 of 28");
    expect(row.attributes("role")).toBe("button");
    expect(row.attributes("tabindex")).toBe("0");
    expect(wrapper.get("[data-test='queue-headline']").text()).toBe("1 being made · 0 waiting");

    await row.trigger("keydown.enter");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("calls a held print Needs a download first and offers Retry now", async () => {
    const wrapper = await mountView();
    const generation = useGenerationStore();
    generation.jobs = [
      {
        clientId: 2,
        id: "srv-2",
        model: "flux-dev:q8",
        prompt: "held",
        status: "error",
        error: "model not found",
        holdCode: "MODEL_NOT_FOUND",
        holdError: "flux-dev:q8 is not on this machine",
        retryable: true,
        retrying: false,
      } as never,
    ];
    const retry = vi.spyOn(generation, "retryHeld").mockResolvedValue();
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    expect(row.text()).toContain("Needs a download first");
    expect(row.text()).not.toContain("Failed");

    await row.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).toContain("Retry now");
    const entry = useContextMenuStore().entries.find(
      (e) => !isSeparator(e) && e.label === "Retry now",
    );
    if (!entry || isSeparator(entry)) throw new Error("no retry entry");
    entry.action?.();
    await flushPromises();
    expect(retry).toHaveBeenCalledWith(2);
  });

  // A long clip the host chains and stitches is ONE print row, never a row
  // of its own — so Stop everything reaches it exactly as it reaches a still.
  it("stops a running clip from Stop everything, as one print row", async () => {
    const wrapper = await mountView();
    const generation = useGenerationStore();
    generation.jobs = [
      { clientId: 1, id: "job-1", model: "ltx-video", prompt: "a long clip", status: "denoising" },
    ] as never;
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(true);
    await flushPromises();

    expect(wrapper.find("[data-test='queue-row-sequence']").exists()).toBe(false);
    expect(wrapper.get("[data-test='queue-row-print']").text()).toContain("a long clip");

    // Stop everything is the widest destructive action in the app: it asks
    // first, from every door, and the one dialog lives in the shell.
    await wrapper.get("[data-test='queue-stop-all']").trigger("click");
    await flushPromises();
    expect(cancel).not.toHaveBeenCalled();
    const commands = queueCommands();
    expect(commands.stopEverythingOpen.value).toBe(true);
    expect(commands.stopEverythingSummary.value).toContain("Anything part-finished is lost.");

    await commands.confirmStopEverything();
    await flushPromises();
    expect(cancel).toHaveBeenCalledWith(1);
    expect(commands.stopEverythingOpen.value).toBe(false);
  });

  it("closes the Stop everything confirm without stopping anything", async () => {
    const wrapper = await mountView();
    const generation = useGenerationStore();
    generation.jobs = [
      { clientId: 1, id: "job-1", model: "ltx-video", prompt: "a long clip", status: "denoising" },
    ] as never;
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(true);
    await flushPromises();

    await wrapper.get("[data-test='queue-stop-all']").trigger("click");
    const commands = queueCommands();
    commands.cancelStopEverything();
    await flushPromises();
    expect(commands.stopEverythingOpen.value).toBe(false);
    expect(cancel).not.toHaveBeenCalled();
  });
  it("lets a waiting print jump the line where its host can reorder", async () => {
    const wrapper = await mountView();
    useGenerationStore().jobs = [
      { clientId: 1, id: "srv-1", model: "flux-dev:q8", prompt: "first", status: "queued" },
      { clientId: 2, id: "srv-2", model: "flux-dev:q8", prompt: "second", status: "queued" },
    ] as never;
    const jobs = useJobsStore();
    const snapshot = {
      hostId: "local",
      entries: [
        { id: "srv-1", state: "queued" },
        { id: "srv-2", state: "queued" },
      ],
      paused: false,
      caps: { canPause: false, canCancelAll: false, canReorder: true },
      gpuOrdinals: [],
      error: null,
    } as unknown as HostQueueSnapshot;
    jobs.queues.local = snapshot;
    const reorder = vi.spyOn(jobs, "reorderQueued").mockResolvedValue(true);
    await flushPromises();

    const rows = wrapper.findAll("[data-test='queue-row-print']");
    const second = rows.find((row) => row.text().includes("second"))!;
    await second.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels().slice(0, 3)).toEqual(["Jump the line", "Move earlier", "Move later"]);
    const jump = useContextMenuStore().entries.find(
      (e) => !isSeparator(e) && e.label === "Jump the line",
    );
    if (!jump || isSeparator(jump)) throw new Error("no jump entry");
    jump.action?.();
    await flushPromises();
    expect(reorder).toHaveBeenCalledWith("local", "srv-2", 0);

    // Without the capability the entries never appear.
    snapshot.caps = { canPause: false, canCancelAll: false, canReorder: false };
    await flushPromises();
    await second.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).not.toContain("Jump the line");
  });

  /**
   * The explainer says "Drag a row to reorder". Nothing here was draggable —
   * reordering lived only behind the row's ⋯ — so the sentence instructed an
   * interaction the view did not implement.
   */
  it("reorders by dragging a waiting row onto the slot it should take", async () => {
    const wrapper = await mountView();
    useGenerationStore().jobs = [
      { clientId: 1, id: "srv-1", model: "flux-dev:q8", prompt: "first", status: "queued" },
      { clientId: 2, id: "srv-2", model: "flux-dev:q8", prompt: "second", status: "queued" },
    ] as never;
    const jobs = useJobsStore();
    const snapshot = {
      hostId: "local",
      entries: [
        { id: "srv-1", state: "queued" },
        { id: "srv-2", state: "queued" },
      ],
      paused: false,
      caps: { canPause: false, canCancelAll: false, canReorder: true },
      gpuOrdinals: [],
      error: null,
    } as unknown as HostQueueSnapshot;
    jobs.queues.local = snapshot;
    const reorder = vi.spyOn(jobs, "reorderQueued").mockResolvedValue(true);
    await flushPromises();

    const rows = wrapper.findAll("[data-test='queue-row-print']");
    const first = rows.find((row) => row.text().includes("first"))!;
    const second = rows.find((row) => row.text().includes("second"))!;
    expect(second.attributes("draggable")).toBe("true");

    await second.trigger("dragstart", { dataTransfer: { setData: vi.fn() } });
    await first.trigger("drop");
    await flushPromises();
    expect(reorder).toHaveBeenCalledWith("local", "srv-2", 0);
  });

  it("offers no drag at all where the host cannot reorder", async () => {
    const wrapper = await mountView();
    useGenerationStore().jobs = [
      { clientId: 1, id: "srv-1", model: "flux-dev:q8", prompt: "only", status: "queued" },
    ] as never;
    useJobsStore().queues.local = {
      hostId: "local",
      entries: [{ id: "srv-1", state: "queued" }],
      paused: false,
      caps: { canPause: false, canCancelAll: false, canReorder: false },
      gpuOrdinals: [],
      error: null,
    } as unknown as HostQueueSnapshot;
    await flushPromises();
    expect(wrapper.get("[data-test='queue-row-print']").attributes("draggable")).toBe("false");
  });

  it("marks the control the explainer names, and shouts its column header", async () => {
    const wrapper = await mountView();
    // "Jump the line" names a control; unmarked it reads as prose (mock:598).
    expect(wrapper.get("[data-test='queue-jump-name']").text()).toBe("Jump the line");
    // Every other ms-group-label in the app adds `uppercase`; the kit leaves
    // the case to the caller, so this header rendered in sentence case.
    expect(wrapper.get("[data-test='queue-columns']").classes()).toContain("uppercase");
  });

  /**
   * `hasFinished` read this client's job list while the table renders
   * `queue.rows`. The button that says "Clear finished" must be enabled by
   * exactly the rows it is going to clear, on every machine.
   */
  it("enables Clear finished from the rows the table actually shows", async () => {
    const wrapper = await mountView();
    const generation = useGenerationStore();
    expect(wrapper.get("[data-test='queue-clear-finished']").attributes("disabled")).toBeDefined();

    generation.jobs = [
      {
        clientId: 1,
        id: "srv-1",
        hostId: "plato",
        model: "flux-dev:q8",
        prompt: "done elsewhere",
        status: "complete",
      },
    ] as never;
    await flushPromises();
    const clear = wrapper.get("[data-test='queue-clear-finished']");
    expect(clear.attributes("disabled")).toBeUndefined();

    const prune = vi.spyOn(generation, "prune").mockImplementation(() => undefined);
    await clear.trigger("click");
    expect(prune).toHaveBeenCalledWith(0);
  });

  it("pauses and resumes one waiting print only where the host offers it", async () => {
    const wrapper = await mountView();
    useGenerationStore().jobs = [
      { clientId: 1, id: "srv-1", model: "flux-dev:q8", prompt: "waiting", status: "queued" },
    ] as never;
    const jobs = useJobsStore();
    const snapshot = {
      hostId: "local",
      entries: [{ id: "srv-1", state: "queued" }],
      paused: false,
      caps: { canPause: false, canPauseJob: false, canCancelAll: false, canReorder: false },
      gpuOrdinals: [],
      error: null,
    } as unknown as HostQueueSnapshot;
    jobs.queues.local = snapshot;
    const setPaused = vi.spyOn(jobs, "setJobPaused").mockResolvedValue(undefined);
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    await row.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).not.toContain("Pause");

    snapshot.caps = {
      canPause: false,
      canPauseJob: true,
      canCancelAll: false,
      canReorder: false,
    } as never;
    await flushPromises();
    await row.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).toContain("Pause");
    const pause = useContextMenuStore().entries.find(
      (entry) => !isSeparator(entry) && entry.label === "Pause",
    );
    if (!pause || isSeparator(pause)) throw new Error("no pause entry");
    pause.action?.();
    await flushPromises();
    expect(setPaused).toHaveBeenCalledWith("local", "srv-1", true);

    snapshot.entries[0]!.state = "paused";
    await flushPromises();
    await row.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).toContain("Resume");
  });

  /**
   * The ETA is a countdown against a ticking clock, so the assertion needs a
   * fixed one: rounding `(finish - Date.now()) / 1000` at render turns 45 s
   * into 44 s after half a second of real drift between setup and assertion.
   */
  it("counts today's prints from the gallery and states the fleet's time left", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-04T12:00:00Z"));
    const wrapper = await mountView();
    const gallery = useGalleryStore();
    const now = Date.now();
    gallery.buckets.local = {
      items: [
        { filename: "today.png", timestamp: Math.floor(now / 1000) },
        { filename: "yesterday.png", timestamp: Math.floor(now / 1000) - 86_400 * 2 },
      ],
      error: null,
    } as never;
    const jobs = useJobsStore();
    jobs.queues.local = {
      hostId: "local",
      entries: [],
      paused: false,
      caps: null,
      gpuOrdinals: [],
      plan: { work_items: [{ work_id: "srv-1", estimated_finish_unix_ms: now + 45_000 }] },
      error: null,
    } as unknown as HostQueueSnapshot;
    await flushPromises();

    const stats = wrapper.findAll("[data-test='queue-stat']");
    expect(stats[2]!.text()).toContain("Done today");
    expect(stats[2]!.text()).toContain("all saved to My images");
    expect(stats[2]!.text()).toContain("1");
    expect(wrapper.get("[data-test='queue-total-eta']").text()).toBe("about 45s left in total");

    // The clock is the composable's own 1s tick, so the countdown moves
    // between queue refreshes instead of freezing at the value it was read at.
    await vi.advanceTimersByTimeAsync(10_000);
    await flushPromises();
    expect(wrapper.get("[data-test='queue-total-eta']").text()).toBe("about 35s left in total");
    vi.useRealTimers();
  });
});
