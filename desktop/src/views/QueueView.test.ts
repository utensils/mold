import { describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import QueueView from "./QueueView.vue";
import { useChainJobsStore } from "../stores/chainJobs";
import { useConnectionStore } from "../stores/connection";
import { isSeparator, useContextMenuStore } from "../stores/contextMenu";
import { useGenerationStore } from "../stores/generation";
import { useHostsStore } from "../stores/hosts";

const stub = { template: "<div />" };
let router: Router;

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

  it("stops a running clip from its row and from Stop everything", async () => {
    const wrapper = await mountView();
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: [
        {
          id: "job-1",
          state: "running",
          model: "ltx-video",
          stage_count: 3,
          current_stage: 1,
          created_at_unix_ms: Date.now(),
          updated_at_unix_ms: Date.now(),
          error: null,
        },
      ],
      error: null,
    };
    const cancel = vi.spyOn(chains, "cancel").mockResolvedValue(undefined as never);
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-sequence']");
    expect(row.text()).toContain("Making scene 2 of 3");
    await row.get("[data-test='queue-row-menu']").trigger("click");
    expect(menuLabels()).toEqual(["Open", "Stop"]);

    await wrapper.get("[data-test='queue-stop-all']").trigger("click");
    await flushPromises();
    expect(cancel).toHaveBeenCalledWith("local", "job-1");
  });
});
