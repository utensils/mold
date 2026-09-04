import { describe, expect, it } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import StatusBar from "./StatusBar.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore } from "../../stores/generation";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useHostsStore } from "../../stores/hosts";
import { useJobsStore } from "../../stores/jobs";
import { shortcutLabel } from "../../lib/platform";

const stub = { template: "<div />" };
let router: Router;

async function mountBar() {
  router = createRouter({
    history: createMemoryHistory(),
    routes: ["/create", "/queue", "/models", "/machines", "/machines/:id"].map((path) => ({
      path,
      component: stub,
    })),
  });
  router.push("/create");
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const wrapper = mount(StatusBar, { global: { plugins: [pinia, router] } });
  await flushPromises();
  return wrapper;
}

function connectLocal() {
  const connection = useConnectionStore();
  connection.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  connection.status = "ready";
}

/** The bar's key hints as pairs, in the order the mock draws them. */
function hints(wrapper: VueWrapper) {
  const spans = wrapper.findAll("span.keycap");
  return spans.map((keycap) => [keycap.text(), keycap.element.nextElementSibling?.textContent]);
}

describe("StatusBar", () => {
  it("says which machine, and why it is not answering", async () => {
    const wrapper = await mountBar();
    expect(wrapper.get("[data-test='status-machine']").text()).toBe("no machine");

    connectLocal();
    await flushPromises();
    const label = useHostsStore().primaryHost!.label;
    expect(wrapper.get("[data-test='status-machine']").text()).toBe(label);

    useConnectionStore().status = "error";
    await flushPromises();
    expect(wrapper.get("[data-test='status-machine']").text()).toBe(`${label} · offline`);
  });

  it("states how deep the queue is, and says so plainly when it is paused", async () => {
    const wrapper = await mountBar();
    connectLocal();
    await flushPromises();
    expect(wrapper.get("[data-test='status-queue']").text()).toBe("nothing waiting");

    useGenerationStore().jobs = [
      { clientId: 1, model: "flux-dev:q8", prompt: "one", status: "denoising", step: 1, total: 28 },
      { clientId: 2, model: "flux-dev:q8", prompt: "two", status: "queued" },
    ] as never;
    await flushPromises();
    expect(wrapper.get("[data-test='status-queue']").text()).toBe("1 image being made · 1 waiting");

    useJobsStore().queues.local = {
      entries: [],
      caps: { canPause: true },
      paused: true,
    } as never;
    await flushPromises();
    expect(wrapper.get("[data-test='status-queue']").text()).toBe("queue paused · 1 waiting");
  });

  it("reads out graphics and system memory only where the host reports them", async () => {
    const wrapper = await mountBar();
    connectLocal();
    await flushPromises();
    expect(wrapper.find("[data-test='status-vram']").exists()).toBe(false);
    expect(wrapper.find("[data-test='status-ram']").exists()).toBe(false);

    useHostStatusStore().snapshot = {
      gpus: [{ ordinal: 0, vram_used: 8_000_000_000, vram_total: 24_000_000_000 }],
      system_ram: { used: 32_000_000_000, total: 64_000_000_000 },
    } as never;
    await flushPromises();
    expect(wrapper.get("[data-test='status-vram']").text()).toBe("vram 8.0 GB / 24.0 GB");
    expect(wrapper.get("[data-test='status-ram']").text()).toBe("ram 32.0 GB/64.0 GB");
  });

  it("advertises the three chords the shell really binds", async () => {
    const wrapper = await mountBar();
    expect(hints(wrapper)).toEqual([
      [shortcutLabel("↩"), "Generate"],
      [shortcutLabel("K"), "Search"],
      ["Space", "Pause queue"],
    ]);
  });
});
