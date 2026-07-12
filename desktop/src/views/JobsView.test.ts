import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const apiJsonTo = vi.fn();
const apiFetchTo = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetchTo: (...a: unknown[]) => apiFetchTo(...a),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../lib/notify", () => ({ notifyGenerated: vi.fn(), notifyGenerationFailed: vi.fn() }));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { saveOutputBytes: vi.fn() },
}));

import JobsView from "./JobsView.vue";
import { useConnectionStore } from "../stores/connection";

const stub = { template: "<div />" };

function installApi() {
  apiJsonTo.mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/queue") {
      return Promise.resolve({
        entries: [
          {
            id: "srv-1",
            model: "flux2-klein",
            state: "queued",
            started_at_unix_ms: 1,
            position: 0,
          },
        ],
      });
    }
    if (path === "/api/status") return Promise.resolve({ version: "x", queue_paused: false });
    if (path === "/api/capabilities") {
      return Promise.resolve({ queue: { can_pause: true, can_cancel_all: true } });
    }
    return Promise.reject(new Error(`unexpected ${path}`));
  });
}

async function mountView() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/generate", component: stub },
    ],
  });
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const wrapper = mount(JobsView, {
    global: { plugins: [pinia, router], stubs: { DevelopCanvas: stub } },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  installApi();
});

describe("JobsView", () => {
  it("renders each host's queue with its entries", async () => {
    const wrapper = await mountView();
    expect(wrapper.findAll("[data-test='host-queue']")).toHaveLength(1);
    const row = wrapper.get("[data-test='queue-row']");
    expect(row.text()).toContain("flux2-klein");
    expect(row.text()).toContain("QUEUED");
    expect(row.text()).toContain("OTHER CLIENT");
  });

  it("pauses the queue via the host's control endpoint", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='pause-toggle']").trigger("click");
    await flushPromises();
    const call = apiFetchTo.mock.calls.find((c) => c[1] === "/api/queue/pause");
    expect(call).toBeDefined();
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Resume");
    expect(wrapper.find("[data-test='paused-chip']").exists()).toBe(true);
  });

  it("cancel-all requires a second confirming click", async () => {
    const wrapper = await mountView();
    const btn = wrapper.get("[data-test='cancel-all']");
    await btn.trigger("click");
    expect(
      apiFetchTo.mock.calls.some((c) => c[1] === "/api/queue" && c[2]?.method === "DELETE"),
    ).toBe(false);
    expect(btn.text()).toContain("Cancel all?");
    await btn.trigger("click");
    await flushPromises();
    expect(
      apiFetchTo.mock.calls.some((c) => c[1] === "/api/queue" && c[2]?.method === "DELETE"),
    ).toBe(true);
  });

  it("cancels another client's queued job against its host", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='cancel-entry']").trigger("click");
    await flushPromises();
    expect(apiFetchTo.mock.calls.some((c) => c[1] === "/api/queue/srv-1")).toBe(true);
  });
});
