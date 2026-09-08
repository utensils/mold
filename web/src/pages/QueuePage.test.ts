import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { Job } from "../composables/useGenerateStream";
import type { FleetActiveWork } from "@studio/api/activity";
import QueuePage from "./QueuePage.vue";
import { takeLocalJobHandoff } from "../composables/useGenerationHandoff";

const state = vi.hoisted(() => ({
  jobs: [] as Job[],
  rows: [] as FleetActiveWork[],
  cancel: vi.fn(),
  retry: vi.fn(),
  remove: vi.fn(),
  push: vi.fn(),
  openShared: vi.fn(),
  submit: vi.fn(),
}));
vi.mock("vue-router", () => ({ useRouter: () => ({ push: state.push }) }));
vi.mock("../composables/useGenerateStream", () => ({
  useGenerateStream: () => ({
    jobs: ref(state.jobs),
    cancel: state.cancel,
    retry: state.retry,
    remove: state.remove,
    submit: state.submit,
  }),
}));
vi.mock("../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: ref([{ id: "box", label: "Box", status: "ready" }]),
    queueStatus: ref(null),
  }),
}));
vi.mock("../composables/useLiveActivity", () => ({
  useLiveActivity: () => ({ rows: ref(state.rows) }),
}));
vi.mock("../composables/useOpenLiveWork", () => ({
  useOpenLiveWork: () => state.openShared,
}));

function job(id: string, hostId = "box"): Job {
  return {
    id,
    serverId: id,
    hostId,
    hostLabel: hostId,
    request: { model: "flux-dev", prompt: `Prompt ${id}` },
    progress: { stage: "Queued", step: null, totalSteps: null },
    state: "running",
    workStarted: false,
    startedAt: 10,
    settledAt: null,
    error: null,
    retryable: false,
  } as Job;
}
function row(id: string, hostId = "box"): FleetActiveWork {
  return {
    key: `${hostId}:generation:${id}`,
    id,
    hostId,
    hostLabel: hostId,
    kind: "generation",
    model: "flux-dev",
    phase: "running",
    can_cancel: true,
    created_at_unix_ms: 10,
    updated_at_unix_ms: 10,
    stale: false,
  } as FleetActiveWork;
}
function render() {
  return mount(QueuePage, {
    global: {
      stubs: {
        RouterLink: { props: ["to"], template: '<a :href="to"><slot /></a>' },
      },
    },
  });
}
beforeEach(() => {
  vi.clearAllMocks();
  state.jobs = [];
  state.rows = [];
  takeLocalJobHandoff();
});
describe("Queue", () => {
  it("renders every queued job and only restores one when explicitly opened", async () => {
    state.jobs = [job("one"), job("two"), job("three")];
    const wrapper = render();
    expect(wrapper.findAll('[data-test^="activity-queued-"]')).toHaveLength(3);
    expect(state.submit).not.toHaveBeenCalled();
    expect(takeLocalJobHandoff()).toBeNull();
    await wrapper.get('[data-test="activity-queued-two"]').trigger("click");
    expect(takeLocalJobHandoff()).toBe("two");
    expect(state.push).toHaveBeenCalledWith("/create");
    expect(state.submit).not.toHaveBeenCalled();
  });
  it("deduplicates by exact host and job while retaining another machine's matching ID", () => {
    state.jobs = [job("one")];
    state.rows = [row("one"), row("one", "other")];
    const wrapper = render();
    expect(wrapper.findAll('[data-test^="activity-queued-"]')).toHaveLength(1);
    expect(
      wrapper.findAll('[data-test^="live-activity-select-"]'),
    ).toHaveLength(1);
    expect(wrapper.text()).toContain("other");
  });
  it("uses the original local cancel/retry actions without opening or submitting", async () => {
    state.jobs = [
      { ...job("held"), retryable: true, holdError: "Style missing" },
    ];
    const wrapper = render();
    await wrapper.get('[data-test="activity-retry-held"]').trigger("click");
    await wrapper.get('[data-test="activity-cancel-held"]').trigger("click");
    await flushPromises();
    expect(state.retry).toHaveBeenCalledWith("held");
    expect(state.cancel).toHaveBeenCalledWith("held");
    expect(state.push).not.toHaveBeenCalled();
    expect(state.submit).not.toHaveBeenCalled();
  });
  it("keeps old failures available here after the compact Create expiry", () => {
    state.jobs = [
      { ...job("failed"), state: "error", error: "Failed", settledAt: 1 },
    ];
    expect(render().find('[data-test="activity-error-failed"]').exists()).toBe(
      true,
    );
  });
  it("offers usable navigation when empty", () => {
    const wrapper = render();
    expect(wrapper.text()).toContain("Nothing waiting here");
    expect(wrapper.get('a[href="/create"]').text()).toBe("New image");
    expect(wrapper.get('a[href="/library"]').text()).toBe("My images");
  });
});
