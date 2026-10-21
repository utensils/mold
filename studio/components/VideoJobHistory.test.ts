import { flushPromises, mount } from "@vue/test-utils";
import { computed, reactive } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ caps: vi.fn(), list: vi.fn() }));
const initialHosts = [
  {
    id: "offline",
    label: "Offline",
    instanceId: "old",
    ready: true,
    target: { baseUrl: "http://offline" },
  },
  {
    id: "plato",
    label: "Plato",
    instanceId: "install",
    ready: true,
    target: { baseUrl: "http://plato", apiKey: "private" },
  },
];
const hosts = reactive(structuredClone(initialHosts));
vi.mock("../api/client", () => ({ apiJsonTo: api.caps }));
vi.mock("../api/videoJobList", () => ({ listVideoJobs: api.list }));
vi.mock("../composables/useHeldQueueTransfer", () => ({
  useHeldQueueTransfer: () => ({ hosts: computed(() => hosts) }),
}));
import VideoJobHistory from "./VideoJobHistory.vue";
import { saveVideoJobOperation } from "../lib/videoJobRecovery";
beforeEach(() => {
  localStorage.clear();
  hosts.splice(0, hosts.length, ...structuredClone(initialHosts));
  vi.clearAllMocks();
  api.caps.mockImplementation(async (target) => {
    if (target.baseUrl.includes("offline"))
      throw new Error("Offline unavailable");
    return { video_jobs: { supported: true, literal_prompts: true } };
  });
  api.list.mockResolvedValue({
    jobs: [
      {
        id: "failed",
        state: "failed",
        completed_clips: "2",
        plan: { clip_count: "7" },
      },
    ],
    has_more: false,
    next_cursor: null,
  });
});
it("keeps another machine's recoverable jobs reachable when one machine fails", async () => {
  const wrapper = mount(VideoJobHistory, {
    global: { stubs: { VideoJobControl: true } },
  });
  await flushPromises();
  expect(wrapper.text()).toContain("Plato · failed");
  expect(wrapper.text()).toContain("Offline");
  wrapper.unmount();
});
it("shows an unanswered saved attempt even when its machine is absent and suppresses existing parent duplicates", async () => {
  saveVideoJobOperation({
    kind: "video_job",
    operationId: "known",
    hostId: "removed",
    instanceId: "gone",
    baseUrl: "http://gone",
    createdAt: 1,
  });
  api.caps.mockResolvedValue({
    video_jobs: { supported: true, literal_prompts: true },
  });
  const wrapper = mount(VideoJobHistory, {
    props: { existingKeys: ["plato:install:failed", "offline:old:failed"] },
    global: { stubs: { VideoJobControl: true } },
  });
  await flushPromises();
  expect(wrapper.text()).toContain(
    "Unanswered video attempt · removed · known",
  );
  expect(wrapper.text()).not.toContain("Plato · failed");
  wrapper.unmount();
});

function mountedHistory() {
  return mount(VideoJobHistory, {
    global: { stubs: { VideoJobControl: true } },
  });
}
it("discovers saved parents when the mounted machine becomes ready", async () => {
  hosts.splice(0, 1);
  hosts[0]!.ready = false;
  const wrapper = mountedHistory();
  await flushPromises();
  expect(api.list).not.toHaveBeenCalled();
  hosts[0]!.ready = true;
  await flushPromises();
  expect(wrapper.text()).toContain("Plato · failed");
  expect(api.list).toHaveBeenCalledTimes(1);
  wrapper.unmount();
});
it.each(["apiKey", "baseUrl", "instanceId", "connectionMode"])(
  "rebinds saved rows after %s changes without polling on telemetry",
  async (field) => {
    hosts.splice(0, 1);
    const wrapper = mountedHistory();
    await flushPromises();
    const host = hosts[0]!;
    if (field === "instanceId") host.instanceId = "replacement";
    else
      Object.assign(host.target, {
        [field]: field === "connectionMode" ? "regular" : "replacement",
      });
    await flushPromises();
    expect(api.list).toHaveBeenCalledTimes(2);
    const captured = wrapper
      .findComponent({ name: "VideoJobControl" })
      .props("owner");
    expect(captured.instanceId).toBe(host.instanceId);
    expect(captured.target).toEqual(host.target);
    Object.assign(host, { uptime: 999, busy: true });
    await flushPromises();
    expect(api.list).toHaveBeenCalledTimes(2);
    wrapper.unmount();
  },
);
it("supersedes an in-flight read and never lets its late result replace the new owner", async () => {
  hosts.splice(0, 1);
  let finish!: (value: unknown) => void;
  api.list.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const wrapper = mountedHistory();
  await flushPromises();
  hosts[0]!.target.apiKey = "replacement";
  hosts[0]!.target.baseUrl = "http://replacement";
  await flushPromises();
  expect(api.list).toHaveBeenCalledTimes(2);
  expect(wrapper.text()).toContain("Plato · failed");
  finish({
    jobs: [{ id: "stale", state: "paused", plan: { clip_count: "12" } }],
    has_more: false,
    next_cursor: null,
  });
  await flushPromises();
  expect(wrapper.text()).not.toContain("paused");
  expect(
    wrapper.findComponent({ name: "VideoJobControl" }).props("owner").target
      .apiKey,
  ).toBe("replacement");
  wrapper.unmount();
});
it("does not rediscover unsupported older servers on ordinary telemetry", async () => {
  hosts.splice(0, 1);
  api.caps.mockResolvedValue({});
  const wrapper = mountedHistory();
  await flushPromises();
  Object.assign(hosts[0]!, { uptime: 100 });
  await flushPromises();
  expect(api.caps).toHaveBeenCalledTimes(1);
  expect(api.list).not.toHaveBeenCalled();
  wrapper.unmount();
});

it("reconnects while capabilities are pending and ignores the old refusal", async () => {
  hosts.splice(0, 1);
  let finish!: (value: unknown) => void;
  api.caps.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const wrapper = mountedHistory();
  await flushPromises();
  hosts[0]!.ready = false;
  hosts[0]!.target.apiKey = "reconnected";
  hosts[0]!.ready = true;
  await flushPromises();
  expect(api.list).toHaveBeenCalledTimes(1);
  finish({});
  await flushPromises();
  expect(wrapper.text()).toContain("Plato · failed");
  expect(wrapper.get("button").attributes("disabled")).toBeUndefined();
  expect(
    api.caps.mock.calls.every((call) => call[1] === "/api/capabilities"),
  ).toBe(true);
  wrapper.unmount();
});
it("does not publish a stale load-more page or clear the newer discovery's busy state", async () => {
  hosts.splice(0, 1);
  api.list.mockResolvedValueOnce({
    jobs: [{ id: "first", state: "paused", plan: { clip_count: "12" } }],
    has_more: true,
    next_cursor: "older",
  });
  const wrapper = mountedHistory();
  await flushPromises();
  let finishMore!: (value: unknown) => void;
  api.list.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finishMore = resolve;
      }),
  );
  await wrapper
    .findAll("button")
    .find((button) => button.text().includes("Load more"))!
    .trigger("click");
  let finishNew!: (value: unknown) => void;
  api.list.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finishNew = resolve;
      }),
  );
  hosts[0]!.instanceId = "new-installation";
  await flushPromises();
  finishMore({
    jobs: [{ id: "stale", state: "interrupted", plan: { clip_count: "9" } }],
    has_more: false,
    next_cursor: null,
  });
  await flushPromises();
  expect(wrapper.text()).not.toContain("interrupted");
  expect(wrapper.get("button").attributes("disabled")).toBeDefined();
  finishNew({
    jobs: [{ id: "new", state: "failed", plan: { clip_count: "7" } }],
    has_more: false,
    next_cursor: null,
  });
  await flushPromises();
  expect(wrapper.text()).toContain("Plato · failed");
  expect(wrapper.get("button").attributes("disabled")).toBeUndefined();
  wrapper.unmount();
});
