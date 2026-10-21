import { flushPromises, mount } from "@vue/test-utils";
import { computed } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ caps: vi.fn(), list: vi.fn() }));
const hosts = [
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
vi.mock("../api/client", () => ({ apiJsonTo: api.caps }));
vi.mock("../api/videoJobList", () => ({ listVideoJobs: api.list }));
vi.mock("../composables/useHeldQueueTransfer", () => ({
  useHeldQueueTransfer: () => ({ hosts: computed(() => hosts) }),
}));
import VideoJobHistory from "./VideoJobHistory.vue";
import { saveVideoJobOperation } from "../lib/videoJobRecovery";
beforeEach(() => {
  localStorage.clear();
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
