import { ref } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({
  owner: vi.fn(),
  library: vi.fn(),
  get: vi.fn(),
  mutate: vi.fn(),
  validate: vi.fn(),
  admit: vi.fn(),
}));
vi.mock("../api/videoJobs", () => ({
  assertVideoJobOwner: api.owner,
  getVideoJob: api.get,
  mutateVideoJob: api.mutate,
  validateVideoJob: api.validate,
  admitVideoJob: api.admit,
}));
vi.mock("../api/videoLibraryOutput", () => ({
  videoLibraryOutputExists: api.library,
}));
import { VIDEO_LIBRARY_NAVIGATION } from "../lib/videoLibraryNavigation";
import VideoJobControl from "./VideoJobControl.vue";
import { ApiError } from "../api/client";
import {
  saveVideoJobOperation,
  readVideoJobOperations,
} from "../lib/videoJobRecovery";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const operation = {
  kind: "video_job" as const,
  operationId: "known",
  hostId: "plato",
  instanceId: "installation",
  baseUrl: "http://plato",
  createdAt: 1,
};
const wrappers: ReturnType<typeof mount>[] = [];
beforeEach(() => {
  localStorage.clear();
  Object.values(api).forEach((mock) => mock.mockReset());
  api.owner.mockResolvedValue(undefined);
  api.library.mockResolvedValue(true);
  api.get.mockResolvedValue(null);
});
afterEach(() => {
  wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
  document.body.innerHTML = "";
});
it("makes an unanswered saved attempt reachable and persists lost Stop intent", async () => {
  saveVideoJobOperation(operation);
  const wrapper = mount(VideoJobControl, {
    props: {
      owner,
      jobId: "known",
      operation,
      isCurrent: () => true,
      autoOpen: true,
    },
    attachTo: document.body,
  });
  wrappers.push(wrapper);
  await flushPromises();
  const stop = Array.from(document.body.querySelectorAll("button")).find(
    (button) => button.textContent?.trim() === "Stop video",
  )!;
  expect(stop).toBeTruthy();
  api.mutate.mockRejectedValue(new Error("lost Stop reply"));
  stop.click();
  await flushPromises();
  expect(readVideoJobOperations()[0]?.cancelRequested).toBe(true);
  expect(document.body.textContent).toContain("lost Stop reply");
  api.mutate.mockResolvedValue(undefined);
  api.get.mockRejectedValue(
    new ApiError("cancelled", 410, { code: "VIDEO_OPERATION_CANCELLED" }),
  );
  Array.from(document.body.querySelectorAll("button"))
    .find((button) => button.textContent?.trim() === "Check queue")!
    .click();
  await flushPromises();
  expect(readVideoJobOperations()).toEqual([]);
  expect(document.body.textContent).toContain("Cancelled before queueing");
});
it("does not publish a late answer after the captured installation changed", async () => {
  let finish!: (value: unknown) => void,
    current = true;
  api.get.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  saveVideoJobOperation(operation);
  const wrapper = mount(VideoJobControl, {
    props: {
      owner,
      jobId: "known",
      operation,
      isCurrent: () => current,
      autoOpen: true,
    },
    attachTo: document.body,
  });
  wrappers.push(wrapper);
  await flushPromises();
  current = false;
  finish({ id: "known", state: "completed", output_filename: "other.mp4" });
  await flushPromises();
  expect(document.body.textContent).not.toContain("Saved to Library");
  expect(readVideoJobOperations()).toEqual([operation]);
});
it("does not close and reread an open job for an equivalent refreshed owner object", async () => {
  const wrapper = mount(VideoJobControl, {
    props: { owner, jobId: "known", isCurrent: () => true, autoOpen: true },
    attachTo: document.body,
  });
  wrappers.push(wrapper);
  await flushPromises();
  await wrapper.setProps({ owner: { ...owner, target: { ...owner.target } } });
  await flushPromises();
  expect(api.get).toHaveBeenCalledTimes(1);
});
it("keeps historical stopped videos available for a fresh attempt and server discard", async () => {
  api.get.mockResolvedValue({
    id: "known",
    state: "cancelled",
    completed_clips: "1",
    plan: { clip_count: "2", total_duration_ms: "6000" },
    can_cancel: false,
    can_resume: false,
    can_delete: true,
  });
  const wrapper = mount(VideoJobControl, {
    props: { owner, jobId: "known", isCurrent: () => true, autoOpen: true },
    global: { stubs: { ParentVideoRestart: true } },
    attachTo: document.body,
  });
  wrappers.push(wrapper);
  await flushPromises();
  expect(wrapper.findComponent({ name: "ParentVideoRestart" }).exists()).toBe(
    true,
  );
  expect(document.body.textContent).toContain("Delete recovery data");
  expect(document.body.textContent).not.toContain("Resume video");
});

it("disables a captured stale Details owner without sending any mutation", async () => {
  const connected = ref(true);
  const wrapper = mount(VideoJobControl, {
    props: { owner, jobId: "known", isCurrent: () => connected.value },
  });
  wrappers.push(wrapper);
  connected.value = false;
  await flushPromises();
  expect(wrapper.get("button").attributes("disabled")).toBeDefined();
  await wrapper.get("button").trigger("click");
  expect(api.get).not.toHaveBeenCalled();
  expect(api.mutate).not.toHaveBeenCalled();
  expect(api.admit).not.toHaveBeenCalled();
});

const completed = {
  id: "known",
  state: "completed",
  completed_clips: "2",
  output_filename: "movie.mp4",
  plan: { clip_count: "2", total_duration_ms: "6000" },
  can_cancel: false,
  can_resume: false,
  can_delete: true,
};
function outputControl(navigate = vi.fn()) {
  const wrapper = mount(VideoJobControl, {
    props: { owner, jobId: "known", isCurrent: () => true, autoOpen: true },
    global: { provide: { [VIDEO_LIBRARY_NAVIGATION as symbol]: navigate } },
    attachTo: document.body,
  });
  wrappers.push(wrapper);
  return { wrapper, navigate };
}
it("retires an acknowledged missing row instead of offering Stop for a deleted job", async () => {
  const { wrapper } = outputControl();
  await flushPromises();
  expect(wrapper.emitted("removed")).toEqual([["known"]]);
  expect(document.body.textContent).not.toContain("Stop video");
});
it("hides an exact deleted Library copy and preserves recovery-data deletion", async () => {
  api.get.mockResolvedValue(completed);
  api.library.mockResolvedValue(false);
  outputControl();
  await flushPromises();
  expect(document.body.textContent).toContain(
    "no longer in this machine's Library",
  );
  expect(document.body.textContent).not.toContain("View in Library");
  expect(document.body.textContent).toContain("Delete recovery data");
});
it("rechecks a Library copy at click time instead of opening a stale output link", async () => {
  api.get.mockResolvedValue(completed);
  api.library.mockResolvedValueOnce(true).mockResolvedValue(false);
  const { navigate } = outputControl();
  await flushPromises();
  Array.from(document.body.querySelectorAll("button"))
    .find((button) => button.textContent?.trim() === "View in Library")!
    .click();
  await flushPromises();
  expect(navigate).not.toHaveBeenCalled();
  expect(document.body.textContent).toContain(
    "no longer in this machine's Library",
  );
});
it("emits one removal after Delete and lets the creating owner discard its cached result", async () => {
  api.get.mockResolvedValue(completed);
  const { wrapper } = outputControl();
  await flushPromises();
  Array.from(document.body.querySelectorAll("button"))
    .find((button) => button.textContent?.trim() === "Delete recovery data…")!
    .click();
  await flushPromises();
  Array.from(document.body.querySelectorAll("button"))
    .find((button) => button.textContent?.trim() === "Delete recovery data")!
    .click();
  await flushPromises();
  expect(wrapper.emitted("removed")).toEqual([["known"]]);
  expect(document.body.textContent).not.toContain("Saved to Library");
});

it("preserves completed recovery data when the Library connection is unavailable", async () => {
  api.get.mockResolvedValue(completed);
  api.library.mockRejectedValue(new Error("offline"));
  outputControl();
  await flushPromises();
  expect(document.body.textContent).toContain(
    "Could not check the saved video: offline",
  );
  expect(document.body.textContent).not.toContain(
    "no longer in this machine's Library",
  );
  expect(document.body.textContent).toContain("Delete recovery data");
});
