import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({
  owner: vi.fn(),
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
