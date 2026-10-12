import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import QueueBatchCancelControl from "./QueueBatchCancelControl.vue";
import {
  inspectQueueBatch,
  cancelInspectedQueueBatch,
} from "../api/queueBatchCancel";
vi.mock("../api/queueBatchCancel", () => ({
  inspectQueueBatch: vi.fn(),
  cancelInspectedQueueBatch: vi.fn(),
}));
const selection = {
  instanceId: "instance",
  jobId: "job",
  batchId: "batch",
  clientBatchId: "client",
  liveCount: 3,
};
const props = {
  target: { baseUrl: "http://fixture", apiKey: null },
  instanceId: "instance",
  jobId: "job",
  online: true,
};
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(inspectQueueBatch).mockResolvedValue(selection);
});
it("offers a two-step batch cancel, reports success, and never renders it for singletons", async () => {
  const wrapper = mount(QueueBatchCancelControl, { props });
  await flushPromises();
  expect(wrapper.text()).toContain("Cancel batch (3 jobs)");
  await wrapper.get("button").trigger("click");
  expect(cancelInspectedQueueBatch).not.toHaveBeenCalled();
  expect(wrapper.text()).toContain("Cancel entire batch?");
  await wrapper.get("button").trigger("click");
  await flushPromises();
  expect(cancelInspectedQueueBatch).toHaveBeenCalledOnce();
  expect(wrapper.emitted("cancelled")).toHaveLength(1);
  wrapper.unmount();
  vi.mocked(inspectQueueBatch).mockResolvedValue(null);
  const singleton = mount(QueueBatchCancelControl, { props });
  await flushPromises();
  expect(singleton.find("button").exists()).toBe(false);
  singleton.unmount();
});
it("clears confirmation and ignores late reads when the selected machine changes", async () => {
  const wrapper = mount(QueueBatchCancelControl, { props });
  await flushPromises();
  await wrapper.get("button").trigger("click");
  vi.mocked(inspectQueueBatch).mockResolvedValue(null);
  await wrapper.setProps({ instanceId: "other" });
  await flushPromises();
  expect(wrapper.find("button").exists()).toBe(false);
  expect(cancelInspectedQueueBatch).not.toHaveBeenCalled();
  wrapper.unmount();
});
it("shows cancellation failures inline and permits deliberate retry", async () => {
  vi.mocked(cancelInspectedQueueBatch).mockRejectedValue(
    new Error("Transfer is reserved"),
  );
  const wrapper = mount(QueueBatchCancelControl, { props });
  await flushPromises();
  await wrapper.get("button").trigger("click");
  await wrapper.get("button").trigger("click");
  await flushPromises();
  expect(wrapper.get('[role="alert"]').text()).toContain(
    "Transfer is reserved",
  );
  expect(wrapper.emitted("cancelled")).toBeUndefined();
  expect(wrapper.get("button").attributes("disabled")).toBeUndefined();
  wrapper.unmount();
});
