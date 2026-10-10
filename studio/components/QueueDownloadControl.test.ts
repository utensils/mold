import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import QueueDownloadControl from "./QueueDownloadControl.vue";
const recovery = vi.hoisted(() => ({ probe: vi.fn(), start: vi.fn() }));
vi.mock("../composables/useQueueDownloadRecovery", () => ({
  missingQueueModel: recovery.probe,
  queueDownloadState: () => null,
  startQueueDownloadRecovery: recovery.start,
}));
beforeEach(() => vi.clearAllMocks());
it("renders a verified repair action without probing and preserves exact authority on activation", async () => {
  const target = { baseUrl: "http://machine", apiKey: "credential" };
  const wrapper = mount(QueueDownloadControl, {
    props: {
      target,
      instance: "instance",
      job: "held",
      host: "Plato",
      controls: true,
      downloadAvailable: true,
    },
  });
  await flushPromises();
  expect(recovery.probe).not.toHaveBeenCalled();
  await wrapper.get("button").trigger("click");
  expect(recovery.start).toHaveBeenCalledWith(
    target,
    "instance",
    "held",
    "Plato",
  );
  wrapper.unmount();
});
it("does not probe or offer a malformed server hint", async () => {
  const wrapper = mount(QueueDownloadControl, {
    props: {
      target: { baseUrl: "http://machine", apiKey: null },
      instance: "instance",
      job: "held",
      host: "Plato",
      controls: true,
      downloadAvailable: false,
    },
  });
  await flushPromises();
  expect(recovery.probe).not.toHaveBeenCalled();
  expect(wrapper.find("button").exists()).toBe(false);
  wrapper.unmount();
});
