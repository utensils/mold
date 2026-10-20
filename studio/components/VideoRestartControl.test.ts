import { computed, ref } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({
  options: vi.fn(),
  export: vi.fn(),
  admit: vi.fn(),
  owner: vi.fn(),
  get: vi.fn(),
}));
vi.mock("../api/chainRestart", () => ({
  assertChainRestartDestination: api.owner,
  chainRestartOptions: api.options,
  exportChainRestart: api.export,
  admitChainRestart: api.admit,
}));
vi.mock("../api/chainJobs", async (original) => ({
  ...(await original<typeof import("../api/chainJobs")>()),
  assertChainOwner: api.owner,
  getChainJob: api.get,
}));
vi.mock("../composables/useHeldQueueTransfer", () => ({
  useHeldQueueTransfer: () => transfer,
}));
import VideoRestartControl from "./VideoRestartControl.vue";
const transfer = {
  hosts: computed(() => [
    {
      id: "plato",
      label: "Plato",
      instanceId: "source",
      ready: true,
      target: { baseUrl: "http://plato", apiKey: null },
    },
    {
      id: "hal",
      label: "HAL9000",
      instanceId: "target",
      ready: true,
      target: { baseUrl: "http://hal", apiKey: null },
    },
  ]),
  busy: ref(false),
};
beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  api.options.mockResolvedValue({
    source_instance_id: "source",
    source_revision: "revision",
    restart: { supported: true },
    send: { supported: true },
  });
  api.export.mockResolvedValue({
    source_instance_id: "source",
    source_revision: "revision",
    destination_instance_id: "target",
    request_json: "opaque",
  });
  api.owner.mockResolvedValue(undefined);
  api.get.mockResolvedValue(null);
});
it("opens a reviewable destination choice without generating and requires explicit confirmation", async () => {
  const wrapper = mount(VideoRestartControl, {
    props: {
      hostId: "plato",
      jobId: "failed-parent",
      state: "failed",
      chain: true,
    },
    global: { stubs: { Teleport: true } },
    attachTo: document.body,
  });
  expect(api.options).not.toHaveBeenCalled();
  await wrapper.get('[data-test="video-restart-open"]').trigger("click");
  await flushPromises();
  expect(wrapper.text()).toContain("The original stays on its machine");
  expect(wrapper.get('[data-test="video-restart-submit"]').text()).toBe(
    "Start again",
  );
  expect(api.export).not.toHaveBeenCalled();
  expect(api.admit).not.toHaveBeenCalled();
  await wrapper.get("select").setValue("hal");
  expect(wrapper.get('[data-test="video-restart-submit"]').text()).toBe(
    "Send video",
  );
  await wrapper.get('[data-test="video-restart-submit"]').trigger("click");
  await flushPromises();
  expect(api.admit).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({
      sourceJobId: "failed-parent",
      requestJson: "opaque",
      destination: expect.objectContaining({
        hostId: "hal",
        instanceId: "target",
      }),
    }),
  );
  wrapper.unmount();
});
it("offers no restart mutation for an ordinary or live generation", () => {
  for (const props of [
    { chain: false, state: "failed" },
    { chain: true, state: "running" },
  ]) {
    const wrapper = mount(VideoRestartControl, {
      props: { hostId: "plato", jobId: "job", ...props },
    });
    expect(wrapper.find('[data-test="video-restart-open"]').exists()).toBe(
      false,
    );
    wrapper.unmount();
  }
  expect(api.options).not.toHaveBeenCalled();
  expect(api.admit).not.toHaveBeenCalled();
});
