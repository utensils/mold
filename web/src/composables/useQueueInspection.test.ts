import { mount } from "@vue/test-utils";
import { defineComponent, ref, computed } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetActiveWork } from "@studio/api/activity";
import type { QueueJobEntry } from "@studio/api/queuePlan";
import type { HostRouting } from "./useHostRouting";
import type { Job } from "./useGenerateStream";
import { useQueueInspection } from "./useQueueInspection";
const api = vi.hoisted(() => ({
  read: vi.fn(),
  get: vi.fn(),
  cancel: vi.fn(),
  retry: vi.fn(),
  pause: vi.fn(),
  toast: vi.fn(),
}));
vi.mock("@studio/api/client", () => ({ apiJsonTo: api.read }));
vi.mock("@studio/api/queuePlan", () => ({
  getQueueJob: api.get,
  cancelQueueJob: api.cancel,
  retryQueueJobRecoveringAmbiguity: api.retry,
  setQueueJobPaused: api.pause,
}));
vi.mock("../lib/toasts", () => ({ toast: api.toast }));
const row = {
  id: "job",
  key: "box:job",
  hostId: "box",
  hostLabel: "Box",
  routeUrl: "https://box",
  instanceId: "instance",
  kind: "generation",
  phase: "running",
  can_cancel: true,
  stale: false,
} as FleetActiveWork;
const entry = (): QueueJobEntry => ({
  job: {
    id: "job",
    model: "flux-dev",
    state: "held",
    started_at_unix_ms: 1,
    position: 0,
    retryable: true,
    batch_id: "batch",
    client_batch_id: "client",
    metadata: {
      prompt: "Server prompt",
      model: "flux-dev",
      width: 1024,
      height: 1024,
    },
  },
});
const cleanups: Array<() => void> = [];
function setup() {
  const hosts = ref([
    {
      id: "box",
      label: "Box",
      url: "https://box",
      apiKey: "secret",
      status: "ready",
    },
  ]);
  const routing = {
    hosts,
    installedModels: computed(() => []),
    capabilitiesByHost: ref({}),
  } as unknown as HostRouting;
  const refresh = vi.fn().mockResolvedValue(undefined);
  let subject!: ReturnType<typeof useQueueInspection>;
  const wrapper = mount(
    defineComponent({
      setup() {
        subject = useQueueInspection(routing, refresh);
        return () => null;
      },
    }),
  );
  cleanups.push(() => wrapper.unmount());
  return { subject, hosts, refresh };
}
beforeEach(() => {
  vi.resetAllMocks();
  api.read.mockImplementation(async (_target, path) =>
    path === "/api/status"
      ? { instance_id: "instance" }
      : { queue: { cooperative_cancellation: true, can_pause_job: true } },
  );
  api.get.mockResolvedValue(entry());
  api.retry.mockResolvedValue({ kind: "accepted" });
});
afterEach(() => {
  cleanups.splice(0).forEach((fn) => fn());
  document.body.innerHTML = "";
});
describe("Queue inspection", () => {
  it("reads and mutates the captured machine with fresh instance checks on both sides", async () => {
    const { subject } = setup();
    await subject.open(row);
    expect(api.read.mock.calls.map((call) => call[1])).toEqual([
      "/api/status",
      "/api/capabilities",
      "/api/status",
    ]);
    await subject.act("retry");
    expect(api.retry).toHaveBeenCalledExactlyOnceWith(
      { baseUrl: "https://box", apiKey: "secret" },
      {
        instanceId: "instance",
        jobId: "job",
        batchId: "batch",
        clientBatchId: "client",
      },
    );
  });
  it("rejects an instance replacement during the detail read", async () => {
    const { subject } = setup();
    api.get.mockImplementation(async () => {
      api.read.mockResolvedValue({ instance_id: "replacement" });
      return entry();
    });
    await subject.open(row);
    expect(subject.detail.value).toBeNull();
    expect(subject.error.value).toContain("different Mold server");
    expect(api.cancel).not.toHaveBeenCalled();
  });
  it("refuses changed routes and a mismatched returned job", async () => {
    const { subject, hosts } = setup();
    await subject.open(row);
    hosts.value[0]!.url = "https://other";
    await subject.act("cancel");
    expect(api.cancel).not.toHaveBeenCalled();
    expect(subject.error.value).toContain("original machine");
    hosts.value[0]!.url = "https://box";
    api.get.mockResolvedValue({ job: { ...entry().job, id: "other" } });
    await subject.open(row);
    expect(subject.error.value).toContain("different queue job");
  });
  it.each([
    { state: "running" },
    { batch_id: "  " },
    { client_batch_id: "\t" },
    { retryable: false },
  ])("refuses incomplete or non-held retry authority %j", async (patch) => {
    const { subject } = setup();
    api.get.mockResolvedValue({ job: { ...entry().job, ...patch } });
    await subject.open(row);
    await subject.act("retry");
    expect(api.retry).not.toHaveBeenCalled();
  });
  it("fails capability-gated mutations closed when fresh capabilities disappear", async () => {
    const { subject } = setup();
    api.get.mockResolvedValue({ job: { ...entry().job, state: "running" } });
    await subject.open(row);
    api.read.mockImplementation(async (_target, path) => {
      if (path === "/api/capabilities") throw new Error("offline");
      return { instance_id: "instance" };
    });
    await subject.act("cancel");
    expect(api.cancel).not.toHaveBeenCalled();
    api.get.mockResolvedValue({ job: { ...entry().job, state: "queued" } });
    await subject.act("pause");
    expect(api.pause).not.toHaveBeenCalled();
  });
  it("preserves local ownership and delegates local mutations only to the stream owner", async () => {
    const { subject } = setup();
    const local = {
      job: {
        id: "local",
        serverId: "job",
        request: {
          prompt: "Local prompt",
          model: "flux-dev",
          collection: { id: "collection", name: "My album" },
          lora: { path: "look.safetensors", scale: 0.75 },
        },
        retryable: true,
        durableBatch: {
          serverBatchId: "batch",
          clientBatchId: "client",
          expectedInstanceId: "instance",
        },
      } as Job,
      cancel: vi.fn().mockResolvedValue(undefined),
      retry: vi.fn().mockResolvedValue(undefined),
    };
    api.get.mockResolvedValue({
      job: {
        ...entry().job,
        batch_id: null,
        client_batch_id: null,
        metadata: null,
      },
    });
    await subject.open(row, local);
    expect(subject.model.value?.facts).toContainEqual({
      label: "Owner",
      value: "This app",
    });
    expect(subject.model.value?.metadataSource).toBe("local");
    expect(JSON.stringify(subject.model.value?.groups)).toContain("My album");
    expect(JSON.stringify(subject.model.value?.groups)).toContain(
      "look.safetensors",
    );
    await subject.act("retry");
    expect(local.retry).toHaveBeenCalledOnce();
    expect(api.retry).not.toHaveBeenCalled();
    await subject.open(row, local);
    await subject.act("cancel");
    expect(local.cancel).toHaveBeenCalledOnce();
    expect(api.cancel).not.toHaveBeenCalled();
  });
  it("ignores a closed detail read and restores the exact opener", async () => {
    const { subject } = setup();
    const button = document.createElement("button");
    document.body.append(button);
    let resolve!: (value: QueueJobEntry) => void;
    api.get.mockImplementation(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    );
    const pending = subject.open(row, undefined, button);
    await vi.waitFor(() => expect(api.get).toHaveBeenCalledOnce());
    subject.close();
    resolve(entry());
    await pending;
    expect(subject.detail.value).toBeNull();
    expect(subject.selected.value).toBeNull();
    expect(document.activeElement).toBe(button);
  });
  it("reports uncertain mutation outcomes after closing without retrying the POST", async () => {
    const { subject } = setup();
    let resolve!: (value: unknown) => void;
    api.retry.mockImplementation(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    );
    await subject.open(row);
    const pending = subject.act("retry");
    await vi.waitFor(() => expect(api.retry).toHaveBeenCalledOnce());
    subject.close();
    resolve({
      kind: "uncertain",
      error: "Check the machine before trying again.",
    });
    await pending;
    expect(api.toast).toHaveBeenCalledWith(
      "error",
      "Box: Check the machine before trying again.",
    );
    expect(api.retry).toHaveBeenCalledOnce();
  });
  it("returns a verified reuse snapshot and pauses only the captured queued job", async () => {
    const { subject } = setup();
    api.get.mockResolvedValue({ job: { ...entry().job, state: "queued" } });
    await subject.open(row);
    expect((await subject.snapshot())?.detail.job.metadata).toEqual(
      entry().job.metadata,
    );
    await subject.act("pause");
    expect(api.pause).toHaveBeenCalledExactlyOnceWith(
      { baseUrl: "https://box", apiKey: "secret" },
      "job",
      true,
    );
  });
});
