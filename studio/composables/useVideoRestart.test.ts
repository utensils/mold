import { computed, ref } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({
  options: vi.fn(),
  export: vi.fn(),
  admit: vi.fn(),
  owner: vi.fn(),
  get: vi.fn(),
  cancel: vi.fn(),
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
  cancelChainOperation: api.cancel,
}));
import { createVideoRestart, VIDEO_RESTART_KEY } from "./useVideoRestart";
import { ApiError } from "../api/client";
const source = {
  id: "plato",
  label: "Plato",
  instanceId: "source",
  ready: true,
  target: { baseUrl: "http://plato", apiKey: "private-source" },
};
const target = {
  id: "hal",
  label: "HAL9000",
  instanceId: "target",
  ready: true,
  target: { baseUrl: "http://hal", apiKey: "private-target" },
};
const options = {
  source_instance_id: "source",
  source_revision: "revision",
  restart: { supported: true },
  send: { supported: true },
};
beforeEach(() => {
  localStorage.clear();
  Object.values(api).forEach((mock) => mock.mockReset());
  api.options.mockResolvedValue(options);
  api.owner.mockResolvedValue(undefined);
  api.get.mockResolvedValue(null);
  api.export.mockResolvedValue({
    ...options,
    destination_instance_id: "target",
    request_json: "opaque wire bytes",
  });
});

it("opens read-only and names same-host restart separately from sending", async () => {
  const state = createVideoRestart(
    computed(() => [source, target]),
    ref(false),
  );
  await state.open("plato", "old-job");
  expect(state.options.value).toEqual(options);
  expect(state.selected.value).toBe("plato");
  expect(state.destinations.value.map((host) => host.id)).toEqual([
    "plato",
    "hal",
  ]);
  expect(api.export).not.toHaveBeenCalled();
  expect(api.admit).not.toHaveBeenCalled();
});

it("drops stale source-option replies after the dialog closes", async () => {
  let resolve!: (value: typeof options) => void;
  api.options.mockReturnValue(
    new Promise((done) => {
      resolve = done;
    }),
  );
  const state = createVideoRestart(
    computed(() => [source, target]),
    ref(false),
  );
  const opening = state.open("plato", "old-job");
  state.close();
  resolve(options);
  await opening;
  expect(state.options.value).toBeNull();
  expect(state.active.value).toBe(false);
});

it("keeps lost admission replies recoverable and never makes a second operation on reopen", async () => {
  const state = createVideoRestart(
    computed(() => [source, target]),
    ref(false),
  );
  api.admit.mockImplementation(async (args) => {
    args.persist({
      operationId: "known-operation",
      hostId: "hal",
      instanceId: "target",
      baseUrl: "http://hal",
      createdAt: 1,
      sourceJobId: "old-job",
      sourceHostId: "plato",
      sourceInstanceId: "source",
      sourceRevision: "revision",
      destinationInstanceId: "target",
    });
    throw new Error("lost reply");
  });
  await state.open("plato", "old-job");
  state.selected.value = "hal";
  await state.start();
  expect(state.admission.value?.operationId).toBe("known-operation");
  expect(localStorage.getItem(VIDEO_RESTART_KEY)).not.toMatch(/private|opaque/);
  state.close();
  await state.open("plato", "old-job");
  expect(state.admission.value?.operationId).toBe("known-operation");
  expect(api.admit).toHaveBeenCalledTimes(1);
  expect(api.get).toHaveBeenLastCalledWith(
    expect.objectContaining({ instanceId: "target" }),
    "known-operation",
  );
});

it("fences a replaced destination before exporting or admitting inputs", async () => {
  const hosts = ref([source, target]);
  const state = createVideoRestart(
    computed(() => hosts.value),
    ref(false),
  );
  await state.open("plato", "old-job");
  state.selected.value = "hal";
  api.owner.mockImplementation(async () => {
    hosts.value = [source, { ...target, instanceId: "replacement" }];
  });
  await state.start();
  expect(api.export).not.toHaveBeenCalled();
  expect(api.admit).not.toHaveBeenCalled();
  expect(state.error.value).toContain("changed");
});

it("can cancel a known operation while admission is unanswered", async () => {
  const state = createVideoRestart(
    computed(() => [source, target]),
    ref(false),
  );
  let finish!: () => void;
  api.admit.mockImplementation(async (args) => {
    args.persist({
      operationId: "known-operation",
      hostId: "hal",
      instanceId: "target",
      baseUrl: "http://hal",
      createdAt: 1,
      sourceJobId: "old-job",
      sourceHostId: "plato",
      sourceInstanceId: "source",
      sourceRevision: "revision",
      destinationInstanceId: "target",
    });
    await new Promise<void>((resolve) => {
      finish = resolve;
    });
  });
  await state.open("plato", "old-job");
  state.selected.value = "hal";
  const starting = state.start();
  await vi.waitFor(() => expect(state.admission.value).not.toBeNull());
  await state.cancel();
  expect(api.cancel).toHaveBeenCalledWith(
    expect.objectContaining({ instanceId: "target" }),
    "known-operation",
    expect.any(Function),
  );
  expect(state.admission.value?.cancelRequested).toBe(true);
  finish();
  await starting;
});

it("does not install a late cancellation reply into another open video", async () => {
  const state = createVideoRestart(
    computed(() => [source, target]),
    ref(false),
  );
  const known = {
    operationId: "known",
    hostId: "hal",
    instanceId: "target",
    baseUrl: "http://hal",
    createdAt: 1,
    sourceJobId: "old-job",
    sourceHostId: "plato",
    sourceInstanceId: "source",
    sourceRevision: "revision",
    destinationInstanceId: "target",
  };
  localStorage.setItem(VIDEO_RESTART_KEY, JSON.stringify([known]));
  await state.open("plato", "old-job");
  let finish!: () => void;
  api.cancel.mockReturnValue(
    new Promise<void>((resolve) => {
      finish = resolve;
    }),
  );
  const cancelling = state.cancel();
  await vi.waitFor(() => expect(api.cancel).toHaveBeenCalled());
  state.close();
  await state.open("plato", "different-job");
  finish();
  await cancelling;
  expect(state.admission.value).toBeNull();
  expect(JSON.parse(localStorage.getItem(VIDEO_RESTART_KEY)!)[0]).toMatchObject(
    { operationId: "known", cancelAcknowledged: true },
  );
});

it("clears a typed destination refusal but keeps an ambiguous conflict recoverable", async () => {
  for (const [code, retained] of [
    ["CHAIN_DESTINATION_CHANGED", false],
    ["UNKNOWN_CONFLICT", true],
  ] as const) {
    localStorage.clear();
    const state = createVideoRestart(
      computed(() => [source, target]),
      ref(false),
    );
    api.admit.mockImplementation(async (args) => {
      args.persist({
        operationId: "known",
        hostId: "hal",
        instanceId: "target",
        baseUrl: "http://hal",
        createdAt: 1,
        sourceJobId: "old-job",
        sourceHostId: "plato",
        sourceInstanceId: "source",
        sourceRevision: "revision",
        destinationInstanceId: "target",
      });
      throw new ApiError("conflict", 409, { code });
    });
    await state.open("plato", "old-job");
    state.selected.value = "hal";
    await state.start();
    expect(state.admission.value !== null).toBe(retained);
  }
});
