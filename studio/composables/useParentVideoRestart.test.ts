import { computed, ref } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
const transport = vi.hoisted(() => ({
  options: vi.fn(),
  export: vi.fn(),
  transfer: vi.fn(),
  validate: vi.fn(),
  admit: vi.fn(),
  caps: vi.fn(),
  get: vi.fn(),
}));
vi.mock("../api/videoJobRestart", () => ({
  videoRestartOptions: transport.options,
  exportVideoRestart: transport.export,
  transferVideoRestartPrefix: transport.transfer,
  validateVideoRestart: transport.validate,
  admitVideoRestart: transport.admit,
}));
vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  apiJsonTo: transport.caps,
}));
vi.mock("../api/videoJobs", () => ({
  assertVideoJobOwner: vi.fn(),
  getVideoJob: transport.get,
}));
import { createParentVideoRestart } from "./useParentVideoRestart";
import { readVideoJobOperations } from "../lib/videoJobRecovery";
const host = {
  id: "plato",
  label: "Plato",
  instanceId: "install",
  ready: true,
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const source = {
  hostId: host.id,
  label: host.label,
  instanceId: host.instanceId,
  target: host.target,
};
const options = {
  source_instance_id: "install",
  source_revision: "revision",
  restart: { supported: true },
  send: { supported: true },
};
beforeEach(() => {
  localStorage.clear();
  Object.values(transport).forEach((v) => v.mockReset());
  transport.options.mockResolvedValue(options);
  transport.caps.mockResolvedValue({ video_jobs: { restart: true } });
  transport.export.mockResolvedValue({ request_json: "opaque" });
  transport.validate.mockResolvedValue({ validation_revision: "proof" });
  transport.get.mockResolvedValue(null);
});
function setup() {
  const hosts = ref([host]);
  return {
    hosts,
    state: createParentVideoRestart(computed(() => hosts.value)),
  };
}
it("refuses a legacy target before private export and never cancels or deletes the original", async () => {
  const { state } = setup();
  await state.open(source, "failed");
  transport.export.mockRejectedValue(new Error("Update this machine"));
  await state.start();
  expect(transport.transfer).not.toHaveBeenCalled();
  expect(transport.admit).not.toHaveBeenCalled();
  expect(state.error.value).toContain("Update");
});
it("persists the exact new operation before a lost reply without private export fields", async () => {
  const { state } = setup();
  await state.open(source, "failed");
  transport.admit.mockImplementation(async (args) => {
    args.persist({
      kind: "video_job",
      operationId: args.operationId,
      hostId: "plato",
      instanceId: "install",
      baseUrl: "http://plato",
      createdAt: 1,
      sourceRevision: "revision",
      sourceJobId: "failed",
      sourceHostId: "plato",
      sourceInstanceId: "install",
    });
    expect(readVideoJobOperations()[0]?.operationId).toBe(args.operationId);
    throw new Error("lost reply");
  });
  await state.start();
  const known = readVideoJobOperations()[0]!;
  expect(known).toBeTruthy();
  expect(JSON.stringify(known)).not.toContain("private");
  expect(JSON.stringify(known)).not.toContain("opaque");
  await state.start();
  expect(transport.admit).toHaveBeenCalledTimes(1);
  expect(state.run.admission.value?.operationId).toBe(known.operationId);
});
it("refuses a destination replacement while an export is in flight", async () => {
  const { state, hosts } = setup();
  await state.open(source, "failed");
  let finish!: (value: unknown) => void;
  transport.export.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const pending = state.start();
  await vi.waitFor(() => expect(transport.export).toHaveBeenCalled());
  hosts.value = [{ ...host, instanceId: "replacement" }];
  finish({ request_json: "opaque" });
  await pending;
  expect(transport.admit).not.toHaveBeenCalled();
  expect(readVideoJobOperations()).toEqual([]);
});
