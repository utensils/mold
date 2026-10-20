import { beforeEach, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
const fetch = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: json,
  apiFetchTo: fetch,
}));
import {
  exportChainRestart,
  admitChainRestart,
  discardChainRecovery,
  cancelChainRestart,
} from "./chainRestart";

const source = {
  hostId: "plato",
  label: "Plato",
  instanceId: "source-install",
  target: { baseUrl: "http://plato", apiKey: "source-secret" },
};
const destination = {
  hostId: "hal",
  label: "HAL9000",
  instanceId: "destination-install",
  target: { baseUrl: "http://hal", apiKey: "destination-secret" },
};
const options = {
  source_instance_id: "source-install",
  source_revision: "frozen-revision",
  restart: { supported: true },
  send: { supported: true },
};
// The opaque wire payload deliberately exceeds JavaScript's integer range.
const requestJson =
  '{"model":"opaque","seed":18446744073709551615,"stages":[{"prompt":"literal\\\\n","seed_offset":18446744073709551614,"frames":107}]}';
beforeEach(() => {
  json.mockReset();
  fetch.mockReset();
});

it("discards only explicitly advertised recovery data with the captured owner and revision", async () => {
  json.mockResolvedValue({ instance_id: "source-install" });
  fetch.mockResolvedValue(new Response(null, { status: 204 }));
  const authority = { ...options, discard: { supported: true } };
  await discardChainRecovery(source, "job/id", authority);
  expect(fetch).toHaveBeenCalledExactlyOnceWith(
    source.target,
    "/api/chain-jobs/job%2Fid/discard",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_instance_id: "source-install",
        source_revision: "frozen-revision",
      }),
    },
  );
  fetch.mockClear();
  await expect(discardChainRecovery(source, "job", options)).rejects.toThrow(
    "Update",
  );
  expect(fetch).not.toHaveBeenCalled();
  await expect(
    discardChainRecovery(source, "job", authority, () => false),
  ).rejects.toThrow("changed");
  expect(fetch).not.toHaveBeenCalled();
});

it("exports only the captured source revision without changing the original", async () => {
  json.mockResolvedValue({
    ...options,
    request_json: requestJson,
    destination_instance_id: "destination-install",
  });
  await expect(
    exportChainRestart(source, "job/id", options, "destination-install"),
  ).resolves.toMatchObject({
    request_json: requestJson,
    destination_instance_id: "destination-install",
  });
  expect(json).toHaveBeenCalledExactlyOnceWith(
    source.target,
    "/api/chain-jobs/job%2Fid/restart-export",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_instance_id: "source-install",
        source_revision: "frozen-revision",
        destination_instance_id: "destination-install",
      }),
    },
  );
});

it("refuses a changed source or export revision", async () => {
  await expect(
    exportChainRestart(
      { ...source, instanceId: "replacement" },
      "job",
      options,
      "destination-install",
    ),
  ).rejects.toThrow("installation");
  expect(json).not.toHaveBeenCalled();
  json.mockResolvedValue({
    ...options,
    source_revision: "different",
    request_json: requestJson,
    destination_instance_id: "destination-install",
  });
  await expect(
    exportChainRestart(source, "job", options, "destination-install"),
  ).rejects.toThrow("changed");
});

it("persists source and destination authority before admission and posts wire bytes unchanged", async () => {
  const order: string[] = [];
  json.mockImplementation(async (_, path) => {
    if (path === "/api/status") return { instance_id: "destination-install" };
    if (path === "/api/capabilities") return { queue: { chain_restart: true } };
    order.push("post");
    throw new Error("lost reply");
  });
  let saved: unknown;
  await expect(
    admitChainRestart({
      source,
      sourceJobId: "old-job",
      options,
      destination,
      requestJson,
      operationId: "same-operation",
      persist: (value) => {
        order.push("persist");
        saved = value;
      },
    }),
  ).rejects.toThrow("lost reply");
  expect(order).toEqual(["persist", "post"]);
  expect(saved).toMatchObject({
    operationId: "same-operation",
    sourceJobId: "old-job",
    sourceRevision: "frozen-revision",
    destinationInstanceId: "destination-install",
  });
  expect(JSON.stringify(saved)).not.toMatch(/secret|opaque|184467/);
  expect(json.mock.calls.at(-1)).toEqual([
    destination.target,
    "/api/chain-jobs",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-mold-operation-id": "same-operation",
        "x-mold-destination-instance": "destination-install",
      },
      body: requestJson,
    },
  ]);
});

it("refuses destination replacement, stale route, and storage failure before POST", async () => {
  const base = {
    source,
    sourceJobId: "old-job",
    options,
    destination,
    requestJson,
    operationId: "operation",
    persist: vi.fn(),
  };
  json.mockResolvedValue({ instance_id: "replacement" });
  await expect(admitChainRestart(base)).rejects.toThrow("installation");
  expect(json).toHaveBeenCalledTimes(1);
  json
    .mockClear()
    .mockImplementation(async (_, path) =>
      path === "/api/status"
        ? { instance_id: "destination-install" }
        : { queue: { chain_restart: true } },
    );
  await expect(
    admitChainRestart({ ...base, isCurrent: () => false }),
  ).rejects.toThrow("changed");
  expect(base.persist).not.toHaveBeenCalled();
  await expect(
    admitChainRestart({
      ...base,
      persist: () => {
        throw new Error("storage full");
      },
    }),
  ).rejects.toThrow("storage full");
  expect(
    json.mock.calls.every(([, path]) =>
      ["/api/status", "/api/capabilities"].includes(path),
    ),
  ).toBe(true);
});

it("requires affirmative destination header enforcement before sending a new attempt", async () => {
  json.mockImplementation(async (_, path) =>
    path === "/api/status" ? { instance_id: "destination-install" } : {},
  );
  const persist = vi.fn();
  await expect(
    admitChainRestart({
      source,
      sourceJobId: "old-job",
      options,
      destination,
      requestJson,
      operationId: "operation",
      persist,
    }),
  ).rejects.toThrow("Update");
  expect(persist).not.toHaveBeenCalled();
  expect(json.mock.calls.some(([, path]) => path === "/api/chain-jobs")).toBe(
    false,
  );
});

it("requires affirmative safe cancellation and sends the captured destination installation", async () => {
  json.mockImplementation(async (_, path) =>
    path === "/api/status"
      ? { instance_id: "destination-install" }
      : { queue: { chain_restart: true } },
  );
  await expect(
    cancelChainRestart(destination, "known-operation"),
  ).rejects.toThrow("Update");
  expect(fetch).not.toHaveBeenCalled();
  json.mockImplementation(async (_, path) =>
    path === "/api/status"
      ? { instance_id: "destination-install" }
      : { queue: { chain_restart_controls: true } },
  );
  await cancelChainRestart(destination, "known-operation");
  expect(fetch).toHaveBeenCalledExactlyOnceWith(
    destination.target,
    "/api/chain-jobs/known-operation/operations/known-operation/cancel",
    {
      method: "POST",
      headers: { "x-mold-destination-instance": "destination-install" },
    },
  );
  fetch.mockClear();
  await expect(
    cancelChainRestart(destination, "known-operation", () => false),
  ).rejects.toThrow("changed");
  expect(fetch).not.toHaveBeenCalled();
});
