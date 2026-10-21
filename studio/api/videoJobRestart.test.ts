import { beforeEach, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
const fetch = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: json,
  apiFetchTo: fetch,
}));
vi.mock("./videoJobs", () => ({
  assertVideoJobOwner: vi.fn(),
  checkedVideoJobPlan: (value: unknown) => value,
}));
import {
  exportVideoRestart,
  transferVideoRestartPrefix,
  admitVideoRestart,
} from "./videoJobRestart";
const source = {
  hostId: "plato",
  instanceId: "source",
  label: "Plato",
  target: { baseUrl: "http://plato", apiKey: "source-secret" },
};
const destination = {
  hostId: "hal",
  instanceId: "destination",
  label: "HAL",
  target: { baseUrl: "http://hal", apiKey: "destination-secret" },
};
const options = {
  source_instance_id: "source",
  source_revision: "revision",
  restart: { supported: true },
  send: { supported: true },
};
const raw =
  '{"generation":{"seed":18446744073709551615,"prompt":"literal\\\\n"},"duration_ms":"30000"}';
const spec = '{"metadata":{"seed":18446744073709551615},"size_bytes":"3"}';
const prefix = {
  filename: "old.mp4",
  archive_identity_sha256: "a".repeat(64),
  sha256: "b".repeat(64),
  size_bytes: "3",
  source_ref: "locator",
  upload_spec_json: spec,
};
const exported = {
  ...options,
  destination_instance_id: "destination",
  request_json: raw,
  prefix,
};
const uploadId = "c7f8554e-53e7-4bd1-a203-346b72032e31";
beforeEach(() => {
  json.mockReset();
  fetch.mockReset();
});
it("keeps full-range request and metadata opaque and fences export revision", async () => {
  json.mockImplementation(async (_target, path) =>
    path === "/api/capabilities" ? { video_jobs: { restart: true } } : exported,
  );
  expect(
    (await exportVideoRestart(source, "old/id", options, destination))
      .request_json,
  ).toBe(raw);
  expect(
    json.mock.calls.find((call) => call[1].includes("restart-export"))?.[1],
  ).toBe("/api/video-jobs/old%2Fid/restart-export");
  json.mockImplementation(async (_target, path) =>
    path === "/api/capabilities"
      ? { video_jobs: { restart: true } }
      : { ...exported, source_revision: "changed" },
  );
  await expect(
    exportVideoRestart(source, "old", options, destination),
  ).rejects.toThrow("changed");
});
it("relays bounded bytes using each owner's transport and forwards metadata unchanged", async () => {
  json.mockImplementation(async (_target, path, init) => {
    if (path === "/api/capabilities")
      return { video_jobs: { restart: true, source_transfer: true } };
    if (path.endsWith("/complete"))
      return { handle: uploadId, sha256: prefix.sha256, size_bytes: "3" };
    if (init?.method === "PUT")
      return { handle: uploadId, offset_bytes: "3", completed: false };
    return { handle: uploadId, offset_bytes: "0", completed: false };
  });
  fetch.mockResolvedValue(new Response(new Uint8Array([1, 2, 3])));
  expect(
    await transferVideoRestartPrefix({
      source,
      sourceJobId: "old/id",
      destination,
      exported,
      uploadId,
    }),
  ).toEqual({ handle: uploadId, sha256: prefix.sha256, size_bytes: "3" });
  const begin = json.mock.calls.find((call) => call[2]?.body === spec)!;
  expect(begin[0]).toEqual(destination.target);
  expect(begin[2].headers["x-mold-destination-instance"]).toBe("destination");
  expect(fetch.mock.calls[0]?.[0]).toEqual(source.target);
  expect(fetch.mock.calls[0]?.[2].headers["x-mold-source-instance"]).toBe(
    "source",
  );
});
it("refuses oversized source chunks and a replaced owner before upload", async () => {
  json.mockImplementation(async (_target, path) =>
    path === "/api/capabilities"
      ? { video_jobs: { restart: true, source_transfer: true } }
      : { handle: uploadId, offset_bytes: "0", completed: false },
  );
  fetch.mockResolvedValue(new Response(new Uint8Array(4 * 1024 * 1024 + 1)));
  await expect(
    transferVideoRestartPrefix({
      source,
      sourceJobId: "old",
      destination,
      exported,
      uploadId,
    }),
  ).rejects.toThrow();
  expect(json.mock.calls.some((call) => call[2]?.method === "PUT")).toBe(false);
  await expect(
    transferVideoRestartPrefix({
      source,
      sourceJobId: "old",
      destination: { ...destination, instanceId: "replacement" },
      exported,
      uploadId,
    }),
  ).rejects.toThrow("changed");
});
it("uses only a same-instance sealed prefix grant without relaying its bytes", async () => {
  json.mockResolvedValue({
    video_jobs: { restart: true, source_transfer: true },
  });
  const local = {
    ...exported,
    destination_instance_id: "source",
    prefix: {
      ...prefix,
      retained_upload: {
        handle: uploadId,
        sha256: prefix.sha256,
        size_bytes: "3",
      },
    },
  };
  expect(
    await transferVideoRestartPrefix({
      source,
      sourceJobId: "old",
      destination: source,
      exported: local,
      uploadId,
    }),
  ).toEqual(local.prefix.retained_upload);
  expect(fetch).not.toHaveBeenCalled();
  expect(json.mock.calls.every((call) => call[1] === "/api/capabilities")).toBe(
    true,
  );
  await expect(
    transferVideoRestartPrefix({
      source,
      sourceJobId: "old",
      destination,
      exported: { ...local, destination_instance_id: "destination" },
      uploadId,
    }),
  ).rejects.toThrow();
});

it("persists scalar authority before POST and preserves opaque JSON through lost replies", async () => {
  const order: string[] = [];
  json.mockImplementation(async (_target, path) => {
    if (path === "/api/capabilities") return { video_jobs: { restart: true } };
    order.push("post");
    throw new Error("lost reply");
  });
  let saved: unknown;
  await expect(
    admitVideoRestart({
      source,
      sourceJobId: "old",
      options,
      destination,
      exported,
      proof: "validated",
      operationId: uploadId,
      upload: { handle: uploadId, sha256: prefix.sha256, size_bytes: "3" },
      persist: (value) => {
        order.push("persist");
        saved = value;
      },
    }),
  ).rejects.toThrow("lost reply");
  expect(order).toEqual(["persist", "post"]);
  const post = json.mock.calls.find((call) => call[1] === "/api/video-jobs")!;
  expect(post[2].body).toBe(raw);
  expect(post[2].headers["x-mold-video-validation-revision"]).toBe("validated");
  expect(JSON.stringify(saved)).not.toContain("secret");
  expect(JSON.stringify(saved)).not.toContain(raw);
});

it("checks both affirmative capabilities before any private export", async () => {
  json.mockResolvedValue({ video_jobs: {} });
  await expect(
    exportVideoRestart(source, "old", options, destination),
  ).rejects.toThrow("Update");
  expect(json.mock.calls.every((call) => call[1] === "/api/capabilities")).toBe(
    true,
  );
});
