import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ read: vi.fn(), owner: vi.fn() }));
vi.mock("./client", () => ({ apiJsonTo: api.read }));
vi.mock("./videoJobs", async (original) => ({
  ...(await original<typeof import("./videoJobs")>()),
  assertVideoJobOwner: api.owner,
}));
import { listVideoJobs } from "./videoJobList";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "install",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
beforeEach(() => {
  vi.clearAllMocks();
  api.owner.mockResolvedValue(undefined);
  api.read.mockResolvedValue({
    jobs: [],
    has_more: true,
    next_cursor: "opaque cursor",
  });
});
it("uses a stable recoverable cursor bound to the captured installation", async () => {
  await listVideoJobs(owner, "opaque cursor");
  const [, path, init] = api.read.mock.calls[0]!;
  expect(path).toContain("recoverable_only=true");
  expect(path).toContain("cursor=opaque+cursor");
  expect(init.headers["x-mold-destination-instance"]).toBe("install");
});
it("refuses an owner change or malformed paging response before publishing rows", async () => {
  api.read.mockImplementation(async () => ({
    jobs: [],
    has_more: true,
    next_cursor: null,
  }));
  await expect(listVideoJobs(owner)).rejects.toThrow("pagination");
  await expect(listVideoJobs(owner, undefined, () => false)).rejects.toThrow(
    "changed",
  );
});
it("keeps a known progress phase and drops a malformed one", async () => {
  const plan = {
    validation_revision: "r",
    model: "m",
    width: 8,
    height: 8,
    fps: 24,
    single_clip_max_frames: 345,
    overlap_frames: 1,
    join_mode: "continuous",
    clip_count: "3",
    part_count: "1",
    added_frames: "1",
    source_frames: "0",
    total_frames: "1",
    added_duration_ms: "1",
    total_duration_ms: "1",
    warnings: [],
  };
  api.read.mockResolvedValue({
    jobs: [
      { id: "a", state: "running", progress_phase: "waiting", plan },
      { id: "b", state: "running", progress_phase: { odd: true }, plan },
    ],
    has_more: false,
    next_cursor: null,
  });
  const page = await listVideoJobs(owner);
  expect(page.jobs[0]!.progress_phase).toBe("waiting");
  expect(page.jobs[1]!.progress_phase).toBeUndefined();
});
