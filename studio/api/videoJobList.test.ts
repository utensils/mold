import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ read: vi.fn(), owner: vi.fn() }));
vi.mock("./client", () => ({ apiJsonTo: api.read }));
vi.mock("./videoJobs", () => ({ assertVideoJobOwner: api.owner }));
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
