import { beforeEach, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
const ownerCheck = vi.hoisted(() => vi.fn());
vi.mock("./client", () => ({ apiJsonTo: json }));
vi.mock("./videoJobs", () => ({ assertVideoJobOwner: ownerCheck }));
import { videoLibraryOutputExists } from "./videoLibraryOutput";
const owner = {
  hostId: "machine",
  label: "Machine",
  instanceId: "instance",
  target: { baseUrl: "http://machine", apiKey: "secret" },
};
beforeEach(() => {
  json.mockReset();
  ownerCheck.mockReset();
});
it("recognizes only an exact active Library filename on the owning installation", async () => {
  json.mockResolvedValue([{ filename: "movie.mp4" }]);
  expect(await videoLibraryOutputExists(owner, "movie.mp4")).toBe(true);
  expect(ownerCheck).toHaveBeenCalledTimes(2);
  json.mockResolvedValue([]);
  expect(await videoLibraryOutputExists(owner, "movie.mp4")).toBe(false);
});
it("does not call offline, malformed or changed-owner output deleted", async () => {
  json.mockRejectedValue(new Error("offline"));
  await expect(videoLibraryOutputExists(owner, "movie.mp4")).rejects.toThrow(
    "offline",
  );
  json.mockResolvedValue([{ filename: "another.mp4" }]);
  await expect(videoLibraryOutputExists(owner, "movie.mp4")).rejects.toThrow(
    "invalid",
  );
  let current = true;
  json.mockImplementation(async () => {
    current = false;
    return [];
  });
  await expect(
    videoLibraryOutputExists(owner, "movie.mp4", () => current),
  ).rejects.toThrow("changed");
});
