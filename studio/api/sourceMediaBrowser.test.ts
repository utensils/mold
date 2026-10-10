import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  sourceMediaInventories,
  retainedSourceMediaContent,
} from "./sourceMediaBrowser";
import { ApiError } from "./client";
import { sourceMemberKind, sourceMemberLabel } from "../lib/sourceMediaBrowser";
const fetchMock = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiFetchTo: fetchMock,
}));
const target = { baseUrl: "http://owner", apiKey: "private" };
const member = {
  member_id: "opaque",
  role: "keyframes",
  display_name: "frame.json",
  size_bytes: 8,
};
beforeEach(() => {
  fetchMock.mockReset();
});
describe("source media browser boundaries", () => {
  it("does not guess heterogeneous references from filenames", () => {
    expect(
      sourceMemberKind({
        ...member,
        role: "references",
        display_name: "photo.png",
      }),
    ).toBe("unknown");
    expect(sourceMemberKind(member)).toBe("image");
    expect(sourceMemberLabel({ ...member, frame_index: 48 })).toContain("48");
  });
  it("rejects oversized inventory requests before HTTP", async () => {
    await expect(
      sourceMediaInventories(
        Array.from({ length: 101 }, (_, i) => `${i}.png`),
        target,
      ),
    ).rejects.toThrow("100");
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("does not downgrade authentication or transport failures to empty", async () => {
    fetchMock.mockRejectedValue(new ApiError("Unauthorized", 401));
    await expect(sourceMediaInventories(["owner.png"], target)).rejects.toThrow(
      "Unauthorized",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it("falls back on an unsupported batch endpoint only", async () => {
    fetchMock
      .mockRejectedValueOnce(new ApiError("old", 404))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ availability: "available", members: [member] }),
        ),
      );
    const answer = await sourceMediaInventories(["owner.png"], target);
    expect(answer.inventories["owner.png"]?.members).toEqual([member]);
    expect(fetchMock.mock.calls[1]?.[1]).toBe(
      "/api/gallery/source-media/owner.png",
    );
  });
  it("rejects mismatched instances", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          instance_id: "new",
          inventories: { "a.png": { availability: "available", members: [] } },
        }),
      ),
    );
    await expect(
      sourceMediaInventories(["a.png"], target, undefined, "old"),
    ).rejects.toThrow("changed");
  });
  it("never falls back to a JSON keyframe as image content", async () => {
    fetchMock.mockRejectedValue(new ApiError("old", 404));
    await expect(
      retainedSourceMediaContent("a.png", member, target),
    ).rejects.toThrow("Update");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
it("bounds older-server inventory fallback concurrency", async () => {
  let active = 0,
    peak = 0;
  fetchMock.mockRejectedValueOnce(new ApiError("old", 404));
  fetchMock.mockImplementation(async () => {
    active++;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active--;
    return new Response(
      JSON.stringify({ availability: "available", members: [] }),
    );
  });
  const answer = await sourceMediaInventories(
    Array.from({ length: 10 }, (_, n) => `${n}.png`),
    target,
  );
  expect(Object.keys(answer.inventories)).toHaveLength(10);
  expect(peak).toBe(4);
});
