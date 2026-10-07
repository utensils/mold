import { describe, expect, it, vi } from "vitest";
import { queueSourceThumbnail, queueInputs } from "./queueSourceThumbnail";
const mocks = vi.hoisted(() => ({ fetch: vi.fn(), json: vi.fn() }));
vi.mock("./client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./client")>()),
  apiFetchTo: mocks.fetch,
  apiJsonTo: mocks.json,
}));
import { ApiError } from "./client";
describe("retained queue source thumbnail", () => {
  const target = { baseUrl: "https://machine/prefix", apiKey: "key" };
  it("uses the owning authenticated machine and escapes the job identity", async () => {
    mocks.fetch.mockResolvedValue(
      new Response(new Uint8Array([1, 2]), {
        headers: { "Content-Type": "image/png" },
      }),
    );
    const signal = new AbortController().signal;
    const blob = await queueSourceThumbnail(target, "job/a", signal);
    expect(mocks.fetch).toHaveBeenLastCalledWith(
      target,
      "/api/queue/job%2Fa/input-thumbnail",
      { signal },
    );
    expect(blob.size).toBe(2);
    expect(blob.type).toBe("image/png");
  });
  it("refuses oversized bodies even without a content length", async () => {
    mocks.fetch.mockResolvedValue(
      new Response(new Uint8Array(2 * 1024 * 1024 + 1), {
        headers: { "Content-Type": "image/png" },
      }),
    );
    await expect(queueSourceThumbnail(target, "job")).rejects.toThrow(
      "too large",
    );
  });
  it("refuses non-image responses", async () => {
    mocks.fetch.mockResolvedValue(
      new Response("error", { headers: { "Content-Type": "text/html" } }),
    );
    await expect(queueSourceThumbnail(target, "job")).rejects.toThrow("image");
  });
});

it("falls back only when the additive input route is absent", async () => {
  mocks.json.mockRejectedValueOnce(new ApiError("missing", 404));
  expect(
    await queueInputs({ baseUrl: "http://box", apiKey: null }, "j"),
  ).toEqual([{ label: "Source", preview: true }]);
  mocks.json.mockRejectedValueOnce(new ApiError("unauthorized", 401));
  await expect(
    queueInputs({ baseUrl: "http://box", apiKey: null }, "j"),
  ).rejects.toThrow("unauthorized");
  mocks.json.mockResolvedValueOnce([]);
  expect(
    await queueInputs({ baseUrl: "http://box", apiKey: null }, "j"),
  ).toEqual([]);
  mocks.json.mockResolvedValueOnce([
    { index: -1, label: "bad", preview: true },
  ]);
  await expect(
    queueInputs({ baseUrl: "http://box", apiKey: null }, "j"),
  ).rejects.toThrow("Invalid");
});
it("addresses an exact ordered input without changing the captured target", async () => {
  mocks.fetch.mockResolvedValueOnce(
    new Response(new Uint8Array([1]), {
      headers: { "Content-Type": "image/png" },
    }),
  );
  const target = { baseUrl: "http://box/mold", apiKey: "key" };
  await queueSourceThumbnail(target, "job/a", undefined, 3);
  expect(mocks.fetch).toHaveBeenLastCalledWith(
    target,
    "/api/queue/job%2Fa/input-thumbnail?index=3",
    {},
  );
});
