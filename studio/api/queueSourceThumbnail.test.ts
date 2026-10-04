import { describe, expect, it, vi } from "vitest";
import { queueSourceThumbnail } from "./queueSourceThumbnail";
const mocks = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("./client", () => ({ apiFetchTo: mocks.fetch }));
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
