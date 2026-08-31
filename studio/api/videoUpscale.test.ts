import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createFramewiseUpscale,
  transitionFramewiseUpscale,
} from "./videoUpscale";

const target = { baseUrl: "http://plato:7680", apiKey: "secret" };
afterEach(() => vi.unstubAllGlobals());

describe("Framewise upscale API", () => {
  it("sends gallery authority rather than video bytes or a server path", async () => {
    let request: RequestInit | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url, init) => {
        request = init;
        return Response.json({
          id: "vup-1",
          state: "queued",
          disclosure: "Framewise upscale",
        });
      }),
    );
    await createFramewiseUpscale(
      target,
      "clip.mp4",
      "real-esrgan-x4plus:fp16",
      256,
    );
    expect(JSON.parse(String(request?.body))).toEqual({
      source: { kind: "library", filename: "clip.mp4" },
      model: "real-esrgan-x4plus:fp16",
      tile_size: 256,
    });
    expect(new Headers(request?.headers).get("x-api-key")).toBe("secret");
  });

  it("uses explicit lifecycle endpoints", async () => {
    const calls: Array<[string, string]> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url, init) => {
        calls.push([String(url), init?.method ?? "GET"]);
        return Response.json({
          id: "vup/1",
          state: "paused",
          disclosure: "Framewise upscale",
        });
      }),
    );
    await transitionFramewiseUpscale(target, "vup/1", "pause");
    await transitionFramewiseUpscale(target, "vup/1", "cancel");
    expect(calls).toEqual([
      ["http://plato:7680/api/video-upscale-jobs/vup%2F1/pause", "POST"],
      ["http://plato:7680/api/video-upscale-jobs/vup%2F1", "DELETE"],
    ]);
  });
});
