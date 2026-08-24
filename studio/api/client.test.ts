import { afterEach, describe, expect, it, vi } from "vitest";
import {
  IncompatibleHostError,
  apiFetchTo,
  conditionalApiJsonTo,
  parseCurrentServerStatus,
} from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("target-explicit Studio API", () => {
  it("keeps durable API keys in headers and out of URLs", async () => {
    let captured: [RequestInfo | URL, RequestInit | undefined] | null = null;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = [input, init];
        return new Response("{}");
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    await apiFetchTo(
      { baseUrl: "http://studio:7680", apiKey: "secret" },
      "/api/status",
    );
    const [url, init] = captured!;
    expect(url).toBe("http://studio:7680/api/status");
    expect(String(url)).not.toContain("secret");
    expect((init?.headers as Headers).get("x-api-key")).toBe("secret");
  });

  it("rejects hosts that do not implement the current web contract", () => {
    expect(() => parseCurrentServerStatus({ hostname: "old" })).toThrow(
      IncompatibleHostError,
    );
  });

  it("reuses an unchanged gallery snapshot on 304", async () => {
    const rows = [{ filename: "cat.png" }];
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(rows), {
          headers: { "content-type": "application/json", etag: '"gallery-1"' },
        }),
      )
      .mockImplementationOnce(async (_input, init?: RequestInit) => {
        expect((init?.headers as Headers).get("if-none-match")).toBe(
          '"gallery-1"',
        );
        return new Response(null, { status: 304 });
      });
    vi.stubGlobal("fetch", fetchMock);
    const target = { baseUrl: "http://etag-test:7680", apiKey: "secret" };

    const first = await conditionalApiJsonTo<typeof rows>(
      target,
      "/api/gallery",
    );
    const second = await conditionalApiJsonTo<typeof rows>(
      target,
      "/api/gallery",
    );

    expect(second).toBe(first);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
