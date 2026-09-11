import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  IncompatibleHostError,
  apiFetchTo,
  conditionalApiJsonTo,
  parseCurrentServerStatus,
} from "./client";

afterEach(() => vi.unstubAllGlobals());

const failureReaders = [
  {
    name: "ordinary",
    read: () =>
      apiFetchTo(
        { baseUrl: "http://failure-test:7680", apiKey: null },
        "/api/status",
      ),
  },
  {
    name: "conditional",
    read: () =>
      conditionalApiJsonTo(
        { baseUrl: "http://failure-test:7680", apiKey: null },
        "/api/gallery",
      ),
  },
] as const;

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

  it.each(failureReaders)(
    "preserves structured HTTP errors for $name requests",
    async ({ read }) => {
      const body = { error: "invalid request", code: "INVALID_REQUEST" };
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(body), {
            status: 422,
            statusText: "Unprocessable Content",
            headers: { "content-type": "application/json" },
          }),
        ),
      );

      await expect(read()).rejects.toEqual(
        expect.objectContaining<ApiError>({
          name: "ApiError",
          message: "invalid request",
          status: 422,
          body,
        }),
      );
    },
  );

  it.each(failureReaders)(
    "falls back to status text for non-JSON $name failures",
    async ({ read }) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response("bad gateway", {
            status: 502,
            statusText: "Bad Gateway",
          }),
        ),
      );

      await expect(read()).rejects.toEqual(
        expect.objectContaining<ApiError>({
          name: "ApiError",
          message: "Bad Gateway",
          status: 502,
          body: null,
        }),
      );
    },
  );

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

describe("throwApiError without a reason phrase", () => {
  it("names the status when the response carries no statusText and no error body", async () => {
    const { apiFetchTo, ApiError } = await import("./client");
    const original = globalThis.fetch;
    globalThis.fetch = (async () =>
      ({
        ok: false,
        status: 503,
        statusText: "",
        clone() {
          return { json: async () => ({}) };
        },
      }) as unknown as Response) as typeof fetch;
    try {
      await expect(
        apiFetchTo({ baseUrl: "http://m", apiKey: null }, "/api/config"),
      ).rejects.toMatchObject({ message: "HTTP 503", status: 503 });
      await expect(
        apiFetchTo({ baseUrl: "http://m", apiKey: null }, "/api/config"),
      ).rejects.toBeInstanceOf(ApiError);
    } finally {
      globalThis.fetch = original;
    }
  });
});
