import { afterEach, describe, expect, it, vi } from "vitest";
import {
  IncompatibleHostError,
  apiFetchTo,
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
});
