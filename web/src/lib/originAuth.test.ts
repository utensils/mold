import { originHost } from "./hostRegistry";
import { fetchModels } from "../api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  originAuthenticatedFetch,
  originApiTarget,
  setOriginApiKey,
} from "./originAuth";

describe("serving origin credentials", () => {
  beforeEach(() => sessionStorage.clear());
  afterEach(() => vi.unstubAllGlobals());
  it("adds the origin key only to same-origin API requests and fences redirects", async () => {
    const mock = vi.fn().mockResolvedValue(new Response());
    vi.stubGlobal("fetch", mock);
    setOriginApiKey("origin-secret");
    await originAuthenticatedFetch("/api/status");
    expect(new Headers(mock.mock.calls[0][1].headers).get("x-api-key")).toBe(
      "origin-secret",
    );
    expect(mock.mock.calls[0][1].redirect).toBe("error");
    await originAuthenticatedFetch("https://foreign.example/api/status");
    expect(new Headers(mock.mock.calls[1][1]?.headers).has("x-api-key")).toBe(
      false,
    );
    await originAuthenticatedFetch("/logo.png");
    expect(new Headers(mock.mock.calls[2][1]?.headers).has("x-api-key")).toBe(
      false,
    );
  });
  it("retains explicit target keys and scopes storage to the exact origin", async () => {
    const mock = vi.fn().mockResolvedValue(new Response());
    vi.stubGlobal("fetch", mock);
    setOriginApiKey("origin-secret");
    await originAuthenticatedFetch("/api/status", {
      headers: { "x-api-key": "explicit" },
    });
    expect(new Headers(mock.mock.calls[0][1].headers).get("x-api-key")).toBe(
      "explicit",
    );
    expect(originApiTarget().apiKey).toBe("origin-secret");
    setOriginApiKey("");
    expect(originApiTarget().apiKey).toBeNull();
    expect(sessionStorage.length).toBe(0);
  });
  it("uses the session key for origin routing and ordinary web API calls", async () => {
    const mock = vi.fn().mockResolvedValue(new Response("[]"));
    vi.stubGlobal("fetch", mock);
    sessionStorage.setItem(
      "mold.web.origin-key.v1:https://foreign.example",
      "foreign",
    );
    expect(originApiTarget().apiKey).toBeNull();
    setOriginApiKey("origin");
    expect(originHost().apiKey).toBe("origin");
    await fetchModels();
    expect(new Headers(mock.mock.calls[0][1].headers).get("x-api-key")).toBe(
      "origin",
    );
    await originAuthenticatedFetch("https://foreign.example/api/status", {
      headers: { "x-api-key": "foreign-explicit" },
      redirect: "follow",
    });
    expect(new Headers(mock.mock.calls[1][1].headers).get("x-api-key")).toBe(
      "foreign-explicit",
    );
    expect(mock.mock.calls[1][1].redirect).toBe("error");
  });
});
