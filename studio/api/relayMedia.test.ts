import { afterEach, expect, it, vi } from "vitest";
import { resolveRelayMedia } from "./relayMedia";
afterEach(() => vi.unstubAllGlobals());
it("polls pending media and validates the ready object URL", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue(
      Response.json({
        state: "ready",
        url: "https://host.example/_mold/objects/a?Signature=short",
        expires_at: 9999999999,
      }),
    );
  vi.stubGlobal("fetch", fetch);
  expect(
    await resolveRelayMedia(
      { relay: { id: "job", state: "pending" } },
      "https://host.example",
      { "x-api-key": "secret" },
    ),
  ).toContain("/_mold/objects/a");
  expect(fetch.mock.calls[0]?.[0]).toBe(
    "https://host.example/_mold/relay/media/job",
  );
  expect(new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-api-key")).toBe(
    "secret",
  );
});
it("retains old media ticket behavior and refuses foreign media", async () => {
  expect(
    await resolveRelayMedia({ token: "old" }, "https://host.example", {}),
  ).toBeNull();
  await expect(
    resolveRelayMedia(
      { url: "https://evil.example/_mold/objects/x" },
      "https://host.example",
      {},
    ),
  ).rejects.toThrow();
});
