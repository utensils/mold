import { afterEach, describe, expect, it, vi } from "vitest";
import {
  claimPairingSession,
  mobilePairingUrl,
  parseMobilePairingPayload,
} from "./pairing";

const payload = {
  type: "mold.mobile-pairing" as const,
  version: 1 as const,
  base_url: "http://studio.local:7680",
  token: "one-time",
  expires_at: 1_900_000_000,
  instance_id: "server-id",
  name: "Studio Mac",
};

afterEach(() => vi.unstubAllGlobals());

describe("parseMobilePairingPayload", () => {
  it("accepts the versioned one-time pairing envelope", () => {
    expect(parseMobilePairingPayload(JSON.stringify(payload))).toMatchObject({
      base_url: "http://studio.local:7680",
      token: "one-time",
    });
  });

  it("round-trips an app-opening mobile pairing URL", () => {
    const url = mobilePairingUrl(payload);

    expect(url).toMatch(/^mold:\/\/pair\?/);
    expect(url).not.toContain("api_key");
    expect(parseMobilePairingPayload(url)).toEqual(payload);
  });

  it("preserves credential-free pairing in an app-opening URL", () => {
    const url = mobilePairingUrl({ ...payload, token: null, expires_at: null });

    expect(parseMobilePairingPayload(url)).toEqual({
      ...payload,
      token: null,
      expires_at: null,
    });
  });

  it("rejects foreign, malformed, and non-http codes", () => {
    expect(() => parseMobilePairingPayload("not json")).toThrow(
      "not a Mold pairing code",
    );
    expect(() =>
      parseMobilePairingPayload(
        JSON.stringify({
          type: "mold.mobile-pairing",
          version: 2,
          base_url: "file:///tmp/key",
          token: null,
          expires_at: null,
          instance_id: "server-id",
          name: "Studio",
        }),
      ),
    ).toThrow("not a supported Mold pairing code");
  });
});

describe("claimPairingSession", () => {
  it("identifies the client without putting its new credential in the request", async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            api_key: "mold_pair_secret",
            instance_id: "host-id",
            hostname: "studio",
          }),
          { headers: { "content-type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetch);

    await claimPairingSession("http://studio:7680", "one-use", {
      name: "Mold on iPhone",
      kind: "iphone",
    });

    const [, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      token: "one-use",
      client_name: "Mold on iPhone",
      client_kind: "iphone",
    });
    expect(String(init.body)).not.toContain("mold_pair_secret");
  });
});
