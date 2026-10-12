import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type MobilePairingPayload,
  mobilePairingUrl,
  parseMobilePairingPayload,
} from "./pairing";
import fixtures from "./pairing.fixtures.json";

const payload = {
  type: "mold.mobile-pairing" as const,
  version: 2 as const,
  ...fixtures.links[0]!.payload,
} as MobilePairingPayload;

afterEach(() => vi.unstubAllGlobals());

describe("parseMobilePairingPayload", () => {
  it("rejects credentials and request paths in advertised API origins", () => {
    for (const base_url of [
      "https://user:secret@host.test",
      "https://host.test/api/status",
      "https://host.test?key=secret",
      "https://host.test#secret",
    ]) {
      expect(() =>
        parseMobilePairingPayload(JSON.stringify({ ...payload, base_url })),
      ).toThrow(/pairing code/);
    }
    for (const websocket_url of [
      "https://host.test/ws",
      "wss://user:secret@host.test/ws",
      "wss://host.test/ws?key=secret",
      "wss://host.test/ws#fragment",
    ]) {
      expect(() =>
        parseMobilePairingPayload(
          JSON.stringify({ ...payload, websocket_url }),
        ),
      ).toThrow(/pairing code/);
    }
  });
  it("accepts the versioned one-time pairing envelope", () => {
    expect(parseMobilePairingPayload(JSON.stringify(payload))).toMatchObject({
      base_url: "http://studio.local:7680",
      host_key: fixtures.links[0]!.payload.host_key,
    });
  });

  it("round-trips an app-opening mobile pairing URL", () => {
    const url = mobilePairingUrl(payload);

    expect(url).toMatch(/^https:\/\/utensils\.io\/mold\/pair#/);
    expect(url).not.toContain("api_key");
    expect(parseMobilePairingPayload(url)).toEqual(payload);
  });

  it("refuses old pairings instead of silently downgrading", () => {
    expect(() =>
      parseMobilePairingPayload(JSON.stringify({ ...payload, version: 1 })),
    ).toThrow("not a supported Mold pairing code");
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

describe("the shared pairing-link contract (pairing.fixtures.json)", () => {
  // The Swift port (MoldClient's MobilePairingPayload) reads the same file.
  const envelope = (
    p: (typeof fixtures.links)[number]["payload"],
  ): MobilePairingPayload =>
    ({
      type: "mold.mobile-pairing" as const,
      version: 2 as const,
      ...p,
    }) as MobilePairingPayload;

  it.each(fixtures.links)("prints $link", (fixture) => {
    expect(mobilePairingUrl(envelope(fixture.payload))).toBe(fixture.link);
  });

  it.each(fixtures.links)(
    "reads $link and its equivalent fragment spellings",
    (fixture) => {
      for (const raw of [fixture.link, ...fixture.aliases]) {
        expect(parseMobilePairingPayload(raw)).toEqual(
          envelope(fixture.payload),
        );
      }
    },
  );

  it.each(fixtures.rejected)("refuses %s", (raw) => {
    expect(() => parseMobilePairingPayload(raw)).toThrow(/pairing code/);
  });
});

describe("Regular pairing", () => {
  const regular = {
    type: "mold.mobile-pairing",
    version: 2,
    mode: "regular",
    base_url: "http://203.0.113.7:7680",
    instance_id: "host-instance",
    name: "Headless",
    pairing_token: "a".repeat(64),
    expires_at: 4000000000,
  };
  it("round-trips an explicit Regular invitation without any secure fields", () => {
    const parsed = parseMobilePairingPayload(JSON.stringify(regular));
    expect(parsed).toEqual(regular);
    expect(parseMobilePairingPayload(mobilePairingUrl(parsed))).toEqual(
      regular,
    );
  });
  it("rejects mixed authority and proxy addresses", () => {
    for (const extra of [
      { base_url: "https://x.mold-link.urandom.io" },
      { host_key: "a".repeat(64) },
      { endpoints: [{ kind: "relay", url: "https://proxy.example" }] },
      { websocket_url: "wss://proxy.example" },
    ]) {
      expect(() =>
        parseMobilePairingPayload(JSON.stringify({ ...regular, ...extra })),
      ).toThrow(/pairing code/);
    }
  });
});

it("accepts the backend Regular credential shape and rejects Secure claim keys", async () => {
  const { claimPairingSession } = await import("./pairing");
  const invitation = parseMobilePairingPayload(
    JSON.stringify({
      type: "mold.mobile-pairing",
      version: 2,
      mode: "regular",
      base_url: "http://203.0.113.7:7680",
      instance_id: "regular-instance",
      name: "Host",
      pairing_token: "a".repeat(64),
      expires_at: 4000000000,
    }),
  );
  const apiKey = `mold_regular_${"b".repeat(64)}`;
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            mode: "regular",
            api_key: apiKey,
            instance_id: "regular-instance",
            hostname: "Host",
            base_url: invitation.base_url,
          }),
          { headers: { "content-type": "application/json" } },
        ),
    ),
  );
  await expect(claimPairingSession(invitation)).resolves.toMatchObject({
    mode: "regular",
    api_key: apiKey,
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            mode: "regular",
            api_key: "mold_pair_secure",
            instance_id: "regular-instance",
            hostname: "Host",
            base_url: invitation.base_url,
          }),
          { headers: { "content-type": "application/json" } },
        ),
    ),
  );
  await expect(claimPairingSession(invitation)).rejects.toThrow(
    "invalid Regular pairing",
  );
});
