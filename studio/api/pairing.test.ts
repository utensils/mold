import { describe, expect, it } from "vitest";
import { parseMobilePairingPayload } from "./pairing";

describe("parseMobilePairingPayload", () => {
  it("accepts the versioned one-time pairing envelope", () => {
    expect(
      parseMobilePairingPayload(
        JSON.stringify({
          type: "mold.mobile-pairing",
          version: 1,
          base_url: "http://studio.local:7680",
          token: "one-time",
          expires_at: 1_900_000_000,
          instance_id: "server-id",
          name: "Studio Mac",
        }),
      ),
    ).toMatchObject({
      base_url: "http://studio.local:7680",
      token: "one-time",
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
