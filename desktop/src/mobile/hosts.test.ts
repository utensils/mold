import { describe, expect, it } from "vitest";
import { normalizeRemoteAddress, remoteHostId } from "./hosts";

describe("mobile remote hosts", () => {
  it("accepts Tailscale MagicDNS names and applies Mold's default port", () => {
    expect(normalizeRemoteAddress("studio.tailnet.ts.net")).toBe(
      "http://studio.tailnet.ts.net:7680",
    );
  });

  it("preserves explicit HTTPS ports", () => {
    expect(normalizeRemoteAddress("https://mold.example.com:8443/")).toBe(
      "https://mold.example.com:8443",
    );
  });

  it("uses the standard HTTPS port when a complete HTTPS URL is entered", () => {
    expect(normalizeRemoteAddress("https://mold.example.com/")).toBe("https://mold.example.com");
  });

  it("creates a stable URL slug for legacy hosts without instance ids", () => {
    expect(remoteHostId("http://192.168.1.20:7680")).toBe("192-168-1-20-7680");
  });
});
