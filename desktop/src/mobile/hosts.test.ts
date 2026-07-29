import { describe, expect, it } from "vitest";
import {
  mobileHostMatchesRoute,
  normalizeRemoteAddress,
  remoteHostId,
  type MobileHost,
} from "./hosts";
import type { HostRoute } from "../stores/hosts";

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

  it.each([
    ["URL", { baseUrl: "http://replacement:7680" }],
    ["API key", { apiKey: "replacement-key" }],
    ["instance", { instanceId: "replacement-instance" }],
  ])("rejects a frozen placement route after the host %s changes", (_label, patch) => {
    const route: HostRoute = {
      hostId: "studio-id",
      label: "Studio",
      kind: "remote",
      target: {
        baseUrl: "http://studio.tailnet.ts.net:7680",
        apiKey: "secret",
      },
      instanceId: "studio-instance",
    };
    const host: MobileHost = {
      id: route.hostId,
      name: route.label,
      baseUrl: route.target.baseUrl,
      apiKey: route.target.apiKey ?? "",
      hostname: "studio",
      version: "0.18.0",
      instanceId: route.instanceId ?? undefined,
      online: true,
      ...patch,
    };

    expect(mobileHostMatchesRoute(route, host)).toBe(false);
  });

  it("accepts only the unchanged online host for a frozen placement route", () => {
    const route: HostRoute = {
      hostId: "studio-id",
      label: "Studio",
      kind: "remote",
      target: {
        baseUrl: "http://studio.tailnet.ts.net:7680",
        apiKey: "secret",
      },
      instanceId: "studio-instance",
    };
    expect(
      mobileHostMatchesRoute(route, {
        id: route.hostId,
        name: route.label,
        baseUrl: route.target.baseUrl,
        apiKey: route.target.apiKey ?? "",
        hostname: "studio",
        version: "0.18.0",
        instanceId: route.instanceId ?? undefined,
        online: true,
      }),
    ).toBe(true);
  });
});
