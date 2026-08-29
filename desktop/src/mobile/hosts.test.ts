import { describe, expect, it } from "vitest";
import {
  mobileHostHealthLabel,
  mobileHostMatchesRoute,
  mergeMobileHostsByInstanceId,
  normalizeRemoteAddress,
  recordMobileHostAuthorityRejection,
  recordMobileHostProbeFailure,
  recordMobileHostStatus,
  remoteHostId,
  type MobileHost,
} from "./hosts";
import type { ServerStatus } from "../lib/api/types";
import type { HostRoute } from "../stores/hosts";

function status(overrides: Partial<ServerStatus> = {}): ServerStatus {
  return {
    version: "0.18.0",
    models_loaded: [],
    uptime_secs: 60,
    hostname: "studio",
    instance_id: "studio-instance",
    ...overrides,
  };
}

function verifiedHost(overrides: Partial<MobileHost> = {}): MobileHost {
  return {
    id: "studio-id",
    name: "Studio",
    baseUrl: "http://studio.tailnet.ts.net:7680",
    apiKey: "secret",
    hostname: "studio",
    version: "0.18.0",
    instanceId: "studio-instance",
    online: true,
    ...overrides,
  };
}

describe("mobile remote hosts", () => {
  it("collapses IP and hostname aliases by UUID onto the last successful address", () => {
    const byName = verifiedHost({
      id: "hal9000-7680",
      baseUrl: "http://hal9000:7680",
      instanceId: "same-uuid",
      lastConnectedAtMs: 10,
    });
    const byIp = verifiedHost({
      id: "100-123-198-98-7681",
      baseUrl: "http://100.123.198.98:7681",
      instanceId: "same-uuid",
      lastConnectedAtMs: 20,
    });

    expect(mergeMobileHostsByInstanceId([byName, byIp])).toEqual({
      hosts: [byIp],
      dropped: [{ loser: byName.id, survivor: byIp.id }],
    });
  });

  it("does not merge missing or different server UUIDs", () => {
    const unknown = verifiedHost({ id: "unknown", instanceId: " " });
    const first = verifiedHost({ id: "first", instanceId: "uuid-1" });
    const second = verifiedHost({ id: "second", instanceId: "uuid-2" });
    expect(mergeMobileHostsByInstanceId([unknown, first, second]).hosts).toEqual([
      unknown,
      first,
      second,
    ]);
  });

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

  it("uses the HTTPS scheme default when a complete URL omits a port", () => {
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

  it("keeps a verified host stale through one or many transient failures, then recovers", () => {
    const host = verifiedHost();

    recordMobileHostProbeFailure(host, new Error("status timeout"));
    expect(host).toMatchObject({
      online: true,
      stale: true,
      instanceId: "studio-instance",
      version: "0.18.0",
      healthError: "status timeout",
    });
    expect(mobileHostHealthLabel(host)).toBe("reconnecting…");

    recordMobileHostProbeFailure(host, new Error("backend busy"));
    expect(host).toMatchObject({
      online: true,
      stale: true,
      instanceId: "studio-instance",
      healthError: "backend busy",
    });

    expect(
      recordMobileHostStatus(
        host,
        status({ version: "0.19.0", uptime_secs: 90, instance_id: "studio-instance" }),
      ),
    ).toBe("verified");
    expect(host).toMatchObject({
      online: true,
      stale: false,
      instanceId: "studio-instance",
      version: "0.19.0",
    });
    expect(host.healthError).toBeUndefined();
    expect(mobileHostHealthLabel(host)).toBe("v0.19.0");
  });

  it("keeps a never-verified host unreachable and non-authoritative", () => {
    const host = verifiedHost({ online: false, instanceId: undefined, version: undefined });

    recordMobileHostProbeFailure(host, new Error("network unreachable"));

    expect(host).toMatchObject({
      online: false,
      stale: false,
      healthError: "network unreachable",
    });
    expect(mobileHostHealthLabel(host)).toBe("unreachable");
  });

  it("fences an explicit instance mismatch without adopting replacement identity", () => {
    const host = verifiedHost();

    expect(recordMobileHostStatus(host, status({ instance_id: "replacement-instance" }))).toBe(
      "instance_mismatch",
    );
    expect(host).toMatchObject({
      online: false,
      stale: false,
      instanceId: "studio-instance",
      instanceMismatch: {
        expected: "studio-instance",
        reported: "replacement-instance",
      },
    });
    expect(mobileHostHealthLabel(host)).toBe("identity changed");

    recordMobileHostProbeFailure(host, new Error("replacement timed out"));
    expect(host.instanceMismatch).toEqual({
      expected: "studio-instance",
      reported: "replacement-instance",
    });
    expect(host.online).toBe(false);
  });

  it("retires last-good authority when the exact credential is rejected", () => {
    const host = verifiedHost();

    recordMobileHostAuthorityRejection(host, new Error("API key was rejected"));

    expect(host).toMatchObject({
      online: false,
      stale: false,
      authorityRejected: true,
      instanceId: "studio-instance",
      healthError: "API key was rejected",
    });
    expect(mobileHostHealthLabel(host)).toBe("access denied");
  });
});
