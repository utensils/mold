import { describe, expect, it } from "vitest";
import {
  hostIdFromUrl,
  normalizeHostUrl,
  pickAutoHost,
  pickDisplayHost,
  type RoutableHost,
} from "./hosts";

describe("hostIdFromUrl", () => {
  it("mirrors the Rust host_id slugs exactly", () => {
    // Twin of desktop/src-tauri/src/connection.rs::host_ids_are_stable_slugs —
    // per-host secret names are derived on both sides and must agree.
    expect(hostIdFromUrl("http://hal9000:7680")).toBe("hal9000-7680");
    expect(hostIdFromUrl("https://mold.example.com")).toBe("mold-example-com");
    expect(hostIdFromUrl("https://abc123-7680.proxy.runpod.net")).toBe(
      "abc123-7680-proxy-runpod-net",
    );
    expect(hostIdFromUrl("http://Studio.local:7680")).toBe("studio-local-7680");
    expect(hostIdFromUrl("studio.local:7680")).toBe("studio-local-7680");
  });
});

describe("normalizeHostUrl", () => {
  it("mirrors the Rust normalize_host_url rules", () => {
    expect(normalizeHostUrl("hal9000")).toBe("http://hal9000:7680");
    expect(normalizeHostUrl("studio.local:7680")).toBe("http://studio.local:7680");
    expect(normalizeHostUrl("http://studio.local:7680///")).toBe("http://studio.local:7680");
    expect(normalizeHostUrl("https://mold.example.com/")).toBe("https://mold.example.com");
    expect(() => normalizeHostUrl("   ")).toThrow();
  });
});

function host(overrides: Partial<RoutableHost>): RoutableHost {
  return {
    id: "h",
    kind: "remote",
    status: "ready",
    queueDepth: 0,
    ...overrides,
  };
}

describe("pickAutoHost", () => {
  it("routes to the least-busy ready host", () => {
    const local = host({ id: "local", kind: "local", queueDepth: 3 });
    const idle = host({ id: "idle", queueDepth: 0 });
    const busy = host({ id: "busy", queueDepth: 5 });
    expect(pickAutoHost([local, busy, idle])?.id).toBe("idle");
  });

  it("prefers the local host on a queue-depth tie", () => {
    const local = host({ id: "local", kind: "local", queueDepth: 1 });
    const remote = host({ id: "remote", queueDepth: 1 });
    expect(pickAutoHost([remote, local])?.id).toBe("local");
  });

  it("skips hosts that are not ready and treats unknown depth as busiest", () => {
    const down = host({ id: "down", status: "error", queueDepth: 0 });
    const unknown = host({ id: "unknown", queueDepth: null });
    const known = host({ id: "known", queueDepth: 9 });
    expect(pickAutoHost([down, unknown, known])?.id).toBe("known");
  });

  it("returns null when nothing is ready", () => {
    expect(pickAutoHost([host({ status: "error" })])).toBeNull();
    expect(pickAutoHost([])).toBeNull();
  });
});

describe("pickDisplayHost", () => {
  it("returns the primary when nothing is generating", () => {
    expect(pickDisplayHost([], "local")).toBe("local");
  });

  it("follows the most recently submitted live job's host", () => {
    expect(pickDisplayHost(["local", "hal9000-7680"], "local")).toBe("hal9000-7680");
    expect(pickDisplayHost(["hal9000-7680", "local"], "local")).toBe("local");
  });

  it("skips jobs without a routed host (single-host submissions)", () => {
    expect(pickDisplayHost(["hal9000-7680", null], "local")).toBe("hal9000-7680");
    expect(pickDisplayHost([null, null], "local")).toBe("local");
  });
});
