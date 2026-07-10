import { describe, expect, it } from "vitest";
import { addressLabel, dedupeHosts, prepareHosts, sortHosts, versionLabel } from "./discovery";
import type { DiscoveredHost } from "./ipc";

function host(overrides: Partial<DiscoveredHost> = {}): DiscoveredHost {
  return {
    name: "box-7680",
    url: "http://192.168.1.11:7680",
    host: "192.168.1.11",
    port: 7680,
    version: "0.14.0",
    authRequired: false,
    isThisMachine: false,
    ...overrides,
  };
}

describe("sortHosts", () => {
  it("puts this machine first", () => {
    const others = host({ name: "aaa-7680", url: "http://1/" });
    const self = host({ name: "zzz-7680", url: "http://2/", isThisMachine: true });
    const sorted = sortHosts([others, self]);
    expect(sorted[0]).toBe(self);
  });

  it("sorts the rest by name case-insensitively", () => {
    const b = host({ name: "Bravo-7680", url: "http://b/" });
    const a = host({ name: "alpha-7680", url: "http://a/" });
    const sorted = sortHosts([b, a]);
    expect(sorted.map((h) => h.name)).toEqual(["alpha-7680", "Bravo-7680"]);
  });

  it("does not mutate the input array", () => {
    const input = [host({ name: "b" }), host({ name: "a" })];
    const copy = [...input];
    sortHosts(input);
    expect(input).toEqual(copy);
  });
});

describe("dedupeHosts", () => {
  it("removes duplicate urls, keeping the first", () => {
    const first = host({ name: "first", url: "http://same/" });
    const dup = host({ name: "second", url: "http://same/" });
    const other = host({ url: "http://other/" });
    const out = dedupeHosts([first, dup, other]);
    expect(out).toHaveLength(2);
    expect(out[0]!.name).toBe("first");
  });
});

describe("prepareHosts", () => {
  it("dedupes then sorts self-first", () => {
    const self = host({ name: "self", url: "http://self/", isThisMachine: true });
    const dup = host({ name: "self", url: "http://self/", isThisMachine: true });
    const remote = host({ name: "remote", url: "http://remote/" });
    const out = prepareHosts([remote, self, dup]);
    expect(out).toHaveLength(2);
    expect(out[0]!.isThisMachine).toBe(true);
  });
});

describe("labels", () => {
  it("addressLabel is host:port", () => {
    expect(addressLabel(host({ host: "10.0.0.5", port: 7681 }))).toBe("10.0.0.5:7681");
  });

  it("versionLabel falls back when unknown", () => {
    expect(versionLabel(host({ version: "0.14.0" }))).toBe("mold 0.14.0");
    expect(versionLabel(host({ version: null }))).toBe("mold");
  });
});
