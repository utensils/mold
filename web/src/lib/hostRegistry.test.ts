import { beforeEach, describe, expect, it } from "vitest";
import {
  ORIGIN_HOST_ID,
  HOSTS_STORAGE_KEY,
  addHost,
  dedupeByInstanceId,
  getGenerateTargetId,
  hostIdFromUrl,
  listHosts,
  listKnownHosts,
  listStoredHosts,
  normalizeHostAddress,
  originHost,
  recordSuccessfulHostInstance,
  reconcileOriginInstanceId,
  removeHost,
  setGenerateTargetId,
  setHostConnected,
  updateHost,
} from "./hostRegistry";

beforeEach(() => {
  localStorage.clear();
});

describe("normalizeHostAddress", () => {
  it("defaults schemeless input to http and keeps the port", () => {
    expect(normalizeHostAddress("192.168.1.42:51789")).toBe(
      "http://192.168.1.42:51789",
    );
  });

  it("keeps an explicit https scheme (MagicDNS / TLS host)", () => {
    expect(normalizeHostAddress("https://box.tail1234.ts.net")).toBe(
      "https://box.tail1234.ts.net",
    );
  });

  it("drops any path, query, and trailing slash to the bare origin", () => {
    expect(
      normalizeHostAddress("http://studio.local:7680/api/status?x=1"),
    ).toBe("http://studio.local:7680");
  });

  it("defaults a bare hostname to mold's port, like desktop and iOS", () => {
    expect(normalizeHostAddress("studio.local")).toBe(
      "http://studio.local:7680",
    );
  });

  it("defaults a bare IP to mold's port", () => {
    expect(normalizeHostAddress("100.105.134.43")).toBe(
      "http://100.105.134.43:7680",
    );
  });

  it("uses the scheme default when a complete URL omits a port", () => {
    expect(normalizeHostAddress("http://100.105.134.43")).toBe(
      "http://100.105.134.43",
    );
  });

  it("returns null for empty or unparseable input", () => {
    expect(normalizeHostAddress("   ")).toBeNull();
    expect(normalizeHostAddress("http://")).toBeNull();
  });
});

describe("hostIdFromUrl", () => {
  it("slugifies host and port", () => {
    expect(hostIdFromUrl("http://192.168.1.42:51789")).toBe(
      "192-168-1-42-51789",
    );
  });

  it("never collides with the reserved primary id", () => {
    expect(hostIdFromUrl("http://origin")).toBe("origin-1");
  });
});

describe("primary host immutability", () => {
  it("always lists the origin first as an unstored 'this server'", () => {
    const hosts = listHosts();
    expect(hosts[0]?.id).toBe(ORIGIN_HOST_ID);
    expect(hosts[0]?.name).toBe("this server");
    expect(hosts[0]?.url).toBe(originHost().url);
  });

  it("refuses to store, update, or remove the primary", () => {
    expect(updateHost(ORIGIN_HOST_ID, { name: "hacked" })).toBeNull();
    removeHost(ORIGIN_HOST_ID);
    expect(listHosts()[0]?.id).toBe(ORIGIN_HOST_ID);
    // Adding the origin URL again does not create a stored duplicate.
    addHost({ url: originHost().url, name: "dup" });
    expect(listStoredHosts()).toHaveLength(0);
  });
});

describe("CRUD", () => {
  it("adds a remote host and lists it after the origin", () => {
    const entry = addHost({ url: "192.168.1.20:7680", name: "Studio" });
    expect(entry.id).toBe("192-168-1-20-7680");
    expect(entry.url).toBe("http://192.168.1.20:7680");
    const hosts = listHosts();
    expect(hosts).toHaveLength(2);
    expect(hosts[1]?.name).toBe("Studio");
  });

  it("stores an api key on the entry, never in the url", () => {
    const entry = addHost({ url: "box.local", name: "Box", apiKey: "sekret" });
    expect(entry.apiKey).toBe("sekret");
    expect(entry.url).not.toContain("sekret");
  });

  it("updates a stored host's name and key", () => {
    const entry = addHost({ url: "box.local", name: "Box" });
    const updated = updateHost(entry.id, { name: "Bench", apiKey: "k2" });
    expect(updated?.name).toBe("Bench");
    expect(updated?.apiKey).toBe("k2");
    expect(listStoredHosts()[0]?.name).toBe("Bench");
  });

  it("removes a stored host", () => {
    const entry = addHost({ url: "box.local", name: "Box" });
    removeHost(entry.id);
    expect(listStoredHosts()).toHaveLength(0);
  });

  it("keeps a disconnected host remembered but out of the active mix", () => {
    const entry = addHost({ url: "box.local", name: "Box", apiKey: "secret" });
    setHostConnected(entry.id, false);
    expect(listHosts().map((host) => host.id)).not.toContain(entry.id);
    expect(listKnownHosts().find((host) => host.id === entry.id)).toMatchObject(
      {
        connected: false,
        apiKey: "secret",
      },
    );

    setHostConnected(entry.id, true);
    expect(listHosts().map((host) => host.id)).toContain(entry.id);
  });
});

describe("dedupe by instance id", () => {
  it("finds an existing entry recorded with the same instance id", () => {
    addHost({ url: "192.168.1.20:7680", name: "Studio", instanceId: "uuid-1" });
    const match = dedupeByInstanceId("uuid-1");
    expect(match?.name).toBe("Studio");
    expect(dedupeByInstanceId("nope")).toBeNull();
    expect(dedupeByInstanceId("")).toBeNull();
  });

  it("merges a re-add reached by a different address into the earliest entry", () => {
    const first = addHost({
      url: "192.168.1.20:7680",
      name: "Studio",
      instanceId: "uuid-1",
    });
    const merged = addHost({
      url: "studio.local:7680",
      name: "Studio (mDNS)",
      instanceId: "uuid-1",
    });
    // Same row: keeps the earliest id, updates url + name.
    expect(merged.id).toBe(first.id);
    expect(merged.url).toBe("http://studio.local:7680");
    expect(merged.name).toBe("Studio (mDNS)");
    expect(listStoredHosts()).toHaveLength(1);
  });

  it("migrates stored aliases onto the most recently successful address", () => {
    localStorage.setItem(
      HOSTS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "hal9000-7680",
          name: "hal9000",
          url: "http://hal9000:7680",
          instanceId: "uuid-1",
          lastConnectedAtMs: 10,
        },
        {
          id: "100-123-198-98-7681",
          name: "hal9000",
          url: "http://100.123.198.98:7681",
          instanceId: "uuid-1",
          lastConnectedAtMs: 20,
        },
      ]),
    );
    setGenerateTargetId("hal9000-7680");

    expect(listStoredHosts()).toEqual([
      expect.objectContaining({
        id: "100-123-198-98-7681",
        url: "http://100.123.198.98:7681",
      }),
    ]);
    expect(getGenerateTargetId()).toBe("100-123-198-98-7681");
  });

  it("persists a polled UUID, selects that working address, and remaps recovery state", () => {
    const hostname = addHost({ url: "hal9000:7680", name: "hal9000", instanceId: "uuid-1" });
    const ip = addHost({ url: "100.123.198.98:7681", name: "hal9000 IP" });
    localStorage.setItem(
      "mold.create.tracked-sequences.v1",
      JSON.stringify([{ hostId: hostname.id, jobId: "chain-1" }]),
    );
    localStorage.setItem(
      "mold.generate.jobs",
      JSON.stringify({ version: 1, jobs: [{ hostId: hostname.id }] }),
    );
    localStorage.setItem(
      "mold.generate.jobs.recovery.batch-1",
      JSON.stringify({ version: 1, jobs: [{ hostId: hostname.id }] }),
    );
    setGenerateTargetId(hostname.id);

    const canonical = recordSuccessfulHostInstance(ip.id, " uuid-1 ");

    expect(canonical).toMatchObject({ id: ip.id, url: "http://100.123.198.98:7681" });
    expect(listStoredHosts()).toHaveLength(1);
    expect(getGenerateTargetId()).toBe(ip.id);
    expect(localStorage.getItem("mold.create.tracked-sequences.v1")).toContain(ip.id);
    expect(localStorage.getItem("mold.generate.jobs")).toContain(ip.id);
    expect(localStorage.getItem("mold.generate.jobs.recovery.batch-1")).toContain(ip.id);
  });

  it("keeps different UUIDs separate even when the URL slug is reused", () => {
    const old = addHost({ url: "render:7680", name: "Old", instanceId: "uuid-old" });
    const replacement = addHost({
      url: "render:7680",
      name: "Replacement",
      instanceId: "uuid-new",
    });
    expect(replacement.id).not.toBe(old.id);
    expect(listStoredHosts()).toHaveLength(2);
    expect(listStoredHosts().find((host) => host.id === old.id)?.connected).toBe(false);
  });

  it("removes a stored alias when its UUID is the browser origin", () => {
    const alias = addHost({
      url: "127.0.0.1:7680",
      name: "Local alias",
      instanceId: "origin-uuid",
    });
    setGenerateTargetId(alias.id);

    reconcileOriginInstanceId(" origin-uuid ");

    expect(listStoredHosts()).toEqual([]);
    expect(getGenerateTargetId()).toBe(ORIGIN_HOST_ID);
  });

  it("merges a re-add of the same address by slug", () => {
    addHost({ url: "192.168.1.20:7680", name: "Studio" });
    addHost({ url: "http://192.168.1.20:7680", name: "Studio 2" });
    expect(listStoredHosts()).toHaveLength(1);
    expect(listStoredHosts()[0]?.name).toBe("Studio 2");
  });
});

describe("generation target", () => {
  it("defaults to model-aware Auto and persists a selection", () => {
    // Desktop parity (null = auto): a fresh install pinned to the origin hid
    // every connected machine's models from the Create picker.
    expect(getGenerateTargetId()).toBe("auto");
    setGenerateTargetId("192-168-1-20-7680");
    expect(getGenerateTargetId()).toBe("192-168-1-20-7680");
  });
});
