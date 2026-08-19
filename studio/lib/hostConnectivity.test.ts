import { describe, expect, it } from "vitest";
import {
  HOST_OFFLINE_DESCRIPTION,
  HOST_RECONNECTING_LABEL,
  hostOfflineTitle,
  hostReconnectedTitle,
  reconcileHostConnectivity,
  snapshotHostStatuses,
} from "./hostConnectivity";

function hosts(...rows: [string, string][]) {
  return rows.map(([id, status]) => ({ id, label: id, status }));
}

describe("copy", () => {
  it("names the host in both directions and promises the retry", () => {
    expect(hostOfflineTitle("plato")).toBe("Can't reach plato");
    expect(hostReconnectedTitle("plato")).toBe("Reconnected to plato");
    expect(HOST_OFFLINE_DESCRIPTION).toContain("Retrying");
    expect(HOST_RECONNECTING_LABEL).toBe("reconnecting…");
  });
});

describe("reconcileHostConnectivity", () => {
  it("reports a drop once per ready → error edge and never for a first contact by default", () => {
    const previous = snapshotHostStatuses(
      hosts(["a", "ready"], ["b", "ready"]),
    );
    const changes = reconcileHostConnectivity({
      previous,
      current: hosts(["a", "error"], ["b", "ready"], ["c", "error"]),
      warned: [],
    });
    // "c" has never been reachable — desktop stays quiet about it.
    expect(changes.offline.map((h) => h.id)).toEqual(["a"]);
  });

  const base = { previous: {}, current: [], warned: [] as string[] };

  it("reports a drop once and then stays quiet while it is still warned", () => {
    const first = reconcileHostConnectivity({
      ...base,
      previous: snapshotHostStatuses(hosts(["a", "ready"])),
      current: hosts(["a", "error"]),
    });
    expect(first.offline.map((h) => h.id)).toEqual(["a"]);

    const second = reconcileHostConnectivity({
      previous: first.next,
      current: hosts(["a", "error"]),
      warned: ["a"],
    });
    expect(second.offline).toEqual([]);
    expect(second.reconnected).toEqual([]);
  });

  it("never celebrates a recovery it did not warn about", () => {
    // The desktop boot probe is deliberately quiet, so a machine asleep at
    // launch is errored but unannounced. It coming back is not news.
    const changes = reconcileHostConnectivity({
      previous: snapshotHostStatuses(hosts(["a", "error"])),
      current: hosts(["a", "ready"]),
      warned: [],
    });
    expect(changes.reconnected).toEqual([]);
    expect(
      reconcileHostConnectivity({
        previous: snapshotHostStatuses(hosts(["a", "error"])),
        current: hosts(["a", "ready"]),
        warned: ["a"],
      }).reconnected.map((h) => h.id),
    ).toEqual(["a"]);
  });

  it("only warns about a never-reachable host when the surface asks for it", () => {
    const quiet = reconcileHostConnectivity({
      ...base,
      current: hosts(["a", "error"]),
    });
    expect(quiet.offline).toEqual([]);
    const loud = reconcileHostConnectivity({
      ...base,
      current: hosts(["a", "error"]),
      warnOnFirstContact: true,
    });
    expect(loud.offline.map((h) => h.id)).toEqual(["a"]);
  });

  it("retires a notice whose host left the list", () => {
    const changes = reconcileHostConnectivity({
      previous: snapshotHostStatuses(hosts(["a", "error"], ["b", "ready"])),
      current: hosts(["b", "ready"]),
      warned: ["a"],
    });
    expect(changes.retired).toEqual(["a"]);
    expect(changes.reconnected).toEqual([]);
    expect(changes.next).toEqual({ b: "ready" });
  });

  it("carries the settled status forward through a probe", () => {
    const changes = reconcileHostConnectivity({
      previous: snapshotHostStatuses(hosts(["a", "error"])),
      current: hosts(["a", "connecting"]),
      warned: ["a"],
    });
    expect(changes.next["a"]).toBe("error");
    expect(changes.offline).toEqual([]);
    expect(changes.reconnected).toEqual([]);
  });
});
