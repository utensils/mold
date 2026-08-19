import { describe, expect, it } from "vitest";
import {
  HOST_OFFLINE_DESCRIPTION,
  HOST_RECONNECTING_LABEL,
  detectOfflineTransitions,
  detectReconnectTransitions,
  hostOfflineTitle,
  hostReconnectedTitle,
  reconcileHostConnectivity,
  snapshotHostStatuses,
} from "./hostConnectivity";

function hosts(...rows: [string, string][]) {
  return rows.map(([id, status]) => ({ id, label: id, status }));
}

describe("detectOfflineTransitions", () => {
  it("reports only hosts that went ready → error", () => {
    const previous = snapshotHostStatuses(
      hosts(["a", "ready"], ["b", "ready"]),
    );
    const offline = detectOfflineTransitions(
      previous,
      hosts(["a", "error"], ["b", "ready"]),
    );
    expect(offline.map((h) => h.id)).toEqual(["a"]);
  });

  it("stays quiet while a host remains errored across polls", () => {
    const previous = snapshotHostStatuses(hosts(["a", "error"]));
    expect(detectOfflineTransitions(previous, hosts(["a", "error"]))).toEqual(
      [],
    );
  });

  it("stays quiet for a host that was never reachable", () => {
    expect(detectOfflineTransitions({}, hosts(["a", "error"]))).toEqual([]);
    const previous = snapshotHostStatuses(hosts(["a", "connecting"]));
    expect(detectOfflineTransitions(previous, hosts(["a", "error"]))).toEqual(
      [],
    );
  });
});

describe("detectReconnectTransitions", () => {
  it("reports hosts that came back error → ready", () => {
    const previous = snapshotHostStatuses(
      hosts(["a", "error"], ["b", "error"]),
    );
    const back = detectReconnectTransitions(
      previous,
      hosts(["a", "ready"], ["b", "error"]),
    );
    expect(back.map((h) => h.id)).toEqual(["a"]);
  });

  it("does not celebrate a host that was already reachable", () => {
    const previous = snapshotHostStatuses(hosts(["a", "ready"]));
    expect(detectReconnectTransitions(previous, hosts(["a", "ready"]))).toEqual(
      [],
    );
  });

  it("does not celebrate a first successful connection", () => {
    expect(detectReconnectTransitions({}, hosts(["a", "ready"]))).toEqual([]);
    const previous = snapshotHostStatuses(hosts(["a", "connecting"]));
    expect(detectReconnectTransitions(previous, hosts(["a", "ready"]))).toEqual(
      [],
    );
  });

  it("treats a retry that lands after an explicit reconnect attempt as recovery", () => {
    // The manual Retry drives error → connecting → ready. "connecting" is a
    // probe, not evidence of reachability, so the snapshot carries the last
    // settled status forward and the recovery still fires exactly once.
    const errored = snapshotHostStatuses(hosts(["a", "error"]));
    expect(
      detectReconnectTransitions(errored, hosts(["a", "connecting"])),
    ).toEqual([]);
    const probing = snapshotHostStatuses(hosts(["a", "connecting"]), errored);
    expect(probing["a"]).toBe("error");
    expect(
      detectReconnectTransitions(probing, hosts(["a", "ready"])).map(
        (h) => h.id,
      ),
    ).toEqual(["a"]);
  });
});

describe("copy", () => {
  it("names the host in both directions and promises the retry", () => {
    expect(hostOfflineTitle("plato")).toBe("Can't reach plato");
    expect(hostReconnectedTitle("plato")).toBe("Reconnected to plato");
    expect(HOST_OFFLINE_DESCRIPTION).toContain("Retrying");
    expect(HOST_RECONNECTING_LABEL).toBe("reconnecting…");
  });
});

describe("reconcileHostConnectivity", () => {
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
