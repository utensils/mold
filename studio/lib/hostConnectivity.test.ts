import { describe, expect, it } from "vitest";
import {
  HOST_OFFLINE_DESCRIPTION,
  HOST_RECONNECTING_LABEL,
  detectOfflineTransitions,
  detectReconnectTransitions,
  hostOfflineTitle,
  hostReconnectedTitle,
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
