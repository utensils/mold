import { describe, expect, it } from "vitest";
import source from "./App.vue?raw";

describe("desktop host bootstrap", () => {
  it("starts remembered-host reconnect without waiting for This Mac", () => {
    const reconnect = source.indexOf("const hostStartup = hostsStore.init()");
    const local = source.indexOf("const connectionStartup = connection.init()");
    const waitForBoth = source.indexOf(
      "await Promise.allSettled([connectionStartup, hostStartup])",
    );

    expect(reconnect).toBeGreaterThan(-1);
    expect(local).toBeGreaterThan(-1);
    expect(waitForBoth).toBeGreaterThan(-1);
    expect(reconnect).toBeLessThan(waitForBoth);
    expect(local).toBeLessThan(waitForBoth);
  });

  it("keeps offline hosts actionable from the compact toast", () => {
    // The copy itself lives in the shared studio policy so web says the same.
    expect(source).toContain("HOST_OFFLINE_DESCRIPTION");
    expect(source).toContain('label: "Open Machines"');
    expect(source).toContain('router.push("/machines")');
    expect(source).toContain("sticky: true");
  });

  it("warns on a drop and confirms the automatic reconnect in green", () => {
    // A host that stays listed and is re-probed every poll is a warning, not a
    // failure; the recovery withdraws that toast and reports success.
    expect(source).toContain('toasts.push(hostOfflineTitle(host.label), "warning"');
    expect(source).toContain('toasts.push(hostReconnectedTitle(host.label), "success")');
    expect(source).toContain("detectReconnectTransitions(hostStatusSnapshot, current)");
    expect(source).toContain("toasts.dismiss(stale)");
  });
});
