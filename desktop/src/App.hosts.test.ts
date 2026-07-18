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
});
