import { describe, expect, it } from "vitest";
import {
  MOBILE_GENERATE_TARGET_KEY,
  loadMobileGenerateTarget,
  mobileAutoRoutingAvailable,
  mobileGenerateTargetLabel,
  mobileModelAvailabilityTag,
  mobileRoutingHosts,
  resolveMobileGenerateTarget,
  saveMobileGenerateTarget,
} from "./generateTarget";
import type { MobileHost } from "./hosts";

function host(overrides: Partial<MobileHost> & { id: string }): MobileHost {
  return {
    name: overrides.id,
    baseUrl: `http://${overrides.id}:7680`,
    apiKey: "",
    hostname: undefined,
    version: undefined,
    online: true,
    ...overrides,
  };
}

function memoryStorage(initial: Record<string, string> = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    store,
  };
}

describe("mobile automatic-routing visibility", () => {
  it("needs two reachable connected machines", () => {
    expect(mobileAutoRoutingAvailable([host({ id: "studio" })])).toBe(false);
    expect(mobileAutoRoutingAvailable([host({ id: "studio" }), host({ id: "plato" })])).toBe(true);
  });

  it("does not count offline or disconnected machines", () => {
    expect(
      mobileAutoRoutingAvailable([host({ id: "studio" }), host({ id: "plato", online: false })]),
    ).toBe(false);
    expect(
      mobileAutoRoutingAvailable([host({ id: "studio" }), host({ id: "plato", connected: false })]),
    ).toBe(false);
    expect(mobileRoutingHosts([host({ id: "a" }), host({ id: "b", online: false })])).toHaveLength(
      1,
    );
  });
});

describe("mobile generate-target persistence", () => {
  it("defaults a fresh install to Auto and round-trips a saved value", () => {
    const storage = memoryStorage();
    expect(loadMobileGenerateTarget(storage)).toBe("auto");
    saveMobileGenerateTarget("capable", storage);
    expect(storage.store.get(MOBILE_GENERATE_TARGET_KEY)).toBe("capable");
    expect(loadMobileGenerateTarget(storage)).toBe("capable");
    saveMobileGenerateTarget("plato", storage);
    expect(loadMobileGenerateTarget(storage)).toBe("plato");
  });

  it("survives storage that refuses to write", () => {
    expect(() =>
      saveMobileGenerateTarget("auto", {
        getItem: () => null,
        setItem: () => {
          throw new Error("quota exceeded");
        },
      }),
    ).not.toThrow();
  });
});

describe("resolveMobileGenerateTarget", () => {
  const fleet = [host({ id: "studio" }), host({ id: "plato" })];

  it("keeps an automatic policy while two machines are reachable", () => {
    expect(resolveMobileGenerateTarget("auto", fleet, "studio")).toBe("auto");
    expect(resolveMobileGenerateTarget("capable", fleet, "studio")).toBe("capable");
  });

  it("degrades to the browsed machine while only one is reachable", () => {
    const single = [host({ id: "studio" }), host({ id: "plato", online: false })];
    expect(resolveMobileGenerateTarget("auto", single, "studio")).toBe("studio");
    expect(resolveMobileGenerateTarget("capable", single, "studio")).toBe("studio");
    // The saved value is untouched, so reconnecting restores Auto.
    expect(resolveMobileGenerateTarget("auto", fleet, "studio")).toBe("auto");
  });

  it("honours a pinned machine and drops a forgotten one", () => {
    expect(resolveMobileGenerateTarget("plato", fleet, "studio")).toBe("plato");
    expect(resolveMobileGenerateTarget("ghost", fleet, "studio")).toBe("auto");
    const single = [host({ id: "studio" })];
    expect(resolveMobileGenerateTarget("ghost", single, "studio")).toBe("studio");
    expect(resolveMobileGenerateTarget("ghost", [], "")).toBe("");
  });
});

describe("labels", () => {
  const fleet = [host({ id: "studio", name: "Studio" }), host({ id: "plato", name: "plato" })];

  it("names the policies and machines", () => {
    expect(mobileGenerateTargetLabel("auto", fleet)).toBe("Auto");
    expect(mobileGenerateTargetLabel("capable", fleet)).toBe("Most capable");
    expect(mobileGenerateTargetLabel("studio", fleet)).toBe("Studio");
    expect(mobileGenerateTargetLabel("ghost", fleet)).toBe("ghost");
  });

  it("tags a model only when it is not on every reachable machine", () => {
    expect(mobileModelAvailabilityTag(["studio"], fleet)).toBe("Studio");
    expect(mobileModelAvailabilityTag(["studio", "plato"], fleet)).toBeNull();
    expect(mobileModelAvailabilityTag([], fleet)).toBeNull();
    const three = [...fleet, host({ id: "hal", name: "hal9000" })];
    expect(mobileModelAvailabilityTag(["studio", "plato"], three)).toBe("2 machines");
  });
});
