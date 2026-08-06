import { describe, expect, it } from "vitest";
import type { PreparedExpansionInputs } from "../lib/preparedExpansion";
import type { HostRoute } from "../stores/hosts";
import {
  createMobileExpansionRecovery,
  mobileExpansionRecoveryStaleReason,
} from "./mobileExpansionRecovery";

const inputs: PreparedExpansionInputs = {
  sourcePrompt: "a lighthouse",
  model: "flux-dev:q8",
  family: "flux",
  task: "text-to-image",
  requestedCount: 3,
  stylePreset: null,
  selectedHostPolicy: "studio",
};
const route: HostRoute = {
  hostId: "studio",
  label: "Studio",
  kind: "remote",
  target: { baseUrl: "https://studio.test:7680", apiKey: "secret" },
  instanceId: "studio-instance",
};

describe("mobile expansion recovery", () => {
  it("deep-snapshots inputs and derives its pull host only from the frozen route", () => {
    const mutableInputs = { ...inputs };
    const mutableRoute = { ...route, target: { ...route.target } };
    const recovery = createMobileExpansionRecovery({
      id: 7,
      leaseId: "lease-7",
      model: "qwen3-expand:q8",
      inputs: mutableInputs,
      route: mutableRoute,
      requestToken: 11,
      replacePrepared: true,
    });
    mutableInputs.sourcePrompt = "mutated";
    mutableRoute.target.baseUrl = "https://wrong.test";
    mutableRoute.target.apiKey = "wrong";
    mutableRoute.instanceId = "wrong-instance";

    expect(recovery.inputs).toEqual(inputs);
    expect(recovery.route).toEqual(route);
    expect(recovery.host).toMatchObject({
      id: route.hostId,
      name: route.label,
      baseUrl: route.target.baseUrl,
      apiKey: route.target.apiKey,
      instanceId: route.instanceId,
      online: true,
    });
    expect(Object.isFrozen(recovery)).toBe(true);
    expect(Object.isFrozen(recovery.inputs)).toBe(true);
    expect(Object.isFrozen(recovery.route.target)).toBe(true);
  });

  it.each([
    ["endpoint", { baseUrl: "https://new.test:7680" }],
    ["Keychain credential", { apiKey: "rotated" }],
    ["server identity", { instanceId: "replacement-instance" }],
  ])("rejects an in-place %s mutation", (_label, patch) => {
    const recovery = createMobileExpansionRecovery({
      id: 1,
      leaseId: "lease-1",
      model: "qwen3-expand:q8",
      inputs,
      route,
      requestToken: 2,
      replacePrepared: false,
    });
    const currentHost = {
      ...recovery.host,
      ...patch,
    };

    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs,
        currentHost,
        tokenCurrent: true,
      }),
    ).toContain("connection details changed");
  });

  it("freezes the style chip and stales a pull whose style drifts", () => {
    const styled: PreparedExpansionInputs = { ...inputs, stylePreset: "cinematic" };
    const recovery = createMobileExpansionRecovery({
      id: 3,
      leaseId: "lease-3",
      model: "qwen3-expand:q8",
      inputs: styled,
      route,
      requestToken: 5,
      replacePrepared: false,
    });

    expect(recovery.inputs.stylePreset).toBe("cinematic");
    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...styled },
        currentHost: recovery.host,
        tokenCurrent: true,
      }),
    ).toBeNull();
    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...inputs, stylePreset: null },
        currentHost: recovery.host,
        tokenCurrent: true,
      }),
    ).toContain("inputs changed");
  });

  it("treats a legacy style id and its canonical twin as the same frozen style", () => {
    const recovery = createMobileExpansionRecovery({
      id: 4,
      leaseId: "lease-4",
      model: "qwen3-expand:q8",
      inputs: { ...inputs, stylePreset: "photographic" },
      route,
      requestToken: 6,
      replacePrepared: false,
    });

    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...inputs, stylePreset: "photoreal" },
        currentHost: recovery.host,
        tokenCurrent: true,
      }),
    ).toBeNull();
  });

  it("rejects selected-host and form changes without altering the frozen record", () => {
    const recovery = createMobileExpansionRecovery({
      id: 1,
      leaseId: "lease-1",
      model: "qwen3-expand:q8",
      inputs,
      route,
      requestToken: 2,
      replacePrepared: false,
    });

    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...inputs, sourcePrompt: "a changed lighthouse", selectedHostPolicy: "render" },
        currentHost: recovery.host,
        tokenCurrent: true,
      }),
    ).toContain("inputs changed");
    expect(recovery.inputs).toEqual(inputs);
    expect(recovery.route).toEqual(route);
  });

  it("rejects conditioning-task drift during a frozen pull", () => {
    const recovery = createMobileExpansionRecovery({
      id: 8,
      leaseId: "lease-8",
      model: "qwen3-expand:q8",
      inputs,
      route,
      requestToken: 9,
      replacePrepared: false,
    });

    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...inputs, task: "image-to-video" },
        currentHost: recovery.host,
        tokenCurrent: true,
      }),
    ).toContain("inputs changed");
  });

  it("freezes and compares remix source, dimensions, and conditioning identity", () => {
    const remix = {
      sourcePrompt: "a lighthouse",
      rootPrompt: "a lighthouse",
      sourceKind: "original" as const,
      dimensions: ["composition", "lighting"] as const,
      conditioningFingerprint: "media-a",
    };
    const recovery = createMobileExpansionRecovery({
      id: 9,
      leaseId: "lease-9",
      model: "qwen3-expand:q8",
      inputs: { ...inputs, kind: "remix" },
      route,
      requestToken: 10,
      replacePrepared: false,
      remix,
    });

    expect(Object.isFrozen(recovery.remix)).toBe(true);
    expect(Object.isFrozen(recovery.remix?.dimensions)).toBe(true);
    expect(
      mobileExpansionRecoveryStaleReason(recovery, {
        inputs: { ...inputs, kind: "remix" },
        currentHost: recovery.host,
        tokenCurrent: true,
        remix: { ...remix, dimensions: ["composition", "mood"] },
      }),
    ).toContain("Remix inputs changed");
  });
});
