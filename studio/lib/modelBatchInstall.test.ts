import { describe, expect, it } from "vitest";
import {
  isAlreadyQueuedError,
  planBatchInstallTargets,
} from "./modelBatchInstall";

const local = { id: "local", label: "This device" };
const studio = { id: "studio", label: "Studio GPU" };

describe("planBatchInstallTargets", () => {
  it("keeps only machines that can receive every selected model", () => {
    const plans = planBatchInstallTargets([
      {
        modelId: "flux",
        targets: [
          { host: local, action: "install" as const },
          { host: studio, action: "repair" as const },
        ],
      },
      {
        modelId: "ltx",
        targets: [{ host: local, action: "install" as const }],
      },
    ]);

    expect(plans).toEqual([
      {
        host: local,
        items: [
          { modelId: "flux", action: "install" },
          { modelId: "ltx", action: "install" },
        ],
        installCount: 2,
        repairCount: 0,
      },
    ]);
  });

  it("summarises mixed install and repair work for each machine", () => {
    const plans = planBatchInstallTargets([
      {
        modelId: "flux",
        targets: [{ host: studio, action: "repair" as const }],
      },
      {
        modelId: "ltx",
        targets: [{ host: studio, action: "install" as const }],
      },
    ]);

    expect(plans[0]).toMatchObject({
      host: studio,
      installCount: 1,
      repairCount: 1,
    });
  });

  it("returns no target for an incompatible selection", () => {
    expect(
      planBatchInstallTargets([
        {
          modelId: "flux",
          targets: [{ host: local, action: "install" as const }],
        },
        {
          modelId: "ltx",
          targets: [{ host: studio, action: "install" as const }],
        },
      ]),
    ).toEqual([]);
  });
});

describe("isAlreadyQueuedError", () => {
  it("recognises transport errors that preserve an HTTP conflict status", () => {
    expect(isAlreadyQueuedError({ status: 409 })).toBe(true);
    expect(isAlreadyQueuedError({ status: 500 })).toBe(false);
    expect(isAlreadyQueuedError(new Error("already queued"))).toBe(false);
  });
});
