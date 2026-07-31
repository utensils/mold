import { describe, expect, it } from "vitest";
import { planModelInstall } from "./modelInstallTargets";

interface TestHost {
  id: string;
  label: string;
}

const local: TestHost = { id: "local", label: "This device" };
const bender: TestHost = { id: "bender", label: "bender" };
const hal: TestHost = { id: "hal9000", label: "hal9000" };

describe("planModelInstall", () => {
  it("offers an install on every machine that does not have the model", () => {
    const plan = planModelInstall([local, bender, hal], ["bender"]);

    expect(plan.canInstall).toBe(true);
    expect(plan.label).toBe("Pull");
    expect(plan.targets).toEqual([
      { host: local, action: "install" },
      { host: hal, action: "install" },
      { host: bender, action: "repair" },
    ]);
  });

  it("only repairs once every eligible machine has the model", () => {
    const plan = planModelInstall([local, bender], ["local", "bender"]);

    expect(plan.canInstall).toBe(false);
    expect(plan.label).toBe("Repair");
    expect(plan.targets).toEqual([
      { host: local, action: "repair" },
      { host: bender, action: "repair" },
    ]);
  });

  it("keeps a model nobody has a plain pull", () => {
    const plan = planModelInstall([local, bender], []);

    expect(plan.canInstall).toBe(true);
    expect(plan.label).toBe("Pull");
    expect(plan.targets.map((t) => t.action)).toEqual(["install", "install"]);
  });

  it("labels the action Pull even when no machine is reachable", () => {
    // Nothing to target, but the row must not claim the model is repairable
    // when in fact it is installed nowhere.
    expect(planModelInstall([], []).label).toBe("Pull");
    expect(planModelInstall([], ["bender"]).label).toBe("Repair");
  });

  it("never offers a machine whose inventory has not been read", () => {
    const plan = planModelInstall([local, bender, hal], ["bender"], {
      inventoryKnown: (host) => host.id !== "hal9000",
    });

    expect(plan.targets).toEqual([
      { host: local, action: "install" },
      { host: bender, action: "repair" },
    ]);
  });

  it("still repairs on an owning machine whose inventory looks unread", () => {
    // Ownership is itself positive knowledge — it cannot be contradicted by a
    // missing inventory snapshot.
    const plan = planModelInstall([bender], ["bender"], {
      inventoryKnown: () => false,
    });

    expect(plan.targets).toEqual([{ host: bender, action: "repair" }]);
    expect(plan.canInstall).toBe(false);
  });

  it("tolerates a missing owner list", () => {
    const plan = planModelInstall([local], null);

    expect(plan.canInstall).toBe(true);
    expect(plan.targets).toEqual([{ host: local, action: "install" }]);
  });

  it("ignores owner ids that are not eligible machines", () => {
    // A model installed only on an offline or forgotten host still installs
    // onto the machines that are actually reachable.
    const plan = planModelInstall([local], ["retired-box"]);

    expect(plan.canInstall).toBe(true);
    expect(plan.label).toBe("Pull");
    expect(plan.targets).toEqual([{ host: local, action: "install" }]);
  });
});
