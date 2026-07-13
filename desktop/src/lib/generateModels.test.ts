import { describe, expect, it } from "vitest";
import type { ModelEntry } from "./api/types";
import {
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "./generateModels";

function model(name: string, family = "flux"): ModelEntry {
  return {
    name,
    family,
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
  } as ModelEntry;
}

describe("Generate model availability", () => {
  it("merges remote-only models and prefers the primary copy of duplicates", () => {
    const primary = model("flux-dev:q8");
    const merged = mergeInstalledModels(
      [primary],
      [
        { ...primary, description: "remote", hostIds: ["hal9000-7680"] },
        { ...model("z-image:q8", "zimage"), hostIds: ["hal9000-7680"] },
      ],
    );

    expect(merged.map((entry) => entry.name)).toEqual(["flux-dev:q8", "z-image:q8"]);
    expect(merged[0]).toBe(primary);
  });

  it("shows the starter only after every host has initialized and refreshed empty", () => {
    const base = {
      connectionReady: true,
      primaryLoading: false,
      hostsInitialized: true,
      hostModelsLoading: false,
      allReadyHostsFetched: true,
      installed: [] as ModelEntry[],
    };

    expect(shouldShowStarterCards(base)).toBe(true);
    expect(shouldShowStarterCards({ ...base, hostsInitialized: false })).toBe(false);
    expect(shouldShowStarterCards({ ...base, hostModelsLoading: true })).toBe(false);
    expect(shouldShowStarterCards({ ...base, allReadyHostsFetched: false })).toBe(false);
    expect(shouldShowStarterCards({ ...base, installed: [model("flux-dev:q8")] })).toBe(false);
  });

  it("selects and resolves models that exist only on a remote host", () => {
    const remote = model("z-image:q8", "zimage");
    const installed = mergeInstalledModels([], [{ ...remote, hostIds: ["hal9000-7680"] }]);

    expect(preferredInstalledModel(installed)).toMatchObject({ name: "z-image:q8" });
    expect(findInstalledModel(installed, "z-image:q8")?.family).toBe("zimage");
  });
});
