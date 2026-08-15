import { describe, expect, it } from "vitest";
import { profileConflictMessage, profileHashConflict } from "./profileFleet";

const model = (hash: string | null, downloaded = true) => ({
  name: "z-image-turbo:q4",
  downloaded,
  generation_profile: hash ? { profile_hash: hash } : null,
});

describe("profileHashConflict", () => {
  it("accepts automatic routing when every eligible owner agrees", () => {
    expect(
      profileHashConflict(
        { local: [model("same")], remote: [model("same")] },
        "z-image-turbo:q4",
        ["local", "remote"],
      ),
    ).toBeNull();
  });

  it("requires an explicit machine for different or missing hashes", () => {
    expect(
      profileHashConflict(
        { local: [model("old")], remote: [model("new")] },
        "z-image-turbo:q4",
        ["local", "remote"],
      ),
    ).toEqual({
      hostIds: ["local", "remote"],
      hashesByHost: { local: "old", remote: "new" },
    });
    expect(
      profileHashConflict(
        { local: [model("same")], remote: [model(null)] },
        "z-image-turbo:q4",
        ["local", "remote"],
      ),
    ).not.toBeNull();
  });

  it("keeps an all-legacy fleet routable during the compatibility release", () => {
    expect(
      profileHashConflict(
        { local: [model(null)], remote: [model(null)] },
        "z-image-turbo:q4",
        ["local", "remote"],
      ),
    ).toBeNull();
  });

  it("ignores ineligible hosts and copies that are not installed", () => {
    expect(
      profileHashConflict(
        {
          local: [model("old")],
          remote: [model("new")],
          catalog: [model("different", false)],
        },
        "z-image-turbo:q4",
        ["local", "catalog"],
      ),
    ).toBeNull();
  });
});

describe("profileConflictMessage", () => {
  it("names conflicting machines and gives an immediate recovery action", () => {
    expect(
      profileConflictMessage([
        { label: "This Mac", profileHash: "local-profile" },
        { label: "plato", profileHash: "remote-profile" },
      ]),
    ).toBe(
      "Auto can't safely choose a machine because This Mac and plato use different generation settings for this model. They may be running different Mold versions or builds, so the same controls could produce different results. Update and reconnect them, or choose one machine for this print. Nothing was queued.",
    );
  });

  it("explains when an older machine may lack a generation profile", () => {
    expect(
      profileConflictMessage([
        { label: "This Mac", profileHash: "current-profile" },
        { label: "studio", profileHash: null },
      ]),
    ).toContain("At least one may be running an older Mold version");
  });
});
