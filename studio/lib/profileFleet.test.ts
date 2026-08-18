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

  it("allows profile drift within one Mold major version", () => {
    expect(
      profileHashConflict(
        { local: [model("old")], remote: [model("new")] },
        "z-image-turbo:q4",
        ["local", "remote"],
        { local: "0.23.1", remote: "0.23.0" },
      ),
    ).toBeNull();
    expect(
      profileHashConflict(
        { local: [model("same")], remote: [model(null)] },
        "z-image-turbo:q4",
        ["local", "remote"],
        { local: "0.23.1", remote: "0.22.4" },
      ),
    ).toBeNull();
  });

  it("requires an explicit machine when profile drift crosses a major version", () => {
    expect(
      profileHashConflict(
        { local: [model("old")], remote: [model("new")] },
        "z-image-turbo:q4",
        ["local", "remote"],
        { local: "0.23.1", remote: "1.0.0" },
      ),
    ).toEqual({
      hostIds: ["local", "remote"],
      hashesByHost: { local: "old", remote: "new" },
    });
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
        { label: "This Mac", profileHash: "local-profile", version: "0.23.1" },
        { label: "plato", profileHash: "remote-profile", version: "1.0.0" },
      ]),
    ).toBe(
      "Auto can't safely choose a machine because This Mac and plato use incompatible major Mold versions for this model. Update and reconnect them, or choose one machine for this print. Nothing was queued.",
    );
  });
});
