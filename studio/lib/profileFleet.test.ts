import { describe, expect, it } from "vitest";
import { profileHashConflict } from "./profileFleet";

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
