import { describe, expect, it } from "vitest";
import { mediaTypeFromQuery, mediaTypeMatches, MEDIA_TYPE_KIND } from "./modelAvailability";

describe("mediaTypeFromQuery", () => {
  it("reads the explicit media-type query", () => {
    expect(mediaTypeFromQuery({ type: "image" })).toBe("image");
    expect(mediaTypeFromQuery({ type: "video" })).toBe("video");
    expect(mediaTypeFromQuery({ type: "mesh" })).toBe("mesh");
  });

  it("defaults an absent or unknown type to all", () => {
    expect(mediaTypeFromQuery({})).toBe("all");
    expect(mediaTypeFromQuery({ type: "wat" })).toBe("all");
    expect(mediaTypeFromQuery({ type: ["image", "video"] })).toBe("all");
  });

  it("maps the legacy catalog deep link to all without crashing", () => {
    expect(mediaTypeFromQuery({ tab: "catalog" })).toBe("all");
  });

  it("maps legacy availability queries to all without crashing", () => {
    expect(mediaTypeFromQuery({ availability: "installed" })).toBe("all");
    expect(mediaTypeFromQuery({ availability: "available" })).toBe("all");
  });
});

/*
 * The kind filter is the Create toolbar's own partition (`outputKindForModel`)
 * and nothing else, so a style filters under exactly the kind it is offered
 * under in Create.
 */
describe("mediaTypeMatches", () => {
  it("maps every filter value onto one Create section", () => {
    expect(MEDIA_TYPE_KIND).toEqual({ image: "still", video: "clip", mesh: "mesh" });
  });

  it("passes everything under All", () => {
    for (const family of ["flux", "ltx-video", "hunyuan3d"]) {
      expect(mediaTypeMatches("all", family)).toBe(true);
    }
  });

  it("sorts a style by the same partition Create offers it under", () => {
    expect(mediaTypeMatches("image", "flux")).toBe(true);
    expect(mediaTypeMatches("video", "flux")).toBe(false);
    expect(mediaTypeMatches("mesh", "flux")).toBe(false);

    expect(mediaTypeMatches("video", "ltx-video")).toBe(true);
    expect(mediaTypeMatches("image", "ltx-video")).toBe(false);

    expect(mediaTypeMatches("mesh", "hunyuan3d")).toBe(true);
    expect(mediaTypeMatches("image", "hunyuan3d")).toBe(false);
    expect(mediaTypeMatches("video", "hunyuan3d")).toBe(false);
  });
});
