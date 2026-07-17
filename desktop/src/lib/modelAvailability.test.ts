import { describe, expect, it } from "vitest";
import { mediaTypeFromQuery } from "./modelAvailability";

describe("mediaTypeFromQuery", () => {
  it("reads the explicit media-type query", () => {
    expect(mediaTypeFromQuery({ type: "image" })).toBe("image");
    expect(mediaTypeFromQuery({ type: "video" })).toBe("video");
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
