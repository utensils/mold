import { describe, expect, it } from "vitest";
import { galleryMediaPath } from "./media";

describe("galleryMediaPath", () => {
  it("keeps engine gallery media on the authenticated API", () => {
    expect(galleryMediaPath("print one.png", "engine")).toBe("/api/gallery/image/print%20one.png");
  });

  it("routes local gallery media through the restricted native protocol", () => {
    expect(galleryMediaPath("print one.png", "local", true)).toBe(
      "mold-local://localhost/print%20one.png",
    );
  });
});
