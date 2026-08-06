import { describe, expect, it } from "vitest";
import { videoExportFilename, videoExportPath } from "./videoExport";

describe("video export wire helpers", () => {
  it("encodes the owning gallery filename into the export route", () => {
    expect(videoExportPath("rain dance #1.mp4")).toBe(
      "/api/gallery/export/rain%20dance%20%231.mp4",
    );
  });

  it("replaces the source extension for each exported animation", () => {
    expect(videoExportFilename("rain.dance.mp4", "gif")).toBe("rain.dance.gif");
    expect(videoExportFilename("rain.dance.mp4", "apng")).toBe(
      "rain.dance.png",
    );
  });
});
