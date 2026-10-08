import { describe, expect, it } from "vitest";
import { projectRetainedDraftMedia } from "./retainedDraftMedia";
const member = (role: string, display_name = role) => ({
  role,
  display_name,
  member_id: role,
  size_bytes: 1,
});
describe("visible retained draft media", () => {
  it.each([false, true])(
    "restores Wan endpoints on web=%s without moving the canvas",
    (web) => {
      const patch = projectRetainedDraftMedia(
        {
          keyframes: [
            { frame: 0, image: "first", name: "start" },
            { frame: 96, image: "last", name: "end" },
          ],
        },
        [member("keyframes")],
        { web, boundary: true, sourceMode: "single", h3: false },
      );
      expect(patch.endFrame).toMatchObject({ base64: "last", filename: "end" });
      expect(web ? patch.imageAttachments : patch.sourceImage).toEqual(
        web ? [expect.objectContaining({ base64: "first" })] : "first",
      );
      expect(patch).not.toHaveProperty("width");
      expect(patch.keyframes).toEqual([]);
    },
  );
  it("preserves LTX indices and ordered edit references", () => {
    const frames = [
      { frame: 17, image: "mid", name: "middle" },
      { frame: 0, image: "first", name: "opening" },
    ];
    const patch = projectRetainedDraftMedia(
      { keyframes: frames, edit_images: ["target", "reference"] },
      [
        member("keyframes"),
        member("edit_images", "target"),
        member("edit_images", "reference"),
      ],
      { web: false, boundary: false, sourceMode: "replaces", h3: false },
    );
    expect(patch.keyframes).toEqual(
      frames.map((f) => ({
        frame: f.frame,
        image: { base64: f.image, filename: f.name },
      })),
    );
    expect(patch.imageAttachments).toEqual(["target", "reference"]);
  });
  it("restores H3 opening and closing wells instead of a hidden list", () => {
    const patch = projectRetainedDraftMedia(
      {
        source_image: "first",
        keyframes: [{ frame: 140, image: "last", name: "end" }],
      },
      [member("source_image", "start"), member("keyframes")],
      { web: false, boundary: false, sourceMode: "none", h3: true },
    );
    expect(patch.h3Authoring).toMatchObject({
      firstFrame: { data: "first" },
      lastFrame: { data: "last" },
    });
    expect(patch.keyframes).toEqual([]);
  });
  it("refuses unsupported plural identity rather than silently dropping photos", () => {
    expect(() =>
      projectRetainedDraftMedia({ id_images: ["a", "b"] }, [], {
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
    ).toThrow();
  });
});
