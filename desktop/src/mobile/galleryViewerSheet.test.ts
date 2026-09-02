import { describe, expect, it } from "vitest";
import type { GalleryImage } from "../lib/api/types";
import {
  meshExportChoices,
  resolveSheetGesture,
  viewerKindLabel,
  viewerPeekSummary,
} from "./galleryViewerSheet";

const print: GalleryImage = {
  filename: "print one.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:q8",
    seed: 42,
    steps: 4,
    guidance: 3.5,
    width: 1024,
    height: 1024,
  },
};

describe("viewerKindLabel", () => {
  it("names every media kind the phone viewer can stage", () => {
    expect(viewerKindLabel("image")).toBe("Image");
    expect(viewerKindLabel("video")).toBe("Video");
    expect(viewerKindLabel("audio")).toBe("Audio");
    expect(viewerKindLabel("mesh")).toBe("3-D");
  });
});

describe("viewerPeekSummary", () => {
  const mesh: GalleryImage = { ...print, filename: "armchair 01.glb", format: "glb" };
  const clip: GalleryImage = { ...print, filename: "clip.mp4", format: "mp4" };

  it("measures a mesh in the triangles and vertices its viewer loaded", () => {
    expect(
      viewerPeekSummary("mesh", mesh, { mesh: { triangleCount: 49_152, vertexCount: 24_576 } }),
    ).toBe("49.2k tris · 24.6k verts");
  });

  /** A gallery row records no counts, so an unloaded mesh names its container. */
  it("falls back to the stored container before the mesh has loaded", () => {
    expect(viewerPeekSummary("mesh", mesh)).toBe("GLB");
  });

  it("measures a clip by the running time its player reported", () => {
    expect(viewerPeekSummary("video", clip, { durationMs: 4_040 })).toBe("0:04");
  });

  it("derives a clip's duration from its frame grid before playback loads", () => {
    expect(
      viewerPeekSummary("video", {
        ...clip,
        metadata: { ...print.metadata, video_frames: 97, video_fps: 24 },
      }),
    ).toBe("0:04");
  });

  it("measures audio by the transport's own duration", () => {
    expect(
      viewerPeekSummary(
        "audio",
        { ...print, filename: "score.wav", format: "wav" },
        {
          durationMs: 83_000,
        },
      ),
    ).toBe("1:23");
  });

  it("measures a still by its pixel dimensions", () => {
    expect(viewerPeekSummary("image", print)).toBe("1024×1024");
  });
});

describe("resolveSheetGesture", () => {
  it("expands on a decisive upward drag and collapses on a downward one", () => {
    expect(resolveSheetGesture({ deltaX: 4, deltaY: -80, expanded: false, scrolled: false })).toBe(
      "expand",
    );
    expect(resolveSheetGesture({ deltaX: 4, deltaY: 80, expanded: true, scrolled: false })).toBe(
      "collapse",
    );
  });

  it("ignores a drag that is short, horizontal, or already at the end state", () => {
    expect(resolveSheetGesture({ deltaX: 0, deltaY: -12, expanded: false, scrolled: false })).toBe(
      "none",
    );
    expect(
      resolveSheetGesture({ deltaX: -160, deltaY: -60, expanded: false, scrolled: false }),
    ).toBe("none");
    expect(resolveSheetGesture({ deltaX: 0, deltaY: -80, expanded: true, scrolled: false })).toBe(
      "none",
    );
    expect(resolveSheetGesture({ deltaX: 0, deltaY: 80, expanded: false, scrolled: false })).toBe(
      "none",
    );
  });

  /** A scrolled body owns its own downward drag: the sheet must not steal it. */
  it("leaves a downward drag to a scrolled sheet body", () => {
    expect(resolveSheetGesture({ deltaX: 0, deltaY: 90, expanded: true, scrolled: true })).toBe(
      "none",
    );
  });
});

describe("meshExportChoices", () => {
  it("lists the advertised geometry containers and one turntable entry", () => {
    expect(meshExportChoices(["glb", "obj", "stl", "ply"], ["gif", "apng", "webp"])).toEqual([
      { value: "glb", label: "GLB" },
      { value: "obj", label: "OBJ" },
      { value: "stl", label: "STL" },
      { value: "ply", label: "PLY" },
      { value: "turntable", label: "Turntable" },
    ]);
  });

  it("offers no turntable when the host advertises no animated container", () => {
    expect(meshExportChoices(["obj"], [])).toEqual([{ value: "obj", label: "OBJ" }]);
  });

  it("offers the turntable alone when geometry is all the host withholds", () => {
    expect(meshExportChoices([], ["gif"])).toEqual([{ value: "turntable", label: "Turntable" }]);
  });
});
