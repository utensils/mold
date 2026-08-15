import { describe, expect, it, vi } from "vitest";
import {
  fetchH3BoundaryMedia,
  h3BoundariesNeedingMedia,
} from "./h3BoundaryRestore";
import type { MinimaxH3AuthoringState } from "./minimaxH3Authoring";

const descriptor = (filename: string) => ({
  filename,
  mimeType: "image/*",
  width: 0,
  height: 0,
  data: "",
  sha256: "a".repeat(64),
});

const populated = (filename: string) => ({
  ...descriptor(filename),
  data: "QUJD",
});

describe("h3BoundariesNeedingMedia", () => {
  it("lists bytes-less named descriptors only", () => {
    const authoring: MinimaxH3AuthoringState = {
      firstFrame: descriptor("opening.png"),
      lastFrame: populated("closing.png"),
      references: [],
    };
    expect(h3BoundariesNeedingMedia(authoring)).toEqual([
      {
        endpoint: "firstFrame",
        filename: "opening.png",
        sha256: "a".repeat(64),
      },
    ]);
    expect(h3BoundariesNeedingMedia(null)).toEqual([]);
  });
});

describe("fetchH3BoundaryMedia", () => {
  it("fetches bytes for each descriptor and reports misses", async () => {
    const authoring: MinimaxH3AuthoringState = {
      firstFrame: descriptor("opening.png"),
      lastFrame: descriptor("closing.png"),
      references: [],
    };
    const fetchByFilename = vi.fn(async (name: string) =>
      name === "opening.png" ? "T1BFTg==" : null,
    );
    const outcome = await fetchH3BoundaryMedia(authoring, fetchByFilename);
    expect(outcome.restored).toEqual([
      { endpoint: "firstFrame", filename: "opening.png", base64: "T1BFTg==" },
    ]);
    expect(outcome.failed).toEqual(["closing.png"]);
  });

  it("does nothing when every slot already has bytes", async () => {
    const authoring: MinimaxH3AuthoringState = {
      firstFrame: populated("opening.png"),
      lastFrame: null,
      references: [],
    };
    const fetchByFilename = vi.fn();
    const outcome = await fetchH3BoundaryMedia(authoring, fetchByFilename);
    expect(outcome.restored).toEqual([]);
    expect(outcome.failed).toEqual([]);
    expect(fetchByFilename).not.toHaveBeenCalled();
  });

  it("treats a fetch rejection as a miss instead of failing the restore", async () => {
    const authoring: MinimaxH3AuthoringState = {
      firstFrame: descriptor("opening.png"),
      lastFrame: null,
      references: [],
    };
    const outcome = await fetchH3BoundaryMedia(
      authoring,
      vi.fn().mockRejectedValue(new Error("host offline")),
    );
    expect(outcome.failed).toEqual(["opening.png"]);
  });
});
