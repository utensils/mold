import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  downloadVideoExport,
  shareVideoExport,
  videoExportFilename,
  videoExportPath,
} from "./videoExport";

describe("video export wire helpers", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:video-export"),
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: vi.fn(),
      configurable: true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

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

  it("downloads on desktop and web even when Web Share is available", () => {
    const share = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "share", {
      value: share,
      configurable: true,
    });
    Object.defineProperty(navigator, "canShare", {
      value: vi.fn(() => true),
      configurable: true,
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    downloadVideoExport(new Blob(["gif"], { type: "image/gif" }), "export.gif");

    expect(click).toHaveBeenCalledOnce();
    expect(share).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:video-export");
  });

  it("shares an image file only when the iOS caller opts in", async () => {
    const share = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "share", {
      value: share,
      configurable: true,
    });
    Object.defineProperty(navigator, "canShare", {
      value: vi.fn(() => true),
      configurable: true,
    });

    await expect(
      shareVideoExport(new Blob(["gif"], { type: "image/gif" }), "export.gif"),
    ).resolves.toBe("shared");
    expect(share).toHaveBeenCalledWith({
      files: [
        expect.objectContaining({ name: "export.gif", type: "image/gif" }),
      ],
      title: "export.gif",
    });
  });

  it("treats dismissing the iOS share sheet as cancellation", async () => {
    Object.defineProperty(navigator, "share", {
      value: vi.fn(async () => {
        throw new DOMException("Share cancelled", "AbortError");
      }),
      configurable: true,
    });
    Object.defineProperty(navigator, "canShare", {
      value: vi.fn(() => true),
      configurable: true,
    });

    await expect(
      shareVideoExport(new Blob(["gif"], { type: "image/gif" }), "export.gif"),
    ).resolves.toBe("cancelled");
  });
});

import { validGifPause, turntableFrameLimit } from "./videoExport";
describe("export timing capability validation", () => {
  it("keeps explicit zero and refuses malformed advertisements", () => {
    expect(validGifPause({ min: 0, max: 5000, step: 10, default: 0 })).toBe(
      true,
    );
    expect(validGifPause(undefined)).toBe(false);
    expect(validGifPause({ min: 0, max: 5000, step: 0, default: 0 })).toBe(
      false,
    );
    expect(validGifPause({ min: 10, max: 0, step: 10, default: 0 })).toBe(
      false,
    );
  });
  it("shares the mesh frame buffer budget including transparency", () => {
    expect(turntableFrameLimit(2048, false)).toBe(21);
    expect(turntableFrameLimit(2048, true)).toBe(16);
    expect(turntableFrameLimit(512, true)).toBe(180);
  });
});
