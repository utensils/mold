import { describe, expect, it } from "vitest";
import { previewPrintFilename, previewSequenceFilename } from "./printFilename";

describe("previewSequenceFilename", () => {
  it("uses the chain grammar and elides the job digest", () => {
    expect(previewSequenceFilename("Smurfs")).toBe("mold-chain-…-take-0~smurfs.mp4");
    expect(previewSequenceFilename(null)).toBe("mold-chain-…-take-0.mp4");
    expect(previewSequenceFilename("日本語")).toBe("mold-chain-…-take-0.mp4");
  });
});

describe("previewPrintFilename", () => {
  it("mirrors the server's untitled grammar", () => {
    expect(
      previewPrintFilename({ model: "z-image-turbo:bf16", timestamp: 1787320481000, ext: "png" }),
    ).toBe("mold-z-image-turbo-bf16-1787320481000.png");
  });

  it("appends the title slug", () => {
    expect(
      previewPrintFilename({
        model: "z-image-turbo:bf16",
        timestamp: 1787320481000,
        ext: "png",
        title: "Smurfs",
      }),
    ).toBe("mold-z-image-turbo-bf16-1787320481000~smurfs.png");
  });

  it("omits the slug when the title has nothing sluggable in it", () => {
    expect(
      previewPrintFilename({
        model: "flux-dev",
        timestamp: 10,
        ext: "png",
        title: "日本語",
      }),
    ).toBe("mold-flux-dev-10.png");
  });

  it("adds the batch index only for a batch larger than one", () => {
    expect(
      previewPrintFilename({
        model: "flux-dev",
        timestamp: 10,
        ext: "png",
        batchSize: 1,
        index: 0,
      }),
    ).toBe("mold-flux-dev-10.png");
    expect(
      previewPrintFilename({
        model: "flux-dev",
        timestamp: 10,
        ext: "png",
        batchSize: 4,
        index: 2,
      }),
    ).toBe("mold-flux-dev-10-2.png");
  });

  it("normalizes the extension and tolerates an empty model", () => {
    expect(previewPrintFilename({ model: "", timestamp: 10, ext: ".MP4" })).toBe(
      "mold-model-10.mp4",
    );
  });
});
