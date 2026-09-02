import { describe, expect, it } from "vitest";

import type { GalleryImage, OutputMetadata } from "../api/types";
import { isUpscaledImage } from "./upscaled";

function print(filename: string, metadata: Partial<OutputMetadata>): GalleryImage {
  return {
    filename,
    timestamp: 1_788_304_027,
    format: filename.endsWith(".mp4") ? "mp4" : "png",
    metadata: {
      prompt: "a thermalnuclear explosion",
      model: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
      seed: 570911661,
      steps: 5,
      guidance: 0,
      width: 960,
      height: 960,
      ...metadata,
    } as OutputMetadata,
  } as GalleryImage;
}

describe("isUpscaledImage", () => {
  it("badges a Framewise-upscaled video from its recorded source dimensions", () => {
    // The hal9000 print: 960×960 ×4, `upscale_model` beside the source size.
    const item = print("clip-framewise-upscaled-268a8049.mp4", {
      width: 3072,
      height: 3072,
      generation_width: 960,
      generation_height: 960,
      upscale_model: "real-esrgan-x4plus:fp16",
      source_video_path: "clip.mp4",
    });
    expect(isUpscaledImage(item)).toBe(true);
  });

  it("badges a Framewise video by its server-issued filename when the fit lands on the source size", () => {
    // A 4096×1716 source ×2 is fitted back to 4096×1716 by the edge cap.
    const item = print("wide-framewise-upscaled-0a1b2c3d.mp4", {
      width: 4096,
      height: 1716,
      generation_width: 4096,
      generation_height: 1716,
      upscale_model: "real-esrgan-x2plus:fp16",
      source_video_path: "wide.mp4",
    });
    expect(isUpscaledImage(item)).toBe(true);
  });

  it("leaves a generated video unbadged when an upscaler was merely requested", () => {
    const item = print("clip.mp4", {
      width: 960,
      height: 960,
      generation_width: 960,
      generation_height: 960,
      upscale_model: "real-esrgan-x4plus:fp16",
    });
    expect(isUpscaledImage(item)).toBe(false);
  });

  it("leaves the separately saved original unbadged", () => {
    expect(isUpscaledImage(print("clip.mp4", {}))).toBe(false);
    expect(isUpscaledImage(print("still.png", {}))).toBe(false);
  });

  it("still badges stills by filename or by recorded dimensions", () => {
    expect(isUpscaledImage(print("still-upscaled.png", {}))).toBe(true);
    expect(
      isUpscaledImage(
        print("renamed.png", {
          width: 2048,
          height: 2048,
          generation_width: 512,
          generation_height: 512,
          upscale_model: "real-esrgan-x4plus:fp16",
        }),
      ),
    ).toBe(true);
  });
});
