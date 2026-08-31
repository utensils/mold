import { describe, expect, it } from "vitest";
import {
  defaultUpscaler,
  framewiseProgress,
  framewiseStatus,
  libraryUpscaleLabel,
  shouldPollFramewiseJob,
} from "./upscale";

describe("Library upscale presentation", () => {
  it("names image and video actions without conflating temporal processing", () => {
    expect(libraryUpscaleLabel("image")).toBe("Upscale…");
    expect(libraryUpscaleLabel("video")).toBe("Framewise upscale…");
  });

  it("prefers an installed general-purpose model", () => {
    expect(
      defaultUpscaler([
        { name: "real-esrgan-x4plus-anime:fp16", downloaded: true },
        { name: "real-esrgan-x4plus:fp16", downloaded: true },
      ]),
    ).toBe("real-esrgan-x4plus:fp16");
  });

  it("describes durable progress and recovery", () => {
    const job = {
      id: "vup-1",
      contract_version: 1,
      state: "running" as const,
      model: "real-esrgan-x4plus:fp16",
      completed_frames: 4,
      total_frames: 10,
      disclosure: "Framewise",
    };
    expect(framewiseProgress(job)).toBe(0.4);
    expect(framewiseStatus(job)).toBe("Upscaling frame 5 of 10");
    expect(shouldPollFramewiseJob(job)).toBe(true);
    expect(shouldPollFramewiseJob({ ...job, state: "paused" })).toBe(false);
  });
});
