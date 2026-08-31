import { describe, expect, it } from "vitest";
import {
  recoverableFramewiseUpscale,
  type VideoUpscaleJob,
} from "./videoUpscale";

function job(
  id: string,
  filename: string,
  state: VideoUpscaleJob["state"],
): VideoUpscaleJob {
  return {
    contract_version: 1,
    id,
    state,
    source: { kind: "library", filename },
    model: "real-esrgan-x4plus:fp16",
    completed_frames: 0,
    total_frames: 4,
    disclosure: "Framewise",
  };
}

describe("recoverableFramewiseUpscale", () => {
  it("restores the newest active or restart-paused job for a Library video", () => {
    expect(
      recoverableFramewiseUpscale(
        [job("paused", "clip.mp4", "paused"), job("other", "other.mp4", "running")],
        "clip.mp4",
      )?.id,
    ).toBe("paused");
  });

  it("does not restore completed, failed, or cancelled history", () => {
    expect(
      recoverableFramewiseUpscale(
        [
          job("done", "clip.mp4", "completed"),
          job("failed", "clip.mp4", "failed"),
          job("cancelled", "clip.mp4", "cancelled"),
        ],
        "clip.mp4",
      ),
    ).toBeNull();
  });
});
