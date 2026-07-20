import { describe, expect, it } from "vitest";
import type { GenerateRequest, SseChainCompleteEvent } from "./api/types";
import { applyChainProgress, chainCompleteToComplete, newJob } from "./generationJob";

const request: GenerateRequest = {
  prompt: "a lighthouse through a storm",
  model: "ltx-2.3-22b-distilled:fp8",
  width: 1536,
  height: 640,
  steps: 24,
  guidance: 3.5,
  seed: 42,
  frames: 241,
  fps: 24,
  output_format: "mp4",
};

describe("chain Job adapters", () => {
  it("tracks the additive job id and maps per-clip progress to overall progress", () => {
    const job = newJob(request);

    applyChainProgress(job, {
      type: "chain_start",
      stage_count: 3,
      estimated_total_frames: 241,
      job_id: "chain-1",
    });
    expect(job).toMatchObject({
      id: "chain-1",
      status: "loading",
      chainStageCount: 3,
      stage: "Preparing 3 clips",
    });

    applyChainProgress(job, {
      type: "denoise_step",
      stage_idx: 1,
      step: 7,
      total: 24,
      job_id: "chain-1",
    });
    expect(job).toMatchObject({
      id: "chain-1",
      status: "denoising",
      chainStageIndex: 1,
      step: 31,
      total: 72,
      stage: "Clip 2 of 3",
    });

    applyChainProgress(job, {
      type: "stitching",
      total_frames: 241,
      job_id: "chain-1",
    });
    expect(job).toMatchObject({ status: "finishing", step: 72, stage: "Stitching 241 frames" });
  });

  it("adapts full and metadata-only chain completions to CompleteEvent", () => {
    const metadata = {
      prompt: request.prompt,
      model: request.model,
      seed: 99,
      steps: 24,
      guidance: 3.5,
      width: 1536,
      height: 640,
    };
    const event = {
      video: "video-b64",
      format: "mp4",
      width: 1536,
      height: 640,
      frames: 241,
      fps: 24,
      thumbnail: "thumb-b64",
      gif_preview: "gif-b64",
      has_audio: true,
      duration_ms: 10_042,
      audio_sample_rate: 48_000,
      audio_channels: 2,
      stage_count: 3,
      gpu: 1,
      generation_time_ms: 12_345,
      filename: "chain-99.mp4",
      metadata,
      script: {} as SseChainCompleteEvent["script"],
    } satisfies SseChainCompleteEvent;

    expect(chainCompleteToComplete(event, request)).toEqual({
      image: "video-b64",
      format: "mp4",
      width: 1536,
      height: 640,
      original_image: null,
      original_width: null,
      original_height: null,
      seed_used: 99,
      generation_time_ms: 12_345,
      model: request.model,
      video_frames: 241,
      video_fps: 24,
      video_thumbnail: "thumb-b64",
      video_gif_preview: "gif-b64",
      video_has_audio: true,
      video_duration_ms: 10_042,
      video_audio_sample_rate: 48_000,
      video_audio_channels: 2,
      gpu: 1,
      filename: "chain-99.mp4",
      original_filename: null,
      metadata,
    });

    expect(
      chainCompleteToComplete(
        {
          ...event,
          video: "",
          thumbnail: null,
          gif_preview: null,
        },
        request,
      ),
    ).toMatchObject({ image: "", filename: "chain-99.mp4", seed_used: 99 });
  });
});
