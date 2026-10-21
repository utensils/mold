import type { VideoJobRequest, VideoJobSource } from "./api/videoJobTypes";
export interface VideoDurationScene {
  id: string;
  prompt: string;
  durationMs: string | null;
}
export interface VideoDurationExtension {
  source: Pick<VideoJobSource, "filename" | "expected_archive_identity_sha256">;
  owner: {
    hostId: string;
    instanceId: string;
    baseUrl: string;
    connectionMode?: string;
  };
  title: string;
  machine: string;
  references: { name: string; role: string }[];
}
export interface VideoDurationAuthoring {
  reuseNotice?: string;
  extension?: VideoDurationExtension;
  durationMs: string | null;
  scenesEnabled: boolean;
  scenes: VideoDurationScene[];
}
export function initialVideoDuration(
  frames: number | null | undefined,
  fps: number | null | undefined,
): VideoDurationAuthoring {
  const valid =
    Number.isSafeInteger(frames) &&
    frames! > 0 &&
    Number.isSafeInteger(fps) &&
    fps! > 0;
  const value = valid ? (BigInt(frames!) * 1000n) / BigInt(fps!) : null;
  return {
    durationMs: value !== null && value > 0n ? String(value) : null,
    scenesEnabled: false,
    scenes: [],
  };
}
function duration(value: string | null): string {
  if (
    !value ||
    !/^[1-9]\d*$/.test(value) ||
    value.length > 20 ||
    BigInt(value) > 18_446_744_073_709_551_615n
  )
    throw new Error(
      "Enter a positive duration in seconds, with up to three decimal places.",
    );
  return value;
}
export function buildDurationVideoRequest<Generation extends object>(
  draft: VideoDurationAuthoring,
  generation: Generation,
  source?: VideoJobSource,
): VideoJobRequest<Generation> {
  const durationMs = duration(draft.durationMs);
  if (draft.scenes.length > 4096)
    throw new Error("Use at most 4,096 scenes in one video.");
  const scenes = draft.scenesEnabled
    ? draft.scenes.map((scene) => {
        if (
          !scene.prompt.trim() ||
          new TextEncoder().encode(scene.prompt).byteLength > 77_000
        )
          throw new Error(
            "Every scene needs a prompt of at most 77,000 bytes.",
          );
        return {
          prompt: scene.prompt,
          duration_ms: duration(scene.durationMs),
        };
      })
    : [];
  if (
    scenes.length &&
    scenes.reduce((sum, scene) => sum + BigInt(scene.duration_ms), 0n) !==
      BigInt(durationMs)
  )
    throw new Error(
      "Scene durations must add up to the video's requested duration.",
    );
  return {
    generation,
    duration_ms: durationMs,
    ...((source ?? draft.extension?.source)
      ? { source: source ?? draft.extension!.source }
      : {}),
    ...(scenes.length ? { scenes } : {}),
  };
}

/** Never admit fewer movies than the copy count the author selected. */
export function durationVideoCopies(
  requested: number,
  prepared: number,
): number {
  if (
    !Number.isSafeInteger(requested) ||
    requested < 1 ||
    requested !== prepared
  )
    throw new Error(
      `This machine can prepare ${prepared} video copies at once, but ${requested} were requested. Choose a supported copy count before generating.`,
    );
  return requested;
}
