import type { ExpandTask, ExpansionTaskRequest } from "./expandTask";

export type RemixSourceKind = "original" | "current" | "direct";
export type RemixDimension =
  | "composition"
  | "camera"
  | "lighting"
  | "setting"
  | "mood"
  | "movement"
  | "style";

export interface PromptSource {
  prompt: string;
  rootPrompt?: string;
  kind: RemixSourceKind;
}

export interface RemixVariant {
  prompt: string;
  dimensions: RemixDimension[];
}

export const DEFAULT_REMIX_VARIATIONS = 3;

export function promptSource(
  currentPrompt: string,
  originalPrompt?: string | null,
  requested?: RemixSourceKind,
): PromptSource {
  const current = currentPrompt.trim();
  const root = originalPrompt?.trim() || undefined;
  if ((requested === undefined || requested === "original") && root) {
    return { prompt: root, rootPrompt: root, kind: "original" };
  }
  return {
    prompt: current,
    ...(root ? { rootPrompt: root } : {}),
    kind: root ? "current" : "direct",
  };
}

export function remixDimensionsForTask(
  task: ExpandTask,
  styleLocked = false,
): RemixDimension[] {
  let allowed: RemixDimension[];
  switch (task) {
    case "text-to-image":
    case "text-to-video":
      allowed = [
        "composition",
        "camera",
        "lighting",
        "setting",
        "mood",
        "movement",
        "style",
      ];
      break;
    case "text-to-audio":
      allowed = ["mood", "movement"];
      break;
    case "image-to-video":
    case "video-to-video":
    case "retake":
    case "keyframe-interpolation":
    case "audio-driven-video":
    case "reference-to-audio-video":
      allowed = ["movement"];
      break;
  }
  return styleLocked
    ? allowed.filter((dimension) => dimension !== "style")
    : allowed;
}

export function defaultRemixDimensions(
  task: ExpandTask,
  styleLocked = false,
): RemixDimension[] {
  return remixDimensionsForTask(task, styleLocked);
}

export function validateRemixVariants(
  variants: readonly RemixVariant[],
  expected = DEFAULT_REMIX_VARIATIONS,
): RemixVariant[] {
  if (variants.length !== expected) {
    throw new Error(
      `Expected exactly ${expected} remix variants, but the host returned ${variants.length}.`,
    );
  }
  return variants.map((variant, index) => {
    const prompt = variant.prompt.trim();
    if (!prompt) throw new Error(`Remix variant ${index + 1} was empty.`);
    return { prompt, dimensions: [...variant.dimensions] };
  });
}

function referenceFingerprint(reference: unknown): unknown {
  if (!reference || typeof reference !== "object") return reference;
  const value = reference as Record<string, unknown>;
  const media = value.media;
  if (!media || typeof media !== "object") return value;
  const authority = (media as Record<string, unknown>).authority;
  return {
    ...value,
    // Provenance digest + exact descriptors carry semantic identity. Raw
    // video/audio bytes, upload handles, and server paths do not belong in a
    // synchronous UI fingerprint and can be hundreds of megabytes.
    media: { authority },
  };
}

/** Client-only identity used to stale reviewed work when conditioned media changes. */
export function conditioningFingerprint(request: ExpansionTaskRequest): string {
  const value = JSON.stringify({
    source_image: request.source_image ?? null,
    source_video: request.source_video ?? null,
    source_video_path: request.source_video_path?.trim() || null,
    extend_video: request.extend_video ?? null,
    extend_video_path: request.extend_video_path?.trim() || null,
    audio_file: request.audio_file ?? null,
    audio_file_path: request.audio_file_path?.trim() || null,
    keyframes: request.keyframes ?? null,
    pipeline: request.pipeline ?? null,
    retake_range: request.retake_range ?? null,
    references: request.references?.map(referenceFingerprint) ?? null,
    // The identity photo is conditioning media like any other: swapping the
    // face behind a reviewed rewrite must stale it through this one rule, not
    // a second identity-only staleness check.
    id_image: request.id_image ?? null,
  });
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}
