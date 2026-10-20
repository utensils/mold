import type {
  ChainContinuation,
  ChainRequestWire,
  ChainStageWire,
} from "./api/chainTypes";

export interface VideoClipDraft {
  id: string;
  prompt: string;
  frames: number;
}
export interface ClipsBaseRequest {
  model: string;
  prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance?: number;
  frames?: number | null;
  fps?: number | null;
}

const SUPPORTED = new Set([
  "model",
  "prompt",
  "frames",
  "width",
  "height",
  "fps",
  "steps",
  "guidance",
  "seed",
  "strength",
  "enable_audio",
  "negative_prompt",
  "source_image",
  "source_image_name",
  "source_fit",
  "keyframes",
  "references",
  "loras",
  "title",
  "tags",
  "collection",
  "output_format",
  "original_prompt",
  "prompt_transform",
  "batch_id",
  "batch_index",
  "batch_count",
  "batch_size",
  "save_to_gallery",
]);
/** Every nonempty ordinary option is either represented or refused by name. */
export function chainInputRefusal<T extends ClipsBaseRequest>(
  request: T,
): string | null {
  const unsupported = Object.entries(request)
    .filter(
      ([key, value]) =>
        !SUPPORTED.has(key) &&
        value != null &&
        value !== false &&
        value !== "" &&
        (!Array.isArray(value) || value.length > 0),
    )
    .map(([key]) => key);
  const extra = request as ClipsBaseRequest & {
    seed?: number;
    batch_size?: number;
    output_format?: string;
    save_to_gallery?: boolean;
  };
  if (
    extra.seed != null &&
    (!Number.isSafeInteger(extra.seed) || extra.seed < 0)
  )
    unsupported.push("seed (choose a whole number from 0 to 9007199254740991)");
  const keyframes = (
    request as ClipsBaseRequest & { keyframes?: Array<{ frame: number }> }
  ).keyframes;
  if (
    keyframes?.some(
      (frame) => frame.frame !== 0 && frame.frame !== (request.frames ?? 0) - 1,
    )
  )
    unsupported.push("intermediate keyframes (use first and last frames)");
  if ((extra.batch_size ?? 1) !== 1) unsupported.push("batch_size");
  if (extra.output_format && extra.output_format !== "mp4")
    unsupported.push("output_format (choose MP4)");
  if (extra.save_to_gallery === false)
    unsupported.push("save_to_gallery (video parts are saved to Library)");
  return unsupported.length
    ? `Video clips cannot preserve these options: ${unsupported.join(", ")}. Clear them to continue.`
    : null;
}

export function editClipOrder(
  clips: readonly VideoClipDraft[],
  id: string,
  delta: -1 | 1,
): VideoClipDraft[] {
  const result = [...clips];
  const from = result.findIndex((clip) => clip.id === id),
    to = from + delta;
  if (from < 0 || to < 0 || to >= result.length) return result;
  [result[from], result[to]] = [result[to]!, result[from]!];
  return result;
}

export function buildClipsRequest<T extends ClipsBaseRequest>(
  base: T,
  clips: readonly VideoClipDraft[],
  motionTail: number,
  continuation?: ChainContinuation,
): ChainRequestWire & { stages: ChainStageWire[] } {
  const refusal = chainInputRefusal(base);
  if (refusal) throw new Error(refusal);
  if (
    !clips.length ||
    clips.length > 16 ||
    clips.some((clip) => !clip.prompt.trim())
  )
    throw new Error("Write a prompt for every clip (up to 16 clips per part).");
  const fields = base as ClipsBaseRequest &
    Partial<ChainRequestWire> & {
      source_image?: string;
      negative_prompt?: string;
      keyframes?: ChainStageWire["keyframes"];
      references?: ChainStageWire["generation_references"];
      loras?: ChainStageWire["loras"];
    };
  if (
    continuation &&
    motionTail > 0 &&
    (fields.source_image ||
      fields.keyframes?.some((frame) => frame.frame === 0))
  )
    throw new Error(
      "A continued video already carries its first frame. Remove the authored first frame.",
    );
  // Compute repeated reference costs before constructing stages or serializing.
  // These are the protocol's bounded-part budgets, including an owned incoming tail.
  const references = fields.references ?? [];
  if (references.some((reference) => reference.media?.authority !== "inline"))
    throw new Error(
      "Video clips require original inline reference media. Reattach the original inputs before making this video.",
    );
  const decodedBytes = (data: string) =>
    Math.floor((data.length * 3) / 4) -
    (data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0);
  const referenceBytes = references.reduce(
    (total, reference) =>
      total +
      (reference.media.authority === "inline"
        ? decodedBytes(reference.media.data)
        : 0),
    0,
  );
  const mediaBytes =
    referenceBytes * clips.length +
    (fields.source_image ? decodedBytes(fields.source_image) : 0) +
    (fields.keyframes ?? []).reduce(
      (total, frame) => total + decodedBytes(frame.image),
      0,
    );
  if (mediaBytes > 32 * 1024 * 1024)
    throw new Error(
      "These video clips exceed 32 MiB of total reference media. Use fewer clips or smaller originals in this part.",
    );
  const members =
    references.length * clips.length +
    Number(Boolean(fields.source_image)) +
    (fields.keyframes?.length ?? 0) +
    (continuation ? motionTail : 0);
  if (members > 64)
    throw new Error(
      "These video clips exceed 64 retained inputs. Use fewer clips or references in this part.",
    );
  const stages: ChainStageWire[] = clips.map((clip, index) => ({
    prompt: clip.prompt,
    frames: clip.frames,
    transition: "smooth",
    ...(fields.negative_prompt
      ? { negative_prompt: fields.negative_prompt }
      : {}),
    ...(fields.loras?.length ? { loras: fields.loras } : {}),
    ...(fields.references?.length
      ? { generation_references: fields.references }
      : {}),
    ...(index === 0 && fields.source_image
      ? { source_image: fields.source_image }
      : {}),
    ...(fields.keyframes?.length
      ? {
          keyframes: fields.keyframes
            .filter((frame) =>
              frame.frame === 0 ? index === 0 : index === clips.length - 1,
            )
            .map((frame) => ({
              ...frame,
              frame: frame.frame === 0 ? 0 : clip.frames - 1,
            })),
        }
      : {}),
  }));
  return {
    model: base.model,
    width: base.width,
    height: base.height,
    fps: base.fps ?? 24,
    steps: base.steps,
    guidance: base.guidance ?? 1,
    stages,
    motion_tail_frames: motionTail,
    output_format: "mp4",
    output_mode: "sequence",
    ...(continuation ? { continue_from: continuation } : {}),
    ...Object.fromEntries(
      [
        "seed",
        "strength",
        "enable_audio",
        "title",
        "tags",
        "collection",
        "original_prompt",
        "prompt_transform",
        "batch_id",
        "batch_index",
        "batch_count",
        "source_fit",
      ]
        .filter(
          (key) =>
            (fields as unknown as Record<string, unknown>)[key] !== undefined,
        )
        .map((key) => [
          key,
          (fields as unknown as Record<string, unknown>)[key],
        ]),
    ),
  };
}
