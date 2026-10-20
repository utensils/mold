import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import {
  applyMinimaxH3ReferenceCrops,
  minimaxH3ReferenceProjection,
  type MinimaxH3AuthoringState,
} from "./minimaxH3Authoring";
import { applySourceFitPreprocess } from "./sourceFitPreprocess";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  parseSourceFitPolicy,
} from "./sourceFit";
import type { ClipsBaseRequest } from "./chainDraft";
import type { ChainStageWire } from "./api/chainTypes";

/** Request-only pixels: preserve the editable originals, typed references and names.
 * Call only after Make video; Check length must never start an upscaler job.
 */
export async function prepareClipsInputs<T extends ClipsBaseRequest>(
  request: T,
  deps: Parameters<typeof applySourceFitPreprocess>[1] & {
    current: () => boolean;
  },
  referenceAuthoring?: MinimaxH3AuthoringState | null,
): Promise<T> {
  const assertCurrent = () => {
    if (!deps.current())
      throw new Error(
        "The video inputs or machine changed while preparing. Review the clips and make the video again.",
      );
  };
  assertCurrent();
  const prepared = (await prepareClipsReferences(
    request,
    referenceAuthoring,
    deps,
  )) as T & {
    source_image?: string | null;
    source_fit?: unknown;
    keyframes?: ChainStageWire["keyframes"];
  };
  const hasFrames = !!prepared.source_image || !!prepared.keyframes?.length;
  const parsedPolicy = parseSourceFitPolicy(prepared.source_fit);
  if (hasFrames && prepared.source_fit != null && !parsedPolicy)
    throw new Error(
      "This source fit option is not supported. Choose a source fit option again before making the video.",
    );
  const policy = coerceSourceFitForMaskless(
    parsedPolicy ?? defaultSourceFitPolicy(),
  );
  if (
    hasFrames &&
    policy.mode === "upscale-then-fit" &&
    (!policy.upscalerModel || !deps.upscale)
  )
    throw new Error(
      "Choose a source upscaler before making this video, or choose another source fit option.",
    );
  const guarded = {
    ...deps,
    cache: deps.cache ?? new SourceFitPreprocessCache(),
    ops: {
      imageSize: async (data: string) => {
        assertCurrent();
        const result = await deps.ops.imageSize(data);
        assertCurrent();
        return result;
      },
      fitImage: async (...args: Parameters<typeof deps.ops.fitImage>) => {
        assertCurrent();
        const result = await deps.ops.fitImage(...args);
        assertCurrent();
        return result;
      },
      buildMask: deps.ops.buildMask,
    },
    ...(deps.upscale
      ? {
          upscale: async (data: string, model: string) => {
            assertCurrent();
            const result = await deps.upscale!(data, model);
            assertCurrent();
            return result;
          },
        }
      : {}),
  };
  if (!hasFrames) return prepared;
  const fit = async (source: string) => {
    assertCurrent();
    const result = await applySourceFitPreprocess(
      {
        source,
        mask: null,
        policy,
        target: { width: request.width, height: request.height },
      },
      guarded,
    );
    assertCurrent();
    return result.source ?? source;
  };
  if (prepared.source_image)
    prepared.source_image = await fit(prepared.source_image);
  if (prepared.keyframes) {
    const frames = [];
    for (const frame of prepared.keyframes)
      frames.push({ ...frame, image: await fit(frame.image) });
    prepared.keyframes = frames;
  }
  prepared.source_fit = policy;
  assertCurrent();
  return prepared;
}

/** CPU-only canonical reference crops; safe for read-only length checks. */
export async function prepareClipsReferences<T extends ClipsBaseRequest>(
  request: T,
  authoring: MinimaxH3AuthoringState | null | undefined,
  deps: {
    ops: Pick<import("./sourceFitCanvas").SourceFitCanvasOps, "fitImage">;
    current: () => boolean;
  },
): Promise<T> {
  const check = () => {
    if (!deps.current())
      throw new Error(
        "The video inputs or machine changed while preparing. Review the clips again.",
      );
  };
  check();
  const activeReferences = (request as T & { references?: unknown[] })
    .references;
  if (
    !activeReferences?.length ||
    !authoring?.references.some((entry) => entry.crop)
  )
    return { ...request };
  const active = JSON.stringify(activeReferences);
  if (
    active !== JSON.stringify(minimaxH3ReferenceProjection(authoring)) &&
    active !==
      JSON.stringify(authoring.references.map((entry) => entry.reference))
  )
    return { ...request };
  const cropped = await applyMinimaxH3ReferenceCrops(authoring, {
    fitImage: async (...args) => {
      check();
      const data = await deps.ops.fitImage(...args);
      check();
      return data;
    },
  });
  check();
  return { ...request, references: minimaxH3ReferenceProjection(cropped) };
}
