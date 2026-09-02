export type MinimaxH3Task = "fl2va" | "ref2va";

export const MINIMAX_H3_FL2VA_OFFICIAL = "minimax-h3-fl2va:official-bf16";
export const MINIMAX_H3_REF2VA_OFFICIAL = "minimax-h3-ref2va:official-bf16";
export const MINIMAX_H3_FL2VA_COMFY = "minimax-h3-fl2va:comfy-pruned-int8";
export const MINIMAX_H3_REF2VA_COMFY = "minimax-h3-ref2va:comfy-pruned-int8";
export const MINIMAX_H3_FL2VA_COMFY_NVFP4 =
  "minimax-h3-fl2va:comfy-pruned-nvfp4";
export const MINIMAX_H3_REF2VA_COMFY_NVFP4 =
  "minimax-h3-ref2va:comfy-pruned-nvfp4";
export const MINIMAX_H3_FL2VA_COMFY_TURBO_8STEP =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step";
export const MINIMAX_H3_FL2VA_COMFY_TURBO_4STEP_768P =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p";
export const MINIMAX_H3_FL2VA_COMFY_TURBO_4STEP_768P_V11 =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1";
export const MINIMAX_H3_FL2VA_COMFY_TURBO_8STEP_768P =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p";
export const MINIMAX_H3_REF2VA_COMFY_TURBO_4STEP =
  "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step";

function normalize(value: string): string {
  return value.trim().toLowerCase().replaceAll("_", "-");
}

/** Resolve only the released H3 aliases and exact task/layout partitions. */
export function canonicalMinimaxH3ModelName(
  model: string | null | undefined,
): string | null {
  const value = normalize(model ?? "");
  if (
    value === "minimax-h3" ||
    value === "minimaxh3" ||
    value === "minimax-h3-fl2va"
  ) {
    return MINIMAX_H3_FL2VA_COMFY;
  }
  if (value === "minimax-h3-ref2va") {
    return MINIMAX_H3_REF2VA_COMFY;
  }
  if (
    value === MINIMAX_H3_FL2VA_OFFICIAL ||
    value === MINIMAX_H3_REF2VA_OFFICIAL ||
    value === MINIMAX_H3_FL2VA_COMFY ||
    value === MINIMAX_H3_REF2VA_COMFY ||
    value === MINIMAX_H3_FL2VA_COMFY_NVFP4 ||
    value === MINIMAX_H3_REF2VA_COMFY_NVFP4 ||
    value === MINIMAX_H3_FL2VA_COMFY_TURBO_8STEP ||
    value === MINIMAX_H3_FL2VA_COMFY_TURBO_4STEP_768P ||
    value === MINIMAX_H3_FL2VA_COMFY_TURBO_4STEP_768P_V11 ||
    value === MINIMAX_H3_FL2VA_COMFY_TURBO_8STEP_768P ||
    value === MINIMAX_H3_REF2VA_COMFY_TURBO_4STEP
  ) {
    return value;
  }
  return null;
}

export function isMinimaxH3Family(family: string | null | undefined): boolean {
  const value = normalize(family ?? "").replaceAll("-", "");
  return value === "minimaxh3";
}

/** An opaque catalog ID stays unresolved instead of being guessed into a task. */
export function minimaxH3TaskForModel(
  model: string | null | undefined,
): MinimaxH3Task | null {
  const value = canonicalMinimaxH3ModelName(model);
  if (value?.startsWith("minimax-h3-ref2va:")) return "ref2va";
  if (value?.startsWith("minimax-h3-fl2va:")) return "fl2va";
  return null;
}

/** Family metadata may identify H3 even when an opaque model ID has no task. */
export function isMinimaxH3Identity(
  family: string | null | undefined,
  model: string | null | undefined,
): boolean {
  if (isMinimaxH3Family(family)) return true;
  const value = normalize(model ?? "");
  return (
    value === "minimax-h3" ||
    value === "minimaxh3" ||
    value === "minimax-h3-fl2va" ||
    value === "minimax-h3-ref2va" ||
    value.startsWith("minimax-h3-fl2va:") ||
    value.startsWith("minimax-h3-ref2va:")
  );
}
