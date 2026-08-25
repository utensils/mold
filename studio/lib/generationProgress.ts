export type GenerationWorkPhase = "preparing" | "denoising" | "finalizing";

export interface GenerationProgressCopy {
  phase: GenerationWorkPhase;
  step: number;
  total: number;
  stage?: string | null;
}

function meaningfulStage(stage: string | null | undefined): string | null {
  if (!stage || stage === "Denoising" || stage.endsWith(" (done)")) return null;
  return stage;
}

/**
 * A StageStart can be a nested phase inside a denoise evaluation (MiniMax H3
 * streams 50 transformer blocks per evaluation) or genuinely post-denoise
 * work. Only the completed denoise counter can distinguish those cases.
 */
export function phaseForStageStart(
  current: GenerationWorkPhase,
  step: number,
  total: number,
): GenerationWorkPhase {
  if (current === "finalizing") return "finalizing";
  if (current !== "denoising") return "preparing";
  return total > 0 && step >= total ? "finalizing" : "denoising";
}

/** One progress sentence shared by web, desktop, iPhone, and Android. */
export function generationProgressCopy({
  phase,
  step,
  total,
  stage,
}: GenerationProgressCopy): string {
  const detail = meaningfulStage(stage);
  if (phase === "denoising") {
    const count = total > 0 ? `${step}/${total}` : String(step);
    return `Developing ${count}${detail ? ` — ${detail}` : ""}`;
  }
  if (phase === "finalizing") {
    return `Finalizing${detail ? ` — ${detail}` : ""}`;
  }
  return detail ?? "Preparing";
}
