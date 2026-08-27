import { describeTransportError } from "../lib/api/errors";
import type { CompleteEvent } from "../lib/api/types";
import { isCancelledError, type Job } from "../lib/generationJob";

export interface MobileGenerationOutcome {
  advisories: string[];
  completed: Job[];
  latestCompleted: Job | null;
  status: { message: string; isError: boolean } | null;
  announcement: string | null;
  refreshGallery: boolean;
}

export function mobileCompletionSummary(result: CompleteEvent): string {
  const timing =
    result.generation_time_ms > 0 ? `${(result.generation_time_ms / 1000).toFixed(1)}s · ` : "";
  return `${timing}seed ${result.seed_used}`;
}

/**
 * One sentence naming WHICH sibling of a batch failed and the prompt it was
 * reviewed with. Shared so the sequence path and the durable print path
 * cannot word a partial failure differently.
 */
export function preparedVariationFailure(
  oneBasedIndex: number,
  prompt: string,
  detail: string,
): string {
  const trimmed = prompt.length > 120 ? `${prompt.slice(0, 117)}…` : prompt;
  return `Variation ${oneBasedIndex}, “${trimmed}”, failed: ${detail}`;
}

export function summarizeMobileGenerationOutcome(
  jobs: readonly Job[],
  options: { hostLabel: string; prepared: boolean },
): MobileGenerationOutcome {
  const completed = jobs.filter(
    (candidate): candidate is Job & { result: CompleteEvent } =>
      candidate.status === "complete" && candidate.result !== null,
  );
  const latestCompleted = completed.at(-1) ?? null;
  const unconfirmedCancellation = jobs.find((candidate) =>
    candidate.error?.includes("remote cancellation was not confirmed"),
  );
  const failed = jobs.find((candidate) => candidate.error && !isCancelledError(candidate.error));
  const failedError = failed?.error
    ? describeTransportError(failed.error, options.hostLabel)
    : null;
  const failedVariations = options.prepared
    ? jobs.flatMap((candidate, index) => {
        if (!candidate.error || isCancelledError(candidate.error)) return [];
        const prompt =
          candidate.prompt.length > 120 ? `${candidate.prompt.slice(0, 117)}…` : candidate.prompt;
        return [
          `Variation ${index + 1}, “${prompt}”, failed: ${describeTransportError(
            candidate.error,
            options.hostLabel,
          )}`,
        ];
      })
    : [];
  const preparedFailureSummary = failedVariations.join(" ");
  const failedCount = jobs.filter(
    (candidate) => candidate.error && !isCancelledError(candidate.error),
  ).length;
  const cancelled = jobs.some((candidate) => isCancelledError(candidate.error));
  let status: MobileGenerationOutcome["status"] = null;
  let announcement: string | null = null;

  if (latestCompleted?.result) {
    if (latestCompleted.resultError) {
      const previewDetail = describeTransportError(latestCompleted.resultError, options.hostLabel);
      status = { message: previewDetail, isError: true };
      announcement = `${completed.length} of ${jobs.length} generations completed, but the latest preview is unavailable. ${previewDetail}`;
    } else {
      status = {
        message: `${completed.length > 1 ? `${completed.length} prints · ` : ""}${mobileCompletionSummary(latestCompleted.result)}`,
        isError: false,
      };
      announcement =
        completed.length === 1 && jobs.length === 1
          ? "Generation completed."
          : `${completed.length} of ${jobs.length} generations completed.`;
    }
    if (unconfirmedCancellation?.error || failedError) {
      status = {
        message: [
          `${completed.length} of ${jobs.length} completed`,
          failedError,
          unconfirmedCancellation?.error,
        ]
          .filter(Boolean)
          .join(" · "),
        isError: true,
      };
      announcement = [
        `${completed.length} generations completed.`,
        failedError
          ? options.prepared
            ? `${failedCount} failed. ${preparedFailureSummary}`
            : `${failedCount} failed. ${failedError}`
          : "",
        unconfirmedCancellation?.error
          ? `Cancellation failed. ${unconfirmedCancellation.error}`
          : "",
      ]
        .filter(Boolean)
        .join(" ");
    }
  } else if (unconfirmedCancellation?.error || failedError) {
    status = {
      message: [failedError, unconfirmedCancellation?.error].filter(Boolean).join(" · "),
      isError: true,
    };
    announcement = [
      failedError
        ? options.prepared
          ? `Generation failed. ${preparedFailureSummary}`
          : `Generation failed. ${failedError}`
        : "",
      unconfirmedCancellation?.error ? `Cancellation failed. ${unconfirmedCancellation.error}` : "",
    ]
      .filter(Boolean)
      .join(" ");
  } else if (cancelled) {
    status = { message: "Cancelled", isError: false };
    announcement = `${jobs.length} generation${jobs.length === 1 ? "" : "s"} cancelled.`;
  }

  return {
    advisories: [...new Set(jobs.flatMap((candidate) => candidate.requestWarnings))],
    completed,
    latestCompleted,
    status,
    announcement,
    refreshGallery: latestCompleted !== null,
  };
}
