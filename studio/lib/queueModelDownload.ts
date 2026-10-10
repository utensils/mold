/** Undefined means an older row needs the legacy batch-code lookup. A present
 * but malformed hint never falls through to a different recovery authority. */
export function queueModelDownloadAvailability(row: {
  state: string;
  retryable?: boolean | null;
  model: string;
  model_download?: unknown;
}): boolean | undefined {
  if (row.state !== "held" || row.retryable !== true) return false;
  if (!Object.prototype.hasOwnProperty.call(row, "model_download"))
    return undefined;
  const hint = row.model_download;
  return !!(
    hint &&
    typeof hint === "object" &&
    "model" in hint &&
    typeof hint.model === "string" &&
    hint.model.trim().length > 0 &&
    hint.model === row.model
  );
}
