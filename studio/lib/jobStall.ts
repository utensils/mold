/** Additive server evidence: stalled work remains a live job and may hold its device. */
export interface JobStallDiagnostic {
  stage: string;
  last_completed_work_unix_ms: number;
  elapsed_stall_ms: number;
  cancel_requested: boolean;
  device_still_held: boolean;
}
export function parseJobStall(value: unknown): JobStallDiagnostic | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Record<string, unknown>;
  if (
    typeof row.stage !== "string" ||
    !row.stage.trim() ||
    !Number.isSafeInteger(row.last_completed_work_unix_ms) ||
    Number(row.last_completed_work_unix_ms) < 0 ||
    !Number.isSafeInteger(row.elapsed_stall_ms) ||
    Number(row.elapsed_stall_ms) < 0 ||
    typeof row.cancel_requested !== "boolean" ||
    typeof row.device_still_held !== "boolean"
  )
    return null;
  return {
    stage: row.stage,
    last_completed_work_unix_ms: Number(row.last_completed_work_unix_ms),
    elapsed_stall_ms: Number(row.elapsed_stall_ms),
    cancel_requested: row.cancel_requested,
    device_still_held: row.device_still_held,
  };
}
export function cancellationRequestedLabel(deviceStillHeld: boolean): string {
  return deviceStillHeld
    ? "Cancel requested; device still held"
    : "Cancel requested";
}
export function jobStallLabel(stall: JobStallDiagnostic): string {
  return stall.cancel_requested
    ? cancellationRequestedLabel(stall.device_still_held)
    : `Stalled · ${stall.stage}`;
}
export interface QueueCancellationResult {
  terminal: boolean;
  deviceStillHeld: boolean;
  stall: JobStallDiagnostic | null;
}
/** HTTP 202 acknowledges intent, never termination, even with an unreadable body. */
export async function parseQueueCancellationResponse(
  response: Response,
): Promise<QueueCancellationResult> {
  if (response.status !== 202)
    return { terminal: true, deviceStillHeld: false, stall: null };
  const value: unknown = await response.json().catch(() => null);
  const row =
    value && typeof value === "object"
      ? (value as Record<string, unknown>)
      : {};
  const stall = parseJobStall(row.stall);
  return {
    terminal: false,
    deviceStillHeld:
      row.device_still_held === true || stall?.device_still_held === true,
    stall,
  };
}
