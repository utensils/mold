import type { VideoJobAdmission } from "../api/videoJobs";

export const VIDEO_JOB_OPERATIONS_KEY = "mold.video-jobs.operations.v1";
const strings = ["operationId", "hostId", "instanceId", "baseUrl"] as const;
const allowed = new Set([
  "kind",
  ...strings,
  "connectionMode",
  "createdAt",
  "cancelRequested",
  "cancelAcknowledged",
]);
function valid(value: unknown): value is VideoJobAdmission {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return (
    record.kind === "video_job" &&
    strings.every(
      (field) =>
        typeof record[field] === "string" &&
        (record[field] as string).length > 0,
    ) &&
    typeof record.createdAt === "number" &&
    Number.isFinite(record.createdAt) &&
    (record.connectionMode === undefined ||
      typeof record.connectionMode === "string") &&
    [record.cancelRequested, record.cancelAcknowledged].every(
      (field) => field === undefined || typeof field === "boolean",
    ) &&
    Object.keys(record).every((field) => allowed.has(field))
  );
}
function unreadable(): Error {
  return new Error(
    "This device’s video recovery journal is unreadable or newer. Keep it before repairing storage.",
  );
}
export function readVideoJobOperations(): VideoJobAdmission[] {
  let value: unknown;
  try {
    value = JSON.parse(localStorage.getItem(VIDEO_JOB_OPERATIONS_KEY) ?? "[]");
  } catch {
    throw unreadable();
  }
  if (!Array.isArray(value) || value.some((record) => !valid(record)))
    throw unreadable();
  return value as VideoJobAdmission[];
}
function sameOperation(
  left: VideoJobAdmission,
  right: VideoJobAdmission,
): boolean {
  return (
    left.operationId === right.operationId &&
    left.hostId === right.hostId &&
    left.instanceId === right.instanceId &&
    left.baseUrl === right.baseUrl &&
    left.connectionMode === right.connectionMode
  );
}
export function saveVideoJobOperation(record: VideoJobAdmission): void {
  if (!valid(record)) throw unreadable();
  const entries = readVideoJobOperations().filter(
    (entry) => !sameOperation(entry, record),
  );
  localStorage.setItem(
    VIDEO_JOB_OPERATIONS_KEY,
    JSON.stringify([...entries, record]),
  );
}
export function forgetVideoJobOperation(record: VideoJobAdmission): void {
  const entries = readVideoJobOperations().filter(
    (entry) => !sameOperation(entry, record),
  );
  localStorage.setItem(VIDEO_JOB_OPERATIONS_KEY, JSON.stringify(entries));
}
