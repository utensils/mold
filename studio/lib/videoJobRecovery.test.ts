import { beforeEach, expect, it } from "vitest";
import {
  VIDEO_JOB_OPERATIONS_KEY,
  readVideoJobOperations,
  saveVideoJobOperation,
  forgetVideoJobOperation,
} from "./videoJobRecovery";
const record = {
  kind: "video_job" as const,
  operationId: "known",
  hostId: "plato",
  instanceId: "installation",
  baseUrl: "http://plato",
  createdAt: 1,
};
beforeEach(() => localStorage.clear());
it("keeps only operation and owner authority, never request inputs or credentials", () => {
  saveVideoJobOperation(record);
  expect(readVideoJobOperations()).toEqual([record]);
  expect(JSON.parse(localStorage.getItem(VIDEO_JOB_OPERATIONS_KEY)!)).toEqual([
    record,
  ]);
});
it("removes only the exact original owner even if another machine uses the same UUID", () => {
  const another = {
    ...record,
    hostId: "hal",
    instanceId: "other-installation",
    baseUrl: "http://hal",
  };
  saveVideoJobOperation(record);
  saveVideoJobOperation(another);
  forgetVideoJobOperation(record);
  expect(readVideoJobOperations()).toEqual([another]);
});
it("preserves an unreadable or newer journal instead of overwriting known attempts", () => {
  for (const original of [
    "broken JSON",
    JSON.stringify([{ ...record, request: { source_image: "private" } }]),
    JSON.stringify([{ ...record, cancelRequested: null }]),
  ]) {
    localStorage.setItem(VIDEO_JOB_OPERATIONS_KEY, original);
    expect(() => saveVideoJobOperation(record)).toThrow("recovery");
    expect(localStorage.getItem(VIDEO_JOB_OPERATIONS_KEY)).toBe(original);
  }
});
