import { apiJsonTo } from "./client";
import {
  assertVideoJobOwner,
  checkedVideoJobPlan,
  type VideoJobOwner,
} from "./videoJobs";
import type { VideoJob } from "../lib/api/videoJobTypes";
export interface VideoJobPage {
  jobs: VideoJob[];
  has_more: boolean;
  next_cursor: string | null;
}
export async function listVideoJobs(
  owner: VideoJobOwner,
  cursor?: string,
  isCurrent: () => boolean = () => true,
): Promise<VideoJobPage> {
  const check = () => {
    if (!isCurrent())
      throw new Error("The video's machine changed. Refresh Queue.");
  };
  check();
  await assertVideoJobOwner(owner);
  check();
  const query = new URLSearchParams({ recoverable_only: "true", limit: "50" });
  if (cursor) query.set("cursor", cursor);
  const value = await apiJsonTo<VideoJobPage>(
    owner.target,
    `/api/video-jobs?${query}`,
    { headers: { "x-mold-destination-instance": owner.instanceId! } },
  );
  check();
  if (
    !Array.isArray(value.jobs) ||
    value.jobs.length > 50 ||
    typeof value.has_more !== "boolean" ||
    (value.next_cursor !== null && typeof value.next_cursor !== "string") ||
    (value.has_more && !value.next_cursor) ||
    (!value.has_more && value.next_cursor !== null) ||
    new Set(value.jobs.map((job) => job?.id)).size !== value.jobs.length ||
    value.jobs.some(
      (job) =>
        !job ||
        typeof job.id !== "string" ||
        !job.id ||
        typeof job.state !== "string",
    )
  )
    throw new Error(
      "This machine returned invalid video Queue pagination. Refresh before continuing.",
    );
  for (const job of value.jobs) {
    checkedVideoJobPlan(job.plan);
    if (
      ![
        "queued",
        "running",
        "paused",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
      ].includes(job.state)
    )
      throw new Error("This machine returned an unsupported video job state.");
  }
  return value;
}
