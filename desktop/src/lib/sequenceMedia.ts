import { apiJsonTo, type ApiTarget } from "./api/client";

interface MediaTicket {
  token: string | null;
  expires_at: number | null;
  auth_required?: boolean;
}

export function sequenceStageMediaPath(jobId: string, stageIdx: number): string {
  return `/api/chain-jobs/${encodeURIComponent(jobId)}/stages/${encodeURIComponent(
    String(stageIdx),
  )}/media`;
}

export async function sequenceStageMediaUrl(
  target: ApiTarget,
  jobId: string,
  stageIdx: number,
): Promise<string> {
  const path = sequenceStageMediaPath(jobId, stageIdx);
  const direct = `${target.baseUrl.replace(/\/$/, "")}${path}`;
  if (!target.apiKey) return direct;

  const ticket = await apiJsonTo<MediaTicket>(target, `${path}-token`, {
    method: "POST",
  });
  if (ticket.auth_required === false) return direct;
  if (!ticket.token || !Number.isSafeInteger(ticket.expires_at)) {
    throw new Error("host returned an invalid stage media ticket");
  }
  const url = new URL(direct);
  url.searchParams.set("media_token", ticket.token);
  url.searchParams.set("expires", String(ticket.expires_at));
  return url.toString();
}
