import type { StreamTarget } from "../api";

interface MediaTicket {
  token: string | null;
  expires_at: number | null;
  auth_required?: boolean;
}

function base(target?: StreamTarget): string {
  return target?.baseUrl?.replace(/\/$/, "") ?? "";
}

export function sequenceStageMediaPath(
  jobId: string,
  stageIdx: number,
): string {
  return `/api/chain-jobs/${encodeURIComponent(jobId)}/stages/${encodeURIComponent(
    String(stageIdx),
  )}/media`;
}

export async function sequenceStageMediaUrl(
  jobId: string,
  stageIdx: number,
  target?: StreamTarget,
): Promise<string> {
  const path = sequenceStageMediaPath(jobId, stageIdx);
  const direct = `${base(target)}${path}`;
  if (!target?.apiKey) return direct;

  const ticketPath = `${path}-token`;
  const response = await fetch(`${base(target)}${ticketPath}`, {
    method: "POST",
    headers: { "x-api-key": target.apiKey },
  });
  if (!response.ok) {
    throw new Error(`stage media ticket failed: ${response.status}`);
  }
  const ticket = (await response.json()) as MediaTicket;
  if (ticket.auth_required === false) return direct;
  if (!ticket.token || !Number.isSafeInteger(ticket.expires_at)) {
    throw new Error("host returned an invalid stage media ticket");
  }
  const url = new URL(direct, window.location.origin);
  url.searchParams.set("media_token", ticket.token);
  url.searchParams.set("expires", String(ticket.expires_at));
  return url.toString();
}
