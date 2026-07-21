import { ApiError } from "./client";

/**
 * Human copy for transport-level failures.
 *
 * `fetch` rejections surface engine internals — WebKit's "Load failed" /
 * "The network connection was lost.", Chromium's "Failed to fetch" — that
 * mean nothing to the person holding the phone, and an `ApiError` built from
 * `response.statusText` can be completely empty over HTTP/2. This module maps
 * both onto directed human sentences while passing server-authored error copy
 * (which often embeds its own fix) through verbatim.
 */

/** Messages the browser engines emit when the socket itself failed. */
const TRANSPORT_MESSAGE = new RegExp(
  [
    "^load failed$",
    "^fetch failed$",
    "^failed to fetch$",
    "^network error$",
    "networkerror when attempting to fetch resource",
    "network connection was lost",
    "^the request timed out\\.?$",
    "hostname could not be found",
    "internet connection appears to be offline",
    "could not connect to the server",
    "software caused connection abort",
    "socket is not connected",
  ].join("|"),
  "i",
);

const SSE_HTTP_FAILURE = /^SSE request failed with HTTP (\d+)$/;

function hostLabel(hostName: string | null | undefined): string {
  return hostName?.trim() || "the host";
}

function messageOf(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  return error == null ? "" : String(error);
}

/**
 * Whether this failure is the connection itself dying (offline, suspension,
 * DNS, timeout) rather than the server answering. HTTP replies — `ApiError`s
 * of any status — are never transport failures.
 */
export function isTransportFailure(error: unknown): boolean {
  if (error instanceof ApiError) return false;
  // Fetch signals every network-level failure as a TypeError; the message
  // text varies by engine, so the instance check is the reliable signal.
  if (error instanceof TypeError) return true;
  return TRANSPORT_MESSAGE.test(messageOf(error).trim());
}

function describeHttpFailure(status: number, detail: string, host: string): string {
  if (status === 401 || status === 403) {
    return `${host} didn’t accept the API key. Update it in Machines and try again.`;
  }
  // Server-authored copy (it often embeds its own fix) survives verbatim.
  if (detail.trim()) return detail;
  return `${host} replied HTTP ${status}.`;
}

/**
 * Convert any thrown value from a host request into human copy, naming the
 * host when the caller knows it. Server-authored `ApiError` detail and
 * ordinary domain errors pass through unchanged.
 */
export function describeTransportError(error: unknown, hostName?: string | null): string {
  const host = hostLabel(hostName);
  if (error instanceof ApiError) return describeHttpFailure(error.status, error.message, host);
  if ((error as { name?: unknown } | null | undefined)?.name === "AbortError") return "Cancelled";
  const message = messageOf(error).trim();
  const sse = SSE_HTTP_FAILURE.exec(message);
  if (sse) return describeHttpFailure(Number(sse[1]), "", host);
  if (isTransportFailure(error)) {
    return `Couldn’t reach ${host}. Check the connection and try again.`;
  }
  if (
    error instanceof SyntaxError ||
    /JSON Parse error|Unexpected token|unexpected end of JSON/i.test(message)
  ) {
    return `${host} sent an unexpected reply.`;
  }
  if (message) return message;
  return `Something went wrong talking to ${host}.`;
}
