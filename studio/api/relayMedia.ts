import { relayFetch, resolveRelayObjectUrl } from "./relayTransport";
const MediaURL = URL;
export interface RelayMediaTicket {
  url?: string;
  relay?: { id: string; state: string };
  token?: string | null;
}
/** Pending staging is separate from the player's short-lived HTTP read URL. */
export async function resolveRelayMedia(
  ticket: RelayMediaTicket,
  baseUrl: string,
  headers: HeadersInit,
  signal?: AbortSignal,
): Promise<string | null> {
  const origin = new MediaURL(baseUrl || window.location.origin).origin;
  if (ticket.url) return resolveRelayObjectUrl(ticket.url, origin, signal);
  if (!ticket.relay) return null;
  if (!ticket.relay.id || ticket.relay.state !== "pending")
    throw new Error("The relay returned an invalid media transfer.");
  const deadline = Date.now() + 840000;
  while (Date.now() < deadline) {
    signal?.throwIfAborted();
    const response = await relayFetch(
      `${origin}/_mold/relay/media/${encodeURIComponent(ticket.relay.id)}`,
      { headers, signal: signal ?? null, redirect: "error" },
    );
    if (!response.ok)
      throw new Error(`Relay media staging failed: ${response.status}`);
    const status = (await response.json()) as {
      state: string;
      url?: string;
      expires_at?: number;
    };
    if (status.state === "ready" && status.url)
      return resolveRelayObjectUrl(status.url, origin, signal);
    if (status.state !== "pending")
      throw new Error("The relay could not stage this media file.");
    await new Promise<void>((resolve, reject) => {
      const abort = () => {
        clearTimeout(timer);
        reject(signal?.reason ?? new Error("Media request cancelled."));
      };
      const timer = setTimeout(() => {
        signal?.removeEventListener("abort", abort);
        resolve();
      }, 1000);
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
    });
  }
  throw new Error("Media staging timed out. Try again.");
}
