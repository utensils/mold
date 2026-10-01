import { fetchEventSource } from "@microsoft/fetch-event-source";
import { originApiKey, originAuthenticatedFetch } from "./originAuth";

/** Fetch-based EventSource for authenticated serving-origin GET streams. */
export function createOriginEventSource(url: string): EventSource {
  if (!originApiKey()) return new EventSource(url);
  const target = new EventTarget();
  const controller = new AbortController();
  const source = Object.assign(target, {
    onerror: null as ((event: Event) => void) | null,
    onopen: null as ((event: Event) => void) | null,
    onmessage: null as ((event: MessageEvent) => void) | null,
    close: () => controller.abort(),
  });
  void fetchEventSource(url, {
    fetch: originAuthenticatedFetch,
    signal: controller.signal,
    openWhenHidden: true,
    async onopen(response) {
      if (
        !response.ok ||
        !response.headers.get("content-type")?.includes("text/event-stream")
      ) {
        throw new Error("Event stream was refused");
      }
      source.onopen?.(new Event("open"));
    },
    onmessage(message) {
      const event = new MessageEvent(message.event || "message", {
        data: message.data,
        lastEventId: message.id,
      });
      source.dispatchEvent(event);
      if (event.type === "message") source.onmessage?.(event);
    },
    onerror(error) {
      throw error;
    },
    onclose() {
      throw new Error("Event stream closed");
    },
  }).catch(() => {
    if (!controller.signal.aborted) source.onerror?.(new Event("error"));
  });
  return source as unknown as EventSource;
}
