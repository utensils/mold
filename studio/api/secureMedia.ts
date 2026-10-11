import { CONNECTION_MODE_HEADER } from "./connectionMode";
import {
  assertRegularRequest,
  isLoopbackOrigin,
  relayFetch,
  secureRequestUrl,
} from "./relayTransport";
import { secureHostForOrigin } from "./secureStore";

interface MediaEntry {
  url: string;
  headers: Headers;
  touched: number;
  active: Set<() => void>;
  dispose: () => void;
}
const entries = new Map<string, MediaEntry>();
let workerScript: string | undefined;
let registration: Promise<ServiceWorkerRegistration> | undefined;
let installed = false;
let nativeAdapter:
  | { open: (url: URL) => Promise<string>; close: (url: string) => void }
  | undefined;
export function configureSecureMediaAdapter(
  adapter: typeof nativeAdapter,
): void {
  nativeAdapter = adapter;
}

export function configureSecureMediaWorker(url: string): void {
  workerScript = url;
}
async function ensureWorker(): Promise<ServiceWorkerRegistration> {
  if (!workerScript || !navigator.serviceWorker)
    throw new Error("This surface needs its native secure media bridge.");
  registration ??= (async () => {
    const script = new URL(workerScript!, location.href);
    const scope = new URL(".", script).pathname;
    const registered = await navigator.serviceWorker.register(script.href, {
      scope,
    });
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      await new Promise<void>((resolve, reject) => {
        const done = () => {
          clearTimeout(timer);
          navigator.serviceWorker.removeEventListener("controllerchange", done);
          resolve();
        };
        const timer = setTimeout(() => {
          navigator.serviceWorker.removeEventListener("controllerchange", done);
          reject(
            new Error(
              "Secure media did not become ready. Reload Studio and retry.",
            ),
          );
        }, 10000);
        navigator.serviceWorker.addEventListener("controllerchange", done, {
          once: true,
        });
        if (navigator.serviceWorker.controller) done();
      });
    }
    if (!installed) {
      navigator.serviceWorker.addEventListener("message", onMediaRequest);
      installed = true;
    }
    return registered;
  })().catch((error: unknown) => {
    registration = undefined;
    throw error;
  });
  return registration;
}

/** Returns a local capability URL; every Range/HEAD request still goes through Noise. */
export async function secureMediaUrl(
  input: string,
  headers: HeadersInit = {},
  signal?: AbortSignal,
): Promise<string | null> {
  signal?.throwIfAborted();
  const url = secureRequestUrl(input);
  const mode = new Headers(headers).get(CONNECTION_MODE_HEADER) ?? "secure";
  const profile = await secureHostForOrigin(url.origin, mode !== "regular");
  if (mode === "regular") {
    assertRegularRequest(url, new Headers(headers), profile);
    return null;
  }
  if (!profile && isLoopbackOrigin(url)) return null;
  if (!profile)
    throw new Error("Pair this machine securely to play its media.");
  if (nativeAdapter) {
    const result = await nativeAdapter.open(url);
    if (signal?.aborted) {
      nativeAdapter.close(result);
      signal.throwIfAborted();
    }
    signal?.addEventListener("abort", () => nativeAdapter?.close(result), {
      once: true,
    });
    return result;
  }
  const worker = await ensureWorker();
  signal?.throwIfAborted();
  const capability = crypto.randomUUID();
  const requestHeaders = new Headers(headers);
  requestHeaders.set(CONNECTION_MODE_HEADER, "secure");
  requestHeaders.delete("x-api-key");
  requestHeaders.delete("authorization");
  requestHeaders.delete("cookie");
  const result = new URL(`__mold_media/${capability}`, worker.scope).href;
  const abort = () => releaseSecureMediaUrl(result);
  entries.set(capability, {
    url: url.href,
    headers: requestHeaders,
    touched: Date.now(),
    active: new Set(),
    dispose: () => signal?.removeEventListener("abort", abort),
  });
  signal?.addEventListener("abort", abort, { once: true });
  if (entries.size > 128) {
    const oldest = [...entries].sort((a, b) => a[1].touched - b[1].touched)[0];
    if (oldest)
      releaseSecureMediaUrl(
        new URL(`__mold_media/${oldest[0]}`, worker.scope).href,
      );
  }
  return result;
}

export function releaseSecureMediaUrl(value: string): void {
  nativeAdapter?.close(value);
  try {
    const capability = new URL(value).pathname.split("/__mold_media/")[1] ?? "";
    const entry = entries.get(capability);
    entries.delete(capability);
    entry?.dispose();
    for (const cancel of entry?.active ?? []) cancel();
  } catch {
    /* Ordinary blob URL. */
  }
}

function onMediaRequest(event: MessageEvent): void {
  if (
    event.source !== navigator.serviceWorker.controller ||
    event.data?.type !== "mold-secure-media-v3" ||
    event.ports.length !== 1
  )
    return;
  const port = event.ports[0]!;
  const entry = entries.get(event.data.capability as string);
  if (!entry || !["GET", "HEAD"].includes(event.data.method)) {
    port.postMessage({ type: "error" });
    port.close();
    return;
  }
  entry.touched = Date.now();
  const controller = new AbortController();
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let reading = false;
  let cancelled = false;
  const cancel = () => {
    if (cancelled) return;
    cancelled = true;
    entry.active.delete(cancel);
    controller.abort();
    void reader?.cancel().catch(() => {});
    port.postMessage({ type: "error" });
    port.close();
  };
  entry.active.add(cancel);
  port.onmessage = ({ data }: MessageEvent) => {
    if (data?.type === "cancel") {
      cancel();
      return;
    }
    if (data?.type !== "pull" || reading || cancelled || !reader) return;
    reading = true;
    void reader
      .read()
      .then((part) => {
        reading = false;
        if (cancelled) return;
        if (part.done) {
          port.postMessage({ type: "end" });
          cancel();
        } else {
          if (part.value.length > 16384)
            throw new Error("Secure media chunk exceeded its bound.");
          const bytes = new Uint8Array(part.value).buffer;
          port.postMessage({ type: "data", bytes }, [bytes]);
        }
      })
      .catch(() => {
        if (!cancelled) port.postMessage({ type: "error" });
        cancel();
      });
  };
  const headers = new Headers(entry.headers);
  for (const [field, name] of [
    ["range", "range"],
    ["ifRange", "if-range"],
    ["ifNoneMatch", "if-none-match"],
    ["ifModifiedSince", "if-modified-since"],
  ] as const) {
    if (typeof event.data[field] === "string")
      headers.set(name, event.data[field]);
  }
  void relayFetch(entry.url, {
    method: event.data.method as string,
    headers,
    signal: controller.signal,
  })
    .then((response) => {
      if (cancelled) {
        void response.body?.cancel();
        return;
      }
      reader = response.body?.getReader();
      port.postMessage({
        type: "headers",
        status: response.status,
        headers: [...response.headers.entries()],
      });
      if (!reader) {
        port.postMessage({ type: "end" });
        cancel();
      }
    })
    .catch(() => {
      if (!cancelled) port.postMessage({ type: "error" });
      cancel();
    });
}
