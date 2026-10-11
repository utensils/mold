// Copied verbatim by the Vite plugin into each trusted client distribution.
// No credentials or plaintext media are cached by this worker.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
  event.waitUntil(self.clients.claim()),
);
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const prefix = new URL("__mold_media/", self.registration.scope).pathname;
  if (url.origin !== self.location.origin || !url.pathname.startsWith(prefix))
    return;
  const capability = url.pathname.slice(prefix.length);
  if (
    !/^[0-9a-f-]{36}$/.test(capability) ||
    url.search ||
    !["GET", "HEAD"].includes(event.request.method)
  ) {
    event.respondWith(new Response("Invalid media request", { status: 400 }));
    return;
  }
  let finish;
  const lifetime = new Promise((resolve) => {
    finish = resolve;
  });
  event.waitUntil(lifetime);
  event.respondWith(
    (async () => {
      const client = event.clientId && (await self.clients.get(event.clientId));
      if (!client || !client.url.startsWith(self.registration.scope)) {
        finish();
        return new Response("Open Mold Studio to play this media", {
          status: 403,
        });
      }
      if (event.request.signal.aborted) {
        finish();
        return new Response("Secure media cancelled", { status: 499 });
      }
      const { port1, port2 } = new MessageChannel();
      let controller;
      let resolved = false;
      let stopped = false;
      let completeHeaders;
      let completePull;
      let timer;
      const stop = () => {
        if (stopped) return;
        stopped = true;
        clearTimeout(timer);
        event.request.signal.removeEventListener("abort", abort);
        completePull?.();
        completePull = undefined;
        port1.postMessage({ type: "cancel" });
        port1.close();
        finish();
      };
      const fail = (message, status = 502) => {
        if (stopped) return;
        if (!resolved) {
          resolved = true;
          completeHeaders(new Response(message, { status }));
        } else controller?.error(new Error(message));
        stop();
      };
      const abort = () => fail("Secure media cancelled", 499);
      // The timer covers pending work only. Paused playback holds bounded credit
      // and may remain paused indefinitely without losing its connection.
      const arm = () => {
        clearTimeout(timer);
        timer = setTimeout(
          () => fail("Media connection timed out", 504),
          20000,
        );
      };
      event.request.signal.addEventListener("abort", abort, { once: true });
      return new Promise((resolve) => {
        completeHeaders = resolve;
        port1.onmessage = ({ data }) => {
          if (stopped) return;
          if (data?.type === "headers" && !resolved) {
            resolved = true;
            clearTimeout(timer);
            const body =
              event.request.method === "HEAD" ||
              [204, 205, 304].includes(data.status)
                ? null
                : new ReadableStream(
                    {
                      start(value) {
                        controller = value;
                      },
                      pull() {
                        return new Promise((done) => {
                          completePull = done;
                          arm();
                          port1.postMessage({ type: "pull" });
                        });
                      },
                      cancel: stop,
                    },
                    { highWaterMark: 0 },
                  );
            try {
              resolve(
                new Response(body, {
                  status: data.status,
                  headers: data.headers,
                }),
              );
            } catch {
              resolve(new Response("Invalid media response", { status: 502 }));
              stop();
            }
            if (!body) stop();
          } else if (
            data?.type === "data" &&
            resolved &&
            completePull &&
            data.bytes instanceof ArrayBuffer &&
            data.bytes.byteLength > 0 &&
            data.bytes.byteLength <= 16384
          ) {
            clearTimeout(timer);
            controller.enqueue(new Uint8Array(data.bytes));
            completePull();
            completePull = undefined;
          } else if (data?.type === "end" && resolved) {
            controller?.close();
            stop();
          } else fail("Secure media interrupted");
        };
        if (event.request.signal.aborted) {
          abort();
          return;
        }
        arm();
        client.postMessage(
          {
            type: "mold-secure-media-v3",
            capability,
            method: event.request.method,
            range: event.request.headers.get("range"),
            ifRange: event.request.headers.get("if-range"),
            ifNoneMatch: event.request.headers.get("if-none-match"),
            ifModifiedSince: event.request.headers.get("if-modified-since"),
          },
          [port2],
        );
      });
    })().catch(() => {
      finish();
      return new Response("Secure media unavailable", { status: 503 });
    }),
  );
});
