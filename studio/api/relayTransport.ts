import { sha256 } from "@noble/hashes/sha2.js";
/** The Lambda facade preserves HTTP; only bulk bodies and object responses need adaptation. */
const TransportURL = URL;
const threshold = 2 * 1024 * 1024;
interface RelayInfo {
  protocol: number;
  upload_threshold: number;
  max_body_bytes: number;
  object_origin?: string;
}
const infoCache = new Map<string, Promise<RelayInfo | null>>();
export function clearRelayInfoCache(): void {
  infoCache.clear();
}
function s3Identity(url: URL): string | null {
  if (
    url.protocol !== "https:" ||
    (url.port && url.port !== "443") ||
    url.username ||
    url.password ||
    url.href.includes("#")
  )
    return null;
  const match =
    /^mold-relay-(\d{12})-([a-z]{2}(?:-gov)?-[a-z]+-\d)\.s3\.(?:dualstack\.)?([a-z]{2}(?:-gov)?-[a-z]+-\d)\.amazonaws\.com$/.exec(
      url.hostname,
    );
  return match && match[2] === match[3] ? `${match[1]}:${match[2]}` : null;
}
function validatedObjectOrigin(value: string): URL {
  const origin = new TransportURL(value);
  if (!s3Identity(origin) || origin.pathname !== "/" || origin.search)
    throw new Error("The relay returned an unsafe object origin.");
  return origin;
}
export function validateRelayObjectUrl(
  value: string,
  origin: string,
  objectOrigin?: string,
): string {
  const url = new TransportURL(value);
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.href.includes("#") ||
    !url.pathname.startsWith("/_mold/objects/")
  )
    throw new Error("The relay returned an unsafe object URL.");
  if (url.origin !== origin) {
    if (
      !objectOrigin ||
      !s3Identity(url) ||
      s3Identity(url) !== s3Identity(validatedObjectOrigin(objectOrigin))
    )
      throw new Error("The relay returned an unsafe object URL.");
    const one = (key: string) =>
      url.searchParams.getAll(key).length === 1
        ? url.searchParams.get(key)
        : null;
    const expires = one("X-Amz-Expires");
    if (
      one("X-Amz-Algorithm") !== "AWS4-HMAC-SHA256" ||
      !/^[a-fA-F0-9]{64}$/.test(one("X-Amz-Signature") ?? "") ||
      !/^\d+$/.test(expires ?? "") ||
      Number(expires) < 1 ||
      Number(expires) > 900
    )
      throw new Error("The relay returned an unsigned or invalid object URL.");
  }
  return url.href;
}
export async function resolveRelayObjectUrl(
  value: string,
  origin: string,
  signal: AbortSignal | null = null,
): Promise<string> {
  if (new TransportURL(value).origin === origin)
    return validateRelayObjectUrl(value, origin);
  if (!s3Identity(new TransportURL(value)))
    throw new Error("The relay returned an unsafe object URL.");
  const info = await relayInfo(origin, signal);
  return validateRelayObjectUrl(value, origin, info?.object_origin);
}
async function digest(bytes: ArrayBuffer): Promise<string> {
  return Array.from(
    globalThis.crypto?.subtle
      ? new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))
      : sha256(new Uint8Array(bytes)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
}
function requestHeaders(headers: Headers): Record<string, string> {
  const excluded = new Set([
    "authorization",
    "x-api-key",
    "cookie",
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-amz-content-sha256",
    "x-mold-request-target",
    ...(headers.get("connection") ?? "")
      .split(",")
      .map((name) => name.trim().toLowerCase()),
  ]);
  return Object.fromEntries(
    [...headers.entries()].filter(
      ([name]) =>
        !excluded.has(name) &&
        !name.startsWith("x-mold-viewer-") &&
        !name.startsWith("x-mold-relay-"),
    ),
  );
}
async function boundedDiscoveryText(response: Response): Promise<string> {
  const limit = 65536;
  if (Number(response.headers.get("content-length")) > limit) {
    await response.body?.cancel();
    throw new Error("Relay discovery response exceeds 64 KiB.");
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const bytes = new Uint8Array(limit);
  let size = 0;
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      if (size + part.value.byteLength > limit) {
        await reader.cancel();
        throw new Error("Relay discovery response exceeds 64 KiB.");
      }
      bytes.set(part.value, size);
      size += part.value.byteLength;
    }
    return new TextDecoder().decode(bytes.subarray(0, size));
  } finally {
    reader.releaseLock();
  }
}
async function relayInfo(
  origin: string,
  signal: AbortSignal | null = null,
  originalHeaders?: Headers,
): Promise<RelayInfo | null> {
  let pending = infoCache.get(origin);
  if (!pending) {
    const credentials = new Headers();
    for (const name of ["x-api-key", "authorization"]) {
      const value = originalHeaders?.get(name);
      if (value) credentials.set(name, value);
    }
    pending = globalThis
      .fetch(`${origin}/_mold/relay/info`, {
        headers: credentials,
        redirect: "error",
        credentials: "omit",
        signal: AbortSignal.timeout(10_000),
      })
      .then(async (response) => {
        if (response.status === 404) {
          await response.body?.cancel();
          return null;
        }
        if (!response.ok) {
          await response.body?.cancel();
          throw new Error(`Relay discovery failed: ${response.status}`);
        }
        const text = await boundedDiscoveryText(response);
        if (
          response.status === 200 &&
          response.headers
            .get("content-type")
            ?.split(";")[0]
            ?.trim()
            .toLowerCase() === "text/html" &&
          !response.headers.has("x-mold-relay-protocol") &&
          ((text.includes("<title>mold — studio</title>") &&
            text.includes('<div id="app"></div>')) ||
            (text.includes("<title>mold</title>") &&
              text.includes("<h1>mold is running</h1>") &&
              text.includes(
                "This binary was built without the web gallery UI bundled.",
              )))
        )
          return null;
        const info = JSON.parse(text) as RelayInfo;
        if (
          info.protocol !== 2 ||
          !Number.isSafeInteger(info.upload_threshold) ||
          info.upload_threshold <= 0 ||
          !Number.isSafeInteger(info.max_body_bytes) ||
          info.max_body_bytes > 67108864 ||
          info.max_body_bytes < info.upload_threshold
        )
          throw new Error("The relay returned invalid upload limits.");
        if (info.object_origin !== undefined)
          validatedObjectOrigin(info.object_origin);
        return info;
      });
    infoCache.set(origin, pending);
    void pending.catch(() => {
      if (infoCache.get(origin) === pending) infoCache.delete(origin);
    });
  }
  if (!signal) return pending;
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    pending!
      .then(resolve, reject)
      .finally(() => signal.removeEventListener("abort", abort));
  });
}
async function control(
  origin: string,
  path: string,
  body: unknown,
  headers: Headers,
  signal: AbortSignal | null = null,
): Promise<Response> {
  const serialized = JSON.stringify(body);
  const controlHeaders = new Headers({ "content-type": "application/json" });
  const key = headers.get("x-api-key");
  if (key) controlHeaders.set("x-api-key", key);
  controlHeaders.set(
    "x-amz-content-sha256",
    await digest(new TextEncoder().encode(serialized).buffer),
  );
  return globalThis.fetch(`${origin}${path}`, {
    method: "POST",
    headers: controlHeaders,
    body: serialized,
    redirect: "error",
    signal,
  });
}
/** Explicit adapter, never a global fetch patch. Credentials never follow redirects or object URLs. */
export async function relayFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const url = new TransportURL(
    input instanceof Request ? input.url : String(input),
    typeof window === "undefined" ? undefined : window.location.origin,
  );
  const headers = new Headers(
    init?.headers ?? (input instanceof Request ? input.headers : undefined),
  );
  const method = (
    init?.method ?? (input instanceof Request ? input.method : "GET")
  ).toUpperCase();
  const signal =
    init?.signal ?? (input instanceof Request ? input.signal : null);
  let response: Response;
  const mutation = ["POST", "PUT", "PATCH", "DELETE"].includes(method);
  const apiRequest =
    url.protocol === "https:" && url.pathname.startsWith("/api/");
  if (apiRequest)
    headers.set("x-mold-request-target", url.pathname + url.search);
  if (
    url.protocol === "https:" &&
    (url.pathname.startsWith("/api/") ||
      url.pathname.startsWith("/_mold/relay/")) &&
    mutation
  ) {
    const request = new Request(
      input instanceof Request ? input.clone() : url.href,
      {
        ...init,
        method,
        headers,
      } as RequestInit,
    );
    const bytes = await request.arrayBuffer();
    const hash = await digest(bytes);
    headers.set("x-amz-content-sha256", hash);
    if (!headers.has("content-type") && request.headers.has("content-type"))
      headers.set("content-type", request.headers.get("content-type")!);
    const info =
      bytes.byteLength > threshold
        ? await relayInfo(url.origin, signal, headers)
        : null;
    if (info && bytes.byteLength > info.upload_threshold) {
      if (bytes.byteLength > info.max_body_bytes)
        throw new Error("This relay accepts request bodies up to 64 MiB.");
      const grantResponse = await control(
        url.origin,
        "/_mold/relay/uploads",
        {
          method,
          path: url.pathname + url.search,
          headers: requestHeaders(headers),
          size: bytes.byteLength,
          sha256: hash,
        },
        headers,
        signal,
      );
      if (!grantResponse.ok) return grantResponse;
      const grant = (await grantResponse.json()) as {
        id: string;
        url: string;
        headers: Record<string, string>;
        expires_at: number;
      };
      const upload = new TransportURL(grant.url);
      const uploadHeaders = new Headers(grant.headers);
      if (
        !grant.id ||
        upload.protocol !== "https:" ||
        upload.username ||
        upload.password ||
        !s3Identity(upload) ||
        !info.object_origin ||
        s3Identity(upload) !==
          s3Identity(validatedObjectOrigin(info.object_origin)) ||
        uploadHeaders.has("x-api-key") ||
        uploadHeaders.has("authorization") ||
        [...uploadHeaders.keys()].some((name) => name.startsWith("x-mold-")) ||
        !Number.isSafeInteger(grant.expires_at) ||
        grant.expires_at <= Date.now() / 1000
      )
        throw new Error("The relay returned an unsafe upload grant.");
      const uploaded = await globalThis.fetch(upload.href, {
        method: "PUT",
        headers: uploadHeaders,
        body: bytes,
        redirect: "error",
        credentials: "omit",
        signal,
      });
      if (!uploaded.ok)
        throw new Error(`Relay upload failed: ${uploaded.status}`);
      response = await control(
        url.origin,
        "/_mold/relay/request",
        { id: grant.id },
        headers,
        signal,
      );
    } else
      response = await globalThis.fetch(input, {
        ...init,
        method,
        headers,
        body: init?.body ?? (input instanceof Request ? bytes : null),
        ...(headers.has("x-api-key") ? { redirect: "error" as const } : {}),
      });
  } else {
    const outgoingInit = apiRequest
      ? {
          ...init,
          headers,
          ...(headers.has("x-api-key") ? { redirect: "error" as const } : {}),
        }
      : headers.has("x-api-key")
        ? { ...init, redirect: "error" }
        : init;
    response = outgoingInit
      ? await globalThis.fetch(input, outgoingInit as RequestInit)
      : await globalThis.fetch(input);
  }
  if (response.headers?.get("x-mold-relay-object") !== "1") return response;
  const object = (await response.json()) as {
    url: string;
    status: number;
    headers: Record<string, string>;
  };
  const location = await resolveRelayObjectUrl(object.url, url.origin, signal);
  if (
    !Number.isInteger(object.status) ||
    object.status < 200 ||
    object.status > 599
  )
    throw new Error("The relay returned an invalid object response.");
  const downloaded = await globalThis.fetch(location, {
    redirect: "error",
    credentials: "omit",
    method: method === "HEAD" ? "HEAD" : "GET",
    signal,
  });
  if (!downloaded.ok)
    throw new Error(`Relay object download failed: ${downloaded.status}`);
  return new Response(
    method === "HEAD" || [204, 205, 304].includes(object.status)
      ? null
      : downloaded.body,
    { status: object.status, headers: object.headers },
  );
}
