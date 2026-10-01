/** The Lambda facade preserves HTTP; only bulk bodies and object responses need adaptation. */
const TransportURL = URL;
const threshold = 2 * 1024 * 1024;
interface RelayInfo {
  protocol: number;
  upload_threshold: number;
  max_body_bytes: number;
}
const infoCache = new Map<string, Promise<RelayInfo | null>>();
export function clearRelayInfoCache(): void {
  infoCache.clear();
}
export function validateRelayObjectUrl(value: string, origin: string): string {
  const url = new TransportURL(value);
  if (
    url.protocol !== "https:" ||
    url.origin !== origin ||
    url.username ||
    url.password ||
    !url.pathname.startsWith("/_mold/objects/")
  )
    throw new Error("The relay returned an unsafe object URL.");
  return url.href;
}
async function digest(bytes: ArrayBuffer): Promise<string> {
  return Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
}
function requestHeaders(headers: Headers): Record<string, string> {
  return Object.fromEntries(headers.entries());
}
async function relayInfo(
  origin: string,
  signal: AbortSignal | null = null,
): Promise<RelayInfo | null> {
  let pending = infoCache.get(origin);
  if (!pending) {
    pending = globalThis
      .fetch(`${origin}/_mold/relay/info`, {
        redirect: "error",
        credentials: "omit",
        signal,
      })
      .then(async (response) => {
        if (response.status === 404) return null;
        if (!response.ok)
          throw new Error(`Relay discovery failed: ${response.status}`);
        const info = (await response.json()) as RelayInfo;
        if (
          info.protocol !== 2 ||
          !Number.isSafeInteger(info.upload_threshold) ||
          info.upload_threshold <= 0 ||
          !Number.isSafeInteger(info.max_body_bytes) ||
          info.max_body_bytes > 67108864 ||
          info.max_body_bytes < info.upload_threshold
        )
          throw new Error("The relay returned invalid upload limits.");
        return info;
      });
    infoCache.set(origin, pending);
    void pending.catch(() => {
      if (infoCache.get(origin) === pending) infoCache.delete(origin);
    });
  }
  return pending;
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
  const mutation = ["POST", "PUT", "PATCH"].includes(method);
  if (
    url.protocol === "https:" &&
    url.pathname.startsWith("/api/") &&
    mutation
  ) {
    const request = new Request(url.href, {
      ...(input instanceof Request
        ? {
            method: input.method,
            headers: input.headers,
            body: input.body,
            duplex: "half",
          }
        : {}),
      ...init,
      method,
      headers,
    } as RequestInit);
    const bytes = await request.arrayBuffer();
    const hash = await digest(bytes);
    headers.set("x-amz-content-sha256", hash);
    if (!headers.has("content-type") && request.headers.has("content-type"))
      headers.set("content-type", request.headers.get("content-type")!);
    const info =
      bytes.byteLength > threshold ? await relayInfo(url.origin, signal) : null;
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
        !/^[a-z0-9.-]+\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$/.test(
          upload.hostname,
        ) ||
        uploadHeaders.has("x-api-key") ||
        uploadHeaders.has("authorization") ||
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
        body: bytes,
        ...(headers.has("x-api-key") ? { redirect: "error" as const } : {}),
      });
  } else
    response = await globalThis.fetch(
      input,
      headers.has("x-api-key") ? { ...init, redirect: "error" } : init,
    );
  if (response.headers?.get("x-mold-relay-object") !== "1") return response;
  const object = (await response.json()) as {
    url: string;
    status: number;
    headers: Record<string, string>;
  };
  const location = validateRelayObjectUrl(object.url, url.origin);
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
