import { hmac } from "@noble/hashes/hmac.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

export interface ConnectionEndpoint {
  url: string;
  kind: "lan" | "tailscale" | "relay";
}
export interface ConnectionAddresses {
  version: 1;
  instance_id: string;
  endpoints: ConnectionEndpoint[];
}
const OriginURL = URL;
const pairedCredential = (secret: string): boolean =>
  /^mold_pair_[A-Za-z0-9_-]{43}$/.test(secret);
export function parseConnectionEndpoints(value: unknown): ConnectionEndpoint[] {
  if (!Array.isArray(value) || value.length > 8)
    throw new Error("Invalid connection addresses.");
  const result: ConnectionEndpoint[] = [];
  for (const item of value) {
    if (
      !item ||
      typeof item.url !== "string" ||
      item.url.length > 2048 ||
      /[?#]/.test(item.url) ||
      !["lan", "tailscale", "relay"].includes(item.kind)
    )
      throw new Error("Invalid connection address.");
    const url = new OriginURL(item.url);
    const hostname = url.hostname
      .toLowerCase()
      .replace(/^\[|\]$/g, "")
      .replace(/\.$/, "");
    const forbidden =
      hostname === "localhost" ||
      hostname.endsWith(".localhost") ||
      hostname === "::" ||
      hostname === "::1" ||
      hostname === "0.0.0.0" ||
      hostname.startsWith("127.") ||
      hostname.startsWith("169.254.") ||
      /^fe[89ab][0-9a-f]:/.test(hostname) ||
      hostname.includes("%");
    if (
      forbidden ||
      (url.port !== "" && (Number(url.port) < 1 || Number(url.port) > 65535)) ||
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      url.pathname !== "/" ||
      (item.kind === "relay" && url.protocol !== "https:")
    )
      throw new Error("Connection addresses must be plain HTTP(S) origins.");
    if (!result.some((e) => e.url === url.origin))
      result.push({ url: url.origin, kind: item.kind });
  }
  return result;
}
async function boundedJSON(response: Response): Promise<unknown> {
  if (!response.ok || Number(response.headers.get("content-length")) > 4096)
    throw new Error("Connection proof refused.");
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Missing connection proof.");
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > 4096) throw new Error("Oversized connection proof.");
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}
/** Prove possession without forwarding the API key or one-use pairing bearer. */
export async function selectConnectionRoute(options: {
  endpoints: readonly ConnectionEndpoint[];
  expectedInstanceId: string;
  secret: string;
  kind: "api" | "pairing";
  secureContext?: boolean;
  signal?: AbortSignal | undefined;
}): Promise<string> {
  if (options.kind === "api" && !pairedCredential(options.secret))
    throw new Error("Automatic roaming requires a paired credential.");
  if (!options.expectedInstanceId || !options.secret)
    throw new Error("Connection proof requires an identity and credential.");
  const secure =
    options.secureContext ??
    (typeof location !== "undefined" && location.protocol === "https:");
  const endpoints = parseConnectionEndpoints(options.endpoints).filter(
    (e) => !secure || e.url.startsWith("https:"),
  );
  const key = sha256(new TextEncoder().encode(options.secret));
  const results = await Promise.all(
    endpoints.map(async (endpoint) => {
      const nonce = bytesToHex(crypto.getRandomValues(new Uint8Array(32)));
      const timeout = AbortSignal.timeout(2500);
      const signal = options.signal
        ? AbortSignal.any([timeout, options.signal])
        : timeout;
      try {
        const response = await globalThis.fetch(
          `${endpoint.url}/api/connection-probe`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              kind: options.kind,
              key_tag: bytesToHex(key).slice(0, 16),
              nonce,
            }),
            credentials: "omit",
            redirect: "error",
            signal,
          },
        );
        const value = (await boundedJSON(response)) as {
          instance_id?: unknown;
          proof?: unknown;
        };
        if (
          value.instance_id !== options.expectedInstanceId ||
          typeof value.proof !== "string" ||
          !/^[a-f0-9]{64}$/.test(value.proof)
        )
          return null;
        const expected = bytesToHex(
          hmac(
            sha256,
            key,
            new TextEncoder().encode(
              `mold-connection-proof-v1\n${options.kind}\n${nonce}\n${options.expectedInstanceId}`,
            ),
          ),
        );
        // Equal-length values; compare every byte rather than an early prefix.
        let difference = 0;
        for (let i = 0; i < 64; i++)
          difference |= expected.charCodeAt(i) ^ value.proof.charCodeAt(i);
        return difference === 0 ? endpoint : null;
      } catch {
        return null;
      }
    }),
  );
  options.signal?.throwIfAborted();
  const priority = { lan: 0, tailscale: 1, relay: 2 };
  const winner = results
    .filter((e): e is ConnectionEndpoint => e !== null)
    .sort((a, b) => priority[a.kind] - priority[b.kind])[0];
  if (!winner)
    throw new Error("No approved address could prove this Mold machine.");
  return winner.url;
}

const ROUTES_KEY = "mold.connection-routes.v1";
interface RouteRecord {
  instanceId: string;
  keyTag: string;
  endpoints: ConnectionEndpoint[];
  checkedAt?: number;
  learnedAt?: number;
  activeURL?: string;
  originalURL?: string;
  originalProof?: string;
}
const records = new Map<string, RouteRecord>();
const revisions = new Map<string, number>();
const learning = new Set<string>();
function secretTag(secret: string): string {
  return bytesToHex(sha256(new TextEncoder().encode(secret))).slice(0, 16);
}
export function forgetConnectionRoutes(hostId: string): void {
  revisions.set(hostId, (revisions.get(hostId) ?? 0) + 1);
  records.delete(hostId);
  try {
    const data = JSON.parse(localStorage.getItem(ROUTES_KEY) ?? "{}");
    delete data[hostId];
    localStorage.setItem(
      ROUTES_KEY,
      JSON.stringify(data, (key, value) =>
        key === "keyTag" ? undefined : value,
      ),
    );
  } catch {
    /* optional persistence */
  }
}
function originalProof(
  hostId: string,
  instanceId: string,
  secret: string,
  url: string,
): string {
  return bytesToHex(
    hmac(
      sha256,
      sha256(new TextEncoder().encode(secret)),
      new TextEncoder().encode(
        `mold-connection-original-v1\n${hostId}\n${instanceId}\n${url}`,
      ),
    ),
  );
}
function saveRecord(hostId: string, record: RouteRecord): void {
  try {
    const data = JSON.parse(localStorage.getItem(ROUTES_KEY) ?? "{}");
    data[hostId] = {
      instanceId: record.instanceId,
      endpoints: record.endpoints,
      ...(record.originalURL
        ? {
            originalURL: record.originalURL,
            originalProof: record.originalProof,
          }
        : {}),
    };
    localStorage.setItem(
      ROUTES_KEY,
      JSON.stringify(data, (key, value) =>
        key === "keyTag" ? undefined : value,
      ),
    );
  } catch {
    /* optional persistence */
  }
}
function approveOriginal(
  hostId: string,
  record: RouteRecord,
  secret: string,
  url: string,
): void {
  const parsed = new OriginURL(url);
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash ||
    parsed.pathname !== "/"
  )
    throw new Error("Invalid original connection address.");
  record.originalURL = parsed.origin;
  record.originalProof = originalProof(
    hostId,
    record.instanceId,
    secret,
    parsed.origin,
  );
  saveRecord(hostId, record);
}
export function rememberConnectionRoutes(
  hostId: string,
  instanceId: string,
  secret: string,
  endpoints: unknown,
  originalURL?: string,
): void {
  if (!pairedCredential(secret)) {
    forgetConnectionRoutes(hostId);
    return;
  }
  const record: RouteRecord = {
    instanceId,
    keyTag: secretTag(secret),
    endpoints: parseConnectionEndpoints(endpoints),
    learnedAt: Date.now(),
  };
  const previous = records.get(hostId);
  const original =
    originalURL ??
    (previous?.instanceId === instanceId && previous.keyTag === record.keyTag
      ? previous.originalURL
      : undefined);
  if (original) approveOriginal(hostId, record, secret, original);
  records.set(hostId, record);
  saveRecord(hostId, record);
}
function knownRoutes(
  hostId: string,
  instanceId: string,
  secret: string,
): RouteRecord | null {
  let record = records.get(hostId);
  let invalidOriginal = false;
  if (!record) {
    try {
      const raw = JSON.parse(localStorage.getItem(ROUTES_KEY) ?? "{}")[hostId];
      if (raw) {
        invalidOriginal =
          raw.originalURL !== undefined &&
          (typeof raw.originalURL !== "string" ||
            raw.originalProof !==
              originalProof(hostId, raw.instanceId, secret, raw.originalURL));
        record = {
          instanceId: raw.instanceId,
          // A cached catalog is only a candidate list; prove with the current key before using alternatives.
          keyTag: secretTag(secret),
          endpoints: parseConnectionEndpoints(raw.endpoints),
          ...(typeof raw.originalURL === "string" &&
          raw.originalProof ===
            originalProof(hostId, raw.instanceId, secret, raw.originalURL)
            ? { originalURL: raw.originalURL, originalProof: raw.originalProof }
            : {}),
        };
        if ("keyTag" in raw) saveRecord(hostId, record);
      }
    } catch {
      /* ignore corrupt optional cache */
    }
  }
  if (invalidOriginal)
    throw new Error(
      "The saved original connection address could not be verified. Remove and re-add this machine to approve its address again.",
    );
  if (!record) return null;
  if (record.instanceId !== instanceId || record.keyTag !== secretTag(secret)) {
    forgetConnectionRoutes(hostId);
    return null;
  }
  records.set(hostId, record);
  return record;
}
/** Only health reads roam. A selected route never causes a mutation replay. */
export async function connectionHealth<
  T extends { instance_id?: string | null | undefined },
>(options: {
  hostId: string;
  baseUrl: string;
  apiKey: string | null | undefined;
  instanceId?: string | null | undefined;
  read: (baseUrl: string) => Promise<T>;
  signal?: AbortSignal | undefined;
  secureContext?: boolean;
  isCurrent?: () => boolean;
}): Promise<{ value: T; baseUrl: string }> {
  let revision = revisions.get(options.hostId) ?? 0;
  const assertCurrent = () => {
    options.signal?.throwIfAborted();
    if (
      (revisions.get(options.hostId) ?? 0) !== revision ||
      (options.isCurrent && !options.isCurrent())
    )
      throw new DOMException("Host connection changed.", "AbortError");
  };
  assertCurrent();
  const secret =
    options.apiKey && pairedCredential(options.apiKey) ? options.apiKey : null;
  if (!secret) forgetConnectionRoutes(options.hostId);
  let url = options.baseUrl;
  const record =
    secret && options.instanceId
      ? knownRoutes(options.hostId, options.instanceId, secret)
      : null;
  revision = revisions.get(options.hostId) ?? 0;
  if (record && !record.originalURL)
    approveOriginal(options.hostId, record, secret!, options.baseUrl);
  const approvedOriginal = record?.originalURL ?? options.baseUrl;
  const age =
    record?.checkedAt === undefined ? Infinity : Date.now() - record.checkedAt;
  const cachedHealthy = !!record?.activeURL && age >= 0 && age <= 30_000;
  if (
    record &&
    !cachedHealthy &&
    (record.endpoints.length > 1 || url !== approvedOriginal)
  ) {
    try {
      url = await selectConnectionRoute({
        endpoints: record.endpoints,
        expectedInstanceId: record.instanceId,
        secret: secret!,
        kind: "api",
        ...(options.signal ? { signal: options.signal } : {}),
        ...(options.secureContext !== undefined
          ? { secureContext: options.secureContext }
          : {}),
      });
      record.checkedAt = Date.now();
      record.activeURL = url;
    } catch (error) {
      assertCurrent();
      if (url !== approvedOriginal) {
        const secure =
          options.secureContext ??
          (typeof location !== "undefined" && location.protocol === "https:");
        if (secure && !approvedOriginal.startsWith("https:")) throw error;
        url = approvedOriginal;
        record.learnedAt = 0;
        record.activeURL = url;
        record.checkedAt = Date.now();
      }
    }
  } else if (cachedHealthy && record?.activeURL) url = record.activeURL;
  assertCurrent();
  let value: T;
  try {
    value = await options.read(url);
  } catch (error) {
    const status = (error as { status?: number })?.status;
    // Revocation/refusal is authoritative, not transport reachability.
    if (
      !record ||
      !secret ||
      status === 401 ||
      status === 403 ||
      options.signal?.aborted
    )
      throw error;
    let winner: string;
    try {
      winner = await selectConnectionRoute({
        endpoints: record.endpoints,
        expectedInstanceId: record.instanceId,
        secret,
        kind: "api",
        signal: options.signal,
        ...(options.secureContext !== undefined
          ? { secureContext: options.secureContext }
          : {}),
      });
    } catch {
      assertCurrent();
      const secure =
        options.secureContext ??
        (typeof location !== "undefined" && location.protocol === "https:");
      if (
        !record.originalURL ||
        record.originalURL === url ||
        (secure && !record.originalURL.startsWith("https:"))
      )
        throw error;
      winner = record.originalURL;
      record.learnedAt = 0;
    }
    assertCurrent();
    if (winner === url) throw error;
    url = winner;
    value = await options.read(url);
    record.checkedAt = Date.now();
    record.activeURL = url;
  }
  assertCurrent();
  if (
    options.instanceId &&
    value.instance_id &&
    value.instance_id !== options.instanceId
  ) {
    if (url !== options.baseUrl)
      throw new Error("This address now reaches a different Mold machine.");
    return { value, baseUrl: url };
  }
  const instanceId = value.instance_id;
  if (
    secret &&
    instanceId &&
    !learning.has(options.hostId) &&
    Date.now() - (record?.learnedAt ?? 0) > 60_000
  ) {
    learning.add(options.hostId);
    void (async () => {
      try {
        const response = await globalThis.fetch(
          `${url}/api/connection-addresses`,
          {
            headers: { "x-api-key": secret },
            redirect: "error",
            credentials: "omit",
            signal: options.signal
              ? AbortSignal.any([options.signal, AbortSignal.timeout(2500)])
              : AbortSignal.timeout(2500),
          },
        );
        const addresses = (await boundedJSON(response)) as ConnectionAddresses;
        assertCurrent();
        if (addresses.version === 1 && addresses.instance_id === instanceId) {
          rememberConnectionRoutes(
            options.hostId,
            instanceId,
            secret,
            addresses.endpoints,
            approvedOriginal,
          );
          const learned = records.get(options.hostId)!;
          learned.activeURL = url;
          learned.checkedAt = Date.now();
        }
      } catch {
        /* Additive discovery never invalidates authenticated status. */
      } finally {
        learning.delete(options.hostId);
      }
    })();
  }
  return { value, baseUrl: url };
}
