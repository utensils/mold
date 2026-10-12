import {
  CONNECTION_MODE_HEADER,
  directOrigin,
  type ConnectionMode,
} from "./connectionMode";
import { relayFetch } from "./relayTransport";
import { secureHostForOrigin, updateSecureHostRoutes } from "./secureStore";

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
  if (!response.ok)
    throw Object.assign(new Error(`The machine returned ${response.status}.`), {
      status: response.status,
    });
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The machine returned no response.");
  const bytes = new Uint8Array(65536);
  let count = 0;
  try {
    for (;;) {
      const part = await reader.read();
      if (part.done) break;
      if (count + part.value.length > bytes.length)
        throw new Error("The machine returned an oversized response.");
      bytes.set(part.value, count);
      count += part.value.length;
    }
    return JSON.parse(new TextDecoder().decode(bytes.subarray(0, count)));
  } finally {
    await reader.cancel();
  }
}

/** Candidates come from a pinned host. Every probe authenticates Noise before HTTP. */
export async function selectConnectionRoute(options: {
  endpoints: readonly ConnectionEndpoint[];
  expectedInstanceId: string;
  signal?: AbortSignal | undefined;
}): Promise<string> {
  if (!options.expectedInstanceId)
    throw new Error("A saved machine identity is required.");
  options.signal?.throwIfAborted();
  // relayFetch sends authenticated Noise over WebSocket, not plaintext HTTP.
  // The browser itself enforces local-network permission/mixed-content rules.
  const endpoints = parseConnectionEndpoints(options.endpoints);
  const priority = { lan: 0, tailscale: 1, relay: 2 };
  const first = Math.min(...endpoints.map((e) => priority[e.kind]));
  const controller = new AbortController();
  return new Promise((resolve, reject) => {
    let settled = false;
    let winner: ConnectionEndpoint | undefined;
    const pending = new Set(endpoints);
    const timers: ReturnType<typeof setTimeout>[] = [];
    const finish = (error?: unknown) => {
      if (settled) return;
      settled = true;
      timers.forEach(clearTimeout);
      controller.abort();
      options.signal?.removeEventListener("abort", abort);
      if (error) reject(error);
      else resolve(winner!.url);
    };
    const abort = () =>
      finish(
        options.signal?.reason ?? new DOMException("Cancelled", "AbortError"),
      );
    options.signal?.addEventListener("abort", abort, { once: true });
    const consider = () => {
      if (
        winner &&
        ![...pending].some((e) => priority[e.kind] < priority[winner!.kind])
      )
        finish();
      else if (!pending.size)
        finish(
          new Error(
            "This machine is not reachable at any of its secure addresses.",
          ),
        );
    };
    for (const endpoint of endpoints) {
      const run = async () => {
        if (settled) return;
        let valid = false;
        try {
          const profile = await secureHostForOrigin(endpoint.url);
          if (!profile || profile.instance_id !== options.expectedInstanceId)
            throw new Error("Unknown machine identity.");
          const signal = AbortSignal.any([
            controller.signal,
            AbortSignal.timeout(2500),
          ]);
          const value = (await boundedJSON(
            await relayFetch(`${endpoint.url}/api/status`, {
              headers: { [CONNECTION_MODE_HEADER]: "secure" },
              signal,
            }),
          )) as { instance_id?: string };
          signal.throwIfAborted();
          valid = value.instance_id === options.expectedInstanceId;
        } catch {
          /* A candidate cannot answer or cannot prove the pinned identity. */
        }
        if (settled) return;
        pending.delete(endpoint);
        if (
          valid &&
          (!winner || priority[endpoint.kind] < priority[winner.kind])
        )
          winner = endpoint;
        consider();
      };
      const delay = (priority[endpoint.kind] - first) * 150;
      if (!delay) void run();
      else
        timers.push(
          setTimeout(() => {
            void run();
          }, delay),
        );
    }
    consider();
  });
}

const revisions = new Map<string, number>();
const selected = new Map<
  string,
  { namespace: string; url: string; checkedAt: number }
>();
const learning = new Set<string>();
const learnedAt = new Map<string, number>();
export function forgetConnectionRoutes(hostId: string): void {
  revisions.set(hostId, (revisions.get(hostId) ?? 0) + 1);
  selected.delete(hostId);
}
/** Only health reads roam; failed mutations are never dispatched a second time. */
export async function connectionHealth<
  T extends { instance_id?: string | null | undefined },
>(options: {
  hostId: string;
  baseUrl: string;
  apiKey: string | null | undefined;
  connectionMode?: ConnectionMode | undefined;
  instanceId?: string | null | undefined;
  read: (baseUrl: string) => Promise<T>;
  signal?: AbortSignal | undefined;
  isCurrent?: () => boolean;
}): Promise<{ value: T; baseUrl: string }> {
  const revision = revisions.get(options.hostId) ?? 0;
  const assertCurrent = () => {
    options.signal?.throwIfAborted();
    if (
      (revisions.get(options.hostId) ?? 0) !== revision ||
      (options.isCurrent && !options.isCurrent())
    )
      throw new DOMException("Host connection changed.", "AbortError");
  };
  assertCurrent();
  if ((options.connectionMode ?? "secure") === "regular") {
    directOrigin(
      new URL(
        options.baseUrl || "/",
        typeof location !== "undefined" ? location.href : undefined,
      ).origin,
    );
    const value = await options.read(options.baseUrl);
    assertCurrent();
    return { value, baseUrl: options.baseUrl };
  }
  const profile = await secureHostForOrigin(
    new URL(
      options.baseUrl || "/",
      typeof location !== "undefined" ? location.href : undefined,
    ).origin,
  );
  assertCurrent();
  if (!profile)
    return {
      value: await options.read(options.baseUrl),
      baseUrl: options.baseUrl,
    };
  if (options.instanceId && options.instanceId !== profile.instance_id)
    throw new Error("Pair this machine again: its saved identity changed.");
  const expectedInstanceId = profile.instance_id;
  let url = options.baseUrl;
  const cached = selected.get(options.hostId);
  if (
    cached?.namespace === profile.namespace &&
    Date.now() - cached.checkedAt < 30000
  )
    url = cached.url;
  else if (profile.endpoints.length) {
    try {
      url = await selectConnectionRoute({
        endpoints: profile.endpoints,
        expectedInstanceId,
        signal: options.signal,
      });
    } catch {
      assertCurrent(); /* The originally pinned route can still answer. */
    }
  }
  assertCurrent();
  let value: T;
  try {
    value = await options.read(url);
  } catch (error) {
    assertCurrent();
    if ([401, 403].includes((error as { status?: number }).status ?? 0))
      throw error;
    const winner = await selectConnectionRoute({
      endpoints: profile.endpoints,
      expectedInstanceId,
      signal: options.signal,
    });
    assertCurrent();
    if (winner === url) throw error;
    url = winner;
    value = await options.read(url);
  }
  assertCurrent();
  if (value.instance_id !== expectedInstanceId)
    throw new Error("This address now reaches a different Mold machine.");
  selected.set(options.hostId, {
    namespace: profile.namespace,
    url,
    checkedAt: Date.now(),
  });
  if (
    !learning.has(options.hostId) &&
    Date.now() - (learnedAt.get(options.hostId) ?? 0) > 60000
  ) {
    learning.add(options.hostId);
    const authority = profile.grant.signature;
    void (async () => {
      try {
        const addresses = (await boundedJSON(
          await relayFetch(`${url}/api/connection-addresses`, {
            headers: { [CONNECTION_MODE_HEADER]: "secure" },
            signal: AbortSignal.timeout(2500),
          }),
        )) as ConnectionAddresses;
        assertCurrent();
        if (
          addresses.version !== 1 ||
          addresses.instance_id !== expectedInstanceId
        )
          return;
        const endpoints = parseConnectionEndpoints(addresses.endpoints);
        // Only the encrypted, authenticated host can introduce address aliases.
        learnedAt.set(options.hostId, Date.now());
        await updateSecureHostRoutes(
          profile.namespace,
          authority,
          endpoints,
          [...new Set([options.baseUrl, ...endpoints.map((e) => e.url)])].slice(
            0,
            8,
          ),
        );
      } catch {
        /* Discovery cannot invalidate a successfully authenticated status. */
      } finally {
        learning.delete(options.hostId);
      }
    })();
  }
  return { value, baseUrl: url };
}
