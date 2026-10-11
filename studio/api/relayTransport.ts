import { SecureChannel } from "./secureChannel";
import {
  deviceIdentity,
  secureHostForOrigin,
  renewSecureHostGrant,
  assertSecureHostCurrent,
} from "./secureStore";
import {
  checkGrant,
  type SecureHostProfile,
  type SecureGrant,
} from "./secureCrypto";

type RequestAdapter = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;
interface ChannelEntry {
  promise: Promise<SecureChannel>;
  controller: AbortController;
  waiters: number;
  channel?: SecureChannel;
  authority: string;
}
const channels = new Map<string, ChannelEntry>();
let defaultOrigin: (() => string | null) | undefined;
let injectedAdapter: RequestAdapter | undefined;

/** Shell-independent injection for platform adapters and deterministic tests. */
export function configureSecureRequestAdapter(adapter?: RequestAdapter): void {
  injectedAdapter = adapter;
}
/** The independently hosted Studio maps its relative API calls to its selected paired host. */
export function configureSecureOrigin(provider: () => string | null): void {
  defaultOrigin = provider;
}
export function clearRelayInfoCache(): void {
  for (const entry of channels.values()) entry.controller.abort();
  channels.clear();
}
export function isLoopbackOrigin(url: URL): boolean {
  return url.hostname === "127.0.0.1" || url.hostname === "[::1]";
}
export function secureRequestUrl(input: RequestInfo | URL): URL {
  const url = new URL(
    input instanceof Request ? input.url : String(input),
    typeof location === "undefined" ? undefined : location.href,
  );
  if (
    defaultOrigin &&
    typeof location !== "undefined" &&
    url.origin === location.origin &&
    url.pathname.startsWith("/api/")
  ) {
    const target = defaultOrigin();
    if (!target) throw new Error("Pair a machine to use Mold Studio.");
    return new URL(url.pathname + url.search, target);
  }
  return url;
}
export function secureWebsocketFor(
  profile: SecureHostProfile,
  origin: string,
): string | undefined {
  const url = new URL(origin);
  if (
    profile.endpoints.find((endpoint) => endpoint.url === origin)?.kind !==
      "relay" &&
    url.protocol === "http:"
  ) {
    url.protocol = "ws:";
    url.pathname = "/_mold/secure";
    return url.href;
  }
  if (
    profile.endpoints.find((endpoint) => endpoint.url === origin)?.kind !==
      "relay" ||
    !profile.websocket_url
  ) {
    url.protocol = "wss:";
    url.pathname = "/_mold/secure";
    return url.href;
  }
  return undefined;
}
async function channelFor(
  profile: SecureHostProfile,
  origin: string,
  signal: AbortSignal,
): Promise<SecureChannel> {
  signal.throwIfAborted();
  assertSecureHostCurrent(profile);
  const key = `${profile.namespace}|${origin}`;
  // Renewal preserves authority; a replacement grant or host pin does not.
  const authority = JSON.stringify([
    profile.host_key,
    profile.host_noise_key,
    profile.host_binding_signature,
    profile.websocket_url,
    profile.grant.id,
    profile.grant.epoch,
    profile.apiKey,
  ]);
  let entry = channels.get(key);
  if (entry && entry.authority !== authority) {
    entry.controller.abort();
    channels.delete(key);
    entry = undefined;
  }
  if (entry?.channel?.closed) {
    channels.delete(key);
    entry = undefined;
  }
  if (!entry) {
    const controller = new AbortController();
    const created: ChannelEntry = {
      promise: Promise.resolve(null as unknown as SecureChannel),
      controller,
      waiters: 0,
      authority,
    };
    created.promise = (async () => {
      const identity = await deviceIdentity();
      try {
        controller.signal.throwIfAborted();
        assertSecureHostCurrent(profile);
        const websocket = secureWebsocketFor(profile, origin);
        const channel = await SecureChannel.connect({
          host: profile,
          identity,
          grant: profile.grant,
          signal: controller.signal,
          ...(websocket ? { websocket } : {}),
        });
        controller.signal.addEventListener("abort", () => channel.close(), {
          once: true,
        });
        if (controller.signal.aborted) {
          channel.close();
          controller.signal.throwIfAborted();
        }
        try {
          assertSecureHostCurrent(profile);
          if (profile.grant.expiresAt - Date.now() / 1000 < 7 * 86400) {
            const response = await channel.fetch(
              new Request(new URL("/api/pairing/secure/renew", origin), {
                method: "POST",
                headers: { "x-api-key": profile.apiKey },
                signal: controller.signal,
              }),
            );
            if (
              !response.ok ||
              Number(response.headers.get("content-length")) > 4096
            )
              throw new Error(
                "This pairing could not be renewed. Pair again from the host.",
              );
            const reader = response.body?.getReader();
            let text = "";
            if (!reader) throw new Error("Missing renewal response.");
            try {
              for (;;) {
                const part = await reader.read();
                if (part.done) break;
                text += new TextDecoder().decode(part.value);
                if (text.length > 4096)
                  throw new Error("Invalid renewal response.");
              }
            } finally {
              await reader.cancel();
            }
            const { grant } = JSON.parse(text) as { grant: SecureGrant };
            checkGrant(profile, identity, grant);
            if (
              grant.id !== profile.grant.id ||
              grant.epoch !== profile.grant.epoch
            )
              throw new Error("Pairing authority changed during renewal.");
            if (!(await renewSecureHostGrant(profile, grant)))
              throw new Error(
                "This pairing changed while reconnecting. Connect again.",
              );
            profile.grant = grant;
          }
          assertSecureHostCurrent(profile);
          return channel;
        } catch (error) {
          channel.close();
          throw error;
        }
      } finally {
        identity.close();
        identity.free();
      }
    })();
    channels.set(key, created);
    void created.promise.then(
      (channel) => {
        created.channel = channel;
        if (controller.signal.aborted) channel.close();
      },
      () => {
        if (channels.get(key) === created) channels.delete(key);
      },
    );
    entry = created;
  }
  const owner = entry;
  owner.waiters += 1;
  return new Promise((resolve, reject) => {
    let finished = false;
    const done = () => {
      if (finished) return false;
      finished = true;
      signal.removeEventListener("abort", abort);
      owner.waiters -= 1;
      if (!owner.waiters && !owner.channel) {
        owner.controller.abort();
        if (channels.get(key) === owner) channels.delete(key);
      }
      return true;
    };
    const abort = () => {
      if (done()) reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
    owner.promise.then(
      (channel) => {
        if (done()) resolve(channel);
      },
      (error) => {
        if (done()) reject(error);
      },
    );
    if (signal.aborted) abort();
  });
}

/** Every remote Mold request is encrypted. Loopback is the embedded engine boundary. */
export async function relayFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  if (injectedAdapter) return injectedAdapter(input, init);
  const url = secureRequestUrl(input);
  const api = url.pathname.startsWith("/api/") || url.pathname === "/health";
  if (!api) return globalThis.fetch(input, init);
  const profile = await secureHostForOrigin(url.origin);
  if (!profile && isLoopbackOrigin(url))
    return globalThis.fetch(
      input instanceof Request ? new Request(url, input) : url,
      init,
    );
  if (!profile)
    throw new Error(
      "This machine needs secure pairing. Create a new pairing code in its settings.",
    );
  const request = new Request(input instanceof Request ? input : url, init);
  const headers = new Headers(request.headers);
  headers.set("x-api-key", profile.apiKey);
  headers.delete("authorization");
  headers.delete("cookie");
  headers.delete("x-mold-secure-peer");
  const encrypted = new Request(url, {
    method: request.method,
    headers,
    body: request.body,
    signal: request.signal,
    ...(request.body ? { duplex: "half" } : {}),
  } as RequestInit);
  const channel = await channelFor(profile, url.origin, request.signal);
  assertSecureHostCurrent(profile);
  return channel.fetch(encrypted);
}

if (typeof window !== "undefined")
  window.addEventListener("mold:secure-pairings-changed", clearRelayInfoCache);
