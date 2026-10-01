import type { InjectionKey } from "vue";
import { relayFetch } from "@studio/api/relayTransport";
export const ORIGIN_ACCESS_CHANGE_KEY: InjectionKey<() => void> = Symbol(
  "mold.origin.change-key",
);
/** Browser credentials belong to this exact serving origin and this tab's session. */
const BrowserURL = URL;
const storageKey = () => `mold.web.origin-key.v1:${window.location.origin}`;
export function originApiKey(): string | null {
  try {
    return sessionStorage.getItem(storageKey()) || null;
  } catch {
    return null;
  }
}
export function setOriginApiKey(key: string): void {
  const value = key.trim();
  if (value) sessionStorage.setItem(storageKey(), value);
  else sessionStorage.removeItem(storageKey());
}
export function originApiTarget() {
  return { baseUrl: "", apiKey: originApiKey() };
}
/** Never carry an origin credential to another host, assets, or a redirect. */
export function originAuthenticatedFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const url = new BrowserURL(
    input instanceof Request ? input.url : String(input),
    window.location.origin,
  );
  const headers = new Headers(
    init?.headers ?? (input instanceof Request ? input.headers : undefined),
  );
  let injected = false;
  if (
    url.origin === window.location.origin &&
    url.pathname.startsWith("/api/") &&
    !headers.has("x-api-key")
  ) {
    const key = originApiKey();
    if (key) {
      headers.set("x-api-key", key);
      injected = true;
    }
  }
  if (!headers.has("x-api-key")) return relayFetch(input, init);
  return relayFetch(input, {
    ...init,
    ...(injected ? { headers } : {}),
    redirect: "error",
  });
}
