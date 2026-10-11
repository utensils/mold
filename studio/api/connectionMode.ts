/** Security is an explicit client choice, never a reachability fallback. */
export type ConnectionMode = "secure" | "regular";
export const CONNECTION_MODE_HEADER = "x-mold-connection-mode";
const PREFIX = "mold.connection-mode.v1:";

export function isManagedProxyOrigin(value: string): boolean {
  const hostname = new URL(value).hostname.toLowerCase().replace(/\.$/, "");
  return (
    hostname === "mold-link.urandom.io" ||
    hostname.endsWith(".mold-link.urandom.io")
  );
}
/** QR codes name an origin; direct HTTP works on LAN, VPN and WAN alike. */
export function directOrigin(value: string): string {
  const url = new URL(value);
  if (
    value !== value.trim() ||
    /[\s\u0000-\u001f\u007f]/.test(value) ||
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    value.includes("?") ||
    value.includes("#") ||
    url.port === "0" ||
    url.search ||
    url.hash
  )
    throw new Error(
      "Enter a direct HTTP or HTTPS machine address without credentials or a request path.",
    );
  if (isManagedProxyOrigin(value))
    throw new Error("The Mold proxy always requires a Secure connection.");
  return url.origin;
}
export function connectionModeFor(origin: string): ConnectionMode {
  try {
    return localStorage.getItem(
      PREFIX + new URL(origin, location.href).origin,
    ) === "regular"
      ? "regular"
      : "secure";
  } catch {
    return "secure";
  }
}
/** Only call after explicit pairing/connection. No credentials are stored here. */
export function rememberConnectionMode(
  origin: string,
  mode: ConnectionMode,
): void {
  const key = new URL(origin).origin;
  if (mode === "regular") directOrigin(key);
  localStorage.setItem(PREFIX + key, mode);
}
export function forgetConnectionMode(origin: string): void {
  localStorage.removeItem(PREFIX + new URL(origin).origin);
}
