import { isLoopbackOrigin } from "@studio/api/relayTransport";

export interface StudioPrimary {
  id: string;
  name: string;
  url: string;
  instanceId?: string;
}
export const STUDIO_PRIMARY_KEY = "mold.web.primary.v3";
export const STUDIO_PRIMARY_CHANGED = "mold:studio-primary-changed";
// Every consumer in this tab uses the same accepted selection. Reading storage
// on each request could retarget mounted UI when another tab switches machines.
let primary: StudioPrimary | null | undefined;
/** Hosted Studio is a client; its static origin is never a Mold server. */
export function standaloneStudio(): boolean {
  return (
    import.meta.env.VITE_STANDALONE_STUDIO === "1" ||
    !isLoopbackOrigin(new URL(window.location.origin))
  );
}
function readPrimary(): StudioPrimary | null {
  try {
    const value = JSON.parse(
      localStorage.getItem(STUDIO_PRIMARY_KEY) ?? "null",
    ) as StudioPrimary | null;
    return value &&
      typeof value.id === "string" &&
      typeof value.name === "string" &&
      typeof value.url === "string" &&
      new URL(value.url).origin === value.url
      ? value
      : null;
  } catch {
    return null;
  }
}
export function studioPrimary(): StudioPrimary | null {
  if (primary === undefined) {
    const stored = readPrimary();
    primary = stored ? Object.freeze(stored) : null;
  }
  return primary;
}
/** Suspend this tab without overwriting another tab's saved selection. */
export function suspendStudioPrimary(): void {
  primary = null;
}
export function setStudioPrimary(host: StudioPrimary | null): void {
  const next = host
    ? Object.freeze({
        id: host.id,
        name: host.name,
        url: host.url,
        instanceId: host.instanceId,
      })
    : null;
  if (next) localStorage.setItem(STUDIO_PRIMARY_KEY, JSON.stringify(next));
  else {
    // Clearing this tab must not erase a newer selection saved by another tab
    // before its asynchronous storage event reaches us.
    const stored = readPrimary();
    const current = studioPrimary();
    if (
      !stored ||
      (current &&
        stored.id === current.id &&
        stored.url === current.url &&
        stored.instanceId === current.instanceId)
    )
      localStorage.removeItem(STUDIO_PRIMARY_KEY);
  }
  primary = next;
  window.dispatchEvent(new CustomEvent(STUDIO_PRIMARY_CHANGED));
}
