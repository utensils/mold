import { isLoopbackOrigin } from "@studio/api/relayTransport";

export interface StudioPrimary {
  id: string;
  name: string;
  url: string;
  instanceId?: string;
}
const PRIMARY = "mold.web.primary.v3";
/** Hosted Studio is a client; its static origin is never a Mold server. */
export function standaloneStudio(): boolean {
  return (
    import.meta.env.VITE_STANDALONE_STUDIO === "1" ||
    !isLoopbackOrigin(new URL(window.location.origin))
  );
}
export function studioPrimary(): StudioPrimary | null {
  try {
    const value = JSON.parse(
      localStorage.getItem(PRIMARY) ?? "null",
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
export function setStudioPrimary(host: StudioPrimary | null): void {
  if (host)
    localStorage.setItem(
      PRIMARY,
      JSON.stringify({
        id: host.id,
        name: host.name,
        url: host.url,
        instanceId: host.instanceId,
      }),
    );
  else localStorage.removeItem(PRIMARY);
}
