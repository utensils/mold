import type { ApiTarget } from "../lib/api/client";
import type { HostRoute } from "../stores/hosts";

// Mobile and desktop intentionally share host normalization, secret-key
// slugs, and the host shape consumed by the reusable queue/download stores.
// Keeping one implementation prevents the same remote from acquiring
// different identities or subtly different API routing across clients.
export {
  hostIdFromUrl as remoteHostId,
  normalizeHostUrl as normalizeRemoteAddress,
} from "../lib/hosts";

export interface MobileHost {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  hostname: string | undefined;
  version: string | undefined;
  /** Stable server installation id, kept separate from the URL-based row id. */
  instanceId?: string | undefined;
  online: boolean;
}

export function mobileHostTarget(host: MobileHost): ApiTarget {
  return { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
}

/** Exact authority fence for prepared work and placement previews. */
export function mobileHostMatchesRoute(route: HostRoute, host: MobileHost | undefined): boolean {
  if (!host || !host.online || host.id !== route.hostId) return false;
  const target = mobileHostTarget(host);
  return (
    target.baseUrl === route.target.baseUrl &&
    target.apiKey === route.target.apiKey &&
    (route.instanceId === undefined || (host.instanceId ?? null) === route.instanceId)
  );
}
