import type { ApiTarget } from "../lib/api/client";
import type { HostView } from "../stores/hosts";

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

/**
 * Adapt the remote-only iPhone record to the shared desktop host contract.
 * The iOS shell deliberately does not initialize `useHostsStore`: that store
 * owns a desktop-local engine and invokes native commands the phone does not
 * ship. Pure consumers such as the jobs and downloads stores can still share
 * their proven per-host logic through this narrow adapter.
 */
export function mobileHostView(host: MobileHost): HostView {
  return {
    id: host.id,
    label: host.name,
    kind: "remote",
    baseUrl: host.baseUrl,
    apiKey: host.apiKey || null,
    status: host.online ? "ready" : "error",
    // "Selected" on iPhone is a generation preference, not a desktop-local
    // primary connection. Keep every mobile host remote for shared stores.
    primary: false,
    queueDepth: null,
    queueCapacity: null,
    version: host.version ?? null,
    instanceId: host.instanceId ?? null,
  };
}
