import type { ApiTarget } from "../lib/api/client";

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
