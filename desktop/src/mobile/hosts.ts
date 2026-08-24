import type { ApiTarget } from "../lib/api/client";
import type { ServerStatus } from "../lib/api/types";
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
  /** False only after an explicit disconnect; the address and Keychain key remain. */
  connected?: boolean;
  /** A status read succeeded for this exact host target during this app session. */
  online: boolean;
  /** The latest status read failed after this exact target was verified. */
  stale?: boolean;
  /** Ephemeral transport detail for an unreachable or reconnecting host. */
  healthError?: string | undefined;
  /** The exact target rejected its HTTP credential; last-good authority is retired. */
  authorityRejected?: boolean | undefined;
  /** A successful probe reached a different server installation at this URL. */
  instanceMismatch?:
    | {
        expected: string;
        reported: string;
      }
    | undefined;
}

export type MobileHostStatusOutcome = "verified" | "instance_mismatch";

function normalizedInstanceId(value: string | null | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

/**
 * Apply one successful `/api/status` response to the mobile host authority.
 * A different non-empty instance id is a hard fence: keep the remembered
 * identity, mark the replacement explicitly, and require the user to remove
 * and re-add the machine before anything can route to it.
 */
export function recordMobileHostStatus(
  host: MobileHost,
  status: ServerStatus,
): MobileHostStatusOutcome {
  const expected = normalizedInstanceId(host.instanceId);
  const reported = normalizedInstanceId(status.instance_id);
  if (expected && reported && expected !== reported) {
    host.online = false;
    host.stale = false;
    host.healthError = undefined;
    host.authorityRejected = false;
    host.instanceMismatch = { expected, reported };
    return "instance_mismatch";
  }

  host.online = true;
  host.stale = false;
  host.healthError = undefined;
  host.authorityRejected = false;
  host.instanceMismatch = undefined;
  host.version = status.version;
  host.hostname = status.hostname ?? undefined;
  host.instanceId = reported ?? host.instanceId;
  return "verified";
}

/**
 * A transport failure is not evidence that a previously verified host died.
 * Keep last-good authority routable and label it stale; a never-verified host
 * remains unreachable, and an explicit identity mismatch remains fenced.
 */
export function recordMobileHostProbeFailure(host: MobileHost, error: unknown): void {
  if (host.instanceMismatch || host.authorityRejected) return;
  host.stale = host.online;
  host.healthError = error instanceof Error ? error.message : String(error);
}

/** Authentication rejection is positive authority evidence, not congestion. */
export function recordMobileHostAuthorityRejection(host: MobileHost, error: unknown): void {
  host.online = false;
  host.stale = false;
  host.authorityRejected = true;
  host.healthError = error instanceof Error ? error.message : String(error);
  host.instanceMismatch = undefined;
}

/** Compact, truthful state text shared by the Machines list and detail. */
export function mobileHostHealthLabel(host: MobileHost): string {
  if (host.connected === false) return "disconnected";
  if (host.instanceMismatch) return "identity changed";
  if (host.authorityRejected) return "access denied";
  if (host.stale) return "reconnecting\u2026";
  if (host.online) return host.version ? `v${host.version}` : "online";
  if (host.healthError) return "unreachable";
  return "connecting\u2026";
}

export function mobileHostTarget(host: MobileHost): ApiTarget {
  return { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
}

/** Exact authority fence for prepared work and placement previews. */
export function mobileHostMatchesRoute(route: HostRoute, host: MobileHost | undefined): boolean {
  if (!host || host.connected === false || !host.online || host.id !== route.hostId) return false;
  const target = mobileHostTarget(host);
  return (
    target.baseUrl === route.target.baseUrl &&
    target.apiKey === route.target.apiKey &&
    (route.instanceId === undefined || (host.instanceId ?? null) === route.instanceId)
  );
}
