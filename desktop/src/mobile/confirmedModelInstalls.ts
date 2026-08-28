import type { ModelEntry } from "../lib/api/types";
import type { HostRoute } from "../stores/hosts";
import type { MobileHost } from "./hosts";

export interface ConfirmedModelInstall {
  route: HostRoute;
  modelExpiries: Record<string, number>;
}

export type ConfirmedModelInstalls = Record<string, ConfirmedModelInstall>;
export const CONFIRMED_MODEL_INSTALL_TTL_MS = 30_000;

/** Record an exact successful download without weakening its frozen identity. */
export function confirmModelInstall(
  claims: ConfirmedModelInstalls,
  route: HostRoute,
  model: string,
  nowMs = Date.now(),
): ConfirmedModelInstalls {
  const prior = claims[route.hostId];
  const sameRoute =
    prior?.route.target.baseUrl === route.target.baseUrl &&
    prior.route.target.apiKey === route.target.apiKey &&
    prior.route.instanceId === route.instanceId;
  const livePrior = sameRoute
    ? Object.fromEntries(
        Object.entries(prior.modelExpiries).filter(([, expiresAtMs]) => expiresAtMs > nowMs),
      )
    : {};
  return {
    ...claims,
    [route.hostId]: {
      route,
      modelExpiries: {
        ...livePrior,
        [model]: nowMs + CONFIRMED_MODEL_INSTALL_TTL_MS,
      },
    },
  };
}

/** A fresh inventory accounts for claims it now reports. Empty/stale-looking
 * snapshots cannot revoke a successful download; host authority retirement
 * handles removal or identity replacement. */
export function accountForConfirmedInventory(
  claims: ConfirmedModelInstalls,
  hostId: string,
  entries: readonly ModelEntry[],
): ConfirmedModelInstalls {
  const claim = claims[hostId];
  if (!claim) return claims;
  const installed = new Set(entries.filter((entry) => entry.downloaded).map((entry) => entry.name));
  const remaining = Object.fromEntries(
    Object.entries(claim.modelExpiries).filter(([model]) => !installed.has(model)),
  );
  if (Object.keys(remaining).length === Object.keys(claim.modelExpiries).length) return claims;
  const next = { ...claims };
  if (Object.keys(remaining).length === 0) delete next[hostId];
  else next[hostId] = { ...claim, modelExpiries: remaining };
  return next;
}

export function retireConfirmedModelAuthority(
  claims: ConfirmedModelInstalls,
  hostId: string,
): ConfirmedModelInstalls {
  if (!claims[hostId]) return claims;
  const next = { ...claims };
  delete next[hostId];
  return next;
}

export function confirmedModelHostIds(
  claims: ConfirmedModelInstalls,
  model: string,
  hosts: readonly MobileHost[],
  routeMatches: (route: HostRoute, host: MobileHost) => boolean,
  nowMs = Date.now(),
): string[] {
  return hosts
    .filter((host) => {
      const claim = claims[host.id];
      return Boolean(
        claim && (claim.modelExpiries[model] ?? 0) > nowMs && routeMatches(claim.route, host),
      );
    })
    .map((host) => host.id);
}
