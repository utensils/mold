import type { PreparedExpansionInputs } from "../lib/preparedExpansion";
import type { HostRoute } from "../stores/hosts";
import type { MobileHost } from "./hosts";

export interface MobileExpansionRecoveryRecord {
  readonly id: number;
  readonly leaseId: string;
  readonly model: string;
  readonly inputs: Readonly<PreparedExpansionInputs>;
  readonly route: Readonly<HostRoute>;
  readonly host: Readonly<MobileHost>;
  readonly requestToken: number;
  readonly replacePrepared: boolean;
}

interface MobileExpansionRecoveryDraft {
  id: number;
  leaseId: string;
  model: string;
  inputs: PreparedExpansionInputs;
  route: HostRoute;
  requestToken: number;
  replacePrepared: boolean;
}

export interface CurrentMobileExpansionRecovery {
  inputs: PreparedExpansionInputs;
  currentHost: MobileHost | undefined;
  tokenCurrent: boolean;
}

/** Deep, immutable recovery state. The pull host is derived only from the frozen route. */
export function createMobileExpansionRecovery(
  draft: MobileExpansionRecoveryDraft,
): MobileExpansionRecoveryRecord {
  const inputs = Object.freeze({ ...draft.inputs });
  const target = Object.freeze({ ...draft.route.target });
  const route = Object.freeze({ ...draft.route, target });
  const host = Object.freeze<MobileHost>({
    id: route.hostId,
    name: route.label,
    baseUrl: target.baseUrl,
    apiKey: target.apiKey ?? "",
    hostname: undefined,
    version: undefined,
    instanceId: route.instanceId ?? undefined,
    online: true,
  });
  return Object.freeze({
    id: draft.id,
    leaseId: draft.leaseId,
    model: draft.model,
    inputs,
    route,
    host,
    requestToken: draft.requestToken,
    replacePrepared: draft.replacePrepared,
  });
}

function sameInputs(
  current: PreparedExpansionInputs,
  frozen: Readonly<PreparedExpansionInputs>,
): boolean {
  return (
    current.sourcePrompt === frozen.sourcePrompt &&
    current.model === frozen.model &&
    current.family === frozen.family &&
    current.requestedCount === frozen.requestedCount &&
    current.selectedHostPolicy === frozen.selectedHostPolicy
  );
}

/** One directed reason suitable for inline recovery UI. */
export function mobileExpansionRecoveryStaleReason(
  recovery: MobileExpansionRecoveryRecord,
  current: CurrentMobileExpansionRecovery,
): string | null {
  if (!current.tokenCurrent) return "This expansion recovery is no longer current.";
  if (!sameInputs(current.inputs, recovery.inputs)) {
    return "Expansion inputs changed after the missing model was detected.";
  }
  const host = current.currentHost;
  if (!host || !host.online || host.id !== recovery.route.hostId) {
    return `${recovery.route.label} is no longer reachable.`;
  }
  if (
    host.baseUrl !== recovery.route.target.baseUrl ||
    (host.apiKey || null) !== recovery.route.target.apiKey ||
    (host.instanceId ?? null) !== (recovery.route.instanceId ?? null)
  ) {
    return `${recovery.route.label}'s connection details changed.`;
  }
  return null;
}
