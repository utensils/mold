import type { CopyReceipt } from "../api/galleryCopy";
/** Hide-only evidence for an output whose retained inputs/organization are unfinished.
 * It is deliberately not a verified copy link or an organization ownership record. */
export interface ProvisionalCopyPrivacy {
  source: string;
  destination: string;
  sourceFilename: string;
  destinationFilename: string;
}
export function provisionalCopyPrivacy(
  receipts: Record<string, CopyReceipt>,
): ProvisionalCopyPrivacy[] {
  return Object.values(receipts).flatMap((receipt) =>
    receipt.stage !== "complete" &&
    receipt.destinationFilename &&
    receipt.sourceOrganization?.collections.some(
      (collection) => collection.hidden === true,
    )
      ? [
          {
            source: receipt.source,
            destination: receipt.destination,
            sourceFilename: receipt.sourceFilename,
            destinationFilename: receipt.destinationFilename,
          },
        ]
      : [],
  );
}

/** Resolve pending privacy only against the currently configured destination.
 * An unknown installation withholds the row until checked; a replacement cannot
 * inherit evidence. This set is never consulted for copying or ownership. */
export function provisionalCopyHiddenKeys(
  evidence: readonly ProvisionalCopyPrivacy[],
  hosts: readonly {
    id: string;
    url: string;
    instanceId?: string;
    connectionMode?: string;
  }[],
): Set<string> {
  const keys = new Set<string>();
  for (const edge of evidence) {
    let identity: unknown;
    try {
      identity = JSON.parse(edge.destination);
    } catch {
      continue;
    }
    if (
      !Array.isArray(identity) ||
      identity.length !== 4 ||
      !identity.every((v) => typeof v === "string")
    )
      continue;
    const [id, url, mode, instance] = identity;
    const host = hosts.find((candidate) => candidate.id === id);
    if (
      host &&
      host.url === url &&
      (host.connectionMode ?? "secure") === mode &&
      (!host.instanceId || host.instanceId === instance)
    )
      keys.add(`${id}|${edge.destinationFilename}`);
  }
  return keys;
}

/** Completion cannot expose a row before the fresh physical inventory arrives. */
export function retainCopyPrivacyUntilConfirmed(
  pending: ProvisionalCopyPrivacy[],
  previous: readonly ProvisionalCopyPrivacy[],
  receipts: Record<string, CopyReceipt>,
  confirmedHidden: ReadonlySet<string>,
): ProvisionalCopyPrivacy[] {
  const result = [...pending];
  for (const old of previous) {
    let id: unknown;
    try {
      id = JSON.parse(old.destination)[0];
    } catch {
      continue;
    }
    if (confirmedHidden.has(`${id}|${old.destinationFilename}`)) continue;
    if (
      Object.values(receipts).some(
        (row) =>
          row.source === old.source &&
          row.destination === old.destination &&
          row.sourceFilename === old.sourceFilename &&
          row.destinationFilename === old.destinationFilename &&
          row.stage === "complete" &&
          row.sourceOrganization?.collections.some(
            (collection) => collection.hidden === true,
          ),
      )
    )
      result.push(old);
  }
  return result;
}

/** Confirm privacy only from a row's own successfully listed host inventory.
 * Logical replica grouping can hide a peer but cannot prove its membership. */
export function confirmedPhysicalCopyPrivacyKeys(
  rows: readonly { hostId: string; filename: string; collections?: string[] }[],
  hosts: readonly {
    hostId: string;
    collectionsListingOk?: boolean;
    collections: readonly { id: string; hidden?: boolean }[];
  }[],
): Set<string> {
  const hiddenByHost = new Map(
    hosts
      .filter((host) => host.collectionsListingOk === true)
      .map((host) => [
        host.hostId,
        new Set(
          host.collections
            .filter((collection) => collection.hidden === true)
            .map((collection) => collection.id),
        ),
      ]),
  );
  return new Set(
    rows
      .filter((row) =>
        row.collections?.some((id) => hiddenByHost.get(row.hostId)?.has(id)),
      )
      .map((row) => `${row.hostId}|${row.filename}`),
  );
}
