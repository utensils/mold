/**
 * Profile-consistency checks for automatic multi-host generation routing.
 *
 * A model name is not a sufficient execution contract: two machines may run
 * different Mold versions and advertise different settings for that model.
 * Explicit routing is always safe because the selected machine is the
 * authority. Automatic routing is safe only when every eligible owner agrees
 * on the exact profile hash.
 */

export interface FleetProfileModel {
  name: string;
  downloaded: boolean;
  generation_profile?: { profile_hash?: string | null } | null;
}

export interface ProfileHashConflict {
  hostIds: string[];
  hashesByHost: Record<string, string | null>;
}

export interface ProfileConflictHost {
  label: string;
  profileHash: string | null;
}

function formatMachineList(hosts: readonly ProfileConflictHost[]): string {
  const labels = hosts.map((host) => host.label);
  if (labels.length < 2) return labels[0] ?? "the available machines";
  if (labels.length === 2) return `${labels[0]} and ${labels[1]}`;
  return `${labels.slice(0, -1).join(", ")}, and ${labels.at(-1)}`;
}

/** Explain why automatic routing cannot safely choose between model owners. */
export function profileConflictMessage(
  hosts: readonly ProfileConflictHost[],
): string {
  const owners = formatMachineList(hosts);
  const hasLegacyOwner = hosts.some((host) => !host.profileHash);
  const cause = hasLegacyOwner
    ? "At least one may be running an older Mold version, so the same controls could produce different results."
    : "They may be running different Mold versions or builds, so the same controls could produce different results.";
  return `Auto can't safely choose a machine because ${owners} use different generation settings for this model. ${cause} Update and reconnect them, or choose one machine for this print. Nothing was queued.`;
}

export function profileHashConflict(
  modelsByHost: Readonly<Record<string, readonly FleetProfileModel[]>>,
  modelName: string,
  eligibleHostIds: readonly string[],
): ProfileHashConflict | null {
  const owners = eligibleHostIds.flatMap((hostId) => {
    const model = modelsByHost[hostId]?.find(
      (candidate) => candidate.name === modelName && candidate.downloaded,
    );
    return model ? [{ hostId, model }] : [];
  });
  if (owners.length <= 1) return null;

  const hashesByHost = Object.fromEntries(
    owners.map(({ hostId, model }) => [
      hostId,
      model.generation_profile?.profile_hash?.trim() || null,
    ]),
  );
  const hashes = Object.values(hashesByHost);
  // During the one-release compatibility window, a fleet made entirely of
  // legacy servers has one shared (legacy) contract. The unsafe case is a
  // partially upgraded fleet or two concrete hashes that disagree.
  if (
    hashes.every((hash) => hash === null) ||
    (hashes.every((hash) => hash !== null) && new Set(hashes).size === 1)
  ) {
    return null;
  }
  return { hostIds: owners.map(({ hostId }) => hostId), hashesByHost };
}
