/**
 * Profile-consistency checks for automatic multi-host generation routing.
 *
 * A model name is not a sufficient execution contract across incompatible
 * protocol generations. Explicit routing is always safe because the selected
 * machine is the authority. Automatic routing only stops at a Mold major-
 * version boundary; minor, patch, and build drift is intentionally compatible.
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
  version?: string | null;
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
  return `Auto can't safely choose a machine because ${owners} use incompatible major Mold versions for this model. Update and reconnect them, or choose one machine for this print. Nothing was queued.`;
}

function majorVersion(version: string | null | undefined): number | null {
  const match = version?.trim().match(/^v?(\d+)(?:\.|$)/);
  if (!match) return null;
  const major = Number(match[1]);
  return Number.isSafeInteger(major) ? major : null;
}

export function profileHashConflict(
  modelsByHost: Readonly<Record<string, readonly FleetProfileModel[]>>,
  modelName: string,
  eligibleHostIds: readonly string[],
  versionsByHost: Readonly<Record<string, string | null | undefined>> = {},
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
  const majors = new Set(
    owners
      .map(({ hostId }) => majorVersion(versionsByHost[hostId]))
      .filter((major): major is number => major !== null),
  );
  // Profile hashes naturally drift as defaults and capabilities evolve. They
  // remain useful diagnostics, but only a definite major-version split is an
  // automatic-routing incompatibility. Unknown versions fail open here; the
  // selected host's placement preview still validates the concrete request.
  if (majors.size <= 1) return null;
  return { hostIds: owners.map(({ hostId }) => hostId), hashesByHost };
}
