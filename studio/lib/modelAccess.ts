/**
 * Additive `/api/capabilities.model_access` contract shared by every Studio
 * surface. Older servers omit it, which means the client has no advertised
 * restriction to apply; current servers remain the final authority.
 */
export interface ModelAccessRestriction {
  code: string;
  family: string;
  message: string;
  license_url: string;
  authorization_url: string;
}

export interface ModelAccessCapabilityRecord {
  model_access?: {
    restrictions?: readonly ModelAccessRestriction[] | null;
  } | null;
}

export interface ModelAccessIdentity {
  model?: string | null | undefined;
  family?: string | null | undefined;
}

function normalizeIdentity(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function identityContainsFamily(identity: string, family: string): boolean {
  const normalizedIdentity = normalizeIdentity(identity);
  const normalizedFamily = normalizeIdentity(family);
  if (!normalizedIdentity || !normalizedFamily) return false;
  const separated =
    normalizedIdentity === normalizedFamily ||
    normalizedIdentity.startsWith(`${normalizedFamily}-`) ||
    normalizedIdentity.endsWith(`-${normalizedFamily}`) ||
    normalizedIdentity.includes(`-${normalizedFamily}-`);
  if (separated) return true;

  // Official implementation class names often compact the family identity
  // inside one token (for example MiniMaxH3Scheduler and
  // AutoencoderKLMiniMaxH3). Mirror the server's boundary: the family must
  // begin the token (apart from the conventional AutoencoderKL prefix), and a
  // suffix may be absent or begin with a letter. This rejects lookalikes such
  // as minimaxh30 and notminimaxh3.
  const compactFamily = normalizedFamily.replaceAll("-", "");
  return normalizedIdentity.split("-").some((token) => {
    const prefixes = [compactFamily, `autoencoderkl${compactFamily}`];
    return prefixes.some((prefix) => {
      if (!token.startsWith(prefix)) return false;
      const suffix = token.slice(prefix.length);
      return suffix.length === 0 || /^[a-z]/.test(suffix);
    });
  });
}

/** Return the first server-advertised restriction matching model or family. */
export function modelAccessRestrictionFor(
  capabilities: ModelAccessCapabilityRecord | null | undefined,
  identity: ModelAccessIdentity,
): ModelAccessRestriction | null {
  const identities = [identity.model, identity.family].filter(
    (value): value is string =>
      typeof value === "string" && value.trim().length > 0,
  );
  return (
    capabilities?.model_access?.restrictions?.find((restriction) =>
      identities.some((value) =>
        identityContainsFamily(value, restriction.family),
      ),
    ) ?? null
  );
}

export function isModelAccessRestricted(
  capabilities: ModelAccessCapabilityRecord | null | undefined,
  identity: ModelAccessIdentity,
): boolean {
  return modelAccessRestrictionFor(capabilities, identity) !== null;
}

/** Filter model rows using both their request identity and resolved family. */
export function filterRestrictedModels<
  T extends { name: string; family?: string | null },
>(
  models: readonly T[],
  capabilities: ModelAccessCapabilityRecord | null | undefined,
): T[] {
  return models.filter(
    (model) =>
      !isModelAccessRestricted(capabilities, {
        model: model.name,
        family: model.family,
      }),
  );
}
