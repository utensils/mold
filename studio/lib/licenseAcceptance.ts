export interface LicenseTerms {
  id: string;
  name: string;
  url: string;
  canonical: string;
  sha256: string;
  summary: string;
}

export interface LicenseAcceptance {
  id: string;
  url: string;
  sha256: string;
}

/** A style a licence gates, in the registry's own words (additive). */
export interface LicensedStyle {
  name: string;
  description: string;
}

export interface ThirdPartyLicenseStatus extends LicenseTerms {
  accepted: boolean;
  required_by: string[];
  /** The same styles as `required_by`, each with its plain description, so a
   * row can lead with what the licence unlocks. Absent on an older host. */
  required_by_styles?: LicensedStyle[] | null;
}

/**
 * What a licence row leads with: the styles it unlocks, in the registry's
 * plain words when the host lists them, else their ids, else the licence's
 * own name. The licence id never leads — it is the machine's handle.
 */
export function licenseFriendlyLine(license: ThirdPartyLicenseStatus): string {
  const styles = license.required_by_styles ?? [];
  const described = styles
    .map((style) => style.description.trim())
    .filter((description) => description.length > 0);
  if (described.length > 0) return described.join(" · ");
  if (license.required_by.length > 0) return license.required_by.join(" · ");
  return license.name;
}

/** The mono line under the friendly one: the licence's name and its summary
 * when styles lead, or the id and summary when the name itself leads. */
export function licenseDetailLine(license: ThirdPartyLicenseStatus): string {
  const leadsWithName = licenseFriendlyLine(license) === license.name;
  return `${leadsWithName ? license.id : license.name} · ${license.summary}`;
}

export interface LicenseListing {
  licenses: ThirdPartyLicenseStatus[];
}

export interface LicensedPendingDownload {
  install_model?: string | null;
  licenses?: LicenseTerms[] | null;
}

export interface LicenseRequirement {
  installModel: string;
  licenses: LicenseTerms[];
}

export function acceptanceFor(license: LicenseTerms): LicenseAcceptance {
  return { id: license.id, url: license.url, sha256: license.sha256 };
}

function validTerms(value: unknown): value is LicenseTerms {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    return false;
  const row = value as Record<string, unknown>;
  return ["id", "name", "url", "canonical", "sha256", "summary"].every(
    (key) => typeof row[key] === "string" && row[key].length > 0,
  );
}

/** Registry-shaped requirements carried by any placement preview.
 *
 * The UI deliberately knows no model or license ids. Adding a gated manifest
 * server-side automatically feeds this same consent/download/resume workflow.
 */
export function licenseRequirements(
  downloads: readonly LicensedPendingDownload[] | null | undefined,
): LicenseRequirement[] {
  const byModel = new Map<string, Map<string, LicenseTerms>>();
  for (const download of downloads ?? []) {
    const model = download.install_model?.trim();
    if (!model || !Array.isArray(download.licenses)) continue;
    for (const candidate of download.licenses) {
      if (!validTerms(candidate)) continue;
      const licenses = byModel.get(model) ?? new Map<string, LicenseTerms>();
      licenses.set(
        `${candidate.id}\0${candidate.url}\0${candidate.sha256.toLowerCase()}`,
        candidate,
      );
      byModel.set(model, licenses);
    }
  }
  return [...byModel.entries()].map(([installModel, licenses]) => ({
    installModel,
    licenses: [...licenses.values()],
  }));
}

/** The requirement implied by a host's refusal.
 *
 * The bundle name comes from the CALLER's own request, never from the payload:
 * the server refuses by license, and only the caller knows what it asked to
 * install. Keeps the UI registry-blind — it still names no model and no
 * license id.
 */
export function licenseRequirementFromError(
  body: unknown,
  installModel: string,
): LicenseRequirement | null {
  const terms = licenseFromErrorBody(body);
  if (!terms) return null;
  const model = installModel.trim();
  if (!model) return null;
  return { installModel: model, licenses: [terms] };
}

export function licenseFromErrorBody(body: unknown): LicenseTerms | null {
  if (typeof body !== "object" || body === null || Array.isArray(body))
    return null;
  const row = body as Record<string, unknown>;
  return validTerms(row.license) ? row.license : null;
}
