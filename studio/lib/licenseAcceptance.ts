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

export interface ThirdPartyLicenseStatus extends LicenseTerms {
  accepted: boolean;
  required_by: string[];
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

export function licenseFromErrorBody(body: unknown): LicenseTerms | null {
  if (typeof body !== "object" || body === null || Array.isArray(body))
    return null;
  const row = body as Record<string, unknown>;
  return validTerms(row.license) ? row.license : null;
}
