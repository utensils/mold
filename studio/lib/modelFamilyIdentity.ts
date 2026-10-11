/**
 * A family's short mark and its tone — what every place draws for a model
 * it has no picture of: the catalog card's placeholder (web and desktop) and
 * a starting point with no photo of its own. The family's NAME is not here:
 * it is `familyLabel`, the one authority every surface reads.
 */
export interface FamilyIdentity {
  mark: string;
  tone: "cool" | "warm" | "neutral";
}

export function familyIdentity(family: string): FamilyIdentity {
  const normalized = family.trim().toLowerCase();
  if (normalized === "flux2" || normalized.startsWith("flux.2")) {
    return { mark: "F2", tone: "cool" };
  }
  if (normalized.startsWith("flux")) return { mark: "F", tone: "cool" };
  if (normalized === "sdxl") return { mark: "XL", tone: "warm" };
  if (normalized === "sd15" || normalized.includes("1.5")) {
    return { mark: "1.5", tone: "warm" };
  }
  if (normalized.startsWith("sd3")) return { mark: "3", tone: "warm" };
  if (normalized.startsWith("qwen")) return { mark: "Q", tone: "neutral" };
  if (normalized.startsWith("zimage") || normalized.startsWith("z-image")) {
    return { mark: "Z", tone: "cool" };
  }
  if (normalized.startsWith("ltx")) return { mark: "LTX", tone: "neutral" };
  if (normalized.startsWith("wuerstchen")) return { mark: "W", tone: "warm" };
  const mark = normalized.replaceAll("-", " ").toUpperCase() || "MODEL";
  return { mark: mark.slice(0, 3), tone: "neutral" };
}
