/**
 * A family's short mark, its spelled-out name and its tone — the one mapping
 * behind every place a model shows a picture it does not have: the catalog
 * card's placeholder and a starting point with no photo of its own.
 */
export interface FamilyIdentity {
  mark: string;
  label: string;
  tone: "cool" | "warm" | "neutral";
}

export function familyIdentity(family: string): FamilyIdentity {
  const normalized = family.trim().toLowerCase();
  if (normalized === "flux2" || normalized.startsWith("flux.2")) {
    return { mark: "F2", label: "FLUX.2", tone: "cool" };
  }
  if (normalized.startsWith("flux")) return { mark: "F", label: "FLUX", tone: "cool" };
  if (normalized === "sdxl") return { mark: "XL", label: "SDXL", tone: "warm" };
  if (normalized === "sd15" || normalized.includes("1.5")) {
    return { mark: "1.5", label: "SD 1.5", tone: "warm" };
  }
  if (normalized.startsWith("sd3")) return { mark: "3", label: "SD 3", tone: "warm" };
  if (normalized.startsWith("qwen")) return { mark: "Q", label: "QWEN", tone: "neutral" };
  if (normalized.startsWith("zimage") || normalized.startsWith("z-image")) {
    return { mark: "Z", label: "Z-IMAGE", tone: "cool" };
  }
  if (normalized.startsWith("ltx")) return { mark: "LTX", label: "LTX VIDEO", tone: "neutral" };
  if (normalized.startsWith("wuerstchen")) {
    return { mark: "W", label: "WUERSTCHEN", tone: "warm" };
  }
  const label = normalized.replaceAll("-", " ").toUpperCase() || "MODEL";
  return { mark: label.slice(0, 3), label, tone: "neutral" };
}
