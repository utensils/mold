export interface RunPodAccount {
  email: string;
  balance: number;
  spendPerHour: number;
  spendLimit: number | null;
}

export interface RunPodGpu {
  id: string | null;
  displayName: string;
  gpuId?: string;
  memoryInGb: number;
  secureCloud: boolean;
  communityCloud: boolean;
  stockStatus: string | null;
  available: boolean;
}

export interface RunPodDatacenter {
  id: string;
  name: string;
  location: string | null;
  gpuAvailability: Array<{
    displayName: string;
    gpuId: string;
    stockStatus: string | null;
  }>;
}

export interface RunPodNetworkVolume {
  id: string;
  name: string;
  dataCenterId: string;
  size: number;
}

export interface RunPodPod {
  id: string;
  name: string | null;
  desiredStatus: string;
  imageName: string | null;
  gpuCount: number;
  costPerHr: number;
  uptimeSeconds: number;
  memoryInGb: number;
  vcpuCount: number;
  volumeInGb: number;
  machine: {
    gpuDisplayName: string | null;
    gpuTypeId: string | null;
    dataCenterId: string | null;
    location: string | null;
  } | null;
  gpu: { id: string | null; displayName: string | null; count: number | null } | null;
  networkVolumeId: string | null;
  networkVolume: RunPodNetworkVolume | null;
}

export interface RunPodNetworkVolumeCreateInput {
  name: string;
  sizeGb: number;
  datacenterId: string;
}

export interface RunPodNetworkVolumeUpdateInput {
  id: string;
  name: string | null;
  sizeGb: number | null;
}

export interface RunPodOverview {
  configured: boolean;
  credentialSource: "app" | "environment" | "config" | null;
  account: RunPodAccount | null;
  pods: RunPodPod[];
  gpus: RunPodGpu[];
  datacenters: RunPodDatacenter[];
  networkVolumes: RunPodNetworkVolume[];
}

export interface RunPodCreateInput {
  name: string | null;
  gpuTypeId: string;
  gpuDisplayName: string;
  cloudType: "SECURE" | "COMMUNITY";
  datacenterId: string | null;
  containerDiskGb: number;
  volumeGb: number;
  networkVolumeId: string | null;
  model: string | null;
  includeHfToken: boolean;
}

// Production REST currently exposes pod-only datacenters alongside locations
// that can host network volumes. Keep the volume picker fail-closed so users
// never discover the distinction through a billable API error.
export const NETWORK_VOLUME_DATACENTER_IDS = new Set([
  "AP-IN-2",
  "AP-JP-1",
  "CA-MTL-3",
  "CA-MTL-4",
  "EU-CZ-1",
  "EU-FR-1",
  "EU-NL-1",
  "EU-RO-1",
  "EUR-IS-1",
  "EUR-IS-3",
  "EUR-NO-1",
  "EUR-NO-2",
  "US-CA-2",
  "US-IL-1",
  "US-MO-2",
  "US-NC-1",
  "US-NE-1",
  "US-TX-3",
  "US-WA-1",
]);

const REGION_NAMES: Record<string, string> = {
  IN: "India",
  JP: "Japan",
  MTL: "Montreal, Canada",
  CZ: "Czech Republic, Europe",
  FR: "France, Europe",
  NL: "Netherlands, Europe",
  RO: "Romania, Europe",
  IS: "Iceland, Europe",
  NO: "Norway, Europe",
  CA: "California, United States",
  IL: "Illinois, United States",
  MO: "Missouri, United States",
  NC: "North Carolina, United States",
  NE: "Nebraska, United States",
  TX: "Texas, United States",
  WA: "Washington, United States",
};

export const isNetworkVolumeDatacenter = (id: string): boolean =>
  NETWORK_VOLUME_DATACENTER_IDS.has(id);

export function runPodRegionLabel(id: string, fallback?: string | null): string {
  const regionCode = id.split("-").at(-2) ?? "";
  const region = REGION_NAMES[regionCode] ?? fallback ?? "Location unavailable";
  return `${region} · ${id}`;
}

export function friendlyRunPodError(message: string): string {
  let detail = message;
  const jsonStart = message.indexOf("{");
  if (jsonStart >= 0) {
    try {
      const payload = JSON.parse(message.slice(jsonStart)) as { error?: string };
      if (payload.error) detail = payload.error;
    } catch {
      // Preserve the original response when RunPod returns malformed JSON.
    }
  }
  const unsupported = detail.match(/Data center [\\"']*([A-Z]{2,3}-[A-Z]{2,3}-\d+)[\\"']*/i);
  if (unsupported?.[1] && detail.includes("does not support network volumes")) {
    return `${unsupported[1]} does not support network volumes. Choose a supported region and try again.`;
  }
  return detail.replace(/^create network volume:\s*/i, "");
}

const STOCK_RANK: Record<string, number> = { High: 3, Medium: 2, Low: 1, None: 0 };

export function rankRunPodGpus(gpus: RunPodGpu[]): RunPodGpu[] {
  return [...gpus].sort((a, b) => {
    const stock =
      (STOCK_RANK[b.stockStatus ?? "None"] ?? 0) - (STOCK_RANK[a.stockStatus ?? "None"] ?? 0);
    return stock || a.memoryInGb - b.memoryInGb || a.displayName.localeCompare(b.displayName);
  });
}

export const podProxyUrl = (id: string): string => `https://${id}-7680.proxy.runpod.net`;

function endpointOrigin(value: string): string | null {
  try {
    return new URL(value).origin.toLowerCase();
  } catch {
    return null;
  }
}

/** Match a routed Mold host to the running RunPod proxy that owns it. */
export function runPodForHostUrl(
  pods: readonly RunPodPod[],
  hostUrl: string | null | undefined,
): RunPodPod | null {
  if (!hostUrl) return null;
  const origin = endpointOrigin(hostUrl);
  if (!origin) return null;
  return (
    pods.find(
      (pod) =>
        pod.desiredStatus.toUpperCase() === "RUNNING" &&
        endpointOrigin(podProxyUrl(pod.id)) === origin,
    ) ?? null
  );
}

export const podGpuName = (pod: RunPodPod): string =>
  pod.gpu?.displayName ??
  pod.machine?.gpuDisplayName ??
  pod.machine?.gpuTypeId ??
  pod.gpu?.id ??
  "GPU details unavailable";

/** Cost of the session so far: uptime × hourly rate, floored at zero. */
export function estimatedPodCost(costPerHr: number, uptimeSeconds: number): number {
  if (!(costPerHr > 0) || !(uptimeSeconds > 0)) return 0;
  return (uptimeSeconds / 3600) * costPerHr;
}

/**
 * Live session cost for the running-cost meter (§08 G9): the server reports
 * `uptimeSeconds` only when the overview is polled, so the meter ticks by
 * adding the wall-clock time elapsed since that snapshot was read. Pure so the
 * meter math is unit-testable without timers.
 */
export function liveAccruedCost(
  costPerHr: number,
  snapshotUptimeSeconds: number,
  snapshotAtMs: number,
  nowMs: number,
): number {
  const elapsedSeconds = Math.max(0, (nowMs - snapshotAtMs) / 1000);
  return estimatedPodCost(costPerHr, snapshotUptimeSeconds + elapsedSeconds);
}

/** USD formatter shared by every cost surface so the meter reads identically. */
export function formatUsd(value: number): string {
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(value);
}

/** One-line hardware summary ("2× GPU · 16 vCPU · 62 GB RAM · 40 GB disk"). */
export function podHardwareSummary(pod: RunPodPod): string {
  const parts: string[] = [];
  if (pod.gpuCount > 1) parts.push(`${pod.gpuCount}× GPU`);
  if (pod.vcpuCount > 0) parts.push(`${pod.vcpuCount} vCPU`);
  if (pod.memoryInGb > 0) parts.push(`${pod.memoryInGb} GB RAM`);
  if (pod.volumeInGb > 0) parts.push(`${pod.volumeInGb} GB disk`);
  if (pod.machine?.dataCenterId) parts.push(pod.machine.dataCenterId);
  return parts.join(" · ");
}

export const emptyRunPodOverview = (): RunPodOverview => ({
  configured: false,
  credentialSource: null,
  account: null,
  pods: [],
  gpus: [],
  datacenters: [],
  networkVolumes: [],
});
