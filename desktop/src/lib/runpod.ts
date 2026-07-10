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
  machine: { gpuDisplayName: string | null; location: string | null } | null;
}

export interface RunPodOverview {
  configured: boolean;
  credentialSource: "keychain" | "environment" | "config" | null;
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

const STOCK_RANK: Record<string, number> = { High: 3, Medium: 2, Low: 1, None: 0 };

export function rankRunPodGpus(gpus: RunPodGpu[]): RunPodGpu[] {
  return [...gpus].sort((a, b) => {
    const stock =
      (STOCK_RANK[b.stockStatus ?? "None"] ?? 0) - (STOCK_RANK[a.stockStatus ?? "None"] ?? 0);
    return stock || a.memoryInGb - b.memoryInGb || a.displayName.localeCompare(b.displayName);
  });
}

export const podProxyUrl = (id: string): string => `https://${id}-7680.proxy.runpod.net`;

export const emptyRunPodOverview = (): RunPodOverview => ({
  configured: false,
  credentialSource: null,
  account: null,
  pods: [],
  gpus: [],
  datacenters: [],
  networkVolumes: [],
});
