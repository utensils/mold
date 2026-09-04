import { gpuFleetLabel } from "./api/gpuStatus";
import { inferBackendFromGpuName } from "./hosts";
import { formatUptime } from "./format";
import type { GpuSnapshot } from "./api/types";

/** What a machine sentence needs to know about the box it describes. */
export interface MachineIdentity {
  kind: "local" | "remote";
  baseUrl?: string | null;
}

/** "4× L40S · CUDA" — every card in the box, and what runs them. Empty before
 *  any telemetry has arrived, so the sentence simply starts at where it is. */
export function machineHardware(gpus: readonly GpuSnapshot[]): string {
  const fleet = gpuFleetLabel(gpus);
  const first = gpus[0];
  if (!fleet || !first) return fleet;
  return `${fleet} · ${(first.backend || inferBackendFromGpuName(first.name)).toUpperCase()}`;
}

/** Where the machine lives, in the lexicon's words. The address rides the
 *  machine card; the machine pane leaves it to the name's tooltip. */
export function machineLocation(host: MachineIdentity, withAddress = false): string {
  if (host.kind === "local") return "this device";
  const address = host.baseUrl?.replace(/^https?:\/\//, "") ?? "";
  if (/\.runpod\.net/.test(address)) return "rented cloud GPU";
  return withAddress && address ? `on your network at ${address}` : "on your network";
}

/**
 * The ONE plain sentence about a machine: what it is, where it lives, and how
 * long it has been up. The Machines list and the machine pane both say it, so
 * a 4× L40S box can never read as one card in one place and four in the other.
 */
export function machineSentence(
  host: MachineIdentity,
  gpus: readonly GpuSnapshot[],
  options: { address?: boolean; uptimeSeconds?: number | null } = {},
): string {
  const uptime = options.uptimeSeconds;
  return [
    machineHardware(gpus),
    machineLocation(host, options.address === true),
    uptime === null || uptime === undefined ? "" : `up ${formatUptime(uptime)}`,
  ]
    .filter(Boolean)
    .join(" · ");
}
