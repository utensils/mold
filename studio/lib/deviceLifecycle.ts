import type { DeviceInfo } from "../api/devices";

/** Minimal capability shape shared by web, desktop, and iPhone clients. */
export interface DeviceLifecycleCapabilities {
  devices?: {
    lifecycle?: boolean;
    restart_enable?: boolean;
  } | null;
  dispatch?: {
    v2_authoritative?: boolean;
  } | null;
}

export type DeviceLifecycleMode = "live" | "restart_only" | "unsupported";

export function deviceLifecycleMode(
  capabilities: DeviceLifecycleCapabilities | null | undefined,
): DeviceLifecycleMode {
  if (
    capabilities?.devices?.lifecycle === true &&
    capabilities.dispatch?.v2_authoritative === true
  ) {
    return "live";
  }
  if (capabilities?.devices?.restart_enable === true) return "restart_only";
  return "unsupported";
}

export function canMutateDevice(
  device: DeviceInfo,
  capabilities: DeviceLifecycleCapabilities | null | undefined,
): boolean {
  if (device.admin_state === "startup_excluded") return false;
  const mode = deviceLifecycleMode(capabilities);
  if (mode === "live") return true;
  return (
    mode === "restart_only" &&
    !device.desired_enabled &&
    !device.restart_required
  );
}

export function deviceActionLabel(
  device: DeviceInfo,
  capabilities: DeviceLifecycleCapabilities | null | undefined,
): string {
  if (device.restart_required) return "Enabled on restart";
  if (
    device.admin_state !== "startup_excluded" &&
    !device.desired_enabled &&
    deviceLifecycleMode(capabilities) === "restart_only"
  ) {
    return "Enable on restart";
  }
  return device.desired_enabled ? "Disable" : "Enable";
}

export function deviceStateLabel(device: DeviceInfo): string {
  if (device.restart_required) return "Restart required";
  if (device.admin_state === "draining") return "Finishing current work";
  if (device.admin_state === "starting") return "Starting";
  if (device.admin_state === "startup_excluded") return "Excluded at startup";
  if (device.health !== "healthy") return device.health;
  return device.admin_state;
}

export function deviceLifecycleMessage(
  capabilities: DeviceLifecycleCapabilities | null | undefined,
): string {
  switch (deviceLifecycleMode(capabilities)) {
    case "live":
      return "Disabling a busy GPU lets its current stage finish, then removes it from scheduling.";
    case "restart_only":
      return "Live GPU controls require Scheduler V2. Disabled GPUs can be enabled for the next server restart.";
    case "unsupported":
      return "Live GPU controls are unavailable on this server.";
  }
}
