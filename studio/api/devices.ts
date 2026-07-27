import { IncompatibleHostError, apiJsonTo, type ApiTarget } from "./client";

export type DeviceBackend = "cuda" | "metal";
export type DeviceKind = "full_gpu" | "mig" | "unknown_cuda" | "metal";
export type DeviceAdminState =
  "startup_excluded" | "starting" | "enabled" | "draining" | "disabled";
export type DeviceHealth = "healthy" | "degraded" | "unavailable" | "poisoned";
export type DeviceActivity =
  | "idle"
  | "loading"
  | "generating"
  | "upscaling"
  | "admin_loading"
  | "stopping";

export interface DeviceMemory {
  total_bytes: number | null;
  used_bytes: number | null;
  mold_used_bytes: number | null;
  other_used_bytes: number | null;
}

export interface DeviceTelemetry {
  utilization_percent: number | null;
  temperature_c: number | null;
  power_w: number | null;
}

export interface DeviceInfo {
  id: string;
  backend: DeviceBackend;
  ordinal: number | null;
  device_kind: DeviceKind;
  nvml_uuid: string | null;
  physical_uuid: string | null;
  mig_uuid: string | null;
  mig_parent_uuid: string | null;
  mig_profile: string | null;
  name: string;
  pci_bus_id: string | null;
  compute_capability: string | null;
  memory: DeviceMemory;
  telemetry: DeviceTelemetry;
  desired_enabled: boolean;
  restart_required?: boolean;
  admin_state: DeviceAdminState;
  health: DeviceHealth;
  activity: DeviceActivity;
  schedulable: boolean;
  unschedulable_reason: string | null;
  loaded_models: string[];
  active_work_id: string | null;
  planned_work_ids: string[];
}

export interface DeviceListResponse {
  devices: DeviceInfo[];
  plan_version: number;
}

const backends = new Set<DeviceBackend>(["cuda", "metal"]);
const deviceKinds = new Set<DeviceKind>([
  "full_gpu",
  "mig",
  "unknown_cuda",
  "metal",
]);
const adminStates = new Set<DeviceAdminState>([
  "startup_excluded",
  "starting",
  "enabled",
  "draining",
  "disabled",
]);
const healthStates = new Set<DeviceHealth>([
  "healthy",
  "degraded",
  "unavailable",
  "poisoned",
]);
const activities = new Set<DeviceActivity>([
  "idle",
  "loading",
  "generating",
  "upscaling",
  "admin_loading",
  "stopping",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isNullableNumber(value: unknown): value is number | null {
  return (
    value === null || (typeof value === "number" && Number.isFinite(value))
  );
}

function isNullableNonnegativeNumber(value: unknown): value is number | null {
  return isNullableNumber(value) && (value === null || value >= 0);
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function parseDevice(
  value: unknown,
  index: number,
  missing: string[],
): DeviceInfo | null {
  if (!isRecord(value)) {
    missing.push(`devices[${index}]`);
    return null;
  }
  const requireField = (
    key: string,
    predicate: (candidate: unknown) => boolean,
  ): void => {
    if (!predicate(value[key])) missing.push(`devices[${index}].${key}`);
  };

  requireField(
    "id",
    (candidate) => typeof candidate === "string" && candidate.length > 0,
  );
  requireField(
    "backend",
    (candidate) =>
      typeof candidate === "string" && backends.has(candidate as DeviceBackend),
  );
  requireField(
    "ordinal",
    (candidate) =>
      candidate === null ||
      (typeof candidate === "number" &&
        Number.isInteger(candidate) &&
        candidate >= 0),
  );
  requireField(
    "device_kind",
    (candidate) =>
      typeof candidate === "string" && deviceKinds.has(candidate as DeviceKind),
  );
  for (const key of [
    "nvml_uuid",
    "physical_uuid",
    "mig_uuid",
    "mig_parent_uuid",
    "mig_profile",
    "pci_bus_id",
    "compute_capability",
    "unschedulable_reason",
    "active_work_id",
  ]) {
    requireField(key, isNullableString);
  }
  requireField(
    "name",
    (candidate) => typeof candidate === "string" && candidate.length > 0,
  );
  requireField(
    "desired_enabled",
    (candidate) => typeof candidate === "boolean",
  );
  requireField(
    "admin_state",
    (candidate) =>
      typeof candidate === "string" &&
      adminStates.has(candidate as DeviceAdminState),
  );
  requireField(
    "health",
    (candidate) =>
      typeof candidate === "string" &&
      healthStates.has(candidate as DeviceHealth),
  );
  requireField(
    "activity",
    (candidate) =>
      typeof candidate === "string" &&
      activities.has(candidate as DeviceActivity),
  );
  requireField("schedulable", (candidate) => typeof candidate === "boolean");
  requireField("loaded_models", isStringArray);
  requireField("planned_work_ids", isStringArray);

  const memory = value.memory;
  if (!isRecord(memory)) {
    missing.push(`devices[${index}].memory`);
  } else {
    if (!isNullableNonnegativeNumber(memory.total_bytes)) {
      missing.push(`devices[${index}].memory.total_bytes`);
    }
    for (const key of ["used_bytes", "mold_used_bytes", "other_used_bytes"]) {
      if (!isNullableNonnegativeNumber(memory[key])) {
        missing.push(`devices[${index}].memory.${key}`);
      }
    }
  }

  const telemetry = value.telemetry;
  if (!isRecord(telemetry)) {
    missing.push(`devices[${index}].telemetry`);
  } else {
    for (const key of ["utilization_percent", "temperature_c", "power_w"]) {
      if (!isNullableNumber(telemetry[key])) {
        missing.push(`devices[${index}].telemetry.${key}`);
      }
    }
  }

  return {
    ...(value as unknown as DeviceInfo),
    restart_required:
      typeof value.restart_required === "boolean"
        ? value.restart_required
        : false,
  };
}

export function parseDeviceListResponse(value: unknown): DeviceListResponse {
  if (!isRecord(value)) {
    throw new IncompatibleHostError(["devices", "plan_version"]);
  }

  const missing: string[] = [];
  const rawDevices = value.devices;
  if (!Array.isArray(rawDevices)) missing.push("devices");
  if (
    typeof value.plan_version !== "number" ||
    !Number.isInteger(value.plan_version) ||
    value.plan_version < 0
  ) {
    missing.push("plan_version");
  }

  const devices = Array.isArray(rawDevices)
    ? rawDevices
        .map((device, index) => parseDevice(device, index, missing))
        .filter((device): device is DeviceInfo => device !== null)
    : [];

  if (missing.length > 0) throw new IncompatibleHostError(missing);
  return { devices, plan_version: value.plan_version as number };
}

export async function listDevices(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<DeviceListResponse> {
  const value = await apiJsonTo<unknown>(target, "/api/devices", {
    signal: signal ?? null,
  });
  return parseDeviceListResponse(value);
}

export async function setDeviceEnabled(
  target: ApiTarget,
  deviceId: string,
  enabled: boolean,
): Promise<DeviceInfo> {
  const value = await apiJsonTo<unknown>(
    target,
    `/api/devices/${encodeURIComponent(deviceId)}`,
    {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ enabled }),
    },
  );
  const missing: string[] = [];
  const device = parseDevice(value, 0, missing);
  if (!device || missing.length > 0) throw new IncompatibleHostError(missing);
  return device;
}
