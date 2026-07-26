import { describe, expect, it } from "vitest";
import type { DeviceInfo } from "../api/devices";
import {
  canMutateDevice,
  deviceActionLabel,
  deviceLifecycleMessage,
  deviceStateLabel,
} from "./deviceLifecycle";

const device = (overrides: Partial<DeviceInfo> = {}): DeviceInfo => ({
  id: "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  backend: "cuda",
  ordinal: 0,
  device_kind: "full_gpu",
  nvml_uuid: "GPU-a",
  physical_uuid: "GPU-a",
  mig_uuid: null,
  mig_parent_uuid: null,
  mig_profile: null,
  name: "NVIDIA RTX 3090",
  pci_bus_id: null,
  compute_capability: "8.6",
  memory: {
    total_bytes: 24_000_000_000,
    used_bytes: 4_000_000_000,
    mold_used_bytes: null,
    other_used_bytes: null,
  },
  telemetry: { utilization_percent: 10, temperature_c: null, power_w: null },
  desired_enabled: true,
  restart_required: false,
  admin_state: "enabled",
  health: "healthy",
  activity: "idle",
  schedulable: true,
  unschedulable_reason: null,
  loaded_models: [],
  active_work_id: null,
  planned_work_ids: [],
  ...overrides,
});

const authoritative = {
  devices: { lifecycle: true, restart_enable: false },
  dispatch: { v2_authoritative: true },
};
const restartOnly = {
  devices: { lifecycle: true, restart_enable: true },
  dispatch: { v2_authoritative: false },
};

describe("shared device lifecycle presentation", () => {
  it("requires both lifecycle support and authoritative V2 for live mutation", () => {
    expect(canMutateDevice(device(), authoritative)).toBe(true);
    expect(canMutateDevice(device(), restartOnly)).toBe(false);
    expect(canMutateDevice(device(), null)).toBe(false);
    expect(
      canMutateDevice(
        device({ admin_state: "startup_excluded" }),
        authoritative,
      ),
    ).toBe(false);
  });

  it("offers only disabled devices restart recovery outside authoritative V2", () => {
    const disabled = device({
      desired_enabled: false,
      admin_state: "disabled",
      schedulable: false,
    });
    expect(canMutateDevice(disabled, restartOnly)).toBe(true);
    expect(deviceActionLabel(disabled, restartOnly)).toBe("Enable on restart");
    expect(deviceLifecycleMessage(restartOnly)).toContain(
      "next server restart",
    );
  });

  it("presents a persisted restart recovery as pending instead of unavailable", () => {
    const pending = device({
      restart_required: true,
      health: "unavailable",
      schedulable: false,
      unschedulable_reason: "device_unavailable",
    });
    expect(canMutateDevice(pending, restartOnly)).toBe(false);
    expect(deviceActionLabel(pending, restartOnly)).toBe("Enabled on restart");
    expect(deviceStateLabel(pending)).toBe("Restart required");
  });
});
