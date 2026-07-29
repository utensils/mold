import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { setDeviceEnabled } = vi.hoisted(() => ({ setDeviceEnabled: vi.fn() }));
vi.mock("@studio/api/devices", () => ({ setDeviceEnabled }));

import DeviceSettingsPanel from "./DeviceSettingsPanel.vue";
import type { DeviceInfo } from "@studio/api/devices";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";

const device: DeviceInfo = {
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
  admin_state: "enabled",
  health: "healthy",
  activity: "idle",
  schedulable: true,
  unschedulable_reason: null,
  loaded_models: [],
  active_work_id: null,
  planned_work_ids: [],
};

describe("DeviceSettingsPanel", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    setDeviceEnabled.mockReset().mockResolvedValue(undefined);
    const connection = useConnectionStore();
    connection.info = {
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-secret",
      mode: "local",
    };
    connection.status = "ready";
    const hosts = useHostsStore();
    hosts.telemetry.local = {
      queueDepth: 0,
      queueCapacity: 8,
      version: "0.20.2",
      devices: [device],
    };
    hosts.capabilities.local = {
      gallery: { can_delete: true },
      devices: { available: true, lifecycle: true, restart_enable: false },
      dispatch: { active_mode: "v2", v2_authoritative: true, observes_v2_decisions: false },
    };
    vi.spyOn(hosts, "refresh").mockResolvedValue(undefined);
  });

  it("targets this device with its local API key", async () => {
    const wrapper = mount(DeviceSettingsPanel);
    expect(wrapper.text()).toContain("NVIDIA RTX 3090");

    await wrapper.get("[data-test='settings-device-toggle-0']").trigger("click");
    await flushPromises();

    expect(setDeviceEnabled).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-secret" },
      device.id,
      false,
    );
    expect(useHostsStore().refresh).toHaveBeenCalledOnce();
  });

  it("offers only restart recovery when live lifecycle is unavailable", async () => {
    const hosts = useHostsStore();
    hosts.telemetry.local!.devices = [
      { ...device, desired_enabled: false, admin_state: "disabled", schedulable: false },
    ];
    hosts.capabilities.local = {
      gallery: { can_delete: true },
      devices: { available: true, lifecycle: false, restart_enable: true },
      dispatch: {
        active_mode: "legacy",
        v2_authoritative: false,
        observes_v2_decisions: false,
      },
    };
    const wrapper = mount(DeviceSettingsPanel);
    const button = wrapper.get("[data-test='settings-device-toggle-0']");
    expect(button.text()).toBe("Enable on restart");
    await button.trigger("click");
    await flushPromises();
    expect(setDeviceEnabled).toHaveBeenCalledWith(expect.any(Object), device.id, true);
  });
});
