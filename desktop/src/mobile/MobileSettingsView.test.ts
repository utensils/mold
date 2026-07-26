import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MobileSettingsView from "./MobileSettingsView.vue";

const { apiJsonTo, openExternalMock, setDeviceEnabled } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  openExternalMock: vi.fn(),
  setDeviceEnabled: vi.fn(),
}));
vi.mock("../lib/openExternal", () => ({ openExternal: openExternalMock }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo,
}));
vi.mock("@studio/api/devices", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/devices")>()),
  setDeviceEnabled,
}));

beforeEach(() => {
  openExternalMock.mockClear();
  apiJsonTo.mockReset();
  setDeviceEnabled.mockReset().mockResolvedValue(undefined);
});

describe("MobileSettingsView", () => {
  it("offers accessible theme choices and emits immediate updates", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "system", themeFamily: "mold", autoSavePhotos: true },
        hostCount: 2,
        appVersion: "0.18.0",
      },
    });

    expect(wrapper.findAll("fieldset")).toHaveLength(3);
    expect(wrapper.text()).toContain("Change the chrome without changing the color of your prints");
    expect(wrapper.text()).toContain("2 hosts saved");
    expect(wrapper.text()).toContain("0.18.0");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
    expect(wrapper.text()).not.toMatch(/equal (project )?owners/i);

    await wrapper.get('input[name="mobile-theme-family"][value="safelight"]').setValue(true);
    await wrapper.get('input[name="mobile-theme-appearance"][value="light"]').setValue(true);
    await wrapper.get('input[name="mobile-auto-save-photos"]').setValue(false);

    expect(wrapper.emitted("update")).toEqual([
      [{ themeFamily: "safelight" }],
      [{ theme: "light" }],
      [{ autoSavePhotos: false }],
    ]);
  });

  it("routes host management through an explicit action", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "dark", themeFamily: "mold", autoSavePhotos: true },
        hostCount: 0,
        appVersion: "Development build",
      },
    });

    expect(wrapper.text()).toContain("No hosts saved");
    await wrapper.get(".mobile-settings-manage").trigger("click");
    expect(wrapper.emitted("manage-hosts")).toHaveLength(1);
  });

  it("opens the public privacy policy from About", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "system", themeFamily: "safelight", autoSavePhotos: true },
        hostCount: 1,
        appVersion: "0.20.2",
      },
    });

    await wrapper.get("[data-test='mobile-privacy-policy']").trigger("click");

    expect(openExternalMock).toHaveBeenCalledOnce();
    expect(openExternalMock).toHaveBeenCalledWith("https://utensils.io/mold/privacy");
  });

  it("controls the selected host device with its Keychain-supplied route", async () => {
    const host = {
      id: "studio-id",
      name: "Studio",
      baseUrl: "http://studio.tailnet.ts.net:7680",
      apiKey: "studio-secret",
      hostname: "studio",
      version: "0.20.2",
      online: true,
    };
    const device = {
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
      telemetry: {
        utilization_percent: 10,
        temperature_c: null,
        power_w: null,
      },
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
    let load = 0;
    apiJsonTo.mockImplementation(async () => {
      load += 1;
      return {
        devices: [
          load === 1
            ? device
            : {
                ...device,
                desired_enabled: false,
                admin_state: "disabled",
                schedulable: false,
                unschedulable_reason: "device_disabled",
              },
        ],
        plan_version: load,
      };
    });
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "system", themeFamily: "mold", autoSavePhotos: true },
        hostCount: 1,
        appVersion: "0.20.2",
        host,
      },
    });
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='mobile-settings-devices']").exists()).toBe(true),
    );

    await wrapper.get("[data-test='mobile-settings-device-toggle-0']").trigger("click");
    await vi.waitFor(() => expect(apiJsonTo).toHaveBeenCalledTimes(2));

    expect(setDeviceEnabled).toHaveBeenCalledWith(
      { baseUrl: host.baseUrl, apiKey: host.apiKey },
      device.id,
      false,
    );
    expect(wrapper.get("[data-test='mobile-settings-device-toggle-0']").text()).toBe("Enable");
  });
});
