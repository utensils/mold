import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { DeviceInfo } from "../api/devices";
import DevicePanel from "./DevicePanel.vue";

function device(index: number, patch: Partial<DeviceInfo> = {}): DeviceInfo {
  return {
    id: `cuda:${String(index).padStart(32, "0")}`,
    backend: "cuda",
    ordinal: index,
    device_kind: "full_gpu",
    nvml_uuid: `GPU-${index}`,
    physical_uuid: `GPU-${index}`,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name: `GPU ${index}`,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24 * 1024 ** 3,
      used_bytes: index * 1024 ** 3,
      mold_used_bytes: null,
      other_used_bytes: null,
    },
    telemetry: {
      utilization_percent: index * 10,
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
    ...patch,
  };
}

describe("DevicePanel", () => {
  for (const count of [1, 2, 8]) {
    it(`renders all ${count} devices without a cardinality ceiling`, () => {
      const wrapper = mount(DevicePanel, {
        props: { devices: Array.from({ length: count }, (_, i) => device(i)) },
      });
      expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(count);
      expect(
        wrapper
          .get('[data-test="device-panel"]')
          .attributes("data-device-count"),
      ).toBe(String(count));
    });
  }

  it("renders disabled, draining, unavailable, and MIG identity explicitly", () => {
    const wrapper = mount(DevicePanel, {
      props: {
        mutable: true,
        devices: [
          device(0, { desired_enabled: false, admin_state: "disabled" }),
          device(1, { desired_enabled: false, admin_state: "draining" }),
          device(2, { health: "unavailable", schedulable: false }),
          device(3, {
            device_kind: "mig",
            mig_uuid: "MIG-one",
            mig_parent_uuid: "GPU-parent",
            mig_profile: "1g.10gb",
          }),
        ],
      },
    });
    expect(wrapper.text()).toContain("disabled");
    expect(wrapper.text()).toContain("Finishing current work");
    expect(wrapper.text()).toContain("unavailable");
    expect(wrapper.text()).toContain("MIG 1g.10gb");
  });

  it("keeps a single device compact and emits stable-id mutation", async () => {
    const wrapper = mount(DevicePanel, {
      props: { devices: [device(0)], mutable: true },
    });
    expect(wrapper.get('[data-test="device-panel"]').classes()).toContain(
      "device-panel--compact",
    );
    await wrapper.get('[data-test="device-toggle-0"]').trigger("click");
    expect(wrapper.emitted("toggle")?.[0]).toEqual([device(0).id, false]);
  });

  it("offers Auto and re-enable recovery for a disabled hard pin", async () => {
    const pinned = device(0, {
      desired_enabled: false,
      admin_state: "disabled",
      schedulable: false,
    });
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [pinned],
        mutable: true,
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "job-1",
              parent_id: "job-1",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              hard_pinned_device_id: pinned.id,
              estimate_confidence: "low",
              reason: "hard_pin_unavailable",
            },
          ],
        },
      },
    });
    const buttons = wrapper.findAll(".device-panel__blocked-action");
    expect(buttons.map((button) => button.text())).toEqual([
      "Use Auto",
      "Re-enable",
    ]);
    await buttons[0]!.trigger("click");
    await buttons[1]!.trigger("click");
    expect(wrapper.emitted("unpin")?.[0]).toEqual(["job-1"]);
    expect(wrapper.emitted("toggle")?.[0]).toEqual([pinned.id, true]);
  });
});
