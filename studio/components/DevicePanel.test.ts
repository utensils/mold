import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
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
  it("separates a remote Metal budget from installed shared RAM", () => {
    const metal = device(0, { backend: "metal", device_kind: "metal" });
    metal.memory.metal_memory = {
      wired_limit: { mode: "explicit", mib: 16384 },
      physical_bytes: 48 * 1024 ** 3,
      available_host_bytes: 32 * 1024 ** 3,
      recommended_bytes: 37 * 1024 ** 3,
      allocated_bytes: 4 * 1024 ** 3,
      effective_capacity_bytes: 16 * 1024 ** 3,
      allocation_headroom_bytes: 12 * 1024 ** 3,
      error: null,
    };
    const wrapper = mount(DevicePanel, { props: { devices: [metal] } });
    expect(wrapper.text()).toContain("Metal capacity 16.0 GiB");
    expect(wrapper.text()).toContain("Allocation headroom 12.0 GiB");
    expect(wrapper.text()).toContain("Kernel limit 16384 MiB");
    expect(wrapper.text()).toContain("Shared RAM");
    expect(wrapper.text()).not.toContain("sudo");
    metal.memory.metal_memory.effective_capacity_bytes = null;
    metal.memory.metal_memory.allocation_headroom_bytes = null;
    metal.memory.metal_memory.error = "Cannot read Metal device";
    const failed = mount(DevicePanel, { props: { devices: [metal] } });
    expect(failed.text()).toContain("Metal capacity unavailable");
    expect(failed.text()).toContain("Cannot read Metal device");
  });

  for (const count of [1, 2, 8, 64]) {
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

  it("separates running work from the queued lane without exposing internal ids", () => {
    const gpu = device(0, { active_work_id: "active-internal-id" });
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [gpu],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "active-internal-id",
              parent_id: "active-parent",
              work_kind: "video_generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              planned_device_id: gpu.id,
              planned_lane_kind: "device",
              lane_order: 0,
              estimate_confidence: "high",
            },
            {
              work_id: "queued-internal-id",
              parent_id: "queued-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 1,
              bypass_count: 0,
              planned_device_id: gpu.id,
              planned_lane_kind: "device",
              lane_order: 1,
              estimate_confidence: "low",
            },
          ],
        },
      },
    });

    expect(wrapper.get(".device-card__line-label").text()).toBe("Running now");
    expect(wrapper.get(".device-card__line-value").text()).toBe(
      "Video generation",
    );
    expect(
      wrapper
        .get('[data-test="device-lane"]')
        .findAll("li")
        .map((row) => row.text()),
    ).toHaveLength(1);
    expect(wrapper.get('[data-test="device-lane"]').text()).toContain(
      "Next · Generation",
    );
    expect(wrapper.text()).not.toContain("active-internal-id");
    expect(wrapper.text()).not.toContain("queued-internal-id");
    expect(wrapper.find('[title="active-internal-id"]').exists()).toBe(false);
    expect(wrapper.find('[title="queued-internal-id"]').exists()).toBe(false);
  });

  it("disables every device with an in-flight mutation", () => {
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [device(0), device(1), device(2)],
        mutable: true,
        busyDeviceIds: [device(0).id, device(2).id],
      },
    });

    expect(
      wrapper.get('[data-test="device-toggle-0"]').attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get('[data-test="device-toggle-1"]').attributes("disabled"),
    ).toBeUndefined();
    expect(
      wrapper.get('[data-test="device-toggle-2"]').attributes("disabled"),
    ).toBeDefined();
  });

  it("hides lifecycle mutations when the host does not advertise them", () => {
    const wrapper = mount(DevicePanel, {
      props: { devices: [device(0)], mutable: false },
    });

    expect(wrapper.find('[data-test="device-toggle-0"]').exists()).toBe(false);
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

  it("renders future typed blocked reasons without a client allowlist", () => {
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [device(0)],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "job-future",
              parent_id: "job-future",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              estimate_confidence: "low",
              blocked_reason: "thermal_throttle",
            },
          ],
        },
      },
    });

    expect(wrapper.get(".device-panel__blocked").text()).toContain(
      "thermal throttle",
    );
  });

  it("keeps an ordered CPU utility lane visible at zero, one, and many GPUs", () => {
    for (const count of [0, 1, 8]) {
      const wrapper = mount(DevicePanel, {
        props: {
          devices: Array.from({ length: count }, (_, index) => device(index)),
          mutable: true,
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "optimized",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "future-parent-2",
                parent_id: "parent-2",
                work_kind: "future_utility",
                priority_class: "user",
                queue_rank: 1,
                bypass_count: 0,
                planned_device_id: null,
                planned_lane_kind: "host_utility",
                lane_order: 2,
                estimated_finish_unix_ms: Date.now() + 7_000,
                estimate_confidence: "low",
                activity_phase: "queued",
              },
              {
                work_id: "expand-parent-1",
                parent_id: "parent-1",
                work_kind: "prompt_expansion",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                planned_device_id: null,
                planned_lane_kind: "host_utility",
                lane_order: 1,
                estimated_finish_unix_ms: Date.now() + 5_000,
                estimate_confidence: "medium",
                activity_phase: "cpu",
              },
            ],
          },
        },
      });

      expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(count);
      expect(
        wrapper
          .get('[data-test="device-panel"]')
          .classes()
          .includes("device-panel--compact"),
      ).toBe(count === 0);
      expect(
        wrapper.get('[data-test="device-panel"]').attributes("data-lane-count"),
      ).toBe(String(count + 1));
      const utility = wrapper.get('[data-test="cpu-utility-lane"]');
      expect(utility.text()).toContain("Host utility");
      expect(utility.text()).toContain("CPU");
      expect(utility.find("button").exists()).toBe(false);
      if (count === 0) {
        expect(wrapper.text()).not.toContain("Live GPU controls");
      }
      expect(
        utility.findAll(".device-card__work").map((row) => row.text()),
      ).toEqual(["Next · Prompt expansion", "After that · Future utility"]);
      wrapper.unmount();
    }
  });

  it("keeps future typed non-device lanes visible without parsing their IDs", () => {
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "future-lane-work",
              parent_id: "parent",
              work_kind: "future_utility",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              planned_device_id: null,
              planned_lane_kind: "future_host_lane",
              lane_order: 0,
              estimate_confidence: "low",
              activity_phase: "queued",
            },
          ],
        },
      },
    });

    expect(wrapper.get('[data-test="other-compute-lane"]').text()).toContain(
      "Next · Future utility",
    );
  });

  it("keeps typed collisions authoritative beside a typed GPU lane on the same device", () => {
    const visible = device(0);
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [visible],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "future-collision",
              parent_id: "parent",
              work_kind: "future_utility",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              planned_device_id: visible.id,
              planned_lane_kind: "future_accelerator_lane",
              lane_order: 0,
              estimate_confidence: "low",
              activity_phase: "queued",
            },
            {
              work_id: "typed-device-work",
              parent_id: "typed-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 1,
              bypass_count: 0,
              planned_device_id: visible.id,
              planned_lane_kind: "device",
              lane_order: 1,
              estimate_confidence: "low",
              activity_phase: "queued",
            },
          ],
        },
      },
    });

    expect(wrapper.get('[data-test="other-compute-lane"]').text()).toContain(
      "Future utility",
    );
    expect(wrapper.get('[data-test="device-lane"]').text()).toContain(
      "Generation",
    );
    expect(
      wrapper.get('[data-test="other-compute-lane"]').text(),
    ).not.toContain("HOST");
  });

  it("partitions unmatched device work into an explicit unknown lane", () => {
    const visible = device(0);
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [visible],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "known-device-work",
              parent_id: "known-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              planned_device_id: visible.id,
              planned_lane_kind: "device",
              lane_order: 0,
              estimate_confidence: "high",
            },
            {
              work_id: "missing-device-work",
              parent_id: "missing-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 1,
              bypass_count: 0,
              planned_device_id: "cuda:not-in-this-snapshot",
              planned_lane_kind: "device",
              lane_order: 1,
              estimate_confidence: "medium",
            },
            {
              work_id: "unplaced-waiting-work",
              parent_id: "unplaced-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 2,
              bypass_count: 0,
              planned_device_id: null,
              lane_order: 2,
              estimate_confidence: "low",
              reason: "not_ready",
              blocked_reason: "dependency_wait",
            },
            {
              work_id: "future-lane-missing-device",
              parent_id: "future-parent",
              work_kind: "future_utility",
              priority_class: "user",
              queue_rank: 3,
              bypass_count: 0,
              planned_device_id: "cuda:future-device-not-in-snapshot",
              planned_lane_kind: "future_accelerator_lane",
              lane_order: 3,
              estimate_confidence: "low",
            },
            {
              work_id: "host-utility-work",
              parent_id: "utility-parent",
              work_kind: "prompt_expansion",
              priority_class: "user",
              queue_rank: 4,
              bypass_count: 0,
              planned_device_id: null,
              planned_lane_kind: "host_utility",
              lane_order: 0,
              estimate_confidence: "low",
            },
            {
              work_id: "blocked-unassigned-work",
              parent_id: "blocked-parent",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 5,
              bypass_count: 0,
              planned_device_id: null,
              planned_lane_kind: "device",
              lane_order: 3,
              estimate_confidence: "low",
              blocked_reason: "no_capacity",
            },
          ],
        },
      },
    });

    const known = wrapper.get('[data-test="device-lane"]');
    const unknown = wrapper.get('[data-test="unassigned-device-lane"]');
    const utility = wrapper.get('[data-test="cpu-utility-lane"]');
    const blocked = wrapper.get('[data-test="blocked-work"]');

    expect(known.text()).toContain("Next · Generation");
    expect(unknown.text()).toContain("Unassigned / unknown device");
    expect(unknown.text()).toContain("Generation");
    expect(unknown.text()).toContain("Future utility");
    // Not yet placed on any lane (no lane kind, no device) while it waits on
    // a dependency: ordinary queued work, never an "unknown device" row.
    expect(wrapper.text()).not.toContain("unplaced-waiting-work");
    expect(utility.text()).toContain("Prompt expansion");
    expect(blocked.text()).toContain("Generation · no capacity");
    expect(blocked.text()).not.toContain("blocked-unassigned-work");
    for (const workId of [
      "known-device-work",
      "missing-device-work",
      "future-lane-missing-device",
      "host-utility-work",
    ]) {
      expect(wrapper.text()).not.toContain(workId);
    }
  });

  it("does not invent a host utility card without CPU-planned work", () => {
    const wrapper = mount(DevicePanel, {
      props: { devices: [device(0)], plan: null },
    });

    expect(wrapper.find('[data-test="cpu-utility-lane"]').exists()).toBe(false);
  });

  it("shows blocked host utility work only in the blocked recovery section", () => {
    const wrapper = mount(DevicePanel, {
      props: {
        devices: [],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "blocked-cpu-work",
              parent_id: "parent",
              work_kind: "prompt_expansion",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              planned_device_id: null,
              planned_lane_kind: "host_utility",
              lane_order: 0,
              estimate_confidence: "low",
              blocked_reason: "host_ram",
            },
          ],
        },
      },
    });

    expect(wrapper.find('[data-test="cpu-utility-lane"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="blocked-work"]').text()).toContain(
      "Prompt expansion · host ram",
    );
  });

  it("updates ETA and replan countdowns while the snapshot is otherwise static", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(10_000);
    try {
      const wrapper = mount(DevicePanel, {
        props: {
          devices: [device(0)],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "debouncing",
            dirty_since_unix_ms: 10_000,
            next_replan_at_unix_ms: 13_500,
            work_items: [
              {
                work_id: "job-1",
                parent_id: "job-1",
                work_kind: "generation",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                planned_device_id: device(0).id,
                planned_lane_kind: "device",
                lane_order: 0,
                estimated_finish_unix_ms: 14_500,
                estimate_confidence: "high",
              },
            ],
          },
        },
      });
      expect(wrapper.get('[data-test="replan-countdown"]').text()).toContain(
        "Updating order in 4s",
      );
      expect(wrapper.get('[data-test="device-lane"]').text()).toContain(
        "under a minute",
      );

      await vi.advanceTimersByTimeAsync(1_100);

      expect(wrapper.get('[data-test="replan-countdown"]').text()).toContain(
        "Updating order in 3s",
      );
      expect(wrapper.get('[data-test="device-lane"]').text()).toContain(
        "under a minute",
      );
      wrapper.unmount();
    } finally {
      vi.useRealTimers();
    }
  });
});
