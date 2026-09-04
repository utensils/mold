import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiJsonToMock, listDevicesMock, listQueueMock, subscribeMock } = vi.hoisted(() => ({
  apiJsonToMock: vi.fn().mockResolvedValue({ queue_capacity: 8 }),
  listDevicesMock: vi.fn().mockResolvedValue({ devices: [], plan_version: 0 }),
  listQueueMock: vi.fn().mockResolvedValue({ entries: [], plan: null }),
  subscribeMock: vi.fn(),
}));

vi.mock("@studio/api/devices", () => ({
  listDevices: listDevicesMock,
  setDeviceEnabled: vi.fn(),
}));
vi.mock("@studio/api/queuePlan", () => ({
  listQueue: listQueueMock,
  queuePageRequestForCapacity: (capacity: unknown) =>
    typeof capacity === "number" && capacity > 0 ? { limit: capacity } : undefined,
  setQueueDevicePin: vi.fn(),
}));
vi.mock("../../lib/api/deviceEvents", () => ({
  subscribeToDeviceSnapshots: subscribeMock,
}));
vi.mock("../../lib/api/client", () => ({
  apiJsonTo: apiJsonToMock,
}));

import AdvancedSection from "./AdvancedSection.vue";
import { useConnectionStore } from "../../stores/connection";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  apiJsonToMock.mockResolvedValue({ queue_capacity: 8 });
});

describe("AdvancedSection device snapshots", () => {
  function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((next) => {
      resolve = next;
    });
    return { promise, resolve };
  }

  it("bounds its queue-plan snapshot by the server's queue capacity", async () => {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
    };
    connection.status = "ready";

    mount(AdvancedSection, {
      global: {
        stubs: {
          ConfigRowItem: true,
          ConfigSettingRow: true,
          PlacementSection: true,
        },
      },
    });

    await vi.waitFor(() => expect(listQueueMock).toHaveBeenCalled());
    expect(listQueueMock).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:7680", apiKey: "desktop-secret" },
      { limit: 8 },
    );
  });

  it("renders typed host utility work when the local server has no GPUs", async () => {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
    };
    connection.status = "ready";
    listDevicesMock.mockResolvedValueOnce({ devices: [], plan_version: 1 });
    listQueueMock.mockResolvedValueOnce({
      entries: [],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "desktop-cpu-work",
            parent_id: "parent",
            work_kind: "prompt_expansion",
            priority_class: "user",
            queue_rank: 0,
            bypass_count: 0,
            planned_device_id: null,
            planned_lane_kind: "host_utility",
            lane_order: 0,
            estimate_confidence: "low",
          },
        ],
      },
    });

    const wrapper = mount(AdvancedSection, {
      global: {
        stubs: {
          ConfigRowItem: true,
          ConfigSettingRow: true,
          PlacementSection: true,
        },
      },
    });
    await vi.waitFor(() => expect(wrapper.text()).toContain("Prompt expansion"));

    expect(wrapper.get('[data-test="cpu-utility-lane"]').text()).toContain("Machine utility");
    expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(0);
  });

  it("keeps a colliding future typed lane out of the desktop GPU lane", async () => {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
    };
    connection.status = "ready";
    const device = {
      id: "cuda:desktop-stable",
      backend: "cuda",
      ordinal: 0,
      device_kind: "full_gpu",
      nvml_uuid: "GPU-desktop",
      physical_uuid: "GPU-desktop",
      mig_uuid: null,
      mig_parent_uuid: null,
      mig_profile: null,
      name: "Desktop GPU",
      pci_bus_id: null,
      compute_capability: "8.6",
      memory: {
        total_bytes: 24_000_000_000,
        used_bytes: 0,
        mold_used_bytes: 0,
        other_used_bytes: 0,
      },
      telemetry: {
        utilization_percent: 0,
        temperature_c: null,
        power_w: null,
      },
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
    };
    listDevicesMock.mockResolvedValueOnce({ devices: [device], plan_version: 1 });
    listQueueMock.mockResolvedValueOnce({
      entries: [],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "desktop-future-collision",
            parent_id: "parent",
            work_kind: "future_utility",
            priority_class: "user",
            queue_rank: 0,
            bypass_count: 0,
            planned_device_id: device.id,
            planned_lane_kind: "future_accelerator_lane",
            lane_order: 0,
            estimate_confidence: "low",
          },
          {
            work_id: "desktop-typed-device",
            parent_id: "typed-parent",
            work_kind: "generation",
            priority_class: "user",
            queue_rank: 1,
            bypass_count: 0,
            planned_device_id: device.id,
            planned_lane_kind: "device",
            lane_order: 1,
            estimate_confidence: "low",
          },
        ],
      },
    });

    const wrapper = mount(AdvancedSection, {
      global: {
        stubs: {
          ConfigRowItem: true,
          ConfigSettingRow: true,
          PlacementSection: true,
        },
      },
    });
    await vi.waitFor(() => expect(wrapper.text()).toContain("Future utility"));

    expect(wrapper.get('[data-test="other-compute-lane"]').text()).toContain("Future utility");
    expect(wrapper.get('[data-test="device-lane"]').text()).toContain("Generation");
  });

  it("subscribes to the exact authenticated target and refetches on invalidation", async () => {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
    };
    connection.status = "ready";

    const wrapper = mount(AdvancedSection, {
      global: {
        stubs: {
          ConfigRowItem: true,
          ConfigSettingRow: true,
          DevicePanel: true,
          PlacementSection: true,
        },
      },
    });
    await vi.waitFor(() => expect(listDevicesMock).toHaveBeenCalled());
    expect(wrapper.find("device-panel-stub").exists()).toBe(true);
    expect(subscribeMock).toHaveBeenCalledWith(
      {
        baseUrl: "http://127.0.0.1:7680",
        apiKey: "desktop-secret",
      },
      expect.any(AbortSignal),
      expect.any(Function),
    );

    const refresh = subscribeMock.mock.calls[0]?.[2] as () => void;
    listDevicesMock.mockClear();
    listQueueMock.mockClear();
    refresh();
    await vi.waitFor(() => expect(listDevicesMock).toHaveBeenCalledOnce());
    expect(listQueueMock).toHaveBeenCalledOnce();

    wrapper.unmount();
    const signal = subscribeMock.mock.calls[0]?.[1] as AbortSignal;
    expect(signal.aborted).toBe(true);
  });

  it("ignores an older same-target device response after an event refetch", async () => {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
    };
    connection.status = "ready";
    const initial = { devices: [testDevice("INITIAL")], plan_version: 1 };
    listDevicesMock.mockResolvedValueOnce(initial);
    const wrapper = mount(AdvancedSection, {
      global: {
        stubs: {
          ConfigRowItem: true,
          ConfigSettingRow: true,
          PlacementSection: true,
        },
      },
    });
    await vi.waitFor(() => expect(wrapper.text()).toContain("INITIAL"));
    const older = deferred<unknown>();
    const newer = deferred<unknown>();
    listDevicesMock
      .mockImplementationOnce(() => older.promise)
      .mockImplementationOnce(() => newer.promise);
    const refresh = subscribeMock.mock.calls[0]?.[2] as () => void;
    refresh();
    refresh();
    newer.resolve({ devices: [testDevice("NEWER")], plan_version: 3 });
    await vi.waitFor(() => expect(wrapper.text()).toContain("NEWER"));
    older.resolve({ devices: [testDevice("OLDER")], plan_version: 2 });
    await Promise.resolve();
    await Promise.resolve();

    expect(wrapper.text()).toContain("NEWER");
    expect(wrapper.text()).not.toContain("OLDER");
  });
});

function testDevice(name: string) {
  return {
    id: `cuda:${name}`,
    backend: "cuda",
    ordinal: 0,
    device_kind: "full_gpu",
    nvml_uuid: name,
    physical_uuid: name,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24_000_000_000,
      used_bytes: 0,
      mold_used_bytes: 0,
      other_used_bytes: 0,
    },
    telemetry: {
      utilization_percent: 0,
      temperature_c: 30,
      power_w: 20,
    },
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
  };
}
