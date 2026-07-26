import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { listDevicesMock, listQueueMock, subscribeMock } = vi.hoisted(() => ({
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
  setQueueDevicePin: vi.fn(),
}));
vi.mock("../../lib/api/deviceEvents", () => ({
  subscribeToDeviceSnapshots: subscribeMock,
}));

import AdvancedSection from "./AdvancedSection.vue";
import { useConnectionStore } from "../../stores/connection";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("AdvancedSection device snapshots", () => {
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
});
