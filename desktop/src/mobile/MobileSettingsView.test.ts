import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../lib/api/client";
import MobileSettingsView from "./MobileSettingsView.vue";

const { apiJsonTo, listQueueMock, openExternalMock, setDeviceEnabled, subscribeMock } = vi.hoisted(
  () => ({
    apiJsonTo: vi.fn(),
    listQueueMock: vi.fn(),
    openExternalMock: vi.fn(),
    setDeviceEnabled: vi.fn(),
    subscribeMock: vi.fn(),
  }),
);
vi.mock("../lib/openExternal", () => ({ openExternal: openExternalMock }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo,
}));
vi.mock("@studio/api/devices", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/devices")>()),
  setDeviceEnabled,
}));
vi.mock("@studio/api/queuePlan", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/queuePlan")>()),
  listQueue: listQueueMock,
}));
vi.mock("../lib/api/deviceEvents", () => ({
  subscribeToDeviceSnapshots: subscribeMock,
}));

beforeEach(() => {
  openExternalMock.mockClear();
  apiJsonTo.mockReset();
  listQueueMock.mockReset().mockResolvedValue({ entries: [], plan: null });
  setDeviceEnabled.mockReset().mockResolvedValue(undefined);
  subscribeMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("MobileSettingsView", () => {
  function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((next) => {
      resolve = next;
    });
    return { promise, resolve };
  }

  it("offers accessible theme choices and emits immediate updates", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
        hostCount: 2,
        appVersion: "0.18.0",
      },
    });

    expect(wrapper.findAll("fieldset")).toHaveLength(4);
    expect(wrapper.text()).toContain("Change the chrome without changing the color of your prints");
    expect(wrapper.text()).toContain("2 hosts saved");
    expect(wrapper.text()).toContain("0.18.0");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
    expect(wrapper.text()).not.toMatch(/equal (project )?owners/i);

    await wrapper.get('input[name="mobile-theme-family"][value="safelight"]').setValue(true);
    await wrapper.get('input[name="mobile-theme-appearance"][value="light"]').setValue(true);
    await wrapper.get('input[name="mobile-auto-save-photos"]').setValue(false);
    await wrapper.get('input[name="mobile-auto-tag-title"]').setValue(false);

    expect(wrapper.emitted("update")).toEqual([
      [{ themeFamily: "safelight" }],
      [{ theme: "light" }],
      [{ autoSavePhotos: false }],
      [{ autoTagTitle: false }],
    ]);
  });

  it("names the auto-tag preference as the removable default it files", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: false,
        },
        hostCount: 1,
        appVersion: "0.23.0",
      },
    });

    const toggle = wrapper.get('input[name="mobile-auto-tag-title"]');
    expect((toggle.element as HTMLInputElement).checked).toBe(false);
    expect(wrapper.get("[data-test='mobile-settings-library']").text()).toContain(
      "Tag new prints with their title",
    );

    await toggle.setValue(true);
    expect(wrapper.emitted("update")).toEqual([[{ autoTagTitle: true }]]);
  });

  it("routes host management through an explicit action", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "dark", themeFamily: "mold", autoSavePhotos: true, autoTagTitle: true },
        hostCount: 0,
        appVersion: "Development build",
      },
    });

    expect(wrapper.text()).toContain("No hosts saved");
    await wrapper.get(".mobile-settings-manage").trigger("click");
    expect(wrapper.emitted("manage-hosts")).toHaveLength(1);
  });

  it("keeps CPU utility work visible when a host reports no GPUs", async () => {
    const host = {
      id: "cpu-host",
      name: "CPU Host",
      baseUrl: "http://cpu-host:7680",
      apiKey: "secret",
      hostname: "cpu-host",
      version: "0.20.2",
      online: true,
    };
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") return { devices: [], plan_version: 1 };
      if (path === "/api/capabilities") {
        return {
          devices: { available: true, lifecycle: false },
          dispatch: { active_mode: "v2", v2_authoritative: true },
        };
      }
      throw new Error(`Unexpected path ${path}`);
    });
    listQueueMock.mockResolvedValue({
      entries: [],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "mobile-cpu-work",
            parent_id: "parent",
            work_kind: "post_upscale",
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

    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
        hostCount: 1,
        appVersion: "0.20.2",
        host,
      },
    });
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='mobile-settings-devices']").exists()).toBe(true),
    );

    expect(wrapper.text()).toContain("Compute devices");
    expect(wrapper.text()).toContain("Host utility");
    expect(wrapper.text()).toContain("mobile-cpu-work");
  });

  it("keeps a queue-plan compute lane visible when device inventory is unavailable", async () => {
    const host = {
      id: "older-device-api-host",
      name: "Older Host",
      baseUrl: "http://older-host:7680",
      apiKey: "secret",
      hostname: "older-host",
      version: "0.20.2",
      online: true,
    };
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") throw new Error("device inventory unavailable");
      if (path === "/api/capabilities") return {};
      throw new Error(`Unexpected path ${path}`);
    });
    listQueueMock.mockResolvedValue({
      entries: [],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "future-mobile-lane",
            parent_id: "parent",
            work_kind: "future_utility",
            priority_class: "user",
            queue_rank: 0,
            bypass_count: 0,
            planned_device_id: null,
            planned_lane_kind: "future_host_lane",
            lane_order: 0,
            estimate_confidence: "low",
          },
        ],
      },
    });

    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
        hostCount: 1,
        appVersion: "0.20.2",
        host,
      },
    });
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='mobile-settings-devices']").exists()).toBe(true),
    );

    expect(wrapper.get("[data-test='other-compute-lane']").text()).toContain("future-mobile-lane");
  });

  it("opens the public privacy policy from About", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "safelight",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
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
    let load = 0;
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/capabilities")
        return {
          devices: { available: true, lifecycle: true, restart_enable: false },
          dispatch: { active_mode: "v2", v2_authoritative: true },
        };
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
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
        hostCount: 1,
        appVersion: "0.20.2",
        host,
      },
    });
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='mobile-settings-devices']").exists()).toBe(true),
    );

    await wrapper.get("[data-test='device-toggle-0']").trigger("click");
    await vi.waitFor(() => expect(apiJsonTo).toHaveBeenCalledTimes(4));

    expect(setDeviceEnabled).toHaveBeenCalledWith(
      { baseUrl: host.baseUrl, apiKey: host.apiKey },
      device.id,
      false,
    );
    await vi.waitFor(() =>
      expect(wrapper.get("[data-test='device-toggle-0']").text()).toBe("Enable"),
    );
    expect(subscribeMock).toHaveBeenCalledWith(
      { baseUrl: host.baseUrl, apiKey: host.apiKey },
      expect.any(AbortSignal),
      expect.any(Function),
    );

    wrapper.unmount();
    const signal = subscribeMock.mock.calls[0]?.[1] as AbortSignal;
    expect(signal.aborted).toBe(true);
  });

  it("ignores an older same-host device bootstrap after an event refresh", async () => {
    const host = {
      id: "studio-id",
      name: "Studio",
      baseUrl: "http://studio:7680",
      apiKey: "secret",
      hostname: "studio",
      version: "0.20.2",
      online: true,
    };
    const enabled = {
      ...deviceWireForRace(),
      desired_enabled: true,
      admin_state: "enabled",
      schedulable: true,
    };
    const disabled = {
      ...enabled,
      desired_enabled: false,
      admin_state: "disabled",
      schedulable: false,
      unschedulable_reason: "device_disabled",
    };
    const older = deferred<unknown>();
    const newer = deferred<unknown>();
    let deviceCall = 0;
    apiJsonTo.mockImplementation((_target, path) => {
      if (path === "/api/devices") return deviceCall++ === 0 ? older.promise : newer.promise;
      if (path === "/api/capabilities")
        return Promise.resolve({
          devices: { available: true, lifecycle: true, restart_enable: false },
          dispatch: { active_mode: "v2", v2_authoritative: true },
        });
      if (path === "/api/queue") return Promise.resolve({ entries: [], plan: null });
      throw new Error(`unexpected ${path}`);
    });
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: {
          theme: "system",
          themeFamily: "mold",
          autoSavePhotos: true,
          autoTagTitle: true,
        },
        hostCount: 1,
        appVersion: "0.20.2",
        host,
      },
    });
    await vi.waitFor(() => expect(subscribeMock).toHaveBeenCalledOnce());
    const refresh = subscribeMock.mock.calls[0]?.[2] as () => void;
    refresh();
    older.resolve({ devices: [enabled], plan_version: 1 });
    await Promise.resolve();
    await Promise.resolve();
    newer.resolve({ devices: [disabled], plan_version: 2 });
    await vi.waitFor(() =>
      expect(wrapper.get("[data-test='device-toggle-0']").text()).toBe("Enable"),
    );

    expect(wrapper.get("[data-test='device-toggle-0']").text()).toBe("Enable");
  });

  it("polls devices every five seconds when the SSE stream is absent or stops delivering", async () => {
    vi.useFakeTimers();
    const host = mobileHost("poll-host");
    let deviceCalls = 0;
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        return {
          devices: [deviceWireForRace(deviceCalls === 1 ? "INITIAL GPU" : "POLLED GPU")],
          plan_version: deviceCalls,
        };
      }
      if (path === "/api/capabilities") return deviceCapabilities();
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();
    expect(wrapper.text()).toContain("INITIAL GPU");

    // The mocked stream never calls onOpen or emits an invalidation. Polling
    // must still recover from an absent or subsequently dropped stream.
    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    expect(deviceCalls).toBe(2);
    expect(wrapper.text()).toContain("POLLED GPU");
    wrapper.unmount();
  });

  it("surfaces a rejected device fetch and clears the error after a poll recovers", async () => {
    vi.useFakeTimers();
    const host = mobileHost("recovering-host");
    let deviceCalls = 0;
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        if (deviceCalls === 1) {
          throw new ApiError("device inventory crashed", 500);
        }
        return {
          devices: [deviceWireForRace("RECOVERED GPU")],
          plan_version: 2,
        };
      }
      if (path === "/api/capabilities") return deviceCapabilities();
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();

    const alert = wrapper.get("[role='alert']");
    expect(alert.text()).toContain("Couldn’t refresh compute devices");
    expect(alert.text()).toContain("device inventory crashed");
    expect(alert.text()).toContain("try again automatically");

    // The 5 s retry timer may not be registered yet when this advance runs —
    // the failed fetch's promise chain schedules it — so advance one full
    // poll period per waitFor attempt until the recovery poll has landed
    // (#1334; a single advance + flush raced this under CI load).
    await vi.waitFor(
      async () => {
        await vi.advanceTimersByTimeAsync(5_000);
        await flushPromises();
        expect(wrapper.find("[role='alert']").exists()).toBe(false);
        expect(wrapper.text()).toContain("RECOVERED GPU");
      },
      { timeout: 10_000 },
    );
    expect(deviceCalls).toBeGreaterThanOrEqual(2);
    wrapper.unmount();
  });

  it("keeps the deliberate older-server state quiet when capabilities omit the device API", async () => {
    const host = mobileHost("legacy-host");
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") throw new ApiError("Not Found", 404);
      if (path === "/api/capabilities") return {};
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();

    expect(wrapper.find("[role='alert']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-settings-devices']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("keeps polling after a malformed capabilities payload", async () => {
    vi.useFakeTimers();
    const host = mobileHost("malformed-capabilities-host");
    let deviceCalls = 0;
    apiJsonTo.mockImplementation(async (_target, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        if (deviceCalls === 2) {
          throw new ApiError("device inventory unavailable", 500);
        }
        return {
          devices: [
            deviceWireForRace(deviceCalls === 1 ? "INITIAL GPU" : "RECOVERED AFTER MALFORMED"),
          ],
          plan_version: deviceCalls,
        };
      }
      if (path === "/api/capabilities") {
        return deviceCalls === 2 ? null : deviceCapabilities();
      }
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();
    expect(wrapper.text()).toContain("INITIAL GPU");

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    expect(wrapper.get("[role='alert']").text()).toContain("device inventory unavailable");

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    expect(deviceCalls).toBe(3);
    expect(wrapper.find("[role='alert']").exists()).toBe(false);
    expect(wrapper.text()).toContain("RECOVERED AFTER MALFORMED");
    wrapper.unmount();
  });

  it("coalesces overlapping poll and SSE refreshes", async () => {
    vi.useFakeTimers();
    const host = mobileHost("slow-host");
    const slowPoll = deferred<unknown>();
    let deviceCalls = 0;
    apiJsonTo.mockImplementation((_target, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        if (deviceCalls === 1) {
          return Promise.resolve({
            devices: [deviceWireForRace("INITIAL GPU")],
            plan_version: 1,
          });
        }
        if (deviceCalls === 2) return slowPoll.promise;
        return Promise.resolve({
          devices: [deviceWireForRace("COALESCED GPU")],
          plan_version: 3,
        });
      }
      if (path === "/api/capabilities") return Promise.resolve(deviceCapabilities());
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();
    const refresh = subscribeMock.mock.calls[0]?.[2] as () => void;

    await vi.advanceTimersByTimeAsync(5_000);
    refresh();
    refresh();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(deviceCalls).toBe(2);

    slowPoll.resolve({
      devices: [deviceWireForRace("SLOW GPU")],
      plan_version: 2,
    });
    await flushPromises();

    expect(deviceCalls).toBe(3);
    await flushPromises();
    expect(wrapper.text()).toContain("COALESCED GPU");
    expect(wrapper.text()).not.toContain("SLOW GPU");
    wrapper.unmount();
  });

  it("retargets polling on host identity changes and ignores stale-host results", async () => {
    vi.useFakeTimers();
    const first = mobileHost("first-host");
    const second = {
      ...mobileHost("second-host"),
      baseUrl: "http://second-host.tailnet.ts.net:7680",
      apiKey: "second-secret",
    };
    const stale = deferred<unknown>();
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/devices") {
        if (target.baseUrl === first.baseUrl) return stale.promise;
        return Promise.resolve({
          devices: [deviceWireForRace("SECOND GPU")],
          plan_version: 1,
        });
      }
      if (path === "/api/capabilities") return Promise.resolve(deviceCapabilities());
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(first);
    await flushPromises();
    const firstSignal = subscribeMock.mock.calls[0]?.[1] as AbortSignal;

    await wrapper.setProps({ host: second });
    await flushPromises();
    expect(firstSignal.aborted).toBe(true);
    expect(wrapper.text()).toContain("SECOND GPU");

    stale.resolve({
      devices: [deviceWireForRace("STALE FIRST GPU")],
      plan_version: 99,
    });
    await flushPromises();
    expect(wrapper.text()).not.toContain("STALE FIRST GPU");

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    const deviceTargets = apiJsonTo.mock.calls
      .filter(([, path]) => path === "/api/devices")
      .map(([target]) => target);
    expect(deviceTargets.at(-1)).toEqual({
      baseUrl: second.baseUrl,
      apiKey: second.apiKey,
    });
    wrapper.unmount();
  });

  it("stops polling and suppresses an in-flight result after unmount", async () => {
    vi.useFakeTimers();
    const host = mobileHost("unmount-host");
    const pending = deferred<unknown>();
    let deviceCalls = 0;
    apiJsonTo.mockImplementation((_target, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        return pending.promise;
      }
      if (path === "/api/capabilities") return Promise.resolve(deviceCapabilities());
      throw new Error(`unexpected ${path}`);
    });

    const wrapper = mountSettings(host);
    await flushPromises();
    const signal = subscribeMock.mock.calls[0]?.[1] as AbortSignal;
    wrapper.unmount();

    expect(signal.aborted).toBe(true);
    pending.resolve({
      devices: [deviceWireForRace("UNMOUNTED GPU")],
      plan_version: 1,
    });
    await vi.advanceTimersByTimeAsync(20_000);
    await flushPromises();

    expect(deviceCalls).toBe(1);
  });
});

function mobileHost(id: string) {
  return {
    id,
    name: id,
    baseUrl: `http://${id}:7680`,
    apiKey: `${id}-secret`,
    hostname: id,
    version: "0.20.2",
    online: true,
  };
}

function mountSettings(host: ReturnType<typeof mobileHost>) {
  return mount(MobileSettingsView, {
    props: {
      settings: { theme: "system", themeFamily: "mold", autoSavePhotos: true, autoTagTitle: true },
      hostCount: 1,
      appVersion: "0.20.2",
      host,
    },
  });
}

function deviceCapabilities() {
  return {
    devices: { available: true, lifecycle: true, restart_enable: false },
    dispatch: { active_mode: "v2", v2_authoritative: true },
  };
}

function deviceWireForRace(name = "Race GPU") {
  return {
    id: `cuda:${name}`,
    backend: "cuda",
    ordinal: 0,
    device_kind: "full_gpu",
    nvml_uuid: "GPU-race",
    physical_uuid: "GPU-race",
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
