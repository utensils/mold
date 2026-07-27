import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsPage from "./SettingsPage.vue";
import settingsPageSource from "./SettingsPage.vue?raw";
import { theme, themeFamily } from "../lib/theme";
import { resetNotifications, useNotifications } from "../lib/toasts";
import { originHost } from "../lib/hostRegistry";
import type { ServerStatus } from "../types";

const statusRef = vi.hoisted(() => ({ value: null as ServerStatus | null }));
const subscribeToDeviceSnapshots = vi.hoisted(() => vi.fn());

vi.mock("../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({ status: statusRef }),
}));
vi.mock("../lib/deviceEvents", () => ({ subscribeToDeviceSnapshots }));

const originalFetch = globalThis.fetch;

function deviceWire(enabled = true, adminState = "enabled") {
  return {
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
    desired_enabled: enabled,
    admin_state: adminState,
    health: "healthy",
    activity: "idle",
    schedulable: enabled,
    unschedulable_reason: enabled ? null : "device_disabled",
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

describe("SettingsPage", () => {
  it("keeps its padded content inside narrow web viewports", () => {
    const settingsRule = settingsPageSource.match(/\.settings\s*\{([^}]*)\}/s);
    expect(settingsRule).not.toBeNull();

    const settingsDeclarations = settingsRule?.[1] ?? "";
    expect(settingsDeclarations).toMatch(/width:\s*100%/);
    expect(settingsDeclarations).toMatch(/box-sizing:\s*border-box/);
  });

  beforeEach(() => {
    statusRef.value = null;
    subscribeToDeviceSnapshots.mockClear();
    resetNotifications();
    localStorage.clear();
    globalThis.fetch = vi.fn(
      async (input) =>
        ({
          ok: true,
          json: async () => {
            if (String(input).endsWith("/profiles"))
              return { profiles: ["default"], active: "default" };
            if (String(input).endsWith("/api/catalog/credentials")) {
              return {
                hf: { configured: false, source: null, masked: null },
                civitai: { configured: false, source: null, masked: null },
              };
            }
            return { entries: [] };
          },
        }) as Response,
    ) as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    theme.value = "system";
    themeFamily.value = "mold";
    vi.restoreAllMocks();
  });

  it("renders the Settings title and About rows", () => {
    statusRef.value = {
      version: "9.9.9",
      models_loaded: [],
      busy: false,
      uptime_secs: 1,
    };
    const wrapper = mount(SettingsPage);

    expect(wrapper.get("h1").text()).toBe("Settings");
    expect(wrapper.get('[data-test="about-version"]').text()).toBe("9.9.9");
    expect(wrapper.text()).toContain("local + your hosts");
    expect(wrapper.text()).toContain("Core contributors");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
    expect(wrapper.text()).not.toMatch(/equal (project )?owners/i);
  });

  it("falls back to an em dash when the server version is unknown", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.get('[data-test="about-version"]').text()).toBe("—");
  });

  it("refreshes device truth on SSE connect and semantic invalidations", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(1);
    const [target, _signal, refresh] =
      subscribeToDeviceSnapshots.mock.calls[0]!;
    expect(target).toEqual({
      baseUrl: originHost().url,
      apiKey: originHost().apiKey ?? null,
    });
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const before = fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/devices"),
    ).length;
    expect(before).toBe(1);

    refresh();
    await flushPromises();

    expect(
      fetchMock.mock.calls.filter(([input]) =>
        String(input).endsWith("/api/devices"),
      ),
    ).toHaveLength(before + 1);
    wrapper.unmount();
  });

  it("ignores an older same-origin device response after an event refetch", async () => {
    const older = deferred<Response>();
    const newer = deferred<Response>();
    let deviceCall = 0;
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/devices"))
        return deviceCall++ === 0 ? older.promise : newer.promise;
      return {
        ok: true,
        json: async () => {
          if (url.endsWith("/profiles"))
            return { profiles: ["default"], active: "default" };
          if (url.endsWith("/api/catalog/credentials"))
            return {
              hf: { configured: false, source: null, masked: null },
              civitai: { configured: false, source: null, masked: null },
            };
          if (url.endsWith("/api/capabilities"))
            return {
              devices: { available: true, lifecycle: true },
              dispatch: { active_mode: "v2", v2_authoritative: true },
            };
          return { entries: [], plan: null };
        },
      } as Response;
    }) as typeof fetch;
    const wrapper = mount(SettingsPage, {
      global: { stubs: { DeviceSettingsPanel: true } },
    });
    await vi.waitFor(() =>
      expect(subscribeToDeviceSnapshots).toHaveBeenCalled(),
    );
    const refresh = subscribeToDeviceSnapshots.mock.calls[0]?.[2] as () => void;
    refresh();
    newer.resolve({
      ok: true,
      json: async () => ({
        devices: [deviceWire(false, "disabled")],
        plan_version: 2,
      }),
    } as Response);
    await vi.waitFor(() => expect(wrapper.text()).toContain("disabled"));
    older.resolve({
      ok: true,
      json: async () => ({
        devices: [deviceWire(true, "enabled")],
        plan_version: 1,
      }),
    } as Response);
    await flushPromises();

    expect(wrapper.text()).toContain("disabled");
  });

  it("persists the theme family through the shared lib/theme refs", async () => {
    const wrapper = mount(SettingsPage);
    const button = wrapper
      .get('[data-test="seg-theme"]')
      .findAll("button")
      .find((b) => b.text() === "Safelight");
    await button?.trigger("click");
    await flushPromises();
    expect(themeFamily.value).toBe("safelight");
  });

  it("persists the appearance through the shared lib/theme refs", async () => {
    const wrapper = mount(SettingsPage);
    const button = wrapper
      .get('[data-test="seg-appearance"]')
      .findAll("button")
      .find((b) => b.text() === "Dark");
    await button?.trigger("click");
    await flushPromises();
    expect(theme.value).toBe("dark");
  });

  it("stores the Hugging Face token on the server and shows a masked state", async () => {
    localStorage.clear();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      if (
        String(input).endsWith("/api/catalog/credentials/hf") &&
        init?.method === "PUT"
      ) {
        return {
          ok: true,
          json: async () => ({
            hf: {
              configured: true,
              source: "server",
              masked: "hf_••••1234",
            },
            civitai: { configured: false, source: null, masked: null },
          }),
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : {
                hf: { configured: false, source: null, masked: null },
                civitai: { configured: false, source: null, masked: null },
              },
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await wrapper.get("input[name=hf_token]").setValue("hf_secretvalue1234");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    const saveCall = fetchMock.mock.calls.find((c) =>
      String(c[0]).endsWith("/api/catalog/credentials/hf"),
    );
    expect(saveCall?.[1]).toEqual(
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ token: "hf_secretvalue1234" }),
      }),
    );
    expect(localStorage.getItem("mold.web.accounts.v1")).toBeNull();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••1234");
    expect(useNotifications().toasts.some((t) => /server/.test(t.text))).toBe(
      true,
    );
  });

  it("reflects an already-saved server token on load and can clear it", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const configured = {
        hf: { configured: false, source: null, masked: null },
        civitai: {
          configured: true,
          source: "server",
          masked: "cv_••••7890",
        },
      };
      const cleared = {
        hf: { configured: false, source: null, masked: null },
        civitai: { configured: false, source: null, masked: null },
      };
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : init?.method === "DELETE"
              ? cleared
              : configured,
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();
    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();
    expect(wrapper.find('[data-test="civitai-mask"]').exists()).toBe(false);
    expect(
      fetchMock.mock.calls.some(
        ([input, init]) =>
          String(input).endsWith("/api/catalog/credentials/civitai") &&
          init?.method === "DELETE",
      ),
    ).toBe(true);
  });

  it("lets a server-saved token override an environment fallback and clear back to it", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const environment = {
        hf: {
          configured: true,
          source: "environment",
          masked: "hf_••••base",
        },
        civitai: { configured: false, source: null, masked: null },
      };
      const override = {
        hf: {
          configured: true,
          source: "server",
          masked: "hf_••••ride",
        },
        civitai: { configured: false, source: null, masked: null },
      };
      return {
        ok: true,
        json: async () => {
          if (String(input).endsWith("/profiles"))
            return { profiles: ["default"], active: "default" };
          if (init?.method === "PUT") return override;
          return environment;
        },
      } as Response;
    });

    const wrapper = mount(SettingsPage);
    await flushPromises();
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe("Environment");
    await wrapper.get('[data-test="replace-hf"]').trigger("click");
    await wrapper.get("input[name=hf_token]").setValue("hf_user_override");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••ride");
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe(
      "Saved override",
    );

    await wrapper.get('[data-test="clear-hf"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••base");
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe("Environment");
  });

  it("keeps the token in the field when the server rejects the save", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "disk full",
    });

    await wrapper.get("input[name=hf_token]").setValue("hf_secretvalue1234");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    const input = wrapper.get("input[name=hf_token]")
      .element as HTMLInputElement;
    expect(input.value).toBe("hf_secretvalue1234");
    expect(wrapper.find('[data-test="hf-mask"]').exists()).toBe(false);

    const { toasts } = useNotifications();
    expect(toasts.some((t) => t.kind === "success")).toBe(false);
    const error = toasts.find((t) => t.kind === "error");
    expect(error?.text).toContain("disk full");
  });

  it("does not claim a server token was removed when the clear fails", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      if (init?.method === "DELETE") {
        return {
          ok: false,
          status: 500,
          text: async () => "read-only filesystem",
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : {
                hf: { configured: false, source: null, masked: null },
                civitai: {
                  configured: true,
                  source: "server",
                  masked: "cv_••••7890",
                },
              },
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();

    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    const error = useNotifications().toasts.find((t) => t.kind === "error");
    expect(error?.text).toContain("read-only filesystem");
  });

  it("does not render a fictional default-scheduler control", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.find("select[name=default_scheduler]").exists()).toBe(false);
  });

  it("does not render an NSFW visibility control", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.find("input[name=catalog_show_nsfw]").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Show NSFW");
  });

  it("shows CPU utility work when the server reports no GPUs", async () => {
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);
      const body = url.endsWith("/api/devices")
        ? { devices: [], plan_version: 1 }
        : url.endsWith("/api/queue")
          ? {
              entries: [],
              plan: {
                plan_version: 1,
                state_version: 1,
                optimizer_state: "optimized",
                dirty_since_unix_ms: null,
                next_replan_at_unix_ms: null,
                work_items: [
                  {
                    work_id: "cpu-expand",
                    parent_id: "parent",
                    work_kind: "prompt_expansion",
                    priority_class: "user",
                    queue_rank: 0,
                    bypass_count: 0,
                    planned_device_id: "cpu:utility:0",
                    lane_order: 0,
                    estimate_confidence: "low",
                    activity_phase: "cpu",
                  },
                ],
              },
            }
          : url.endsWith("/api/capabilities")
            ? {
                devices: { available: true, lifecycle: true },
                dispatch: { active_mode: "v2", v2_authoritative: true },
              }
            : url.endsWith("/profiles")
              ? { profiles: ["default"], active: "default" }
              : url.endsWith("/api/catalog/credentials")
                ? {
                    hf: { configured: false, source: null, masked: null },
                    civitai: { configured: false, source: null, masked: null },
                  }
                : { entries: [] };
      return { ok: true, json: async () => body } as Response;
    }) as typeof fetch;

    const wrapper = mount(SettingsPage);
    await vi.waitFor(() => expect(wrapper.text()).toContain("Host utility"));

    expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(0);
    expect(wrapper.get('[data-test="cpu-utility-lane"]').text()).toContain(
      "cpu-expand",
    );
  });

  it("controls the origin server GPU from Advanced settings", async () => {
    let enabled = true;
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/api/devices") && !init?.method) {
        return {
          ok: true,
          json: async () => ({
            devices: [deviceWire(enabled, enabled ? "enabled" : "disabled")],
            plan_version: enabled ? 1 : 2,
          }),
        } as Response;
      }
      if (url.endsWith("/api/capabilities")) {
        return {
          ok: true,
          json: async () => ({
            devices: {
              available: true,
              lifecycle: true,
              restart_enable: false,
            },
            dispatch: { active_mode: "v2", v2_authoritative: true },
          }),
        } as Response;
      }
      if (url.includes("/api/devices/cuda%3A") && init?.method === "PATCH") {
        enabled = false;
        return {
          ok: true,
          json: async () => deviceWire(false, "disabled"),
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          url.endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : url.endsWith("/api/catalog/credentials")
              ? {
                  hf: { configured: false, source: null, masked: null },
                  civitai: { configured: false, source: null, masked: null },
                }
              : { entries: [] },
      } as Response;
    });

    const wrapper = mount(SettingsPage);
    await flushPromises();
    expect(
      wrapper.get("[data-test='settings-device-controls']").text(),
    ).toContain("NVIDIA RTX 3090");
    expect(
      wrapper.get("[data-test='device-panel']").attributes("data-device-count"),
    ).toBe("1");

    await wrapper
      .get("[data-test='settings-device-toggle-0']")
      .trigger("click");
    await flushPromises();

    const patch = fetchMock.mock.calls.find(
      ([input, init]) =>
        String(input).includes("/api/devices/cuda%3A") &&
        init?.method === "PATCH",
    );
    expect(patch?.[1]).toEqual(
      expect.objectContaining({
        body: JSON.stringify({ enabled: false }),
      }),
    );
    expect(wrapper.get("[data-test='settings-device-toggle-0']").text()).toBe(
      "Enable",
    );
  });
});
