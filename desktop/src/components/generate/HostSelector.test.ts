import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import HostSelector from "./HostSelector.vue";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { AppSettings } from "../../lib/ipc";

const appSettingsSet = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsSet: (...a: unknown[]) => appSettingsSet(...a),
  },
}));

function baseSettings(): AppSettings {
  return {
    mode: "local",
    remoteUrl: null,
    remoteApiKey: null,
    lastRoute: null,
    engineEnv: {},
    theme: "system",
    themeFamily: "mold",
    notifications: true,
    dockBadge: true,
    restoreLastRoute: false,
    runpodIncludeHfToken: false,
    runpodNetworkVolumeId: null,
    uiScalePercent: 100,
    updateChannel: "stable",
    savedHosts: [],
    connectedHostIds: [],
    generateTargetHost: null,
  };
}

function setup(withExtra: boolean) {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  const hosts = useHostsStore();
  if (withExtra) {
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
    });
    hosts.telemetry["hal9000-7680"] = { queueDepth: 2, queueCapacity: 8, version: null };
  }
  useAppPrefsStore().settings = baseSettings();
  return mount(HostSelector);
}

beforeEach(() => {
  appSettingsSet.mockClear();
});

describe("HostSelector", () => {
  it("renders nothing with a single host", () => {
    const wrapper = setup(false);
    expect(wrapper.find("[data-test='host-selector']").exists()).toBe(false);
  });

  it("lists Auto plus every host with live queue depth", () => {
    const wrapper = setup(true);
    const options = wrapper.findAll("option");
    expect(options.map((o) => o.text())).toEqual([
      "Auto — least busy",
      "This Mac",
      "hal9000 · 2 queued",
    ]);
  });

  it("persists the picked host and maps Auto back to null", async () => {
    const wrapper = setup(true);
    await wrapper.get("select").setValue("hal9000-7680");
    await flushPromises();
    expect(appSettingsSet).toHaveBeenCalledWith(
      expect.objectContaining({ generateTargetHost: "hal9000-7680" }),
    );
    await wrapper.get("select").setValue("auto");
    await flushPromises();
    expect(appSettingsSet).toHaveBeenLastCalledWith(
      expect.objectContaining({ generateTargetHost: null }),
    );
  });

  it("shows a stale persisted pick as Auto when the host is gone", () => {
    const wrapper = setup(true);
    useAppPrefsStore().settings!.generateTargetHost = "vanished-host";
    expect((wrapper.get("select").element as HTMLSelectElement).value).toBe("auto");
  });
});
