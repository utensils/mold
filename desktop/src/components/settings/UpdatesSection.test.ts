import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/ipc", () => ({
  ipc: {
    onUpdaterProgress: () => Promise.resolve(() => {}),
    checkForUpdates: () =>
      Promise.resolve({
        supported: true,
        channel: "stable",
        currentVersion: "0.16.0",
        checkedAt: "2026-07-12T18:00:00Z",
        candidate: null,
      }),
    appSettingsSet: () => Promise.resolve(),
    installPendingUpdate: () => Promise.resolve(),
  },
}));

import UpdatesSection from "./UpdatesSection.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useUpdaterStore } from "../../stores/updater";

beforeEach(() => {
  setActivePinia(createPinia());
  useAppPrefsStore().settings = {
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
    saveRemoteOutputs: true,
    navRailWidth: null,
    generateParamsWidth: null,
    sidebarCollapsed: false,
  };
});

function mountSection() {
  const updater = useUpdaterStore();
  updater.initialized = true;
  return { updater, wrapper: mount(UpdatesSection) };
}

describe("UpdatesSection", () => {
  it("labels the stable and nightly channel selector", () => {
    const { wrapper } = mountSection();
    const select = wrapper.get("select[aria-label='Update channel']");
    expect(select.findAll("option").map((option) => option.text())).toEqual(["Stable", "Nightly"]);
  });

  it("renders an available update with selectable notes and the explicit restart action", async () => {
    const { updater, wrapper } = mountSection();
    updater.currentVersion = "0.16.0";
    updater.phase = "available";
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: "2026-07-12T17:00:00Z",
      notes: "Improved Metal startup.",
    };
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain("Mold 0.17.0 is available");
    expect(wrapper.get("[data-test='update-notes']").attributes()).toHaveProperty(
      "data-selectable",
    );
    expect(wrapper.get("[data-test='install-update']").text()).toBe("Update and restart");
  });

  it("exposes determinate download progress to assistive technology", async () => {
    const { updater, wrapper } = mountSection();
    updater.phase = "downloading";
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };
    updater.downloadedBytes = 25;
    updater.totalBytes = 100;
    await wrapper.vm.$nextTick();

    const progress = wrapper.get("[role='progressbar']");
    expect(progress.attributes("aria-valuemin")).toBe("0");
    expect(progress.attributes("aria-valuemax")).toBe("100");
    expect(progress.attributes("aria-valuenow")).toBe("25");
    expect(progress.attributes("aria-label")).toBe("Downloading Mold 0.17.0");
  });

  it("keeps failure details visible and states that the current app was unchanged", async () => {
    const { updater, wrapper } = mountSection();
    updater.currentVersion = "0.16.0";
    updater.phase = "failed";
    updater.error = {
      code: "signature",
      message: "The update signature is invalid.",
      disposition: "unchanged",
      retryable: true,
    };
    await wrapper.vm.$nextTick();

    const alert = wrapper.get("[role='alert']");
    expect(alert.text()).toContain("The update signature is invalid.");
    expect(alert.text()).toContain(
      "Mold 0.16.0 remains installed because the update did not complete.",
    );
    expect(alert.find("[data-selectable]").exists()).toBe(true);
  });

  it("describes complete preflight verification before installation", async () => {
    const { updater, wrapper } = mountSection();
    updater.phase = "staging";
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain(
      "Running complete signature, identity, Gatekeeper, and install-location checks",
    );
  });

  it("explains that browser and unsigned builds cannot self-update", async () => {
    const { updater, wrapper } = mountSection();
    updater.phase = "unsupported";
    await wrapper.vm.$nextTick();
    expect(wrapper.text()).toContain(
      "Automatic updates are currently available only in signed macOS builds",
    );
  });
});
