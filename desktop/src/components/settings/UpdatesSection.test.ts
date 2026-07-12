import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/ipc", () => ({
  ipc: {
    onUpdaterProgress: () => Promise.resolve(() => {}),
    takeUpdateRecovery: () => Promise.resolve(null),
    confirmUpdateHealthy: () => Promise.resolve(),
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
    expect(alert.text()).toContain("Mold 0.16.0 is still installed. No app files were changed.");
    expect(alert.find("[data-selectable]").exists()).toBe(true);
  });

  it("announces a recovered prior version with the failed version for context", async () => {
    const { updater, wrapper } = mountSection();
    updater.recovery = {
      restoredVersion: "0.16.0",
      failedVersion: "0.17.0",
      message: "Mold restored the previous version after startup failed.",
    };
    await wrapper.vm.$nextTick();

    const recovery = wrapper.get("[data-test='update-recovery']");
    expect(recovery.attributes("role")).toBe("status");
    expect(recovery.text()).toContain("Mold 0.16.0 was restored");
    expect(recovery.text()).toContain("0.17.0");
  });

  it("shows the preserved backup path and manual recovery action when rollback fails", async () => {
    const { updater, wrapper } = mountSection();
    updater.recovery = {
      restoredVersion: "0.16.0",
      failedVersion: "0.17.0",
      message: "Automatic rollback could not replace the installed app.",
      rollbackFailed: true,
      backupPath: "/Users/me/Library/Application Support/Mold/updater/backup/Mold.app",
    };
    await wrapper.vm.$nextTick();

    const recovery = wrapper.get("[data-test='update-recovery']");
    expect(recovery.text()).toContain("Automatic rollback needs attention");
    expect(recovery.text()).toContain("Copy the recovery app to Applications");
    expect(recovery.text()).toContain("/Users/me/Library/Application Support");
    expect(recovery.text()).not.toContain("Dismiss");
    expect(wrapper.get("select[aria-label='Update channel']").attributes("disabled")).toBeDefined();
    expect(wrapper.get("button").attributes("disabled")).toBeDefined();
  });

  it("explains that browser and unsigned builds cannot self-update", async () => {
    const { updater, wrapper } = mountSection();
    updater.phase = "unsupported";
    await wrapper.vm.$nextTick();
    expect(wrapper.text()).toContain("Automatic updates are available in signed desktop builds");
  });
});
