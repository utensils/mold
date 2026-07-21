import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsPage from "./SettingsPage.vue";
import { theme, themeFamily } from "../lib/theme";
import { resetNotifications, useNotifications } from "../lib/toasts";
import type { ServerStatus } from "../types";

const statusRef = vi.hoisted(() => ({ value: null as ServerStatus | null }));

vi.mock("../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({ status: statusRef }),
}));

const originalFetch = globalThis.fetch;

describe("SettingsPage", () => {
  beforeEach(() => {
    statusRef.value = null;
    resetNotifications();
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true }) as typeof fetch;
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
  });

  it("falls back to an em dash when the server version is unknown", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.get('[data-test="about-version"]').text()).toBe("—");
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

  it("saves the Hugging Face token via POST /api/settings/set", async () => {
    const wrapper = mount(SettingsPage);
    await wrapper.get("input[name=hf_token]").setValue("hf_secret");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    expect(fm).toHaveBeenCalledWith(
      "/api/settings/set",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "huggingface.token", value: "hf_secret" });
    expect(useNotifications().toasts.some((t) => t.text === "saved")).toBe(
      true,
    );
  });

  it("saves the Civitai token via POST /api/settings/set", async () => {
    const wrapper = mount(SettingsPage);
    await wrapper.get("input[name=civitai_token]").setValue("cv_secret");
    await wrapper.get('[data-test="save-civitai"]').trigger("click");
    await flushPromises();

    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "civitai.token", value: "cv_secret" });
  });

  it("saves the default scheduler on change", async () => {
    const wrapper = mount(SettingsPage);
    const select = wrapper.get("select[name=default_scheduler]");
    await select.setValue("ddim");
    await flushPromises();

    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "generate.scheduler", value: "ddim" });
  });

  it("does not render an NSFW visibility control", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.find("input[name=catalog_show_nsfw]").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Show NSFW");
  });
});
