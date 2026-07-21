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

  it("stores the Hugging Face token in this browser and shows a masked state", async () => {
    localStorage.clear();
    const wrapper = mount(SettingsPage);
    await wrapper.get("input[name=hf_token]").setValue("hf_secretvalue1234");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    // No fictional network call — the token lives in localStorage.
    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    const calledSettings = fm.mock.calls.some((c) =>
      String(c[0]).includes("/api/settings/set"),
    );
    expect(calledSettings).toBe(false);
    expect(
      JSON.parse(localStorage.getItem("mold.web.accounts.v1") ?? "{}").hfToken,
    ).toBe("hf_secretvalue1234");
    // Masked indicator + honest toast.
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••1234");
    expect(
      useNotifications().toasts.some((t) => /this browser/.test(t.text)),
    ).toBe(true);
  });

  it("reflects an already-saved token on load and can clear it", async () => {
    localStorage.setItem(
      "mold.web.accounts.v1",
      JSON.stringify({ civitaiToken: "cv_abcdef7890" }),
    );
    const wrapper = mount(SettingsPage);
    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();
    expect(wrapper.find('[data-test="civitai-mask"]').exists()).toBe(false);
    expect(localStorage.getItem("mold.web.accounts.v1")).toBeNull();
  });

  it("keeps the token in the field and says so when the browser blocks storage", async () => {
    localStorage.clear();
    const wrapper = mount(SettingsPage);
    vi.spyOn(localStorage, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
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
    expect(error?.text).toBe(
      "This browser blocked storage — the token wasn't saved.",
    );
  });

  it("does not claim a token was removed when the clear could not persist", async () => {
    localStorage.setItem(
      "mold.web.accounts.v1",
      JSON.stringify({ civitaiToken: "cv_abcdef7890" }),
    );
    const wrapper = mount(SettingsPage);
    vi.spyOn(localStorage, "removeItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    const error = useNotifications().toasts.find((t) => t.kind === "error");
    expect(error?.text).toBe(
      "This browser blocked storage — the token wasn't removed.",
    );
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
});
