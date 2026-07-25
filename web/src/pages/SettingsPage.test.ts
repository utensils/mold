import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsPage from "./SettingsPage.vue";
import settingsPageSource from "./SettingsPage.vue?raw";
import { theme, themeFamily } from "../lib/theme";
import { resetNotifications, useNotifications } from "../lib/toasts";
import type { ServerStatus } from "../types";

const statusRef = vi.hoisted(() => ({ value: null as ServerStatus | null }));

vi.mock("../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({ status: statusRef }),
}));

const originalFetch = globalThis.fetch;

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
});
