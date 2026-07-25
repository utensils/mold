import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ConfigSettingsPanel from "./ConfigSettingsPanel.vue";

const rows = [
  { key: "models_dir", value: "/models", source: "file" },
  { key: "default_steps", value: 28, source: "db", profile: "default" },
  { key: "expand.enabled", value: true, source: "default" },
  {
    key: "default_width",
    value: 1024,
    source: "env",
    env_var: "MOLD_DEFAULT_WIDTH",
  },
  { key: "future.option", value: "visible", source: "default" },
];

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 404 ? "Not Found" : "OK",
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

describe("ConfigSettingsPanel", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      if (url === "/api/config" && !init?.method)
        return response({ profile: "default", entries: rows });
      if (url === "/api/config/profiles")
        return response({
          profiles: ["default", "quality"],
          active: "default",
        });
      return response({});
    }) as typeof fetch;
  });

  afterEach(() => vi.restoreAllMocks());

  it("renders curated and future config with provenance and locked env values", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();

    expect(wrapper.text()).toContain("Models directory");
    expect(wrapper.text()).toContain("Default steps");
    expect(wrapper.text()).toContain("Enable prompt expansion");
    expect(wrapper.text()).toContain("future.option");
    expect(wrapper.get('[data-test="source-models_dir"]').text()).toBe("file");
    expect(
      wrapper.get('[data-test="reset-models_dir"]').attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get('[data-test="reset-default_steps"]').attributes("disabled"),
    ).toBeUndefined();
    expect(
      wrapper.get('[data-test="config-default_width"]').attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.text()).toContain("MOLD_DEFAULT_WIDTH");
  });

  it("uses the shared icon-led, accented treatment for every All settings group", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();

    const groups = wrapper.findAll("[data-test='config-group']");
    expect(groups).toHaveLength(4);
    for (const group of groups) {
      expect(group.find(".config-group__plate").exists()).toBe(true);
      expect(group.find(".config-group__summary").text()).not.toBe("");
      expect(group.find(".config-card--accented").exists()).toBe(true);
    }
  });

  it("searches labels, help, and raw keys", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    await wrapper.get('[data-test="config-search"]').setValue("future.option");

    expect(wrapper.text()).toContain("future.option");
    expect(wrapper.text()).not.toContain("Models directory");
  });

  it("writes typed values and resets individual keys", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    await wrapper.get('[data-test="config-default_steps"]').setValue("36");
    await wrapper.get('[data-test="save-default_steps"]').trigger("click");
    await flushPromises();
    await wrapper.get('[data-test="reset-default_steps"]').trigger("click");
    await flushPromises();

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    expect(fetchMock).toHaveBeenCalledWith("/api/config/default_steps", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value: 36 }),
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/config/default_steps", {
      method: "DELETE",
    });
  });

  it("does not coerce an empty numeric draft to zero", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    await wrapper.get('[data-test="config-default_steps"]').setValue("");
    await wrapper.get('[data-test="save-default_steps"]').trigger("click");
    await flushPromises();

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          url === "/api/config/default_steps" && init?.method === "PUT",
      ),
    ).toBe(false);
  });

  it("switches to existing profiles and creates new ones through the same endpoint", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    await wrapper.get('[data-test="profile-select"]').setValue("quality");
    await wrapper.get('[data-test="profile-name"]').setValue("drafts");
    await wrapper.get('[data-test="profile-create"]').trigger("click");
    await flushPromises();

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    expect(fetchMock).toHaveBeenCalledWith("/api/config/profile", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "quality" }),
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/config/profile", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "drafts" }),
    });
  });
});
