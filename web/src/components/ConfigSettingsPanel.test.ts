import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ConfigSettingsPanel from "./ConfigSettingsPanel.vue";
import { AUTO_TAG_SETTING_WEB } from "@studio/lib/fileUnder";
import { autoTagTitle, reloadAutoTagTitle } from "../lib/fileUnder";

const rows = [
  { key: "models_dir", value: "/models", source: "file" },
  { key: "output_dir", value: "/prints", source: "file" },
  { key: "default_steps", value: 28, source: "db", profile: "default" },
  { key: "expand.enabled", value: true, source: "default" },
  {
    key: "default_width",
    value: 1024,
    source: "env",
    env_var: "MOLD_DEFAULT_WIDTH",
  },
  { key: "future.option", value: "visible", source: "default" },
  {
    key: "scheduler.replan_debounce_ms",
    value: 2000,
    source: "db",
    restart_required: true,
  },
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
    expect(
      wrapper.get('[data-test="config-output_dir"]').attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get('[data-test="save-output_dir"]').attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.text()).toContain("Startup-only");
    expect(
      wrapper.get('[data-test="restart-scheduler.replan_debounce_ms"]').text(),
    ).toBe("Restart server to apply");
    expect(
      wrapper
        .get('[data-test="reset-scheduler.replan_debounce_ms"]')
        .attributes("disabled"),
    ).toBeUndefined();
  });

  it("uses the shared icon-led, accented treatment for every All settings group", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();

    const groups = wrapper.findAll("[data-test='config-group']");
    // Four server sections plus Library, which this host contributes no rows
    // to but which still carries the browser-local auto-tag preference.
    expect(groups).toHaveLength(5);
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

  it("renders trash retention as labelled choices and writes it back as a number", async () => {
    rows.push({
      key: "gallery.trash_retention_days",
      value: 30,
      source: "db",
      profile: "default",
    });
    try {
      const wrapper = mount(ConfigSettingsPanel);
      await flushPromises();
      const select = wrapper.get<HTMLSelectElement>(
        '[data-test="config-gallery.trash_retention_days"]',
      );
      expect(select.element.tagName).toBe("SELECT");
      const labels = select.findAll("option").map((o) => o.text());
      expect(labels).toEqual([
        "1 day",
        "7 days",
        "30 days",
        "90 days",
        "1 year",
        "Forever",
      ]);
      expect(wrapper.text()).toContain("Prints moved to the trash");
      expect(
        wrapper
          .get('[data-test="reset-gallery.trash_retention_days"]')
          .attributes("disabled"),
      ).toBeUndefined();

      await select.setValue("0");
      await wrapper
        .get('[data-test="save-gallery.trash_retention_days"]')
        .trigger("click");
      await flushPromises();
      const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/config/gallery.trash_retention_days",
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ value: 0 }),
        },
      );
    } finally {
      rows.pop();
    }
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

  it("offers the browser-local auto-tag preference in Settings ▸ Library", async () => {
    localStorage.clear();
    reloadAutoTagTitle();
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();

    const toggle = wrapper.get('[data-test="config-auto-tag-title"]');
    expect(wrapper.text()).toContain("Tag new prints with their title");
    expect(wrapper.text()).toContain("it never rewrites existing ones");
    // On by default, and stored in this browser rather than on the host —
    // so there is nothing to Save or Reset.
    expect((toggle.element as HTMLInputElement).checked).toBe(true);
    expect(wrapper.get('[data-test="source-auto-tag-title"]').text()).toBe(
      "this browser",
    );
    expect(
      wrapper.find('[data-test="save-mold.create.autoTagTitle.v1"]').exists(),
    ).toBe(false);

    await toggle.setValue(false);
    expect(autoTagTitle.value).toBe(false);
    expect(localStorage.getItem(AUTO_TAG_SETTING_WEB)).toBe("false");
  });

  it("keeps the Library heading for the local row when the host has no Library config", async () => {
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      if (url === "/api/config" && !init?.method)
        return response({
          profile: "default",
          entries: [{ key: "models_dir", value: "/models", source: "file" }],
        });
      return response({ profiles: ["default"], active: "default" });
    }) as typeof fetch;
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    expect(wrapper.find('[data-test="config-auto-tag-title"]').exists()).toBe(
      true,
    );
  });

  it("hides the local row when it does not match the settings search", async () => {
    const wrapper = mount(ConfigSettingsPanel);
    await flushPromises();
    await wrapper
      .get('[data-test="config-search"]')
      .setValue("prompt expansion");
    expect(wrapper.find('[data-test="config-auto-tag-title"]').exists()).toBe(
      false,
    );
    await wrapper.get('[data-test="config-search"]').setValue("title");
    expect(wrapper.find('[data-test="config-auto-tag-title"]').exists()).toBe(
      true,
    );
  });
});
