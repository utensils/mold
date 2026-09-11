import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import type { ConfigRow } from "../../api/config";
import ConfigSettingRow from "./ConfigSettingRow.vue";

/*
 * The schema-driven row. It autosaves — there is no Save button — so the four
 * commit rules ported from `ConfigRowItem.test.ts` are what stop it writing
 * rows nobody edited: a blur with no change is silent, and a blank or
 * non-numeric field restores what the engine still holds instead of sending
 * `Number("")` (0) or `Number("abc")` (null).
 */

function row(over: Partial<ConfigRow> = {}): ConfigRow {
  return {
    key: "scheduler.replan_debounce_ms",
    value: 250,
    source: "db",
    env_var: null,
    restart_required: false,
    ...over,
  };
}

function mountRow(schemaKey: string, over: Partial<ConfigRow> = {}) {
  return mount(ConfigSettingRow, {
    props: { schemaKey, row: row({ key: schemaKey, ...over }) },
  });
}

describe("ConfigSettingRow commits", () => {
  it("stays silent when a numeric row is only tabbed through", async () => {
    const wrapper = mountRow("scheduler.replan_debounce_ms");
    await wrapper.get("input").trigger("change");
    expect(wrapper.emitted("save")).toBeUndefined();
  });

  it("never turns a cleared numeric field into 0", async () => {
    const wrapper = mountRow("scheduler.replan_debounce_ms");
    const input = wrapper.get("input");
    await input.setValue("");
    expect(wrapper.emitted("save")).toBeUndefined();
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("never sends a non-numeric entry as null", async () => {
    const wrapper = mountRow("scheduler.replan_debounce_ms");
    const input = wrapper.get("input");
    await input.setValue("abc");
    expect(wrapper.emitted("save")).toBeUndefined();
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("saves a real change once, naming the key", async () => {
    const wrapper = mountRow("scheduler.replan_debounce_ms");
    await wrapper.get("input").setValue("400");
    expect(wrapper.emitted("save")).toEqual([
      ["scheduler.replan_debounce_ms", 400],
    ]);
  });
});

describe("ConfigSettingRow editors", () => {
  it("renders the editor the schema declares", () => {
    expect(
      mountRow("embed_metadata", { value: true })
        .get("button")
        .attributes("role"),
    ).toBe("switch");
    expect(mountRow("t5_variant", { value: "" }).find("select").exists()).toBe(
      true,
    );
    expect(
      mountRow("expand.top_p", { value: 0.9 }).get("input").attributes("type"),
    ).toBe("range");
    expect(
      mountRow("expand.backend", { value: "local" })
        .get("input")
        .attributes("type"),
    ).toBe("text");
  });

  it("edits a cloud key as a secret, reading the engine's <set> as present", () => {
    const wrapper = mountRow("runpod.api_key", {
      value: "<set>",
      source: "file",
    });
    expect(wrapper.text()).not.toContain("<set>");
    expect(wrapper.text()).toContain("set");
  });

  it("clears a text row as null rather than as an empty string", async () => {
    const wrapper = mountRow("default_negative_prompt", { value: "blurry" });
    await wrapper.get("input").setValue("");
    expect(wrapper.emitted("save")).toEqual([
      ["default_negative_prompt", null],
    ]);
  });

  it("gets a native folder picker only when one is handed to it", () => {
    expect(
      mountRow("models_dir", { value: "/m", source: "file" }).text(),
    ).not.toContain("Choose…");
    const native = mount(ConfigSettingRow, {
      props: {
        schemaKey: "models_dir",
        row: row({ key: "models_dir", value: "/m", source: "file" }),
        pickDirectory: async () => "/picked",
      },
    });
    expect(native.text()).toContain("Choose…");
  });
});

describe("ConfigSettingRow locks", () => {
  it("says a startup-only key cannot move while the server runs", () => {
    const wrapper = mountRow("output_dir", { value: "/out", source: "file" });
    expect(wrapper.text()).toContain(
      "Startup-only while the server is running",
    );
    expect(wrapper.get("input").attributes("disabled")).toBeDefined();
  });

  it("names the environment variable that is winning", () => {
    const wrapper = mountRow("default_steps", {
      value: 30,
      source: "env",
      env_var: "MOLD_DEFAULT_STEPS",
    });
    expect(wrapper.text()).toContain("MOLD_DEFAULT_STEPS");
    expect(wrapper.text()).toContain("unset it to edit here");
  });
});

describe("ConfigSettingRow reset", () => {
  /* ↺ is the only affordance — no per-row Save or Reset buttons. It appears
   * only where `DELETE /api/config/:key` will actually work: offering it on a
   * config.toml key is offering a button the host refuses. */
  it("offers ↺ on a DB-surface key and emits the key", async () => {
    const wrapper = mountRow("expand.top_p", { value: 0.9 });
    expect(wrapper.findAll("button").map((b) => b.text())).not.toContain(
      "Save",
    );
    await wrapper.get("[data-test='setting-reset']").trigger("click");
    expect(wrapper.emitted("reset")).toEqual([["expand.top_p"]]);
  });

  it("offers no ↺ on a key that lives in config.toml", () => {
    // `logging.*` is File-surface, so DELETE refuses it outright.
    const logging = mountRow("logging.level", {
      value: "info",
      source: "file",
    });
    expect(logging.find("[data-test='setting-reset']").exists()).toBe(false);
    const port = mountRow("server_port", { value: 7680, source: "file" });
    expect(port.find("[data-test='setting-reset']").exists()).toBe(false);
  });
});

describe("ConfigSettingRow absences", () => {
  it("renders nothing for a key the host does not report", () => {
    const wrapper = mount(ConfigSettingRow, {
      props: { schemaKey: "expand.top_p", row: null },
    });
    expect(wrapper.text()).toBe("");
  });
});
