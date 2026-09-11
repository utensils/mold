import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import type { ConfigRow } from "../../api/config";
import { PER_STYLE_FIELDS } from "../../lib/settingsSchema";
import PerStyleDefaultsRow from "./PerStyleDefaultsRow.vue";

/*
 * hal9000 reports 104 `models.<style>.<field>` rows over 13 styles. Flat, that
 * is most of the Settings page; one disclosure per style is the whole point of
 * this component.
 */

function rowsFor(
  style: string,
  over: Partial<Record<string, unknown>> = {},
): ConfigRow[] {
  return PER_STYLE_FIELDS.map((field) => ({
    key: `models.${style}.${field}`,
    value: (over[field] ?? null) as ConfigRow["value"],
    source: "db" as const,
  }));
}

function mountRow(style: string, over: Partial<Record<string, unknown>> = {}) {
  return mount(PerStyleDefaultsRow, {
    props: { style, rows: rowsFor(style, over) },
  });
}

/** happy-dom does not toggle a <details> from a summary click, so open it the
 *  way the browser would and let the component hear its own event. */
async function expand(wrapper: ReturnType<typeof mountRow>) {
  const details = wrapper.get("details");
  (details.element as HTMLDetailsElement).open = true;
  await details.trigger("toggle");
}

describe("PerStyleDefaultsRow", () => {
  it("is collapsed until it is opened", () => {
    const wrapper = mountRow("flux-dev:q4");
    expect(wrapper.get("details").attributes("open")).toBeUndefined();
    expect(wrapper.findAll("input")).toHaveLength(0);
  });

  it("leads with the friendly name over the runnable id", () => {
    const wrapper = mount(PerStyleDefaultsRow, {
      props: {
        style: "flux-dev:q4",
        rows: rowsFor("flux-dev:q4"),
        displayName: "FLUX Dev",
      },
    });
    expect(wrapper.get("[data-test='per-style-name']").text()).toBe("FLUX Dev");
    expect(wrapper.get("[data-test='per-style-id']").text()).toBe(
      "flux-dev:q4",
    );
  });

  it("falls back to the id when there is no friendlier name", () => {
    const wrapper = mountRow("flux-dev:q4");
    expect(wrapper.get("[data-test='per-style-name']").text()).toBe(
      "flux-dev:q4",
    );
  });

  it("counts only the fields this style actually overrides", () => {
    expect(
      mountRow("z-image").get("[data-test='per-style-count']").text(),
    ).toBe("0 overrides");
    expect(
      mountRow("z-image", { default_steps: 30 })
        .get("[data-test='per-style-count']")
        .text(),
    ).toBe("1 override");
    expect(
      mountRow("z-image", { default_steps: 30, lora: "anime", lora_scale: 0.8 })
        .get("[data-test='per-style-count']")
        .text(),
    ).toBe("3 overrides");
  });

  it("expands to every field a style may override, in the engine's order", async () => {
    const wrapper = mountRow("z-image");
    await expand(wrapper);
    const rows = wrapper.findAll("[data-test^='per-style-row-']");
    expect(rows.map((el) => el.attributes("data-test"))).toEqual(
      PER_STYLE_FIELDS.map((field) => `per-style-row-${field}`),
    );
    // Each row is NAMED for the engine field, because that is what
    // `mold config set models.<style>.<field>` takes.
    for (const [index, field] of PER_STYLE_FIELDS.entries()) {
      expect(rows[index]!.text(), field).toContain(field);
    }
  });

  it("saves and resets by the full key, so the caller needs no parsing", async () => {
    const wrapper = mountRow("z-image", { default_steps: 30 });
    await expand(wrapper);
    await wrapper
      .get(
        "[data-test='per-style-row-default_steps'] [data-test='setting-reset']",
      )
      .trigger("click");
    expect(wrapper.emitted("reset")).toEqual([
      ["models.z-image.default_steps"],
    ]);
  });
});
