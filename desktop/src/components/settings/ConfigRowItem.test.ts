import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ConfigRowItem from "./ConfigRowItem.vue";
import type { ConfigRow } from "../../lib/api/types";

/*
 * Advanced's raw rows write straight to `/api/config`, and this component
 * commits on blur — which means every row a user TABS PAST is a write. It
 * must therefore write only what the user actually changed, and never turn a
 * cleared numeric field into a value the engine will honour: `Number("")` is
 * 0, and `Number("abc")` is NaN, which the config client serialises as null.
 */

function row(over: Partial<ConfigRow> = {}): ConfigRow {
  return {
    key: "scheduler.replan_debounce_ms",
    value: 250,
    source: "db",
    env_var: null,
    restart_required: false,
    ...over,
  } as ConfigRow;
}

function mountRow(over: Partial<ConfigRow> = {}) {
  const wrapper = mount(ConfigRowItem, { props: { row: row(over) } });
  return { wrapper, input: wrapper.get("input") };
}

describe("ConfigRowItem commits", () => {
  it("never turns a cleared numeric field into 0", async () => {
    const { wrapper, input } = mountRow();
    await input.setValue("");
    await input.trigger("blur");

    expect(wrapper.emitted("save")).toBeUndefined();
    // The field snaps back to the value the engine still holds, so the row
    // does not read as blank while the engine reads 250.
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("never sends a non-numeric entry as null", async () => {
    const { wrapper, input } = mountRow();
    await input.setValue("abc");
    await input.trigger("blur");

    expect(wrapper.emitted("save")).toBeUndefined();
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("stays silent when a row is only tabbed through", async () => {
    const { wrapper, input } = mountRow({ value: "hello", key: "some.text" });
    await input.trigger("blur");
    await input.trigger("keydown.enter");

    expect(wrapper.emitted("save")).toBeUndefined();
  });

  it("still saves a real numeric change, as a number", async () => {
    const { wrapper, input } = mountRow();
    await input.setValue("400");
    await input.trigger("blur");

    expect(wrapper.emitted("save")).toEqual([[400]]);
  });

  it("still saves a real text change, and can clear a text row", async () => {
    const { wrapper, input } = mountRow({ value: "hello", key: "some.text" });
    await input.setValue("");
    await input.trigger("blur");

    expect(wrapper.emitted("save")).toEqual([[""]]);
  });

  it("saves a checkbox on change", async () => {
    const wrapper = mount(ConfigRowItem, {
      props: { row: row({ value: false, key: "some.flag" }) },
    });
    const box = wrapper.get("input[type='checkbox']");
    await box.setValue(true);

    expect(wrapper.emitted("save")).toEqual([[true]]);
  });
});
