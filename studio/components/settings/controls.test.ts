import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { secretValuePresent } from "../../api/config";
import NumberControl from "./NumberControl.vue";
import PathControl from "./PathControl.vue";
import SecretControl from "./SecretControl.vue";
import SelectControl from "./SelectControl.vue";
import SliderControl from "./SliderControl.vue";
import TextControl from "./TextControl.vue";
import ToggleControl from "./ToggleControl.vue";

/*
 * The kit's controls are shared by web and desktop, so their contract is the
 * ACCESSIBLE one — `role`, `aria-*`, and the `commit` emit — never a class
 * name. Desktop's originals were asserted on Tailwind classes (`rounded-control`,
 * `bg-accent`, `bg-fg-dim`), which do not exist on web at all and could not
 * survive the move; the behaviour they stood for is pinned here instead.
 */

describe("ToggleControl", () => {
  const make = (modelValue: boolean, disabled = false) =>
    mount(ToggleControl, {
      props: { modelValue, disabled, ariaLabel: "Save every result" },
    });

  it("is a switch named by its aria-label", () => {
    const button = make(false).get("button");
    expect(button.attributes("role")).toBe("switch");
    expect(button.attributes("aria-checked")).toBe("false");
    expect(button.attributes("aria-label")).toBe("Save every result");
    expect(make(true).get("button").attributes("aria-checked")).toBe("true");
  });

  it("commits the opposite value on click", async () => {
    const wrapper = make(false);
    await wrapper.get("button").trigger("click");
    expect(wrapper.emitted("commit")).toEqual([[true]]);
  });

  it("commits nothing while disabled", async () => {
    const wrapper = make(false, true);
    await wrapper.get("button").trigger("click");
    expect(wrapper.emitted("commit")).toBeUndefined();
  });
});

describe("SelectControl", () => {
  const options = [
    { value: "", label: "auto" },
    { value: "q8_0", label: "q8_0" },
  ];

  it("offers every option and commits the chosen value", async () => {
    const wrapper = mount(SelectControl, {
      props: {
        modelValue: "",
        options,
        ariaLabel: "How FLUX reads your words",
      },
    });
    expect(wrapper.findAll("option").map((o) => o.text())).toEqual([
      "auto",
      "q8_0",
    ]);
    expect(wrapper.get("select").attributes("aria-label")).toBe(
      "How FLUX reads your words",
    );
    await wrapper.get("select").setValue("q8_0");
    expect(wrapper.emitted("commit")).toEqual([["q8_0"]]);
  });
});

describe("NumberControl", () => {
  const make = (modelValue: number | null = 250) =>
    mount(NumberControl, {
      props: { modelValue, ariaLabel: "Queue replan debounce" },
    });

  /* This control writes on blur, so every row a person TABS PAST is a write
   * unless it guards. `Number("")` is 0 and `Number("abc")` is NaN, either of
   * which the config client would happily send. */
  it("restores the stored value rather than sending a cleared field", async () => {
    const wrapper = make();
    const input = wrapper.get("input");
    // `setValue` raises `change`, which is the blur this control writes on.
    await input.setValue("");
    expect(wrapper.emitted("commit")).toBeUndefined();
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("restores the stored value rather than sending a non-numeric entry", async () => {
    const wrapper = make();
    const input = wrapper.get("input");
    await input.setValue("abc");
    expect(wrapper.emitted("commit")).toBeUndefined();
    expect((input.element as HTMLInputElement).value).toBe("250");
  });

  it("stays silent when a row is only tabbed through", async () => {
    const wrapper = make();
    await wrapper.get("input").trigger("change");
    expect(wrapper.emitted("commit")).toBeUndefined();
  });

  it("commits a real change, as a number", async () => {
    const wrapper = make();
    await wrapper.get("input").setValue("400");
    expect(wrapper.emitted("commit")).toEqual([[400]]);
  });
});

describe("TextControl", () => {
  it("commits only a real change, and can clear the field", async () => {
    const wrapper = mount(TextControl, {
      props: { modelValue: "hello", ariaLabel: "Backend" },
    });
    const input = wrapper.get("input");
    await input.trigger("change");
    expect(wrapper.emitted("commit")).toBeUndefined();
    await input.setValue("");
    expect(wrapper.emitted("commit")).toEqual([[""]]);
  });
});

describe("SliderControl", () => {
  it("shows the value it is dragging and commits on release", async () => {
    const wrapper = mount(SliderControl, {
      props: {
        modelValue: 0.5,
        min: 0,
        max: 2,
        step: 0.05,
        ariaLabel: "Temperature",
      },
    });
    const input = wrapper.get("input");
    // A drag is `input` events; only releasing raises `change`.
    (input.element as HTMLInputElement).value = "1.2";
    await input.trigger("input");
    expect(wrapper.text()).toContain("1.2");
    expect(wrapper.emitted("commit")).toBeUndefined();
    await input.trigger("change");
    expect(wrapper.emitted("commit")).toEqual([[1.2]]);
  });
});

describe("PathControl", () => {
  /* The native folder picker is injected, not imported: `desktop/src/lib/ipc`
   * is Tauri, and `scripts/tests/frontend-architecture.sh` refuses any Tauri
   * reference under studio/. A browser tab gets an editable path field. */
  it("offers Choose… only when a picker is supplied", () => {
    const web = mount(PathControl, {
      props: { modelValue: "/models", title: "Where styles are kept" },
    });
    expect(web.text()).not.toContain("Choose…");
    expect(web.find("input").exists()).toBe(true);

    const native = mount(PathControl, {
      props: {
        modelValue: "/models",
        title: "Where styles are kept",
        pick: async () => "/picked",
      },
    });
    expect(native.text()).toContain("Choose…");
  });

  it("commits what the picker returns, and nothing when it is dismissed", async () => {
    const wrapper = mount(PathControl, {
      props: {
        modelValue: "/models",
        title: "Where styles are kept",
        pick: async (title: string) => (title ? "/picked" : null),
      },
    });
    await wrapper.get("button").trigger("click");
    await Promise.resolve();
    expect(wrapper.emitted("commit")).toEqual([["/picked"]]);

    const dismissed = mount(PathControl, {
      props: { modelValue: "/models", title: "", pick: async () => null },
    });
    await dismissed.get("button").trigger("click");
    await Promise.resolve();
    expect(dismissed.emitted("commit")).toBeUndefined();
  });

  it("commits an edited path in a browser", async () => {
    const wrapper = mount(PathControl, {
      props: { modelValue: "/models", title: "Where styles are kept" },
    });
    await wrapper.get("input").setValue("/elsewhere");
    expect(wrapper.emitted("commit")).toEqual([["/elsewhere"]]);
  });
});

describe("SecretControl", () => {
  /* `get_static_value` answers a stored cloud key with the literal string
   * "<set>" (config_keys.rs:517,541). Rendering that as a value would put
   * `<set>` in the field and let someone save it as the key. */
  it("reads the engine's <set> sentinel as present, never as a value", () => {
    expect(secretValuePresent("<set>")).toBe(true);
    expect(secretValuePresent("sk-real-key")).toBe(true);
    expect(secretValuePresent("")).toBe(false);
    expect(secretValuePresent(null)).toBe(false);

    const wrapper = mount(SecretControl, {
      props: { present: true, ariaLabel: "RunPod key" },
    });
    expect(wrapper.text()).not.toContain("<set>");
    expect(wrapper.text()).toContain("set");
    expect(wrapper.find("input").exists()).toBe(false);
  });

  it("says so when nothing is stored", () => {
    const wrapper = mount(SecretControl, { props: { present: false } });
    expect(wrapper.text()).toContain("not set");
  });

  it("saves what is typed and clears what is stored", async () => {
    const wrapper = mount(SecretControl, {
      props: { present: true, clearable: true, ariaLabel: "RunPod key" },
    });
    await wrapper.get("[data-test='secret-edit']").trigger("click");
    const input = wrapper.get("input");
    expect(input.attributes("type")).toBe("password");
    await input.setValue("  sk-typed  ");
    await wrapper.get("[data-test='secret-save']").trigger("click");
    expect(wrapper.emitted("save")).toEqual([["sk-typed"]]);

    await wrapper.setProps({ present: true });
    await wrapper.get("[data-test='secret-clear']").trigger("click");
    expect(wrapper.emitted("clear")).toEqual([[]]);
  });

  it("saves nothing blank", async () => {
    const wrapper = mount(SecretControl, { props: { present: false } });
    await wrapper.get("[data-test='secret-edit']").trigger("click");
    await wrapper.get("input").setValue("   ");
    await wrapper.get("[data-test='secret-save']").trigger("click");
    expect(wrapper.emitted("save")).toBeUndefined();
  });
});

describe("SelectControl with a value the options do not list", () => {
  it("shows the machine's own value rather than rendering blank", () => {
    const wrapper = mount(SelectControl, {
      props: {
        modelValue: "q6_k",
        options: [
          { value: "auto", label: "auto" },
          { value: "q8_0", label: "q8_0" },
        ],
      },
    });
    const select = wrapper.get("select").element as HTMLSelectElement;
    expect(select.value).toBe("q6_k");
    expect(wrapper.findAll("option").map((o) => o.text())).toEqual([
      "q6_k",
      "auto",
      "q8_0",
    ]);
  });
});
