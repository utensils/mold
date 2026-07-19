import { afterEach, describe, expect, it } from "vitest";
import { DOMWrapper, mount } from "@vue/test-utils";
import MissingModelDialog from "./MissingModelDialog.vue";

describe("MissingModelDialog", () => {
  afterEach(() => (document.body.innerHTML = ""));

  function bodyGet<T extends Element>(selector: string): DOMWrapper<T> {
    const element = document.body.querySelector<T>(selector);
    expect(element).not.toBeNull();
    return new DOMWrapper(element as T);
  }

  it("names the model, host, and size, and emits confirm / close", async () => {
    const wrapper = mount(MissingModelDialog, {
      props: { model: "jibmix-flux:fp8", hostLabel: "plato", sizeGb: 11.4 },
      attachTo: document.body,
    });
    const dialog = bodyGet("[data-test='missing-model-dialog']");
    expect(dialog.text()).toContain("jibmix-flux:fp8");
    expect(dialog.text()).toContain("plato");
    expect(dialog.text()).toContain("11.4 GB");

    await bodyGet("[data-test='missing-model-pull']").trigger("click");
    expect(wrapper.emitted("confirm")).toHaveLength(1);
    await bodyGet("[data-test='missing-model-cancel']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("omits the size parenthetical when unknown", () => {
    mount(MissingModelDialog, {
      props: { model: "m", hostLabel: "plato", sizeGb: null },
      attachTo: document.body,
    });
    expect(bodyGet("[data-test='missing-model-dialog']").text()).not.toContain("GB");
  });
});
