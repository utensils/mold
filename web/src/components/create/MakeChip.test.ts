import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MakeChip from "./MakeChip.vue";
import Stepper from "@ui/components/Stepper.vue";

function factory(props: Partial<InstanceType<typeof MakeChip>["$props"]> = {}) {
  return mount(MakeChip, {
    props: { modelValue: 4, ...props },
    attachTo: document.body,
  });
}

/** The popover teleports to <body>, so the menu is read off the document. */
function menu(): HTMLElement | null {
  return document.body.querySelector("[data-test='make-menu']");
}

describe("MakeChip", () => {
  it("reads the count it will make", () => {
    const wrapper = factory();
    expect(wrapper.get("[data-test='make-chip']").text()).toContain("Make 4");
    wrapper.unmount();
  });

  it("opens a stepper popover and writes the new count", async () => {
    const wrapper = factory();
    expect(menu()).toBeNull();
    await wrapper.get("[data-test='make-chip']").trigger("click");
    const stepper = wrapper.findComponent(Stepper);
    expect(stepper.exists()).toBe(true);
    stepper.vm.$emit("update:modelValue", 6);
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([6]);
    wrapper.unmount();
  });

  /*
   * A batch is not one render with more pictures in it — every print is its
   * own job on the machine, which is what makes a large Make a queue decision
   * rather than a slider.
   */
  it("says what a batch actually is", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='make-chip']").trigger("click");
    expect(menu()?.textContent).toContain("Each one is queued separately");
    wrapper.unmount();
  });

  it("carries the bounds it is given through to the stepper", async () => {
    const wrapper = factory({ min: 2, max: 12 });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    const stepper = wrapper.findComponent(Stepper);
    expect(stepper.props("min")).toBe(2);
    expect(stepper.props("max")).toBe(12);
    wrapper.unmount();
  });

  /*
   * An edit recipe renders one print at a time. The chip then states the one
   * it will make and explains itself, rather than offering a count the
   * request validator would refuse.
   */
  it("locks to one, with the recipe's reason, on a batch-locked recipe", async () => {
    const wrapper = factory({
      modelValue: 4,
      locked: true,
      lockedReason: "Edit models render one print at a time.",
    });
    const chip = wrapper.get("[data-test='make-chip']");
    expect(chip.text()).toContain("Make 1");
    expect(chip.attributes("disabled")).toBeDefined();
    expect(chip.attributes("title")).toBe(
      "Edit models render one print at a time.",
    );
    await chip.trigger("click");
    expect(menu()).toBeNull();
    wrapper.unmount();
  });

  it("opens nothing while disabled", async () => {
    const wrapper = factory({ disabled: true });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    expect(menu()).toBeNull();
    wrapper.unmount();
  });
});
