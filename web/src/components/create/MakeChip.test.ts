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
    // The hover text says what pressing the chip shows; the popover carries
    // the reason itself, so the two no longer say the same sentence.
    expect(chip.attributes("title")).toBe("Why one at a time");
    wrapper.unmount();
  });

  /*
   * A LOCKED chip is not a disabled one. Locked means "this style makes one
   * at a time, and here is why" — a fact worth reading — so the chip keeps its
   * caret, still opens, and answers the question instead of dimming and
   * swallowing the click. Only the hover title said anything, and a hover
   * title is not an answer on a touch screen.
   */
  it("opens and explains itself when locked, with no stepper to offer", async () => {
    const wrapper = factory({
      modelValue: 4,
      locked: true,
      lockedReason: "Edit models render one print at a time.",
    });
    const chip = wrapper.get("[data-test='make-chip']");
    expect(chip.attributes("disabled")).toBeUndefined();
    await chip.trigger("click");

    const opened = menu();
    expect(opened).not.toBeNull();
    expect(wrapper.findComponent(Stepper).exists()).toBe(false);
    expect(opened?.textContent).toContain("Make 1");
    expect(opened?.textContent).toContain(
      "Edit models render one print at a time.",
    );
    // The batch sentence is about a batch; there is no batch here.
    expect(opened?.textContent).not.toContain("Each one is queued separately");
    wrapper.unmount();
  });

  // A screen reader should not be asked "How many to make" by a panel that
  // offers no count.
  it("announces the locked panel for the question it answers", async () => {
    const wrapper = factory({ modelValue: 4, locked: true });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    const panel = document.body.querySelector("[role='dialog']");
    expect(panel?.getAttribute("aria-label")).toBe("Why one at a time");
    wrapper.unmount();
  });

  it("keeps the counting panel's own name when it is not locked", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='make-chip']").trigger("click");
    const panel = document.body.querySelector("[role='dialog']");
    expect(panel?.getAttribute("aria-label")).toBe("How many to make");
    wrapper.unmount();
  });

  it("opens with the count alone when a lock carries no reason", async () => {
    const wrapper = factory({ modelValue: 4, locked: true });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    expect(menu()?.textContent).toContain("Make 1");
    expect(wrapper.findComponent(Stepper).exists()).toBe(false);
    wrapper.unmount();
  });

  it("never writes a count while locked", async () => {
    const wrapper = factory({ modelValue: 4, locked: true });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    wrapper.unmount();
  });

  it("opens nothing while disabled", async () => {
    const wrapper = factory({ disabled: true });
    await wrapper.get("[data-test='make-chip']").trigger("click");
    expect(menu()).toBeNull();
    wrapper.unmount();
  });
});
