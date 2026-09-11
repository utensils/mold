import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MachineCard from "./MachineCard.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import StatusDot from "@ui/components/StatusDot.vue";

/** Decimal GB, the unit every Mold byte readout is written in. */
const GB = 1_000_000_000;

function factory(
  props: Partial<InstanceType<typeof MachineCard>["$props"]> = {},
  slots: Record<string, string> = {},
) {
  return mount(MachineCard, {
    props: { name: "studio-rack", status: "ready", ...props },
    slots,
    global: {
      stubs: {
        RouterLink: { props: ["to"], template: '<a :href="to"><slot /></a>' },
      },
    },
  });
}

describe("MachineCard", () => {
  it("leads with the machine it is talking to and a way to change it", () => {
    const wrapper = factory();
    expect(wrapper.findComponent(StatusDot).props("state")).toBe("online");
    expect(wrapper.get("[data-test='machine-card-name']").text()).toBe(
      "studio-rack",
    );
    const change = wrapper.get("[data-test='machine-card-change']");
    expect(change.text()).toBe("Change");
    expect(change.attributes("href")).toBe("/machines");
  });

  /*
   * The one sentence the mock puts on this card. A browser tab is not where
   * the work happens, and someone who closes it should already know that.
   */
  it("explains that the work outlives the tab", () => {
    expect(factory().get("[data-test='machine-card-sentence']").text()).toBe(
      "This tab is talking to a machine on your network. Close the tab and it keeps working.",
    );
  });

  it("says something else when it is handed something else", () => {
    const wrapper = factory({ sentence: "Making pictures on this Mac." });
    expect(wrapper.get("[data-test='machine-card-sentence']").text()).toBe(
      "Making pictures on this Mac.",
    );
  });

  it("reads a machine that is not ready with its own dot", () => {
    expect(
      factory({ status: "connecting" }).findComponent(StatusDot).props("state"),
    ).toBe("unknown");
    expect(
      factory({ status: "error" }).findComponent(StatusDot).props("state"),
    ).toBe("offline");
    expect(
      factory({ status: "offline" }).findComponent(StatusDot).props("state"),
    ).toBe("offline");
  });

  // The one byte formatter every other machine readout uses, so the rail and
  // the Machines card cannot state a machine's memory differently.
  it("meters memory in mono, and only once both numbers exist", () => {
    const wrapper = factory({ used: 14.9 * GB, total: 24 * GB });
    expect(wrapper.get("[data-test='machine-card-memory']").text()).toBe(
      "14.9 / 24.0 GB",
    );
    const bar = wrapper.findComponent(ProgressBar);
    expect(bar.props("height")).toBe(5);
    expect(Math.round(bar.props("value"))).toBe(62);
  });

  it("shows no meter for a machine that reports no memory", () => {
    const wrapper = factory({ used: null, total: null });
    expect(wrapper.findComponent(ProgressBar).exists()).toBe(false);
    expect(wrapper.find("[data-test='machine-card-memory']").exists()).toBe(
      false,
    );
  });

  it("says the queue depth it is told, zero included", () => {
    expect(
      factory({ queue: 0 }).get("[data-test='machine-card-queue']").text(),
    ).toBe("queue 0");
    expect(
      factory({ queue: 3 }).get("[data-test='machine-card-queue']").text(),
    ).toBe("queue 3");
    expect(factory().find("[data-test='machine-card-queue']").exists()).toBe(
      false,
    );
  });

  /*
   * "Where it runs" is only a question on a fleet. One machine has nothing to
   * choose between, so the picker does not take up the card.
   */
  it("mounts the routing picker only where more than one machine is reachable", () => {
    const picker = { picker: "<span data-test='picker'>Auto</span>" };
    expect(factory({}, picker).find("[data-test='picker']").exists()).toBe(
      false,
    );
    expect(
      factory({ multiHost: true }, picker)
        .find("[data-test='picker']")
        .exists(),
    ).toBe(true);
  });
});
