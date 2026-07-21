import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import PodCostMeter from "./PodCostMeter.vue";

describe("PodCostMeter", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the accrued cost and hourly rate", () => {
    // 1h of uptime at $1.20/hr → $1.20 accrued.
    const wrapper = mount(PodCostMeter, {
      props: { costPerHr: 1.2, uptimeSeconds: 3600 },
    });
    expect(wrapper.get("[data-test='pod-accrued']").text()).toBe("≈$1.20");
    expect(wrapper.text()).toContain("$1.20/hr");
  });

  it("ticks the accrued cost forward over time", async () => {
    const wrapper = mount(PodCostMeter, {
      props: { costPerHr: 3.6, uptimeSeconds: 0 },
    });
    expect(wrapper.get("[data-test='pod-accrued']").text()).toBe("≈$0.00");
    // Advance one hour of wall-clock time; the 30 s tick re-reads the clock.
    vi.advanceTimersByTime(60 * 60 * 1000);
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='pod-accrued']").text()).toBe("≈$3.60");
  });

  it("only shows Stop when stoppable, and emits it", async () => {
    const hidden = mount(PodCostMeter, { props: { costPerHr: 1, uptimeSeconds: 10 } });
    expect(hidden.find("[data-test='pod-cost-stop']").exists()).toBe(false);

    const wrapper = mount(PodCostMeter, {
      props: { costPerHr: 1, uptimeSeconds: 10, stoppable: true },
    });
    await wrapper.get("[data-test='pod-cost-stop']").trigger("click");
    expect(wrapper.emitted("stop")).toHaveLength(1);
  });

  it("re-anchors when a fresh uptime snapshot arrives so it never drifts", async () => {
    const wrapper = mount(PodCostMeter, { props: { costPerHr: 3.6, uptimeSeconds: 0 } });
    vi.advanceTimersByTime(30 * 60 * 1000); // 30 min elapsed → ≈$1.80
    await wrapper.vm.$nextTick();
    // The next poll delivers uptime=3600 (1h) at this instant; accrued jumps to $3.60.
    await wrapper.setProps({ uptimeSeconds: 3600 });
    expect(wrapper.get("[data-test='pod-accrued']").text()).toBe("≈$3.60");
  });
});
