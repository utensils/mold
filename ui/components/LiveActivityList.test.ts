import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { FleetActiveWork } from "@studio/api/activity";
import LiveActivityList from "./LiveActivityList.vue";

const row: FleetActiveWork = {
  id: "job-1",
  key: "host-1:generation:job-1",
  kind: "generation",
  phase: "queued",
  model: "FLUX",
  created_at_unix_ms: 1,
  updated_at_unix_ms: 2,
  can_cancel: true,
  hostId: "host-1",
  hostLabel: "hal9000",
  routeUrl: "http://hal9000:7680",
  instanceId: "instance-1",
  stale: false,
  hostError: null,
};

function touch(type: string, x: number, y: number, ended = false): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  const point = { clientX: x, clientY: y };
  Object.defineProperty(event, "touches", { value: ended ? [] : [point] });
  Object.defineProperty(event, "changedTouches", { value: [point] });
  return event;
}

function mountList() {
  return mount(LiveActivityList, {
    props: { rows: [row], interactive: true, swipeActions: true },
    slots: { actions: '<button type="button">Cancel</button>' },
  });
}

describe("LiveActivityList swipe actions", () => {
  it("opens actions for a clearly horizontal swipe", async () => {
    const wrapper = mountList();
    const item = wrapper.get(".live-activity-row");
    item.element.dispatchEvent(touch("touchstart", 260, 100));
    item.element.dispatchEvent(touch("touchmove", 160, 102));
    item.element.dispatchEvent(touch("touchend", 160, 102, true));
    await wrapper.vm.$nextTick();
    expect(item.classes()).toContain("live-activity-row--actions-open");
  });

  it("keeps actions closed when diagonal startup becomes a vertical scroll", async () => {
    const wrapper = mountList();
    const item = wrapper.get(".live-activity-row");
    item.element.dispatchEvent(touch("touchstart", 300, 100));
    const ambiguous = touch("touchmove", 287, 111);
    item.element.dispatchEvent(ambiguous);
    expect(ambiguous.defaultPrevented).toBe(false);
    expect(item.classes()).not.toContain("live-activity-row--actions-open");

    const vertical = touch("touchmove", 280, 160);
    item.element.dispatchEvent(vertical);
    item.element.dispatchEvent(touch("touchend", 280, 160, true));
    await wrapper.vm.$nextTick();
    expect(vertical.defaultPrevented).toBe(false);
    expect(item.classes()).not.toContain("live-activity-row--actions-open");
  });
});
