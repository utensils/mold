import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { QueuePlan } from "../api/queuePlan";
import QueuePlanWorkList from "./QueuePlanWorkList.vue";

function plan(): QueuePlan {
  return {
    plan_version: 1,
    state_version: 1,
    optimizer_state: "optimized",
    dirty_since_unix_ms: null,
    next_replan_at_unix_ms: null,
    work_items: [
      {
        work_id: "chain:67fa2076-stage-2",
        parent_id: "chain:67fa2076",
        work_kind: "chain_stage",
        chain_stage: 1,
        priority_class: "user",
        queue_rank: 0,
        bypass_count: 0,
        gpu: 2,
        lane_order: 0,
        estimate_confidence: "medium",
        activity_phase: "active",
      },
    ],
  };
}

describe("QueuePlanWorkList", () => {
  it("shows scheduler work even when no durable queue row represents it", () => {
    const wrapper = mount(QueuePlanWorkList, { props: { plan: plan() } });

    expect(wrapper.get('[data-test="planned-queue-row"]').text()).toContain(
      "Chain stage · stage 2",
    );
    expect(wrapper.text()).toContain("active");
    expect(wrapper.text()).toContain("GPU 2");
    expect(wrapper.get("code").attributes("title")).toBe("chain:67fa2076");
  });

  it("does not duplicate work already represented by a queue parent", () => {
    const wrapper = mount(QueuePlanWorkList, {
      props: { plan: plan(), excludeIds: ["chain:67fa2076"] },
    });

    expect(wrapper.find('[data-test="planned-queue-row"]').exists()).toBe(
      false,
    );
  });

  it("leaves blocked work to the recovery section when phase is absent", () => {
    const blocked = plan();
    delete blocked.work_items[0]!.activity_phase;
    blocked.work_items[0]!.blocked_reason = "no_capacity";
    const wrapper = mount(QueuePlanWorkList, { props: { plan: blocked } });

    expect(wrapper.find('[data-test="planned-queue-row"]').exists()).toBe(
      false,
    );
  });

  it("leaves phase-only blocked work to the recovery section", () => {
    const blocked = plan();
    blocked.work_items[0]!.activity_phase = "blocked";
    blocked.work_items[0]!.blocked_reason = null;
    const wrapper = mount(QueuePlanWorkList, { props: { plan: blocked } });

    expect(wrapper.find('[data-test="planned-queue-row"]').exists()).toBe(
      false,
    );
  });
});
