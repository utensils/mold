import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import QueueCard from "./QueueCard.vue";
import type { QueueEntry } from "../../types";

/** [running, A, B]: one running job in front of two queued jobs. Server queue
 *  positions count the running job, but the reorder PATCH indexes among queued
 *  jobs only, so the card must send queued-subset indices. */
function listing(): QueueEntry[] {
  return [
    {
      id: "run",
      model: "m",
      state: "running",
      started_at_unix_ms: 1,
      position: 0,
    },
    {
      id: "A",
      model: "m",
      state: "queued",
      started_at_unix_ms: 2,
      position: 1,
    },
    {
      id: "B",
      model: "m",
      state: "queued",
      started_at_unix_ms: 3,
      position: 2,
    },
  ];
}

function mountCard() {
  return mount(QueueCard, {
    props: { entries: listing(), gpuOrdinals: [0], canReorder: true },
  });
}

describe("QueueCard reorder index", () => {
  it("exposes advertised queue controls and disables lane changes for running jobs", async () => {
    const wrapper = mount(QueueCard, {
      props: {
        entries: listing(),
        gpuOrdinals: [0, 1],
        canReorder: true,
        canPause: true,
        canCancelAll: true,
        paused: false,
      },
    });
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Pause");
    await wrapper.get("[data-test='pause-toggle']").trigger("click");
    await wrapper.get("[data-test='cancel-all']").trigger("click");
    expect(wrapper.emitted("togglePause")).toHaveLength(1);
    expect(wrapper.emitted("cancelAll")).toHaveLength(1);
    expect(
      wrapper.findAll("[data-test='queue-lane']")[0]!.attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.findAll("[data-test='queue-lane']")[1]!.attributes("disabled"),
    ).toBeUndefined();
  });

  it("offers running cancellation only when the host advertises cooperative support", () => {
    const legacy = mount(QueueCard, {
      props: { entries: listing(), gpuOrdinals: [0] },
    });
    expect(legacy.findAll("[data-test='queue-cancel']")).toHaveLength(2);

    const current = mount(QueueCard, {
      props: { entries: listing(), gpuOrdinals: [0], canCancelRunning: true },
    });
    expect(current.findAll("[data-test='queue-cancel']")).toHaveLength(3);
  });

  it("preserves non-contiguous advertised GPU ordinals in lane values", () => {
    const wrapper = mount(QueueCard, {
      props: {
        entries: listing(),
        gpuOrdinals: [1, 3],
      },
    });

    const options = wrapper
      .find("[data-test='queue-lane']")
      .findAll("option")
      .map((option) => ({
        value: option.attributes("value"),
        label: option.text(),
      }));

    expect(options).toEqual([
      { value: "", label: "Auto" },
      { value: "1", label: "GPU 1" },
      { value: "3", label: "GPU 3" },
    ]);
  });

  it("moves the last queued job up to queued index 0, not its listing position", async () => {
    const wrapper = mountCard();
    // Only the two queued rows expose reorder buttons: index 0 = A, 1 = B.
    const upButtons = wrapper.findAll("[data-test='queue-up']");
    expect(upButtons).toHaveLength(2);
    await upButtons[1]!.trigger("click"); // move B up

    expect(wrapper.emitted("move")).toEqual([["B", 0]]);
  });

  it("moves the first queued job down to queued index 1, skipping the running job", async () => {
    const wrapper = mountCard();
    const downButtons = wrapper.findAll("[data-test='queue-down']");
    expect(downButtons).toHaveLength(2);
    await downButtons[0]!.trigger("click"); // move A down

    expect(wrapper.emitted("move")).toEqual([["A", 1]]);
  });
});

describe("held rows", () => {
  it("shows why a held job is parked, and still offers to clear it", async () => {
    // A held job exceeded its replay or dispatch budget: it will never start
    // on its own, so the reason and the cancel action are the only two things
    // that make the row actionable rather than merely puzzling.
    const wrapper = mount(QueueCard, {
      props: {
        gpuOrdinals: [],
        canCancelAll: true,
        entries: [
          {
            id: "srv-held",
            model: "flux2-klein",
            state: "held" as const,
            started_at_unix_ms: 1,
            position: 2,
            held_reason: "dispatch attempts exhausted",
          },
        ],
      },
    });

    expect(wrapper.text()).toContain("Held");
    expect(wrapper.text()).toContain("dispatch attempts exhausted");
    expect(wrapper.find("[data-test='queue-cancel']").exists()).toBe(true);
    expect(wrapper.find("[data-test='cancel-all']").exists()).toBe(false);
  });
});

describe("host-wide pause presentation", () => {
  it("updates queued row status while preserving a held row", () => {
    const wrapper = mount(QueueCard, {
      props: {
        paused: true,
        gpuOrdinals: [],
        entries: [
          { ...listing()[1]!, position: 0 },
          {
            ...listing()[2]!,
            state: "held" as const,
            held_reason: "operator action",
          },
        ],
      },
    });

    const rows = wrapper.findAll("[data-test='queue-row']");
    expect(rows[0]!.text()).toContain("Queue paused");
    expect(rows[1]!.text()).toContain("Held");
  });
});

describe("cancel all", () => {
  it("is hidden when every visible job is already running", () => {
    const wrapper = mount(QueueCard, {
      props: {
        canCancelAll: true,
        gpuOrdinals: [],
        entries: [listing()[0]!],
      },
    });

    expect(wrapper.find("[data-test='cancel-all']").exists()).toBe(false);
  });
});

describe("restart-paused rows", () => {
  it("offers Resume even when the global pause flag is false", async () => {
    const wrapper = mount(QueueCard, {
      props: {
        canPause: true,
        canCancelAll: true,
        paused: false,
        gpuOrdinals: [],
        entries: [
          {
            id: "srv-paused",
            model: "flux2-klein",
            state: "paused" as const,
            started_at_unix_ms: 1,
            position: 0,
          },
        ],
      },
    });

    expect(wrapper.get("[data-test='paused-chip']").text()).toContain(
      "paused after restart",
    );
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Resume");
    expect(wrapper.find("[data-test='cancel-all']").exists()).toBe(true);
    await wrapper.get("[data-test='pause-toggle']").trigger("click");
    expect(wrapper.emitted("togglePause")).toHaveLength(1);
  });

  it("keeps an ordinary queued row waiting when only another row is restart-paused", () => {
    const wrapper = mount(QueueCard, {
      props: {
        canPause: true,
        paused: false,
        gpuOrdinals: [],
        entries: [
          {
            id: "srv-paused",
            model: "flux2-klein",
            state: "paused" as const,
            started_at_unix_ms: 1,
            position: 0,
          },
          { ...listing()[1]!, position: 1 },
        ],
      },
    });

    const rows = wrapper.findAll("[data-test='queue-row']");
    expect(rows[0]!.text()).toContain("Paused");
    expect(rows[1]!.text()).toContain("#1 in line");
    expect(rows[1]!.text()).not.toContain("Queue paused");
    expect(wrapper.get("[data-test='pause-toggle']").text()).toBe("Resume");
  });
});

describe("scheduler plan work", () => {
  it("does not claim the queue is empty when the scheduler has active work", () => {
    const wrapper = mount(QueueCard, {
      props: {
        entries: [],
        gpuOrdinals: [0],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "chain-stage-2",
              parent_id: "chain-parent",
              work_kind: "chain_stage",
              chain_stage: 1,
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              gpu: 0,
              estimate_confidence: "medium",
              activity_phase: "active",
            },
          ],
        },
      },
    });

    expect(wrapper.find("[data-test='queue-empty']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='planned-queue-row']")).toHaveLength(1);
  });
});
