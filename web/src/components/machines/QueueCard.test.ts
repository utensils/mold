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
    props: { entries: listing(), gpuCount: 1, canReorder: true },
  });
}

describe("QueueCard reorder index", () => {
  it("exposes advertised queue controls and disables lane changes for running jobs", async () => {
    const wrapper = mount(QueueCard, {
      props: {
        entries: listing(),
        gpuCount: 2,
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
