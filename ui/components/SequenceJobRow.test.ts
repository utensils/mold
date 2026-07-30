import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SequenceJobRow from "./SequenceJobRow.vue";
import { sequenceToVM } from "@studio/lib/activity";
import type { ActivityJobVM } from "@studio/lib/activity";
import type { ChainJobSummary } from "@studio/lib/api/chainTypes";

function vm(
  overrides: Partial<ChainJobSummary> = {},
): ActivityJobVM & { kind: "sequence" } {
  const summary: ChainJobSummary = {
    id: "job-1",
    state: "running",
    model: "ltx-2.3-22b-distilled:fp8",
    stage_count: 5,
    current_stage: 2,
    created_at_unix_ms: 10,
    updated_at_unix_ms: 20,
    error: null,
    ...overrides,
  };
  const row = sequenceToVM(
    summary,
    { hostId: "plato-7680", hostLabel: "plato" },
    {
      step: 4,
      total: 8,
    },
  );
  if (row.kind !== "sequence") throw new Error("unreachable");
  return row;
}

describe("SequenceJobRow", () => {
  it("renders state, model, clip count, and host", () => {
    const wrapper = mount(SequenceJobRow, {
      props: { vm: vm(), modelLabel: "LTX-2.3 22B Distilled" },
    });
    const text = wrapper.text();
    expect(text).toContain("running");
    expect(text).toContain("LTX-2.3 22B Distilled");
    expect(text).toContain("5 clips");
    expect(text).toContain("3/5");
    expect(text).toContain("plato");
  });

  it("falls back to the raw model id when no display label is supplied", () => {
    const wrapper = mount(SequenceJobRow, { props: { vm: vm() } });
    expect(wrapper.text()).toContain("ltx-2.3-22b-distilled:fp8");
  });

  it("emits the action id the button carries", async () => {
    const wrapper = mount(SequenceJobRow, { props: { vm: vm() } });
    await wrapper.get("[data-test='seq-cancel']").trigger("click");
    expect(wrapper.emitted("action")?.[0]?.[0]).toBe("cancel");
    expect(wrapper.emitted("select")).toBeUndefined();
  });

  it("selects the row by pointer or keyboard without stealing action clicks", async () => {
    const wrapper = mount(SequenceJobRow, { props: { vm: vm() } });
    const row = wrapper.get("[data-test='sequence-job-row']");
    expect(row.attributes("role")).toBe("button");
    await row.trigger("click");
    await row.trigger("keydown", { key: "Enter" });
    await row.trigger("keydown", { key: " " });
    expect(wrapper.emitted("select")).toHaveLength(3);
  });

  it("labels watch as Open once the job has settled", () => {
    const running = mount(SequenceJobRow, { props: { vm: vm() } });
    expect(running.get("[data-test='seq-watch']").text()).toBe("Watch");
    const done = mount(SequenceJobRow, {
      props: { vm: vm({ state: "completed" }) },
    });
    expect(done.get("[data-test='seq-watch']").text()).toBe("Open");
  });

  it("shows the error text of a failed job", () => {
    const wrapper = mount(SequenceJobRow, {
      props: { vm: vm({ state: "failed", error: "chain stage 2 blew up" }) },
    });
    expect(wrapper.text()).toContain("chain stage 2 blew up");
    expect(wrapper.find("[data-test='seq-resume']").exists()).toBe(true);
  });

  it("expands a failed job so its complete diagnostic can be inspected", async () => {
    const diagnostic =
      "CUDA allocation failed while reserving the transformer cache on device 1; requested 8.4 GiB with 3.1 GiB available";
    const wrapper = mount(SequenceJobRow, {
      props: { vm: vm({ state: "failed", error: diagnostic }) },
    });
    const disclosure = wrapper.get("[data-test='seq-error-disclosure']");

    expect(disclosure.attributes("aria-expanded")).toBe("false");
    expect(disclosure.text()).toContain(
      "plato ran out of memory. Try a smaller model, image size, or batch.",
    );
    await disclosure.trigger("click");

    expect(disclosure.attributes("aria-expanded")).toBe("true");
    expect(disclosure.classes()).toContain("ms-seqrow__error--expanded");
    expect(wrapper.emitted("select")).toBeUndefined();
  });

  it("drops the progress region in the dense variant", () => {
    expect(
      mount(SequenceJobRow, { props: { vm: vm() } })
        .find("[data-test='seq-progress']")
        .exists(),
    ).toBe(true);
    expect(
      mount(SequenceJobRow, { props: { vm: vm(), dense: true } })
        .find("[data-test='seq-progress']")
        .exists(),
    ).toBe(false);
  });

  it("renders only the actions it is told to, when narrowed", async () => {
    const wrapper = mount(SequenceJobRow, {
      props: { vm: vm({ state: "completed" }), actions: ["delete"] },
    });
    expect(wrapper.find("[data-test='seq-watch']").exists()).toBe(false);
    await wrapper.get("[data-test='seq-delete']").trigger("click");
    expect(wrapper.emitted("action")?.[0]?.[0]).toBe("delete");
  });
});
