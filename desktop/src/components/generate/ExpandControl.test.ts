import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ExpandControl from "./ExpandControl.vue";

const wrappers: ReturnType<typeof mount>[] = [];

function mountControl(
  props: Partial<{
    prompt: string;
    batchSize: number;
    running: boolean;
    hostLabel: string | null;
    canUndo: boolean;
  }> = {},
) {
  const wrapper = mount(ExpandControl, {
    props: {
      prompt: "a cat",
      batchSize: 1,
      running: false,
      hostLabel: "Studio 4090",
      canUndo: false,
      ...props,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("ExpandControl", () => {
  it("preserves the Batch 1 quick-expand and undo interaction", async () => {
    const wrapper = mountControl({ canUndo: true });
    await wrapper.get('button[title="Expand prompt"]').trigger("click");
    expect(wrapper.emitted("expand")).toHaveLength(1);

    await wrapper.get('button[aria-label="Restore original prompt"]').trigger("click");
    expect(wrapper.emitted("restore")).toHaveLength(1);
  });

  it("turns Batch N into a preparation action", async () => {
    const wrapper = mountControl({ batchSize: 3 });
    const button = wrapper.get('[data-test="expand-action"]');
    expect(button.text()).toContain("Prepare 3 variations");
    await button.trigger("click");
    expect(wrapper.emitted("expand")).toHaveLength(1);
    expect(wrapper.find('[aria-label="Restore original prompt"]').exists()).toBe(false);
  });

  it("announces progress with the frozen host and requested count", () => {
    const wrapper = mountControl({ batchSize: 5, running: true });
    expect(wrapper.get('[role="status"]').text()).toBe("Expanding 5 prompts on Studio 4090…");
    expect(wrapper.get('[data-test="expand-action"]').attributes("disabled")).toBeDefined();
  });

  it("keeps the exposed keyboard action wired to the visible action", () => {
    const wrapper = mountControl({ batchSize: 3 });
    (wrapper.vm as unknown as { expand: () => void }).expand();
    expect(wrapper.emitted("expand")).toHaveLength(1);
  });
});
