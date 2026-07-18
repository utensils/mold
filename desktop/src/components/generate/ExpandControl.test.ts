import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ExpandControl from "./ExpandControl.vue";
import { expandPrompt } from "../../lib/api/expand";

vi.mock("../../lib/api/expand", () => ({
  expandPrompt: vi.fn(),
}));
vi.mock("../../lib/api/catalog", () => ({
  startCatalogDownload: vi.fn(),
}));

const expandPromptMock = vi.mocked(expandPrompt);

// The component registers document-level mousedown/keydown listeners on mount,
// so every mounted wrapper must be unmounted to avoid cross-test leakage.
const wrappers: ReturnType<typeof mount>[] = [];

function mountControl(props: Partial<{ prompt: string; family: string }> = {}) {
  const wrapper = mount(ExpandControl, {
    props: { prompt: "a cat", family: "flux", ...props },
  });
  wrappers.push(wrapper);
  return wrapper;
}

async function openPopover(wrapper: ReturnType<typeof mountControl>) {
  await wrapper.get('[data-test="expand-options"]').trigger("click");
}

beforeEach(() => {
  setActivePinia(createPinia());
  expandPromptMock.mockReset();
});

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("ExpandControl one-shot fast path (regression)", () => {
  it("sends a single variation and applies the first candidate", async () => {
    expandPromptMock.mockResolvedValue({
      original: "a cat",
      expanded: ["a detailed cat", "another cat"],
    });
    const wrapper = mountControl();
    await wrapper.get('button[title="Expand prompt"]').trigger("click");
    await flushPromises();

    expect(expandPromptMock).toHaveBeenCalledTimes(1);
    expect(expandPromptMock).toHaveBeenCalledWith("a cat", {
      variations: 1,
      modelFamily: "flux",
    });
    expect(wrapper.emitted("apply")).toEqual([[{ expanded: "a detailed cat", original: "a cat" }]]);
  });

  it("omits the family when none is set", async () => {
    expandPromptMock.mockResolvedValue({ original: "a cat", expanded: ["x"] });
    const wrapper = mountControl({ family: "" });
    await wrapper.get('button[title="Expand prompt"]').trigger("click");
    await flushPromises();
    expect(expandPromptMock).toHaveBeenCalledWith("a cat", { variations: 1 });
  });

  it("undo restores the original prompt", async () => {
    expandPromptMock.mockResolvedValue({ original: "a cat", expanded: ["a detailed cat"] });
    const wrapper = mountControl();
    await wrapper.get('button[title="Expand prompt"]').trigger("click");
    await flushPromises();

    await wrapper.get('button[aria-label="Restore original prompt"]').trigger("click");
    expect(wrapper.emitted("restore")).toEqual([["a cat"]]);
  });
});

describe("ExpandControl variations popover", () => {
  it("preview fetches the selected variation count without touching the prompt", async () => {
    expandPromptMock.mockResolvedValue({ original: "a cat", expanded: ["a", "b", "c"] });
    const wrapper = mountControl();
    await openPopover(wrapper);
    await wrapper.get('[data-test="expand-variations-3"]').trigger("click");
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();

    expect(expandPromptMock).toHaveBeenCalledWith("a cat", {
      variations: 3,
      modelFamily: "flux",
    });
    // Preview must not mutate the prompt — no apply, only a candidate list.
    expect(wrapper.emitted("apply")).toBeUndefined();
    const candidates = wrapper.findAll('[data-test="expand-candidate"]');
    expect(candidates.map((c) => c.text())).toEqual(["a", "b", "c"]);
  });

  it("clicking a candidate applies exactly that candidate and undo restores", async () => {
    expandPromptMock.mockResolvedValue({ original: "a cat", expanded: ["a", "b", "c"] });
    const wrapper = mountControl();
    await openPopover(wrapper);
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();

    await wrapper.findAll('[data-test="expand-candidate"]')[1]!.trigger("click");
    expect(wrapper.emitted("apply")).toEqual([[{ expanded: "b", original: "a cat" }]]);

    await wrapper.get('button[aria-label="Restore original prompt"]').trigger("click");
    expect(wrapper.emitted("restore")).toEqual([["a cat"]]);
  });

  it("sends the family override instead of the current family when provided", async () => {
    expandPromptMock.mockResolvedValue({ original: "a cat", expanded: ["a"] });
    const wrapper = mountControl();
    await openPopover(wrapper);
    await wrapper.get('[data-test="expand-family"]').setValue("sdxl");
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();

    expect(expandPromptMock).toHaveBeenCalledWith("a cat", {
      variations: 1,
      modelFamily: "sdxl",
    });
  });

  it("shows the current family as the override placeholder", async () => {
    const wrapper = mountControl();
    await openPopover(wrapper);
    expect(wrapper.get('[data-test="expand-family"]').attributes("placeholder")).toBe("flux");
  });

  it("surfaces preview errors inline without applying anything", async () => {
    expandPromptMock.mockRejectedValue(new Error("backend down"));
    const wrapper = mountControl();
    await openPopover(wrapper);
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-test="expand-preview-error"]').text()).toContain("backend down");
    expect(wrapper.emitted("apply")).toBeUndefined();
  });

  it("clears stale candidates when a later preview fails", async () => {
    expandPromptMock.mockResolvedValueOnce({ original: "a cat", expanded: ["a", "b"] });
    const wrapper = mountControl();
    await openPopover(wrapper);
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();
    expect(wrapper.findAll('[data-test="expand-candidate"]')).toHaveLength(2);

    // A failed re-preview must not leave the earlier suggestions clickable.
    expandPromptMock.mockRejectedValueOnce(new Error("backend down"));
    await wrapper.get('[data-test="expand-preview"]').trigger("click");
    await flushPromises();

    expect(wrapper.findAll('[data-test="expand-candidate"]')).toHaveLength(0);
    expect(wrapper.get('[data-test="expand-preview-error"]').text()).toContain("backend down");
  });
});
