import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { PROMPT_IGNORED_TRANSFORM_REASON } from "@studio/lib/promptTransform";
import MobileRemixReview from "./MobileRemixReview.vue";

const wrappers: ReturnType<typeof mount>[] = [];

function mountReview(overrides: Record<string, unknown> = {}) {
  const wrapper = mount(MobileRemixReview, {
    props: {
      sourceKind: "original" as const,
      sourcePrompt: "an armchair shaped like an avocado",
      variants: [
        { id: "remix-1", prompt: "one", dimensions: ["composition" as const], selected: false },
        { id: "remix-2", prompt: "two", dimensions: ["camera" as const], selected: false },
      ],
      hostLabel: "Studio",
      staleReasons: [],
      running: false,
      error: "",
      ...overrides,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("MobileRemixReview", () => {
  it("re-remixes on the frozen route by default", async () => {
    const wrapper = mountReview();
    const reremix = wrapper.get("[data-test='mobile-reremix']");
    expect(reremix.attributes("disabled")).toBeUndefined();
    await reremix.trigger("click");
    expect(wrapper.emitted("reremix")).toHaveLength(1);
  });

  /**
   * Reviewed variants survive a model switch; the recipe they would be
   * re-requested against may not read a prompt at all. Re-remix is refused
   * with the reason, while applying or discarding what is already reviewed
   * stays available.
   */
  it("refuses Re-remix once the recipe ignores the prompt", async () => {
    const wrapper = mountReview({ blockedReason: PROMPT_IGNORED_TRANSFORM_REASON });
    const reremix = wrapper.get("[data-test='mobile-reremix']");
    expect(reremix.attributes("disabled")).toBe("");
    await reremix.trigger("click");
    expect(wrapper.emitted("reremix")).toBeUndefined();
    expect(wrapper.get("[data-test='mobile-remix-transform-blocked']").text()).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
    expect(
      wrapper.get("[data-test='mobile-remix-discard']").attributes("disabled"),
    ).toBeUndefined();
  });
});
