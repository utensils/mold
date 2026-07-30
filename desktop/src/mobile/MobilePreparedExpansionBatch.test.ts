import { afterEach, describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPreparedExpansionBatch } from "../lib/preparedExpansion";
import MobilePreparedExpansionBatch from "./MobilePreparedExpansionBatch.vue";

const route = {
  hostId: "studio",
  label: "Studio",
  kind: "remote" as const,
  target: { baseUrl: "https://studio.test", apiKey: "secret" },
  instanceId: "studio-instance",
};
const batch = () =>
  createPreparedExpansionBatch(
    {
      sourcePrompt: "source",
      model: "flux-dev:q8",
      family: "flux",
      requestedCount: 3,
      stylePreset: null,
      selectedHostPolicy: "studio",
    },
    route,
    ["one", "two", "three"],
    1,
    "mobile-batch",
  );

const wrappers: ReturnType<typeof mount>[] = [];
function mountBatch(overrides: Record<string, unknown> = {}) {
  const wrapper = mount(MobilePreparedExpansionBatch, {
    attachTo: document.body,
    props: {
      batch: batch(),
      staleReasons: [],
      preparing: false,
      error: "",
      submitting: false,
      ...overrides,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
  document.body.innerHTML = "";
});

describe("MobilePreparedExpansionBatch", () => {
  it("renders numbered 16px editors and 44pt touch actions", () => {
    const wrapper = mountBatch();
    expect(wrapper.findAll("textarea")).toHaveLength(3);
    expect(wrapper.get('textarea[aria-label="Variation 1 of 3"]').classes()).toContain(
      "mobile-prepared-editor",
    );
    expect(wrapper.get('[data-test="mobile-prepared-remove"]').classes()).toContain(
      "mobile-touch-action",
    );
    expect(wrapper.get('[data-test="mobile-develop-prepared"]').text()).toBe(
      "Develop 3 variations",
    );
  });

  it("edits and removes while retaining work", async () => {
    const wrapper = mountBatch();
    await wrapper.findAll("textarea")[0]!.setValue("edited one");
    expect(wrapper.emitted("edit")).toEqual([[{ id: "prepared-1-1", text: "edited one" }]]);
    await wrapper.findAll('[data-test="mobile-prepared-remove"]')[1]!.trigger("click");
    expect(wrapper.emitted("remove")).toEqual([["prepared-1-2"]]);
  });

  it("keeps large reviews compact until Review all is requested", async () => {
    const large = createPreparedExpansionBatch(
      {
        sourcePrompt: "source",
        model: "flux-dev:q8",
        family: "flux",
        requestedCount: 120,
        stylePreset: null,
        selectedHostPolicy: "studio",
      },
      route,
      Array.from({ length: 120 }, (_, index) => `variation ${index + 1}`),
      1,
      "large-mobile-batch",
    );
    const wrapper = mountBatch({ batch: large });

    expect(wrapper.findAll("textarea")).toHaveLength(8);
    expect(wrapper.get('[data-test="mobile-prepared-compact-review"]').text()).toContain(
      "112 more variations",
    );
    expect(wrapper.get('[data-test="mobile-develop-prepared"]').text()).toBe(
      "Develop 120 variations",
    );

    await wrapper.get('[data-test="mobile-prepared-review-all"]').trigger("click");
    expect(wrapper.findAll("textarea")).toHaveLength(50);
    expect(wrapper.get('[data-test="mobile-prepared-review-range"]').text()).toContain(
      "Variations 1–50 of 120",
    );
    await wrapper.get('[data-test="mobile-prepared-review-next"]').trigger("click");
    expect(wrapper.findAll("textarea")).toHaveLength(50);
    expect(wrapper.get('[data-test="mobile-prepared-review-range"]').text()).toContain(
      "Variations 51–100 of 120",
    );
    expect(wrapper.get('textarea[aria-label="Variation 51 of 120"]').attributes("aria-label")).toBe(
      "Variation 51 of 120",
    );

    await wrapper.get('[data-test="mobile-prepared-review-next"]').trigger("click");
    const shortened = { ...large, prompts: large.prompts.slice(0, 40) };
    await wrapper.setProps({ batch: shortened });
    expect(wrapper.findAll("textarea")).toHaveLength(40);
    expect(wrapper.get('[data-test="mobile-prepared-review-range"]').text()).toContain(
      "Variations 1–40 of 40",
    );
  });

  it("requires explicit confirmation before collapsing two prompts to Batch 1", async () => {
    const two = batch();
    two.prompts.pop();
    two.requestedCount = 2;
    const wrapper = mountBatch({ batch: two });
    await wrapper.get('[data-test="mobile-prepared-remove"]').trigger("click");
    expect(wrapper.emitted("collapse")).toBeUndefined();
    expect(wrapper.get('[role="alertdialog"]').text()).toContain("Continue with one prompt?");
    expect(document.activeElement).toBe(
      wrapper.get('[data-test="mobile-confirm-collapse"]').element,
    );
    await wrapper.get('[data-test="mobile-confirm-collapse"]').trigger("click");
    expect(wrapper.emitted("collapse")).toEqual([["prepared-1-1"]]);
  });

  it("clears a two-to-one confirmation when the prepared batch is replaced", async () => {
    const two = batch();
    two.prompts.pop();
    two.requestedCount = 2;
    const wrapper = mountBatch({ batch: two });
    await wrapper.get('[data-test="mobile-prepared-remove"]').trigger("click");
    expect(wrapper.find('[role="alertdialog"]').exists()).toBe(true);

    const replacement = batch();
    replacement.prompts = replacement.prompts.slice(0, 2);
    replacement.requestedCount = 2;
    replacement.batchId = "replacement-batch";
    await wrapper.setProps({ batch: replacement });
    await flushPromises();

    expect(wrapper.find('[role="alertdialog"]').exists()).toBe(false);
    expect(wrapper.emitted("collapse")).toBeUndefined();
  });

  it("names stale reasons, blocks Develop, and leaves replacement focus to its parent", async () => {
    const wrapper = mountBatch({ staleReasons: ["Studio's connection details changed."] });
    expect(wrapper.get('[role="alert"]').text()).toContain("connection details changed");
    expect(wrapper.get('[data-test="mobile-develop-prepared"]').attributes("disabled")).toBe("");
    const refresh = wrapper.get('[data-test="mobile-refresh-prepared"]');
    (refresh.element as HTMLButtonElement).focus();
    await refresh.trigger("click");
    const replacement = batch();
    replacement.batchId = "replacement";
    await wrapper.setProps({ batch: replacement, staleReasons: [] });
    await flushPromises();
    expect(wrapper.emitted("refresh")).toHaveLength(1);

    await wrapper.setProps({ staleReasons: ["Model changed."] });
    const refreshAgain = wrapper.get('[data-test="mobile-refresh-prepared"]');
    (refreshAgain.element as HTMLButtonElement).focus();
    await refreshAgain.trigger("click");
    const outside = document.createElement("button");
    document.body.append(outside);
    outside.focus();
    const newer = batch();
    newer.batchId = "replacement-2";
    await wrapper.setProps({ batch: newer, staleReasons: [] });
    await flushPromises();
    expect(document.activeElement).toBe(outside);
  });
});
