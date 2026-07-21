import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExpandModal from "./ExpandModal.vue";
import type { ExpandFormState } from "../types";

const expandPromptMock = vi.hoisted(() =>
  vi.fn(async () => ({ original: "a lighthouse", expanded: ["storm light"] })),
);
vi.mock("../api", () => ({ expandPrompt: expandPromptMock }));

const expand: ExpandFormState = {
  enabled: false,
  variations: 1,
  familyOverride: null,
};

function factory(props: Record<string, unknown> = {}) {
  return mount(ExpandModal, {
    attachTo: document.body,
    // The modal teleports to <body>; keep it inline so the wrapper can find it.
    global: { stubs: { teleport: true } },
    props: {
      open: true,
      prompt: "a lighthouse",
      expand,
      currentModel: null,
      ...props,
    },
  });
}

describe("ExpandModal", () => {
  beforeEach(() => {
    expandPromptMock.mockClear();
    document.body.innerHTML = "";
  });

  it("carries the composer's style directive into the expansion request", async () => {
    const wrapper = factory({ styleDirective: "Cinematic look — anamorphic" });
    await wrapper.get("[data-test='expand-preview']").trigger("click");
    await flushPromises();

    expect(expandPromptMock).toHaveBeenCalledWith({
      prompt: "a lighthouse",
      model_family: "flux",
      variations: 1,
      style: "Cinematic look — anamorphic",
    });
  });

  it("omits the style entirely when no preset is active", async () => {
    const wrapper = factory({ styleDirective: null });
    await wrapper.get("[data-test='expand-preview']").trigger("click");
    await flushPromises();

    expect(expandPromptMock).toHaveBeenCalledWith({
      prompt: "a lighthouse",
      model_family: "flux",
      variations: 1,
    });
  });
});
