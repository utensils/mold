import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
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

  describe("overlay contract", () => {
    function dialog(): HTMLElement {
      const el = document.body.querySelector<HTMLElement>("[role=dialog]");
      expect(el).not.toBeNull();
      return el as HTMLElement;
    }

    it("is a labelled modal dialog", () => {
      factory();
      const el = dialog();
      expect(el.getAttribute("aria-modal")).toBe("true");
      expect(el.getAttribute("aria-label")).toBe("Prompt expansion");
    });

    it("closes on Escape and on backdrop click, but not inside the panel", async () => {
      const wrapper = factory();

      const panel = document.body.querySelector(
        ".ms-modal__panel",
      ) as HTMLElement;
      panel.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await nextTick();
      expect(wrapper.emitted("close")).toBeUndefined();

      dialog().dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await nextTick();
      expect(wrapper.emitted("close")).toHaveLength(1);

      dialog().dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
      await nextTick();
      expect(wrapper.emitted("close")).toHaveLength(2);
    });

    it("restores focus to the opener when it closes", async () => {
      const opener = document.createElement("button");
      document.body.appendChild(opener);
      opener.focus();

      const wrapper = factory({ open: false });
      await wrapper.setProps({ open: true });
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(dialog().contains(document.activeElement)).toBe(true);

      await wrapper.setProps({ open: false });
      expect(document.activeElement).toBe(opener);
    });

    it("renders no emoji and no hard-coded hex colors", async () => {
      const wrapper = factory();
      await nextTick();
      expect(wrapper.html()).not.toMatch(
        /[\u{1F000}-\u{1FAFF}\u{2190}-\u{21FF}\u{2300}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}]/u,
      );
      expect(wrapper.html()).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
    });
  });
});
