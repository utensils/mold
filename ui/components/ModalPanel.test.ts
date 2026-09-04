import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ModalPanel from "./ModalPanel.vue";

function make(
  props: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(ModalPanel, {
    props: { open: true, label: "Add a machine", ...props },
    slots: { default: "<p>Modal content</p>", ...slots },
  });
}

describe("ModalPanel", () => {
  it("renders nothing when closed", () => {
    const wrapper = make({ open: false });
    expect(wrapper.find("[role=dialog]").exists()).toBe(false);
  });

  it("is a modal dialog named by its label", () => {
    const wrapper = make();
    const dialog = wrapper.find("[role=dialog]");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Add a machine");
    expect(dialog.attributes("tabindex")).toBe("-1");
  });

  it("applies the width prop to the panel, defaulting to 480", () => {
    expect(make().find(".ms-modal__panel").attributes("style")).toContain(
      "width: 480px",
    );
    expect(
      make({ width: 560 }).find(".ms-modal__panel").attributes("style"),
    ).toContain("width: 560px");
  });

  it("hides the step row when steps is omitted", () => {
    const wrapper = make();
    expect(wrapper.find(".ms-modal__steps").exists()).toBe(false);
  });

  it("renders one bar per step and tints up to the current step", () => {
    const wrapper = make({ steps: 3, step: 2 });
    const dots = wrapper.findAll(".ms-modal__dot");
    expect(dots).toHaveLength(3);
    expect(dots[0]!.attributes("data-on")).toBe("true");
    expect(dots[1]!.attributes("data-on")).toBe("true");
    expect(dots[2]!.attributes("data-on")).toBeUndefined();
  });

  it("tints only the first bar by default (step 1)", () => {
    const dots = make({ steps: 3 }).findAll(".ms-modal__dot");
    expect(dots.map((d) => d.attributes("data-on"))).toEqual([
      "true",
      undefined,
      undefined,
    ]);
  });

  it("renders the default slot body", () => {
    expect(make().find(".ms-modal__body").text()).toContain("Modal content");
  });

  it("omits the footer wrapper unless the footer slot is used", () => {
    expect(make().find(".ms-modal__footer").exists()).toBe(false);
    const wrapper = make({}, { footer: "<button>Continue</button>" });
    expect(wrapper.find(".ms-modal__footer").text()).toContain("Continue");
  });

  it("emits close on backdrop click but not on body click", async () => {
    const wrapper = make();
    await wrapper.find(".ms-modal__panel").trigger("click");
    await wrapper.find(".ms-modal__body").trigger("click");
    expect(wrapper.emitted("close")).toBeUndefined();
    await wrapper.find("[role=dialog]").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close on Escape pressed inside the panel", async () => {
    const wrapper = mount(ModalPanel, {
      props: { open: true, label: "Add a machine" },
      slots: { default: "<input data-test='first' />" },
      attachTo: document.body,
    });
    await wrapper
      .get("[data-test='first']")
      .trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });

  /**
   * The panel is only `aria-modal`, so focus can be anywhere in the app when
   * Escape arrives. A listener on the root closed the dialog only while it
   * happened to hold focus, which is a trap rather than a dialog.
   */
  it("closes on Escape from anywhere in the document while it is open", async () => {
    const wrapper = mount(ModalPanel, {
      props: { open: true, label: "Add a machine" },
      slots: { default: "<p>Modal content</p>" },
      attachTo: document.body,
    });
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(wrapper.emitted("close")).toHaveLength(1);

    await wrapper.setProps({ open: false });
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });

  it("keeps Tab inside the panel", async () => {
    const wrapper = mount(ModalPanel, {
      props: { open: true, label: "Add a machine" },
      slots: {
        default: "<input data-test='first' />",
        footer: "<button data-test='last'>Continue</button>",
      },
      attachTo: document.body,
    });
    const first = wrapper.get("[data-test='first']").element as HTMLElement;
    const last = wrapper.get("[data-test='last']").element as HTMLElement;

    last.focus();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(document.activeElement).toBe(first);

    first.focus();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
    expect(document.activeElement).toBe(last);
    wrapper.unmount();
  });

  it("pulls focus back in when it has escaped the panel", async () => {
    const outside = document.createElement("button");
    document.body.appendChild(outside);
    const wrapper = mount(ModalPanel, {
      props: { open: true, label: "Add a machine" },
      slots: { default: "<input data-test='first' />" },
      attachTo: document.body,
    });
    outside.focus();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='first']").element,
    );
    wrapper.unmount();
    outside.remove();
  });

  it("focuses the overlay root when opened", async () => {
    const wrapper = mount(ModalPanel, {
      props: { open: false },
      attachTo: document.body,
    });
    await wrapper.setProps({ open: true });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(document.activeElement).toBe(wrapper.find("[role=dialog]").element);
    wrapper.unmount();
  });
});
