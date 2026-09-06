import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import ModalPanel from "./ModalPanel.vue";
import ModalPanelSource from "./ModalPanel.vue?raw";

/* The kit's tests are run from `desktop/` (they have no runner of their own),
 * and `import.meta.url` for a file outside that Vite root is not a file URL —
 * so the sheet is found by walking up from the working directory. */
const desktopTokens = (() => {
  for (const candidate of ["../ui/mold-desktop.css", "ui/mold-desktop.css"]) {
    const path = resolve(process.cwd(), candidate);
    if (existsSync(path)) return readFileSync(path, "utf8");
  }
  throw new Error("mold-desktop.css not found from " + process.cwd());
})();

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

describe("ModalPanel stacking", () => {
  function open(label: string) {
    return mount(ModalPanel, {
      props: { open: true, label },
      slots: { default: `<input data-test="${label}" />` },
      attachTo: document.body,
    });
  }

  it("only the topmost dialog answers Escape, and it stops the key there", () => {
    const under = open("under");
    const over = open("over");
    const heardBelow: string[] = [];
    const witness = () => heardBelow.push("window");
    window.addEventListener("keydown", witness);

    // A key no dialog claims still reaches everything below — the fence
    // below is about Escape, not about the listener never firing.
    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "a",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(heardBelow).toEqual(["window"]);

    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(over.emitted("close")).toHaveLength(1);
    expect(under.emitted("close")).toBeUndefined();
    expect(heardBelow).toEqual(["window"]);

    over.unmount();
    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(under.emitted("close")).toHaveLength(1);

    window.removeEventListener("keydown", witness);
    under.unmount();
  });

  it("only the topmost dialog keeps Tab", () => {
    const under = open("under");
    const over = open("over");
    const underField = under.get("[data-test='under']").element as HTMLElement;
    const overField = over.get("[data-test='over']").element as HTMLElement;

    underField.focus();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    // The lower dialog stands down; the top one pulls focus into itself.
    expect(document.activeElement).toBe(overField);

    over.unmount();
    under.unmount();
  });

  it("a dialog closed by its prop hands Escape back to the one below it", async () => {
    const under = open("under");
    const over = open("over");
    await over.setProps({ open: false });
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(under.emitted("close")).toHaveLength(1);
    over.unmount();
    under.unmount();
  });
});

describe("ModalPanel layering and ground", () => {
  /*
   * The dialog is `position: absolute` inside its frame, so without a
   * z-index it painted UNDER the Create bench resizer and the clip lane's
   * a chip inside a positioned panel showed the bar through its own scrim.
   */
  it("paints above every in-view layer, with a fallback for a host without the token", () => {
    expect(ModalPanelSource).toMatch(
      /\.ms-modal \{[^}]*z-index: var\(--mold-z-modal, 100\)/s,
    );
    expect(desktopTokens).toMatch(/--mold-z-modal:\s*100;/);
  });

  /*
   * The panel's ground is the raised surface role, so cards inside a dialog
   * still read as cards. On a host without the desktop role it falls back to
   * the app background rather than to the same colour as its own contents.
   */
  it("stands the panel on the raised-surface role", () => {
    expect(ModalPanelSource).toMatch(
      /\.ms-modal__panel \{[^}]*background: var\(--mold-panel-raised, var\(--mold-bg\)\)/s,
    );
  });
});
