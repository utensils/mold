import { describe, expect, it } from "vitest";
import inspectorSource from "./InspectorPanel.vue?raw";
import segmentedControlSource from "@ui/components/SegmentedControl.vue?raw";

/** The declarations inside the FIRST `selector { … }` rule. */
function rule(source: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return source.match(new RegExp(`\\n${escaped}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
}

describe("inspector local styles", () => {
  /*
   * Vue applies the parent's scope id to a child component's ROOT node, so a
   * class the inspector defines locally also matches any child component
   * whose root carries the same name — at equal specificity, resolved by
   * whichever chunk the bundler emits last. The inspector's Repeat-this-look
   * control was `.ms-seg`, which is exactly the shared SegmentedControl's
   * root class, and the mesh Surface-detail control mounted right beside it.
   */
  it("gives the seed-mode control a name the shared SegmentedControl cannot claim", () => {
    expect(segmentedControlSource).toContain('class="ms-seg"');
    expect(inspectorSource).toContain('<div class="ms-seedmode"');
    expect(inspectorSource).toContain('class="ms-seedmode__btn"');
    expect(inspectorSource).not.toMatch(/class="ms-seg[ "]/);
    expect(rule(inspectorSource, ".ms-seg")).toBe("");
    expect(rule(inspectorSource, ".ms-seedmode")).not.toBe("");
  });

  /*
   * `--mold-radius-3` is the WINDOW and dialog radius (16px in Safelight), so
   * a 34px segment and a 40px door wearing it render as pills — which
   * ui/mold-desktop.css bans by name.
   */
  it("keeps the window radius off in-view controls", () => {
    expect(rule(inspectorSource, ".ms-seedmode")).toContain("var(--mold-radius-2)");
    expect(rule(inspectorSource, ".ms-seedmode__btn")).toContain("var(--mold-radius-1)");
    expect(rule(inspectorSource, ".ms-advanced")).toContain("var(--mold-radius-2)");
    expect(inspectorSource).not.toContain("var(--mold-radius-3)");
  });

  /*
   * The accent is ONE thing (ui/mold-desktop.css §11); status colours are the
   * state tokens. Painted in the accent, the resolution warning was
   * indistinguishable from the "Match source" link directly above it.
   */
  it("paints a warning in the warning colour, not the accent", () => {
    expect(rule(inspectorSource, ".ms-field__hint--warning")).toContain("var(--mold-warning)");
    expect(rule(inspectorSource, ".ms-field__hint--warning")).not.toContain("var(--mold-blue)");
  });

  /*
   * `.ms-card__faces` and `.ms-seed__input` sit on the same element. Equal
   * specificity means the later rule wins, and `.ms-seed__input` is declared
   * later — so the face-budget field's own width and height were dead and it
   * rendered as a full-width 32px input.
   */
  it("lets the mesh face-budget field win over the seed input it shares a class with", () => {
    expect(inspectorSource).toContain("ms-seed__input ms-card__faces");
    expect(rule(inspectorSource, ".ms-seed__input.ms-card__faces")).toContain(
      "width: calc(17ch + 16px + 18px)",
    );
    expect(rule(inspectorSource, ".ms-seed__input.ms-card__faces")).toContain(
      "height: var(--mold-ctl-md)",
    );
    expect(rule(inspectorSource, ".ms-card__faces")).toBe("");
  });

  /*
   * The tab strip sits beside the view toolbar with only the panel divider
   * between them, so its bottom rule has to land on the toolbar's. Padded
   * 8px around a 6px-padded tab it came out 46px against the toolbar's 40,
   * and the two rules met the divider at different heights.
   */
  it("makes the tab strip exactly one view toolbar tall, tabs on the control ladder", () => {
    const strip = rule(inspectorSource, ".ms-inspector__tabs");
    expect(strip).toContain("height: var(--mold-shell-viewbar-h)");
    expect(strip).toContain("flex: 0 0 var(--mold-shell-viewbar-h)");
    expect(strip).toContain("align-items: center");
    expect(strip).not.toMatch(/padding: \d+px \d+px/);
    const tab = rule(inspectorSource, ".ms-inspector__tab");
    expect(tab).toContain("height: var(--mold-ctl-md)");
    expect(tab).toContain("padding: 0;");
  });
});
