import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  resolve(__dirname, "./MeshWorkflowStudio.vue"),
  "utf8",
);
const kit = readFileSync(resolve(__dirname, "../../ui/kit.css"), "utf8");

/** The declarations inside the FIRST `selector { … }` rule. */
function rule(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return css.match(new RegExp(`\\n\\s*${escaped}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
}

describe("the 3-D Studio wears the shell", () => {
  /*
   * README §3: New image and this are the same anatomy. The description, the
   * style chips and Generate belong on the composer, not at the foot of the
   * inspector — which is what made this read as a form dropped into the shell.
   */
  it("puts the description, both style chips and Generate on the composer", () => {
    const composer = source.slice(source.indexOf('data-test="mesh-composer"'));
    expect(composer).toContain("ms-composer__card");
    expect(composer).toContain('aria-label="Describe the object"');
    expect(composer).toContain('name="mesh-picker"');
    expect(composer).toContain('name="image-picker"');
    expect(composer).toContain('data-test="mesh-generate"');
    expect(composer).toContain("ms-composer__key");
  });

  /* The composer's anatomy is the kit's, never a second copy. */
  it("reads the composer's shell from the shared kit", () => {
    expect(rule(kit, ".ms-composer__controls")).toContain("display: flex");
    expect(rule(kit, ".ms-composer__generate")).toContain("var(--mold-blue)");
    expect(source).not.toMatch(/\n\.ms-composer\s*\{/);
    expect(source).not.toMatch(/\n\.ms-composer__generate\s*\{/);
  });

  /*
   * Fixed chrome never shrinks and the canvas absorbs the slack (README §3) —
   * that is the composer's `flex-shrink: 0`. The canvas itself still SCROLLS:
   * clipping it put a long stage list's Cancel and Resume out of reach with no
   * scrollbar to find them.
   */
  it("gives the canvas the height, keeps the composer on its edge, and still scrolls", () => {
    expect(
      rule(source, ".mesh-studio--desktop .mesh-studio__result"),
    ).toContain("overflow: auto");
    const main = rule(source, ".mesh-studio--desktop .mesh-studio__main");
    expect(main).toContain("flex-direction: column");
    // A floor, so an upward style menu is never cut by the view toolbar.
    expect(main).toContain("var(--mesh-canvas-floor");
    expect(rule(source, ".mesh-studio--desktop .mesh-studio__bar")).toContain(
      "flex-shrink: 0",
    );
  });

  /*
   * The view toolbar is the shell's, and the inspector's own header is exactly
   * one toolbar tall so the two rules meet — the metric `InspectorPanel`'s tab
   * strip binds. Both carry a fallback because the phone imports ui/kit.css and
   * ui/tokens.css but never ui/mold-desktop.css.
   */
  it("binds the shell's own toolbar height, with a fallback", () => {
    const header = rule(source, ".mesh-studio--desktop .mesh-studio__header");
    expect(header).toContain("var(--mold-shell-viewbar-h, 40px)");
    expect(header).toContain("var(--mold-chrome)");
    const heading = rule(
      source,
      ".mesh-studio--desktop .mesh-studio__inspector-heading",
    );
    expect(heading).toContain("var(--mold-shell-viewbar-h, 40px)");
    // It lives inside the scrolling settings form, so it has to stick.
    expect(heading).toContain("position: sticky");
  });

  /*
   * A `SwitchToggle`'s `label` IS its accessible name, so the words beside it
   * are the visible half of one control — announcing them again made every
   * toggle read twice.
   */
  it("never announces a switch's own label twice", () => {
    const labels = source.match(/mesh-studio__check-label/g) ?? [];
    expect(labels.length).toBeGreaterThan(0);
    const announced =
      source.match(/mesh-studio__check-label" aria-hidden="true"/g) ?? [];
    expect(announced.length).toBe(labels.length);
  });

  /*
   * Plain words in sans; the format's own vocabulary stays in the mono truth.
   * The words are SENTENCE case in the template — the uppercase is desktop's
   * presentation, applied by CSS. Baking the caps in shouted on web, which
   * renders the same markup without that rule.
   */
  it("names the geometry fields in plain words, and shouts only in CSS", () => {
    expect(source).toContain(">Which way is up</span");
    expect(source).toContain(">How big one unit is</span");
    expect(source).not.toContain("WHICH WAY IS UP");
    expect(source).not.toContain("HOW BIG ONE UNIT IS");
    expect(rule(source, ".mesh-studio--desktop .mesh-studio__label")).toContain(
      "text-transform: uppercase",
    );
    // The format's own vocabulary survives, in the line of mono truth.
    expect(source).toContain("Y-up · as stored (glTF, Blender OBJ)");
  });
});
