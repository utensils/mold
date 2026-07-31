import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/mobile/mobile.css", "utf8");
const mobileHtml = readFileSync("index.mobile.html", "utf8");
const preparedComponent = readFileSync("src/mobile/MobilePreparedExpansionBatch.vue", "utf8");
const pullComponent = readFileSync("src/mobile/MobileExpansionPullStatus.vue", "utf8");
const composerComponent = readFileSync("src/mobile/MobileSequenceComposer.vue", "utf8");
const seamPillComponent = readFileSync("../ui/components/SeamPill.vue", "utf8");

describe("mobile viewport scaling", () => {
  it("disables iPhone page and double-tap zoom without document gesture handlers", () => {
    expect(mobileHtml).toMatch(/maximum-scale=1/);
    expect(mobileHtml).toMatch(/user-scalable=no/);

    const root = css.match(/html,\s*body,\s*#app\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);
    expect(root?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
    expect(content?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
  });
});

describe("mobile editable controls", () => {
  it("keeps every editable control at the iOS no-focus-zoom size", () => {
    const editables = css.match(
      /input,\s*textarea,\s*select,\s*\[contenteditable="true"\]\s*\{([^}]*)\}/s,
    );

    expect(editables?.[1]).toMatch(/font-size:\s*16px\s*;/);
  });

  it("keeps prepared editors at 16px and their actions at least 44pt", () => {
    expect(preparedComponent).toMatch(/\.mobile-prepared-editor\s*\{[^}]*font-size:\s*16px/s);
    expect(preparedComponent).toMatch(/\.mobile-touch-action\s*\{[^}]*min-height:\s*44px/s);
    expect(pullComponent).toMatch(/\.mobile-touch-action\s*\{[^}]*min-height:\s*44px/s);
  });

  it("removes expansion progress motion when reduced motion is requested", () => {
    expect(pullComponent).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*transition:\s*none/,
    );
  });
});

describe("mobile scrolling", () => {
  it("locks the WebView root and contains the one vertical content scroller", () => {
    const root = css.match(/html,\s*body,\s*#app\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);

    expect(root?.[1]).toMatch(/overflow:\s*hidden\s*;/);
    expect(root?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(content?.[1]).toMatch(/min-height:\s*0\s*;/);
    expect(content?.[1]).toMatch(/overflow-x:\s*hidden\s*;/);
    expect(content?.[1]).toMatch(/overflow-y:\s*auto\s*;/);
    expect(content?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(content?.[1]).not.toMatch(/-webkit-overflow-scrolling/);
  });
});

describe("mobile navigation", () => {
  it("visually marks the tab exposed as the current page", () => {
    expect(css).toMatch(/\.mobile-tab\[aria-current="page"\]\s*\{/);
    expect(css).not.toMatch(/\.mobile-tab\[aria-selected="true"\]\s*\{/);
  });

  it("gives every tab a Mold Studio icon column and a 10px monospace caption", () => {
    const tab = css.match(/\.mobile-tab\s*\{([^}]*)\}/s);
    const icon = css.match(/\.mobile-tab svg\s*\{([^}]*)\}/s);
    expect(tab?.[1]).toMatch(/flex-direction:\s*column\s*;/);
    expect(tab?.[1]).toMatch(/font-size:\s*10px\s*;/);
    expect(tab?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(icon?.[1]).toMatch(/width:\s*22px\s*;/);
  });
});

describe("mobile advanced sheet", () => {
  it("is a full-screen overlay that only becomes visible when opened", () => {
    const sheet = css.match(/\.mobile-advanced-sheet\s*\{([^}]*)\}/s);
    const open = css.match(/\.mobile-advanced-sheet\.is-open\s*\{([^}]*)\}/s);
    expect(sheet?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(sheet?.[1]).toMatch(/display:\s*none\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
  });

  it("scrolls its own body with the pinned mobile containment invariants", () => {
    const body = css.match(/\.mobile-advanced-sheet-body\s*\{([^}]*)\}/s);
    expect(body?.[1]).toMatch(/overflow-y:\s*auto\s*;/);
    expect(body?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(body?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
    expect(body?.[1]).toContain("env(safe-area-inset-left)");
    expect(body?.[1]).toContain("env(safe-area-inset-right)");
    expect(body?.[1]).toContain("env(safe-area-inset-bottom)");
  });

  it("keeps the advanced trigger, close, and reset controls at least 44px", () => {
    const trigger = css.match(/\.mobile-advanced-trigger\s*\{([^}]*)\}/s);
    const close = css.match(/\.mobile-advanced-sheet-close\s*\{([^}]*)\}/s);
    const reset = css.match(/\.mobile-advanced-sheet-reset\s*\{([^}]*)\}/s);
    expect(Number(trigger?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(Number(close?.[1]?.match(/min-width:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(Number(close?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(Number(reset?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
  });

  it("uses the mobile utility font token for the guidance count badge", () => {
    const count = css.match(/\.mobile-generate-inline-count\s*\{([^}]*)\}/s);
    expect(count?.[1]).toMatch(/font-family:\s*var\(--font-utility\)\s*;/);
  });

  it("keeps four primary tabs and gives Settings a full-size header control", () => {
    const tabs = css.match(/\.mobile-tabs\s*\{([^}]*)\}/s);
    const settingsControls = css.match(
      /\.mobile-settings-button,\s*\.mobile-settings-back\s*\{([^}]*)\}/s,
    );

    expect(tabs?.[1]).toMatch(/grid-template-columns:\s*repeat\(4,\s*1fr\)\s*;/);
    expect(settingsControls?.[1]).toMatch(/min-width:\s*44px\s*;/);
    expect(settingsControls?.[1]).toMatch(/min-height:\s*44px\s*;/);
  });
});

describe("mobile seam sheet", () => {
  it("is a fixed overlay that only becomes visible when opened", () => {
    // @ui/SheetPanel is `position: absolute; inset: 0` and its `full` variant
    // has no #header slot, so the seam editor gets this bespoke fixed sheet
    // (the MobileAdvancedSheet pattern) instead.
    const sheet = css.match(/\.mobile-seam-sheet\s*\{([^}]*)\}/s);
    const open = css.match(/\.mobile-seam-sheet\.is-open\s*\{([^}]*)\}/s);
    expect(sheet?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(sheet?.[1]).toMatch(/display:\s*none\s*;/);
    expect(sheet?.[1]).toMatch(/inset:\s*0\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
  });

  it("scrolls its own body with the pinned mobile containment invariants", () => {
    const body = css.match(/\.mobile-seam-sheet-body\s*\{([^}]*)\}/s);
    expect(body?.[1]).toMatch(/overflow-y:\s*auto\s*;/);
    expect(body?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(body?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
    expect(body?.[1]).toContain("env(safe-area-inset-left)");
    expect(body?.[1]).toContain("env(safe-area-inset-right)");
    expect(body?.[1]).toContain("env(safe-area-inset-bottom)");
  });

  it("keeps the sheet's Done control and the backdrop at touch size", () => {
    const done = css.match(/\.mobile-seam-sheet-done\s*\{([^}]*)\}/s);
    const backdrop = css.match(/\.mobile-seam-sheet-backdrop\s*\{([^}]*)\}/s);
    expect(Number(done?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(46);
    expect(backdrop?.[1]).toMatch(/position:\s*absolute\s*;/);
    expect(backdrop?.[1]).toMatch(/inset:\s*0\s*;/);
  });

  it("keeps the seam pill that opens it at the iPhone 44pt floor", () => {
    // The pill's touch size comes from the shared kit's `large` variant —
    // assert it there so a kit restyle can't shrink the iPhone target.
    const large = seamPillComponent.match(/\.ms-seam--large\s*\{([^}]*)\}/s);
    expect(Number(large?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
  });
});

describe("mobile sequence composer", () => {
  it("keeps clip prompts at the iOS no-focus-zoom size", () => {
    expect(composerComponent).toMatch(/\.mobile-sequence-prompt\s*\{[^}]*font-size:\s*16px/s);
  });

  it("keeps Add clip and Generate sequence at the sheet-button height", () => {
    for (const selector of [".mobile-sequence-add", ".mobile-sequence-generate"]) {
      const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const rule = composerComponent.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "s"));
      expect(
        Number(rule?.[1]?.match(/min-height:\s*(\d+)px/)?.[1]),
        selector,
      ).toBeGreaterThanOrEqual(46);
    }
  });
});

describe("mobile style row", () => {
  it("renders the collapsed head value as a compact pill, not a 44pt tap chip", () => {
    const styleComponent = readFileSync("src/mobile/MobileStyleChips.vue", "utf8");
    // The head button is itself the 44pt tap target; its value indicator must
    // use the compact class — reusing .mobile-style-chip blockifies the span
    // to 44pt inside the flex head and balloons it into an egg beside STYLE.
    expect(styleComponent).toMatch(/data-test="mobile-style-active"/);
    expect(styleComponent).toMatch(/class="mobile-style-value"/);
    const value = css.match(/\.mobile-style-value\s*\{([^}]*)\}/s);
    expect(value?.[1]).not.toMatch(/min-height/);
    expect(value?.[1]).toMatch(/border-radius:\s*var\(--radius-pill\)\s*;/);
    // The whole-row head keeps the 44pt target; expanded presets stay 44pt.
    const head = css.match(/\.mobile-style-head\s*\{([^}]*)\}/s);
    expect(head?.[1]).toMatch(/min-height:\s*44px\s*;/);
    const chip = css.match(/\.mobile-style-chip\s*\{([^}]*)\}/s);
    expect(chip?.[1]).toMatch(/min-height:\s*44px\s*;/);
  });
});

describe("mobile safe areas", () => {
  it("keeps the shell inside the dynamic viewport and clears both landscape notches", () => {
    const shell = css.match(/\.mobile-shell\s*\{([^}]*)\}/s);
    const header = css.match(/\.mobile-header\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);
    const tabs = css.match(/\.mobile-tabs\s*\{([^}]*)\}/s);

    expect(shell?.[1]).toMatch(/height:\s*100dvh\s*;/);
    expect(shell?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
    for (const rule of [header?.[1], content?.[1], tabs?.[1]]) {
      expect(rule).toContain("env(safe-area-inset-left)");
      expect(rule).toContain("env(safe-area-inset-right)");
    }
  });

  it("keeps frequent resolution and catalog controls at least 44px tall", () => {
    for (const selector of [
      ".mobile-resolution-segment",
      ".mobile-resolution-tier .ms-seg .ms-seg__btn",
      ".mobile-output-mode .ms-seg .ms-seg__btn",
      ".mobile-resolution-aspect",
      ".mobile-catalog-segment button",
      ".mobile-catalog-media button",
      ".mobile-catalog-sources button",
      ".mobile-section-head > button",
      ".mobile-disclosure-button",
      ".mobile-generate-stepper-button",
      ".mobile-media-tile-action",
      ".mobile-token-list button",
      ".mobile-template-actions button",
      "button.mobile-generate-disclosure",
    ]) {
      const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const rules = [...css.matchAll(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "gs"))];
      expect(
        rules.some((rule) => {
          const height = rule[1]?.match(/min-height:\s*(\d+)px\s*;/)?.[1];
          return height !== undefined && Number(height) >= 44;
        }),
        selector,
      ).toBe(true);
    }
  });

  it("renders catalog filters as balanced equal-width tiles", () => {
    const containers = css.match(
      /\.mobile-catalog-media,\s*\.mobile-catalog-sources\s*\{([^}]*)\}/s,
    );
    const media = [...css.matchAll(/\.mobile-catalog-media\s*\{([^}]*)\}/gs)].find((rule) =>
      rule[1]?.includes("grid-template-columns"),
    );
    const sources = [...css.matchAll(/\.mobile-catalog-sources\s*\{([^}]*)\}/gs)].find((rule) =>
      rule[1]?.includes("grid-template-columns"),
    );
    const buttons = css.match(
      /\.mobile-catalog-media button,\s*\.mobile-catalog-sources button\s*\{([^}]*)\}/s,
    );
    const selected = css.match(
      /\.mobile-catalog-media button\[aria-pressed="true"\],\s*\.mobile-catalog-sources button\[aria-pressed="true"\]\s*\{([^}]*)\}/s,
    );

    expect(media?.[1]).toMatch(/grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
    expect(sources?.[1]).toMatch(/grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
    expect(containers?.[1]).toMatch(/gap:\s*8px/);
    expect(buttons?.[1]).toMatch(/min-width:\s*0/);
    expect(buttons?.[1]).toMatch(/border:\s*1px solid var\(--control-edge\)/);
    expect(buttons?.[1]).toMatch(/background:\s*var\(--bench\)/);
    expect(buttons?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(selected?.[1]).toContain("color-mix(in srgb, var(--safelight) 11%, var(--bench))");
    expect(selected?.[1]).toMatch(/color:\s*var\(--safelight\)/);
  });

  it("keeps catalog filters readable on narrow iPhones", () => {
    expect(css).toMatch(
      /@media\s*\(max-width:\s*430px\)[\s\S]*?\.mobile-catalog-filters\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)\s*;/,
    );
    expect(css).toMatch(
      /@media\s*\(max-width:\s*430px\)[\s\S]*?\.mobile-catalog-nsfw\s*\{[^}]*grid-column:\s*1\s*\/\s*-1\s*;/,
    );
  });

  it("wraps full catalog metadata values instead of hiding them behind ellipses", () => {
    const value = css.match(/\.mobile-catalog-detail-meta dd\s*\{([^}]*)\}/s);
    expect(value?.[1]).toMatch(/overflow-wrap:\s*anywhere\s*;/);
    expect(value?.[1]).toMatch(/white-space:\s*normal\s*;/);
    expect(value?.[1]).not.toMatch(/overflow:\s*hidden\s*;/);
    expect(value?.[1]).not.toMatch(/text-overflow:\s*ellipsis\s*;/);
  });

  it("gives aspect choices a proportional visual frame inside a uniform touch tile", () => {
    const aspects = css.match(/\.mobile-resolution-aspects\s*\{([^}]*)\}/s);
    const choice = css.match(/\.mobile-resolution-aspect\s*\{([^}]*)\}/s);
    const shape = css.match(/\.mobile-resolution-aspect-shape\s*\{([^}]*)\}/s);

    expect(aspects?.[1]).toMatch(/display:\s*grid\s*;/);
    expect(aspects?.[1]).toMatch(/minmax\(64px,\s*84px\)/);
    expect(choice?.[1]).toMatch(/min-height:\s*72px\s*;/);
    expect(shape?.[1]).toMatch(/border:\s*1px solid currentColor\s*;/);
  });

  it("keeps the kit tier segments at touch size with legible sublabels", () => {
    // Mobile-scoped overrides of the shared @ui SegmentedControl: the kit's
    // default 7px-padded segments and 9px sub-line are below the iPhone 44pt /
    // 10px floors. Three-class selectors outrank the kit's scoped two-part
    // rules regardless of stylesheet order.
    const button = css.match(/\.mobile-resolution-tier \.ms-seg \.ms-seg__btn\s*\{([^}]*)\}/s);
    const sub = css.match(/\.mobile-resolution-tier \.ms-seg \.ms-seg__sub\s*\{([^}]*)\}/s);
    const dims = css.match(/\.mobile-resolution-tier-dims\s*\{([^}]*)\}/s);

    expect(button?.[1]).toMatch(/min-height:\s*44px\s*;/);
    expect(sub?.[1]).toMatch(/font-size:\s*10px\s*;/);
    expect(dims?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(dims?.[1]).toMatch(/color:\s*var\(--ink-3\)/);
  });

  it("does not keep the redundant resolution summary card", () => {
    for (const selector of [
      ".mobile-resolution-summary",
      ".mobile-resolution-preview",
      ".mobile-resolution-copy",
      ".mobile-resolution-custom-badge",
    ]) {
      expect(css).not.toContain(selector);
    }
  });

  it("renders Output as two flush segments instead of nested rounded pills", () => {
    const control = css.match(/\.mobile-output-mode \.ms-seg\s*\{([^}]*)\}/s);
    const button = css.match(/\.mobile-output-mode \.ms-seg \.ms-seg__btn\s*\{([^}]*)\}/s);
    const first = css.match(
      /\.mobile-output-mode \.ms-seg \.ms-seg__btn:first-child\s*\{([^}]*)\}/s,
    );
    const last = css.match(/\.mobile-output-mode \.ms-seg \.ms-seg__btn:last-child\s*\{([^}]*)\}/s);
    const seam = css.match(
      /\.mobile-output-mode \.ms-seg \.ms-seg__btn \+ \.ms-seg__btn\s*\{([^}]*)\}/s,
    );

    expect(control?.[1]).toMatch(/gap:\s*0\s*;/);
    expect(control?.[1]).toMatch(/padding:\s*0\s*;/);
    expect(control?.[1]).toMatch(/min-height:\s*44px\s*;/);
    expect(control?.[1]).toMatch(/border:\s*0\s*;/);
    expect(control?.[1]).toMatch(/box-shadow:\s*inset 0 0 0 1px var\(--ce\)\s*;/);
    expect(control?.[1]).not.toMatch(/overflow:\s*hidden\s*;/);
    expect(button?.[1]).toMatch(/border-radius:\s*0\s*;/);
    expect(button?.[1]).toMatch(/justify-content:\s*center\s*;/);
    expect(button?.[1]).toMatch(/padding-block:\s*0\s*;/);
    expect(first?.[1]).toMatch(
      /border-radius:\s*calc\(var\(--radius-control\) - 1px\) 0 0 calc\(var\(--radius-control\) - 1px\)\s*;/,
    );
    expect(last?.[1]).toMatch(
      /border-radius:\s*0 calc\(var\(--radius-control\) - 1px\) calc\(var\(--radius-control\) - 1px\) 0\s*;/,
    );
    expect(seam?.[1]).toMatch(/border-left:\s*1px solid var\(--ce\)\s*;/);
  });
});

describe("mobile develop bed", () => {
  it("caps the bed by viewport height without distorting the print ratio", () => {
    // A portrait bed clamped by a plain `max-height: 55vh` keeps width: 100%,
    // so the aspect-ratio box no longer matches the print and the layered
    // preview/grain distort. The cap must ride the width axis instead: the
    // component supplies the print's ratio as `--bed-ar`, and the width cap
    // keeps the ratio-derived height ≤ 55vh.
    const bed = css.match(/\.mobile-develop-bed\s*\{([^}]*)\}/s);
    expect(bed?.[1]).toMatch(
      /max-width:\s*min\(100%,\s*calc\(55vh \* var\(--bed-ar[^)]*\)\)\)\s*;/,
    );
    expect(bed?.[1]).toMatch(/margin-inline:\s*auto\s*;/);
    expect(bed?.[1]).not.toMatch(/max-height\s*:/);

    const app = readFileSync("src/mobile/MobileApp.vue", "utf8");
    expect(app).toMatch(/--bed-ar/);
  });
});
