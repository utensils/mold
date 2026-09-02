import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/mobile/mobile.css", "utf8");
const mobileHtml = readFileSync("index.mobile.html", "utf8");
const mobileAppComponent = readFileSync("src/mobile/MobileApp.vue", "utf8");
const mobileHostDetailComponent = readFileSync("src/mobile/MobileHostDetail.vue", "utf8");
const mobileGenerationQueueCard = readFileSync("src/mobile/MobileGenerationQueueCard.vue", "utf8");
const preparedComponent = readFileSync("src/mobile/MobilePreparedExpansionBatch.vue", "utf8");
const pullComponent = readFileSync("src/mobile/MobileExpansionPullStatus.vue", "utf8");
const composerComponent = readFileSync("src/mobile/MobileSequenceComposer.vue", "utf8");
const sharedParamsComponent = readFileSync("src/mobile/MobileSharedParams.vue", "utf8");
const seamPillComponent = readFileSync("../ui/components/SeamPill.vue", "utf8");
const liveActivityComponent = readFileSync("../ui/components/LiveActivityList.vue", "utf8");
const swipeActionRowComponent = readFileSync("../studio/components/SwipeActionRow.vue", "utf8");
const galleryViewerComponent = readFileSync("src/mobile/MobileGalleryViewer.vue", "utf8");

describe("mobile theme bootstrap", () => {
  it("paints fresh installs as Safelight Dark before Vue mounts", () => {
    expect(mobileHtml).toMatch(/<html[^>]*data-theme-family="safelight"[^>]*data-theme="dark"/);
    expect(mobileHtml).toContain('var family = "safelight"');
    expect(mobileHtml).toContain('var theme = "dark"');
    expect(mobileHtml).toContain('localStorage.getItem("mold.mobile.settings.v1")');
  });
});

describe("mobile viewport scaling", () => {
  it("disables iPhone page and double-tap zoom without document gesture handlers", () => {
    expect(mobileHtml).toMatch(/maximum-scale=1/);
    expect(mobileHtml).toMatch(/user-scalable=no/);

    const root = css.match(/html,\s*body,\s*#app\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);
    expect(root?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
    expect(content?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
  });
});

describe("mobile gallery viewer", () => {
  it("lets fixed edges size the top-layer dialog without iOS inline offset", () => {
    const viewer = css.match(/\.gallery-viewer\s*\{([^}]*)\}/s);

    expect(viewer?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(viewer?.[1]).toMatch(/inset:\s*0\s*;/);
    expect(viewer?.[1]).toMatch(/width:\s*auto\s*;/);
    expect(viewer?.[1]).toMatch(/height:\s*auto\s*;/);
    expect(viewer?.[1]).toMatch(/margin:\s*0\s*;/);
    expect(viewer?.[1]).not.toMatch(/(?:width|height):\s*100(?:%|dvh)\s*;/);
  });
});

describe("mobile Library thumbnail sizing", () => {
  it("drives every gallery column count from the pinch variable", () => {
    // Match every `.gallery-grid` selector, nested or at top level, so a
    // hard-coded count reintroduced inside an at-rule cannot slip past.
    const gridRules = [...css.matchAll(/\.gallery-grid[^{}]*\{([^}]*)\}/gs)];

    expect(gridRules.length).toBeGreaterThan(0);
    const columnDeclarations = gridRules
      .map((rule) => (rule[1] ?? "").match(/grid-template-columns:[^;]*;/)?.[0])
      .filter((declaration): declaration is string => declaration !== undefined);

    expect(columnDeclarations.length).toBeGreaterThan(0);
    for (const declaration of columnDeclarations) {
      expect(declaration).toMatch(/repeat\(var\(--mobile-gallery-columns,\s*3\),/);
    }
  });

  it("reserves the two-finger pinch while one-finger scrolling still works", () => {
    const base = css.match(/\.mobile-gallery-pinch-surface\s*\{([^}]*)\}/s);
    const tile = css.match(/\.gallery-item\s*\{([^}]*)\}/s);
    const media = css.match(/\.gallery-item img,\s*\.gallery-item video\s*\{([^}]*)\}/s);

    expect(base?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
    expect(tile?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
    expect(media?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
    expect(media?.[1]).toMatch(/-webkit-user-drag:\s*none\s*;/);
    expect(base?.[1]).toMatch(/flex:\s*1 0 auto\s*;/);
    expect(base?.[1]).toMatch(/min-height:\s*42vh\s*;/);
    expect(mobileAppComponent).toMatch(
      /class="gallery-grid gallery-grid-virtual"[\s\S]*?'is-android-native': androidNativeRuntime[\s\S]*?@click="handleAndroidGalleryGridClick"/,
    );
    expect(css).toMatch(
      /\.gallery-grid\.is-android-native\s+\.gallery-item\s*\{[^}]*pointer-events:\s*none\s*;/s,
    );
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

  it("keeps the pull-to-refresh control in normal flow above the Library", () => {
    const pull = css.match(/\.mobile-library-pull\s*\{([^}]*)\}/s);

    expect(pull?.[1]).toMatch(/display:\s*grid\s*;/);
    expect(pull?.[1]).toMatch(/flex:\s*none\s*;/);
    expect(pull?.[1]).toMatch(/overflow:\s*hidden\s*;/);
  });

  it("turns wide mobile surfaces into a full-width responsive workspace", () => {
    const tablet = css.match(/@media \(min-width: 640px\) \{([\s\S]*?)\n\}/);
    expect(tablet?.[1]).toMatch(/grid-template-areas:[\s\S]*"tabs header"/);
    expect(tablet?.[1]).toMatch(
      /\.mobile-shell\.is-settings-open\s*\{[\s\S]*grid-template-areas:[\s\S]*"header"[\s\S]*"content"/,
    );
    expect(tablet?.[1]).toMatch(/\.mobile-content\s*\{[\s\S]*width:\s*100%/);
    expect(tablet?.[1]).toMatch(
      /\.mobile-tabs\s*\{[\s\S]*grid-template-columns:\s*minmax\(0,\s*1fr\)/,
    );
    expect(tablet?.[1]).toMatch(/\.mobile-host-form\s*\{[\s\S]*repeat\(2,/);
    expect(tablet?.[1]).toMatch(
      /\.mobile-catalog-detail-scroll\s*\{[\s\S]*padding-right:\s*env\(safe-area-inset-right\)[\s\S]*padding-left:\s*env\(safe-area-inset-left\)/,
    );
    expect(tablet?.[1]).toContain("env(safe-area-inset-left)");

    const roomyTablet = css.match(/@media \(min-width: 768px\) \{([\s\S]*?)\n\}/);
    expect(roomyTablet?.[1]).toMatch(/\.mobile-settings-section\s*\{[\s\S]*minmax\(220px,/);
    expect(roomyTablet?.[1]).toMatch(/\.mobile-catalog-results\s*\{[\s\S]*repeat\(2,/);
    expect(roomyTablet?.[1]).not.toMatch(/\.mobile-catalog-detail-scroll/);

    const wideDetail = css.match(/@media \(min-width: 900px\) \{([\s\S]*?)\n\}/);
    expect(wideDetail?.[1]).toMatch(
      /\.mobile-catalog-detail-scroll\s*\{[\s\S]*minmax\(320px,[\s\S]*minmax\(420px,/,
    );
  });
});

describe("mobile generation status containment", () => {
  it("renders Create and Machines queue jobs through one shared card", () => {
    expect(mobileAppComponent).toContain("<MobileGenerationQueueCard");
    expect(mobileHostDetailComponent).toContain("<MobileGenerationQueueCard");
    expect(mobileGenerationQueueCard).toContain('data-test="mobile-generation-status"');
  });

  it("bounds local queue rows and wraps detailed backend status", () => {
    const row = css.match(/\.mobile-generation-job\s*\{([^}]*)\}/s);
    const detailed = css.match(/\.mobile-generation-job--detailed-status\s*\{([^}]*)\}/s);

    expect(row?.[1]).toMatch(/width:\s*100%\s*;/);
    expect(row?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(row?.[1]).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)/);
    expect(detailed?.[1]).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)\s*;/);
    expect(css).toMatch(
      /\.mobile-generation-job-action\s*>\s*span\s*\{[^}]*max-width:\s*100%\s*;[^}]*overflow-wrap:\s*anywhere\s*;[^}]*white-space:\s*normal\s*;/s,
    );
  });

  it("bounds shared and swipeable activity surfaces before truncating detail", () => {
    expect(liveActivityComponent).toMatch(
      /\.live-activity-surface\s*\{[^}]*width:\s*100%\s*;[^}]*min-width:\s*0\s*;[^}]*box-sizing:\s*border-box\s*;/s,
    );
    expect(liveActivityComponent).toMatch(
      /\.live-activity-copy\s+span\s*\{[^}]*overflow:\s*hidden\s*;[^}]*text-overflow:\s*ellipsis\s*;[^}]*white-space:\s*nowrap\s*;/s,
    );
    expect(swipeActionRowComponent).toMatch(
      /\.swipe-row__surface\s*\{[^}]*width:\s*100%\s*;[^}]*min-width:\s*0\s*;[^}]*box-sizing:\s*border-box\s*;/s,
    );
  });
});

describe("mobile navigation", () => {
  it("reserves a persistent row for the one-shot Develop action", () => {
    const shell = css.match(/\.mobile-shell\s*\{([^}]*)\}/s);
    const action = css.match(/\.mobile-create-action\s*\{([^}]*)\}/s);
    const actionButton = css.match(/\.mobile-create-action \.primary-button\s*\{([^}]*)\}/s);

    expect(shell?.[1]).toMatch(/grid-template-rows:\s*auto minmax\(0, 1fr\) auto auto\s*;/);
    expect(action?.[1]).toMatch(/grid-template-columns:\s*minmax\(0, 1fr\) auto\s*;/);
    expect(action?.[1]).toContain("env(safe-area-inset-left)");
    expect(action?.[1]).toContain("env(safe-area-inset-right)");
    expect(css).toMatch(
      /\.mobile-create-action \.ms-action-blocker\s*\{[^}]*grid-column:\s*1 \/ -1/s,
    );
    expect(Number(actionButton?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(
      48,
    );
  });

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
    expect(sheet?.[1]).toMatch(/inset:\s*0 0 auto\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
  });

  it("scrolls its own body with the pinned mobile containment invariants", () => {
    const panel = css.match(/\.mobile-seam-sheet-panel\s*\{([^}]*)\}/s);
    const body = css.match(/\.mobile-seam-sheet-body\s*\{([^}]*)\}/s);
    expect(panel?.[1]).toMatch(
      /max-height:\s*min\(86dvh,\s*var\(--mobile-visual-viewport-height,\s*100dvh\)\)\s*;/,
    );
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

describe("mobile form spacing", () => {
  it("preserves form rhythm after custom mobile controls", () => {
    const output = css.match(/\.mobile-output-mode\s*\{([^}]*)\}/s);
    const range = css.match(/\.mobile-range-field\s*\{([^}]*)\}/s);
    const duration = css.match(/\.mobile-duration-field\s*\{([^}]*)\}/s);

    expect(output?.[1]).toMatch(/margin:\s*14px 0\s*;/);
    expect(range?.[1]).toMatch(/margin:\s*14px 0\s*;/);
    expect(sharedParamsComponent).toMatch(
      /<VideoDurationSlider[\s\S]*?class="mobile-duration-field"/,
    );
    expect(duration?.[1]).toMatch(/margin-bottom:\s*14px\s*;/);
  });

  it("keeps the 3-D octree segments on the 44pt touch floor", () => {
    const octree = css.match(/\.mobile-mesh-group \.ms-seg \.ms-seg__btn\s*\{([^}]*)\}/s);

    expect(octree?.[1]).toMatch(/min-height:\s*44px\s*;/);
    expect(sharedParamsComponent).toMatch(
      /<div v-if="octreeSegments\.length" class="mobile-mesh-group">/,
    );
  });
});

describe("mobile safe areas", () => {
  it("pins the shell to the unobscured viewport instead of a keyboard-reduced root", () => {
    const shell = css.match(/\.mobile-shell\s*\{([^}]*)\}/s);
    const header = css.match(/\.mobile-header\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);
    const tabs = css.match(/\.mobile-tabs\s*\{([^}]*)\}/s);

    expect(shell?.[1]).toMatch(/height:\s*100lvh\s*;/);
    expect(shell?.[1]).not.toMatch(/height:\s*100%\s*;/);
    expect(shell?.[1]).not.toMatch(/height:\s*100svh\s*;/);
    expect(shell?.[1]).not.toMatch(/height:\s*100dvh\s*;/);
    expect(shell?.[1]).toMatch(
      /transform:\s*translateY\(var\(--mobile-visual-viewport-page-top,\s*0px\)\)\s*;/,
    );
    expect(shell?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
    for (const rule of [header?.[1], content?.[1], tabs?.[1]]) {
      expect(rule).toContain("env(safe-area-inset-left)");
      expect(rule).toContain("env(safe-area-inset-right)");
    }
  });

  it("keeps frequent resolution and catalog controls at least 44px tall", () => {
    for (const selector of [
      ".mobile-resolution-group .ms-shape__btn",
      ".mobile-resolution-tier .ms-seg .ms-seg__btn",
      ".mobile-output-mode .ms-seg .ms-seg__btn",
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

  it("gives shared desktop shape choices a uniform mobile touch tile", () => {
    const group = css.match(/\.mobile-resolution-group \.ms-shape\s*\{([^}]*)\}/s);
    const choice = css.match(/\.mobile-resolution-group \.ms-shape__btn\s*\{([^}]*)\}/s);

    expect(group?.[1]).toMatch(/gap:\s*7px\s*;/);
    expect(choice?.[1]).toMatch(/min-width:\s*60px\s*;/);
    expect(choice?.[1]).toMatch(/min-height:\s*72px\s*;/);
    expect(choice?.[1]).toMatch(/flex:\s*1 1 60px\s*;/);
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

  it("allocates separate disclosure columns for title, filename, and toggle", () => {
    const summary = css.match(/\.mobile-native-disclosure > summary\s*\{([^}]*)\}/s);
    const detail = css.match(/\.mobile-native-disclosure > summary small\s*\{([^}]*)\}/s);
    expect(summary?.[1]).toMatch(/grid-template-columns:\s*auto minmax\(0,\s*1fr\) auto\s*;/);
    expect(detail?.[1]).toMatch(/text-align:\s*right\s*;/);
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

describe("mobile gallery viewer", () => {
  it("contains wide intrinsic media inside the iPhone viewport", () => {
    const viewer = css.match(/\.gallery-viewer\s*\{([^}]*)\}/s);
    const stage = css.match(/\.gallery-viewer-stage\s*\{([^}]*)\}/s);
    const media = css.match(
      /\.gallery-viewer-media,\s*\.gallery-viewer-placeholder\s*\{([^}]*)\}/s,
    );
    expect(viewer?.[1]).toMatch(/overflow:\s*hidden\s*;/);
    expect(stage?.[1]).toMatch(/width:\s*100%\s*;/);
    expect(stage?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(media?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(media?.[1]).toMatch(/max-width:\s*100%\s*;/);
  });

  /**
   * The media is the page: the stage is the whole dialog, the header floats
   * over its top edge, and the details sheet is parked below the bottom one
   * with only its peek showing. Both insets are named once, so the stage can
   * never be sized out from under the picture again.
   */
  it("gives the whole viewport to the media and parks the sheet at its peek", () => {
    const viewer = css.match(/\.gallery-viewer\s*\{([^}]*)\}/s);
    const stage = css.match(/\.gallery-viewer-stage\s*\{([^}]*)\}/s);
    const header = css.match(/\.gallery-viewer-header\s*\{([^}]*)\}/s);
    const sheet = css.match(/(?:^|\n)\.gallery-viewer-sheet\s*\{([^}]*)\}/s);
    const expanded = css.match(/\.gallery-viewer-sheet\.is-expanded\s*\{([^}]*)\}/s);
    const handle = css.match(/\.gallery-viewer-sheet-handle\s*\{([^}]*)\}/s);

    expect(viewer?.[1]).toMatch(
      /--viewer-header-inset:\s*calc\(52px \+ env\(safe-area-inset-top\)\)/,
    );
    expect(viewer?.[1]).toMatch(/--viewer-peek:\s*calc\(68px \+ env\(safe-area-inset-bottom\)\)/);
    expect(stage?.[1]).toMatch(/position:\s*absolute\s*;/);
    expect(stage?.[1]).toMatch(/inset:\s*0\s*;/);
    expect(stage?.[1]).toMatch(
      /padding:\s*var\(--viewer-header-inset\) 0 var\(--viewer-peek\)\s*;/,
    );
    expect(header?.[1]).toMatch(/position:\s*absolute\s*;/);
    // The collapsed sheet shows exactly the peek the stage reserved for it.
    expect(sheet?.[1]).toMatch(
      /transform:\s*translateY\(calc\(100% - var\(--viewer-peek\) \+ var\(--viewer-sheet-drag, 0px\)\)\)\s*;/,
    );
    expect(expanded?.[1]).toMatch(/transform:\s*translateY\(var\(--viewer-sheet-drag, 0px\)\)\s*;/);
    // The handle's own height is the peek the stage reserved, minus the inset.
    expect(handle?.[1]).toMatch(/min-height:\s*68px\s*;/);
  });

  /**
   * The scrim dims the media and swallows its taps, but Close is chrome, not
   * media: it has to stay reachable with the sheet open. Paint order is the
   * whole answer — header over sheet over scrim over stage.
   */
  it("keeps Close reachable above the sheet and its scrim", () => {
    const layer = (source: string, selector: string): number => {
      const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const rule = source.match(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`, "s"));
      return Number(rule?.[1]?.match(/z-index:\s*(\d+)\s*;/)?.[1]);
    };

    expect(layer(css, ".gallery-viewer-header")).toBeGreaterThan(
      layer(css, ".gallery-viewer-sheet"),
    );
    expect(layer(css, ".gallery-viewer-sheet")).toBeGreaterThan(
      layer(css, ".gallery-viewer-sheet-scrim"),
    );
    // The stage's own arrows stay under the scrim: they page the gallery the
    // sheet is describing, so they are media, not chrome.
    expect(layer(css, ".gallery-viewer-sheet-scrim")).toBeGreaterThan(
      layer(galleryViewerComponent, ".gallery-viewer-nav"),
    );
  });

  /** A reveal drag has to reveal something, so the body shows while dragging. */
  it("shows the sheet body while the sheet is being dragged open", () => {
    expect(css).toMatch(
      /\.gallery-viewer-sheet\.is-dragging\s+\.gallery-viewer-details\s*\{\s*visibility:\s*visible;/,
    );
  });

  /** The stage is the viewport now: a transport with no height floats at its top. */
  it("gives the audio transport the whole stage, like every other medium", () => {
    const audio = galleryViewerComponent.match(/\.gallery-viewer-audio\s*\{([^}]*)\}/s);

    expect(audio?.[1]).toMatch(/height:\s*100%\s*;/);
    expect(audio?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
  });

  it("drops the sheet's spring, not the sheet, under reduced motion", () => {
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{\s*\.gallery-viewer-sheet\s*\{\s*transition:\s*none;/,
    );
  });

  it("keeps the header and actions within the same viewport column", () => {
    const header = css.match(/\.gallery-viewer-header\s*\{([^}]*)\}/s);
    const origin = css.match(/\.gallery-viewer-origin\s*\{([^}]*)\}/s);
    const details = css.match(/(?:^|\n)\.gallery-viewer-details\s*\{([^}]*)\}/s);
    const prompt = css.match(/\.gallery-viewer-prompt\s*\{([^}]*)\}/s);
    const promptText = css.match(/\.gallery-viewer-prompt p\s*\{([^}]*)\}/s);
    const actions = css.match(/\.gallery-viewer-actions\s*\{([^}]*)\}/s);
    expect(header?.[1]).toMatch(/width:\s*100%\s*;/);
    expect(header?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(header?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
    expect(origin?.[1]).toMatch(/flex:\s*1 1 0\s*;/);
    expect(origin?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(details?.[1]).toMatch(/width:\s*100%\s*;/);
    expect(details?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(details?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
    expect(details?.[1]).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)\s*;/);
    expect(prompt?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(promptText?.[1]).toMatch(/overflow-wrap:\s*anywhere\s*;/);
    expect(actions?.[1]).toMatch(/min-width:\s*0\s*;/);
  });
});

describe("mobile Library organization", () => {
  it("keeps the Library heading and Select action reachable while the grid scrolls", () => {
    const heading = css.match(/\.mobile-library-heading\s*\{([^}]*)\}/s);

    expect(heading?.[1]).toMatch(/position:\s*sticky\s*;/);
    expect(heading?.[1]).toMatch(/top:\s*-16px\s*;/);
    expect(heading?.[1]).toMatch(/z-index:\s*9\s*;/);
    expect(heading?.[1]).toMatch(/background:/);
    expect(heading?.[1]).toMatch(/backdrop-filter:\s*blur\(/);
  });

  it("keeps the scope row, chips, and tag targets at the 44pt floor", () => {
    for (const selector of [
      ".mobile-library-scope button",
      ".mobile-library-chip",
      ".mobile-library-tag",
      ".mobile-library-banner-link",
      ".mobile-collection-menu-button",
    ]) {
      const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const rule = css.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "s"));
      expect(
        Number(rule?.[1]?.match(/min-(?:height|width):\s*(\d+)px/)?.[1]),
        selector,
      ).toBeGreaterThanOrEqual(44);
    }
  });

  it("never lets the scope row or chip rail capture the grid pinch", () => {
    // The two-finger pinch is reserved for .gallery-grid (touch-action:
    // pan-y); its siblings stay at manipulation so a stray touch on them
    // cannot begin a resize or zoom.
    const scope = css.match(/\.mobile-library-scope\s*\{([^}]*)\}/s);
    const chips = css.match(/\.mobile-library-chips\s*\{([^}]*)\}/s);
    expect(scope?.[1]).toMatch(/touch-action:\s*manipulation\s*;/);
    expect(chips?.[1]).toMatch(/touch-action:\s*pan-x pan-y\s*;/);
    expect(chips?.[1]).toMatch(/overflow-x:\s*auto\s*;/);
  });

  it("keeps tag controls painted above the virtualized print layer", () => {
    const chips = css.match(/\.mobile-library-chips\s*\{([^}]*)\}/s);
    const window = css.match(/\.gallery-grid-window\s*\{([^}]*)\}/s);

    expect(chips?.[1]).toMatch(/flex:\s*none\s*;/);
    expect(chips?.[1]).toMatch(/position:\s*relative\s*;/);
    expect(chips?.[1]).toMatch(/z-index:\s*2\s*;/);
    expect(window?.[1]).toMatch(/z-index:\s*0\s*;/);
  });

  it("keeps Select-mode scrolling native and selection labels inside their buttons", () => {
    const selectingTile = css.match(/\.gallery-grid\.is-selecting \.gallery-item\s*\{([^}]*)\}/s);
    const actions = css.match(/\.mobile-gallery-actions\s*\{([^}]*)\}/s);
    const actionButton = css.match(/\.mobile-gallery-actions button\s*\{([^}]*)\}/s);

    expect(selectingTile?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
    expect(actions?.[1]).toMatch(/flex-shrink:\s*0\s*;/);
    expect(actions?.[1]).toMatch(/grid-template-columns:\s*repeat\(6,\s*minmax\(0,\s*1fr\)\)/);
    expect(actionButton?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
    expect(actionButton?.[1]).toMatch(/overflow-wrap:\s*anywhere\s*;/);
    expect(actionButton?.[1]).toMatch(/white-space:\s*normal\s*;/);
  });

  it("gives the Library sheet the pinned fixed-overlay and body invariants", () => {
    const sheet = css.match(/\.mobile-library-sheet\s*\{([^}]*)\}/s);
    const open = css.match(/\.mobile-library-sheet\.is-open\s*\{([^}]*)\}/s);
    const body = css.match(/\.mobile-library-sheet-body\s*\{([^}]*)\}/s);
    const done = css.match(/\.mobile-library-sheet-done\s*\{([^}]*)\}/s);
    expect(sheet?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(sheet?.[1]).toMatch(/display:\s*none\s*;/);
    expect(sheet?.[1]).toMatch(/inset:\s*0 0 auto\s*;/);
    expect(sheet?.[1]).toMatch(/height:\s*var\(--mobile-visual-viewport-height,\s*100dvh\)\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
    expect(body?.[1]).toMatch(/overflow-y:\s*auto\s*;/);
    expect(body?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(body?.[1]).toMatch(/touch-action:\s*pan-y\s*;/);
    expect(body?.[1]).toContain("env(safe-area-inset-left)");
    expect(body?.[1]).toContain("env(safe-area-inset-right)");
    expect(body?.[1]).toContain("env(safe-area-inset-bottom)");
    expect(Number(done?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
  });

  it("keeps every keyboard-capable sheet inside the visual viewport", () => {
    for (const className of ["mobile-advanced-sheet", "mobile-seam-sheet"]) {
      const sheet = css.match(new RegExp(`\\.${className}\\s*\\{([^}]*)\\}`, "s"));
      expect(sheet?.[1]).toMatch(/inset:\s*0 0 auto\s*;/);
      expect(sheet?.[1]).toMatch(/height:\s*var\(--mobile-visual-viewport-height,\s*100dvh\)\s*;/);
    }
  });

  it("aligns Library sheet actions with their controls instead of the labeled field margin", () => {
    const panel = css.match(/\.mobile-library-sheet-panel\s*\{([^}]*)\}/s);
    const field = css.match(/\.mobile-library-sheet-form \.field\s*\{([^}]*)\}/s);
    expect(panel?.[1]).toMatch(
      /max-height:\s*min\(86dvh,\s*var\(--mobile-visual-viewport-height,\s*100dvh\)\)\s*;/,
    );
    expect(field?.[1]).toMatch(/margin:\s*0\s*;/);
  });

  it("uses the iPhone radii scale for chips, covers, and collection cards", () => {
    const chip = css.match(/\.mobile-library-chip\s*\{([^}]*)\}/s);
    const cover = css.match(/\.mobile-collection-cover\s*\{([^}]*)\}/s);
    const row = css.match(/\.mobile-collection-row\s*\{([^}]*)\}/s);
    expect(chip?.[1]).toMatch(/border-radius:\s*var\(--radius-pill\)\s*;/);
    expect(cover?.[1]).toMatch(/border-radius:\s*var\(--radius-media\)\s*;/);
    expect(row?.[1]).toMatch(/border-radius:\s*var\(--radius-card\)\s*;/);
  });
});
