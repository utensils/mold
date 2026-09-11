import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/mobile/mobile.css", "utf8");
const mobileHtml = readFileSync("index.mobile.html", "utf8");
const mobileAppComponent = readFileSync("src/mobile/MobileApp.vue", "utf8");
const mobileHostDetailComponent = readFileSync("src/mobile/MobileHostDetail.vue", "utf8");
const mobileGenerationQueueCard = readFileSync("src/mobile/MobileGenerationQueueCard.vue", "utf8");
const preparedComponent = readFileSync("src/mobile/MobilePreparedExpansionBatch.vue", "utf8");
const pullComponent = readFileSync("src/mobile/MobileExpansionPullStatus.vue", "utf8");
const sharedParamsComponent = readFileSync("src/mobile/MobileSharedParams.vue", "utf8");
const liveActivityComponent = readFileSync("../ui/components/LiveActivityList.vue", "utf8");
const swipeActionRowComponent = readFileSync("../studio/components/SwipeActionRow.vue", "utf8");
const galleryViewerComponent = readFileSync("src/mobile/MobileGalleryViewer.vue", "utf8");

const tokens = readFileSync("../ui/tokens.css", "utf8");

describe("mobile theme swatches", () => {
  it("copy each theme's chrome, content, telemetry, and accent hexes from ui/tokens.css", () => {
    const themes = [...tokens.matchAll(/\[data-theme="([\w-]+)"\] \{([^}]*)\}/g)];
    expect(themes).toHaveLength(10);
    for (const [, id, body] of themes) {
      const token = (key: string) => body!.match(new RegExp(`--mold-${key}: (#[0-9a-f]{6});`))?.[1];
      const swatch = css.match(
        new RegExp(`\\.mobile-theme-preview\\[data-theme="${id}"\\] \\{([^}]*)\\}`),
      )?.[1];
      expect(swatch, id).toBeDefined();
      const preview = (key: string) =>
        swatch!.match(new RegExp(`--preview-${key}: (#[0-9a-f]{6});`))?.[1];
      expect(preview("bath"), `${id} chrome`).toBe(token("bg-deep"));
      expect(preview("bench"), `${id} content`).toBe(token("bg"));
      expect(preview("halide"), `${id} telemetry`).toBe(token("sapphire"));
      expect(preview("safelight"), `${id} accent`).toBe(token("blue"));
    }
  });
});

describe("mobile theme bootstrap", () => {
  it("paints fresh installs as Safelight before Vue mounts", () => {
    expect(mobileHtml).toMatch(/<html[^>]*data-theme="safelight-dark"/);
    expect(mobileHtml).toContain('var theme = "safelight-dark"');
    expect(mobileHtml).toContain('localStorage.getItem("mold.mobile.settings.v1")');
    // Both older shapes still migrate before paint — the family + appearance
    // pair, and the pre-tone ids — and the tone resolves in the same frame.
    expect(mobileHtml).toContain("parsed.themeFamily");
    expect(mobileHtml).toContain('family === "mold"');
    expect(mobileHtml).toContain('porcelain: "graphite-light"');
    expect(mobileHtml).toContain('matchMedia("(prefers-color-scheme: light)")');
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
  it("keeps sheet text readable against every theme background", () => {
    const sheet = css.match(/\.gallery-viewer-sheet\s*\{([^}]*)\}/s)?.[1];
    expect(sheet).toContain("background: var(--mold-bg-deep)");
    expect(sheet).toContain("color: var(--mold-text)");
    const themes = [...tokens.matchAll(/\[data-theme="([\w-]+)"\] \{([^}]*)\}/g)];
    expect(themes).toHaveLength(10);
    const luminance = (hex: string) => {
      const channels = [1, 3, 5].map((offset) => {
        const value = parseInt(hex.slice(offset, offset + 2), 16) / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
      return channels[0]! * 0.2126 + channels[1]! * 0.7152 + channels[2]! * 0.0722;
    };
    for (const [, theme, body] of themes) {
      const color = (key: string) => {
        const hex = body!.match(new RegExp(`--mold-${key}: (#[0-9a-f]{6});`))?.[1];
        expect(hex, `${theme} ${key}`).toBeDefined();
        return luminance(hex!);
      };
      const background = color("bg-deep");
      for (const key of ["text", "text-2", "text-dim", "blue", "error"]) {
        const foreground = color(key);
        const contrast =
          (Math.max(background, foreground) + 0.05) / (Math.min(background, foreground) + 0.05);
        expect(contrast, `${theme} ${key}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("lets fixed edges size the top-layer dialog without iOS inline offset", () => {
    const viewer = css.match(/\.gallery-viewer\s*\{([^}]*)\}/s);

    expect(viewer?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(viewer?.[1]).toMatch(/inset:\s*0\s*;/);
    expect(viewer?.[1]).toMatch(/width:\s*auto\s*;/);
    expect(viewer?.[1]).toMatch(/height:\s*var\(--mobile-visual-viewport-height, 100dvh\)\s*;/);
    expect(viewer?.[1]).toContain(
      "transform: translateY(var(--mobile-visual-viewport-page-top, 0px))",
    );
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

    expect(editables?.[1]).toMatch(/font-size:\s*max\(16px, 1rem\)\s*!important\s*;/);
  });

  it("keeps prepared editors scalable and their actions at least 44pt", () => {
    expect(preparedComponent).toMatch(/\.mobile-prepared-editor\s*\{[^}]*font-size:\s*1rem/s);
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
    expect(tablet?.[1]).toMatch(/\.mobile-host-form\s*\{[\s\S]*repeat\(auto-fit,/);
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

  it("shows a running print at 64px with a 7px meter and its mono batch line", () => {
    const thumb = css.match(/\.mobile-generation-job-thumb\s*\{([^}]*)\}/s);
    expect(thumb?.[1]).toMatch(/width:\s*64px/);
    expect(thumb?.[1]).toMatch(/height:\s*64px/);

    // The place-in-line square is a finger target where the picture will be.
    const position = css.match(/\.mobile-generation-job-position\s*\{([^}]*)\}/s);
    expect(Number(position?.[1]?.match(/width:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(position?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);

    // Which one of the batch, and where: technical truth, so mono.
    const meta = css.match(/\.mobile-generation-job-meta\s*\{([^}]*)\}/s);
    expect(meta?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(meta?.[1]).toMatch(/font-size:\s*var\(--text-edge-code\)/);

    // The meter is the shared kit bar at the phone's height, not a second one.
    expect(mobileGenerationQueueCard).toContain(':height="7"');
    expect(mobileGenerationQueueCard).toContain("ProgressBar");

    // What a running print is DOING is a sentence, so it takes plain sans;
    // mono uppercase stays the code for a waiting, held or settled row.
    const sentence = css.match(/\.mobile-generation-job-sentence\s*\{([^}]*)\}/s);
    expect(sentence?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(sentence?.[1]).toMatch(/font-size:\s*var\(--text-body\)/);
    expect(sentence?.[1]).toMatch(/color:\s*var\(--mold-text-2\)/);
    expect(sentence?.[1]).not.toMatch(/text-transform:\s*uppercase/);
    // It is a sentence, so it wraps. The mono lines beside it truncate, and
    // inheriting that turned "Denoising (50 steps) · 19/50" into "Denoising…".
    expect(sentence?.[1]).toMatch(/white-space:\s*normal/);
    expect(sentence?.[1]).not.toMatch(/text-overflow:\s*ellipsis/);
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
    expect(action?.[1]).toMatch(
      /grid-template-columns:\s*repeat\(auto-fit, minmax\(min\(100%, 10em\), 1fr\)\)\s*;/,
    );
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

  it("gives every tab a Mold Studio icon column and a readable caption", () => {
    const tab = css.match(/\.mobile-tab\s*\{([^}]*)\}/s);
    const icon = css.match(/\.mobile-tab svg\s*\{([^}]*)\}/s);
    expect(tab?.[1]).toMatch(/flex-direction:\s*column\s*;/);
    expect(tab?.[1]).toMatch(/font-size:\s*var\(--mold-fs-micro\)\s*;/);
    expect(tab?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(icon?.[1]).toMatch(/width:\s*22px\s*;/);
  });
});

describe("mobile advanced sheet", () => {
  it("is a bottom sheet that only becomes visible when opened", () => {
    const sheet = css.match(/\.mobile-advanced-sheet\s*\{([^}]*)\}/s);
    const open = css.match(/\.mobile-advanced-sheet\.is-open\s*\{([^}]*)\}/s);
    expect(sheet?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(sheet?.[1]).toMatch(/display:\s*none\s*;/);
    // It rises from the bottom edge over the composer, rather than replacing
    // the screen: the panel is bounded and the surface behind stays visible.
    expect(sheet?.[1]).toMatch(/justify-content:\s*flex-end\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
  });

  it("gives its header the shared iOS shape and drops the circular Done", () => {
    const head = css.match(/\.mobile-advanced-sheet-head\s*\{([^}]*)\}/s);
    const close = css.match(/\.mobile-advanced-sheet-close\s*\{([^}]*)\}/s);
    expect(head?.[1]).toMatch(/border-bottom:\s*1px solid var\(--mold-border\)/);
    expect(close?.[1]).not.toMatch(/border-radius:\s*50%/);
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

  it("keeps five primary tabs and gives Settings a full-size header control", () => {
    const tabs = css.match(/\.mobile-tabs\s*\{([^}]*)\}/s);
    const settingsControls = css.match(
      /\.mobile-settings-button,\s*\.mobile-settings-back\s*\{([^}]*)\}/s,
    );

    expect(tabs?.[1]).toMatch(/grid-template-columns:\s*repeat\(5,\s*minmax\(0,\s*1fr\)\)\s*;/);
    expect(settingsControls?.[1]).toMatch(/min-width:\s*44px\s*;/);
    expect(settingsControls?.[1]).toMatch(/min-height:\s*44px\s*;/);
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
    expect(value?.[1]).toMatch(/border-radius:\s*var\(--mold-radius-2\)\s*;/);
    // The whole-row head keeps the 44pt target; expanded presets stay 44pt.
    const head = css.match(/\.mobile-style-head\s*\{([^}]*)\}/s);
    expect(head?.[1]).toMatch(/min-height:\s*44px\s*;/);
    const chip = css.match(/\.mobile-style-chip\s*\{([^}]*)\}/s);
    expect(chip?.[1]).toMatch(/min-height:\s*44px\s*;/);
  });
});

describe("mobile form spacing", () => {
  it("preserves form rhythm after custom mobile controls", () => {
    const range = css.match(/\.mobile-range-field\s*\{([^}]*)\}/s);
    const duration = css.match(/\.mobile-duration-field\s*\{([^}]*)\}/s);

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
  it("fits the shell to the visible viewport with a stable full-height fallback", () => {
    const shell = css.match(/\.mobile-shell\s*\{([^}]*)\}/s);
    const header = css.match(/\.mobile-header\s*\{([^}]*)\}/s);
    const content = css.match(/\.mobile-content\s*\{([^}]*)\}/s);
    const tabs = css.match(/\.mobile-tabs\s*\{([^}]*)\}/s);

    expect(shell?.[1]).toMatch(/height:\s*var\(--mobile-visual-viewport-height,\s*100lvh\)\s*;/);
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
    const media = [...css.matchAll(/\.mobile-catalog-media\s*\{([^}]*)\}/gs)].find((rule) =>
      rule[1]?.includes("overflow-x"),
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

    // Four media choices never fit a 393pt row as equal tiles — they wrapped to
    // a second row and pushed the results down. One strip that scrolls instead.
    expect(media?.[1]).toMatch(/display:\s*flex/);
    expect(media?.[1]).toMatch(/flex-wrap:\s*nowrap/);
    expect(media?.[1]).toMatch(/overflow-x:\s*auto/);
    expect(media?.[1]).toMatch(/gap:\s*8px/);
    expect(css).toMatch(/\.mobile-catalog-media button\s*\{[^}]*flex:\s*none/s);
    // The catalog sources are three and stay three equal tiles in the sheet.
    expect(sources?.[1]).toMatch(/grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
    expect(sources?.[1]).toMatch(/gap:\s*8px/);
    expect(buttons?.[1]).toMatch(/min-width:\s*0/);
    expect(buttons?.[1]).toMatch(/border:\s*1px solid var\(--mold-border-control\)/);
    expect(buttons?.[1]).toMatch(/background:\s*var\(--mold-bg\)/);
    expect(buttons?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(selected?.[1]).toContain("color-mix(in srgb, var(--mold-blue) 11%, var(--mold-bg))");
    expect(selected?.[1]).toMatch(/color:\s*var\(--mold-blue\)/);
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

  it("lets shape tiles grow for larger labels while preserving the touch floor", () => {
    const group = css.match(/\.mobile-resolution-group \.ms-shape\s*\{([^}]*)\}/s);
    const choice = css.match(/\.mobile-resolution-group \.ms-shape__btn\s*\{([^}]*)\}/s);

    expect(group?.[1]).toMatch(/gap:\s*7px\s*;/);
    expect(choice?.[1]).toMatch(/min-width:\s*60px\s*;/);
    expect(choice?.[1]).toMatch(/min-height:\s*72px\s*;/);
    expect(choice?.[1]).toMatch(/flex:\s*1 1 max-content\s*;/);
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
    expect(sub?.[1]).toMatch(/font-size:\s*var\(--mold-fs-micro\)\s*;/);
    expect(dims?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(dims?.[1]).toMatch(/color:\s*var\(--mold-text-dim\)/);
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

    // 56px is the header's real height: a 44pt control plus its 6px padding
    // on each side. A smaller inset would leave a band the stage paints under
    // but the header eats, so a swipe there would reach nothing.
    expect(viewer?.[1]).toMatch(
      /--viewer-header-inset:\s*calc\(56px \+ env\(safe-area-inset-top\)\)/,
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

  /**
   * The body stays mounted so every action keeps its identity across the
   * toggle, which only works if the collapsed body is also out of the focus
   * order — and the open sheet has to leave the media the larger half.
   */
  it("hides the collapsed body and leaves the media the larger half", () => {
    const sheet = css.match(/(?:^|\n)\.gallery-viewer-sheet\s*\{([^}]*)\}/s);

    expect(css).toMatch(
      /\.gallery-viewer-sheet:not\(\.is-expanded\)\s+\.gallery-viewer-details\s*\{\s*visibility:\s*hidden;/,
    );
    const cap = Number(sheet?.[1]?.match(/max-height:\s*(\d+)%\s*;/)?.[1]);
    expect(cap).toBeGreaterThanOrEqual(55);
    expect(cap).toBeLessThanOrEqual(70);
  });

  /** The arrows belong to the picture, not to the box the picture sits in. */
  it("centres the paging arrows on the media, not the padded stage", () => {
    const nav = galleryViewerComponent.match(/\.gallery-viewer-nav\s*\{([^}]*)\}/s);

    expect(nav?.[1]).toMatch(
      /top:\s*calc\(\s*50% \+ \(var\(--viewer-header-inset\) - var\(--viewer-peek\)\) \/ 2\s*\)\s*;/,
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
  it("lets large Library headings scroll away and wraps their actions", () => {
    const heading = css.match(/\.mobile-library-heading\s*\{([^}]*)\}/s);

    expect(heading?.[1]).toMatch(/position:\s*relative\s*;/);
    expect(heading?.[1]).toMatch(/flex-wrap:\s*wrap\s*;/);
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
    expect(actions?.[1]).toMatch(/display:\s*flex\s*;/);
    expect(actions?.[1]).toMatch(/flex-wrap:\s*wrap\s*;/);
    expect(actionButton?.[1]).toMatch(/flex:\s*1 1 auto\s*;/);
    expect(actionButton?.[1]).toMatch(/max-width:\s*100%\s*;/);
    expect(actionButton?.[1]).toContain("font-size: min(var(--text-body), 24px)");
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
    for (const className of ["mobile-advanced-sheet", "mobile-crop-sheet"]) {
      const sheet = css.match(new RegExp(`\\.${className}\\s*\\{([^}]*)\\}`, "s"));
      expect(sheet?.[1]).toMatch(/inset:\s*0 0 auto\s*;/);
      expect(sheet?.[1]).toMatch(/height:\s*var\(--mobile-visual-viewport-height,\s*100dvh\)\s*;/);
    }
  });

  it("aligns Library sheet actions with their controls instead of the labeled field margin", () => {
    const panel = css.match(/\.mobile-library-sheet-panel\s*\{([^}]*)\}/s);
    const field = css.match(/\.mobile-library-sheet-form \.field\s*\{([^}]*)\}/s);
    expect(panel?.[1]).toMatch(
      /max-height:\s*min\(\s*86dvh,\s*calc\(var\(--mobile-visual-viewport-height,\s*100dvh\) - env\(safe-area-inset-top\)\)\s*\)\s*;/,
    );
    expect(field?.[1]).toMatch(/margin:\s*0\s*;/);
  });

  it("uses the iPhone radii scale for chips, covers, and collection cards", () => {
    const chip = css.match(/\.mobile-library-chip\s*\{([^}]*)\}/s);
    const cover = css.match(/\.mobile-collection-cover\s*\{([^}]*)\}/s);
    const row = css.match(/\.mobile-collection-row\s*\{([^}]*)\}/s);
    expect(chip?.[1]).toMatch(/border-radius:\s*var\(--mold-radius-2\)\s*;/);
    expect(cover?.[1]).toMatch(/border-radius:\s*var\(--mold-radius-2\)\s*;/);
    expect(row?.[1]).toMatch(/border-radius:\s*var\(--mold-radius-3\)\s*;/);
  });
});

describe("mobile style chip and sheet", () => {
  const styleMenu = readFileSync("../studio/components/StyleMenu.vue", "utf8");

  it("gives the chip a 44pt target with the plain name in sans and the id in mono", () => {
    const chip = css.match(/\.mobile-style-picker-chip\s*\{([^}]*)\}/s);
    const name = css.match(/\.mobile-style-chip-name\s*\{([^}]*)\}/s);
    const id = css.match(/\.mobile-style-id\s*\{([^}]*)\}/s);
    expect(Number(chip?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(name?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(id?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
  });

  it("keeps sheet rows finger-sized and everything editable inside them readable", () => {
    // The shared menu's touch mode is what the phone gets; pin both halves so
    // a desktop-sized row can never reach a thumb.
    const row = styleMenu.match(/\.ms-model__menu--touch \.ms-model__option\s*\{([^}]*)\}/s);
    const filter = styleMenu.match(
      /\.ms-model__menu--touch \.ms-model__filter input\s*\{([^}]*)\}/s,
    );
    expect(Number(row?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    // --mold-fs-md is 1rem: 16px, below which iOS zooms the page on focus.
    expect(row?.[1]).toMatch(/font-size:\s*var\(--mold-fs-md/);
    expect(Number(filter?.[1]?.match(/height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(40);
    expect(filter?.[1]).toMatch(/font-size:\s*var\(--mold-fs-md/);
  });

  // The panel takes focus programmatically so keys stay inside it; it is a
  // container, not a control, and a ring drawn along its rounded top edge
  // read as a stray highlight on every open.
  it("draws no focus ring on the sheet panel itself", () => {
    const focus = css.match(
      /\.mobile-library-sheet-panel:focus,\s*\n\.mobile-sheet-panel:focus\s*\{([^}]*)\}/s,
    );
    expect(focus?.[1]).toMatch(/outline:\s*none\s*;/);
  });

  it("rises from the bottom edge with a grabber, a scrim, and a centred iOS header", () => {
    const sheet = css.match(/\.mobile-sheet\s*\{([^}]*)\}/s);
    const open = css.match(/\.mobile-sheet\.is-open\s*\{([^}]*)\}/s);
    const panel = css.match(/\.mobile-sheet-panel\s*\{([^}]*)\}/s);
    const grabber = css.match(
      /\.mobile-library-sheet-grabber,\s*\n\.mobile-sheet-grabber\s*\{([^}]*)\}/s,
    );
    const scrim = css.match(
      /\.mobile-library-sheet-backdrop,\s*\n\.mobile-sheet-scrim\s*\{([^}]*)\}/s,
    );
    const title = css.match(/\.mobile-sheet-title\s*\{([^}]*)\}/s);
    const action = css.match(/\.mobile-sheet-action\s*\{([^}]*)\}/s);

    expect(sheet?.[1]).toMatch(/position:\s*fixed\s*;/);
    expect(sheet?.[1]).toMatch(/justify-content:\s*flex-end\s*;/);
    expect(open?.[1]).toMatch(/display:\s*flex\s*;/);
    expect(panel?.[1]).toContain("78dvh");
    expect(panel?.[1]).toMatch(/border-radius:\s*18px 18px 0 0\s*;/);
    // The grabber and the scrim are one rule the library sheet shares.
    expect(grabber?.[1]).toMatch(/height:\s*4px\s*;/);
    expect(scrim?.[1]).toMatch(/position:\s*absolute\s*;/);
    // iOS sheet title: 17px semibold sans, centred.
    expect(title?.[1]).toMatch(/font-size:\s*17px/);
    expect(title?.[1]).toMatch(/font-weight:\s*600\s*;/);
    expect(title?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    // Bar buttons are text, not circular glyphs, and still 44pt.
    expect(Number(action?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(action?.[1]).toMatch(/border:\s*0\s*;/);
    expect(action?.[1]).toMatch(/color:\s*var\(--mold-blue\)\s*;/);
  });

  it("keeps the sheet body scrolling under the phone containment invariants", () => {
    const body = css.match(/\.mobile-sheet-body\s*\{([^}]*)\}/s);
    expect(body?.[1]).toMatch(/overflow-y:\s*auto\s*;/);
    expect(body?.[1]).toMatch(/overscroll-behavior:\s*none\s*;/);
    expect(body?.[1]).toContain("env(safe-area-inset-left)");
    expect(body?.[1]).toContain("env(safe-area-inset-right)");
    expect(body?.[1]).toContain("env(safe-area-inset-bottom)");
  });
});

describe("iOS type vocabulary", () => {
  it("says a row label in plain sans and keeps mono uppercase for group headers", () => {
    const label = css.match(/\.field > span\s*\{([^}]*)\}/s);
    // 15px sans, sentence case: a form row's label is a plain word, not a
    // machine token. `--text-body-lg` is the phone bridge's 0.9375rem.
    expect(label?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(label?.[1]).toMatch(/font-size:\s*var\(--text-body-lg\)/);
    expect(label?.[1]).not.toMatch(/text-transform:\s*uppercase/);

    // A GROUP heading is still mono uppercase — that is what separates the
    // two, and it is the only place the utility face is left on this screen.
    const legends = css.match(
      /\.mobile-compact-fieldset > legend,\s*\n\.mobile-generate-legend,[\s\S]*?\{([^}]*)\}/s,
    );
    expect(legends?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(legends?.[1]).toMatch(/text-transform:\s*uppercase/);
  });

  it("says a disclosure line in the same plain sans as every other row label", () => {
    // The type pass that made `.field > span` sans reached the form rows and
    // stopped; the source-media disclosure kept the mono utility face, so the
    // one sentence on Make that is plain words read as a machine token.
    const summary = css.match(
      /\.mobile-native-disclosure > summary,\s*\n\.mobile-disclosure-button\s*\{([^}]*)\}/s,
    );
    expect(summary?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(summary?.[1]).toMatch(/font-size:\s*var\(--text-body-lg\)/);
    expect(summary?.[1]).not.toMatch(/font-family:\s*var\(--font-utility\)/);

    // Its trailing filename summary is the same sans, one step down.
    const detail = css.match(
      /\.mobile-native-disclosure > summary small,\s*\n\.mobile-disclosure-summary\s*\{([^}]*)\}/s,
    );
    expect(detail?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(detail?.[1]).toMatch(/font-size:\s*var\(--text-caption\)/);
  });

  it("gives every screen its own large title instead of one wordmark", () => {
    const title = css.match(/\.mobile-large-title\s*\{([^}]*)\}/s);
    expect(title?.[1]).toMatch(/font-family:\s*var\(--font-display\)/);
    expect(title?.[1]).toMatch(/font-size:\s*min\(var\(--text-display\), 28px\)/);
    expect(title?.[1]).toMatch(/font-weight:\s*700/);
    // The wordmark named the app on all five screens and answered nothing.
    expect(css).not.toContain(".mobile-wordmark");
    expect(mobileAppComponent).not.toContain("mobile-wordmark");

    // One 44pt action per screen, beside the title.
    const action = css.match(/\.mobile-header-action\s*\{([^}]*)\}/s);
    expect(Number(action?.[1]?.match(/width:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(Number(action?.[1]?.match(/height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);

    // The one control that decides where every print goes is a finger target,
    // and it is invisible (opacity 0 over the chip), so the CHIP must show the
    // focus a keyboard or Full Keyboard Access puts on it.
    const chip = css.match(/\.mobile-header-routing-chip\s*\{([^}]*)\}/s);
    expect(Number(chip?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    const select = css.match(/\.mobile-header-routing-select\s*\{([^}]*)\}/s);
    expect(select?.[1]).toMatch(/inset:\s*0/);
    expect(css).toMatch(
      /\.mobile-header-routing-chip:has\(\.mobile-header-routing-select:focus-visible\)\s*\{[^}]*outline:/s,
    );

    // The one action a screen offers shows its own focus too.
    expect(css).toMatch(/\.mobile-header-action:focus-visible\s*\{[^}]*outline:/s);

    // Where the next print lands is pinned above the scroll, not in it.
    const row = css.match(/\.mobile-header-routing\s*\{([^}]*)\}/s);
    expect(row?.[1]).toMatch(/display:\s*flex/);
    const note = css.match(/\.mobile-header-routing-note\s*\{([^}]*)\}/s);
    expect(note?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
  });

  it("makes every Back control a 15px sans accent label with a chevron", () => {
    const back = css.match(/\.mobile-back-button\s*\{([^}]*)\}/s);
    const close = css.match(/\.gallery-viewer-close\s*\{([^}]*)\}/s);
    for (const control of [back, close]) {
      expect(control?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
      expect(control?.[1]).toMatch(/font-size:\s*var\(--text-body-lg\)/);
      expect(Number(control?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    }
    // The viewer's leading control is a Back chevron, never a × dismiss.
    expect(galleryViewerComponent).toMatch(
      /<span aria-hidden="true">‹<\/span>\s*<span>Back<\/span>/,
    );
  });

  it("names the phone's machines lexicon on the host detail Back control", () => {
    expect(mobileHostDetailComponent).toContain("‹</span> Machines");
    expect(mobileHostDetailComponent).not.toContain("‹</span> Hosts");
  });

  it("gives the library sheet head the iOS sheet title, not a mono kicker", () => {
    const head = css.match(/\.mobile-library-sheet-head\s*\{([^}]*)\}/s);
    expect(head?.[1]).toMatch(/font-family:\s*var\(--font-body\)/);
    expect(head?.[1]).toMatch(/font-size:\s*17px/);
    expect(head?.[1]).toMatch(/font-weight:\s*600\s*;/);
    expect(head?.[1]).not.toMatch(/text-transform:\s*uppercase/);
  });
});

describe("mobile library scope control", () => {
  it("keeps the shared segmented control finger-sized in 13px sans", () => {
    const row = css.match(/\.mobile-library-scope\s*\{([^}]*)\}/s);
    const segment = css.match(/\.mobile-library-scope button\s*\{([^}]*)\}/s);
    // The control owns its own layout; the phone only raises the target and
    // the type, and never re-declares `display`.
    expect(row?.[1]).not.toMatch(/display:\s*grid/);
    expect(Number(segment?.[1]?.match(/min-height:\s*(\d+)px/)?.[1])).toBeGreaterThanOrEqual(44);
    expect(segment?.[1]).toMatch(/font-size:\s*var\(--mold-fs-sm\)/);
    expect(segment?.[1]).toMatch(/font-weight:\s*650\s*;/);
    expect(css).not.toContain(".mobile-library-scope-count");
  });
});
