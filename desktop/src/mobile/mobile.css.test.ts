import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/mobile/mobile.css", "utf8");
const mobileHtml = readFileSync("index.mobile.html", "utf8");
const preparedComponent = readFileSync("src/mobile/MobilePreparedExpansionBatch.vue", "utf8");
const pullComponent = readFileSync("src/mobile/MobileExpansionPullStatus.vue", "utf8");

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
      ".mobile-resolution-aspect",
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
    expect(sources?.[1]).toMatch(/grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
    expect(containers?.[1]).toMatch(/gap:\s*8px/);
    expect(buttons?.[1]).toMatch(/min-width:\s*0/);
    expect(buttons?.[1]).toMatch(/border:\s*1px solid var\(--control-edge\)/);
    expect(buttons?.[1]).toMatch(/background:\s*var\(--bench\)/);
    expect(buttons?.[1]).toMatch(/font-family:\s*var\(--font-utility\)/);
    expect(selected?.[1]).toContain("color-mix(in srgb, var(--safelight) 11%, var(--bench))");
    expect(selected?.[1]).toMatch(/color:\s*var\(--safelight\)/);
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
