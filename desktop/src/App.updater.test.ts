import { describe, expect, it } from "vitest";
import source from "./App.vue?raw";

describe("desktop updater integration", () => {
  it("initializes automatic checks only after app preferences are available", () => {
    expect(source.indexOf("await appPrefs.init()")).toBeGreaterThan(-1);
    expect(source.indexOf("updater.init()")).toBeGreaterThan(
      source.indexOf("await appPrefs.init()"),
    );
  });

  it("surfaces automatic update checks in persistent app chrome", () => {
    expect(source).toContain('import UpdateBanner from "./components/shell/UpdateBanner.vue"');
    expect(source).toContain("<UpdateBanner />");
    expect(source).not.toContain("confirmReady");
  });

  it("routes the native Check for Updates action to the Updates settings section", () => {
    expect(source).toContain('case "check-for-updates"');
    expect(source).toContain('section: "updates"');
    expect(source).toContain("void updater.check()");
  });
});
