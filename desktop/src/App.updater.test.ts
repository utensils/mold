import { describe, expect, it } from "vitest";
import source from "./App.vue?raw";

describe("desktop updater integration", () => {
  it("initializes automatic checks only after app preferences are available", () => {
    expect(source.indexOf("await appPrefs.init()")).toBeGreaterThan(-1);
    expect(source.indexOf("updater.init()")).toBeGreaterThan(
      source.indexOf("await appPrefs.init()"),
    );
  });

  it("confirms updater health after the native shell paints without waiting for engine startup", () => {
    expect(source).toContain("const connectionStartup = connection.init()");
    expect(source).toContain("await appWindow.show()");
    expect(source.indexOf("await updater.confirmReady()")).toBeGreaterThan(
      source.indexOf("await appWindow.show()"),
    );
    expect(source.indexOf("await updater.confirmReady()")).toBeLessThan(
      source.indexOf("await connectionStartup"),
    );
  });

  it("routes the native Check for Updates action to the Updates settings section", () => {
    expect(source).toContain('case "check-for-updates"');
    expect(source).toContain('section: "updates"');
    expect(source).toContain("void updater.check()");
  });
});
