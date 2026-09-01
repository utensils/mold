import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const component = readFileSync("../ui/components/UpscaleDialog.vue", "utf8");

describe("UpscaleDialog responsive containment", () => {
  it("includes padding inside the sheet width on narrow viewports", () => {
    const dialog = component.match(/\.upscale-dialog\s*\{([^}]*)\}/s);

    expect(dialog?.[1]).toMatch(/width:\s*min\(460px,\s*100%\)\s*;/);
    expect(dialog?.[1]).toMatch(/max-width:\s*100%\s*;/);
    expect(dialog?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(dialog?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
  });

  it("lets long model names shrink within the sheet", () => {
    const field = component.match(/\.upscale-dialog__field\s*\{([^}]*)\}/s);
    const select = component.match(/select\s*\{([^}]*)\}/s);

    expect(field?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(select?.[1]).toMatch(/min-width:\s*0\s*;/);
    expect(select?.[1]).toMatch(/max-width:\s*100%\s*;/);
    expect(select?.[1]).toMatch(/box-sizing:\s*border-box\s*;/);
  });
});
