import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
const here = dirname(fileURLToPath(import.meta.url));
import { readFileSync } from "node:fs";
import { describe, it, expect } from "vitest";

describe("source inspection touch and ancestor key contracts", () => {
  it("keeps every source inspection action at the Android 48dp minimum", () => {
    const source = readFileSync(
      resolve(here, "OriginalMediaButton.vue"),
      "utf8",
    );
    const minimum = source.match(/min-height:\s*(\d+)px/);
    expect(Number(minimum?.[1])).toBeGreaterThanOrEqual(48);
  });
  it("lets QueueRail row keys act only on the row itself", () => {
    const source = readFileSync(
      resolve(here, "../../desktop/src/components/shell/QueueRail.vue"),
      "utf8",
    );
    for (const action of ["commands.open(active)", "commands.open(row)"]) {
      expect(source).toContain(`@keydown.enter.self.prevent="${action}"`);
      expect(source).toContain(`@keydown.space.self.prevent="${action}"`);
    }
  });
});
