import { describe, expect, it } from "vitest";
import source from "./GenerateView.vue?raw";

function classesFor(testId: string): string {
  const tag = source.match(new RegExp(`<[^>]*data-test="${testId}"[^>]*>`, "s"))?.[0] ?? "";
  return tag.match(/class="([^"]*)"/s)?.[1] ?? "";
}

describe("GenerateView layout", () => {
  it("keeps the composer inside the shell while the preview shrinks", () => {
    expect(classesFor("generate-layout")).toContain("min-h-0");
    expect(classesFor("generate-layout")).toContain("overflow-hidden");
    expect(classesFor("generate-workbench")).toContain("min-h-0");
    expect(classesFor("generate-workbench")).toContain("overflow-hidden");
    expect(classesFor("generate-composer")).toContain("shrink-0");
  });
});
