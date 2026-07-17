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
  it("keeps full model names visible in the picker", () => {
    expect(classesFor("selected-model-name")).not.toContain("truncate");
    expect(classesFor("selected-model-name")).toContain("break-all");
    expect(classesFor("model-option-name")).not.toContain("truncate");
    expect(classesFor("model-option-name")).toContain("break-all");
    expect(classesFor("model-availability")).toContain("whitespace-normal");
    expect(classesFor("model-availability")).toContain("break-all");
  });

  it("renders an instructive blank-canvas placeholder before the first print", () => {
    expect(source).toContain('data-test="empty-canvas"');
    expect(source).toContain(":class=\"job ? 'bg-print-surface' : 'bg-empty-surface'\"");
    expect(source).toContain("No print yet");
    expect(source).toContain("Choose a model, describe your print, then generate.");
  });
});
