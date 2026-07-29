import { describe, expect, it } from "vitest";
import viewSource from "./GenerateView.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";
import modelPickerSource from "../components/create/ModelPicker.vue?raw";
import advancedSource from "../components/create/AdvancedSettings.vue?raw";

function tagFor(source: string, testId: string): string {
  return source.match(new RegExp(`<[^>]*data-test="${testId}"[^>]*>`, "s"))?.[0] ?? "";
}

function classesFor(source: string, testId: string): string {
  return tagFor(source, testId).match(/class="([^"]*)"/s)?.[1] ?? "";
}

describe("GenerateView layout", () => {
  it("keeps the composer pinned inside the shell while the canvas shrinks", () => {
    expect(classesFor(viewSource, "generate-layout")).toContain("min-h-0");
    expect(classesFor(viewSource, "generate-layout")).toContain("overflow-hidden");
    expect(classesFor(viewSource, "generate-workbench")).toContain("min-h-0");
    expect(classesFor(viewSource, "generate-workbench")).toContain("overflow-hidden");
    expect(classesFor(viewSource, "generate-composer")).toContain("shrink-0");
  });

  it("keeps full model names visible in the shared model picker", () => {
    expect(classesFor(modelPickerSource, "selected-model-name")).not.toContain("truncate");
    expect(classesFor(modelPickerSource, "selected-model-name")).toContain("break-all");
    expect(classesFor(modelPickerSource, "model-option-name")).not.toContain("truncate");
    expect(classesFor(modelPickerSource, "model-option-name")).toContain("break-all");
    expect(classesFor(modelPickerSource, "model-availability")).toContain("whitespace-normal");
    expect(classesFor(modelPickerSource, "model-availability")).toContain("break-all");
  });

  it("keeps Advanced in the settings inspector instead of mounting an overlay drawer", () => {
    expect(viewSource).not.toContain("<AdvancedDrawer");
    expect(inspectorSource).toContain("<AdvancedSettings");
    expect(advancedSource).toContain('data-test="inline-advanced"');
  });

  it("renders an instructive brand blank-canvas placeholder before the first print", () => {
    expect(viewSource).toContain('data-test="empty-canvas"');
    expect(tagFor(viewSource, "preview-frame")).toContain("bg-print-surface");
    expect(viewSource).toContain("Your print develops here");
    expect(viewSource).toContain("Describe an image below, pick a look, and press Generate.");
  });

  it("dismisses the Templates popover on document-level Escape and restores trigger focus", () => {
    expect(viewSource).toContain('document.addEventListener("keydown", onDocumentKeydown)');
    expect(viewSource).toContain('document.removeEventListener("keydown", onDocumentKeydown)');
    expect(viewSource).toContain('event.key !== "Escape"');
    expect(viewSource).toContain("templatesToggleEl.value?.focus()");
  });

  it("disables Picture-in-Picture on the generated video preview", () => {
    const previewVideo = viewSource.match(/<video\s+v-if="job\?\.resultUrl[^>]*>/s)?.[0] ?? "";
    expect(previewVideo).toContain("disablepictureinpicture");
  });
});
