import { describe, expect, it } from "vitest";
import viewSource from "./GenerateView.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";
import modelPickerSource from "../components/create/ModelPicker.vue?raw";
import advancedSource from "../components/create/AdvancedSettings.vue?raw";
import sequenceComposerSource from "../components/create/SequenceComposer.vue?raw";
import composerCardSource from "../components/create/ComposerCard.vue?raw";
import activityStripSource from "../components/create/ActivityStrip.vue?raw";

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
    expect(classesFor(viewSource, "create-bottom-panel")).toContain("overflow-hidden");
    expect(classesFor(viewSource, "create-bottom-panel")).not.toContain("overflow-y-auto");
    expect(classesFor(viewSource, "generate-composer")).toContain("flex-1");
  });

  it("keeps activity as the bottom bench's only scroll surface", () => {
    expect(classesFor(activityStripSource, "activity-list-scroll")).toContain("ms-activity__list");
    expect(activityStripSource).toMatch(/\.ms-activity\s*\{[^}]*min-height:\s*0/s);
    expect(activityStripSource).toMatch(/\.ms-activity\s*\{[^}]*max-height:/s);
    expect(activityStripSource).toMatch(/\.ms-activity__list\s*\{[^}]*min-height:\s*0/s);
    expect(activityStripSource).toMatch(/\.ms-activity__list\s*\{[^}]*overflow-y:\s*auto/s);
  });

  it("keeps the canvas visible and makes the bottom bench resizable", () => {
    expect(tagFor(viewSource, "generate-workbench")).toContain('ref="workbenchRef"');
    expect(tagFor(viewSource, "create-bench-resizer")).toContain('role="separator"');
    expect(tagFor(viewSource, "create-bench-resizer")).toContain('@pointerdown="startBenchResize"');
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("height: `${benchHeight}px`");
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("containerType: 'size'");
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("containerName: 'create-bench'");
    expect(viewSource).toContain("min-h-[144px]");
    expect(viewSource).not.toContain('class="absolute inset-0 h-full w-full object-cover"');
  });

  it("fills the bottom panel and pins composer actions to its bottom edge", () => {
    expect(classesFor(viewSource, "create-bottom-panel")).toContain("flex");
    expect(classesFor(viewSource, "create-bottom-panel")).toContain("flex-col");
    expect(classesFor(viewSource, "generate-sequence-shell")).toContain("min-h-[300px]");
    expect(classesFor(viewSource, "generate-sequence-shell")).toContain("flex-[1_0_300px]");
    expect(classesFor(viewSource, "generate-sequence-composer")).toContain("flex-1");
    expect(classesFor(viewSource, "generate-composer")).toContain("flex-1");
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench__footer\s*\{[^}]*margin-top:\s*auto/s);
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench__clip\s*\{[^}]*flex:\s*1/s);
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench__prompt--main\s*\{[^}]*flex:\s*1/s);
    expect(composerCardSource).toMatch(/\.ms-composer__actions\s*\{[^}]*margin-top:\s*auto/s);
    expect(composerCardSource).toMatch(/\.ms-composer__bench\s*\{[^}]*flex:\s*1/s);
    expect(sequenceComposerSource).toContain('data-test="sequence-composer-footer"');
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

  // A conditioned LTX-2 render may go out undescribed. The view owns the
  // shared blocker authority; obvious required inputs can disable the button
  // without taking over the composer with corrective guidance.
  it("keeps Generate gating separate from visible blocker guidance", () => {
    expect(tagFor(composerCardSource, "generate-button")).toContain(':disabled="generateDisabled"');
    expect(composerCardSource).toContain("props.disabled && !props.submitting");
    expect(composerCardSource).toContain('<ActionBlocker v-if="disabledReason"');
    expect(composerCardSource).not.toContain("!form.prompt.trim() || !form.model");

    expect(viewSource).toContain('from "@studio/lib/promptRequirement"');
    // The recipe rides along: `promptInputForForm` projects the form's
    // snapshotted `promptMode` back onto the shared rule, so a recipe that
    // IGNORES the prompt enables Generate with an empty one.
    expect(viewSource).toMatch(
      /const promptMissing = computed\(\s*\(\) => promptRequired\(promptInputForForm\(form\)\) && !form\.prompt\.trim\(\),\s*\);/,
    );
    expect(viewSource).toContain("const generationInputBlockerReason = computed");
    expect(viewSource).toContain("if (generationInputBlockerReason.value ||");
    expect(viewSource).toContain(':disabled="composerDisabled"');
    expect(viewSource).toContain(':disabled-reason="composerBlockerReason"');
  });

  it("takes the blank-canvas guidance from the shared prompt rule", () => {
    expect(viewSource).toContain(':guidance="emptyCanvasGuidance"');
    expect(viewSource).toContain("promptGuidance(");
  });

  it("aspect-fits the settled sequence video inside the canvas instead of clipping it", () => {
    // The settled result must use the same pure-CSS containment as the
    // develop preview frame: a size-container region and a frame sized to
    // the print's own aspect ratio. A width-full frame let the video derive
    // its height from the canvas width and clip top/bottom on short regions.
    expect(classesFor(viewSource, "sequence-result-stage")).toContain("[container-type:size]");
    expect(classesFor(viewSource, "sequence-result-stage")).toContain("min-h-0");
    expect(tagFor(viewSource, "sequence-result-frame")).toContain(':style="settledFrameStyle"');
    expect(viewSource).toMatch(/const settledFrameStyle = computed/);
  });

  it("lets sequence mode shrink the filmstrip on resize instead of growing scrollbars", () => {
    // The bench floor in sequence mode covers the composer's fixed chrome +
    // the filmstrip's minimum height, so dragging the resizer compresses the
    // rail (fluid cqh sizing) rather than overflowing into a scrollbar.
    expect(viewSource).toContain("MIN_SEQUENCE_BENCH_HEIGHT");
    expect(viewSource).toMatch(/function minBenchHeight\(\)/);
    expect(viewSource).toMatch(/Math\.max\(minBenchHeight\(\), available - MIN_CANVAS_HEIGHT\)/);
    expect(viewSource).toMatch(/Math\.max\(minBenchHeight\(\), height\)/);
    // Switching Output re-clamps the persisted height against the new floor.
    expect(viewSource).toMatch(/watch\(isSequence, [\s\S]{0,400}?clampBenchToViewport\(\)/);
    // The bench root must opt out of min-content flooring: floored at auto,
    // it counts the rail's 204px basis (not its 104px floor) and the panel
    // scrolls before the filmstrip's shrink weight ever engages.
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench\s*\{[^}]*min-height:\s*0/s);
    // The preferred rail height must be the flex BASIS, never a `height`: a
    // specified height becomes the wrapper's min-content contribution and
    // resurrects the scrollbar the shrink weight exists to prevent.
    expect(sequenceComposerSource).toMatch(
      /\.ms-seqbench__railwrap\s*\{[^}]*flex:\s*0\s+999\s+204px/s,
    );
    expect(sequenceComposerSource).not.toMatch(
      /\.ms-seqbench__railwrap\s*\{[^}]*[\s;]height:\s*\d/s,
    );
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench__railwrap\s*\{[^}]*min-height:/s);
    expect(sequenceComposerSource).toMatch(/\.ms-seqbench__rail\s*\{[^}]*height:\s*100%/s);
  });

  it("dismisses the Templates popover on document-level Escape and restores trigger focus", () => {
    expect(viewSource).toContain('document.addEventListener("keydown", onDocumentKeydown)');
    expect(viewSource).toContain('document.removeEventListener("keydown", onDocumentKeydown)');
    expect(viewSource).toContain('event.key !== "Escape"');
    expect(viewSource).toContain("templatesToggleEl.value?.focus()");
  });

  it("disables Picture-in-Picture on the generated video preview", () => {
    // `v-else-if` since the audio-only branch takes precedence: an audio
    // print has no frames, so the video probe must not be the first one.
    const previewVideo =
      viewSource.match(/<video\s+v-(?:else-)?if="job\?\.resultUrl[^>]*>/s)?.[0] ?? "";
    expect(previewVideo).toContain("disablepictureinpicture");
  });
});
