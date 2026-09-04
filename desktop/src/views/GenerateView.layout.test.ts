import { describe, expect, it } from "vitest";
import viewSource from "./GenerateView.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";
import modelPickerSource from "../components/create/ModelPicker.vue?raw";
import advancedSource from "../components/create/AdvancedSettings.vue?raw";
import sequenceComposerSource from "../components/create/SequenceComposer.vue?raw";
import composerCardSource from "../components/create/ComposerCard.vue?raw";
import sceneLaneSource from "../components/create/SceneLane.vue?raw";

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
    // The composer takes its own height under the canvas; the canvas absorbs slack.
    expect(classesFor(viewSource, "generate-composer")).toContain("shrink-0");
  });

  it("keeps the canvas visible and makes the bottom bench resizable", () => {
    expect(tagFor(viewSource, "generate-workbench")).toContain('ref="workbenchRef"');
    expect(tagFor(viewSource, "create-bench-resizer")).toContain('role="separator"');
    expect(tagFor(viewSource, "create-bench-resizer")).toContain('@pointerdown="startBenchResize"');
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("height: `${benchHeight}px`");
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("containerType: 'size'");
    expect(tagFor(viewSource, "create-bottom-panel")).toContain("containerName: 'create-bench'");
    expect(viewSource).toContain("min-h-[320px]");
    expect(viewSource).not.toContain('class="absolute inset-0 h-full w-full object-cover"');
  });

  it("fills the bottom panel and pins the timeline's readout to its bottom edge", () => {
    expect(classesFor(viewSource, "create-bottom-panel")).toContain("flex");
    expect(classesFor(viewSource, "create-bottom-panel")).toContain("flex-col");
    expect(classesFor(viewSource, "generate-sequence-shell")).toContain("min-h-[228px]");
    expect(classesFor(viewSource, "generate-sequence-shell")).toContain("flex-[1_0_228px]");
    expect(classesFor(viewSource, "generate-sequence-composer")).toContain("flex-1");
    // The composer takes its own height under the canvas; the canvas absorbs slack.
    expect(classesFor(viewSource, "generate-composer")).toContain("shrink-0");
    expect(sequenceComposerSource).toMatch(/\.ms-timeline__foot\s*\{[^}]*margin-top:\s*auto/s);
    expect(sequenceComposerSource).toContain('data-test="sequence-fit"');
    // The one-shot composer is no longer a bench panel: it is a card under
    // the canvas whose control row carries Generate, so its actions have no
    // bottom edge of their own to pin to.
    expect(composerCardSource).toMatch(/\.ms-composer__controls\s*\{[^}]*display:\s*flex/s);
    expect(tagFor(composerCardSource, "generate-button")).toContain("ms-composer__generate");
  });

  it("keeps full model names visible in the shared model picker", () => {
    expect(classesFor(modelPickerSource, "selected-model-name")).not.toContain("truncate");
    expect(classesFor(modelPickerSource, "selected-model-name")).toContain("break-all");
    expect(classesFor(modelPickerSource, "model-option-name")).not.toContain("truncate");
    expect(classesFor(modelPickerSource, "model-option-name")).toContain("break-all");
    expect(classesFor(modelPickerSource, "model-availability")).not.toContain("truncate");
    expect(classesFor(modelPickerSource, "model-availability")).toContain("break-all");
  });

  it("keeps Advanced in the settings inspector instead of mounting an overlay drawer", () => {
    expect(viewSource).not.toContain("<AdvancedDrawer");
    expect(inspectorSource).toContain("<AdvancedSettings");
    expect(advancedSource).toContain('data-test="inline-advanced"');
  });

  it("renders an instructive brand blank-canvas placeholder before the first print", () => {
    expect(viewSource).toContain('data-test="empty-canvas"');
    expect(tagFor(viewSource, "preview-frame")).toContain("bg-media-bed");
    expect(viewSource).toContain("Your picture appears here");
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
    // One composer answers for both modes. What it refuses for in each is
    // proved by mounting, in GenerateView.sequence.test.ts.
    expect(viewSource).toContain(':disabled="composerLocked"');
    expect(viewSource).toContain(':disabled-reason="composerRefusal"');
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

  it("lets clip mode shrink the scenes lane on resize instead of growing scrollbars", () => {
    // The bench floor in clip mode covers the timeline's fixed chrome + the
    // lane's minimum height, so dragging the resizer compresses the lane
    // rather than overflowing into a scrollbar.
    expect(viewSource).toContain("MIN_SEQUENCE_BENCH_HEIGHT");
    expect(viewSource).toMatch(/function minBenchHeight\(\)/);
    expect(viewSource).toMatch(/Math\.max\(minBenchHeight\(\), available - MIN_CANVAS_HEIGHT\)/);
    expect(viewSource).toMatch(/Math\.max\(minBenchHeight\(\), height\)/);
    // Switching Output re-clamps the persisted height against the new floor.
    expect(viewSource).toMatch(/watch\(isSequence, [\s\S]{0,400}?clampBenchToViewport\(\)/);
    // The timeline root must opt out of min-content flooring: floored at auto,
    // it counts the lane's preferred basis and the panel scrolls before the
    // lane's shrink weight ever engages.
    expect(sequenceComposerSource).toMatch(/\.ms-timeline\s*\{[^}]*min-height:\s*0/s);
    // The preferred lane height must be the flex BASIS, never a `height`: a
    // specified height becomes the wrapper's min-content contribution and
    // resurrects the scrollbar the shrink weight exists to prevent.
    expect(sequenceComposerSource).toMatch(
      /\.ms-timeline__lanewrap\s*\{[^}]*flex:\s*0\s+999\s+96px/s,
    );
    expect(sequenceComposerSource).not.toMatch(
      /\.ms-timeline__lanewrap\s*\{[^}]*[\s;]height:\s*\d/s,
    );
    expect(sequenceComposerSource).toMatch(/\.ms-timeline__lanewrap\s*\{[^}]*min-height:/s);
    // Every block is as wide as the time it plays, so the lane fits its width
    // and never scrolls.
    expect(sceneLaneSource).toMatch(/flexGrow: `\$\{playedFrames\(clip, index\) \/ fps\}`/);
    expect(sceneLaneSource).toMatch(/\.ms-lane\s*\{[^}]*flex:\s*1/s);
  });

  // The floating Templates popover is gone: starting points are a tab in the
  // inspector, so there is no overlay for a document-level Escape to dismiss
  // and no trigger to restore focus to. The tab itself is covered by
  // `InspectorPanel.test.ts` and `CreateHeader.test.ts`.
  it("reaches starting points and recent settings through the inspector's tabs", () => {
    expect(viewSource).not.toContain("templatesToggleEl");
    expect(viewSource).toContain('const inspectorTab = ref<InspectorTab>("settings")');
    expect(viewSource).toContain('@open-tab="inspectorTab = $event"');
    expect(viewSource).toContain('@update:tab="inspectorTab = $event"');
    expect(viewSource).toContain('@load-template="loadTemplate"');
  });

  it("disables Picture-in-Picture on the generated video preview", () => {
    // `v-else-if` since the audio-only branch takes precedence: an audio
    // print has no frames, so the video probe must not be the first one.
    const previewVideo =
      viewSource.match(/<video\s+v-(?:else-)?if="job\?\.resultUrl[^>]*>/s)?.[0] ?? "";
    expect(previewVideo).toContain("disablepictureinpicture");
  });
});
