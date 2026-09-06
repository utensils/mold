import { describe, expect, it, vi } from "vitest";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

import viewSource from "./GenerateView.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";
import modelPickerSource from "../components/create/ModelPicker.vue?raw";
import advancedSource from "../components/create/AdvancedSettings.vue?raw";
import composerCardSource from "../components/create/ComposerCard.vue?raw";
import benchLayoutSource from "../lib/benchLayout?raw";

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

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
    // The composer takes its own height under the canvas; the canvas absorbs slack.
    expect(classesFor(viewSource, "generate-composer")).toContain("shrink-0");
  });

  /*
   * The workbench is overflow-hidden and the composer is its LAST child, so
   * anything the canvas floor fails to leave is cut off the composer — the
   * prompt and the only Generate button. Replaces the old `min-h-[320px]`
   * string check, which pinned a canvas floor that contradicted the number
   * the layout actually reserved.
   */
  it("binds the canvas floor from the one layout authority", () => {
    // One authority for the floor: the canvas binds the same number the
    // layout reserves instead of hard-coding a utility beside it.
    expect(tagFor(viewSource, "generate-canvas")).toContain("minHeight: `${canvasFloor}px`");
    expect(tagFor(viewSource, "generate-canvas")).not.toContain("min-h-[");
    expect(viewSource).toContain('from "../lib/benchLayout"');
    expect(viewSource).toContain("const canvasFloor = MIN_CANVAS_HEIGHT");
    expect(viewSource).not.toContain('class="absolute inset-0 h-full w-full object-cover"');
  });

  /*
   * A clip has ONE way of being made, so the scene bench between the canvas
   * and the composer is retired along with the timeline it held. Nothing may
   * resurrect the resizer, the stored height, or the panel.
   */
  it("keeps no scene bench between the canvas and the composer", () => {
    for (const retired of [
      "create-bench-resizer",
      "create-bottom-panel",
      "generate-sequence-shell",
      "benchHeight",
      "observeComposerHeight",
      "mold.desktop.create-bench-height.v1",
    ]) {
      expect(viewSource, retired).not.toContain(retired);
    }
    expect(benchLayoutSource).not.toContain("clampBenchHeight");
    // The one-shot composer is a card under the canvas whose control row
    // carries Generate, so its actions have no bottom edge of their own.
    expect(composerCardSource).toMatch(/\.ms-composer__controls\s*\{[^}]*display:\s*flex/s);
    expect(tagFor(composerCardSource, "generate-button")).toContain("ms-composer__generate");
  });

  /*
   * The caption is a child of the ASPECT-FITTED frame, so its width is the
   * print's, not the canvas's: an 832×1216 portrait at the minimum window
   * gives it ~287px where the actions need ~420, and `overflow-hidden` cut
   * Make bigger and the ⋯ button off with no scroll and no cue. The frame
   * measures itself and the word actions collapse into the ⋯ menu — which is
   * only safe because that menu offers every one of them.
   */
  describe("canvas caption reachability", () => {
    const captionActions = ["canvas-save", "canvas-variations", "canvas-upscale"] as const;

    it("measures the frame the caption actually lives in", () => {
      expect(tagFor(viewSource, "preview-frame")).toContain("[container-type:inline-size]");
      expect(tagFor(viewSource, "preview-frame")).toContain("[container-name:preview-frame]");
      expect(viewSource).toMatch(
        /@container preview-frame \(max-width: \d+px\) \{\s*\.caption-action--word \{\s*display: none;/,
      );
    });

    it("collapses only the word actions, never the ⋯ menu", () => {
      for (const action of captionActions) {
        expect(classesFor(viewSource, action)).toContain("caption-action--word");
      }
      expect(classesFor(viewSource, "canvas-more")).toContain("caption-action--icon");
      expect(classesFor(viewSource, "canvas-more")).not.toContain("caption-action--word");
    });

    it("offers every collapsible action from the canvas menu", () => {
      const menu = viewSource.slice(viewSource.indexOf("function canvasMenu()"));
      // Save is the mesh/image pair the menu already carried.
      expect(menu).toContain('isMeshResult(j) ? "Save mesh" : "Save image"');
      expect(menu).toContain('label: "Make 4 variations"');
      expect(menu).toContain('label: "Make bigger"');
      // Each menu entry is gated on the same predicate as its caption button.
      expect(menu).toContain("disabled: !canMakeVariations(j)");
      expect(menu).toContain("disabled: !canUpscaleCanvasResult(j)");
      expect(menu).toContain("disabled: !canSaveCanvasResult(j)");
    });
  });

  /*
   * Two colour utilities on one element do not stack: they have equal
   * specificity, so the one Tailwind emits LAST wins whatever the attribute
   * order says. `.text-fg-dim` is emitted after `.text-accent`, which is how
   * the watched sequence's "clip 2/3 · developing…" rendered dim grey while
   * its class list ended in `text-accent`.
   */
  it("never stacks two colour utilities on one element", () => {
    const COLOUR =
      /(?<![\w:-])text-(fg|fg-2|fg-dim|fg-faint|accent|error|success|warning|sapphire|star|on-accent)(?![\w-])/g;
    const offenders = [...viewSource.matchAll(/(?<![:\w-])class="([^"]*)"/g)]
      .map((match) => match[1]!)
      .filter((attr) => [...attr.matchAll(COLOUR)].length > 1);
    expect(offenders).toEqual([]);
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
    expect(viewSource).toContain(':disabled="composerLocked"');
    expect(viewSource).toContain(':disabled-reason="composerRefusal"');
  });

  it("takes the blank-canvas guidance from the shared prompt rule", () => {
    expect(viewSource).toContain(':guidance="emptyCanvasGuidance"');
    expect(viewSource).toContain("promptGuidance(");
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
