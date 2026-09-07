/**
 * The canvas — shape, size, and what a source image does to both — resolved
 * ONCE for every control that offers it.
 *
 * The Create rail and the composer's Shape chip are two doors onto one
 * decision, and while the chip derived its own label from the raw pixels
 * (`outputFamilyLabel(width, height)`) the two could disagree: the rail
 * highlighted **Source** while the chip read "Square · 1024", and an
 * off-ladder size the rail marked approximate the chip stated flatly. Both
 * now read `resolveOutputShape`, which is the repo's Output-shape invariant
 * (`.claude/rules/studio-web.md`): chips, size pills, badge and status
 * sentence all read that one object.
 *
 * Writing is the other half. `onShape`, `onResolution` and `matchSource` are
 * the only paths that set `form.width`/`height`, and each reports the canvas
 * intent it implies — so whichever control the user reached for, the other
 * agrees on the next render.
 */
import { computed, type ComputedRef } from "vue";
import {
  intentForCanvas,
  resolveOutputShape,
  sizeForFamily,
  SOURCE_FAMILY_ID,
  type CanvasIntent,
  type OutputShapeInput,
  type OutputShapeModel,
} from "@studio/lib/outputShape";
import { resolveSourceResolution } from "@studio/lib/sourceResolution";
import type { GenerateForm } from "../lib/generateForm";

export interface OutputShapeOptions {
  form: () => GenerateForm;
  canvasIntent: () => CanvasIntent;
  /** The picked catalog row, when one is known — the shape ladder's author. */
  contractModel: () => OutputShapeModel | null | undefined;
  /** Whether this recipe reads a still at all. A parked image must not
   * project Source controls for a checkpoint that cannot carry it. */
  supportsSourceImage: () => boolean;
  /** Told which intent the write implies, so the owner can persist it. */
  onCanvasIntent: (intent: CanvasIntent) => void;
}

export function useOutputShape(options: OutputShapeOptions) {
  const sourceDimensions = computed(() => {
    if (!options.supportsSourceImage()) return null;
    const form = options.form();
    return form.sourceImageWidth && form.sourceImageHeight
      ? { width: form.sourceImageWidth, height: form.sourceImageHeight }
      : null;
  });
  const sourceResolution = computed(() => {
    const form = options.form();
    return sourceDimensions.value
      ? resolveSourceResolution(
          sourceDimensions.value,
          options.contractModel() ?? form.family,
          form.pipeline,
        )
      : null;
  });
  const shapeInput: ComputedRef<OutputShapeInput> = computed(() => {
    const form = options.form();
    return {
      model: options.contractModel() ?? null,
      family: form.family,
      pipeline: form.pipeline,
      width: form.width,
      height: form.height,
      source: sourceDimensions.value,
      intent: options.canvasIntent(),
    };
  });
  const outputShape = computed(() => resolveOutputShape(shapeInput.value));
  const followsSource = computed(
    () =>
      outputShape.value.state === "follows-source" || outputShape.value.state === "matches-source",
  );
  const resolutionOptions = computed(() =>
    outputShape.value.sizes.map((size) => ({
      id: size.id,
      mp: (size.width * size.height) / 1_000_000,
      label: size.label,
      sub: size.mark ? `${size.megapixels} · ${size.mark}` : size.megapixels,
      width: size.width,
      height: size.height,
    })),
  );

  function write(width: number, height: number, intent: CanvasIntent) {
    const form = options.form();
    options.onCanvasIntent(intent);
    form.width = width;
    form.height = height;
  }
  function onShape(id: string) {
    const size = sizeForFamily(id, shapeInput.value);
    if (!size) return;
    write(size.width, size.height, id === SOURCE_FAMILY_ID ? "source" : "manual");
  }
  function onResolution(id: string | number) {
    const size = outputShape.value.sizes.find((candidate) => candidate.id === id);
    if (!size) return;
    write(size.width, size.height, intentForCanvas(shapeInput.value, size));
  }
  function matchSource() {
    const source = sourceResolution.value;
    if (!source) return;
    write(source.output.width, source.output.height, "source-exact");
  }

  return {
    shapeInput,
    outputShape,
    sourceResolution,
    followsSource,
    shapeOptions: computed(() => outputShape.value.families),
    shapeId: computed(() => outputShape.value.selectedFamilyId),
    shapeApproximate: computed(() => outputShape.value.approximate),
    resolutionRatio: computed(() => options.form().width / options.form().height),
    resolutionOptions,
    resolutionSizeId: computed(() => outputShape.value.selectedSizeId),
    onShape,
    onResolution,
    matchSource,
  };
}
