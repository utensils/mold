/**
 * Runtime-resolved pipeline metadata is provenance, not an authored override.
 *
 * An automatically chained long clip records the pipeline its stages used, and
 * restoring that value as an explicit selector disables the same chain route
 * and collapses the duration control to "1 generation".
 *
 * Scene-by-scene authoring is retired, so every reuse now rebuilds a ONE-SHOT
 * — a print stitched from a scripted sequence included. That removes the one
 * shape whose recorded pipeline used to be worth keeping: replaying a
 * sequence's pipeline onto a one-shot form is exactly the case this guard
 * exists to prevent. Only an explicitly requested pipeline survives.
 */
export function pipelineForSettingsReuse<T>(metadata: {
  pipeline?: T | null;
  pipeline_requested?: boolean | null;
}): T | null {
  return metadata.pipeline_requested === true
    ? (metadata.pipeline ?? null)
    : null;
}

/**
 * The transparent-background toggle a Reuse restores. Only the REQUEST's own
 * `transparent_background` answers: `has_alpha` is a fact about the file (an
 * edit of a transparent reference keeps its alpha with the toggle off), not a
 * setting anybody chose.
 */
export function transparentBackgroundForSettingsReuse(metadata: {
  transparent_background?: boolean | null;
}): boolean {
  return metadata.transparent_background === true;
}
