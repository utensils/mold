/*
 * The binding lexicon (docs/design/README.md §2), as data.
 *
 * Plain words in sans, technical truth in mono, on the same row — and the
 * same plain words on every surface. Desktop's `lexicon.test.ts` and web's
 * read THESE tables rather than each declaring their own, so a word cannot be
 * retired on one surface and survive on the other. No UI imports this module;
 * it exists for the tests and for a future palette that wants the words.
 */

/** The five destinations, in ⌘1–⌘5 order. */
export const DESTINATIONS = [
  "New image",
  "Queue",
  "My images",
  "Styles",
  "Machines",
] as const;

/** Words that may never be a destination's primary label again. */
export const NEVER_A_DESTINATION = [
  "Create",
  "Library",
  "Models",
  "Hosts",
  "Gallery",
  "Catalog",
] as const;

/**
 * Retired words that may not appear in the TEMPLATE TEXT of a Styles or
 * Machines surface (attributes, identifiers and `data-test` hooks are not
 * text a person reads — strip them first with `templateText`).
 */
export const NEVER_SAID_ON_STYLES_AND_MACHINES: readonly RegExp[] = [
  /\bhost\b/i,
  /\bmodel page\b/i,
  /\bPull\b/,
  /\binstalled\b/i,
  /\bInstall\b/,
];

/** The inspector's words for the technical controls. */
export const CONTROL_WORDS = {
  steps: "Detail",
  stepsUnit: "passes",
  guidance: "Stick to my words",
  seed: "Repeat this look",
  seedRandom: "Surprise me",
  seedFixed: "Keep",
  strength: "How much to change it",
  loras: "Add-on looks",
  octree: "Surface detail",
  isoThreshold: "How tight to the photo",
  expand: "Write more for me",
  whereItRuns: "Where it runs",
  resetToStyleDefaults: "Reset to the style's defaults",
} as const;

/** Exact strings that were the pre-lexicon labels and must not come back. */
export const RETIRED_CONTROL_LABELS = [
  "Prompt strength",
  "LoRA stack",
  "Style adapters",
  "Expand prompt",
  "Generation host",
  "Octree detail",
  "Iso threshold",
  "Reset settings to model defaults",
] as const;

/** The Styles surface's two shelves and its one acquisition verb. */
export const STYLES_WORDS = {
  ready: "Ready to use",
  browse: "Browse more",
  get: "Get it",
  getting: "Getting it…",
  readyBadge: "● ready",
} as const;

/** Retired Styles words, never again as a tab label or a button. */
export const RETIRED_STYLES_LABELS = ["Installed", "Discover", "Pull"] as const;

/**
 * Template text only: strips the `<script>` block, every tag's attributes,
 * and every `{{ … }}` interpolation, so an identifier, a route path, or a
 * `data-test` hook can never trip a never-say scan of what a person reads.
 */
export function templateText(source: string): string {
  const template = source
    .replace(/<script[\s\S]*?<\/script>/g, "")
    .replace(/<style[\s\S]*?<\/style>/g, "");
  return (
    template
      // A tag ends at its first `>` OUTSIDE a quoted attribute, so a
      // `v-if="n > 0"` cannot leave half an attribute behind as "text".
      .replace(/<(?:[^>"']|"[^"]*"|'[^']*')*>/g, " ")
      .replace(/\{\{[\s\S]*?\}\}/g, " ")
      .replace(/\s+/g, " ")
  );
}
