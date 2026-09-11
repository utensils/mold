/*
 * The Draft / Good / Best ladder is one rule, and web's Create rail now reads
 * it too, so it lives in `studio/lib/qualityPresets.ts`. This file stays as
 * the re-export its two importers (the inspector's Quality card and the
 * phone's shared params) already point at.
 */
export * from "@studio/lib/qualityPresets";
