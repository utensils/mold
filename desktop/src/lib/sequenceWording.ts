import type { SequenceWording } from "@studio/lib/sequence";

/**
 * Desktop's words for the shared sequence validator: one word, "scene", for
 * every piece on this surface, and the whole thing IS the clip. Web and the
 * phone keep the validator's default `clip` / `sequence`.
 */
export const DESKTOP_SEQUENCE_WORDING: SequenceWording = { piece: "scene", whole: "clip" };
