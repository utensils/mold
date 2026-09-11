/*
 * Web's view of the shared Mold Studio style-preset kit (`ui/lib/stylePresets.ts`).
 * The composer's preset strip is retired on every GUI — the word Style belongs
 * to the style you render with — so nothing here lists presets any more. What
 * survives is the request-time composition a saved print or an in-flight
 * expansion still reads (`composeStyle`, `mergeStyleNegative`, `styleHint`).
 */
export {
  composeStyle,
  mergeStyleNegative,
  styleHint,
} from "@ui/lib/stylePresets";
export type { ComposeStyleOptions, ComposedStyle } from "@ui/lib/stylePresets";
