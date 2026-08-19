/**
 * The engine's missing-expansion-model error embeds its own fix:
 * "local expand model not found — run: mold pull qwen3-expand".
 *
 * The parser itself lives in `@studio` so web reads the same error the same
 * way; this module stays as the desktop import path.
 */
export { parseMissingExpandModel } from "@studio/lib/expansionRouting";
