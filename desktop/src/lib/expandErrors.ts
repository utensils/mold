/**
 * The engine's missing-expansion-model error embeds its own fix:
 * "local expand model not found — run: mold pull qwen3-expand".
 * Extract the model name so the UI can offer the pull directly.
 */
export function parseMissingExpandModel(message: string): string | null {
  const match = /mold pull ([\w.:@/-]+)/.exec(message);
  return match?.[1] ?? null;
}
