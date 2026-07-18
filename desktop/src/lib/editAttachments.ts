/**
 * Pure helpers for the Qwen-Image-Edit Target + Reference picture strip.
 * Ported from the web SPA's Composer (`reorderAttachment`, `moveAttachment`,
 * `roleLabel`, `titleLabel`) so both surfaces agree on ordering semantics:
 * index 0 is the primary edit Target, everything after is a Reference.
 *
 * All list helpers are non-mutating; a no-op (same index, out of range)
 * returns the INPUT array reference so callers can cheaply skip an update.
 */

export function reorderAttachment<T>(list: readonly T[], fromIndex: number, toIndex: number): T[] {
  if (fromIndex === toIndex) return list as T[];
  if (fromIndex < 0 || fromIndex >= list.length) return list as T[];
  if (toIndex < 0 || toIndex >= list.length) return list as T[];
  const next = list.slice();
  const [item] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, item as T);
  return next;
}

export function moveAttachment<T>(list: readonly T[], index: number, delta: -1 | 1): T[] {
  return reorderAttachment(list, index, index + delta);
}

export function attachmentRoleLabel(index: number): "Target" | "Reference" {
  return index === 0 ? "Target" : "Reference";
}

export function attachmentTitleLabel(index: number): string {
  return `Picture ${index + 1}`;
}
