/*
 * Web binding for "File under" — Create-time Library organization.
 *
 * Every rule (ghost tag, collection matching, normalization, request fields,
 * the capability gate, the download grammar) lives in
 * `@studio/lib/fileUnder`. This module owns only what is web-specific: the
 * browser-local "tag new prints with their title" preference and the
 * filename preview line the group renders.
 */
import { ref, watch } from "vue";
import { AUTO_TAG_SETTING_WEB, requestTagKey } from "@studio/lib/fileUnder";
import { titleSlug } from "@studio/lib/libraryOrganization";

// ── "Tag new prints with their title" (Settings ▸ Library) ──────────────────

/** Read the persisted preference. ON for a fresh browser, and ON for any
 * value that is not the literal `"false"` — the ghost chip is a visible,
 * removable default, so a corrupt entry must not silently disable it. */
export function loadAutoTagTitle(): boolean {
  try {
    return localStorage.getItem(AUTO_TAG_SETTING_WEB) !== "false";
  } catch {
    // Storage can be blocked (private mode, embedded webview) — default on.
    return true;
  }
}

export function saveAutoTagTitle(value: boolean): void {
  try {
    localStorage.setItem(AUTO_TAG_SETTING_WEB, value ? "true" : "false");
  } catch {
    // A preference that cannot be stored still applies to this session.
  }
}

/** Live preference, shared by Create's group and Settings ▸ Library. */
export const autoTagTitle = ref(loadAutoTagTitle());

watch(autoTagTitle, (value) => saveAutoTagTitle(value));

/** Re-read the preference from storage (tests, and a storage reset). */
export function reloadAutoTagTitle(): void {
  autoTagTitle.value = loadAutoTagTitle();
}

// ── Filename preview ────────────────────────────────────────────────────────

export interface FileUnderPreviewInput {
  /** Resolved model id, exactly as the request carries it. */
  model: string;
  title?: string | null;
  /** Output format, without a leading dot. */
  ext: string;
  /** Creation stamp. Frozen by the caller so the line does not tick. */
  timestamp: number;
}

/**
 * What the print will be called on disk:
 * `mold-{model}-{timestamp}~{title-slug}.{ext}`, mirroring
 * `mold_core::print_title::default_output_filename_titled` over
 * `default_output_filename`. The model keeps its own text with `:` swapped
 * for `-` (it is NOT slugged there), and an untitled or unsluggable title
 * drops the `~slug` segment so the name is byte-identical to the legacy one.
 *
 * Batch siblings add a `-{index}` before the separator server-side; the
 * preview shows the single-print shape because the group's choice is shared
 * by every sibling anyway.
 */
export function fileUnderPreviewName(input: FileUnderPreviewInput): string {
  const model = input.model.replace(/:/g, "-");
  const stem = `mold-${model}-${input.timestamp}`;
  const trimmed = input.title?.trim();
  const slug = trimmed ? titleSlug(trimmed) : null;
  const ext = input.ext.replace(/^\.+/, "");
  return slug ? `${stem}~${slug}.${ext}` : `${stem}.${ext}`;
}

// ── Reuse ───────────────────────────────────────────────────────────────────

/**
 * Whether a print's recorded tags include the one its title would have
 * derived. A filed print whose title tag is absent opted OUT of the ghost
 * chip, so restoring it must leave the chip retired rather than silently
 * re-adding a tag the user removed. Case-insensitive, like the server.
 */
export function titleTagWasApplied(
  title: string | null | undefined,
  tags: readonly string[],
): boolean {
  const trimmed = title?.trim();
  const slug = trimmed ? titleSlug(trimmed) : null;
  if (!slug) return false;
  const key = requestTagKey(slug);
  return tags.some((tag) => requestTagKey(tag) === key);
}
