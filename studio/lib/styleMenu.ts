/**
 * The least a row needs to be drawn in the shared style menu.
 *
 * Structural on purpose: desktop's `ModelEntry`, web's `ModelInfoExtended` and
 * the phone's own row type all satisfy it, and `studio/` may not import any of
 * them (`scripts/tests/frontend-architecture.sh` keeps studio shell-independent).
 * Every field beyond the id and the family is optional because an older server,
 * a manifest row and a catalog install each omit a different one.
 */
export interface StyleMenuModel {
  /** The runnable id — the wire value, shown in mono beside the plain name. */
  name: string;
  family: string;
  description?: string | null;
  display_name?: string | null;
  disk_usage_bytes?: number | null;
  is_loaded?: boolean | null;
  hf_repo?: string | null;
}
