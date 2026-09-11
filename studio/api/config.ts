import { apiFetchTo, apiJsonTo, type ApiTarget } from "./client";

/**
 * Engine configuration (`/api/config`), addressed by explicit target the way
 * `api/devices.ts` is: the machine a settings page edits is the machine the
 * page is about, and a client that reads its own ambient connection cannot
 * say so.
 *
 * Every reader normalizes defensively. Three response shapes have shipped on
 * the listing endpoint, and a client that understood only the newest would
 * show an empty Settings page against an older machine rather than an error
 * anyone could act on.
 */

export type ConfigValue = string | number | boolean | null;
export type ConfigSource = "db" | "file" | "env" | "default";

/** One row from `GET /api/config`. */
export interface ConfigRow {
  key: string;
  value: ConfigValue;
  source: ConfigSource;
  profile?: string | null;
  /** Name of the environment variable that wins when source is "env". */
  env_var?: string | null;
  /** The persisted value applies when the engine restarts. */
  restart_required?: boolean;
}

/** `GET /api/config/profiles`. */
export interface ConfigProfiles {
  profiles: string[];
  active: string;
}

export async function listConfig(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<ConfigRow[]> {
  const body = await apiJsonTo<
    ConfigRow[] | { entries?: ConfigRow[]; rows?: ConfigRow[] }
  >(target, "/api/config", { signal: signal ?? null });
  if (Array.isArray(body)) return body;
  // The shipped wire shape is `{ profile, entries }` (mold_core::ConfigListing);
  // `rows` is kept for the pre-release drafts.
  return body.entries ?? body.rows ?? [];
}

export async function setConfig(
  target: ApiTarget,
  key: string,
  value: ConfigValue,
): Promise<void> {
  await apiFetchTo(target, `/api/config/${encodeURIComponent(key)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

export async function resetConfig(
  target: ApiTarget,
  key: string,
): Promise<void> {
  await apiFetchTo(target, `/api/config/${encodeURIComponent(key)}`, {
    method: "DELETE",
  });
}

export async function listProfiles(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<ConfigProfiles> {
  const body = await apiJsonTo<Partial<ConfigProfiles>>(
    target,
    "/api/config/profiles",
    { signal: signal ?? null },
  );
  return { profiles: body.profiles ?? [], active: body.active ?? "default" };
}

/** Switching to an unknown name creates the profile (server semantics). */
export async function switchProfile(
  target: ApiTarget,
  name: string,
): Promise<void> {
  await apiFetchTo(target, "/api/config/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export interface ProvenanceTag {
  glyph: string;
  label: string;
}

/** Where the value in hand came from. */
export function provenance(source: ConfigSource): ProvenanceTag {
  switch (source) {
    case "db":
      return { glyph: "⌂", label: "db" };
    case "file":
      return { glyph: "⛁", label: "file" };
    case "env":
      return { glyph: "⚿", label: "env" };
    default:
      return { glyph: "·", label: "default" };
  }
}

/** Env-resolved rows are locked — the environment wins over any stored value. */
export function isRowLocked(row: ConfigRow): boolean {
  return row.source === "env";
}

/** The per-style fields kept in `model_prefs`; the rest are component paths
 *  that stay in config.toml. Mirrors `config_keys::model_field_surface`. */
const PER_STYLE_DB_FIELDS = [
  "default_steps",
  "default_guidance",
  "default_width",
  "default_height",
  "scheduler",
  "negative_prompt",
  "lora",
  "lora_scale",
];

/** Flat generation defaults that persist in the DB under their old CLI names.
 *  Mirrors the `matches!` arm in `config_keys::surface_for_key`. */
const DB_FLAT_KEYS = [
  "default_width",
  "default_height",
  "default_steps",
  "embed_metadata",
  "default_negative_prompt",
  "t5_variant",
  "qwen3_variant",
];

const DB_PREFIXES = [
  "tui.",
  "expand.",
  "generate.",
  "scheduler.",
  "gallery.",
  "queue.",
  "model_prefs.",
];

/**
 * Whether `DELETE /api/config/:key` will reset this key.
 *
 * The mirror of `mold_core::config_keys::effective_surface`: only DB-surface
 * keys reset, because a bootstrap key lives in config.toml and is edited with
 * PUT. Offering ↺ on a file-backed row is offering a button the host refuses.
 */
export function canResetConfig(key: string): boolean {
  if (DB_PREFIXES.some((prefix) => key.startsWith(prefix))) return true;
  if (key.startsWith("models.")) {
    // A style name carries dots, so the field is what follows the LAST dot.
    const field = key.slice(key.lastIndexOf(".") + 1);
    return PER_STYLE_DB_FIELDS.includes(field);
  }
  return DB_FLAT_KEYS.includes(key);
}
