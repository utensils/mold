import {
  RETENTION_OPTIONS,
  retentionLabel,
} from "@studio/lib/libraryOrganization";

export type ConfigValue = string | number | boolean | null;
export type ConfigSource = "db" | "file" | "env" | "default";

export interface ConfigRow {
  key: string;
  value: ConfigValue;
  source: ConfigSource;
  profile?: string | null;
  env_var?: string | null;
  restart_required?: boolean;
}

export interface ConfigProfiles {
  profiles: string[];
  active: string;
}

async function checked(response: Response, action: string): Promise<Response> {
  if (response.ok) return response;
  const detail = await response.text().catch(() => "");
  throw new Error(
    `${action} failed (${response.status})${detail ? `: ${detail}` : ""}`,
  );
}

export async function fetchConfig(): Promise<ConfigRow[]> {
  const response = await checked(await fetch("/api/config"), "Load config");
  const body = (await response.json()) as
    ConfigRow[] | { entries?: ConfigRow[]; rows?: ConfigRow[] };
  return Array.isArray(body) ? body : (body.entries ?? body.rows ?? []);
}

export async function fetchProfiles(): Promise<ConfigProfiles> {
  const response = await checked(
    await fetch("/api/config/profiles"),
    "Load profiles",
  );
  const body = (await response.json()) as Partial<ConfigProfiles>;
  return { profiles: body.profiles ?? [], active: body.active ?? "default" };
}

export async function writeConfig(
  key: string,
  value: ConfigValue,
): Promise<void> {
  await checked(
    await fetch(`/api/config/${encodeURIComponent(key)}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value }),
    }),
    `Save ${key}`,
  );
}

export async function resetConfig(key: string): Promise<void> {
  await checked(
    await fetch(`/api/config/${encodeURIComponent(key)}`, { method: "DELETE" }),
    `Reset ${key}`,
  );
}

export async function switchProfile(name: string): Promise<void> {
  await checked(
    await fetch("/api/config/profile", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    }),
    `Switch to ${name}`,
  );
}

export type ConfigSection =
  | "Storage & server"
  | "Generation"
  | "Library"
  | "Prompt expansion"
  | "Advanced";
export type EditorKind = "toggle" | "number" | "select" | "text";

export interface ConfigSchema {
  key: string;
  section: ConfigSection;
  label: string;
  help: string;
  editor: EditorKind;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Display labels for `options` (by option value); falls back to the raw
   * value (and "auto" for the empty string). */
  optionLabels?: Record<string, string>;
  /** `select` editors submit strings; a numeric key coerces on save. */
  valueType?: "number";
  /** Visible over the live API but only editable before `mold serve` starts. */
  liveReadOnly?: boolean;
  needsEngineRestart?: boolean;
}

export const CONFIG_SECTIONS: ConfigSection[] = [
  "Storage & server",
  "Generation",
  "Library",
  "Prompt expansion",
  "Advanced",
];

/** Settings ▸ Library copy, shared verbatim with the desktop app. */
export const TRASH_RETENTION_HELP =
  "Prints moved to the trash are deleted forever after this long. Forever keeps them until you empty the trash.";

export const CONFIG_SCHEMAS: ConfigSchema[] = [
  {
    key: "models_dir",
    section: "Storage & server",
    label: "Models directory",
    help: "Where pulled model weights live on this host.",
    editor: "text",
  },
  {
    key: "output_dir",
    section: "Storage & server",
    label: "Output directory",
    help: "Where finished prints are written. Startup-only while mold serve is running; stop it, run `mold config set output_dir <path>`, then restart.",
    editor: "text",
    liveReadOnly: true,
  },
  {
    key: "server_port",
    section: "Storage & server",
    label: "Server port",
    help: "Port used by mold serve.",
    editor: "number",
    min: 1,
    max: 65535,
  },
  {
    key: "default_model",
    section: "Generation",
    label: "Default model",
    help: "Model preselected in a fresh composer.",
    editor: "text",
  },
  {
    key: "default_width",
    section: "Generation",
    label: "Default width",
    help: "Print width for new generations.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_height",
    section: "Generation",
    label: "Default height",
    help: "Print height for new generations.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_steps",
    section: "Generation",
    label: "Default steps",
    help: "Denoise steps for new generations.",
    editor: "number",
    min: 1,
    max: 150,
  },
  {
    key: "default_negative_prompt",
    section: "Generation",
    label: "Default negative prompt",
    help: "Used when the composer negative prompt is empty.",
    editor: "text",
  },
  {
    key: "embed_metadata",
    section: "Generation",
    label: "Embed metadata in prints",
    help: "Write prompt, seed, and parameters into PNG and JPEG files.",
    editor: "toggle",
  },
  {
    key: "t5_variant",
    section: "Generation",
    label: "T5 encoder variant",
    help: "Quantization for the FLUX T5 text encoder.",
    editor: "select",
    options: ["", "fp16", "q8_0", "q5_k_m", "q4_k_m"],
  },
  {
    key: "qwen3_variant",
    section: "Generation",
    label: "Qwen3 encoder variant",
    help: "Quantization for the Flux.2 and Z-Image Qwen3 encoder.",
    editor: "select",
    options: ["", "bf16", "q8_0", "q5_k_m", "q4_k_m"],
  },
  {
    key: "gallery.trash_retention_days",
    section: "Library",
    label: "Trash retention",
    help: TRASH_RETENTION_HELP,
    editor: "select",
    valueType: "number",
    options: RETENTION_OPTIONS.map(String),
    optionLabels: Object.fromEntries(
      RETENTION_OPTIONS.map((days) => [String(days), retentionLabel(days)]),
    ),
  },
  {
    key: "expand.enabled",
    section: "Prompt expansion",
    label: "Enable prompt expansion",
    help: "Allow terse prompts to be expanded before generating.",
    editor: "toggle",
  },
  {
    key: "expand.backend",
    section: "Prompt expansion",
    label: "Expansion backend",
    help: "Use local or an Ollama-compatible API URL.",
    editor: "text",
  },
  {
    key: "expand.model",
    section: "Prompt expansion",
    label: "Local expansion model",
    help: "Model used by the local backend.",
    editor: "text",
  },
  {
    key: "expand.api_model",
    section: "Prompt expansion",
    label: "API model",
    help: "Model name sent to the API backend.",
    editor: "text",
  },
  {
    key: "expand.temperature",
    section: "Prompt expansion",
    label: "Temperature",
    help: "Higher values produce more inventive expansions.",
    editor: "number",
    min: 0,
    max: 2,
    step: 0.05,
  },
  {
    key: "expand.top_p",
    section: "Prompt expansion",
    label: "Top-p",
    help: "Nucleus sampling cutoff for expansion.",
    editor: "number",
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: "expand.max_tokens",
    section: "Prompt expansion",
    label: "Max tokens",
    help: "Length budget for the expanded prompt.",
    editor: "number",
    min: 16,
    max: 4096,
  },
  {
    key: "expand.thinking",
    section: "Prompt expansion",
    label: "Thinking mode",
    help: "Let the expansion model reason before writing.",
    editor: "toggle",
  },
  {
    key: "scheduler.replan_debounce_ms",
    section: "Advanced",
    label: "Queue replan debounce",
    help: "Delay after the latest queue change before globally optimizing the plan.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
  {
    key: "scheduler.replan_max_delay_ms",
    section: "Advanced",
    label: "Maximum replan delay",
    help: "Maximum delay from the first unplanned queue change.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
  {
    key: "scheduler.warm_wait_max_ms",
    section: "Advanced",
    label: "Maximum warm-model wait",
    help: "Longest beneficial wait for a compatible warm model.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
];

const SCHEMA_BY_KEY = new Map(
  CONFIG_SCHEMAS.map((schema) => [schema.key, schema]),
);

export function schemaForRow(row: ConfigRow): ConfigSchema {
  return (
    SCHEMA_BY_KEY.get(row.key) ?? {
      key: row.key,
      section: "Advanced",
      label: row.key,
      help: "Server-provided configuration key.",
      editor:
        typeof row.value === "boolean"
          ? "toggle"
          : typeof row.value === "number"
            ? "number"
            : "text",
    }
  );
}

export function matchesConfigSearch(
  query: string,
  schema: ConfigSchema,
): boolean {
  const needle = query.trim().toLowerCase();
  return (
    !needle ||
    [schema.key, schema.label, schema.help].some((value) =>
      value.toLowerCase().includes(needle),
    )
  );
}

/** DELETE resets only DB-backed preference keys; bootstrap keys live in config.toml. */
export function canResetConfig(key: string): boolean {
  if (
    key.startsWith("expand.") ||
    key.startsWith("generate.") ||
    key.startsWith("gallery.") ||
    key.startsWith("scheduler.") ||
    key.startsWith("model_prefs.")
  ) {
    return true;
  }
  if (key.startsWith("models.")) {
    const field = key.slice(key.lastIndexOf(".") + 1);
    return [
      "default_steps",
      "default_guidance",
      "default_width",
      "default_height",
      "scheduler",
      "negative_prompt",
      "lora",
      "lora_scale",
    ].includes(field);
  }
  return [
    "default_width",
    "default_height",
    "default_steps",
    "embed_metadata",
    "default_negative_prompt",
    "t5_variant",
    "qwen3_variant",
  ].includes(key);
}
