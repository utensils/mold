export type ConfigValue = string | number | boolean | null;
export type ConfigSource = "db" | "file" | "env" | "default";

export interface ConfigRow {
  key: string;
  value: ConfigValue;
  source: ConfigSource;
  profile?: string | null;
  env_var?: string | null;
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
  "Storage & server" | "Generation" | "Prompt expansion" | "Advanced";
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
}

export const CONFIG_SECTIONS: ConfigSection[] = [
  "Storage & server",
  "Generation",
  "Prompt expansion",
  "Advanced",
];

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
    help: "Where finished prints are written on this host.",
    editor: "text",
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
  return !["models_dir", "output_dir", "server_port"].includes(key);
}
