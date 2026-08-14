import type { IconName } from "@ui/icons";

/**
 * The Darkroom Bench: curated metadata for the Settings surface.
 *
 * Known engine-config keys (`/api/config`) map to a section and a purposeful
 * editor; anything unknown falls through to Advanced as a raw provenance row,
 * so future engine keys surface without a desktop release. App-side prefs
 * (settings.json) and env knobs declare themselves here too so search spans
 * every section.
 */

export type SectionId =
  | "hosts"
  | "performance"
  | "generation"
  | "media"
  | "expansion"
  | "accounts"
  | "app"
  | "updates"
  | "profiles"
  | "advanced"
  | "about";

export interface SectionInfo {
  id: SectionId;
  label: string;
  /** Icon-led header treatment shared by desktop and web Settings groups. */
  icon?: IconName;
  /** One-line explanation shown beneath the group title. */
  summary?: string;
  /** Extra search terms for sections whose controls carry no curated key
   *  (Accounts tokens, Profiles) so the global search still finds them. */
  keywords?: string[];
}

export interface AccordionSectionInfo extends SectionInfo {
  icon: IconName;
  summary: string;
}

export const SECTIONS: SectionInfo[] = [
  { id: "hosts", label: "Hosts" },
  { id: "performance", label: "Performance" },
  { id: "generation", label: "Generation" },
  { id: "media", label: "Saved media" },
  { id: "expansion", label: "Prompt expansion" },
  { id: "accounts", label: "Accounts & tokens" },
  { id: "app", label: "Appearance & app" },
  { id: "updates", label: "Updates" },
  { id: "profiles", label: "Profiles" },
  { id: "advanced", label: "Advanced" },
  { id: "about", label: "About" },
];

/**
 * The Settings surface is now a single column: Appearance, Updates, and About
 * ride at the top as always-open cards, Hosts is a link to the Machines
 * workspace, and everything deeper collapses into the "All settings" region as
 * one-open-at-a-time accordions. This list is that region, in order — none of
 * it blocks first use (spec G7).
 */
export const ACCORDION_SECTIONS: AccordionSectionInfo[] = [
  {
    id: "performance",
    label: "Performance",
    icon: "scheduler",
    summary: "Memory, previews, queueing, and compute backends",
  },
  {
    id: "generation",
    label: "Generation",
    icon: "image",
    summary: "Defaults for new image and video jobs",
  },
  {
    id: "media",
    label: "Saved media",
    icon: "save",
    summary: "Where photos, videos, and converted exports are saved",
    keywords: [
      "save",
      "save location",
      "default save location",
      "export",
      "download",
      "folder",
      "location",
      "photo",
      "video",
    ],
  },
  {
    id: "expansion",
    label: "Prompt expansion",
    icon: "sparkle",
    summary: "Rewrite behavior, model, and sampling controls",
  },
  {
    id: "accounts",
    label: "Accounts & tokens",
    icon: "lock",
    summary: "Credentials used for catalogs and model downloads",
    keywords: ["hugging face", "huggingface", "hf", "civitai", "token", "api key"],
  },
  {
    id: "profiles",
    label: "Profiles",
    icon: "history",
    summary: "Keep separate sets of engine preferences",
    keywords: ["profile"],
  },
  {
    id: "advanced",
    label: "Advanced",
    icon: "settings",
    summary: "Uncommon and newly discovered engine options",
  },
];

export type EditorKind = "toggle" | "select" | "number" | "text" | "slider" | "path" | "secret";

export interface KeySchema {
  /** Config key (engine keys verbatim; app prefs use `app.*`; env knobs `env.*`). */
  key: string;
  section: SectionId;
  label: string;
  help: string;
  editor: EditorKind;
  /** For select editors. */
  options?: { value: string; label: string }[];
  /** For sliders/numbers. */
  min?: number;
  max?: number;
  step?: number;
  /** Changing this requires an engine restart to take effect. */
  needsEngineRestart?: boolean;
  /** The running server rejects mutation; edit through the CLI while stopped. */
  liveReadOnly?: boolean;
}

/** Engine-config keys (`/api/config`) with curated editors. */
export const ENGINE_KEY_SCHEMAS: KeySchema[] = [
  {
    key: "models_dir",
    section: "hosts",
    label: "Models directory",
    help: "Where pulled model weights live on this host. Moving it does not copy existing models.",
    editor: "path",
    needsEngineRestart: true,
  },
  {
    key: "output_dir",
    section: "hosts",
    label: "Output directory",
    help: "Where finished prints are written. Startup-only: stop the engine, run `mold config set output_dir <path>`, then restart.",
    editor: "path",
    needsEngineRestart: true,
    liveReadOnly: true,
  },
  {
    key: "server_port",
    section: "advanced",
    label: "Server port",
    help: "Port for `mold serve`. The built-in engine always uses an ephemeral port.",
    editor: "number",
    min: 1,
    max: 65535,
  },
  {
    key: "default_model",
    section: "generation",
    label: "Default model",
    help: "Model preselected in a fresh composer.",
    editor: "select",
  },
  {
    key: "default_width",
    section: "generation",
    label: "Default width",
    help: "Print width for new generations.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_height",
    section: "generation",
    label: "Default height",
    help: "Print height for new generations.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_steps",
    section: "generation",
    label: "Default steps",
    help: "Denoise steps for new generations (models override with their own defaults).",
    editor: "number",
    min: 1,
    max: 150,
  },
  {
    key: "default_negative_prompt",
    section: "generation",
    label: "Default negative prompt",
    help: "Applied when the composer's negative prompt is empty (families that support it).",
    editor: "text",
  },
  {
    key: "embed_metadata",
    section: "generation",
    label: "Embed metadata in prints",
    help: "Write prompt, seed, and parameters into PNG/JPEG files so any print can be reproduced.",
    editor: "toggle",
  },
  {
    key: "t5_variant",
    section: "generation",
    label: "T5 encoder variant",
    help: "Quantization for the FLUX T5 text encoder. Smaller variants trade fidelity for VRAM.",
    editor: "select",
    options: [
      { value: "", label: "auto" },
      { value: "fp16", label: "fp16" },
      { value: "q8_0", label: "q8_0" },
      { value: "q5_k_m", label: "q5_k_m" },
      { value: "q4_k_m", label: "q4_k_m" },
    ],
  },
  {
    key: "qwen3_variant",
    section: "generation",
    label: "Qwen3 encoder variant",
    help: "Quantization for the Flux.2/Z-Image Qwen3 text encoder.",
    editor: "select",
    options: [
      { value: "", label: "auto" },
      { value: "bf16", label: "bf16" },
      { value: "q8_0", label: "q8_0" },
      { value: "q5_k_m", label: "q5_k_m" },
      { value: "q4_k_m", label: "q4_k_m" },
    ],
  },
  {
    key: "expand.enabled",
    section: "expansion",
    label: "Enable prompt expansion",
    help: "The primary shortcut plus E rewrites terse prompts before generating.",
    editor: "toggle",
  },
  {
    key: "expand.backend",
    section: "expansion",
    label: "Backend",
    help: "`local` runs the expansion model on this engine; a URL points at an Ollama-compatible API.",
    editor: "text",
  },
  {
    key: "expand.model",
    section: "expansion",
    label: "Local expansion model",
    help: "Model used by the local backend (pull it from Models if missing).",
    editor: "text",
  },
  {
    key: "expand.api_model",
    section: "expansion",
    label: "API model",
    help: "Model name sent to the API backend (e.g. qwen2.5:3b for Ollama).",
    editor: "text",
  },
  {
    key: "expand.temperature",
    section: "expansion",
    label: "Temperature",
    help: "Higher = more inventive expansions.",
    editor: "slider",
    min: 0,
    max: 2,
    step: 0.05,
  },
  {
    key: "expand.top_p",
    section: "expansion",
    label: "Top-p",
    help: "Nucleus sampling cutoff for the expansion model.",
    editor: "slider",
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: "expand.max_tokens",
    section: "expansion",
    label: "Max tokens",
    help: "Length budget for the expanded prompt.",
    editor: "number",
    min: 16,
    max: 4096,
  },
  {
    key: "expand.thinking",
    section: "expansion",
    label: "Thinking mode",
    help: "Let the expansion model reason before writing (slower, sometimes better).",
    editor: "toggle",
  },
  {
    key: "scheduler.replan_debounce_ms",
    section: "performance",
    label: "Queue replan debounce",
    help: "Delay after the latest queue change before globally optimizing the plan.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
  {
    key: "scheduler.replan_max_delay_ms",
    section: "performance",
    label: "Maximum replan delay",
    help: "Maximum delay from the first unplanned queue change.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
  {
    key: "scheduler.warm_wait_max_ms",
    section: "performance",
    label: "Maximum warm-model wait",
    help: "Longest beneficial wait for a compatible warm model.",
    editor: "number",
    min: 0,
    max: 30000,
    needsEngineRestart: true,
  },
];

/**
 * Embedded-engine environment knobs (settings.json `engineEnv`, applied when
 * the engine starts). Hidden when connected to a remote host.
 */
export const ENV_KNOB_SCHEMAS: KeySchema[] = [
  {
    key: "env.MOLD_STEP_PREVIEW",
    section: "performance",
    label: "Live denoise previews",
    help: "Stream a low-fi preview of the forming image after each step. Costs ~ms per step.",
    editor: "select",
    options: [
      { value: "", label: "On (default)" },
      { value: "0", label: "Off" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_KEEP_TE_RAM",
    section: "performance",
    label: "Park text encoders in RAM",
    help: "Keep text encoders on CPU between requests instead of reloading from disk — FP16/BF16 everywhere, plus Qwen-Image's quantized GGUF encoder. Costs several GB of host RAM per parked encoder. No effect on Metal (unified memory).",
    editor: "select",
    options: [
      { value: "", label: "Off (default)" },
      { value: "1", label: "On" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_VAE_TILED",
    section: "performance",
    label: "Tiled VAE decode",
    help: "auto retries with tiling on out-of-memory; force always tiles (slower, tiny VRAM).",
    editor: "select",
    options: [
      { value: "", label: "auto (default)" },
      { value: "force", label: "force" },
      { value: "off", label: "off" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_OFFLOAD",
    section: "performance",
    label: "Block-level offloading",
    help: "Stream FLUX transformer blocks CPU↔GPU one at a time: ~24 GB → 2–4 GB VRAM, 3–5× slower. Auto-enables under pressure.",
    editor: "select",
    options: [
      { value: "", label: "auto (default)" },
      { value: "1", label: "force on" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_ATTN",
    section: "performance",
    label: "Attention backend",
    help: "math is the portable default. flash needs a CUDA build with the flash-attn feature, which the built-in engine is not — it falls back to math here.",
    editor: "select",
    options: [
      { value: "", label: "math (default)" },
      { value: "flash", label: "flash" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_QUEUE_SIZE",
    section: "performance",
    label: "Queue capacity",
    help: "How many generations may wait in line (default 200).",
    editor: "number",
    min: 1,
    max: 10000,
    needsEngineRestart: true,
  },
];

const BY_KEY = new Map([...ENGINE_KEY_SCHEMAS, ...ENV_KNOB_SCHEMAS].map((s) => [s.key, s]));

export function schemaFor(key: string): KeySchema | null {
  return BY_KEY.get(key) ?? null;
}

/**
 * Section for an engine-config key: curated keys go to their section,
 * `tui.*` stays out of a desktop app, everything else lands in Advanced.
 */
export function sectionForConfigKey(key: string): SectionId | null {
  if (key.startsWith("tui.")) return null;
  return schemaFor(key)?.section ?? "advanced";
}

export interface Searchable {
  key: string;
  label: string;
  help?: string;
}

/** Case-insensitive filter across key, label, and help text. */
export function matchesSearch(query: string, item: Searchable): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    item.key.toLowerCase().includes(q) ||
    item.label.toLowerCase().includes(q) ||
    (item.help ?? "").toLowerCase().includes(q)
  );
}

/** Curated schemas that live in a given accordion section. */
export function schemasForSection(sectionId: SectionId): KeySchema[] {
  return [...ENGINE_KEY_SCHEMAS, ...ENV_KNOB_SCHEMAS].filter((s) => s.section === sectionId);
}

/**
 * Whether a settings accordion has anything matching the query — its label,
 * its declared keywords, any curated key it owns, or (Advanced only) a raw
 * engine row. Drives which accordion the global search opens.
 */
export function sectionMatchesSearch(
  query: string,
  section: SectionInfo,
  advancedRowKeys: string[] = [],
): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (section.label.toLowerCase().includes(q)) return true;
  if (section.keywords?.some((keyword) => keyword.includes(q))) return true;
  if (schemasForSection(section.id).some((schema) => matchesSearch(query, schema))) return true;
  if (section.id === "advanced" && advancedRowKeys.some((key) => key.toLowerCase().includes(q)))
    return true;
  return false;
}
