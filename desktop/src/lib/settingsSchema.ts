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
  | "engine"
  | "performance"
  | "generation"
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
}

export const SECTIONS: SectionInfo[] = [
  { id: "engine", label: "Engine" },
  { id: "performance", label: "Performance" },
  { id: "generation", label: "Generation" },
  { id: "expansion", label: "Prompt expansion" },
  { id: "accounts", label: "Accounts & tokens" },
  { id: "app", label: "Appearance & app" },
  { id: "updates", label: "Updates" },
  { id: "profiles", label: "Profiles" },
  { id: "advanced", label: "Advanced" },
  { id: "about", label: "About" },
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
}

/** Engine-config keys (`/api/config`) with curated editors. */
export const ENGINE_KEY_SCHEMAS: KeySchema[] = [
  {
    key: "models_dir",
    section: "engine",
    label: "Models directory",
    help: "Where pulled model weights live. Moving it does not copy existing models.",
    editor: "path",
    needsEngineRestart: true,
  },
  {
    key: "output_dir",
    section: "engine",
    label: "Output directory",
    help: "Where finished prints are written. The Gallery reads from here.",
    editor: "path",
    needsEngineRestart: true,
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
    help: "⌘E rewrites terse prompts into detailed ones before generating.",
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
    help: "Keep FP16/BF16 text encoders on CPU between requests instead of reloading from disk. No effect on Metal (unified memory).",
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
    help: "math is the portable default; flash needs a CUDA build with flash-attn.",
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
