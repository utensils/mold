import { RETENTION_OPTIONS, retentionLabel } from "@studio/lib/libraryOrganization";

/**
 * Settings (README §02 lexicon): curated metadata for the Settings surface.
 *
 * Known engine-config keys (`/api/config`) map to a section and a purposeful
 * editor; anything unknown falls through to Advanced as a raw provenance row,
 * so future engine keys surface without a desktop release. App-side prefs
 * (settings.json) and env knobs declare themselves here too so search spans
 * every section.
 */

export type SectionId =
  | "app"
  | "generation"
  | "expansion"
  | "hosts"
  | "styles"
  | "media"
  | "library"
  | "licenses"
  | "pairing"
  | "performance"
  | "accounts"
  | "profiles"
  | "advanced"
  | "updates";

export interface SectionInfo {
  id: SectionId;
  label: string;
  /** One plain sentence beside the section label. */
  summary: string;
  /** Extra search terms for sections whose controls carry no curated key
   *  (Accounts tokens, Profiles, Look) so the global search still finds them. */
  keywords?: string[];
}

/**
 * The Settings surface is one scrolling page behind a 200px jump nav: every
 * section is always open, in this order, and search narrows both the nav and
 * the page to the sections that match. Nothing here blocks first use (G7);
 * `?section=` deep links (the Library trash banner, the native Check for
 * Updates action) jump to a section by id.
 */
export const SECTIONS: SectionInfo[] = [
  {
    id: "app",
    label: "Look",
    summary: "Theme, interface size, and how the app behaves",
    keywords: ["theme", "appearance", "dark", "light", "match system", "scale", "notifications"],
  },
  {
    id: "generation",
    label: "Defaults for new images",
    summary: "What a fresh New image starts with",
  },
  {
    id: "expansion",
    label: "Write more for me",
    summary: "How the prompt rewriter works and which model it uses",
    keywords: ["expand", "expansion", "rewrite", "prompt"],
  },
  {
    id: "hosts",
    label: "Machines",
    summary: "This device, its key, and the Mold home it works out of",
    keywords: ["hosts", "this device", "engine", "api key", "mold home"],
  },
  {
    id: "styles",
    label: "Styles & disk",
    summary: "Where styles and finished pictures land, and how much disk they take",
    keywords: [
      "models",
      "weights",
      "checkpoints",
      "disk",
      "storage",
      "space",
      "directory",
      "folder",
    ],
  },
  {
    id: "licenses",
    label: "Style licences",
    summary: "Some styles need you to accept their terms once per machine",
    keywords: ["licence", "license", "terms", "accept"],
  },
  {
    id: "library",
    label: "My images & trash",
    summary: "Trash retention on this device",
    keywords: ["trash", "retention", "collections", "albums", "deleted prints", "purge", "library"],
  },
  {
    id: "media",
    label: "Saving pictures & clips",
    summary: "Where saved pictures, clips, and exports go",
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
    id: "pairing",
    label: "Phone pairing",
    summary: "Use mold on your phone",
    keywords: ["phone", "iphone", "pair", "pairing", "qr", "mobile"],
  },
  {
    id: "performance",
    label: "Speed & memory",
    summary: "Memory, previews, queueing, and compute backends",
    keywords: ["performance", "vram", "offload", "preview"],
  },
  {
    id: "accounts",
    label: "Accounts & tokens",
    summary: "Credentials used for catalogs and style downloads",
    keywords: ["hugging face", "huggingface", "hf", "civitai", "token", "api key"],
  },
  {
    id: "profiles",
    label: "Profiles",
    summary: "Keep separate sets of engine preferences",
    keywords: ["profile"],
  },
  {
    id: "advanced",
    label: "Advanced",
    summary: "Uncommon and newly discovered engine options",
  },
  {
    id: "updates",
    label: "Updates & about",
    summary: "Version, update channel, logs, and diagnostics",
    keywords: ["update", "version", "about", "nightly", "stable", "logs", "diagnostics", "privacy"],
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
    section: "styles",
    label: "Where styles are kept",
    help: "Downloaded style weights live here on this machine. Moving it does not copy the styles already there.",
    editor: "path",
    needsEngineRestart: true,
  },
  {
    key: "output_dir",
    section: "styles",
    label: "Where finished pictures are written",
    help: "Startup-only: stop the engine, run `mold config set output_dir <path>`, then restart.",
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
    label: "Style to start with",
    help: "Used whenever you open a new image.",
    editor: "select",
  },
  {
    key: "default_width",
    section: "generation",
    label: "Width",
    help: "How wide a new picture starts. Bigger uses more graphics memory.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_height",
    section: "generation",
    label: "Height",
    help: "How tall a new picture starts. Bigger uses more graphics memory.",
    editor: "number",
    min: 64,
    max: 4096,
    step: 8,
  },
  {
    key: "default_steps",
    section: "generation",
    label: "Detail",
    help: "How many passes a new picture starts with. A style with its own default wins.",
    editor: "number",
    min: 1,
    max: 150,
  },
  {
    key: "default_negative_prompt",
    section: "generation",
    label: "Words to avoid",
    help: "Used when you have not typed any yourself, on styles that read them.",
    editor: "text",
  },
  {
    key: "embed_metadata",
    section: "generation",
    label: "Keep the recipe in the file",
    help: "Writes the words, the seed, and every setting into the PNG or JPEG so the same picture can be made again.",
    editor: "toggle",
  },
  {
    key: "t5_variant",
    section: "generation",
    label: "How FLUX reads your words",
    help: "The size of its T5 text encoder. Smaller trades a little fidelity for graphics memory.",
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
    label: "How Flux.2 and Z-Image read your words",
    help: "The size of their Qwen3 text encoder. Smaller trades a little fidelity for graphics memory.",
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
    key: "gallery.trash_retention_days",
    section: "library",
    label: "Keep deleted pictures for",
    help: "Pictures in the trash are deleted forever after this long. 0 keeps them until you empty the trash.",
    editor: "select",
    options: RETENTION_OPTIONS.map((days) => ({
      value: String(days),
      label: retentionLabel(days),
    })),
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
    help: "Model used by the local backend (get it from Styles if missing).",
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
    label: "Live previews while a picture is made",
    help: "Stream a rough preview of the forming picture after each pass. Costs ~ms per pass.",
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
    help: "Keep text encoders in host RAM between requests instead of re-reading them from disk — BF16/FP16 and quantized GGUF encoders alike. Automatic covers Flux.2 and Z-Image, measuring this machine and parking only when the encoder, the transformer that loads beside it, and a 15%-of-RAM (minimum 8 GB) safety floor all fit, so a 64 GB desktop keeps streaming. Always is the opt-in every other style reads — FLUX and SD3's T5, Wan's encoder, Qwen-Image's — and parks whenever the encoder alone clears that floor. Costs several GB of host RAM per parked encoder. No effect on Metal (unified memory).",
    editor: "select",
    options: [
      { value: "", label: "Automatic (default)" },
      { value: "1", label: "Always, when it fits" },
      { value: "0", label: "Never" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_ATTN",
    section: "performance",
    label: "Attention backend",
    help: "Automatic runs FlashAttention-2 for video (Wan, LTX-2) and for FLUX.1 and Flux.2 wherever the build compiled the kernel, and the byte-stable math path for every other image family. Choosing one applies it to every family. Flash needs a CUDA build with flash-attn and a half-precision tensor; anything else falls back to math.",
    editor: "select",
    options: [
      { value: "", label: "Automatic, per family (default)" },
      { value: "flash", label: "FlashAttention-2 everywhere" },
      { value: "math", label: "Math everywhere (byte-stable)" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_CONV",
    section: "performance",
    label: "Convolution backend",
    help: "Automatic runs cuDNN for video (Wan, LTX-2) and for FLUX.1 and Flux.2 VAE encode/decode wherever the build compiled it, and im2col for every other image family. Choosing one applies it to every family. The two sum in a different order, so they do not agree bit-for-bit.",
    editor: "select",
    options: [
      { value: "", label: "Automatic, per family (default)" },
      { value: "cudnn", label: "cuDNN everywhere" },
      { value: "im2col", label: "im2col everywhere (byte-stable)" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_FLUX_KEEP_TRANSFORMER",
    section: "performance",
    label: "Keep the FLUX transformer on the card",
    help: "Automatic measures the card before each VAE decode — the resident checkpoint, this render's denoise workspace, the decode workspace and an allocator margin — and keeps the FLUX.1 / Flux.2 transformer resident when all four fit, saving a full reload on the next render. Always keep means the same as automatic: an explicit keep still yields to a card that cannot afford it. Always drop frees the VRAM and reloads every render.",
    editor: "select",
    options: [
      { value: "", label: "Automatic, budgeted (default)" },
      { value: "1", label: "Keep when it fits (same as automatic)" },
      { value: "0", label: "Always drop before VAE decode" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_FLUX2_QMATMUL",
    section: "performance",
    label: "Flux.2 quantized fast path",
    help: "Flux.2 GGUF tiers run their linears through candle's quantized CUDA kernels by default — the same algorithm FLUX.1 has always rendered correctly through. Turn it off to dequantize each weight per forward instead; slower, and only worth trying if a Flux.2 render comes out black or blank.",
    editor: "select",
    options: [
      { value: "", label: "On (default)" },
      { value: "0", label: "Off — dequantize per forward" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "env.MOLD_FLUX2_FP8_CACHE",
    section: "performance",
    label: "Flux.2 FP8 weight widening",
    help: "Automatic widens a Flux.2 FP8 checkpoint to the working precision once at load when the card has room for it, instead of casting every weight on every forward — two bytes per parameter at rest for a much faster step. Same arithmetic either way, so the picture does not change. Force either arm if you need the VRAM back or the card's free memory cannot be read.",
    editor: "select",
    options: [
      { value: "", label: "Automatic, budgeted (default)" },
      { value: "1", label: "Widen once at load" },
      { value: "0", label: "Widen on every forward" },
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
    key: "env.MOLD_PNG_ENCODING",
    section: "performance",
    label: "PNG encoding",
    help: "How much CPU a saved PNG is worth. fast uses fdeflate's PNG-tuned ultra-fast deflate; balanced is zlib level 6. PNG is lossless either way — this only trades encode time against file size, never a pixel.",
    editor: "select",
    options: [
      { value: "", label: "fast (default)" },
      { value: "balanced", label: "balanced (smaller files)" },
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
    key: "env.MOLD_QUEUE_SIZE",
    section: "performance",
    label: "Runtime queue window",
    help: "How many jobs may be hydrated for dispatch at once (default 200). The durable backlog remains uncapped.",
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
  if (section.summary.toLowerCase().includes(q)) return true;
  if (section.keywords?.some((keyword) => keyword.includes(q))) return true;
  if (schemasForSection(section.id).some((schema) => matchesSearch(query, schema))) return true;
  if (section.id === "advanced" && advancedRowKeys.some((key) => key.toLowerCase().includes(q)))
    return true;
  return false;
}
