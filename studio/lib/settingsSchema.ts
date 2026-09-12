import { RETENTION_OPTIONS, retentionLabel } from "./libraryOrganization";

/**
 * Settings (README §02 lexicon): curated metadata for the Settings surface,
 * shared by every shell — the desktop app, the web SPA, and the phone.
 *
 * Known engine-config keys (`/api/config`) map to a section and a purposeful
 * editor; anything unknown falls through to Advanced as a raw provenance row,
 * so a key newer than the client surfaces without a release. App-side prefs
 * (settings.json) and env knobs declare themselves here too so search spans
 * every section.
 *
 * `settingsSchema.contract.test.ts` reads `crates/mold-core/src/config_keys.rs`
 * and fails when a registered engine key has no schema here. That is what makes
 * a raw Advanced row MEAN something: it can only ever be a key this build has
 * never heard of, never one nobody got round to curating.
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
  | "cloud"
  | "styleDefaults"
  | "profiles"
  | "advanced"
  | "updates";

/** A shell that renders Settings. The phone reads the desktop list. */
export type SettingsSurface = "web" | "desktop";

export interface SectionInfo {
  id: SectionId;
  label: string;
  /** One plain sentence beside the section label. */
  summary: string;
  /** Extra search terms for sections whose controls carry no curated key
   *  (Accounts tokens, Profiles, Look) so the global search still finds them. */
  keywords?: string[];
  /** Which shells render this section at all. */
  surfaces: readonly SettingsSurface[];
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
    summary: "Theme, light or dark, and how the app behaves",
    keywords: [
      "theme",
      "appearance",
      "dark",
      "light",
      "match system",
      "scale",
      "interface size",
      "notifications",
    ],
    surfaces: ["web", "desktop"],
  },
  {
    id: "generation",
    label: "Defaults for new images",
    summary: "What a fresh New image starts with",
    surfaces: ["web", "desktop"],
  },
  {
    id: "expansion",
    label: "Write more for me",
    summary: "How the prompt rewriter works and which model it uses",
    keywords: ["expand", "expansion", "rewrite", "prompt"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "hosts",
    label: "Machines",
    summary: "This machine, what it runs on, and the others you can reach",
    keywords: ["hosts", "this device", "engine", "api key", "mold home"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "styles",
    label: "Styles & disk",
    summary:
      "Where styles and finished pictures land, and how much disk they take",
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
    surfaces: ["web", "desktop"],
  },
  {
    id: "licenses",
    label: "Style licences",
    summary: "Some styles need you to accept their terms once per machine",
    keywords: ["licence", "license", "terms", "accept"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "library",
    label: "My images & trash",
    summary: "Trash retention on this device",
    keywords: [
      "trash",
      "retention",
      "collections",
      "albums",
      "deleted prints",
      "purge",
      "library",
    ],
    surfaces: ["web", "desktop"],
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
    surfaces: ["desktop"],
  },
  {
    id: "pairing",
    label: "Phone pairing",
    summary: "Use mold on your phone",
    keywords: ["phone", "iphone", "pair", "pairing", "qr", "mobile"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "performance",
    label: "Speed & memory",
    summary: "Memory, previews, queueing, and compute backends",
    keywords: ["performance", "vram", "offload", "preview"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "accounts",
    label: "Accounts & tokens",
    summary: "Credentials used for catalogs and style downloads",
    keywords: [
      "hugging face",
      "huggingface",
      "hf",
      "civitai",
      "token",
      "api key",
    ],
    surfaces: ["web", "desktop"],
  },
  {
    id: "cloud",
    label: "Cloud GPUs",
    summary: "Rented machines, their keys, and what they are allowed to cost",
    keywords: [
      "runpod",
      "lambda",
      "rent",
      "cloud",
      "gpu",
      "pod",
      "hourly",
      "billing",
    ],
    surfaces: ["web", "desktop"],
  },
  {
    id: "styleDefaults",
    label: "Per-style defaults",
    summary: "Per-style overrides saved on this machine",
    keywords: ["model", "style", "per-model", "defaults", "lora", "override"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "profiles",
    label: "Profiles",
    summary: "Keep separate sets of engine preferences",
    keywords: ["profile"],
    surfaces: ["web", "desktop"],
  },
  {
    id: "advanced",
    label: "Advanced",
    summary: "Uncommon and newly discovered engine options",
    surfaces: ["web", "desktop"],
  },
  {
    id: "updates",
    label: "Updates & about",
    summary: "Version, update channel, logs, and diagnostics",
    keywords: [
      "update",
      "version",
      "about",
      "nightly",
      "stable",
      "logs",
      "diagnostics",
      "privacy",
    ],
    surfaces: ["web", "desktop"],
  },
];

export type EditorKind =
  "toggle" | "select" | "number" | "text" | "slider" | "path" | "secret";

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
  /**
   * Changing this requires restarting the whole APP, not just the engine.
   *
   * The desktop build runs the engine as a thread inside the Tauri process and
   * applies engine settings with `set_var` into that same process, while these
   * values are read through `runtime_env`, which freezes them in a `OnceLock`
   * on first use — per PROCESS, not per engine start. Restarting the engine
   * thread therefore cannot change them, and a row that says "RESTART ENGINE"
   * for one of them is telling the user something untrue.
   */
  needsAppRestart?: boolean;
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
    section: "performance",
    label: "Server port",
    help: "Port for `mold serve`. The built-in engine always uses an ephemeral port.",
    editor: "number",
    min: 1,
    max: 65535,
    needsEngineRestart: true,
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
    // The engine's own ladder (`validate_enum` in config_keys.rs); the
    // contract test refuses an option the setter would 422.
    options: ["auto", "fp16", "q8", "q6", "q5", "q4", "q3"].map((value) => ({
      value,
      label: value,
    })),
  },
  {
    // The engine registers this key but has no getter or setter for it yet (#778),
    // so no machine reports it and the row never draws; the schema keeps it so
    // the registry contract holds.
    key: "umt5_variant",
    section: "generation",
    label: "How Wan reads your words",
    help: "The size of its UMT5 text encoder. Smaller trades a little fidelity for graphics memory.",
    editor: "select",
    // The engine's own ladder (`validate_enum` in config_keys.rs); the
    // contract test refuses an option the setter would 422.
    options: ["auto", "fp16", "q8", "q6", "q5", "q4", "q3"].map((value) => ({
      value,
      label: value,
    })),
  },
  {
    key: "qwen3_variant",
    section: "generation",
    label: "How Flux.2 and Z-Image read your words",
    help: "The size of their Qwen3 text encoder. Smaller trades a little fidelity for graphics memory.",
    editor: "select",
    options: ["auto", "bf16", "q8", "q6", "iq4", "q3"].map((value) => ({
      value,
      label: value,
    })),
  },
  {
    key: "gallery.trash_retention_days",
    section: "library",
    label: "Keep deleted pictures for",
    help: "Pictures in the trash are deleted forever after this long. Forever keeps them until you empty the trash.",
    editor: "select",
    options: RETENTION_OPTIONS.map((days) => ({
      value: String(days),
      label: retentionLabel(days),
    })),
  },
  {
    key: "gallery.authority_log",
    section: "library",
    label: "Faster library bookkeeping",
    help: "Record each change to the picture index as it happens instead of rewriting the whole index every time a picture is saved — much faster once the library is large. A mold older than 0.29 cannot save into a library kept this way, so only turn it on where every copy of mold sharing this folder is new enough; `mold system gallery-authority` reports the format and can put it back.",
    editor: "toggle",
    needsEngineRestart: true,
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
  {
    key: "queue.held_retention_days",
    section: "performance",
    label: "Keep held work for",
    help: "Work parked because a machine could not take it is cleared after this long. Forever keeps it until you clear it.",
    editor: "select",
    options: RETENTION_OPTIONS.map((days) => ({
      value: String(days),
      label: retentionLabel(days),
    })),
  },
  {
    key: "generate.auto_tag_title",
    section: "library",
    label: "Tag command-line prints with their title",
    help: "`mold run` adds each titled print's own tag. Each app keeps its own switch for the prints it makes.",
    editor: "toggle",
  },
  {
    key: "logging.level",
    section: "updates",
    label: "Log detail",
    help: "How much the engine writes down. `info` is normal; `debug` is for chasing a problem.",
    editor: "select",
    options: [
      { value: "trace", label: "trace" },
      { value: "debug", label: "debug" },
      { value: "info", label: "info" },
      { value: "warn", label: "warn" },
      { value: "error", label: "error" },
    ],
    needsEngineRestart: true,
  },
  {
    key: "logging.file",
    section: "updates",
    label: "Write a log file",
    help: "Off keeps the log on screen only.",
    editor: "toggle",
    needsEngineRestart: true,
  },
  {
    key: "logging.dir",
    section: "updates",
    label: "Log folder",
    help: "Where log files are written. Empty means `logs/` inside the Mold home.",
    editor: "path",
    needsEngineRestart: true,
  },
  {
    key: "logging.max_days",
    section: "updates",
    label: "Keep logs for",
    help: "Older log files are deleted.",
    editor: "number",
    min: 1,
    max: 3650,
    needsEngineRestart: true,
  },
  {
    key: "runpod.api_key",
    section: "cloud",
    label: "RunPod key",
    help: "Lets Mold rent and release RunPod machines for you.",
    editor: "secret",
  },
  {
    key: "runpod.default_gpu",
    section: "cloud",
    label: "RunPod card to rent",
    help: "Which graphics card a new pod asks for by default.",
    editor: "text",
  },
  {
    key: "runpod.default_datacenter",
    section: "cloud",
    label: "RunPod datacenter",
    help: "Where a new pod is created. Empty lets RunPod choose.",
    editor: "text",
  },
  {
    key: "runpod.default_network_volume_id",
    section: "cloud",
    label: "RunPod storage volume",
    help: "A volume that keeps styles between pods so they are not downloaded twice.",
    editor: "text",
  },
  {
    key: "runpod.auto_teardown",
    section: "cloud",
    label: "Release idle pods automatically",
    help: "A rented machine bills by the minute, whether or not it is working.",
    editor: "toggle",
  },
  {
    key: "runpod.auto_teardown_idle_mins",
    section: "cloud",
    label: "Release after",
    help: "How long a pod may sit idle before it is released.",
    editor: "number",
    min: 0,
    max: 10080,
  },
  {
    key: "runpod.cost_alert_usd",
    section: "cloud",
    label: "Warn me above",
    help: "Warn before a run is expected to cost more than this.",
    editor: "number",
    min: 0,
    max: 1000,
  },
  {
    key: "runpod.endpoint",
    section: "cloud",
    label: "RunPod address",
    help: "Leave empty unless you have been given a different one.",
    editor: "text",
  },
  {
    key: "lambda.api_key",
    section: "cloud",
    label: "Lambda key",
    help: "Lets Mold rent and release Lambda machines for you.",
    editor: "secret",
  },
  {
    key: "lambda.endpoint",
    section: "cloud",
    label: "Lambda address",
    help: "Leave empty unless you have been given a different one.",
    editor: "text",
  },
  {
    key: "lambda.image_repository",
    section: "cloud",
    label: "Lambda image",
    help: "The container image a rented Lambda machine boots.",
    editor: "text",
  },
  {
    key: "lambda.ssh_key_name",
    section: "cloud",
    label: "Lambda SSH key name",
    help: "The key Lambda uses to let Mold in.",
    editor: "text",
  },
  {
    key: "lambda.ssh_private_key_path",
    section: "cloud",
    label: "Lambda SSH key file",
    help: "Where that key lives on this machine.",
    // A FILE, and the injected picker chooses folders: typed until there is a file picker.
    editor: "text",
  },
  {
    key: "lambda.filesystem_prefix",
    section: "cloud",
    label: "Lambda storage name",
    help: "A shared filesystem that keeps styles between machines.",
    editor: "text",
  },
  {
    key: "lambda.filesystem_mount_path",
    section: "cloud",
    label: "Lambda storage path",
    help: "Where that filesystem is mounted inside the machine.",
    editor: "text",
  },
  {
    key: "lambda.confirm_hourly_usd",
    section: "cloud",
    label: "Ask above",
    help: "Confirm before renting a machine that costs more than this an hour.",
    editor: "number",
    min: 0,
    max: 1000,
  },
  {
    key: "lambda.local_port",
    section: "cloud",
    label: "Lambda local port",
    help: "The port on this machine the tunnel to Lambda uses.",
    editor: "number",
    min: 1,
    max: 65535,
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
    needsAppRestart: true,
    section: "performance",
    label: "Park text encoders in RAM",
    help: "Keep text encoders in the machine's RAM between requests instead of re-reading them from disk — BF16/FP16 and quantized GGUF encoders alike. Automatic covers Flux.2 and Z-Image, measuring this machine and parking only when the encoder, the transformer that loads beside it, and a 15%-of-RAM (minimum 8 GB) safety floor all fit, so a 64 GB desktop keeps streaming. Always is the opt-in every other style reads — FLUX and SD3's T5, Wan's encoder, Qwen-Image's — and parks whenever the encoder alone clears that floor. Costs several GB of the machine's RAM per parked encoder. No effect on Metal (unified memory).",
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    needsAppRestart: true,
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
    key: "env.MOLD_RESERVE_VRAM_MB",
    needsAppRestart: true,
    section: "performance",
    label: "Graphics memory held back",
    help: "Megabytes of graphics memory every budget decision leaves alone for the driver, the desktop and the maths libraries' own workspaces — what the card reports free is never quite what the next allocation can take. Leave it empty for this machine's default: 400 on Linux, 600 on Windows, 0 on a Mac, where memory is shared with the system and already has its own headroom. 0 holds nothing back; raising it makes mold stream a large style rather than keep it on the card.",
    editor: "number",
    min: 0,
    max: 65536,
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

const BY_KEY = new Map(
  [...ENGINE_KEY_SCHEMAS, ...ENV_KNOB_SCHEMAS].map((s) => [s.key, s]),
);

export function schemaFor(key: string): KeySchema | null {
  return BY_KEY.get(key) ?? null;
}

/**
 * Section for an engine-config key: `models.<style>.<field>` is a per-style
 * override, curated keys go to their section, `tui.*` stays out of a
 * graphical surface, and everything else lands in Advanced.
 */
export function sectionForConfigKey(key: string): SectionId | null {
  if (key.startsWith("tui.")) return null;
  if (parsePerStyleKey(key)) return "styleDefaults";
  return schemaFor(key)?.section ?? "advanced";
}

/**
 * Split a raw `models.<style>.<field>` key.
 *
 * A style name carries dots and colons (`sd1.5`, `flux-dev:q4`), so the FIELD
 * is what follows the LAST dot — the same rule as
 * `mold_core::config_keys::parse_model_key`. Splitting on the first dot reads
 * `sd1` as the style and `5.default_steps` as a field nothing renders.
 */
export function parsePerStyleKey(
  key: string,
): { style: string; field: string } | null {
  const rest = key.startsWith("models.") ? key.slice("models.".length) : null;
  if (rest === null) return null;
  const lastDot = rest.lastIndexOf(".");
  if (lastDot <= 0) return null;
  const style = rest.slice(0, lastDot);
  const field = rest.slice(lastDot + 1);
  if (!style || !field) return null;
  return { style, field };
}

/** The fields a style may override, in `MODEL_FIELDS` order — which is the
 *  order `mold config list` prints and therefore the one a person has seen.
 *  Pinned against the Rust source by `settingsSchema.contract.test.ts`. */
export const PER_STYLE_FIELDS: readonly string[] = [
  "default_steps",
  "default_guidance",
  "default_width",
  "default_height",
  "scheduler",
  "negative_prompt",
  "lora",
  "lora_scale",
];

/**
 * The inspector's word for each per-style engine field, so a per-style row
 * says "Detail" over `default_steps` the way the inspector says Detail over
 * `28 passes`. An unknown field keeps its engine name.
 */
export const PER_STYLE_FIELD_LABELS: Readonly<Record<string, string>> = {
  default_steps: "Detail",
  default_guidance: "Stick to my words",
  default_width: "Width",
  default_height: "Height",
  scheduler: "Scheduler",
  negative_prompt: "Words to avoid",
  lora: "Add-on look",
  lora_scale: "Add-on look strength",
};

export interface PerStyleGroup<R> {
  style: string;
  rows: R[];
}

/**
 * One disclosure per style: styles alphabetical, fields in `PER_STYLE_FIELDS`
 * order, and any field this build does not know kept after the ones it does
 * rather than dropped. hal9000's 104 flat rows are 13 groups.
 */
export function groupPerStyleRows<R extends { key: string }>(
  rows: readonly R[],
): PerStyleGroup<R>[] {
  const byStyle = new Map<string, R[]>();
  for (const row of rows) {
    const parsed = parsePerStyleKey(row.key);
    if (!parsed) continue;
    const bucket = byStyle.get(parsed.style);
    if (bucket) bucket.push(row);
    else byStyle.set(parsed.style, [row]);
  }
  const rank = (row: R) => {
    const field = parsePerStyleKey(row.key)?.field ?? "";
    const index = PER_STYLE_FIELDS.indexOf(field);
    return index === -1 ? PER_STYLE_FIELDS.length : index;
  };
  return [...byStyle.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([style, styleRows]) => ({
      style,
      rows: [...styleRows].sort(
        (a, b) => rank(a) - rank(b) || a.key.localeCompare(b.key),
      ),
    }));
}

/** Sections a shell renders, in schema order. */
export function sectionsForSurface(surface: SettingsSurface): SectionInfo[] {
  return SECTIONS.filter((section) => section.surfaces.includes(surface));
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
export function schemasForSection(
  sectionId: SectionId,
  surface?: SettingsSurface,
): KeySchema[] {
  // The env knobs configure the desktop's embedded engine; a browser has no
  // engine of its own and never draws them.
  const pool =
    surface === "web"
      ? ENGINE_KEY_SCHEMAS
      : [...ENGINE_KEY_SCHEMAS, ...ENV_KNOB_SCHEMAS];
  return pool.filter((s) => s.section === sectionId);
}

/**
 * Whether a settings section has anything matching the query — its label, its
 * summary, its declared keywords, any curated key it owns, or a raw engine row
 * the caller says this section renders. Drives which sections the global
 * search leaves on the page.
 *
 * The raw evidence is keyed BY SECTION rather than assumed to be Advanced's:
 * per-style overrides are raw rows too, and on a machine with a dozen styles
 * they are most of them.
 */
export function sectionMatchesSearch(
  query: string,
  section: SectionInfo,
  rawKeysBySection: Partial<Record<SectionId, string[]>> = {},
  surface?: SettingsSurface,
): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (section.label.toLowerCase().includes(q)) return true;
  if (section.summary.toLowerCase().includes(q)) return true;
  if (section.keywords?.some((keyword) => keyword.toLowerCase().includes(q)))
    return true;
  if (
    schemasForSection(section.id, surface).some((schema) =>
      matchesSearch(query, schema),
    )
  )
    return true;
  if (
    rawKeysBySection[section.id]?.some((key) => key.toLowerCase().includes(q))
  )
    return true;
  return false;
}
