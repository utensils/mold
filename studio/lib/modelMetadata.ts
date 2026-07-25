export interface ModelKindInput {
  kind?: string | null;
  family?: string | null;
}

const KIND_LABELS: Readonly<Record<string, string>> = {
  checkpoint: "Checkpoint",
  lora: "LoRA",
  vae: "VAE",
  "text-encoder": "Text encoder",
  tokenizer: "Tokenizer",
  clip: "CLIP",
  "control-net": "ControlNet",
  upscaler: "Upscaler",
  "prompt-expander": "Prompt expander",
};

const MODEL_SIZE_LABELS: Readonly<Record<string, string>> = {
  checkpoint: "Checkpoint weights",
  lora: "LoRA weights",
  vae: "VAE weights",
  "text-encoder": "Text encoder weights",
  tokenizer: "Tokenizer files",
  clip: "CLIP weights",
  "control-net": "ControlNet weights",
  upscaler: "Upscaler weights",
  "prompt-expander": "Prompt expander weights",
};

function normalizeKind(kind: string): string {
  return kind.trim().toLowerCase().replaceAll("_", "-").replace(/\s+/g, "-");
}

function sentenceCase(value: string): string {
  const words = value.replaceAll("_", "-").split("-").filter(Boolean);
  if (words.length === 0) return "Model";
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * Resolve the precise catalog kind when supplied, otherwise infer the small
 * set of built-in utility families that `/api/models` exposes without a kind.
 * Standalone generation models are checkpoints.
 */
export function modelKindValue(model: ModelKindInput): string {
  const explicit = model.kind?.trim();
  if (explicit) return normalizeKind(explicit);

  const family = model.family?.trim().toLowerCase() ?? "";
  if (family === "real-esrgan" || family === "upscaler") return "upscaler";
  if (family === "controlnet") return "control-net";
  if (family === "qwen3-expand") return "prompt-expander";
  return "checkpoint";
}

/** Human-readable, singular model-kind label shared by every Studio shell. */
export function modelKindLabel(kind: string): string {
  const normalized = normalizeKind(kind);
  return KIND_LABELS[normalized] ?? sentenceCase(normalized);
}

/** Kind-aware size heading; never call a LoRA/VAE/ControlNet a checkpoint. */
export function modelWeightsLabel(kind: string): string {
  return MODEL_SIZE_LABELS[normalizeKind(kind)] ?? "Model files";
}

/** Readable modality with a future-proof fallback for additive wire values. */
export function modelModalityLabel(modality: string): string {
  return sentenceCase(modality.trim().toLowerCase());
}
