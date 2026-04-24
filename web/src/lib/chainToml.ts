import { parse, stringify } from "smol-toml";

export interface ChainScriptChain {
  model: string;
  width: number;
  height: number;
  fps: number;
  seed?: number;
  steps: number;
  guidance: number;
  strength: number;
  motion_tail_frames: number;
  output_format: "mp4" | "gif" | "apng" | "webp";
}

export interface ChainStageToml {
  prompt: string;
  frames: number;
  source_image?: string;
  source_image_b64?: string;
  negative_prompt?: string;
  seed_offset?: number;
  transition?: "smooth" | "cut" | "fade";
  fade_frames?: number;
}

export interface ChainScriptToml {
  schema: string;
  chain: ChainScriptChain;
  stage: ChainStageToml[];
}

const SCHEMA = "mold.chain.v1";

export function writeChainScript(script: ChainScriptToml): string {
  const doc: Record<string, unknown> = {
    schema: SCHEMA,
    chain: script.chain,
    stage: script.stage,
  };
  return stringify(doc);
}

export function readChainScript(src: string): ChainScriptToml {
  const raw = parse(src) as unknown as Partial<ChainScriptToml>;
  const schema = raw.schema ?? SCHEMA;
  if (schema !== SCHEMA) {
    throw new Error(
      `chain TOML schema '${schema}' is not supported by this mold version (supported: '${SCHEMA}')`,
    );
  }
  if (!raw.chain) throw new Error("chain TOML missing [chain] table");
  if (!raw.stage || raw.stage.length === 0)
    throw new Error("chain TOML missing [[stage]] entries");
  return {
    schema: SCHEMA,
    chain: raw.chain,
    stage: raw.stage,
  };
}
