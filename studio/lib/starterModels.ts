export interface StarterModel {
  model: string;
  displayName: string;
  speed: string;
  size: string;
  recommended: boolean;
}

/**
 * Canonical manifest ids offered on an empty Create surface.
 *
 * Keep the explicit tags here: these values go directly to `/api/downloads`,
 * whose `name:tag` inputs are treated as canonical rather than alias-resolved.
 */
export const STARTER_MODELS = [
  {
    model: "flux2-klein:q4",
    displayName: "FLUX.2 Klein",
    speed: "fast",
    size: "6.9 GB",
    recommended: true,
  },
  {
    model: "z-image-turbo:q8",
    displayName: "Z-Image Turbo",
    speed: "fastest",
    size: "3.4 GB",
    recommended: false,
  },
  {
    model: "sdxl-base:fp16",
    displayName: "SDXL",
    speed: "classic",
    size: "6.6 GB",
    recommended: false,
  },
] as const satisfies readonly StarterModel[];
