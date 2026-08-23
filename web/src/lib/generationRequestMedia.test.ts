import { describe, expect, it } from "vitest";
import type { GenerateRequestWire } from "../types";
import {
  GENERATION_REQUEST_MEDIA_FIELDS,
  requestCarriesGenerationMedia,
} from "./generationRequestMedia";

function request(): GenerateRequestWire {
  return {
    prompt: "a patient red fox",
    model: "flux-dev:fp8",
    width: 512,
    height: 512,
    steps: 8,
    guidance: 3,
  };
}

const PRESENT_MEDIA_VALUE: Record<
  (typeof GENERATION_REQUEST_MEDIA_FIELDS)[number],
  unknown
> = {
  source_image: "source-bytes",
  edit_images: ["edit-bytes"],
  references: [{ type: "image", media: { authority: "inline", data: "ref" } }],
  id_image: "identity-bytes",
  id_images: ["identity-bytes"],
  mask_image: "mask-bytes",
  control_image: "control-bytes",
  audio_file: "audio-bytes",
  audio_file_path: "/private/audio.wav",
  source_video: "source-video-bytes",
  source_video_path: "/private/source.mp4",
  extend_video: "extend-video-bytes",
  extend_video_path: "/private/extend.mp4",
  keyframes: [{ frame: 0, image: "keyframe-bytes" }],
  hdr_exr_dir: "/private/hdr",
};

describe("requestCarriesGenerationMedia", () => {
  it.each(GENERATION_REQUEST_MEDIA_FIELDS)(
    "classifies present %s as media-bearing",
    (field) => {
      expect(
        requestCarriesGenerationMedia({
          ...request(),
          [field]: PRESENT_MEDIA_VALUE[field],
        } as GenerateRequestWire),
      ).toBe(true);
    },
  );

  it("does not classify ordinary settings or unused null media slots as media", () => {
    const ordinary: GenerateRequestWire = {
      ...request(),
      source_image: null,
      edit_images: null,
      references: null,
      id_image: null,
      id_images: null,
      mask_image: null,
      control_image: null,
      audio_file: null,
      audio_file_path: null,
      source_video: null,
      source_video_path: null,
      extend_video: null,
      extend_video_path: null,
      keyframes: null,
      hdr_exr_dir: null,
      source_image_name: "descriptor-only.png",
      id_image_name: "descriptor-only-face.png",
      source_fit: { mode: "pad-fit" },
      loras: [{ path: "model/lora.safetensors", scale: 0.8 }],
    } as GenerateRequestWire;

    expect(requestCarriesGenerationMedia(ordinary)).toBe(false);
  });

  it("conservatively treats present empty collections and paths as media authority", () => {
    expect(
      requestCarriesGenerationMedia({ ...request(), edit_images: [] }),
    ).toBe(true);
    expect(
      requestCarriesGenerationMedia({ ...request(), audio_file_path: "" }),
    ).toBe(true);
  });
});
