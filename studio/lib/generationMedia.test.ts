import { describe, expect, it } from "vitest";
import {
  GENERATION_MEDIA_AUTHORITY_FIELDS,
  redactGenerationMediaForPersistence,
  requestCarriesGenerationMedia,
} from "./generationMedia";

describe("generation media session boundary", () => {
  it.each(GENERATION_MEDIA_AUTHORITY_FIELDS)(
    "classifies present %s authority as session media",
    (field) => {
      expect(requestCarriesGenerationMedia({ [field]: "present" })).toBe(true);
      expect(requestCarriesGenerationMedia({ [field]: null })).toBe(false);
    },
  );

  it("removes bytes, paths, handles, references, and identity metadata without walking bytes", () => {
    const media = {
      toJSON(): never {
        throw new Error("media bytes were serialized");
      },
    };
    const redacted = redactGenerationMediaForPersistence({
      prompt: "safe prompt",
      model: "minimax-h3-ref2va",
      source_image: media,
      source_image_name: "private-source.png",
      id_image: media,
      id_images: [media],
      id_image_name: "face.png",
      id_weight: 1.2,
      id_start_step: 2,
      audio_file_path: "/private/audio.wav",
      source_video_path: "/private/video.mp4",
      references: [
        {
          media: { authority: "upload", handle: "one-use-secret" },
          provenance: { name: "face.png", sha256: "biometric-digest" },
        },
      ],
      stages: [
        {
          prompt: "one",
          frames: 9,
          source_image: media,
          source_image_name: "private-stage-source.png",
        },
      ],
    });

    expect(() => JSON.stringify(redacted)).not.toThrow();
    expect(JSON.stringify(redacted)).toBe(
      JSON.stringify({
        prompt: "safe prompt",
        model: "minimax-h3-ref2va",
        stages: [{ prompt: "one", frames: 9 }],
      }),
    );
  });
});
