import { captureFreshVideoReference } from "./freshVideoReference";
import type { VideoNormalizationCapabilities } from "../api/referenceNormalizationTypes";
import { inputImageBase64, inputImageFacts } from "./inputImage";
import { probeMinimaxH3Mp4, probeMinimaxH3Wav } from "./minimaxH3MediaProbe";
import {
  emptyMinimaxH3AuthoringState,
  minimaxH3ReferenceBudget,
  type MinimaxH3AuthoringState,
  type MinimaxH3ReferenceDraft,
} from "./minimaxH3Authoring";

function base64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("Could not read media."));
    reader.onload = () => {
      const result = String(reader.result ?? "");
      const comma = result.indexOf(",");
      if (comma < 0)
        reject(new Error("The selected media could not be encoded."));
      else resolve(result.slice(comma + 1));
    };
    reader.readAsDataURL(file);
  });
}

async function digestBytes(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Secure media hashing is unavailable on this device.");
  }
  const value = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(value)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function imageDimensions(
  file: File,
): Promise<{ width: number; height: number }> {
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(file);
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not decode the selected image."));
    };
    image.src = url;
  });
}

export async function minimaxH3PickedMediaReference(
  file: File,
  normalization?: VideoNormalizationCapabilities | null,
): Promise<MinimaxH3ReferenceDraft> {
  const mime = file.type.toLowerCase();
  const extension = file.name.split(".").at(-1)?.toLowerCase() ?? "";
  if (
    normalization?.available &&
    normalization.protocol_version === 1 &&
    normalization.input_roles.includes("reference_video") &&
    (mime.startsWith("video/") ||
      normalization.accepted_extensions.includes(extension))
  ) {
    return { reference: await captureFreshVideoReference(file, normalization) };
  }
  if (mime.startsWith("image/")) {
    const data = await inputImageBase64(file);
    const facts = inputImageFacts(data, file.name);
    const bytes = Uint8Array.from(atob(data), (character) =>
      character.charCodeAt(0),
    );
    const sha256 = await digestBytes(bytes.buffer);
    const normalized = new File([bytes], facts.filename, {
      type: facts.mimeType,
    });
    const dimensions = await imageDimensions(normalized);
    return {
      reference: {
        kind: "image",
        media: { authority: "inline", data },
        provenance: { name: facts.filename, sha256 },
        mime_type: facts.mimeType,
        ...dimensions,
      },
    };
  }
  const isMp4 = mime === "video/mp4" || /\.mp4$/i.test(file.name);
  const isWav =
    ["audio/wav", "audio/x-wav", "audio/wave"].includes(mime) ||
    /\.wav$/i.test(file.name);
  if (!isMp4 && !isWav) {
    throw new Error(`${file.name}: choose an image, video, or audio file.`);
  }
  const bytes = await file.arrayBuffer();
  try {
    const facts = isMp4 ? probeMinimaxH3Mp4(bytes) : probeMinimaxH3Wav(bytes);
    if (
      facts.mimeType === "audio/wav" &&
      (facts.channels < 1 || facts.channels > 2)
    ) {
      throw new Error("H3 accepts mono or stereo audio.");
    }
    const [data, sha256] = await Promise.all([
      base64(file),
      digestBytes(bytes),
    ]);
    const media = { authority: "inline" as const, data };
    const provenance = { name: file.name, sha256 };
    if (facts.mimeType === "video/mp4") {
      return {
        reference: {
          kind: "video",
          media,
          provenance,
          mime_type: facts.mimeType,
          width: facts.width,
          height: facts.height,
          frame_count: facts.frameCount,
          duration_ms: facts.durationMs,
          fps: facts.fps,
          has_audio: facts.hasAudio,
          audio_duration_ms: facts.audioDurationMs,
          audio_sample_count: facts.audioSampleCount,
          audio_sample_rate: facts.audioSampleRate,
          audio_channels: facts.audioChannels,
        },
      };
    }
    return {
      reference: {
        kind: "audio",
        media,
        provenance,
        mime_type: facts.mimeType,
        duration_ms: facts.durationMs,
        sample_rate: facts.sampleRate,
        channels: facts.channels,
        sample_count: facts.sampleCount,
      },
    };
  } catch (reason) {
    const message = reason instanceof Error ? reason.message : String(reason);
    throw new Error(`${file.name}: ${message}`);
  }
}

export async function appendMinimaxH3PickedMediaReference(
  state: MinimaxH3AuthoringState | null | undefined,
  file: File,
): Promise<MinimaxH3AuthoringState> {
  const current = state ?? emptyMinimaxH3AuthoringState();
  const reference = await minimaxH3PickedMediaReference(file);
  const references = [...current.references, reference];
  const failures = minimaxH3ReferenceBudget(references).errors.filter(
    (error) =>
      error !==
      "Audio references require at least one image or video reference.",
  );
  if (failures.length) throw new Error(failures.join(" "));
  return { ...current, references };
}
