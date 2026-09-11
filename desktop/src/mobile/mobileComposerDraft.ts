import {
  deleteDraftMediaByPrefix,
  getDraftMedia,
  putDurableMediaBatch,
  type DraftMediaRecord,
} from "@studio/lib/draftMediaStore";
import { createUuid } from "@studio/lib/id";
import { parseSourceFitPolicy } from "@studio/lib/sourceFit";
import type { CanvasIntent } from "@studio/lib/outputShape";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import {
  hydrateMobileDraftMedia,
  MOBILE_DRAFT_MEDIA_PREFIX,
  partitionMobileDraftMedia,
} from "./mobileDraftMedia";

export const MOBILE_COMPOSER_DRAFT_KEY = "mold.mobile.generate-draft.v1";
const DERIVED = new Set([
  "fileUnderAutoTag",
  "fileUnderMatch",
  "guidanceCapabilities",
  "sourceImageCapability",
  "identitySupported",
  "durationPredictionSupported",
  "recipeCapabilities",
  "extendDefaultOverlapFrames",
]);
interface Envelope {
  version: 1;
  revision: string;
  canvasIntent: CanvasIntent;
  form: GenerateForm;
}
interface Persistence {
  storage: Pick<Storage, "getItem" | "setItem" | "removeItem">;
  write: (records: DraftMediaRecord[]) => Promise<boolean>;
  read: (id: string) => Promise<DraftMediaRecord | null>;
  remove: (prefix: string) => Promise<unknown>;
  revision: () => string;
}

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === "string");
const number = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const picked = (value: unknown) =>
  value === null || (object(value) && typeof value.filename === "string" && value.base64 === "");

/** Stored state is versioned data, never an arbitrary object to assign to Vue.
 * Match current primitive/default shapes, then validate variable media/list
 * shapes before any form consumer can run. Request validation remains the
 * authority for values such as model-supported sizes and parameter bounds. */
function validForm(value: unknown): value is GenerateForm {
  if (!object(value)) return false;
  const defaults = newGenerateForm();
  for (const [key, fallback] of Object.entries(defaults)) {
    if (DERIVED.has(key)) continue;
    const actual = value[key];
    if (actual === undefined) continue;
    if (typeof fallback === "string" && typeof actual !== "string") return false;
    if (typeof fallback === "boolean" && typeof actual !== "boolean") return false;
    if (typeof fallback === "number" && !number(actual)) return false;
    if (Array.isArray(fallback) && !Array.isArray(actual)) return false;
    if (object(fallback) && !object(actual)) return false;
  }
  for (const key of ["sourceImage", "maskImage", "controlImage"]) {
    if (value[key] !== undefined && value[key] !== null && value[key] !== "") return false;
  }
  for (const key of [
    "originalPrompt",
    "sourceImageName",
    "cameraControl",
    "pipeline",
    "icLoraControl",
    "exclusiveWell",
  ]) {
    if (value[key] != null && typeof value[key] !== "string") return false;
  }
  for (const key of [
    "sourceImageWidth",
    "sourceImageHeight",
    "identityWeight",
    "identityStartStep",
    "extendOverlapFrames",
  ]) {
    if (value[key] != null && !number(value[key])) return false;
  }
  for (const key of ["endFrame", "identityImage", "sourceVideo", "extendVideo", "audioFile"]) {
    if (value[key] !== undefined && !picked(value[key])) return false;
  }
  if (
    value.imageAttachments !== undefined &&
    (!strings(value.imageAttachments) || value.imageAttachments.some(Boolean))
  )
    return false;
  if (
    value.keyframes !== undefined &&
    !(value.keyframes as unknown[]).every(
      (entry) =>
        object(entry) && number(entry.frame) && entry.image !== null && picked(entry.image),
    )
  )
    return false;
  if (
    value.loras !== undefined &&
    !(value.loras as unknown[]).every(
      (entry) =>
        object(entry) &&
        typeof entry.path === "string" &&
        typeof entry.name === "string" &&
        number(entry.scale) &&
        strings(entry.trainedWords),
    )
  )
    return false;
  if (value.fileUnder !== undefined) {
    const filing = value.fileUnder as Record<string, unknown>;
    if (
      !strings(filing.manualTags) ||
      typeof filing.ghostRemoved !== "boolean" ||
      typeof filing.pickedExplicitly !== "boolean"
    )
      return false;
    if (
      filing.picked != null &&
      (!object(filing.picked) ||
        typeof filing.picked.name !== "string" ||
        (filing.picked.id !== undefined && typeof filing.picked.id !== "string"))
    )
      return false;
    if (filing.clearedMatchSlug != null && typeof filing.clearedMatchSlug !== "string")
      return false;
  }
  if (value.sourceFit !== undefined && !parseSourceFitPolicy(value.sourceFit)) return false;
  for (const key of ["guidanceOverrides", "wanRecipe", "mesh"]) {
    const block = value[key];
    if (
      object(block) &&
      Object.values(block).some(
        (entry) =>
          entry !== null &&
          !number(entry) &&
          typeof entry !== "boolean" &&
          typeof entry !== "string",
      )
    )
      return false;
  }
  if (
    value.retakeRange != null &&
    !(
      object(value.retakeRange) &&
      number(value.retakeRange.start_seconds) &&
      number(value.retakeRange.end_seconds)
    )
  )
    return false;
  if (value.spatialUpscale != null && !["x1-5", "x2"].includes(String(value.spatialUpscale)))
    return false;
  if (value.temporalUpscale != null && value.temporalUpscale !== "x2") return false;
  if (value.namedViews !== undefined) {
    for (const [role, image] of Object.entries(value.namedViews as Record<string, unknown>)) {
      if (
        !["front", "left", "back", "right"].includes(role) ||
        !picked(image) ||
        !object(image) ||
        !number(image.width) ||
        !number(image.height) ||
        typeof image.mimeType !== "string"
      )
        return false;
    }
  }
  if (value.h3Authoring !== undefined) {
    const h3 = value.h3Authoring as Record<string, unknown>;
    for (const key of ["firstFrame", "lastFrame"]) {
      const image = h3[key];
      if (
        image !== null &&
        !(
          object(image) &&
          image.data === "" &&
          typeof image.filename === "string" &&
          typeof image.mimeType === "string" &&
          number(image.width) &&
          number(image.height)
        )
      )
        return false;
    }
    if (
      !Array.isArray(h3.references) ||
      !h3.references.every(
        (entry) =>
          object(entry) &&
          object(entry.reference) &&
          object(entry.reference.media) &&
          entry.reference.media.authority === "descriptor" &&
          typeof entry.reference.kind === "string" &&
          typeof entry.reference.mime_type === "string",
      )
    )
      return false;
  }
  return true;
}

function parse(raw: string): Envelope | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (
      !object(value) ||
      value.version !== 1 ||
      typeof value.revision !== "string" ||
      !/^[a-zA-Z0-9-]+$/.test(value.revision) ||
      !["source", "source-exact", "model-default", "manual"].includes(String(value.canvasIntent)) ||
      !validForm(value.form)
    )
      return null;
    const form = newGenerateForm();
    // Unknown root keys (including credentials, routes and live results) never
    // enter the composer. Derived capabilities must come from fresh inventory.
    for (const key of Object.keys(form) as (keyof GenerateForm)[]) {
      if (!DERIVED.has(key) && Object.hasOwn(value.form, key))
        Object.assign(form, { [key]: value.form[key] });
    }
    form.sourceFit = parseSourceFitPolicy(form.sourceFit)!;
    // The composer's preset chips are retired: a preset a draft saved before
    // then would restyle the prompt at submit with nothing on screen to show
    // or clear it. Never silently change a request.
    form.stylePreset = "";
    return {
      version: 1,
      revision: value.revision,
      canvasIntent: value.canvasIntent as CanvasIntent,
      form,
    };
  } catch {
    return null;
  }
}

export function createMobileComposerDraft(overrides: Partial<Persistence> = {}) {
  const persistence: Persistence = {
    storage: localStorage,
    write: putDurableMediaBatch,
    read: getDraftMedia,
    remove: deleteDraftMediaByPrefix,
    revision: createUuid,
    ...overrides,
  };
  let epoch = 0;
  let savedRevision: string | null = null;
  let savedMedia: DraftMediaRecord[] = [];
  let tail: Promise<unknown> = Promise.resolve();

  async function restore(): Promise<{
    form?: GenerateForm;
    canvasIntent?: CanvasIntent;
    error: string;
    missing: string[];
  }> {
    const ticket = ++epoch;
    let raw: string | null;
    try {
      raw = persistence.storage.getItem(MOBILE_COMPOSER_DRAFT_KEY);
    } catch {
      return { error: "The unfinished draft could not be read.", missing: [] as string[] };
    }
    if (!raw) return { error: "", missing: [] as string[] };
    const envelope = parse(raw);
    if (!envelope)
      return {
        error: "The saved draft is unreadable. Reset it to start a new print.",
        missing: [] as string[],
      };
    try {
      const hydrated = await hydrateMobileDraftMedia(
        envelope.form,
        envelope.revision,
        persistence.read,
      );
      if (ticket !== epoch) return { error: "", missing: [] };
      savedRevision = envelope.revision;
      savedMedia = partitionMobileDraftMedia(hydrated.form, envelope.revision).media;
      return { ...hydrated, canvasIntent: envelope.canvasIntent, error: "" };
    } catch {
      return {
        error: "The saved draft is unreadable. Reset it to start a new print.",
        missing: [],
      };
    }
  }

  function save(form: GenerateForm, canvasIntent: CanvasIntent): Promise<boolean> {
    const ticket = ++epoch;
    const revision = persistence.revision();
    const partition = partitionMobileDraftMedia(form, revision);
    const defaults = newGenerateForm();
    const metadata = Object.fromEntries(
      Object.entries(partition.form).filter(
        ([key]) => Object.hasOwn(defaults, key) && !DERIVED.has(key),
      ),
    );
    const operation = tail
      .catch(() => {})
      .then(async () => {
        if (ticket !== epoch) return false;
        // Typing a title/prompt must not rewrite a parked video on every edit.
        // Reuse the immutable media generation only when all slot bytes match.
        const sameMedia =
          savedRevision !== null &&
          savedMedia.length === partition.media.length &&
          savedMedia.every((record, index) => {
            const next = partition.media[index]!;
            return (
              record.base64 === next.base64 &&
              record.draftId?.split("/").slice(2).join("/") ===
                next.draftId?.split("/").slice(2).join("/")
            );
          });
        const committedRevision = sameMedia ? savedRevision! : revision;
        const envelope = JSON.stringify({
          version: 1,
          revision: committedRevision,
          canvasIntent,
          form: metadata,
        });
        try {
          if (!sameMedia && !(await persistence.write(partition.media))) {
            await persistence.remove(`${MOBILE_DRAFT_MEDIA_PREFIX}${revision}/`).catch(() => {});
            return false;
          }
          if (ticket !== epoch) {
            if (!sameMedia) await persistence.remove(`${MOBILE_DRAFT_MEDIA_PREFIX}${revision}/`);
            return false;
          }
          persistence.storage.setItem(MOBILE_COMPOSER_DRAFT_KEY, envelope);
          const previous = savedRevision;
          savedRevision = committedRevision;
          if (!sameMedia) savedMedia = partition.media;
          if (previous && previous !== committedRevision) {
            // Cleanup failure does not invalidate an already durable new draft.
            await persistence.remove(`${MOBILE_DRAFT_MEDIA_PREFIX}${previous}/`).catch(() => {});
          }
          return true;
        } catch {
          if (!sameMedia)
            await persistence.remove(`${MOBILE_DRAFT_MEDIA_PREFIX}${revision}/`).catch(() => {});
          return false;
        }
      });
    tail = operation;
    return operation;
  }

  function clear(): Promise<void> {
    ++epoch;
    const operation = tail
      .catch(() => {})
      .then(async () => {
        persistence.storage.removeItem(MOBILE_COMPOSER_DRAFT_KEY);
        savedRevision = null;
        savedMedia = [];
        await persistence.remove(MOBILE_DRAFT_MEDIA_PREFIX);
      });
    tail = operation;
    return operation;
  }
  return { restore, save, clear };
}
