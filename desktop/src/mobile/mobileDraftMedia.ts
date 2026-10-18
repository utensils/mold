import type { DraftMediaRecord } from "@studio/lib/draftMediaStore";
import { cloneGenerateForm, type GenerateForm } from "../lib/generateForm";

export const MOBILE_DRAFT_MEDIA_PREFIX = "mold.mobile.composer/";

/** The shared submission clone JSON-copies H3 media. Draft saves must retain
 * those immutable strings by reference until the IndexedDB boundary. */
function cloneDraftForm(form: GenerateForm): GenerateForm {
  const { h3Authoring: state, ...withoutH3 } = form;
  const clone = cloneGenerateForm(withoutH3);
  if (state) {
    clone.h3Authoring = {
      firstFrame: state.firstFrame ? { ...state.firstFrame } : null,
      lastFrame: state.lastFrame ? { ...state.lastFrame } : null,
      references: state.references.map((draft) => ({
        ...draft,
        ...(draft.crop ? { crop: { ...draft.crop } } : {}),
        reference:
          draft.reference.kind === "fresh_video"
            ? { ...draft.reference, original: { ...draft.reference.original } }
            : {
                ...draft.reference,
                media: { ...draft.reference.media },
                ...(draft.reference.provenance
                  ? {
                      provenance: {
                        ...draft.reference.provenance,
                        ...(draft.reference.provenance.crop
                          ? { crop: { ...draft.reference.provenance.crop } }
                          : {}),
                      },
                    }
                  : {}),
              },
      })),
    };
  }
  return clone;
}

interface MediaSlot {
  name: string;
  bytes: string;
  replace: (bytes: string) => void;
}

/** Enumerate authored media, including parked wells, in a private form clone.
 * Empty payloads remain descriptors so a missing asset cannot disappear into
 * an apparently valid text-only draft. Never traverse arbitrary stored paths. */
function mediaSlots(form: GenerateForm): MediaSlot[] {
  const slots: MediaSlot[] = [];
  for (const name of ["sourceImage", "maskImage", "controlImage"] as const) {
    if (form[name] !== null) {
      slots.push({ name, bytes: form[name], replace: (bytes) => (form[name] = bytes) });
    }
  }
  form.imageAttachments.forEach((bytes, index) => {
    slots.push({
      name: `imageAttachments/${index}`,
      bytes,
      replace: (value) => (form.imageAttachments[index] = value),
    });
  });
  for (const name of [
    "endFrame",
    "identityImage",
    "sourceVideo",
    "extendVideo",
    "audioFile",
  ] as const) {
    const image = form[name];
    if (image) {
      slots.push({ name, bytes: image.base64, replace: (bytes) => (image.base64 = bytes) });
    }
  }
  form.keyframes.forEach(({ image }, index) => {
    slots.push({
      name: `keyframes/${index}`,
      bytes: image.base64,
      replace: (bytes) => (image.base64 = bytes),
    });
  });
  for (const role of ["front", "left", "back", "right"] as const) {
    const image = form.namedViews?.[role];
    if (image) {
      // cloneGenerateForm owns the map but its images are immutable references.
      const owned = { ...image };
      form.namedViews![role] = owned;
      slots.push({
        name: `namedViews/${role}`,
        bytes: owned.base64,
        replace: (bytes) => (owned.base64 = bytes),
      });
    }
  }
  for (const name of ["firstFrame", "lastFrame"] as const) {
    const image = form.h3Authoring?.[name];
    if (image) {
      slots.push({
        name: `h3Authoring/${name}`,
        bytes: image.data,
        replace: (bytes) => (image.data = bytes),
      });
    }
  }
  form.h3Authoring?.references.forEach(({ reference }, index) => {
    if (reference.kind === "fresh_video") return;
    slots.push({
      name: `h3Authoring/references/${index}`,
      bytes: reference.media.authority === "inline" ? reference.media.data : "",
      // Upload handles and server paths are request authorities, never durable
      // mobile media. Retain their descriptor and disclose the missing bytes.
      replace: (bytes) => {
        reference.media = bytes
          ? { authority: "inline", data: bytes }
          : { authority: "descriptor" };
      },
    });
  });
  return slots;
}

function mediaId(revision: string, slot: string): string {
  if (!/^[a-zA-Z0-9-]+$/.test(revision)) throw new Error("Invalid mobile draft revision");
  return `${MOBILE_DRAFT_MEDIA_PREFIX}${revision}/${slot}`;
}

/** No JSON clone before separation: large video/audio strings are not doubled
 * merely to remove them. The caller commits this entire batch before metadata. */
export function partitionMobileDraftMedia(form: GenerateForm, revision: string) {
  const metadata = cloneDraftForm(form);
  const media: DraftMediaRecord[] = [];
  for (const slot of mediaSlots(metadata)) {
    const draftId = mediaId(revision, slot.name);
    if (slot.bytes) media.push({ draftId, base64: slot.bytes });
    slot.replace("");
  }
  metadata.h3Authoring?.references.forEach((draft, index) => {
    if (draft.reference.kind !== "fresh_video") return;
    const { body, ...reference } = draft.reference;
    if (body)
      media.push({ draftId: mediaId(revision, `h3Authoring/references/${index}`), blob: body });
    draft.reference = reference;
  });
  return { form: metadata, media };
}

export async function hydrateMobileDraftMedia(
  metadata: GenerateForm,
  revision: string,
  read: (id: string) => Promise<DraftMediaRecord | null>,
): Promise<{ form: GenerateForm; missing: string[] }> {
  const form = cloneDraftForm(metadata);
  const missing: string[] = [];
  // Fixed slot order is also the order shown in recovery. A storage read
  // failure is a missing asset, not a reason to discard the authored draft.
  for (const slot of mediaSlots(form)) {
    let record: DraftMediaRecord | null = null;
    try {
      record = await read(mediaId(revision, slot.name));
    } catch {
      // Report through the same persistent recovery as evicted media.
    }
    if (typeof record?.base64 === "string" && record.base64) slot.replace(record.base64);
    else {
      slot.replace("");
      missing.push(slot.name);
    }
  }
  for (const [index, draft] of (form.h3Authoring?.references ?? []).entries()) {
    if (draft.reference.kind !== "fresh_video") continue;
    const slot = `h3Authoring/references/${index}`;
    const stored = await read(mediaId(revision, slot)).catch(() => null);
    if (stored?.blob instanceof Blob) draft.reference = { ...draft.reference, body: stored.blob };
    else missing.push(slot);
  }
  return { form, missing };
}

export function unavailableMobileDraftMedia(form: GenerateForm): string[] {
  return [
    ...mediaSlots(cloneDraftForm(form))
      .filter((slot) => !slot.bytes)
      .map((slot) => slot.name),
    ...(form.h3Authoring?.references ?? []).flatMap((draft, index) =>
      draft.reference.kind === "fresh_video" && !draft.reference.body
        ? [`h3Authoring/references/${index}`]
        : [],
    ),
  ];
}

/** Explicit recovery action: remove only the unavailable wells, preserving
 * every valid attachment and all authored settings. */
export function discardUnavailableMobileDraftMedia(form: GenerateForm): GenerateForm {
  const clone = cloneDraftForm(form);
  for (const name of ["sourceImage", "maskImage", "controlImage"] as const) {
    if (clone[name] === "") clone[name] = null;
  }
  if (!clone.sourceImage) {
    clone.sourceImageName = null;
    clone.sourceImageWidth = null;
    clone.sourceImageHeight = null;
  }
  clone.imageAttachments = clone.imageAttachments.filter(Boolean);
  for (const name of [
    "endFrame",
    "identityImage",
    "sourceVideo",
    "extendVideo",
    "audioFile",
  ] as const) {
    if (clone[name] && !clone[name].base64) clone[name] = null;
  }
  clone.keyframes = clone.keyframes.filter(({ image }) => !!image.base64);
  for (const role of ["front", "left", "back", "right"] as const) {
    if (clone.namedViews?.[role] && !clone.namedViews[role].base64) delete clone.namedViews[role];
  }
  for (const name of ["firstFrame", "lastFrame"] as const) {
    if (clone.h3Authoring?.[name] && !clone.h3Authoring[name].data) clone.h3Authoring[name] = null;
  }
  if (clone.h3Authoring)
    clone.h3Authoring.references = clone.h3Authoring.references.filter(({ reference }) =>
      reference.kind === "fresh_video"
        ? !!reference.body
        : reference.media.authority === "inline" && !!reference.media.data,
    );
  return clone;
}
