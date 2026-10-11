import { onBeforeUnmount, ref, watch, type Ref } from "vue";
import { useRouter } from "vue-router";
import { retainedSourceMediaContent } from "@studio/api/sourceMediaBrowser";
import {
  sourceMemberKind,
  sourceOwnerIdentity,
  type SourceMediaOwner,
  type SourceMediaSelection,
} from "@studio/lib/sourceMediaBrowser";
import { blobToBase64 } from "@studio/lib/base64";
import { normalizeInputImage } from "@studio/lib/inputImage";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { appendMinimaxH3PickedMediaReference } from "@studio/lib/minimaxH3PickedMedia";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { generationCapabilitiesForFamily } from "./generateCapabilities";
import { routeCreateDrop, applyCreateDrop } from "./createImageDrop";
import {
  getHost,
  originHost,
  ORIGIN_HOST_ID,
  listHosts,
  getGenerateTargetId,
  HOSTS_CHANGED_EVENT,
  GENERATE_TARGET_CHANGED_EVENT,
} from "./hostRegistry";
import { useGenerateForm } from "../composables/useGenerateForm";
import type {
  GalleryImage,
  GenerateFormState,
  ModelInfoExtended,
} from "../types";

export function webSourceOwner(item: GalleryImage): SourceMediaOwner | null {
  const hosted = item as GalleryImage & { hostId?: string; hostLabel?: string };
  const host =
    !hosted.hostId || hosted.hostId === ORIGIN_HOST_ID
      ? originHost()
      : getHost(hosted.hostId);
  if (!host) return null;
  return {
    key: `${host.id}:${item.filename}`,
    filename: item.filename,
    title: item.title || item.metadata.title || item.filename,
    hostLabel: hosted.hostLabel || host.name,
    target: {
      baseUrl: host.url,
      apiKey: host.apiKey ?? null,
      connectionMode: host.connectionMode,
    },
    ...(host.instanceId ? { instanceId: host.instanceId } : {}),
    ...(item.media_version ? { revision: item.media_version } : {}),
  };
}

/** Build against a detached draft, then commit only if nothing authored since the click changed. */
export async function attachSourceMedia(
  selection: SourceMediaSelection,
  state: Ref<GenerateFormState>,
  model: ModelInfoExtended | undefined,
  current: () => boolean,
  read: () => Promise<Blob> = () =>
    retainedSourceMediaContent(
      selection.owner.filename,
      selection.member,
      selection.owner.target,
    ),
) {
  const kind = sourceMemberKind(selection.member);
  if (!["image", "video", "audio"].includes(kind))
    throw new Error("This file type cannot be attached to a new creation.");
  const original = state.value;
  const staged = JSON.parse(JSON.stringify(original)) as GenerateFormState;
  const caps = generationCapabilitiesForFamily(
    model?.family ?? staged.modelFamily,
    staged.model,
    staged.pipeline,
    null,
    model?.source_image ?? staged.sourceImageCapability,
    effectiveGenerationRecipe(model, staged.pipeline),
  );
  const context = {
    plan: sourceMediaPlan(caps),
    referenceMax: caps.referenceImages?.max ?? null,
    refusalReason: null,
    identityVisible: false,
  };
  const destination =
    kind === "image" ? routeCreateDrop(staged, context) : null;
  if (destination && typeof destination !== "string")
    throw new Error(destination.refused);
  if (
    kind === "audio" &&
    context.plan.kind !== "h3-references" &&
    !caps.supportsAudioInput
  )
    throw new Error(
      "Choose a model that accepts source audio before reusing this file.",
    );
  if (
    kind === "video" &&
    context.plan.kind !== "h3-references" &&
    !caps.supportsSourceVideo
  )
    throw new Error(
      "Choose a model that accepts source video before reusing this file.",
    );
  let changed = false;
  const stop = watch(
    state,
    () => {
      changed = true;
    },
    { deep: true, flush: "sync" },
  );
  try {
    const blob = await read();
    const base64 = await blobToBase64(blob);
    if (kind === "image") {
      const prepared = await normalizeInputImage(
        base64,
        selection.member.role === "keyframes"
          ? "keyframe.png"
          : selection.member.display_name,
      );
      const dimensions = imageDimensionsFromBase64(prepared.base64);
      const failure = await applyCreateDrop(
        staged,
        destination as Exclude<typeof destination, null | object>,
        {
          kind: "upload",
          filename: prepared.filename,
          base64: prepared.base64,
          mime: prepared.mimeType,
          ...(dimensions ?? {}),
        },
        context,
      );
      if (failure) throw new Error(failure);
    } else if (context.plan.kind === "h3-references") {
      staged.h3Authoring = await appendMinimaxH3PickedMediaReference(
        staged.h3Authoring,
        new File([blob], selection.member.display_name, {
          type: kind === "video" ? "video/mp4" : "audio/wav",
        }),
      );
    } else {
      const value = {
        kind: "upload" as const,
        filename: selection.member.display_name,
        base64,
        mime: blob.type,
      };
      if (kind === "video") {
        staged.sourceVideo = value;
        staged.sourceVideoPath = "";
      } else {
        staged.audioFile = value;
        staged.audioFilePath = "";
      }
    }
    if (changed || state.value !== original || !current())
      throw new Error(
        "The source or composer changed while loading. Select the source again to keep your latest edits.",
      );
    stop();
    state.value = staged;
  } finally {
    stop();
  }
}

export function useSourceMedia(models: () => ModelInfoExtended[] = () => []) {
  const form = useGenerateForm(),
    router = useRouter();
  const busy = ref(false),
    error = ref("");
  let disposed = false,
    controller: AbortController | undefined;
  onBeforeUnmount(() => {
    disposed = true;
    controller?.abort();
  });
  async function use(
    selection: SourceMediaSelection,
    currentOwner: () => SourceMediaOwner | null = () => selection.owner,
  ): Promise<boolean> {
    if (busy.value) return false;
    busy.value = true;
    error.value = "";
    controller = new AbortController();
    const identity = sourceOwnerIdentity(selection.owner);
    // Keep authoring attached to the machine selection the user saw at click.
    // Events catch even a route edit which is changed back during the download.
    const routeSnapshot = () =>
      JSON.stringify({
        target: getGenerateTargetId(),
        hosts: [originHost(), ...listHosts()].map(
          ({ id, url, apiKey, instanceId }) => ({
            id,
            url,
            apiKey,
            instanceId,
          }),
        ),
      });
    const routes = routeSnapshot();
    let routesChanged = false;
    const changedRoute = () => {
      routesChanged = true;
    };
    window.addEventListener(HOSTS_CHANGED_EVENT, changedRoute);
    window.addEventListener(GENERATE_TARGET_CHANGED_EVENT, changedRoute);
    const current = () => {
      const owner = currentOwner();
      const host = [originHost(), ...listHosts()].find(
        (h) =>
          h.url === selection.owner.target.baseUrl &&
          (h.apiKey ?? null) === selection.owner.target.apiKey,
      );
      return (
        !disposed &&
        !routesChanged &&
        routeSnapshot() === routes &&
        !!owner &&
        !!host &&
        sourceOwnerIdentity(owner) === identity &&
        (!owner.instanceId || host.instanceId === owner.instanceId)
      );
    };
    try {
      await attachSourceMedia(
        selection,
        form.state,
        models().find((m) => m.name === form.state.value.model),
        current,
        () =>
          retainedSourceMediaContent(
            selection.owner.filename,
            selection.member,
            selection.owner.target,
            controller!.signal,
          ),
      );
      await router.push({ name: "create" });
      return true;
    } catch (failure) {
      if (!disposed)
        error.value =
          failure instanceof Error ? failure.message : String(failure);
      return false;
    } finally {
      window.removeEventListener(HOSTS_CHANGED_EVENT, changedRoute);
      window.removeEventListener(GENERATE_TARGET_CHANGED_EVENT, changedRoute);
      busy.value = false;
    }
  }
  return { use, busy, error };
}
