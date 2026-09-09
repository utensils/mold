<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { storeToRefs } from "pinia";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import type {
  MeshWorkflowRequirements,
  MeshWorkflowRoute,
} from "../lib/meshWorkflowRouting";

import { apiFetchTo, apiJsonTo, type ApiTarget } from "../api/client";
import {
  cancelMeshWorkflow,
  type CreateMeshWorkflowRequest,
  createMeshWorkflow,
  deleteMeshWorkflow,
  getMeshWorkflow,
  listMeshWorkflows,
  resumeMeshWorkflow,
  type MeshWorkflowJobDetail,
  type MeshWorkflowJobSummary,
} from "../api/meshWorkflows";
import {
  buildMeshRoundtripWorkflow,
  buildMeshTextureWorkflow,
  buildTextToMeshWorkflow,
  isTextImageWorkflowModel,
  meshWorkflowModes,
  type WorkflowGenerateRequest,
  type WorkflowModel,
} from "../lib/meshWorkflowAuthoring";
import { autoGrowRows } from "../lib/autogrow";
import { isMeshFamily } from "../lib/legacyRecipeRules";
import { useMeshWorkflowDraftStore } from "../stores/meshWorkflowDraft";
import MeshViewer from "./MeshViewer.vue";
import {
  prepareReferenceUploads,
  requestShouldUseReferenceUploads,
  type ReferenceUploadCapabilities,
  type ReferenceUploadLease,
} from "../api/referenceUploads";

const props = defineProps<{
  target: ApiTarget;
  hostLabel?: string;
  desktop?: boolean;
  availableModels?: WorkflowModel[];
  /**
   * A workflow to open on, from the shell's own `?workflow=` deep link — the
   * queue row's route back here. Routing belongs to the shells, so this
   * component is handed the id rather than reading the router itself.
   */
  openWorkflow?: string | null;
  /**
   * The platform's spelling of the Generate chord, e.g. `⌘↩` or `Ctrl↩`.
   *
   * Passed in rather than hard-coded: `studio/` cannot read the shell's
   * platform table, and desktop ships Linux and Windows builds where New
   * image's composer says `Ctrl↩`. Absent renders no keycap, which is right
   * for web — it binds no keyboard Generate on any surface.
   */
  generateShortcut?: string | null;
  resolveTarget?: (
    requirements: MeshWorkflowRequirements,
  ) => Promise<MeshWorkflowRoute>;
}>();

const hostModels = ref<WorkflowModel[]>([]);
const models = computed(() => props.availableModels ?? hostModels.value);
const ownedTarget = ref<ApiTarget>({ ...props.target });
const ownerLabel = ref(props.hostLabel ?? "");
const jobs = ref<MeshWorkflowJobSummary[]>([]);
const detail = ref<MeshWorkflowJobDetail<WorkflowGenerateRequest> | null>(null);

/*
 * The draft lives in a store, not in this component: the router lazy-loads
 * this view and nothing keeps it alive, so every one of these used to be a
 * local `ref` that unmounted with the view — a trip to the Queue and back
 * landed on an empty form. `jobs`, `detail`, the epochs and the result URLs
 * stay local because they belong to this mount and are re-fetched on the next.
 */
const draft = useMeshWorkflowDraftStore();
const {
  mode,
  imageModelName,
  meshModelName,
  prompt,
  texture,
  textureResolution,
  delight,
  meshFile,
  appearanceFile,
  upAxis,
  metersPerUnit,
  selectedId,
} = storeToRefs(draft);
const busy = ref(false);
const loading = ref(true);
const error = ref("");
const resultSrc = ref("");
const resultPoster = ref("");
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let resultEpoch = 0;
let contextEpoch = 0;
let selectionEpoch = 0;
let loadedResult = "";

const meshModels = computed(() =>
  models.value.filter(
    (model) =>
      model.downloaded &&
      model.runtime_available !== false &&
      isMeshFamily(model.family) &&
      meshWorkflowModes(model).some((value) =>
        ["text_to_mesh", "mesh_roundtrip", "mesh_texture"].includes(value),
      ),
  ),
);
const imageModels = computed(() =>
  models.value.filter(isTextImageWorkflowModel),
);
const selectedMeshModel = computed(() =>
  meshModels.value.find((model) => model.name === meshModelName.value),
);
const selectedImageModel = computed(() =>
  imageModels.value.find((model) => model.name === imageModelName.value),
);
const selectedModes = computed(() =>
  selectedMeshModel.value ? meshWorkflowModes(selectedMeshModel.value) : [],
);
const textureAvailable = computed(() =>
  selectedModes.value.includes("mesh_texture"),
);
const delightAvailable = computed(() => {
  const profile = selectedMeshModel.value?.generation_profile;
  const recipe = profile?.recipes.find(
    (value) => value.id === profile.default_recipe_id,
  );
  return recipe?.capabilities.mesh?.delight?.mode === "adjustable";
});
const canSubmit = computed(() => {
  if (busy.value || !selectedMeshModel.value) return false;
  if (mode.value === "text_to_mesh") {
    return (
      selectedModes.value.includes("text_to_mesh") &&
      Boolean(selectedImageModel.value) &&
      Boolean(prompt.value.trim())
    );
  }
  return (
    selectedModes.value.includes(mode.value) &&
    meshFile.value !== null &&
    (mode.value !== "mesh_texture" || appearanceFile.value !== null) &&
    Number.isFinite(metersPerUnit.value) &&
    metersPerUnit.value > 0
  );
});
/**
 * What the empty canvas invites, in the words this workflow needs — the peer
 * of New image's `emptyCanvasGuidance`.
 */
/*
 * Plain word in sans, technical truth in mono, on the same row (README §1).
 * "Up axis" and "Metres per unit" are the file format's words, not a person's.
 * The Y/Z wording is `ui/components/MeshGeometryFields.vue`'s own, so the
 * import side and the export side name the same axes the same way.
 */
const textureSizeOptions = [
  { value: 1024, label: "1024" },
  { value: 2048, label: "2048" },
  { value: 4096, label: "4096" },
];
const upAxisOptions = [
  { value: "y" as const, label: "Y up" },
  { value: "z" as const, label: "Z up" },
];
const upAxisTruth = computed(() =>
  upAxis.value === "y"
    ? "Y-up · as stored (glTF, Blender OBJ)"
    : "Z-up · slicers, CAD, Blender STL/PLY",
);

const emptyCanvasGuidance = computed(() =>
  mode.value === "text_to_mesh"
    ? "Describe an object below, pick a 3-D style, and press Generate. Everything runs on your own machine."
    : mode.value === "mesh_roundtrip"
      ? "Choose a mesh in the settings, then press Generate. Everything runs on your own machine."
      : "Choose a mesh and the picture to paint it with, then press Generate.",
);

const settled = computed(() =>
  detail.value
    ? ["completed", "failed", "cancelled"].includes(detail.value.state)
    : true,
);

const composerPrompt = ref<HTMLTextAreaElement | null>(null);

/*
 * The description grows with what is typed, capped, exactly as New image's
 * does. Without it a `rows="1"` field scrolled inside one line — worse than
 * the `rows="5"` box in the rail it replaced.
 */
function growComposer(): void {
  if (composerPrompt.value) autoGrowRows(composerPrompt.value);
}
watch(prompt, () => void nextTick(growComposer));
onMounted(() => void nextTick(growComposer));

/** The primary-modifier Return from inside the description, the way every
 *  composer behaves. Both modifiers are accepted because this layer cannot
 *  read the shell's platform table; the visible keycap is the caller's. */
function onComposerKeydown(event: KeyboardEvent): void {
  if (event.key !== "Enter" || !(event.metaKey || event.ctrlKey)) return;
  event.preventDefault();
  if (canSubmit.value) void submit();
}

function modelLabel(model: WorkflowModel): string {
  return model.display_name?.trim() || model.name;
}

function stageLabel(kind: string): string {
  return (
    {
      image: "Generate source image",
      matting: "Remove background",
      delight: "Remove baked lighting",
      shape: "Build geometry",
      paint: "Paint PBR materials",
      finalize: "Publish mesh",
    }[kind] ?? kind
  );
}

function clearPoll(): void {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
}

function schedulePoll(): void {
  clearPoll();
  if (
    !selectedId.value ||
    !detail.value ||
    !["queued", "running"].includes(detail.value.state)
  )
    return;
  pollTimer = setTimeout(() => void refreshSelected(), 750);
}

async function refreshJobs(): Promise<void> {
  const target = ownedTarget.value;
  const epoch = contextEpoch;
  const listing = await listMeshWorkflows(target);
  if (epoch === contextEpoch && target === ownedTarget.value)
    jobs.value = listing.jobs;
}

async function refreshSelected(restoreDraft = false): Promise<void> {
  clearPoll();
  const epoch = ++selectionEpoch;
  const id = selectedId.value;
  if (!id) {
    detail.value = null;
    revokeResult();
    return;
  }
  const target = ownedTarget.value;
  try {
    const result = await getMeshWorkflow<WorkflowGenerateRequest>(target, id);
    if (epoch !== selectionEpoch) return;
    detail.value = result;
    if (restoreDraft && result?.request) restoreWorkflowDraft(result.request);
    await loadResult();
  } catch (cause) {
    if (epoch === selectionEpoch)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (epoch === selectionEpoch) schedulePoll();
  }
}

function restoreWorkflowDraft(
  request: CreateMeshWorkflowRequest<WorkflowGenerateRequest>,
): void {
  // History is a different request; never carry another workflow's attachments.
  meshFile.value = null;
  appearanceFile.value = null;
  mode.value = request.mode;
  const mesh =
    request.mode === "text_to_mesh"
      ? request.mesh_request
      : request.mode === "mesh_texture"
        ? request.texture_request
        : request.roundtrip_request;
  meshModelName.value = mesh.model;
  texture.value =
    request.mode === "mesh_texture" || mesh.mesh?.texture === true;
  textureResolution.value = mesh.mesh?.texture_resolution ?? 2048;
  delight.value = mesh.mesh?.delight === true;
  if (request.mode === "text_to_mesh") {
    prompt.value = request.image_request.prompt;
    imageModelName.value = request.image_request.model;
  }
}

function chooseModels(): void {
  if (!meshModels.value.some((model) => model.name === meshModelName.value)) {
    meshModelName.value =
      meshModels.value.find((model) =>
        meshWorkflowModes(model).includes("mesh_texture"),
      )?.name ??
      meshModels.value[0]?.name ??
      "";
  }
  if (!imageModels.value.some((model) => model.name === imageModelName.value))
    imageModelName.value = imageModels.value[0]?.name ?? "";
}

/** A machine's identity for the purpose of owning a workflow. */
function targetIdentity(target: ApiTarget): string {
  return `${target.baseUrl}\u0000${target.apiKey ?? ""}`;
}

async function bootstrap(): Promise<void> {
  const epoch = ++contextEpoch;
  ++selectionEpoch;
  clearPoll();
  revokeResult();
  /*
   * A deep link names the workflow to open on. Without one, the selection is
   * kept whenever this is the SAME machine — `bootstrap` also runs on every
   * mount, so clearing unconditionally meant a trip to the Queue and back
   * emptied the canvas even though the draft beneath it survived. A host
   * change does clear it: the workflow belongs to the machine that ran it.
   */
  const deepLink = props.openWorkflow?.trim() ?? "";
  const machine = targetIdentity(props.target);
  if (deepLink) draft.selectWorkflow(deepLink, machine);
  else if (!draft.selectionBelongsTo(machine)) draft.selectWorkflow("", "");
  detail.value = null;
  ownedTarget.value = { ...props.target };
  ownerLabel.value = props.hostLabel ?? "";
  const target = ownedTarget.value;
  loading.value = true;
  error.value = "";
  try {
    const [availableModels, listing] = await Promise.all([
      apiJsonTo<WorkflowModel[]>(target, "/api/models"),
      listMeshWorkflows(target),
    ]);
    if (epoch !== contextEpoch) return;
    hostModels.value = availableModels;
    jobs.value = listing.jobs;
    chooseModels();
    // `selectedId` was set before the fetch, so its watcher has already run
    // against an empty job list. Restore the deep-linked workflow's draft now
    // that the listing can name it, and say so plainly when it is gone.
    if (selectedId.value) {
      if (listing.jobs.some((job) => job.id === selectedId.value)) {
        /*
         * Restore the DRAFT only for a deep link. A returning view is showing
         * a workflow the person already has, and their unsent description and
         * attachments are the newer thing: `restoreWorkflowDraft` overwrites
         * the prompt, both styles, every stage setting and BOTH file wells, so
         * doing it on every entry reverted the draft to the finished
         * workflow's — the exact loss this PR exists to end, for the normal
         * case of having run something once.
         */
        await refreshSelected(Boolean(deepLink));
      } else {
        draft.selectWorkflow("", "");
        error.value = "That 3-D workflow is no longer on this machine.";
      }
    }
  } catch (cause) {
    if (epoch === contextEpoch)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (epoch === contextEpoch) loading.value = false;
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}

async function filePayload(
  file: File,
): Promise<{ base64: string; sha256: string }> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return {
    base64: bytesToBase64(bytes),
    sha256: Array.from(digest, (value) =>
      value.toString(16).padStart(2, "0"),
    ).join(""),
  };
}

async function submit(): Promise<void> {
  const meshModel = selectedMeshModel.value;
  if (!canSubmit.value || !meshModel) return;
  busy.value = true;
  error.value = "";
  let uploadLease: ReferenceUploadLease<WorkflowGenerateRequest> | null = null;
  const epoch = contextEpoch;
  const submitted = {
    mode: mode.value,
    prompt: prompt.value,
    texture: texture.value,
    textureResolution: textureResolution.value,
    delight: delight.value,
    delightAvailable: delightAvailable.value,
    textureAvailable: textureAvailable.value,
    selectedImageModel: selectedImageModel.value,
    meshFile: meshFile.value,
    appearanceFile: appearanceFile.value,
    upAxis: upAxis.value,
    metersPerUnit: metersPerUnit.value,
  };
  try {
    const route = props.resolveTarget
      ? await props.resolveTarget({
          mode: submitted.mode,
          meshModel: meshModel.name,
          ...(submitted.mode === "text_to_mesh"
            ? { imageModel: submitted.selectedImageModel!.name }
            : {}),
          texture:
            submitted.mode === "mesh_texture" ||
            (submitted.mode === "text_to_mesh" &&
              submitted.textureAvailable &&
              submitted.texture),
          delight:
            submitted.mode !== "mesh_roundtrip" &&
            submitted.delightAvailable &&
            submitted.delight,
        })
      : { target: { ...props.target }, label: props.hostLabel ?? "" };
    if (epoch !== contextEpoch) return;
    const submissionTarget = route.target;
    const [status, capabilities] = await Promise.all([
      apiJsonTo<{ instance_id: string }>(submissionTarget, "/api/status"),
      apiJsonTo<{ reference_uploads?: ReferenceUploadCapabilities | null }>(
        submissionTarget,
        "/api/capabilities",
      ),
    ]);
    if (epoch !== contextEpoch) return;
    const submissionUploads = capabilities.reference_uploads ?? null;
    let request =
      submitted.mode === "text_to_mesh"
        ? buildTextToMeshWorkflow({
            prompt: submitted.prompt,
            imageModel: submitted.selectedImageModel!,
            meshModel,
            texture: submitted.texture,
            textureResolution: submitted.textureResolution,
            delight: submitted.delightAvailable && submitted.delight,
          })
        : await (async () => {
            const mesh = submitted.meshFile!;
            const useUpload =
              submissionUploads?.available === true &&
              Boolean(submissionTarget.apiKey?.trim());
            const meshPayload = useUpload ? null : await filePayload(mesh);
            const meshFormat: "glb" | "obj" = mesh.name
              .toLowerCase()
              .endsWith(".obj")
              ? "obj"
              : "glb";
            const shared = {
              meshModel,
              ...(meshPayload ? { meshBase64: meshPayload.base64 } : {}),
              meshName: mesh.name,
              meshByteLength: mesh.size,
              ...(meshPayload ? { meshSha256: meshPayload.sha256 } : {}),
              meshFormat,
              upAxis: submitted.upAxis,
              metersPerUnit: submitted.metersPerUnit,
            };
            if (submitted.mode === "mesh_roundtrip") {
              return buildMeshRoundtripWorkflow(shared);
            }
            const appearancePayload = await filePayload(
              submitted.appearanceFile!,
            );
            return buildMeshTextureWorkflow({
              ...shared,
              appearanceBase64: appearancePayload.base64,
              textureResolution: submitted.textureResolution,
              delight: submitted.delightAvailable && submitted.delight,
            });
          })();
    const meshRequest =
      request.mode === "mesh_texture"
        ? request.texture_request
        : request.mode === "mesh_roundtrip"
          ? request.roundtrip_request
          : null;
    const directMeshUpload =
      meshRequest?.references?.[0]?.media.authority === "descriptor";
    if (
      meshRequest &&
      (directMeshUpload ||
        requestShouldUseReferenceUploads(
          meshRequest,
          submissionTarget,
          submissionUploads,
        ))
    ) {
      uploadLease = await prepareReferenceUploads({
        target: submissionTarget,
        expectedInstanceId: status.instance_id,
        capabilities: submissionUploads,
        request: meshRequest,
        ...(directMeshUpload
          ? { uploadBodies: new Map([[1, submitted.meshFile!]]) }
          : {}),
      });
      if (request.mode === "mesh_texture") {
        request = { ...request, texture_request: uploadLease.request };
      } else if (request.mode === "mesh_roundtrip") {
        request = { ...request, roundtrip_request: uploadLease.request };
      }
    }
    if (epoch !== contextEpoch) {
      await uploadLease?.cancel().catch(() => undefined);
      return;
    }
    const created = await createMeshWorkflow(submissionTarget, request);
    uploadLease = null;
    if (epoch !== contextEpoch) return;
    clearPoll();
    ++selectionEpoch;
    revokeResult();
    detail.value = null;
    ownedTarget.value = submissionTarget;
    ownerLabel.value = route.label;
    // The run belongs to the machine that accepted it — which under Auto or
    // Most capable is not necessarily the one being browsed.
    draft.selectWorkflow(created.job_id, targetIdentity(submissionTarget));
    await refreshJobs();
  } catch (cause) {
    await uploadLease?.cancel().catch(() => undefined);
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}

async function cancel(): Promise<void> {
  if (!selectedId.value) return;
  busy.value = true;
  try {
    await cancelMeshWorkflow(ownedTarget.value, selectedId.value);
    await Promise.all([refreshJobs(), refreshSelected()]);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}

async function resume(): Promise<void> {
  if (!selectedId.value) return;
  busy.value = true;
  try {
    await resumeMeshWorkflow(ownedTarget.value, selectedId.value);
    await Promise.all([refreshJobs(), refreshSelected()]);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}

async function remove(): Promise<void> {
  if (!selectedId.value || !settled.value) return;
  busy.value = true;
  error.value = "";
  try {
    await deleteMeshWorkflow(ownedTarget.value, selectedId.value);
    draft.selectWorkflow("", "");
    detail.value = null;
    revokeResult();
    await refreshJobs();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}

function revokeResult(): void {
  ++resultEpoch;
  loadedResult = "";
  if (resultSrc.value) URL.revokeObjectURL(resultSrc.value);
  if (resultPoster.value) URL.revokeObjectURL(resultPoster.value);
  resultSrc.value = "";
  resultPoster.value = "";
}

async function loadResult(): Promise<void> {
  const filename = detail.value?.output_filename;
  if (!filename) {
    revokeResult();
    return;
  }
  const target = ownedTarget.value;
  const identity = JSON.stringify([target.baseUrl, target.apiKey, filename]);
  if (loadedResult === identity && resultSrc.value) return;
  const epoch = ++resultEpoch;
  const media = await apiFetchTo(
    target,
    `/api/gallery/image/${encodeURIComponent(filename)}`,
  );
  const poster = await apiFetchTo(
    target,
    `/api/gallery/thumbnail/${encodeURIComponent(filename)}`,
  ).catch(() => null);
  if (!media.ok) throw new Error("Could not load the 3-D result.");
  const blob = await media.blob();
  const posterBlob = poster?.ok ? await poster.blob() : null;
  if (epoch !== resultEpoch) return;
  revokeResult();
  loadedResult = identity;
  resultSrc.value = URL.createObjectURL(blob);
  if (posterBlob) resultPoster.value = URL.createObjectURL(posterBlob);
}

const modeOptions = computed(() =>
  [
    { value: "text_to_mesh" as const, label: "From words" },
    { value: "mesh_roundtrip" as const, label: "Rebuild" },
    { value: "mesh_texture" as const, label: "Add texture" },
  ].filter((option) =>
    meshModels.value.some((model) =>
      meshWorkflowModes(model).includes(option.value),
    ),
  ),
);
watch(mode, (value) => {
  if (!selectedModes.value.includes(value))
    meshModelName.value =
      meshModels.value.find((model) => meshWorkflowModes(model).includes(value))
        ?.name ?? "";
});
watch(models, chooseModels);

watch(meshModelName, () => {
  if (!selectedModes.value.includes(mode.value)) {
    mode.value = selectedModes.value.includes("text_to_mesh")
      ? "text_to_mesh"
      : selectedModes.value.includes("mesh_roundtrip")
        ? "mesh_roundtrip"
        : "mesh_texture";
  }
  if (!textureAvailable.value) texture.value = false;
});
watch(selectedId, () => {
  revokeResult();
  void refreshSelected(true);
});
watch(
  [() => props.target.baseUrl, () => props.target.apiKey],
  () => void bootstrap(),
);
watch(
  () => props.openWorkflow,
  (value) => {
    const id = value?.trim() ?? "";
    if (id && id !== selectedId.value)
      draft.selectWorkflow(id, targetIdentity(ownedTarget.value));
  },
);

/** The shell's ⌘↩ — the same Generate the composer's button runs. */
defineExpose({ generate: () => void submit() });

/*
 * Persist the scalars the moment they settle rather than on unmount: the
 * webview can be closed or reloaded without an unmount hook ever running, and
 * a draft that only survives a graceful exit is not a draft.
 */
watch(
  [
    mode,
    meshModelName,
    imageModelName,
    prompt,
    texture,
    textureResolution,
    delight,
    upAxis,
    metersPerUnit,
  ],
  () => draft.persist(),
);

onMounted(() => void bootstrap());
onBeforeUnmount(() => {
  ++contextEpoch;
  ++selectionEpoch;
  clearPoll();
  revokeResult();
});
</script>

<template>
  <section
    class="mesh-studio"
    :class="{ 'mesh-studio--desktop': desktop }"
    aria-label="3-D workflow studio"
  >
    <header class="mesh-studio__header">
      <SegmentedControl
        v-if="desktop"
        v-model="mode"
        :options="modeOptions"
        :disabled="busy || loading"
        label="3-D workflow"
        variant="neutral"
        compact
      />
      <div v-else>
        <p class="mesh-studio__eyebrow">Hunyuan3D workflow</p>
        <h1>Build and texture a 3-D object</h1>
        <p>
          Every stage is durable. Close the app, restart the engine, and resume
          from the checkpoint.
        </p>
      </div>
      <div class="mesh-studio__header-actions">
        <span v-if="ownerLabel && selectedId" class="mesh-studio__owner">{{
          ownerLabel
        }}</span>
        <select
          v-model="selectedId"
          :disabled="busy"
          aria-label="Previous 3-D workflow"
        >
          <option value="">New workflow</option>
          <option v-for="job in jobs" :key="job.id" :value="job.id">
            {{
              job.mode === "text_to_mesh"
                ? "Text to 3-D"
                : job.mode === "mesh_roundtrip"
                  ? "Rebuild mesh"
                  : "Texture mesh"
            }}
            ·
            {{ job.state }}
          </option>
        </select>
        <slot name="machine" :busy="busy" />
      </div>
    </header>

    <p v-if="error" class="mesh-studio__error" role="alert">{{ error }}</p>
    <p v-if="loading">Loading workflow capabilities…</p>

    <div v-else class="mesh-studio__grid">
      <slot name="inspector-resize" />
      <form class="mesh-studio__composer" @submit.prevent="submit">
        <fieldset class="mesh-studio__fields" :disabled="busy">
          <div v-if="desktop" class="mesh-studio__inspector-heading">
            Settings
          </div>
          <SegmentedControl
            v-if="!desktop"
            v-model="mode"
            :options="modeOptions"
            :disabled="busy || loading"
            label="3-D workflow"
            variant="neutral"
          />

          <slot
            v-if="!desktop"
            name="mesh-picker"
            :models="
              meshModels.filter((model) =>
                meshWorkflowModes(model).includes(mode),
              )
            "
            :selected="meshModelName"
            :select="(name: string) => (meshModelName = name)"
            :disabled="busy"
          >
            <label>
              3-D style
              <select v-model="meshModelName" data-test="mesh-workflow-model">
                <option
                  v-for="model in meshModels.filter((model) =>
                    meshWorkflowModes(model).includes(mode),
                  )"
                  :key="model.name"
                  :value="model.name"
                >
                  {{ modelLabel(model) }}
                </option>
              </select>
            </label>
          </slot>

          <template v-if="mode === 'text_to_mesh'">
            <template v-if="!desktop">
              <label>
                Describe the object
                <textarea
                  v-model="prompt"
                  rows="5"
                  placeholder="A hand-carved wooden fox, centered on a plain background"
                />
              </label>
              <slot
                name="image-picker"
                :models="imageModels"
                :selected="imageModelName"
                :select="(name: string) => (imageModelName = name)"
                :disabled="busy"
              >
                <label>
                  Picture style
                  <select v-model="imageModelName">
                    <option
                      v-for="model in imageModels"
                      :key="model.name"
                      :value="model.name"
                    >
                      {{ modelLabel(model) }}
                    </option>
                  </select>
                </label>
              </slot>
            </template>
            <div
              v-if="textureAvailable"
              class="mesh-studio__check"
              data-test="mesh-workflow-texture"
            >
              <SwitchToggle
                v-model="texture"
                :disabled="busy"
                label="Paint PBR materials after geometry"
              />
              <span class="mesh-studio__check-label" aria-hidden="true"
                >Paint PBR materials after geometry</span
              >
            </div>
            <p
              v-else
              class="mesh-studio__availability"
              data-test="mesh-workflow-texture-unavailable"
            >
              PBR painting is unavailable on this machine. Geometry generation
              remains available.
            </p>
          </template>

          <template v-else>
            <label class="mesh-studio__file">
              Source mesh (GLB or OBJ)
              <input
                type="file"
                accept=".glb,.obj,model/gltf-binary,model/obj"
                @change="
                  meshFile =
                    ($event.target as HTMLInputElement).files?.[0] ?? null
                "
              />
              <span>{{ meshFile?.name || "Choose a mesh" }}</span>
            </label>
            <label v-if="mode === 'mesh_texture'" class="mesh-studio__file">
              Appearance image
              <input
                type="file"
                accept="image/png,image/jpeg"
                @change="
                  appearanceFile =
                    ($event.target as HTMLInputElement).files?.[0] ?? null
                "
              />
              <span>{{ appearanceFile?.name || "Choose an image" }}</span>
            </label>
            <div class="mesh-studio__field">
              <span class="ms-group-label mesh-studio__label"
                >Which way is up</span
              >
              <SegmentedControl
                v-if="desktop"
                v-model="upAxis"
                :options="upAxisOptions"
                :disabled="busy"
                label="Which way is up"
                variant="neutral"
                compact
              />
              <select v-else v-model="upAxis" aria-label="Which way is up">
                <option value="y">Y up</option>
                <option value="z">Z up</option>
              </select>
              <p class="mesh-studio__truth">{{ upAxisTruth }}</p>
            </div>
            <div class="mesh-studio__field">
              <span class="ms-group-label mesh-studio__label"
                >How big one unit is</span
              >
              <input
                v-model.number="metersPerUnit"
                class="mesh-studio__number"
                type="number"
                min="0.000001"
                max="1000000"
                step="any"
                aria-label="How big one unit is"
              />
              <p class="mesh-studio__truth">
                {{ metersPerUnit }} m per unit · what the file calls 1
              </p>
            </div>
          </template>

          <div
            v-if="
              (mode === 'text_to_mesh' && texture) || mode === 'mesh_texture'
            "
            class="mesh-studio__field"
          >
            <span class="ms-group-label mesh-studio__label">Texture size</span>
            <SegmentedControl
              v-if="desktop"
              v-model="textureResolution"
              :options="textureSizeOptions"
              :disabled="busy"
              label="Texture size"
              variant="neutral"
              compact
            />
            <select
              v-else
              v-model.number="textureResolution"
              aria-label="Texture size"
            >
              <option :value="1024">1024</option>
              <option :value="2048">2048</option>
              <option :value="4096">4096</option>
            </select>
            <p class="mesh-studio__truth">
              {{ textureResolution }} px · bigger takes longer to paint
            </p>
          </div>
          <div
            v-if="delightAvailable && mode !== 'mesh_roundtrip'"
            class="mesh-studio__check"
          >
            <SwitchToggle
              v-model="delight"
              :disabled="busy"
              label="Remove baked lighting and highlights before building the mesh"
            />
            <span class="mesh-studio__check-label" aria-hidden="true"
              >Remove baked lighting and highlights before building the
              mesh</span
            >
          </div>
          <button
            v-if="!desktop"
            class="mesh-studio__primary"
            type="submit"
            :disabled="!canSubmit"
          >
            {{
              busy
                ? "Preparing…"
                : mode === "text_to_mesh"
                  ? "Build 3-D object"
                  : mode === "mesh_roundtrip"
                    ? "Rebuild mesh"
                    : "Paint mesh"
            }}
          </button>
        </fieldset>
      </form>

      <div class="mesh-studio__main">
        <article class="mesh-studio__result">
          <MeshViewer
            v-if="resultSrc"
            :src="resultSrc"
            :poster="resultPoster"
            auto-rotate
            expandable
            alt="Generated 3-D object"
          />
          <div v-else-if="detail" class="mesh-studio__progress">
            <h2>
              {{
                detail.state === "completed"
                  ? "Publishing result…"
                  : "Workflow progress"
              }}
            </h2>
            <ol>
              <li
                v-for="stage in detail.stages"
                :key="stage.index"
                :data-state="stage.state"
              >
                <span class="mesh-studio__dot" />
                <span>{{ stageLabel(stage.kind) }}</span>
                <strong>{{ stage.state }}</strong>
              </li>
            </ol>
            <p v-if="detail.error" class="mesh-studio__error">
              {{ detail.error }}
            </p>
            <div class="mesh-studio__actions">
              <button
                v-if="['queued', 'running'].includes(detail.state)"
                type="button"
                :disabled="busy"
                @click="cancel"
              >
                Cancel
              </button>
              <button
                v-if="['paused', 'failed'].includes(detail.state)"
                type="button"
                :disabled="busy"
                @click="resume"
              >
                Resume
              </button>
              <button
                v-if="settled"
                type="button"
                :disabled="busy"
                @click="remove"
              >
                Delete workflow data
              </button>
            </div>
          </div>
          <div v-else class="mesh-studio__empty">
            <EmptyStateBlock
              v-if="desktop"
              data-test="mesh-empty-canvas"
              brand
              icon="layers"
              headline="Your 3-D object appears here"
              :guidance="emptyCanvasGuidance"
            />
            <template v-else>
              <strong>Your 3-D object appears here</strong>
              <span>Choose a workflow and its inputs to begin.</span>
            </template>
          </div>
        </article>

        <!--
        The composer, exactly where New image puts it (README §3): the
        description, the chips that pick the recipe, and one accent Generate
        carrying the shortcut the shell actually honours here. It used to sit
        at the foot of the inspector, which is what made this surface read as
        a form dropped into the shell rather than a view of the app.
      -->
        <form
          v-if="desktop"
          class="ms-composer mesh-studio__bar"
          data-test="mesh-composer"
          @submit.prevent="submit"
        >
          <div class="ms-composer__card">
            <div v-if="mode === 'text_to_mesh'" class="ms-composer__prompt-row">
              <textarea
                v-model="prompt"
                data-selectable
                ref="composerPrompt"
                rows="1"
                aria-label="Describe the object"
                placeholder="Describe the object — “a hand-carved wooden fox, on a plain background”"
                class="ms-composer__input"
                :disabled="busy"
                @keydown="onComposerKeydown"
              />
            </div>
            <div class="ms-composer__controls">
              <slot
                name="mesh-picker"
                :models="
                  meshModels.filter((model) =>
                    meshWorkflowModes(model).includes(mode),
                  )
                "
                :selected="meshModelName"
                :select="(name: string) => (meshModelName = name)"
                :disabled="busy"
              />
              <slot
                v-if="mode === 'text_to_mesh'"
                name="image-picker"
                :models="imageModels"
                :selected="imageModelName"
                :select="(name: string) => (imageModelName = name)"
                :disabled="busy"
              />
              <span class="ms-composer__spacer" />
              <button
                class="ms-composer__generate"
                data-test="mesh-generate"
                type="submit"
                :disabled="!canSubmit"
              >
                {{ busy ? "Preparing…" : "Generate" }}
                <kbd v-if="generateShortcut" class="ms-composer__key">{{
                  generateShortcut
                }}</kbd>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </section>
</template>

<style scoped>
.mesh-studio {
  height: 100%;
  overflow: auto;
  padding: 28px;
  color: var(--mold-text);
  background: var(--mold-bg);
}
.mesh-studio__header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  max-width: 1280px;
  margin: 0 auto 24px;
}
.mesh-studio__header h1 {
  margin: 3px 0 6px;
  font-size: var(--mold-fs-xl);
  font-weight: 700;
}
.mesh-studio__header p {
  margin: 0;
  color: var(--mold-text-2);
}
.mesh-studio__header-actions {
  display: grid;
  gap: 8px;
  min-width: min(100%, 320px);
}
.mesh-studio__eyebrow {
  font: 700 var(--mold-fs-micro) var(--mold-font-mono);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-blue) !important;
}
.mesh-studio__grid {
  display: grid;
  grid-template-columns: minmax(300px, 390px) minmax(420px, 1fr);
  gap: 20px;
  max-width: 1280px;
  min-height: 620px;
  margin: auto;
}
.mesh-studio__composer,
.mesh-studio__result {
  border: 1px solid var(--mold-border);
  background: var(--mold-surface);
  padding: 20px;
}
.mesh-studio__fields {
  display: contents;
}
.mesh-studio__composer {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.mesh-studio label,
.mesh-studio__check {
  display: grid;
  gap: 7px;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
}
/*
 * The rail's own fields. Every selector here EXCLUDES the composer's input:
 * an SFC's scoped style is unlayered and `ui/kit.css` sits in
 * `@layer components`, so an unlayered element selector outranks the kit
 * whatever the specificity (`.claude/rules/desktop.md`, invariant 1). Without
 * the exclusion the composer's description rendered as a bordered, opaque,
 * resizable box with a native grip inside the composer card — nothing like
 * New image's, which is the whole point of having one.
 */
.mesh-studio select:not(.ms-composer__input),
.mesh-studio textarea:not(.ms-composer__input),
.mesh-studio input[type="number"]:not(.ms-composer__input) {
  width: 100%;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-1);
  background: var(--mold-bg);
  color: var(--mold-text);
  padding: 10px;
}
.mesh-studio textarea:not(.ms-composer__input) {
  resize: vertical;
  line-height: 1.45;
}
.mesh-studio__actions button {
  padding: 9px;
  color: var(--mold-text-2);
}
.mesh-studio__check {
  display: flex !important;
  align-items: center;
  gap: 9px !important;
}
.mesh-studio__availability {
  margin: 0;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  line-height: 1.45;
}
.mesh-studio__file input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}
.mesh-studio__file span {
  border: 1px dashed var(--mold-border);
  padding: 14px;
  color: var(--mold-text);
  cursor: pointer;
}
.mesh-studio__primary {
  margin-top: auto;
  border-radius: var(--mold-radius-1);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  padding: 12px 16px;
  font-weight: 750;
}
.mesh-studio__primary:disabled {
  opacity: 0.4;
}
.mesh-studio__result {
  display: grid;
  place-items: stretch;
  min-height: 560px;
}
.mesh-studio__result :deep(.mesh-viewer) {
  min-height: 520px;
}
.mesh-studio__empty,
.mesh-studio__progress {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
  max-width: 560px;
  width: 100%;
  margin: auto;
}
.mesh-studio__empty {
  align-items: center;
  color: var(--mold-text-2);
}
.mesh-studio__empty strong {
  color: var(--mold-text);
  font-size: var(--mold-fs-lg);
}
.mesh-studio__progress h2 {
  margin-bottom: 12px;
  font-size: var(--mold-fs-lg);
}
.mesh-studio__progress ol {
  display: grid;
  gap: 4px;
}
.mesh-studio__progress li {
  display: grid;
  grid-template-columns: 12px 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 10px 0;
  color: var(--mold-text-2);
}
.mesh-studio__progress strong {
  font: 600 var(--mold-fs-micro) var(--mold-font-mono);
  text-transform: uppercase;
}
.mesh-studio__dot {
  width: 8px;
  height: 8px;
  border: 1px solid var(--mold-text-3);
  border-radius: 50%;
}
.mesh-studio__progress li[data-state="running"] .mesh-studio__dot {
  border-color: var(--mold-blue);
  background: var(--mold-blue);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--mold-blue) 18%, transparent);
}
.mesh-studio__progress li[data-state="completed"] .mesh-studio__dot {
  border-color: var(--mold-success);
  background: var(--mold-success);
}
.mesh-studio__progress li[data-state="failed"] .mesh-studio__dot {
  border-color: var(--mold-error);
  background: var(--mold-error);
}
.mesh-studio__actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
.mesh-studio__error {
  max-width: 1280px;
  margin: 0 auto 16px;
  border: 1px solid color-mix(in srgb, var(--mold-error) 45%, transparent);
  background: color-mix(in srgb, var(--mold-error) 9%, transparent);
  padding: 10px 12px;
  color: var(--mold-error);
}
@media (max-width: 820px) {
  .mesh-studio {
    padding: 16px;
  }
  .mesh-studio__header {
    align-items: start;
    flex-direction: column;
  }
  .mesh-studio__grid {
    grid-template-columns: 1fr;
  }
  .mesh-studio__result {
    min-height: 440px;
  }
}
/* Desktop shares the 40px view toolbar and the standard right inspector. */
.mesh-studio--desktop {
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
  min-width: 0;
  background: var(--mold-canvas);
}
.mesh-studio--desktop .mesh-studio__header {
  height: var(--mold-shell-viewbar-h, 40px);
  flex: 0 0 var(--mold-shell-viewbar-h, 40px);
  align-items: center;
  flex-direction: row;
  gap: 10px;
  max-width: none;
  width: 100%;
  margin: 0;
  padding: 0 14px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  /* The shell's chrome plane, the same one CreateHeader sits on — the crust
   * is the tile bed and read a shade off beside the New image toolbar. */
  background: var(--mold-chrome);
}

/* The canvas column: the result takes the height and the composer sits on its
 * bottom edge, both inside the grid's first column so the inspector keeps its
 * own scroll (README §"3-D Studio workflows"). */
.mesh-studio--desktop .mesh-studio__main {
  display: flex;
  flex-direction: column;
  min-width: 0;
  /*
   * A floor, bound from the shell's one layout authority (the desktop view
   * passes `--mesh-canvas-floor` from `benchLayout`'s `MIN_CANVAS_HEIGHT`).
   * The composer's style menus open UPWARD in place, so without a guaranteed
   * canvas the top of a long style list is silently cut by the view toolbar —
   * which is the same reason New image binds its canvas floor rather than
   * hard-coding one beside the layout.
   */
  min-height: var(--mesh-canvas-floor, 320px);
  overflow: hidden;
}
.mesh-studio--desktop .mesh-studio__bar {
  flex-shrink: 0;
}
.mesh-studio--desktop .mesh-studio__empty {
  display: grid;
  place-content: center;
  height: 100%;
}
.mesh-studio--desktop .mesh-studio__header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  margin-left: auto;
}
.mesh-studio--desktop .mesh-studio__header-actions select {
  width: auto;
  max-width: 200px;
}
.mesh-studio--desktop .mesh-studio__grid {
  flex: 1;
  min-height: 0;
  width: 100%;
  max-width: none;
  grid-template-columns: minmax(0, 1fr) var(
      --mesh-inspector-width,
      var(--mold-shell-inspector-w, 300px)
    );
  position: relative;
  gap: 0;
  margin: 0;
}
.mesh-studio--desktop .mesh-studio__composer {
  grid-column: 2;
  grid-row: 1;
  min-height: 0;
  overflow-y: auto;
  gap: 16px;
  padding: 16px;
  border: 0;
  border-left: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-deep);
}
/* The canvas column owns the grid cell; the result takes the height inside it
 * and the composer sits on its bottom edge. */
.mesh-studio--desktop .mesh-studio__main {
  grid-column: 1;
  grid-row: 1;
}
.mesh-studio--desktop .mesh-studio__result {
  flex: 1;
  min-width: 0;
  min-height: 0;
  padding: 24px;
  border: 0;
  background: var(--mold-canvas);
  /*
   * Fixed chrome never shrinks and the canvas absorbs the slack (README §3) —
   * that is what the composer's `flex-shrink: 0` above is for, and it is why
   * the whole surface no longer scrolls away.
   *
   * The canvas itself still SCROLLS, though: `hidden` clipped the stage list
   * at both ends on a short window (it is `justify-content: center`), which
   * put Cancel and Resume out of reach with no scrollbar to find them. The
   * viewer sets its own `height: 100%`, so this never adds a bar for a result.
   */
  overflow: auto;
}
.mesh-studio--desktop .mesh-studio__result :deep(.mesh-viewer) {
  min-height: 0;
  height: 100%;
}
.mesh-studio--desktop select,
.mesh-studio--desktop input[type="number"] {
  height: var(--mold-ctl-md);
  padding: 0 8px;
  font-size: var(--mold-fs-xs);
}
.mesh-studio--desktop textarea:not(.ms-composer__input) {
  padding: 8px;
  font-size: var(--mold-fs-sm);
}
.mesh-studio__owner {
  font: var(--mold-fs-micro) var(--mold-font-mono);
  color: var(--mold-text-2);
}
.mesh-studio--desktop .mesh-studio__error {
  margin: 8px 12px;
  flex-shrink: 0;
}
@media (max-width: 760px) {
  .mesh-studio--desktop .mesh-studio__header {
    height: auto;
    min-height: 40px;
    flex-wrap: wrap;
    padding: 8px;
  }
  .mesh-studio--desktop .mesh-studio__header-actions {
    flex-wrap: wrap;
    gap: 8px;
  }
  .mesh-studio--desktop .mesh-studio__header-actions select {
    max-width: 150px;
  }
}

/* Exactly one view toolbar tall, so its bottom rule lands on the toolbar's —
 * the same metric `InspectorPanel`'s tab strip binds. It also has to be
 * STICKY: it lives inside the scrolling settings form, so it used to slide up
 * out of the panel on the first scroll and leave the rail unlabelled. */
.mesh-studio--desktop .mesh-studio__inspector-heading {
  position: sticky;
  top: 0;
  z-index: 1;
  height: var(--mold-shell-viewbar-h, 40px);
  min-height: var(--mold-shell-viewbar-h, 40px);
  display: flex;
  align-items: center;
  margin: -16px -16px 4px;
  padding: 0 14px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  font-weight: 600;
}

/*
 * Field rhythm: the group label, the control, then one line of mono truth
 * beneath it (README §1).
 *
 * These are NOT desktop-only. The markup they replaced was a `<label>`, which
 * picked up `.mesh-studio label`'s grid, gap, colour and size — so scoping the
 * replacements to `--desktop` left web with bare `<div>`s and an unstyled
 * truth line. Only the uppercase is desktop's presentation.
 */
.mesh-studio__field {
  display: grid;
  gap: 7px;
  min-width: 0;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
}
.mesh-studio--desktop .mesh-studio__field {
  gap: 8px;
}
.mesh-studio--desktop .mesh-studio__label {
  text-transform: uppercase;
}
.mesh-studio__truth {
  margin: 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  font-weight: 400;
  color: var(--mold-text-dim);
}
.mesh-studio input[type="number"].mesh-studio__number {
  height: var(--mold-ctl-md, 26px);
  width: 12ch;
  padding: 0 8px;
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-1);
  background: var(--mold-bg);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.mesh-studio--desktop .mesh-studio__check {
  flex-direction: row-reverse;
  justify-content: space-between;
  line-height: var(--mold-lh-body);
}
</style>
