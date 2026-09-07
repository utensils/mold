<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { apiFetchTo, apiJsonTo, type ApiTarget } from "../api/client";
import {
  cancelMeshWorkflow,
  createMeshWorkflow,
  deleteMeshWorkflow,
  getMeshWorkflow,
  listMeshWorkflows,
  resumeMeshWorkflow,
  type MeshWorkflowJobDetail,
  type MeshWorkflowJobSummary,
} from "../api/meshWorkflows";
import {
  buildMeshTextureWorkflow,
  buildTextToMeshWorkflow,
  isTextImageWorkflowModel,
  meshWorkflowModes,
  type WorkflowGenerateRequest,
  type WorkflowModel,
} from "../lib/meshWorkflowAuthoring";
import MeshViewer from "./MeshViewer.vue";
import {
  prepareReferenceUploads,
  requestShouldUseReferenceUploads,
  type ReferenceUploadCapabilities,
  type ReferenceUploadLease,
} from "../api/referenceUploads";

const props = defineProps<{ target: ApiTarget }>();

const models = ref<WorkflowModel[]>([]);
const jobs = ref<MeshWorkflowJobSummary[]>([]);
const detail = ref<MeshWorkflowJobDetail<WorkflowGenerateRequest> | null>(null);
const selectedId = ref("");
const mode = ref<"text_to_mesh" | "mesh_texture">("text_to_mesh");
const imageModelName = ref("");
const meshModelName = ref("");
const prompt = ref("");
const texture = ref(true);
const textureResolution = ref(2048);
const delight = ref(false);
const meshFile = ref<File | null>(null);
const appearanceFile = ref<File | null>(null);
const upAxis = ref<"y" | "z">("y");
const metersPerUnit = ref(1);
const busy = ref(false);
const loading = ref(true);
const error = ref("");
const resultSrc = ref("");
const resultPoster = ref("");
const instanceId = ref("");
const referenceUploads = ref<ReferenceUploadCapabilities | null>(null);
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let resultEpoch = 0;

const meshModels = computed(() =>
  models.value.filter(
    (model) =>
      model.downloaded &&
      model.runtime_available !== false &&
      model.family === "hunyuan3d" &&
      meshWorkflowModes(model).some((value) =>
        ["text_to_mesh", "mesh_texture"].includes(value),
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
    selectedModes.value.includes("mesh_texture") &&
    meshFile.value !== null &&
    appearanceFile.value !== null &&
    Number.isFinite(metersPerUnit.value) &&
    metersPerUnit.value > 0
  );
});
const settled = computed(() =>
  detail.value
    ? ["completed", "failed", "cancelled"].includes(detail.value.state)
    : true,
);

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
  if (settled.value || !selectedId.value) return;
  pollTimer = setTimeout(() => void refreshSelected(), 750);
}

async function refreshJobs(): Promise<void> {
  const listing = await listMeshWorkflows(props.target);
  jobs.value = listing.jobs;
}

async function refreshSelected(): Promise<void> {
  if (!selectedId.value) return;
  try {
    detail.value = await getMeshWorkflow<WorkflowGenerateRequest>(
      props.target,
      selectedId.value,
    );
    await loadResult();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    schedulePoll();
  }
}

async function bootstrap(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    const [availableModels, status, capabilities] = await Promise.all([
      apiJsonTo<WorkflowModel[]>(props.target, "/api/models"),
      apiJsonTo<{ instance_id: string }>(props.target, "/api/status"),
      apiJsonTo<{ reference_uploads?: ReferenceUploadCapabilities | null }>(
        props.target,
        "/api/capabilities",
      ),
      refreshJobs(),
    ]);
    models.value = availableModels;
    instanceId.value = status.instance_id;
    referenceUploads.value = capabilities.reference_uploads ?? null;
    meshModelName.value =
      meshModels.value.find((model) =>
        meshWorkflowModes(model).includes("mesh_texture"),
      )?.name ??
      meshModels.value[0]?.name ??
      "";
    imageModelName.value = imageModels.value[0]?.name ?? "";
    selectedId.value = jobs.value[0]?.id ?? "";
    await refreshSelected();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    loading.value = false;
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
  try {
    let request =
      mode.value === "text_to_mesh"
        ? buildTextToMeshWorkflow({
            prompt: prompt.value,
            imageModel: selectedImageModel.value!,
            meshModel,
            texture: texture.value,
            textureResolution: textureResolution.value,
            delight: delightAvailable.value && delight.value,
          })
        : await (async () => {
            const mesh = meshFile.value!;
            const appearance = appearanceFile.value!;
            const useUpload =
              referenceUploads.value?.available === true &&
              Boolean(props.target.apiKey?.trim());
            const appearancePayload = await filePayload(appearance);
            const meshPayload = useUpload ? null : await filePayload(mesh);
            const meshFormat = mesh.name.toLowerCase().endsWith(".obj")
              ? "obj"
              : "glb";
            return buildMeshTextureWorkflow({
              meshModel,
              ...(meshPayload ? { meshBase64: meshPayload.base64 } : {}),
              meshName: mesh.name,
              meshByteLength: mesh.size,
              ...(meshPayload ? { meshSha256: meshPayload.sha256 } : {}),
              meshFormat,
              appearanceBase64: appearancePayload.base64,
              upAxis: upAxis.value,
              metersPerUnit: metersPerUnit.value,
              textureResolution: textureResolution.value,
              delight: delightAvailable.value && delight.value,
            });
          })();
    const directMeshUpload =
      request.mode === "mesh_texture" &&
      request.texture_request.references?.[0]?.media.authority === "descriptor";
    if (
      request.mode === "mesh_texture" &&
      (directMeshUpload ||
        requestShouldUseReferenceUploads(
          request.texture_request,
          props.target,
          referenceUploads.value,
        ))
    ) {
      uploadLease = await prepareReferenceUploads({
        target: props.target,
        expectedInstanceId: instanceId.value,
        capabilities: referenceUploads.value,
        request: request.texture_request,
        ...(directMeshUpload
          ? { uploadBodies: new Map([[1, meshFile.value!]]) }
          : {}),
      });
      request = { ...request, texture_request: uploadLease.request };
    }
    const created = await createMeshWorkflow(props.target, request);
    uploadLease = null;
    await refreshJobs();
    selectedId.value = created.job_id;
    await refreshSelected();
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
    await cancelMeshWorkflow(props.target, selectedId.value);
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
    await resumeMeshWorkflow(props.target, selectedId.value);
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
    await deleteMeshWorkflow(props.target, selectedId.value);
    selectedId.value = "";
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
  if (resultSrc.value) URL.revokeObjectURL(resultSrc.value);
  if (resultPoster.value) URL.revokeObjectURL(resultPoster.value);
  resultSrc.value = "";
  resultPoster.value = "";
}

async function loadResult(): Promise<void> {
  const filename = detail.value?.output_filename;
  const epoch = ++resultEpoch;
  if (!filename) {
    revokeResult();
    return;
  }
  const media = await apiFetchTo(
    props.target,
    `/api/gallery/image/${encodeURIComponent(filename)}`,
  );
  const poster = await apiFetchTo(
    props.target,
    `/api/gallery/thumbnail/${encodeURIComponent(filename)}`,
  ).catch(() => null);
  if (epoch !== resultEpoch) return;
  revokeResult();
  resultSrc.value = URL.createObjectURL(await media.blob());
  if (poster) resultPoster.value = URL.createObjectURL(await poster.blob());
}

watch(meshModelName, () => {
  if (!selectedModes.value.includes(mode.value)) {
    mode.value = selectedModes.value.includes("text_to_mesh")
      ? "text_to_mesh"
      : "mesh_texture";
  }
});
watch(selectedId, () => void refreshSelected());
watch(
  () => [props.target.baseUrl, props.target.apiKey],
  () => void bootstrap(),
);
onMounted(() => void bootstrap());
onBeforeUnmount(() => {
  clearPoll();
  revokeResult();
});
</script>

<template>
  <section class="mesh-studio" aria-label="3-D workflow studio">
    <header class="mesh-studio__header">
      <div>
        <p class="mesh-studio__eyebrow">Hunyuan3D workflow</p>
        <h1>Build and texture a 3-D object</h1>
        <p>
          Every stage is durable. Close the app, restart the engine, and resume
          from the checkpoint.
        </p>
      </div>
      <select v-model="selectedId" aria-label="Previous 3-D workflow">
        <option value="">New workflow</option>
        <option v-for="job in jobs" :key="job.id" :value="job.id">
          {{ job.mode === "text_to_mesh" ? "Text to 3-D" : "Texture mesh" }} ·
          {{ job.state }}
        </option>
      </select>
    </header>

    <p v-if="error" class="mesh-studio__error" role="alert">{{ error }}</p>
    <p v-if="loading">Loading workflow capabilities…</p>

    <div v-else class="mesh-studio__grid">
      <form class="mesh-studio__composer" @submit.prevent="submit">
        <div
          class="mesh-studio__tabs"
          role="tablist"
          aria-label="Workflow type"
        >
          <button
            type="button"
            :aria-selected="mode === 'text_to_mesh'"
            :class="{ active: mode === 'text_to_mesh' }"
            :disabled="!selectedModes.includes('text_to_mesh')"
            @click="mode = 'text_to_mesh'"
          >
            Text to 3-D
          </button>
          <button
            type="button"
            :aria-selected="mode === 'mesh_texture'"
            :class="{ active: mode === 'mesh_texture' }"
            :disabled="!selectedModes.includes('mesh_texture')"
            @click="mode = 'mesh_texture'"
          >
            Texture a mesh
          </button>
        </div>

        <label>
          3-D model
          <select v-model="meshModelName" data-test="mesh-workflow-model">
            <option
              v-for="model in meshModels"
              :key="model.name"
              :value="model.name"
            >
              {{ modelLabel(model) }}
            </option>
          </select>
        </label>

        <template v-if="mode === 'text_to_mesh'">
          <label>
            Describe the object
            <textarea
              v-model="prompt"
              rows="5"
              placeholder="A hand-carved wooden fox, centered on a plain background"
            />
          </label>
          <label>
            Image model
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
          <label class="mesh-studio__check">
            <input v-model="texture" type="checkbox" />
            Paint PBR materials after geometry
          </label>
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
          <label class="mesh-studio__file">
            Appearance image
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              @change="
                appearanceFile =
                  ($event.target as HTMLInputElement).files?.[0] ?? null
              "
            />
            <span>{{ appearanceFile?.name || "Choose an image" }}</span>
          </label>
          <div class="mesh-studio__row">
            <label>
              Up axis
              <select v-model="upAxis">
                <option value="y">Y up</option>
                <option value="z">Z up</option>
              </select>
            </label>
            <label>
              Metres per unit
              <input
                v-model.number="metersPerUnit"
                type="number"
                min="0.000001"
                max="1000000"
                step="any"
              />
            </label>
          </div>
        </template>

        <label v-if="texture || mode === 'mesh_texture'">
          Texture size
          <select v-model.number="textureResolution">
            <option :value="1024">1024</option>
            <option :value="2048">2048</option>
            <option :value="4096">4096</option>
          </select>
        </label>
        <label v-if="delightAvailable" class="mesh-studio__check">
          <input v-model="delight" type="checkbox" />
          Remove baked lighting and highlights before building the mesh
        </label>
        <button
          class="mesh-studio__primary"
          type="submit"
          :disabled="!canSubmit"
        >
          {{
            busy
              ? "Preparing…"
              : mode === "text_to_mesh"
                ? "Build 3-D object"
                : "Paint mesh"
          }}
        </button>
      </form>

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
          <strong>Your 3-D object appears here</strong>
          <span>Choose a workflow and its inputs to begin.</span>
        </div>
      </article>
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
.mesh-studio__eyebrow {
  font: 700 var(--mold-fs-micro) var(--mold-font-mono);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-accent) !important;
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
.mesh-studio__composer {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.mesh-studio label {
  display: grid;
  gap: 7px;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  font-weight: 650;
}
.mesh-studio select,
.mesh-studio textarea,
.mesh-studio input[type="number"] {
  width: 100%;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-1);
  background: var(--mold-bg);
  color: var(--mold-text);
  padding: 10px;
}
.mesh-studio textarea {
  resize: vertical;
  line-height: 1.45;
}
.mesh-studio__tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  padding: 4px;
  background: var(--mold-bg);
}
.mesh-studio__tabs button,
.mesh-studio__actions button {
  padding: 9px;
  color: var(--mold-text-2);
}
.mesh-studio__tabs button.active {
  background: var(--mold-surface-2);
  color: var(--mold-text);
}
.mesh-studio__tabs button:disabled {
  opacity: 0.35;
}
.mesh-studio__check {
  display: flex !important;
  align-items: center;
  gap: 9px !important;
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
.mesh-studio__row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.mesh-studio__primary {
  margin-top: auto;
  border-radius: var(--mold-radius-1);
  background: var(--mold-accent);
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
  border-color: var(--mold-accent);
  background: var(--mold-accent);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--mold-accent) 18%, transparent);
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
</style>
