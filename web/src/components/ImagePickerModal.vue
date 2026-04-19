<script setup lang="ts">
import { onMounted, ref } from "vue";
import { listGallery, thumbnailUrl, imageUrl } from "../api";
import { blobToBase64 } from "../lib/base64";
import type { GalleryImage, SourceImageState } from "../types";

defineProps<{ open: boolean }>();
const emit = defineEmits<{
  (e: "pick", v: SourceImageState): void;
  (e: "close"): void;
}>();

const tab = ref<"upload" | "gallery">("upload");
const entries = ref<GalleryImage[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);

onMounted(async () => {
  loading.value = true;
  try {
    entries.value = await listGallery();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});

async function onFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  const b64 = await blobToBase64(file);
  emit("pick", { kind: "upload", filename: file.name, base64: b64 });
  emit("close");
}

async function onDrop(event: DragEvent) {
  event.preventDefault();
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const b64 = await blobToBase64(file);
  emit("pick", { kind: "upload", filename: file.name, base64: b64 });
  emit("close");
}

async function pickFromGallery(item: GalleryImage) {
  const res = await fetch(imageUrl(item.filename));
  if (!res.ok) {
    error.value = `Fetch failed: ${res.status}`;
    return;
  }
  const blob = await res.blob();
  const b64 = await blobToBase64(blob);
  emit("pick", { kind: "gallery", filename: item.filename, base64: b64 });
  emit("close");
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm"
      @click.self="emit('close')"
    >
      <div
        class="glass flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl p-6"
      >
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-slate-100">🖼️ Source image</h2>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-100"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div class="mt-4 flex gap-2">
          <button
            type="button"
            class="rounded-full px-3 py-1 text-sm"
            :class="
              tab === 'upload'
                ? 'bg-brand-500 text-white'
                : 'bg-slate-900/60 text-slate-200'
            "
            @click="tab = 'upload'"
          >
            Upload
          </button>
          <button
            type="button"
            class="rounded-full px-3 py-1 text-sm"
            :class="
              tab === 'gallery'
                ? 'bg-brand-500 text-white'
                : 'bg-slate-900/60 text-slate-200'
            "
            @click="tab = 'gallery'"
          >
            From gallery
          </button>
        </div>

        <div v-if="tab === 'upload'" class="mt-4 flex-1 overflow-y-auto">
          <div
            class="flex h-48 w-full items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/40 text-slate-400"
            @dragover.prevent
            @drop="onDrop"
          >
            <label class="cursor-pointer text-center">
              <span>Drop an image here or click to browse</span>
              <input
                type="file"
                accept="image/*"
                class="hidden"
                @change="onFiles"
              />
            </label>
          </div>
        </div>

        <div v-else class="mt-4 flex-1 overflow-y-auto">
          <p v-if="loading" class="text-sm text-slate-400">Loading…</p>
          <p v-else-if="error" class="text-sm text-rose-300">{{ error }}</p>
          <ul v-else class="grid grid-cols-3 gap-2 sm:grid-cols-5">
            <li v-for="item in entries" :key="item.filename">
              <button
                type="button"
                class="group relative overflow-hidden rounded-xl bg-slate-900/40"
                @click="pickFromGallery(item)"
              >
                <img
                  :src="thumbnailUrl(item.filename)"
                  :alt="item.filename"
                  class="h-24 w-full object-cover transition group-hover:opacity-80"
                />
              </button>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </Teleport>
</template>
