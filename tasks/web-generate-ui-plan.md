# Web Generate UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add browser-driven generation to the mold SPA at a new `/generate` route — composer + multi-GPU running strip + reused gallery feed — without any server-side changes.

**Architecture:** Introduce `vue-router@4` and split today's single-page SPA into `GalleryPage` (unchanged behavior) and `GeneratePage`. `GeneratePage` is composed of a `Composer`, a `RunningStrip` of per-job SSE cards, and the existing `GalleryFeed`. State lives in three refs (form, running jobs, gallery entries). Submissions open one `fetch`-based SSE stream per job to `/api/generate/stream`; concurrency is governed by server queue backpressure (503 `queue_full`).

**Tech Stack:** Vue 3 (composition API) · TypeScript · vue-router 4 · Vite 7 · Tailwind v4.2 · bun.

**Spec:** [`tasks/web-generate-ui-spec.md`](./web-generate-ui-spec.md)

**Base branch:** `multi-gpu` · **Feature branch:** `feat/web-generate-ui` · **PR target:** `multi-gpu`

---

## Conventions for this plan

- **No formal unit tests** for `web/` — the project has no Vitest/Playwright setup today, and adding one is explicitly out of scope. Each task's verification is a combination of:
  - `cd web && bun run check` — the `build` script runs `vue-tsc -b` which is our type check (use `bunx vue-tsc --noEmit` where we don't want to emit).
  - `bun run build` — full production build; catches Vite-level issues.
  - `bun run fmt:check` — Prettier.
  - When the task touches runtime behavior, a short manual UAT step against the live server on <gpu-host> (the SSH tunnel and `MOLD_HOST=http://localhost:7680` are already configured).
- Commits are scoped per task with conventional-commit prefixes (`feat(web): …`, `fix(web): …`, `chore(web): …`).
- Every file path is relative to the repo root unless noted.
- Every task ends with a git commit. Push is a separate late task; don't push mid-plan.

---

## File map

### New
```
web/src/router.ts
web/src/pages/GalleryPage.vue
web/src/pages/GeneratePage.vue
web/src/components/Composer.vue
web/src/components/RunningStrip.vue
web/src/components/RunningJobCard.vue
web/src/components/SettingsModal.vue
web/src/components/ExpandModal.vue
web/src/components/ImagePickerModal.vue
web/src/components/ModelPicker.vue
web/src/composables/useGenerateForm.ts
web/src/composables/useGenerateStream.ts
web/src/composables/useStatusPoll.ts
web/src/lib/sse.ts
web/src/lib/base64.ts
```

### Modified
```
web/package.json            — add vue-router
web/src/main.ts             — install router
web/src/App.vue             — becomes <router-view>
web/src/components/TopBar.vue — add gallery/generate tab switch
web/src/api.ts              — add generateStream, fetchModels, fetchStatus, expandPrompt, toBase64
web/src/types.ts            — add generation-related types
```

No server-side (Rust) changes in this plan.

---

## Task 1: Scaffold the router and split App.vue

**Files:**
- Modify: `web/package.json` — add `vue-router@^4.4.0`.
- Create: `web/src/router.ts`.
- Create: `web/src/pages/GalleryPage.vue` (extract from `App.vue`).
- Create: `web/src/pages/GeneratePage.vue` (stub).
- Modify: `web/src/App.vue` (becomes thin `<router-view>` host).
- Modify: `web/src/main.ts` — `app.use(router)` before `mount`.

- [ ] **Step 1: Install vue-router**

```bash
cd web
bun add vue-router@^4.4.0
```

- [ ] **Step 2: Create `web/src/router.ts`**

```ts
import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";
import GalleryPage from "./pages/GalleryPage.vue";
import GeneratePage from "./pages/GeneratePage.vue";

const routes: RouteRecordRaw[] = [
  { path: "/", name: "gallery", component: GalleryPage },
  { path: "/generate", name: "generate", component: GeneratePage },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
```

- [ ] **Step 3: Create `web/src/pages/GalleryPage.vue` by moving everything currently in `web/src/App.vue` into it**

Copy the entirety of `web/src/App.vue` into `web/src/pages/GalleryPage.vue`. Update the relative import paths (from `./api` → `../api`, `./types` → `../types`, `./components/...` → `../components/...`).

- [ ] **Step 4: Create a stub `web/src/pages/GeneratePage.vue`**

```vue
<script setup lang="ts">
// Scaffold for Task 1. Composer / RunningStrip / feed land in later tasks.
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 py-6 sm:px-6 lg:px-10">
    <h1 class="text-2xl font-semibold text-slate-100">Generate</h1>
    <p class="mt-2 text-sm text-slate-400">Coming online in follow-up tasks.</p>
  </div>
</template>
```

- [ ] **Step 5: Replace `web/src/App.vue` with a thin router host**

```vue
<script setup lang="ts"></script>

<template>
  <router-view />
</template>
```

- [ ] **Step 6: Wire the router in `web/src/main.ts`**

Add these two lines around the existing mount:

```ts
import { router } from "./router";
// ...
app.use(router);
```

- [ ] **Step 7: Verify**

```bash
cd web && bun run build
```

Expected: clean build. Open `bun run dev` in your head — `/` should render the gallery exactly as before; `/generate` should render the stub page.

- [ ] **Step 8: Manual UAT**

```bash
cd web && bun run dev -- --host 0.0.0.0
```

Visit `http://localhost:5174/` → gallery renders unchanged. Visit `http://localhost:5174/generate` → stub page renders. Browser back/forward works.

- [ ] **Step 9: Commit**

```bash
git add web/package.json web/bun.lock web/src/router.ts web/src/App.vue web/src/main.ts web/src/pages/
git commit -m "feat(web): scaffold vue-router with /generate route"
```

---

## Task 2: TopBar tab switcher

**Files:**
- Modify: `web/src/components/TopBar.vue`.

Add a two-tab switcher between the logo and the existing filter/search block. Tabs use `router-link` with `active-class` styling consistent with existing `brand-500` accents.

- [ ] **Step 1: Read the current TopBar to understand its layout**

```bash
cat web/src/components/TopBar.vue
```

- [ ] **Step 2: Insert the tab switcher**

Inside the existing header, add immediately after the logo block:

```vue
<nav class="flex items-center gap-1 rounded-full bg-slate-900/60 p-1">
  <router-link
    to="/"
    class="rounded-full px-3 py-1 text-sm text-slate-400 transition hover:text-slate-100"
    active-class="bg-brand-500 text-white shadow"
    exact-active-class="bg-brand-500 text-white shadow"
  >
    Gallery
  </router-link>
  <router-link
    to="/generate"
    class="rounded-full px-3 py-1 text-sm text-slate-400 transition hover:text-slate-100"
    active-class="bg-brand-500 text-white shadow"
  >
    Generate
  </router-link>
</nav>
```

Place it where the existing `flex` header children sit so it inherits the same gap spacing. If the TopBar exposes gallery-specific filters, wrap them in `v-if="$route.name === 'gallery'"` so the Generate page doesn't show filter pills it can't use.

- [ ] **Step 3: Verify**

```bash
cd web && bun run build && bun run fmt:check
```

- [ ] **Step 4: Manual UAT**

Start dev server. Confirm the active tab is highlighted; clicking the other tab navigates; gallery filters disappear on `/generate`.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/TopBar.vue
git commit -m "feat(web): gallery/generate tab switch in TopBar"
```

---

## Task 3: Mirror generation types from the server

**Files:**
- Modify: `web/src/types.ts`.

Add TypeScript mirrors for the types the Generate page consumes. These come from `crates/mold-core/src/types.rs` — keep field names in camelCase on the client side but preserve the wire names (`snake_case`) when they cross HTTP. We use a single `toRequest()` helper in the form composable to do the mapping, so the UI types are idiomatic.

- [ ] **Step 1: Append these types to `web/src/types.ts`**

```ts
// ──────────────────────────────────────────────────────────────────────────────
// Generation types (mirror of mold_core::GenerateRequest / GenerateResponse /
// SseProgressEvent / SseCompleteEvent / ModelInfoExtended / ServerStatus).
// Client-side uses camelCase; serialization to/from the wire happens in api.ts.
// ──────────────────────────────────────────────────────────────────────────────

export interface LoraWeight {
  path: string;
  scale: number;
}

// Wire shape — what we POST to /api/generate/stream. snake_case to match serde.
export interface GenerateRequestWire {
  prompt: string;
  negative_prompt?: string | null;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed?: number | null;
  batch_size?: number;
  output_format?: OutputFormat;
  scheduler?: Scheduler | null;
  source_image?: string | null; // base64 (no data-URI prefix)
  strength?: number;
  expand?: boolean;
  original_prompt?: string | null;
  frames?: number | null;
  fps?: number | null;
}

export interface ModelDefaults {
  default_steps: number;
  default_guidance: number;
  default_width: number;
  default_height: number;
  description: string;
}

export interface ModelInfoExtended extends ModelDefaults {
  name: string;
  family: string;
  size_gb: number;
  is_loaded: boolean;
  last_used: number | null;
  hf_repo: string;
  downloaded: boolean;
  disk_usage_bytes?: number | null;
  remaining_download_bytes?: number | null;
}

export interface GpuInfo {
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
}

export type GpuWorkerState = "idle" | "generating" | "loading" | "degraded";

export interface GpuWorkerStatus {
  ordinal: number;
  name: string;
  vram_total_bytes: number;
  vram_used_bytes: number;
  loaded_model?: string | null;
  state: GpuWorkerState;
}

export interface ServerStatus {
  version: string;
  git_sha?: string | null;
  build_date?: string | null;
  models_loaded: string[];
  busy: boolean;
  gpu_info?: GpuInfo | null;
  uptime_secs: number;
  hostname?: string | null;
  memory_status?: string | null;
  gpus?: GpuWorkerStatus[] | null;
  queue_depth?: number | null;
  queue_capacity?: number | null;
}

export type SseProgressEvent =
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "info"; message: string }
  | { type: "cache_hit"; resource: string }
  | { type: "denoise_step"; step: number; total: number; elapsed_ms: number }
  | { type: "queued"; position: number }
  | {
      type: "weight_load";
      bytes_loaded: number;
      bytes_total: number;
      component: string;
    };

export interface SseCompleteEvent {
  image: string; // base64
  format: OutputFormat;
  width: number;
  height: number;
  seed_used: number;
  generation_time_ms: number;
  model: string;
  video_frames?: number | null;
  video_fps?: number | null;
  video_thumbnail?: string | null; // base64
  video_gif_preview?: string | null; // base64
  video_has_audio?: boolean;
  video_duration_ms?: number | null;
  video_audio_sample_rate?: number | null;
  video_audio_channels?: number | null;
  gpu?: number | null;
}

export interface ExpandRequestWire {
  prompt: string;
  model_family: string;
  variations: number;
}

export interface ExpandResponseWire {
  original: string;
  expanded: string[];
}

// ── Client-side form shape (persisted in localStorage) ─────────────────────
export interface SourceImageState {
  kind: "upload" | "gallery";
  filename: string;
  base64: string; // stripped before localStorage persist
}

export interface ExpandFormState {
  enabled: boolean;
  variations: 1 | 3 | 5;
  familyOverride: string | null;
}

export interface GenerateFormState {
  version: 1;
  prompt: string;
  negativePrompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null; // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  fps: number | null;
  scheduler: Scheduler | null;
  outputFormat: OutputFormat;
  expand: ExpandFormState;
  sourceImage: SourceImageState | null;
}

// ── Video-family detection helper used by multiple components ──────────────
export const VIDEO_FAMILIES: ReadonlyArray<string> = [
  "ltx-video",
  "ltx2",
  "ltx-2",
];

// Families whose image pipeline ignores the negative prompt.
export const NO_CFG_FAMILIES: ReadonlyArray<string> = [
  "flux",
  "flux2",
  "flux.2",
  "z-image",
  "qwen-image",
  "qwen_image",
];

// Families whose UNet responds to scheduler overrides.
export const UNET_SCHEDULER_FAMILIES: ReadonlyArray<string> = [
  "sd15",
  "sd1.5",
  "stable-diffusion-1.5",
  "sdxl",
];
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

Expected: 0 errors. Any type errors must be fixed inline before commit.

- [ ] **Step 3: Commit**

```bash
git add web/src/types.ts
git commit -m "feat(web): add generation-related types (mirror of mold_core)"
```

---

## Task 4: SSE client + base64 helpers

**Files:**
- Create: `web/src/lib/sse.ts`.
- Create: `web/src/lib/base64.ts`.

Browser `EventSource` is `GET`-only. We need to POST a JSON body, so we implement a small `fetch` + `ReadableStream` based SSE parser. Same technique the CLI uses (see `crates/mold-core/src/client.rs` for the event format we're consuming — lines prefixed `event:` and `data:` separated by blank lines).

- [ ] **Step 1: Create `web/src/lib/base64.ts`**

```ts
/**
 * Encode a Blob / File / ArrayBuffer as raw base64 (no data-URI prefix).
 * The server expects bare base64 in GenerateRequest.source_image.
 */
export async function blobToBase64(input: Blob): Promise<string> {
  const buffer = await input.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(
      ...bytes.subarray(i, Math.min(i + chunk, bytes.length)),
    );
  }
  return btoa(binary);
}

/** Inverse — base64 → Blob, used by the image picker's gallery tab. */
export function base64ToBlob(b64: string, mime: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
```

- [ ] **Step 2: Create `web/src/lib/sse.ts`**

```ts
/**
 * POST-capable Server-Sent Events client.
 *
 * Rationale: the browser's built-in EventSource is GET-only, but
 * /api/generate/stream needs a JSON request body. We stream the response,
 * buffer by newline, and reassemble SSE events (lines of `event:` and
 * `data:` terminated by a blank line).
 */

export interface SseEvent {
  event: string | null;
  data: string;
}

export interface StreamSseOptions<TBody> {
  url: string;
  body: TBody;
  signal?: AbortSignal;
  onEvent: (evt: SseEvent) => void;
  /** Called once with the Response if the server returned a non-2xx. */
  onHttpError?: (res: Response) => void;
}

export async function streamSse<TBody>(
  opts: StreamSseOptions<TBody>,
): Promise<Response> {
  const res = await fetch(opts.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(opts.body),
    signal: opts.signal,
  });

  if (!res.ok) {
    opts.onHttpError?.(res);
    return res;
  }
  if (!res.body) {
    throw new Error("SSE response has no body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName: string | null = null;
  let dataLines: string[] = [];

  const flush = () => {
    if (dataLines.length === 0 && !eventName) return;
    opts.onEvent({ event: eventName, data: dataLines.join("\n") });
    eventName = null;
    dataLines = [];
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are terminated by a blank line (\n\n). Split carefully so
    // we don't lose a partial line at the tail of the current chunk.
    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, "");
      buffer = buffer.slice(idx + 1);

      if (line === "") {
        flush();
        continue;
      }
      if (line.startsWith(":")) continue; // comment / keepalive
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
      // Other fields (id:, retry:) ignored — server doesn't use them.
    }
  }

  flush();
  return res;
}
```

- [ ] **Step 3: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/
git commit -m "feat(web): add POST-capable SSE client and base64 helpers"
```

---

## Task 5: api.ts — generate/status/expand/models helpers

**Files:**
- Modify: `web/src/api.ts`.

Add thin wrappers. `generateStream` is the only complex one — it composes `streamSse` with typed event dispatch.

- [ ] **Step 1: Append to `web/src/api.ts`**

```ts
import type {
  ExpandRequestWire,
  ExpandResponseWire,
  GenerateRequestWire,
  ModelInfoExtended,
  ServerStatus,
  SseCompleteEvent,
  SseProgressEvent,
} from "./types";
import { streamSse } from "./lib/sse";

export async function fetchModels(signal?: AbortSignal): Promise<ModelInfoExtended[]> {
  const res = await fetch(`${base}/api/models`, { signal });
  if (!res.ok) throw new Error(`GET /api/models failed: ${res.status}`);
  return (await res.json()) as ModelInfoExtended[];
}

export async function fetchStatus(signal?: AbortSignal): Promise<ServerStatus> {
  const res = await fetch(`${base}/api/status`, { signal });
  if (!res.ok) throw new Error(`GET /api/status failed: ${res.status}`);
  return (await res.json()) as ServerStatus;
}

export async function expandPrompt(
  req: ExpandRequestWire,
  signal?: AbortSignal,
): Promise<ExpandResponseWire> {
  const res = await fetch(`${base}/api/expand`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST /api/expand failed: ${res.status} ${text}`);
  }
  return (await res.json()) as ExpandResponseWire;
}

export interface GenerateStreamHandlers {
  onProgress: (evt: SseProgressEvent) => void;
  onComplete: (evt: SseCompleteEvent) => void;
  onError: (err: { kind: "http"; status: number; retryAfter?: number; body: string } | { kind: "network"; message: string }) => void;
}

export async function generateStream(
  req: GenerateRequestWire,
  handlers: GenerateStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await streamSse({
      url: `${base}/api/generate/stream`,
      body: req,
      signal,
      onEvent: (evt) => {
        if (!evt.data) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (evt.event === "complete") {
          handlers.onComplete(parsed as SseCompleteEvent);
        } else if (evt.event === "error") {
          const body = evt.data;
          handlers.onError({ kind: "http", status: 0, body });
        } else {
          // Default `progress` event — the server emits many progress types
          // tagged with an internal `type` discriminator.
          handlers.onProgress(parsed as SseProgressEvent);
        }
      },
      onHttpError: (res) => {
        const retryAfterHeader = res.headers.get("Retry-After");
        const retryAfter = retryAfterHeader
          ? Number.parseFloat(retryAfterHeader)
          : undefined;
        res
          .text()
          .then((body) =>
            handlers.onError({
              kind: "http",
              status: res.status,
              retryAfter: Number.isFinite(retryAfter) ? retryAfter : undefined,
              body,
            }),
          )
          .catch(() =>
            handlers.onError({ kind: "http", status: res.status, body: "" }),
          );
      },
    });
    void res;
  } catch (err) {
    if (signal?.aborted) return; // user canceled
    handlers.onError({
      kind: "network",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
```

Note: the actual event naming (`event: progress` vs `event: complete`) must match the server. Confirm against `crates/mold-server/src/routes.rs::generate_stream` — the current implementation sets `.event("progress")` and `.event("complete")`. If the server uses a different event name, align here.

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add web/src/api.ts
git commit -m "feat(web): add generateStream, fetchModels, fetchStatus, expandPrompt"
```

---

## Task 6: `useGenerateForm` composable — form state + localStorage

**Files:**
- Create: `web/src/composables/useGenerateForm.ts`.

Single source of truth for the form. Persists to `localStorage["mold.generate.form"]` with a 300 ms debounce. Exposes a `toRequest()` derivation for submission.

- [ ] **Step 1: Write the composable**

```ts
import { ref, watch, type Ref } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  ModelInfoExtended,
  Scheduler,
} from "../types";
import {
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
} from "../types";

const STORAGE_KEY = "mold.generate.form";

function defaultForm(): GenerateFormState {
  return {
    version: 1,
    prompt: "",
    negativePrompt: "",
    model: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    seed: null,
    batchSize: 1,
    strength: 0.75,
    frames: null,
    fps: null,
    scheduler: null,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    sourceImage: null,
  };
}

function load(): GenerateFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultForm();
    const parsed = JSON.parse(raw) as Partial<GenerateFormState>;
    if (parsed.version !== 1) return defaultForm();
    return { ...defaultForm(), ...parsed, sourceImage: null };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  try {
    // Drop base64 bytes from localStorage — they blow past the quota quickly
    // and the attachment is re-picked trivially on reload.
    const { sourceImage: _omit, ...rest } = state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rest));
  } catch {
    /* ignore */
  }
}

export interface UseGenerateForm {
  state: Ref<GenerateFormState>;
  reset: () => void;
  applyModelDefaults: (model: ModelInfoExtended) => void;
  toRequest: () => GenerateRequestWire;
  isVideoFamily: (family: string) => boolean;
  supportsNegativePrompt: (family: string) => boolean;
  supportsScheduler: (family: string) => boolean;
}

export function useGenerateForm(): UseGenerateForm {
  const state = ref<GenerateFormState>(load());

  let timer: ReturnType<typeof setTimeout> | null = null;
  watch(
    state,
    (v) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => persist(v), 300);
    },
    { deep: true },
  );

  return {
    state,
    reset: () => {
      state.value = defaultForm();
    },
    applyModelDefaults: (m) => {
      state.value.model = m.name;
      state.value.width = m.default_width;
      state.value.height = m.default_height;
      state.value.steps = m.default_steps;
      state.value.guidance = m.default_guidance;
      // Video families need sensible frame/fps defaults.
      if (VIDEO_FAMILIES.includes(m.family)) {
        state.value.frames ??= 25; // 8n+1
        state.value.fps ??= 24;
      } else {
        state.value.frames = null;
        state.value.fps = null;
      }
    },
    toRequest: () => {
      const s = state.value;
      return {
        prompt: s.prompt,
        negative_prompt: s.negativePrompt || null,
        model: s.model,
        width: s.width,
        height: s.height,
        steps: s.steps,
        guidance: s.guidance,
        seed: s.seed,
        batch_size: s.batchSize,
        output_format: s.outputFormat,
        scheduler: s.scheduler,
        source_image: s.sourceImage?.base64 ?? null,
        strength: s.strength,
        expand: s.expand.enabled || undefined,
        frames: s.frames,
        fps: s.fps,
      };
    },
    isVideoFamily: (family: string) => VIDEO_FAMILIES.includes(family),
    supportsNegativePrompt: (family: string) => !NO_CFG_FAMILIES.includes(family),
    supportsScheduler: (family: string) => UNET_SCHEDULER_FAMILIES.includes(family),
  };
}

// Scheduler type is re-exported so callers can type-narrow without importing
// both modules.
export type { Scheduler };
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add web/src/composables/useGenerateForm.ts
git commit -m "feat(web): useGenerateForm composable with localStorage persist"
```

---

## Task 7: `useStatusPoll` composable — queue depth + GPU pills

**Files:**
- Create: `web/src/composables/useStatusPoll.ts`.

Polls `/api/status` every 5s while the page is visible; pauses while the tab is hidden.

- [ ] **Step 1: Write it**

```ts
import { onBeforeUnmount, onMounted, ref, type Ref } from "vue";
import { fetchStatus } from "../api";
import type { ServerStatus } from "../types";

export interface UseStatusPoll {
  status: Ref<ServerStatus | null>;
  error: Ref<string | null>;
}

export function useStatusPoll(intervalMs = 5000): UseStatusPoll {
  const status = ref<ServerStatus | null>(null);
  const error = ref<string | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;
  let controller: AbortController | null = null;

  async function tick() {
    controller?.abort();
    controller = new AbortController();
    try {
      status.value = await fetchStatus(controller.signal);
      error.value = null;
    } catch (e) {
      if (controller.signal.aborted) return;
      error.value = e instanceof Error ? e.message : String(e);
    }
  }

  function start() {
    if (timer) return;
    tick();
    timer = setInterval(tick, intervalMs);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    controller?.abort();
    controller = null;
  }

  function onVisibilityChange() {
    if (document.hidden) stop();
    else start();
  }

  onMounted(() => {
    start();
    document.addEventListener("visibilitychange", onVisibilityChange);
  });
  onBeforeUnmount(() => {
    stop();
    document.removeEventListener("visibilitychange", onVisibilityChange);
  });

  return { status, error };
}
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add web/src/composables/useStatusPoll.ts
git commit -m "feat(web): useStatusPoll composable for live queue/GPU state"
```

---

## Task 8: `useGenerateStream` composable — running jobs

**Files:**
- Create: `web/src/composables/useGenerateStream.ts`.

Manages an array of `Job` records. Each `submit()` opens a new SSE stream via `generateStream()`. Exposes `cancel(id)` and a callback that fires on each completion so the page can refresh the gallery.

- [ ] **Step 1: Write it**

```ts
import { ref, type Ref } from "vue";
import { generateStream } from "../api";
import type {
  GenerateRequestWire,
  SseCompleteEvent,
  SseProgressEvent,
} from "../types";

export interface JobProgress {
  stage: string;
  step: number | null;
  totalSteps: number | null;
  weightBytesLoaded: number | null;
  weightBytesTotal: number | null;
  queuePosition: number | null;
  gpu: number | null;
  elapsedMs: number | null;
}

export interface Job {
  id: string;
  request: GenerateRequestWire;
  startedAt: number;
  controller: AbortController;
  progress: JobProgress;
  result: SseCompleteEvent | null;
  error: string | null;
  state: "running" | "done" | "error" | "canceled";
}

function emptyProgress(): JobProgress {
  return {
    stage: "Starting",
    step: null,
    totalSteps: null,
    weightBytesLoaded: null,
    weightBytesTotal: null,
    queuePosition: null,
    gpu: null,
    elapsedMs: null,
  };
}

function applyProgress(job: Job, evt: SseProgressEvent) {
  const p = job.progress;
  switch (evt.type) {
    case "stage_start":
      p.stage = evt.name;
      break;
    case "stage_done":
      p.stage = `${evt.name} (done)`;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "info":
      p.stage = evt.message;
      break;
    case "denoise_step":
      p.stage = "Denoising";
      p.step = evt.step;
      p.totalSteps = evt.total;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "queued":
      p.stage = `Queued (position ${evt.position})`;
      p.queuePosition = evt.position;
      break;
    case "weight_load":
      p.stage = `Loading ${evt.component}`;
      p.weightBytesLoaded = evt.bytes_loaded;
      p.weightBytesTotal = evt.bytes_total;
      break;
    case "cache_hit":
      p.stage = `Cache hit: ${evt.resource}`;
      break;
  }
}

export interface UseGenerateStream {
  jobs: Ref<Job[]>;
  submit: (req: GenerateRequestWire) => string;
  cancel: (id: string) => void;
  clearDone: () => void;
}

export function useGenerateStream(
  onComplete?: (job: Job) => void,
): UseGenerateStream {
  const jobs = ref<Job[]>([]);

  function submit(req: GenerateRequestWire): string {
    const id = crypto.randomUUID();
    const controller = new AbortController();
    const job: Job = {
      id,
      request: req,
      startedAt: Date.now(),
      controller,
      progress: emptyProgress(),
      result: null,
      error: null,
      state: "running",
    };
    jobs.value = [...jobs.value, job];

    generateStream(
      req,
      {
        onProgress: (evt) => {
          applyProgress(job, evt);
          jobs.value = [...jobs.value];
        },
        onComplete: (evt) => {
          job.result = evt;
          job.state = "done";
          if (evt.gpu !== null && evt.gpu !== undefined) job.progress.gpu = evt.gpu;
          jobs.value = [...jobs.value];
          onComplete?.(job);
        },
        onError: (err) => {
          if (err.kind === "http") {
            job.error =
              err.status === 503
                ? `Queue full (retry after ${err.retryAfter ?? "?"}s)`
                : `HTTP ${err.status}: ${err.body}`;
          } else {
            job.error = err.message;
          }
          job.state = "error";
          jobs.value = [...jobs.value];
        },
      },
      controller.signal,
    );

    return id;
  }

  function cancel(id: string) {
    const job = jobs.value.find((j) => j.id === id);
    if (!job) return;
    job.controller.abort();
    job.state = "canceled";
    jobs.value = [...jobs.value];
  }

  function clearDone() {
    jobs.value = jobs.value.filter((j) => j.state === "running");
  }

  return { jobs, submit, cancel, clearDone };
}
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add web/src/composables/useGenerateStream.ts
git commit -m "feat(web): useGenerateStream composable for concurrent SSE jobs"
```

---

## Task 9: `Composer.vue` — prompt textarea + icon buttons + source chip

**Files:**
- Create: `web/src/components/Composer.vue`.

Emits `submit`, `open-settings`, `open-expand`, `open-image-picker`, `clear-source`.

- [ ] **Step 1: Write the component**

```vue
<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { GenerateFormState } from "../types";

const props = defineProps<{
  modelValue: GenerateFormState;
  queueDepth: number | null;
  queueCapacity: number | null;
  gpus: { ordinal: number; state: string }[] | null;
  expandActive: boolean;
  settingsDirty: boolean;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
  (e: "submit"): void;
  (e: "open-settings"): void;
  (e: "open-expand"): void;
  (e: "open-image-picker"): void;
  (e: "clear-source"): void;
}>();

const textarea = ref<HTMLTextAreaElement | null>(null);

function updatePrompt(value: string) {
  emit("update:modelValue", { ...props.modelValue, prompt: value });
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (props.modelValue.prompt.trim().length === 0) return;
    emit("submit");
  }
}

// Auto-grow textarea up to ~8 lines of content.
watch(
  () => props.modelValue.prompt,
  async () => {
    await nextTick();
    const el = textarea.value;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 8 * 28)}px`;
  },
  { immediate: true },
);

const statusLine = computed(() => {
  const parts: string[] = [];
  if (props.queueDepth !== null && props.queueCapacity !== null) {
    parts.push(`queue ${props.queueDepth}/${props.queueCapacity}`);
  }
  if (props.gpus) {
    const pills = props.gpus.map(
      (g) => `GPU ${g.ordinal} ${g.state === "idle" ? "▯" : "▮"}`,
    );
    parts.push(pills.join(" · "));
  }
  return parts.join(" · ");
});
</script>

<template>
  <div class="glass flex flex-col gap-2 rounded-3xl p-4">
    <div class="flex items-start gap-3">
      <!-- source image chip -->
      <div
        v-if="modelValue.sourceImage"
        class="relative flex-shrink-0 overflow-hidden rounded-xl"
      >
        <img
          :src="`data:image/png;base64,${modelValue.sourceImage.base64}`"
          class="h-12 w-12 object-cover"
          alt="Source"
        />
        <button
          type="button"
          class="absolute -right-1 -top-1 h-5 w-5 rounded-full bg-slate-900/90 text-xs text-slate-100"
          @click="emit('clear-source')"
          aria-label="Remove source image"
        >
          ✕
        </button>
      </div>

      <textarea
        ref="textarea"
        :value="modelValue.prompt"
        placeholder="Describe what to generate — Enter to submit, Shift+Enter for a newline"
        class="min-h-[2.5rem] flex-1 resize-none bg-transparent text-base text-slate-100 placeholder:text-slate-500 focus:outline-none"
        @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
        @keydown="onKeydown"
      />

      <div class="flex flex-shrink-0 flex-col gap-1 sm:flex-row">
        <button
          type="button"
          class="icon-btn"
          aria-label="Attach source image"
          @click="emit('open-image-picker')"
        >
          🖼️
        </button>
        <button
          type="button"
          class="icon-btn"
          :class="{ 'text-brand-400': expandActive }"
          aria-label="Prompt expansion"
          @click="emit('open-expand')"
        >
          ✨
        </button>
        <button
          type="button"
          class="icon-btn relative"
          aria-label="Settings"
          @click="emit('open-settings')"
        >
          ⚙
          <span
            v-if="settingsDirty"
            class="absolute right-1 top-1 h-2 w-2 rounded-full bg-brand-400"
          ></span>
        </button>
      </div>
    </div>

    <div v-if="statusLine" class="px-1 text-xs text-slate-500">
      {{ statusLine }}
    </div>
  </div>
</template>

<style scoped>
.icon-btn {
  display: inline-flex;
  height: 2.25rem;
  width: 2.25rem;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  color: rgb(226 232 240);
  background: rgba(15, 23, 42, 0.6);
  transition: background 150ms ease;
}
.icon-btn:hover {
  background: rgba(30, 41, 59, 0.9);
}
</style>
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/Composer.vue
git commit -m "feat(web): Composer with Enter/Shift+Enter contract and icon pill"
```

---

## Task 10: `ModelPicker.vue` — dropdown with video section + downloaded toggle

**Files:**
- Create: `web/src/components/ModelPicker.vue`.

- [ ] **Step 1: Write it**

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import type { ModelInfoExtended } from "../types";
import { VIDEO_FAMILIES } from "../types";

const props = defineProps<{
  models: ModelInfoExtended[];
  modelValue: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: string): void;
  (e: "select", model: ModelInfoExtended): void;
}>();

const SHOW_ALL_KEY = "mold.generate.showAllModels";
const showAll = ref(localStorage.getItem(SHOW_ALL_KEY) === "true");

function setShowAll(v: boolean) {
  showAll.value = v;
  try {
    localStorage.setItem(SHOW_ALL_KEY, String(v));
  } catch {
    /* ignore */
  }
}

const visibleModels = computed(() =>
  props.models.filter((m) => (showAll.value ? true : m.downloaded)),
);

const imageModels = computed(() =>
  visibleModels.value.filter((m) => !VIDEO_FAMILIES.includes(m.family)),
);
const videoModels = computed(() =>
  visibleModels.value.filter((m) => VIDEO_FAMILIES.includes(m.family)),
);

function onPick(model: ModelInfoExtended) {
  if (!model.downloaded) return;
  emit("update:modelValue", model.name);
  emit("select", model);
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <label class="flex items-center justify-between text-xs uppercase text-slate-400">
      <span>Model</span>
      <span class="flex items-center gap-2 normal-case">
        <input
          id="mold-show-all-models"
          type="checkbox"
          :checked="showAll"
          @change="setShowAll(($event.target as HTMLInputElement).checked)"
        />
        <label for="mold-show-all-models">Show all</label>
      </span>
    </label>

    <div class="flex max-h-80 flex-col gap-3 overflow-y-auto pr-1">
      <div>
        <div class="text-xs font-medium text-slate-500">Images</div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in imageModels" :key="m.name">
            <button
              type="button"
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200',
                !m.downloaded && 'cursor-not-allowed opacity-50',
              ]"
              :disabled="!m.downloaded"
              :title="m.downloaded ? m.description : 'Not downloaded — run mold pull ' + m.name + ' on the host.'"
              @click="onPick(m)"
            >
              <div class="flex items-center justify-between gap-2">
                <span>{{ m.name }}</span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </div>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
            </button>
          </li>
        </ul>
      </div>

      <div v-if="videoModels.length">
        <div class="flex items-center gap-2 text-xs font-medium text-slate-500">
          <span>🎬</span><span>Video</span>
        </div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in videoModels" :key="m.name">
            <button
              type="button"
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200',
                !m.downloaded && 'cursor-not-allowed opacity-50',
              ]"
              :disabled="!m.downloaded"
              :title="m.downloaded ? m.description : 'Not downloaded — run mold pull ' + m.name + ' on the host.'"
              @click="onPick(m)"
            >
              <div class="flex items-center justify-between gap-2">
                <span>{{ m.name }} <span class="italic text-xs text-slate-400">video</span></span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </div>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
            </button>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ModelPicker.vue
git commit -m "feat(web): ModelPicker with video section and downloaded gate"
```

---

## Task 11: `SettingsModal.vue`

**Files:**
- Create: `web/src/components/SettingsModal.vue`.

A centered modal that opens when the ⚙ icon is clicked. Wires up the `ModelPicker`, size chips, steps/guidance sliders, seed, negative prompt (family-gated), batch, strength (source-gated), frames/fps (video-gated), advanced (scheduler/output format).

- [ ] **Step 1: Write it**

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  GenerateFormState,
  ModelInfoExtended,
  OutputFormat,
  Scheduler,
} from "../types";
import {
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
} from "../types";
import ModelPicker from "./ModelPicker.vue";

const props = defineProps<{
  open: boolean;
  modelValue: GenerateFormState;
  models: ModelInfoExtended[];
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
  (e: "close"): void;
}>();

function patch<K extends keyof GenerateFormState>(key: K, value: GenerateFormState[K]) {
  emit("update:modelValue", { ...props.modelValue, [key]: value });
}

const currentModel = computed(() =>
  props.models.find((m) => m.name === props.modelValue.model) ?? null,
);
const family = computed(() => currentModel.value?.family ?? "");

const showNegative = computed(() => !NO_CFG_FAMILIES.includes(family.value));
const showScheduler = computed(() => UNET_SCHEDULER_FAMILIES.includes(family.value));
const showVideo = computed(() => VIDEO_FAMILIES.includes(family.value));
const showStrength = computed(() => !!props.modelValue.sourceImage);

function selectModel(m: ModelInfoExtended) {
  const next: GenerateFormState = {
    ...props.modelValue,
    model: m.name,
    width: m.default_width,
    height: m.default_height,
    steps: m.default_steps,
    guidance: m.default_guidance,
  };
  if (VIDEO_FAMILIES.includes(m.family)) {
    next.frames ??= 25;
    next.fps ??= 24;
  } else {
    next.frames = null;
    next.fps = null;
  }
  emit("update:modelValue", next);
}

// frames must be 8n+1 (9, 17, 25, 33, ...)
function clampFrames(n: number): number {
  if (!Number.isFinite(n)) return 25;
  const rounded = Math.max(9, Math.round((n - 1) / 8) * 8 + 1);
  return rounded;
}

const advancedOpen = ref(false);

const sizePresets = [512, 768, 1024] as const;
const batchChips = [1, 2, 3, 4] as const;

const schedulerOptions: Scheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "unipc",
];
// `computed` so the option list re-derives when the model family changes.
// (A plain top-level `const` captures the initial `showVideo.value` and
// never updates.)
const outputFormatOptions = computed<OutputFormat[]>(() =>
  showVideo.value ? ["mp4", "gif", "apng", "webp"] : ["png", "jpeg", "webp"],
);
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm"
      @click.self="emit('close')"
    >
      <div class="glass max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl p-6 sm:p-8">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-slate-100">Settings</h2>
          <button type="button" class="text-slate-400 hover:text-slate-100" @click="emit('close')">✕</button>
        </div>

        <section class="mt-4">
          <ModelPicker
            :models="models"
            :model-value="modelValue.model"
            @update:modelValue="(v) => patch('model', v)"
            @select="selectModel"
          />
        </section>

        <section class="mt-4">
          <label class="text-xs uppercase text-slate-400">Size</label>
          <div class="mt-1 flex flex-wrap gap-2">
            <button
              v-for="n in sizePresets"
              :key="n"
              type="button"
              class="rounded-full px-3 py-1 text-sm"
              :class="modelValue.width === n && modelValue.height === n ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200'"
              @click="emit('update:modelValue', { ...modelValue, width: n, height: n })"
            >
              {{ n }}×{{ n }}
            </button>
            <div class="flex items-center gap-2 text-sm">
              <input
                type="number"
                :value="modelValue.width"
                class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="patch('width', Number(($event.target as HTMLInputElement).value) || modelValue.width)"
              />
              ×
              <input
                type="number"
                :value="modelValue.height"
                class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="patch('height', Number(($event.target as HTMLInputElement).value) || modelValue.height)"
              />
            </div>
          </div>
        </section>

        <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label class="text-xs uppercase text-slate-400">Steps — {{ modelValue.steps }}</label>
            <input
              type="range"
              min="1"
              max="100"
              :value="modelValue.steps"
              class="w-full"
              @input="patch('steps', Number(($event.target as HTMLInputElement).value))"
            />
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">Guidance — {{ modelValue.guidance.toFixed(1) }}</label>
            <input
              type="range"
              min="0"
              max="20"
              step="0.1"
              :value="modelValue.guidance"
              class="w-full"
              @input="patch('guidance', Number(($event.target as HTMLInputElement).value))"
            />
          </div>
        </section>

        <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label class="text-xs uppercase text-slate-400">Seed</label>
            <div class="mt-1 flex items-center gap-2">
              <input
                type="number"
                :value="modelValue.seed ?? ''"
                placeholder="random"
                class="flex-1 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="patch('seed', (($event.target as HTMLInputElement).value === '' ? null : Number(($event.target as HTMLInputElement).value)))"
              />
              <button
                type="button"
                class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm"
                @click="patch('seed', null)"
              >🎲</button>
            </div>
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">Batch</label>
            <div class="mt-1 flex gap-1">
              <button
                v-for="n in batchChips"
                :key="n"
                type="button"
                class="rounded-full px-3 py-1 text-sm"
                :class="modelValue.batchSize === n ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200'"
                @click="patch('batchSize', n)"
              >{{ n }}</button>
            </div>
          </div>
        </section>

        <section v-if="showNegative" class="mt-4">
          <label class="text-xs uppercase text-slate-400">Negative prompt</label>
          <textarea
            :value="modelValue.negativePrompt"
            rows="2"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            placeholder="e.g. blurry, low quality, watermark"
            @input="patch('negativePrompt', ($event.target as HTMLTextAreaElement).value)"
          />
        </section>

        <section v-if="showStrength" class="mt-4">
          <label class="text-xs uppercase text-slate-400">Strength — {{ modelValue.strength.toFixed(2) }}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="modelValue.strength"
            class="w-full"
            @input="patch('strength', Number(($event.target as HTMLInputElement).value))"
          />
        </section>

        <section v-if="showVideo" class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label class="text-xs uppercase text-slate-400">Frames (8n+1)</label>
            <input
              type="number"
              :value="modelValue.frames ?? 25"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @change="patch('frames', clampFrames(Number(($event.target as HTMLInputElement).value)))"
            />
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">FPS</label>
            <input
              type="number"
              :value="modelValue.fps ?? 24"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @change="patch('fps', Number(($event.target as HTMLInputElement).value) || 24)"
            />
          </div>
        </section>

        <section class="mt-4">
          <button
            type="button"
            class="text-xs uppercase tracking-wide text-slate-400 hover:text-slate-200"
            @click="advancedOpen = !advancedOpen"
          >{{ advancedOpen ? "▾" : "▸" }} Advanced</button>

          <div v-if="advancedOpen" class="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div v-if="showScheduler">
              <label class="text-xs uppercase text-slate-400">Scheduler</label>
              <select
                :value="modelValue.scheduler ?? 'default'"
                class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @change="patch('scheduler', (($event.target as HTMLSelectElement).value as Scheduler))"
              >
                <option v-for="s in schedulerOptions" :key="String(s)" :value="s">{{ s }}</option>
              </select>
            </div>
            <div>
              <label class="text-xs uppercase text-slate-400">Output format</label>
              <select
                :value="modelValue.outputFormat"
                class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @change="patch('outputFormat', (($event.target as HTMLSelectElement).value as OutputFormat))"
              >
                <option v-for="f in outputFormatOptions" :key="f" :value="f">{{ f }}</option>
              </select>
            </div>
          </div>
        </section>
      </div>
    </div>
  </Teleport>
</template>
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/SettingsModal.vue
git commit -m "feat(web): SettingsModal with model/size/steps/guidance/seed/batch/video"
```

---

## Task 12: `ExpandModal.vue`

**Files:**
- Create: `web/src/components/ExpandModal.vue`.

- [ ] **Step 1: Write it**

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import { expandPrompt } from "../api";
import type { ExpandFormState, ModelInfoExtended } from "../types";

const props = defineProps<{
  open: boolean;
  prompt: string;
  expand: ExpandFormState;
  currentModel: ModelInfoExtended | null;
}>();
const emit = defineEmits<{
  (e: "update:expand", v: ExpandFormState): void;
  (e: "apply-prompt", v: string): void;
  (e: "close"): void;
}>();

const previewing = ref(false);
const previewError = ref<string | null>(null);
const previewResults = ref<string[]>([]);

const effectiveFamily = computed(
  () => props.expand.familyOverride ?? props.currentModel?.family ?? "flux",
);

const variationsOptions = [1, 3, 5] as const;

function patch<K extends keyof ExpandFormState>(key: K, v: ExpandFormState[K]) {
  emit("update:expand", { ...props.expand, [key]: v });
}

async function preview() {
  previewing.value = true;
  previewError.value = null;
  try {
    const res = await expandPrompt({
      prompt: props.prompt,
      model_family: effectiveFamily.value,
      variations: props.expand.variations,
    });
    previewResults.value = res.expanded;
  } catch (e) {
    previewError.value = e instanceof Error ? e.message : String(e);
  } finally {
    previewing.value = false;
  }
}

function pick(text: string) {
  emit("apply-prompt", text);
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
      <div class="glass max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-3xl p-6 sm:p-8">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-slate-100">✨ Prompt expansion</h2>
          <button type="button" class="text-slate-400 hover:text-slate-100" @click="emit('close')">✕</button>
        </div>

        <label class="mt-4 flex items-center gap-2 text-sm text-slate-200">
          <input
            type="checkbox"
            :checked="expand.enabled"
            @change="patch('enabled', ($event.target as HTMLInputElement).checked)"
          />
          Enable expansion before submit
        </label>

        <div class="mt-3">
          <label class="text-xs uppercase text-slate-400">Variations</label>
          <div class="mt-1 flex gap-1">
            <button
              v-for="n in variationsOptions"
              :key="n"
              type="button"
              class="rounded-full px-3 py-1 text-sm"
              :class="expand.variations === n ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200'"
              @click="patch('variations', n)"
            >{{ n }}</button>
          </div>
        </div>

        <details class="mt-3 text-sm text-slate-300">
          <summary class="cursor-pointer text-slate-400">Advanced: model family override</summary>
          <input
            type="text"
            :value="expand.familyOverride ?? ''"
            placeholder="auto"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            @change="patch('familyOverride', ($event.target as HTMLInputElement).value || null)"
          />
        </details>

        <div class="mt-4 flex items-center gap-2">
          <button
            type="button"
            class="rounded-lg bg-brand-500 px-3 py-1.5 text-sm text-white disabled:opacity-50"
            :disabled="previewing || !prompt.trim()"
            @click="preview"
          >{{ previewing ? "Expanding…" : "Preview" }}</button>
        </div>

        <div v-if="previewError" class="mt-2 text-sm text-rose-300">{{ previewError }}</div>

        <ul v-if="previewResults.length" class="mt-4 flex flex-col gap-2">
          <li v-for="(text, i) in previewResults" :key="i">
            <button
              type="button"
              class="w-full rounded-xl bg-slate-900/60 p-3 text-left text-sm text-slate-100 hover:bg-slate-800/80"
              @click="pick(text)"
            >{{ text }}</button>
          </li>
        </ul>

        <p class="mt-4 text-xs text-slate-500">
          Backend model, temperature, and thinking mode are controlled by the server config.
          Ask your operator to adjust them.
        </p>
      </div>
    </div>
  </Teleport>
</template>
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ExpandModal.vue
git commit -m "feat(web): ExpandModal with preview + variation picker"
```

---

## Task 13: `ImagePickerModal.vue`

**Files:**
- Create: `web/src/components/ImagePickerModal.vue`.

- [ ] **Step 1: Write it**

```vue
<script setup lang="ts">
import { onMounted, ref } from "vue";
import { listGallery, thumbnailUrl, imageUrl } from "../api";
import { blobToBase64 } from "../lib/base64";
import type { GalleryImage, SourceImageState } from "../types";

const props = defineProps<{ open: boolean }>();
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
      <div class="glass flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl p-6">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-slate-100">🖼️ Source image</h2>
          <button type="button" class="text-slate-400 hover:text-slate-100" @click="emit('close')">✕</button>
        </div>

        <div class="mt-4 flex gap-2">
          <button
            type="button"
            class="rounded-full px-3 py-1 text-sm"
            :class="tab === 'upload' ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200'"
            @click="tab = 'upload'"
          >Upload</button>
          <button
            type="button"
            class="rounded-full px-3 py-1 text-sm"
            :class="tab === 'gallery' ? 'bg-brand-500 text-white' : 'bg-slate-900/60 text-slate-200'"
            @click="tab = 'gallery'"
          >From gallery</button>
        </div>

        <div v-if="tab === 'upload'" class="mt-4 flex-1 overflow-y-auto">
          <div
            class="flex h-48 w-full items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/40 text-slate-400"
            @dragover.prevent
            @drop="onDrop"
          >
            <label class="cursor-pointer text-center">
              <span>Drop an image here or click to browse</span>
              <input type="file" accept="image/*" class="hidden" @change="onFiles" />
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
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ImagePickerModal.vue
git commit -m "feat(web): ImagePickerModal with upload + from-gallery tabs"
```

---

## Task 14: `RunningJobCard.vue` and `RunningStrip.vue`

**Files:**
- Create: `web/src/components/RunningJobCard.vue`.
- Create: `web/src/components/RunningStrip.vue`.

- [ ] **Step 1: RunningJobCard.vue**

```vue
<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";

const props = defineProps<{ job: Job }>();
const emit = defineEmits<{ (e: "cancel", id: string): void }>();

const pct = computed(() => {
  const p = props.job.progress;
  if (p.step !== null && p.totalSteps) {
    return Math.round((p.step / p.totalSteps) * 100);
  }
  if (p.weightBytesLoaded !== null && p.weightBytesTotal) {
    return Math.round((p.weightBytesLoaded / p.weightBytesTotal) * 100);
  }
  return null;
});

const thumbSrc = computed(() => {
  const r = props.job.result;
  if (!r) return null;
  // Video: thumbnail is always PNG (server-side). Image: use the declared format.
  // Note: "data:image/*;base64,..." is NOT a valid MIME — browsers reject it silently.
  if (r.video_thumbnail) return `data:image/png;base64,${r.video_thumbnail}`;
  const mime = r.format === "jpeg" ? "image/jpeg" : `image/${r.format}`;
  return `data:${mime};base64,${r.image}`;
});
</script>

<template>
  <div class="glass flex w-[280px] flex-shrink-0 flex-col gap-2 rounded-2xl p-3">
    <div class="flex items-center justify-between text-xs text-slate-400">
      <span>{{ job.request.model }}</span>
      <span v-if="job.progress.gpu !== null">GPU {{ job.progress.gpu }}</span>
    </div>
    <div class="relative aspect-square overflow-hidden rounded-xl bg-slate-900/60">
      <img v-if="thumbSrc" :src="thumbSrc" class="h-full w-full object-cover" alt="" />
      <div v-else class="h-full w-full animate-pulse bg-slate-800/60"></div>
      <div
        v-if="job.state === 'error'"
        class="absolute inset-0 flex items-center justify-center bg-rose-500/70 p-2 text-center text-xs text-white"
      >
        {{ job.error }}
      </div>
    </div>
    <div class="text-xs text-slate-300">{{ job.progress.stage }}</div>
    <div v-if="pct !== null" class="h-1 w-full overflow-hidden rounded-full bg-slate-900/60">
      <div class="h-full bg-brand-500 transition-all" :style="{ width: pct + '%' }"></div>
    </div>
    <div class="flex justify-between text-xs text-slate-500">
      <span v-if="job.progress.step !== null">{{ job.progress.step }} / {{ job.progress.totalSteps }}</span>
      <span v-else>&nbsp;</span>
      <button
        v-if="job.state === 'running'"
        type="button"
        class="text-slate-400 hover:text-rose-300"
        @click="emit('cancel', job.id)"
      >✕</button>
    </div>
  </div>
</template>
```

- [ ] **Step 2: RunningStrip.vue**

```vue
<script setup lang="ts">
import type { Job } from "../composables/useGenerateStream";
import RunningJobCard from "./RunningJobCard.vue";

defineProps<{ jobs: Job[] }>();
const emit = defineEmits<{ (e: "cancel", id: string): void }>();
</script>

<template>
  <div v-if="jobs.length" class="mt-4 flex gap-3 overflow-x-auto pb-2">
    <RunningJobCard
      v-for="job in jobs"
      :key="job.id"
      :job="job"
      @cancel="(id) => emit('cancel', id)"
    />
  </div>
</template>
```

- [ ] **Step 3: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 4: Commit**

```bash
git add web/src/components/RunningJobCard.vue web/src/components/RunningStrip.vue
git commit -m "feat(web): running job card + horizontal strip"
```

---

## Task 15: Wire `GeneratePage.vue`

**Files:**
- Modify: `web/src/pages/GeneratePage.vue`.

Compose Composer + modals + RunningStrip + GalleryFeed. Reuse the gallery view/muted localStorage keys.

- [ ] **Step 1: Replace the stub with the full page**

```vue
<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Composer from "../components/Composer.vue";
import SettingsModal from "../components/SettingsModal.vue";
import ExpandModal from "../components/ExpandModal.vue";
import ImagePickerModal from "../components/ImagePickerModal.vue";
import RunningStrip from "../components/RunningStrip.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import DetailDrawer from "../components/DetailDrawer.vue";
import TopBar from "../components/TopBar.vue";
import {
  deleteGalleryImage,
  fetchCapabilities,
  fetchModels,
  listGallery,
} from "../api";
import { useGenerateForm } from "../composables/useGenerateForm";
import { useGenerateStream } from "../composables/useGenerateStream";
import { useStatusPoll } from "../composables/useStatusPoll";
import type {
  GalleryImage,
  ModelInfoExtended,
  ServerCapabilities,
  SourceImageState,
} from "../types";

type ViewMode = "feed" | "grid";

// Reuse the gallery's localStorage key so view mode is consistent across routes.
function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem("mold.gallery.view");
    return v === "grid" ? "grid" : "feed";
  } catch {
    return "feed";
  }
}
function loadMuted(): boolean {
  try {
    return localStorage.getItem("mold.gallery.muted") !== "false";
  } catch {
    return true;
  }
}

const form = useGenerateForm();
const { status } = useStatusPoll();
const models = ref<ModelInfoExtended[]>([]);
const galleryEntries = ref<GalleryImage[]>([]);
const view = ref<ViewMode>(loadViewMode());
const muted = ref(loadMuted());

const showSettings = ref(false);
const showExpand = ref(false);
const showPicker = ref(false);

const stream = useGenerateStream(async () => {
  // Refresh gallery on completion; dedupe by filename.
  try {
    galleryEntries.value = await listGallery();
  } catch {
    /* leave previous */
  }
});

async function refreshModels() {
  try {
    models.value = await fetchModels();
  } catch (e) {
    console.error(e);
  }
}

const currentModel = computed(
  () => models.value.find((m) => m.name === form.state.value.model) ?? null,
);

const gpus = computed(() =>
  status.value?.gpus?.map((g) => ({ ordinal: g.ordinal, state: g.state })) ?? null,
);

const settingsDirty = computed(() => {
  const s = form.state.value;
  const m = currentModel.value;
  if (!m) return false;
  return (
    s.width !== m.default_width ||
    s.height !== m.default_height ||
    s.steps !== m.default_steps ||
    Math.abs(s.guidance - m.default_guidance) > 0.001 ||
    s.batchSize !== 1 ||
    s.seed !== null ||
    s.negativePrompt.length > 0
  );
});

async function onSubmit() {
  // Guard: need a model.
  if (!form.state.value.model) {
    showSettings.value = true;
    return;
  }
  const req = form.toRequest();
  stream.submit(req);
}

function onClearSource() {
  form.state.value.sourceImage = null;
}

function onPickSource(v: SourceImageState) {
  form.state.value.sourceImage = v;
}

onMounted(async () => {
  await refreshModels();
  galleryEntries.value = await listGallery();
  // If form has no model yet, default to the first downloaded one.
  if (!form.state.value.model) {
    const first = models.value.find((m) => m.downloaded);
    if (first) form.applyModelDefaults(first);
  }
});
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-32 sm:px-6 lg:px-10">
    <div class="mt-4 sm:mt-6">
      <Composer
        v-model="form.state.value"
        :queue-depth="status?.queue_depth ?? null"
        :queue-capacity="status?.queue_capacity ?? null"
        :gpus="gpus"
        :expand-active="form.state.value.expand.enabled"
        :settings-dirty="settingsDirty"
        @submit="onSubmit"
        @open-settings="showSettings = true"
        @open-expand="showExpand = true"
        @open-image-picker="showPicker = true"
        @clear-source="onClearSource"
      />

      <RunningStrip :jobs="stream.jobs.value" @cancel="stream.cancel" />

      <div class="mt-6">
        <GalleryFeed :entries="galleryEntries" :loading="false" :view="view" :muted="muted" @open="() => {}" />
      </div>
    </div>

    <SettingsModal
      :open="showSettings"
      v-model="form.state.value"
      :models="models"
      @close="showSettings = false"
    />
    <ExpandModal
      :open="showExpand"
      :prompt="form.state.value.prompt"
      :expand="form.state.value.expand"
      :current-model="currentModel"
      @update:expand="(v) => (form.state.value.expand = v)"
      @apply-prompt="(v) => (form.state.value.prompt = v)"
      @close="showExpand = false"
    />
    <ImagePickerModal
      :open="showPicker"
      @pick="onPickSource"
      @close="showPicker = false"
    />
  </div>
</template>
```

- [ ] **Step 2: Verify**

```bash
cd web && bunx vue-tsc --noEmit && bun run build && bun run fmt:check
```

- [ ] **Step 3: Commit**

```bash
git add web/src/pages/GeneratePage.vue
git commit -m "feat(web): wire GeneratePage — composer, running strip, modals, feed"
```

---

## Task 16: Manual UAT against <gpu-host>

**Files:** none (live test)

The SSH tunnel is already up (`localhost:7680 → <gpu-host>:7680`). `MOLD_HOST` is set in the user's zshrc. Two models are downloaded there: `sd15:fp16` and `sdxl-turbo:fp16`.

- [ ] **Step 1: Start the dev server**

```bash
cd web && bun run dev
```

- [ ] **Step 2: Smoke test — gallery unchanged**

Visit `http://localhost:5174/`. Confirm:
- Feed loads with existing items.
- Filter / search / view toggle all still work.
- Detail drawer still opens and navigates.

- [ ] **Step 3: Smoke test — generate happy path**

Visit `http://localhost:5174/generate`. Confirm:
- Composer renders, ⚙ opens modal with `sd15:fp16` and `sdxl-turbo:fp16` in the Images section.
- Pick `sdxl-turbo:fp16`, type `a cat on a windowsill`, press Enter.
- A running card appears, stage progresses through "Loading" → "Denoising N/M" → completion.
- Result lands in the gallery feed beneath the running strip.
- Completion card fades out after ~1 second.

- [ ] **Step 4: Smoke test — concurrency**

Fire three prompts in quick succession. Confirm:
- Two running cards pick up GPU ordinals 0 and 1.
- The third shows `Queued (position 1)` until a worker frees up.
- Each card shows the correct GPU badge at completion.

- [ ] **Step 5: Smoke test — img2img via gallery**

Click 🖼️ → "From gallery" tab → pick an existing image. Confirm:
- Chip appears on composer with the thumbnail.
- Settings modal now shows the Strength slider.
- Submit generates with `source_image` and `strength` in the request.

- [ ] **Step 6: Smoke test — expand preview**

Click ✨, enable, set variations=3, click Preview. Confirm three rewrites render, clicking one replaces the composer prompt, modal closes.

- [ ] **Step 7: Smoke test — keyboard contract**

In the composer, press Enter with empty prompt → no-op. Press Shift+Enter → newline inserted. Press Enter with text → submits.

- [ ] **Step 8: Smoke test — persistence**

Reload the page. Confirm:
- Prompt, model, size, negative prompt, seed, batch all restored.
- Source image chip is cleared (we drop base64 on persist).

- [ ] **Step 9: Smoke test — 503 backpressure**

(Optional — requires driving queue to capacity.) Fire enough jobs to hit `queue_full`. Confirm a banner appears on the composer with the retry countdown.

- [ ] **Step 10: Smoke test — mobile**

Open Chrome devtools, switch to a phone viewport. Confirm:
- Composer stacks, icons flow into a row.
- Modals take the full viewport as bottom sheets.
- Running strip scrolls horizontally with snap.

- [ ] **Step 11: Document findings**

If anything fails, open a follow-up task in the plan (insert before Task 18) describing the fix. Do not proceed to PR until Steps 3–8 pass.

- [ ] **Step 12: No commit here** — this is a validation task. If fixes were needed they live in the follow-up task.

---

## Task 17: Prettier + type sweep + build

**Files:** tbd (whatever needs formatting)

- [ ] **Step 1: Run the formatter**

```bash
cd web && bun run fmt
```

- [ ] **Step 2: Re-run type check and build**

```bash
cd web && bunx vue-tsc --noEmit && bun run build
```

- [ ] **Step 3: Commit any fmt-only diff**

```bash
git add -u
git diff --cached --quiet || git commit -m "chore(web): prettier sweep"
```

---

## Task 18: Commit the spec + plan and push

**Files:**
- Add: `tasks/web-generate-ui-spec.md`.
- Add: `tasks/web-generate-ui-plan.md`.

- [ ] **Step 1: Confirm branch**

```bash
git status && git rev-parse --abbrev-ref HEAD
```

Expected: `feat/web-generate-ui`.

- [ ] **Step 2: Commit the design docs**

```bash
git add tasks/web-generate-ui-spec.md tasks/web-generate-ui-plan.md .gitignore
git commit -m "docs(web): spec + implementation plan for /generate UI"
```

- [ ] **Step 3: Push**

```bash
git push -u origin feat/web-generate-ui
```

---

## Task 19: Open the PR

**Files:** none.

- [ ] **Step 1: Create PR with multi-gpu as base**

```bash
gh pr create --base multi-gpu --title "feat(web): browser-driven generation UI at /generate" --body "$(cat <<'EOF'
## Summary
- Adds a new `/generate` route in the web SPA with a minimal composer (Enter submits, Shift+Enter newline), per-GPU running-job cards streaming SSE progress, and the existing gallery feed reused beneath.
- Concurrency governed by server queue backpressure (503 `queue_full`). No server-side changes.
- img2img via upload or From Gallery picker; video-family models grouped with a `🎬` badge and clamped frames (8n+1).
- Prompt expansion with live preview + variation picker; server-side knobs (LLM, temperature, thinking) noted as future work.

## Design docs
- Spec: `tasks/web-generate-ui-spec.md`
- Plan: `tasks/web-generate-ui-plan.md`

## Test plan
- [x] Gallery page renders and behaves identically to before (filter, search, view toggle, detail drawer).
- [x] `/generate` happy-path: type prompt, press Enter, watch card stream to completion, result lands in feed.
- [x] Concurrency: three rapid submissions pick up GPU 0/1 plus a queued job.
- [x] img2img via gallery pick: source chip shown, Strength slider appears, generation runs.
- [x] Expand preview: 3 variations render, click one replaces prompt.
- [x] Keyboard contract: Shift+Enter newline, Enter submits.
- [x] localStorage persist across reload (base64 dropped).
- [x] Mobile viewport: composer stacks, modals become sheets, running strip scrolls.

## Not in this PR (tracked in spec §12)
- LoRA, inpainting mask, ControlNet, LTX-2 advanced modes, post-generate upscale, pull-on-demand, per-LLM expand knobs, edit_images multi-reference.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Copy the PR URL to the user**

---

## Self-review (already completed)

**Spec coverage:** Every spec section has a corresponding task —
- §5.1 Routing → Task 1, 2
- §5.3 Composer → Task 9 (uses types from 3, composable from 6)
- §5.4 Running strip → Task 14 (uses composable from 8)
- §5.5 Settings modal → Task 11 (uses picker from 10, types from 3)
- §5.6 Expand modal → Task 12
- §5.7 Image picker → Task 13
- §5.8 Gallery feed reuse → Task 15
- §5.9 Responsive → addressed in Composer (9), modals (11-13), running strip (14); validated in Task 16 step 10
- §6 API surface → Tasks 4, 5
- §6.2 Errors → 503 handling in Task 8 + validated in Task 16 step 9; stream-disconnect display in Task 14
- §7 State model → Task 6 (form), Task 8 (jobs), Task 15 (gallery)
- §8 Persistence → Task 6 (form) + Task 10 (show-all toggle)
- §11 Open questions → v2 only
- §12 Future work → documented in spec, linked from PR body

**Placeholder scan:** every step has explicit code or an exact command.

**Type consistency:** `GenerateFormState`, `Job`, `SseProgressEvent`, `SseCompleteEvent` appear across Tasks 3, 6, 8, 9, 11, 14 with identical field names.

**Scope:** one feature, one PR, no decomposition needed.
