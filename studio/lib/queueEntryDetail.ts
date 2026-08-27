/**
 * Everything one server-queue row can say about itself, decided once for web,
 * desktop, and iPhone.
 *
 * `GET /api/queue` is two authorities merged: the bounded live registry, whose
 * rows carry the submitted `metadata`, and the durable SQLite projection,
 * which is deliberately payload-free (`QUEUE_PROJECTION_FIRST_PAGE_SQL` never
 * selects `request_json`, and a regression test proves it). A job admitted
 * through `POST /api/generation-batches` is journalled first and hydrated into
 * the registry by the feeder, so between admission and dispatch — precisely
 * when someone opens a queued job to look at it — the listing has no request
 * to show. That is a gap in the wire, not an old host, so the copy below says
 * what is actually true and never asks anyone to upgrade anything.
 *
 * Two things close most of it client-side: this app already holds the exact
 * request for work IT submitted (`localMetadata`), and every fact the durable
 * projection DOES carry — state, position, durability, replay, dispatch
 * attempts, the hold reason, and the plan's own lane and estimate — is worth
 * rendering on its own.
 */

import type { QueueEntry, QueuePlan, QueueWorkItem } from "../api/queuePlan";
import {
  normalizeBlockedReason,
  queueWaitCode,
  queueWaitLabel,
  resolveQueueWait,
} from "./queuePosition";

/**
 * The structural subset of `OutputMetadata` this model reads. Desktop, web,
 * and iPhone each own their own full type; all three satisfy this, so the
 * shared model never needs a fourth copy of the wire shape.
 */
export interface QueueDetailMetadata {
  prompt?: string | null;
  negative_prompt?: string | null;
  title?: string | null;
  tags?: readonly string[] | null;
  collection?: string | null;
  model?: string | null;
  seed?: number | null;
  steps?: number | null;
  guidance?: number | null;
  width?: number | null;
  height?: number | null;
  frames?: number | null;
  fps?: number | null;
  strength?: number | null;
  output_format?: string | null;
  scheduler?: unknown;
  pipeline?: string | null;
  source_image_name?: string | null;
  source_image_sha256?: string | null;
  id_image_name?: string | null;
  id_weight?: number | null;
  id_start_step?: number | null;
  extend_overlap_frames?: number | null;
  batch_index?: number | null;
  batch_count?: number | null;
  version?: string | null;
  loras?: readonly { path: string; scale: number }[] | null;
  lora?: string | null;
  lora_scale?: number | null;
}

export type QueueDetailMetadataSource = "host" | "local";

export interface QueueDetailField {
  label: string;
  value: string;
  /** Render in the mono face — a number, an identifier, or a dimension. */
  mono?: boolean;
}

export interface QueueDetailGroup {
  title: string;
  fields: QueueDetailField[];
}

export interface QueueDetailAction {
  /** Whether this action exists for a row in this state at all. */
  applicable: boolean;
  /** Whether it can be taken right now. */
  available: boolean;
  /** Why not, in a sentence, whenever it is applicable but unavailable. */
  blockedReason: string | null;
}

export interface QueueDetailProblem {
  title: string;
  detail: string;
}

export interface QueueEntryDetailModel {
  jobId: string;
  /** Raw id retained for requests, matching, and persistence. */
  modelId: string;
  /** Resolved through the host's inventory for display only. */
  modelLabel: string;
  hostLabel: string;
  stateLabel: string;
  stateCode: string;
  waitLabel: string;
  running: boolean;
  held: boolean;
  prompt: string | null;
  negativePrompt: string | null;
  title: string | null;
  groups: QueueDetailGroup[];
  facts: QueueDetailField[];
  metadataSource: QueueDetailMetadataSource | null;
  /** Honest explanation when no request is available, else null. */
  settingsNotice: string | null;
  problem: QueueDetailProblem | null;
  /** Whole, untruncated text for the Copy control. */
  copyText: string;
  reuse: QueueDetailAction;
  cancel: QueueDetailAction;
  retry: QueueDetailAction;
  /** Whether to poll `GET /api/queue/:id/preview` for this row. */
  preview: boolean;
}

/**
 * Said when the host listed a durable row whose request it has not loaded.
 * Deliberately describes the machine's own state — an "upgrade the server"
 * sentence was both wrong (client and server ship together) and unactionable.
 */
export const QUEUE_SETTINGS_PENDING_NOTICE =
  "Settings appear once this machine loads the job — until then it lists the queue row without its request.";

const RUNNING_CANCEL_UNSUPPORTED =
  "This machine cannot stop a running job; it can only be cancelled while it waits.";
const RETRY_NEEDS_AUTHORITY =
  "Only the app that submitted this job can retry it — its batch authority lives with that client.";

export interface QueueEntryDetailInput {
  entry: QueueEntry;
  hostLabel: string;
  modelLabel: string;
  nowMs: number;
  /** The whole host queue plan; the row's own work item is picked out of it. */
  plan?: QueuePlan | null;
  /** Settings the host listed with the row. */
  metadata?: QueueDetailMetadata | null;
  /** The exact request this app submitted, when this row is its own work. */
  localMetadata?: QueueDetailMetadata | null;
  /** Whether this app submitted the row. */
  mine?: boolean;
  /** `capabilities.queue.cooperative_cancellation` for this host. */
  canCancelRunning?: boolean;
  /** Present only when this client holds the row's durable batch authority. */
  retryAuthority?: unknown;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function numberField(label: string, value: unknown): QueueDetailField | null {
  const resolved = finite(value);
  return resolved === null
    ? null
    : { label, value: String(resolved), mono: true };
}

function present(fields: (QueueDetailField | null)[]): QueueDetailField[] {
  return fields.filter((field): field is QueueDetailField => field !== null);
}

function group(
  title: string,
  fields: (QueueDetailField | null)[],
): QueueDetailGroup | null {
  const resolved = present(fields);
  return resolved.length > 0 ? { title, fields: resolved } : null;
}

function schedulerLabel(value: unknown): string | null {
  if (typeof value === "string") return text(value);
  if (typeof value === "object" && value !== null) {
    const [name] = Object.keys(value as Record<string, unknown>);
    return name ? name : null;
  }
  return null;
}

function loraFields(
  metadata: QueueDetailMetadata,
): (QueueDetailField | null)[] {
  const stack = metadata.loras ?? [];
  if (stack.length > 0) {
    return stack.map((lora, index) => ({
      label: stack.length > 1 ? `LoRA ${index + 1}` : "LoRA",
      value: `${lora.path.split("/").pop() ?? lora.path} · ${lora.scale}`,
      mono: true,
    }));
  }
  const single = text(metadata.lora);
  if (single === null) return [];
  const scale = finite(metadata.lora_scale);
  return [
    {
      label: "LoRA",
      value:
        scale === null
          ? (single.split("/").pop() ?? single)
          : `${single.split("/").pop() ?? single} · ${scale}`,
      mono: true,
    },
  ];
}

function workItemFor(
  plan: QueuePlan | null | undefined,
  jobId: string,
): QueueWorkItem | null {
  for (const item of plan?.work_items ?? []) {
    if (item.parent_id === jobId || item.work_id === jobId) return item;
  }
  return null;
}

function laneLabel(item: QueueWorkItem): string | null {
  if (item.gpu != null) return `GPU ${item.gpu}`;
  if (item.planned_lane_kind === "host_utility") return "CPU";
  if (item.planned_device_id) return item.planned_device_id;
  return null;
}

/** Coarse relative time; the drawer re-renders on every queue poll. */
function relative(deltaMs: number): string {
  const seconds = Math.max(0, Math.round(deltaMs / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function timestamp(unixMs: number): string {
  return new Date(unixMs).toLocaleString();
}

function seedField(
  entry: QueueEntry,
  metadata: QueueDetailMetadata,
): QueueDetailField | null {
  const seed = finite(metadata.seed);
  if (seed === null) return null;
  // Seed 0 on the wire means "not pinned" on hosts that predate `seed_pinned`.
  const pinned = entry.seed_pinned ?? seed !== 0;
  return { label: "Seed", value: pinned ? String(seed) : "Random", mono: true };
}

function settingsGroups(
  entry: QueueEntry,
  metadata: QueueDetailMetadata,
): QueueDetailGroup[] {
  const width = finite(metadata.width);
  const height = finite(metadata.height);
  const frames = finite(metadata.frames);
  const fps = finite(metadata.fps);
  const batchIndex = finite(metadata.batch_index);
  const batchCount = finite(metadata.batch_count);
  return present([
    group("Output", [
      width !== null && height !== null
        ? { label: "Size", value: `${width}×${height}`, mono: true }
        : null,
      frames === null
        ? null
        : {
            label: "Frames",
            value: fps === null ? String(frames) : `${frames} @ ${fps} fps`,
            mono: true,
          },
      text(metadata.output_format) === null
        ? null
        : { label: "Format", value: text(metadata.output_format) as string },
      text(metadata.pipeline) === null
        ? null
        : { label: "Pipeline", value: text(metadata.pipeline) as string },
    ]),
    group("Sampling", [
      numberField("Steps", metadata.steps),
      numberField("Guidance", metadata.guidance),
      seedField(entry, metadata),
      schedulerLabel(metadata.scheduler) === null
        ? null
        : {
            label: "Scheduler",
            value: schedulerLabel(metadata.scheduler) as string,
          },
      numberField("Strength", metadata.strength),
    ]),
    group("Conditioning", [
      text(metadata.source_image_name) === null
        ? null
        : {
            label: "Source",
            value: text(metadata.source_image_name) as string,
          },
      text(metadata.id_image_name) === null
        ? null
        : { label: "Identity", value: text(metadata.id_image_name) as string },
      numberField("Identity strength", metadata.id_weight),
      numberField("Identity start step", metadata.id_start_step),
      numberField("Extend overlap", metadata.extend_overlap_frames),
      ...loraFields(metadata),
    ]),
    group("File under", [
      text(metadata.title) === null
        ? null
        : { label: "Title", value: text(metadata.title) as string },
      (metadata.tags ?? []).length > 0
        ? { label: "Tags", value: (metadata.tags as string[]).join(", ") }
        : null,
      text(metadata.collection) === null
        ? null
        : { label: "Collection", value: text(metadata.collection) as string },
    ]),
    group("Provenance", [
      batchIndex !== null && batchCount !== null
        ? {
            label: "Batch",
            value: `${batchIndex + 1} of ${batchCount}`,
            mono: true,
          }
        : null,
      text(metadata.version) === null
        ? null
        : {
            label: "mold",
            value: text(metadata.version) as string,
            mono: true,
          },
    ]),
  ]).filter(
    (entryGroup): entryGroup is QueueDetailGroup => entryGroup !== null,
  );
}

function problemFor(entry: QueueEntry): QueueDetailProblem | null {
  const held = text(entry.held_reason);
  const error = text(entry.error);
  if (held === null && error === null) return null;
  const parts =
    held !== null && error !== null && held !== error
      ? [held, error]
      : [(error ?? held) as string];
  return {
    title: entry.state === "held" ? "Held by the machine" : "Reported problem",
    detail: parts.join("\n\n"),
  };
}

export function queueEntryDetailModel(
  input: QueueEntryDetailInput,
): QueueEntryDetailModel {
  const { entry, hostLabel, modelLabel, nowMs } = input;
  const running = entry.state === "running";
  const held = entry.state === "held";
  const item = workItemFor(input.plan, entry.id);
  const wait = resolveQueueWait({
    position: entry.position,
    blockedReason: item?.blocked_reason ?? item?.reason,
  });

  const hostMetadata =
    typeof input.metadata === "object" && input.metadata !== null
      ? input.metadata
      : null;
  const metadata = hostMetadata ?? input.localMetadata ?? null;
  const metadataSource: QueueDetailMetadataSource | null = hostMetadata
    ? "host"
    : input.localMetadata
      ? "local"
      : null;

  const problem = problemFor(entry);
  const groups = metadata ? settingsGroups(entry, metadata) : [];

  const facts = present([
    { label: "Host", value: hostLabel },
    {
      label: "Owner",
      value: input.mine ? "This app" : "Another client",
    },
    { label: "Submitted", value: timestamp(entry.started_at_unix_ms) },
    running
      ? {
          label: "Elapsed",
          value: relative(nowMs - entry.started_at_unix_ms),
          mono: true,
        }
      : null,
    entry.durable == null
      ? null
      : { label: "Durable", value: entry.durable ? "Yes" : "No" },
    entry.replayed ? { label: "Replayed", value: "Yes" } : null,
    entry.dispatch_attempts == null
      ? null
      : {
          label: "Dispatch attempts",
          value: String(entry.dispatch_attempts),
          mono: true,
        },
    item && laneLabel(item) !== null
      ? { label: "Lane", value: laneLabel(item) as string }
      : null,
    item?.estimated_start_unix_ms != null && !running
      ? {
          label: "Starts in",
          value: relative(item.estimated_start_unix_ms - nowMs),
          mono: true,
        }
      : null,
    item?.estimated_finish_unix_ms != null
      ? {
          label: "Finishes in",
          value: relative(item.estimated_finish_unix_ms - nowMs),
          mono: true,
        }
      : null,
  ]);

  const cancel: QueueDetailAction = running
    ? {
        applicable: true,
        available: input.canCancelRunning === true,
        blockedReason:
          input.canCancelRunning === true ? null : RUNNING_CANCEL_UNSUPPORTED,
      }
    : { applicable: true, available: true, blockedReason: null };

  const retryApplicable = held && entry.retryable === true;
  const retry: QueueDetailAction = {
    applicable: retryApplicable,
    available: retryApplicable && input.retryAuthority != null,
    blockedReason:
      retryApplicable && input.retryAuthority == null
        ? RETRY_NEEDS_AUTHORITY
        : null,
  };

  const copyLines = present([
    { label: "Job", value: entry.id },
    { label: "Model", value: entry.model },
    { label: "Host", value: hostLabel },
    { label: "State", value: entry.state },
    problem ? { label: "Problem", value: problem.detail } : null,
    text(metadata?.prompt) === null
      ? null
      : { label: "Prompt", value: text(metadata?.prompt) as string },
  ])
    .map((field) => `${field.label}: ${field.value}`)
    .concat(
      groups.flatMap((entryGroup) =>
        entryGroup.fields.map(
          (field) => `${entryGroup.title} · ${field.label}: ${field.value}`,
        ),
      ),
    );

  return {
    jobId: entry.id,
    modelId: entry.model,
    modelLabel,
    hostLabel,
    stateLabel: running ? "Running" : held ? "Held" : "Queued",
    stateCode: running
      ? entry.gpu != null
        ? `RUNNING · GPU ${entry.gpu}`
        : "RUNNING"
      : held
        ? "HELD"
        : queueWaitCode(wait),
    waitLabel: running
      ? "Running"
      : held
        ? (normalizeBlockedReason(entry.held_reason) ?? "Held")
        : queueWaitLabel(wait),
    running,
    held,
    prompt: text(metadata?.prompt),
    negativePrompt: text(metadata?.negative_prompt),
    title: text(metadata?.title),
    groups,
    facts,
    metadataSource,
    settingsNotice: metadata ? null : QUEUE_SETTINGS_PENDING_NOTICE,
    problem,
    copyText: copyLines.join("\n"),
    reuse: {
      applicable: true,
      available: metadata !== null,
      blockedReason: metadata === null ? QUEUE_SETTINGS_PENDING_NOTICE : null,
    },
    cancel,
    retry,
    preview: running,
  };
}
