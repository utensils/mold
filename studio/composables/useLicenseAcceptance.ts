import { ref } from "vue";
import { ApiError, type ApiTarget } from "../api/client";
import {
  acceptAndDownload,
  recordLicenseAcceptances,
  type LicenseDownloadProgress,
} from "../api/licenseAcceptance";
import {
  licenseFromErrorBody,
  licenseRequirementFromError,
  type LicenseRequirement,
} from "../lib/licenseAcceptance";

export interface LicensePrompt {
  hostLabel: string;
  target: ApiTarget;
  requirements: LicenseRequirement[];
  /** `"download"` (the default) accepts and fetches the bundle on this host —
   * the generate-time and settings behaviour. `"record"` accepts only, leaving
   * the caller to re-drive its own enqueue so the job lands in that surface's
   * normal downloads queue. */
  intent?: "download" | "record";
}

export interface LicenseConsentOutcome {
  accepted: boolean;
  /** True when this host also fetched the consented bundle, so the caller must
   * NOT enqueue it again. False after a record-only acceptance. */
  downloaded: boolean;
}

const pending = ref<LicensePrompt | null>(null);
const progress = ref<LicenseDownloadProgress | null>(null);
const error = ref<string | null>(null);
const busy = ref(false);
let settle: ((outcome: LicenseConsentOutcome) => void) | null = null;
let controller: AbortController | null = null;

export function useLicenseAcceptance() {
  async function request(prompt: LicensePrompt): Promise<LicenseConsentOutcome> {
    if (prompt.requirements.length === 0)
      return { accepted: true, downloaded: false };
    if (pending.value)
      throw new Error("Another license review is already open.");
    pending.value = prompt;
    progress.value = null;
    error.value = null;
    return new Promise<LicenseConsentOutcome>((resolve) => {
      settle = resolve;
    });
  }

  function close(accepted: boolean, downloaded = false) {
    const resolve = settle;
    settle = null;
    pending.value = null;
    progress.value = null;
    error.value = null;
    busy.value = false;
    controller = null;
    resolve?.({ accepted, downloaded });
  }

  function cancel() {
    if (busy.value) {
      controller?.abort();
    } else {
      close(false);
    }
  }

  async function accept() {
    const prompt = pending.value;
    if (!prompt || busy.value) return;
    busy.value = true;
    controller = new AbortController();
    error.value = null;
    try {
      let downloaded = prompt.intent !== "record";
      for (const requirement of prompt.requirements) {
        if (prompt.intent === "record") {
          try {
            await recordLicenseAcceptances(prompt.target, requirement.licenses);
            continue;
          } catch (cause) {
            // A host predating the standalone accept route can only record
            // consent as a side effect of a download, so take that path and
            // tell the caller the bundle is already on its way.
            const missingRoute =
              cause instanceof ApiError &&
              (cause.status === 404 || cause.status === 405);
            if (!missingRoute) throw cause;
            downloaded = true;
          }
        }
        await acceptAndDownload(
          prompt.target,
          requirement,
          (next) => {
            progress.value = next;
          },
          controller.signal,
        );
      }
      close(true, downloaded);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        close(false);
        return;
      }
      if (cause instanceof ApiError) {
        const current = licenseFromErrorBody(cause.body);
        if (current) {
          const requirement = prompt.requirements.find((row) =>
            row.licenses.some((license) => license.id === current.id),
          );
          if (requirement) {
            requirement.licenses = requirement.licenses.map((license) =>
              license.id === current.id ? current : license,
            );
          }
          error.value =
            "The host pins newer terms. Review the updated terms before accepting.";
        } else {
          error.value = cause.message;
        }
      } else {
        error.value = cause instanceof Error ? cause.message : String(cause);
      }
      busy.value = false;
      controller = null;
      progress.value = null;
    }
  }

  return { pending, progress, error, busy, request, cancel, accept };
}

export type LicenseGatedOutcome<T> =
  | { kind: "ok"; value: T }
  | { kind: "accepted" }
  | { kind: "declined" };

/** Serializes licence reviews across every surface.
 *
 * `request()` deliberately refuses a second concurrent prompt, and a batch
 * install fires N enqueues at once — so consent has to queue here rather than
 * throw there.
 */
let consentGate: Promise<unknown> = Promise.resolve();

/** Run an install, and if the host refuses it pending licence acceptance, take
 * consent and run it again.
 *
 * Every model-manager pull previously threw the structured 403 away and showed
 * a raw error string (or, on one path, nothing at all). This is that missing
 * step, in one place instead of pasted into each call site — and it stays
 * registry-blind: `installModel` is whatever the caller already asked for.
 */
export async function runWithLicenseConsent<T>(options: {
  hostLabel: string;
  target: ApiTarget;
  installModel: string;
  start: () => Promise<T>;
}): Promise<LicenseGatedOutcome<T>> {
  const requirementFor = (error: unknown): LicenseRequirement | null =>
    error instanceof ApiError
      ? licenseRequirementFromError(error.body, options.installModel)
      : null;

  try {
    return { kind: "ok", value: await options.start() };
  } catch (error) {
    // Anything that is not a licence refusal — a 409 already-queued, an
    // unknown model — belongs to the caller exactly as before.
    if (!requirementFor(error)) throw error;
  }

  const run = async (): Promise<LicenseGatedOutcome<T>> => {
    // Re-drive first: a batch sharing one licence produces N concurrent
    // refusals, and after the first acceptance the rest simply succeed. This
    // collapses duplicate prompts with no knowledge of what was refused.
    let requirement: LicenseRequirement | null = null;
    try {
      return { kind: "ok", value: await options.start() };
    } catch (error) {
      requirement = requirementFor(error);
      if (!requirement) throw error;
    }

    const prompt = useLicenseAcceptance();
    const outcome = await prompt.request({
      hostLabel: options.hostLabel,
      target: options.target,
      requirements: [requirement],
      intent: "record",
    });
    if (!outcome.accepted) return { kind: "declined" };
    // An older host recorded consent by downloading; re-enqueueing would
    // duplicate a transfer already finished.
    if (outcome.downloaded) return { kind: "accepted" };
    // Exactly once more. A second refusal means the host re-pinned its terms
    // mid-flight, which the caller must see rather than loop on.
    return { kind: "ok", value: await options.start() };
  };

  const attempt = consentGate.then(run, run);
  consentGate = attempt.catch(() => undefined);
  return attempt;
}
