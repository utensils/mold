import { ref } from "vue";
import { ApiError, type ApiTarget } from "../api/client";
import {
  acceptAndDownload,
  type LicenseDownloadProgress,
} from "../api/licenseAcceptance";
import {
  licenseFromErrorBody,
  type LicenseRequirement,
} from "../lib/licenseAcceptance";

export interface LicensePrompt {
  hostLabel: string;
  target: ApiTarget;
  requirements: LicenseRequirement[];
}

const pending = ref<LicensePrompt | null>(null);
const progress = ref<LicenseDownloadProgress | null>(null);
const error = ref<string | null>(null);
const busy = ref(false);
let settle: ((accepted: boolean) => void) | null = null;
let controller: AbortController | null = null;

export function useLicenseAcceptance() {
  async function request(prompt: LicensePrompt): Promise<boolean> {
    if (prompt.requirements.length === 0) return true;
    if (pending.value)
      throw new Error("Another license review is already open.");
    pending.value = prompt;
    progress.value = null;
    error.value = null;
    return new Promise<boolean>((resolve) => {
      settle = resolve;
    });
  }

  function close(accepted: boolean) {
    const resolve = settle;
    settle = null;
    pending.value = null;
    progress.value = null;
    error.value = null;
    busy.value = false;
    controller = null;
    resolve?.(accepted);
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
      for (const requirement of prompt.requirements) {
        await acceptAndDownload(
          prompt.target,
          requirement,
          (next) => {
            progress.value = next;
          },
          controller.signal,
        );
      }
      close(true);
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
