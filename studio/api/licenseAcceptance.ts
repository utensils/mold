import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "./client";
import {
  acceptanceFor,
  licenseFromErrorBody,
  type LicenseListing,
  type LicenseRequirement,
  type LicenseTerms,
} from "../lib/licenseAcceptance";

interface CreateDownloadResponse {
  id: string;
  position: number;
}

interface DownloadJob {
  id: string;
  model: string;
  status: "queued" | "active" | "completed" | "failed" | "cancelled";
  bytes_done: number;
  bytes_total: number;
  error?: string | null;
}

interface DownloadsListing {
  active_jobs?: DownloadJob[];
  active?: DownloadJob | null;
  queued: DownloadJob[];
  history: DownloadJob[];
}

export interface LicenseDownloadProgress {
  model: string;
  status: DownloadJob["status"] | "starting";
  bytesDone: number;
  bytesTotal: number;
}

export async function fetchLicenseListing(
  target: ApiTarget,
): Promise<LicenseListing> {
  return apiJsonTo<LicenseListing>(target, "/api/licenses");
}

/** Record acceptance on one host WITHOUT downloading the bundle.
 *
 * Consent and acquisition are different acts. `acceptAndDownload` below is the
 * right call when the user asked for the weights; this one is for a client
 * that will re-drive its OWN enqueue afterwards, so the job lands in that
 * surface's normal downloads queue instead of behind the modal's progress bar.
 *
 * Throws `ApiError` 404/405 on a host predating the route, which callers use
 * to fall back to `acceptAndDownload`.
 */
export async function recordLicenseAcceptances(
  target: ApiTarget,
  licenses: readonly LicenseTerms[],
  signal?: AbortSignal,
): Promise<LicenseListing> {
  try {
    const response = await apiFetchTo(target, "/api/licenses/accept", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      ...(signal ? { signal } : {}),
      body: JSON.stringify({ accept_licenses: licenses.map(acceptanceFor) }),
    });
    return (await response.json()) as LicenseListing;
  } catch (error) {
    // Normalize a re-pinned-terms conflict exactly as acceptAndDownload does,
    // so the composable's single mismatch handler covers both intents.
    if (error instanceof ApiError) {
      const current = licenseFromErrorBody(error.body);
      if (current) {
        throw new ApiError(error.message, error.status, {
          ...(error.body as object),
          license: current,
        });
      }
    }
    throw error;
  }
}

function jobs(listing: DownloadsListing): DownloadJob[] {
  const active =
    listing.active_jobs ?? (listing.active ? [listing.active] : []);
  return [...active, ...listing.queued, ...listing.history];
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Download cancelled.", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

async function cancelDownload(target: ApiTarget, id: string) {
  try {
    await apiFetchTo(target, `/api/downloads/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  } catch (error) {
    // Completion and cancellation can race. A missing terminal job needs no
    // further action; every other host failure is still superseded by the
    // user's explicit cancellation.
    if (!(error instanceof ApiError && error.status === 404)) throw error;
  }
}

/** Accept exact pinned terms on one host, download the owning bundle there,
 * and resolve only when that host reports the job terminal. */
export async function acceptAndDownload(
  target: ApiTarget,
  requirement: LicenseRequirement,
  onProgress: (progress: LicenseDownloadProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  onProgress({
    model: requirement.installModel,
    status: "starting",
    bytesDone: 0,
    bytesTotal: 0,
  });
  let created: CreateDownloadResponse;
  try {
    const response = await apiFetchTo(target, "/api/downloads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      ...(signal ? { signal } : {}),
      body: JSON.stringify({
        model: requirement.installModel,
        accept_licenses: requirement.licenses.map(acceptanceFor),
      }),
    });
    created = (await response.json()) as CreateDownloadResponse;
  } catch (error) {
    if (error instanceof ApiError) {
      if (
        error.status === 409 &&
        typeof error.body === "object" &&
        error.body !== null &&
        typeof (error.body as Record<string, unknown>).id === "string"
      ) {
        created = error.body as unknown as CreateDownloadResponse;
      } else {
        const current = licenseFromErrorBody(error.body);
        if (current) {
          throw new ApiError(error.message, error.status, {
            ...(error.body as object),
            license: current,
          });
        }
        throw error;
      }
    } else {
      throw error;
    }
  }
  let missingSnapshots = 0;
  try {
    for (;;) {
      if (signal?.aborted) {
        await cancelDownload(target, created.id);
        throw new DOMException("Download cancelled.", "AbortError");
      }
      const listing = await apiJsonTo<DownloadsListing>(
        target,
        "/api/downloads",
        {
          ...(signal ? { signal } : {}),
        },
      );
      const job = jobs(listing).find(
        (candidate) => candidate.id === created.id,
      );
      if (job) {
        missingSnapshots = 0;
        onProgress({
          model: requirement.installModel,
          status: job.status,
          bytesDone: job.bytes_done,
          bytesTotal: job.bytes_total,
        });
        if (job.status === "completed") return;
        if (job.status === "failed" || job.status === "cancelled") {
          throw new Error(job.error || `Download ${job.status}.`);
        }
      } else {
        missingSnapshots += 1;
        if (missingSnapshots >= 10) {
          throw new Error(
            `The host no longer reports download '${created.id}'. Retry the license download.`,
          );
        }
      }
      await wait(500, signal);
    }
  } catch (error) {
    if (signal?.aborted) {
      await cancelDownload(target, created.id).catch(() => undefined);
      throw new DOMException("Download cancelled.", "AbortError");
    }
    throw error;
  }
}
