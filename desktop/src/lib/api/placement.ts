import { ApiError, apiFetch, apiJson } from "./client";
import type { DevicePlacement } from "../placement";

/**
 * Read the saved per-model placement default. Returns `null` when nothing is
 * saved (server 404), so an editor can fall back to Auto defaults without
 * mistaking "unset" for an error.
 */
export async function getModelPlacement(model: string): Promise<DevicePlacement | null> {
  try {
    return await apiJson<DevicePlacement>(
      `/api/config/model/${encodeURIComponent(model)}/placement`,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

/** Persist a per-model placement default (config.toml, applied at next load). */
export async function putModelPlacement(model: string, placement: DevicePlacement): Promise<void> {
  await apiFetch(`/api/config/model/${encodeURIComponent(model)}/placement`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(placement),
  });
}

/** Clear a per-model placement default. */
export async function deleteModelPlacement(model: string): Promise<void> {
  await apiFetch(`/api/config/model/${encodeURIComponent(model)}/placement`, {
    method: "DELETE",
  });
}
