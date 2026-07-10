import { apiFetch } from "./client";
import type { DevicePlacement } from "../placement";

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
