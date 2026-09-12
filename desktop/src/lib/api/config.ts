import {
  listConfig,
  listProfiles,
  resetConfig as resetConfigOn,
  setConfig as setConfigOn,
  switchProfile,
} from "@studio/api/config";
import { currentTarget } from "./client";
import type { ConfigProfiles, ConfigRow } from "./types";

/**
 * Config surface, bound to whichever engine this app is connected to.
 *
 * The wire logic — the three listing shapes, the escaping, the profile
 * defaults — is `@studio/api/config`, shared with the web app. This file is
 * only the binding: it names the target so the rest of the app does not have
 * to carry one around.
 */

export function fetchConfig(): Promise<ConfigRow[]> {
  return listConfig(currentTarget());
}

export function setConfig(key: string, value: ConfigRow["value"]): Promise<void> {
  return setConfigOn(currentTarget(), key, value);
}

export function resetConfig(key: string): Promise<void> {
  return resetConfigOn(currentTarget(), key);
}

export function fetchProfiles(): Promise<ConfigProfiles> {
  return listProfiles(currentTarget());
}

export function setProfile(name: string): Promise<void> {
  return switchProfile(currentTarget(), name);
}
