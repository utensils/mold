import type { ApiTarget } from "./api/client";

export interface StudioStorage {
  get(key: string): string | null;
  set(key: string, value: string): void;
  remove(key: string): void;
}

export interface StudioPlatform {
  readonly kind: "web" | "desktop" | "mobile";
  readonly storage: StudioStorage;
  targetFor(hostId: string): ApiTarget | null;
  notify(title: string, body: string): Promise<void>;
  chooseImage(): Promise<string | null>;
  saveMedia(filename: string, bytes: Uint8Array): Promise<void>;
  copyText(text: string): Promise<void>;
  openExternal(url: string): Promise<void>;
}

let activePlatform: StudioPlatform | null = null;

export function installStudioPlatform(platform: StudioPlatform): void {
  activePlatform = platform;
}

export function studioPlatform(): StudioPlatform {
  if (!activePlatform)
    throw new Error("Studio platform adapter is not installed");
  return activePlatform;
}
