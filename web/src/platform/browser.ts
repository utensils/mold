import type { StudioPlatform, StudioStorage } from "@studio/platform";
import { getHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";

const storage: StudioStorage = {
  get: (key) => localStorage.getItem(key),
  set: (key, value) => localStorage.setItem(key, value),
  remove: (key) => localStorage.removeItem(key),
};

export const browserPlatform: StudioPlatform = {
  kind: "web",
  storage,
  targetFor(hostId) {
    const host = getHost(hostId);
    if (!host) return null;
    return {
      baseUrl: host.id === ORIGIN_HOST_ID ? "" : host.url,
      apiKey: host.apiKey ?? null,
    };
  },
  async notify(title, body) {
    if (!("Notification" in window) || Notification.permission !== "granted")
      return;
    new Notification(title, { body });
  },
  async chooseImage() {
    return null;
  },
  async saveMedia(filename, bytes) {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const href = URL.createObjectURL(new Blob([copy.buffer]));
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(href);
  },
  async copyText(text) {
    await navigator.clipboard.writeText(text);
  },
  async openExternal(url) {
    window.open(url, "_blank", "noopener,noreferrer");
  },
};
