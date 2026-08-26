import { defineStore } from "pinia";
import { useNotificationsStore, type NotificationAction } from "@studio/stores/notifications";

/** A labeled affordance rendered as a button on the toast (e.g. Undo, View). */
export interface ToastAction {
  label: string;
  run: () => void;
}

export interface Toast {
  id: number;
  message: string;
  /** Severity: green success, yellow warning, red error; info stays neutral. */
  kind: "info" | "success" | "warning" | "error";
  /** Optional supporting copy rendered below the primary message. */
  description?: string;
  /** Optional trailing action button (Undo, View…). Running it dismisses the toast. */
  action?: ToastAction;
  /** Optional body-click handler (e.g. navigate). Defaults to dismiss. */
  onClick?: () => void;
  /** Sticky toasts never auto-dismiss and survive the 3-toast overflow trim. */
  sticky?: boolean;
}

export interface ToastOptions {
  description?: string;
  action?: ToastAction;
  onClick?: () => void;
  sticky?: boolean;
  /** Override the 4 s default lifetime (e.g. a 6 s undo window). Ignored when sticky. */
  durationMs?: number;
  /** Recovery action retained in the notifications center after this toast leaves. */
  notificationAction?: NotificationAction;
}

let nextId = 1;

/** Default lifetime; the undo window overrides it (§08 G12). */
const DEFAULT_DURATION_MS = 4000;

/** Toasts drop in at the top right, newest first; 4 s, max 3 on screen (design spec §6). */
export const useToastStore = defineStore("toasts", {
  state: () => ({ items: [] as Toast[] }),
  getters: {
    /** True while the given toast is still on screen — undo callers poll this. */
    has: (state) => (id: number) => state.items.some((t) => t.id === id),
  },
  actions: {
    /** Queue a toast; returns its id so callers can dismiss it early. */
    push(message: string, kind: Toast["kind"] = "info", options: ToastOptions = {}): number {
      // Toasts vanish in seconds; the notifications bell keeps the record so
      // a long error the user never caught in time stays readable in full.
      useNotificationsStore().record({
        kind,
        text: message,
        description: options.description ?? null,
        action: options.notificationAction ?? null,
      });
      const toast: Toast = {
        id: nextId++,
        message,
        kind,
        ...(options.description ? { description: options.description } : {}),
        ...(options.action ? { action: options.action } : {}),
        ...(options.onClick ? { onClick: options.onClick } : {}),
        ...(options.sticky ? { sticky: true } : {}),
      };
      this.items.push(toast);
      // Trim to three, but never drop a sticky or actionable toast — the user
      // may still need its Undo/View button.
      while (this.items.length > 3) {
        const idx = this.items.findIndex((t) => !t.sticky && !t.action);
        if (idx === -1) break;
        this.items.splice(idx, 1);
      }
      if (!options.sticky) {
        setTimeout(() => this.dismiss(toast.id), options.durationMs ?? DEFAULT_DURATION_MS);
      }
      return toast.id;
    },
    /** Run a toast's action (if any) and dismiss it. */
    runAction(id: number) {
      const toast = this.items.find((t) => t.id === id);
      toast?.action?.run();
      this.dismiss(id);
    },
    /** Run a toast's body-click handler (or just dismiss). */
    click(id: number) {
      const toast = this.items.find((t) => t.id === id);
      if (!toast) return;
      toast.onClick?.();
      this.dismiss(id);
    },
    dismiss(id: number) {
      this.items = this.items.filter((t) => t.id !== id);
    },
  },
});
