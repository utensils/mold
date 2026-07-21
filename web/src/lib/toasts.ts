import { reactive, readonly } from "vue";

/*
 * Web-surface notification store feeding @ui ToastShelf, plus the shared
 * destructive-action helpers (spec §08 G11/G12): toast() for transient
 * feedback, undoableAction() for reversible deletes (optimistic with an
 * undo window), and confirm helpers rendered by App-level modals.
 */

export type ToastKind = "info" | "success" | "error";

export interface Toast {
  id: string;
  kind: ToastKind;
  text: string;
  actionLabel?: string;
  onAction?: () => void;
  /** Fired once when the toast leaves the shelf for any reason other than its
   * action — timeout or manual ✕. undoableAction uses it to commit the pending
   * change (dismiss = "done looking at the undo offer"). */
  onDismiss?: () => void;
}

export interface DialogChoice {
  id: string;
  label: string;
  danger?: boolean;
}

interface DialogRequest {
  id: string;
  kind: "confirm" | "text" | "choice";
  title: string;
  body: string;
  confirmLabel: string;
  danger: boolean;
  /** When set, the user must type this phrase to enable the confirm button. */
  typedPhrase?: string;
  /** kind === "text": input label + initial value. */
  inputLabel?: string;
  inputInitial?: string;
  /** kind === "choice": one button per choice. */
  choices?: DialogChoice[];
  resolve: (result: boolean | string | null) => void;
}

const state = reactive({
  toasts: [] as Toast[],
  confirm: null as DialogRequest | null,
});

let counter = 0;
const timers = new Map<string, ReturnType<typeof setTimeout>>();

function nextId(): string {
  counter += 1;
  return `t${counter}`;
}

export function dismissToast(id: string): void {
  const timer = timers.get(id);
  if (timer) {
    clearTimeout(timer);
    timers.delete(id);
  }
  const index = state.toasts.findIndex((t) => t.id === id);
  if (index === -1) return;
  const [entry] = state.toasts.splice(index, 1);
  // Both the timeout and the manual ✕ route through here; fire onDismiss so a
  // reversible action commits on either. runToastAction runs the action first,
  // so its onDismiss handler is expected to be a no-op after an undo.
  entry?.onDismiss?.();
}

export interface ToastOptions {
  actionLabel?: string;
  onAction?: () => void;
  onDismiss?: () => void;
  /** ms; errors are sticky by default (0 = sticky). */
  timeout?: number;
}

export function toast(
  kind: ToastKind,
  text: string,
  options: ToastOptions = {},
): string {
  const id = nextId();
  const entry: Toast = { id, kind, text };
  if (options.actionLabel) entry.actionLabel = options.actionLabel;
  if (options.onAction) entry.onAction = options.onAction;
  if (options.onDismiss) entry.onDismiss = options.onDismiss;
  state.toasts.push(entry);
  const timeout = options.timeout ?? (kind === "error" ? 0 : 5000);
  if (timeout > 0) {
    timers.set(
      id,
      setTimeout(() => dismissToast(id), timeout),
    );
  }
  return id;
}

export function runToastAction(id: string): void {
  const entry = state.toasts.find((t) => t.id === id);
  entry?.onAction?.();
  dismissToast(id);
}

/**
 * Reversible destructive action (G12): apply an optimistic UI change now,
 * show an undo toast, and only commit for real once the window elapses.
 */
export function undoableAction(options: {
  text: string;
  undo: () => void;
  commit: () => void | Promise<void>;
  windowMs?: number;
}): void {
  // Exactly one of undo/commit runs, exactly once. Clicking Undo undoes;
  // otherwise the change commits — whether the window elapsed (auto-dismiss)
  // or the user closed the toast early (manual ✕). Dismissing is "I'm done
  // with the undo offer", not "forget this ever happened", so a dismissed
  // delete must still delete rather than stranding a hidden-but-undeleted row.
  let settled = false;
  const finish = (undone: boolean) => {
    if (settled) return;
    settled = true;
    if (undone) options.undo();
    else void options.commit();
  };
  toast("info", options.text, {
    actionLabel: "Undo",
    onAction: () => finish(true),
    onDismiss: () => finish(false),
    timeout: options.windowMs ?? 6000,
  });
}

export interface ConfirmOptions {
  title: string;
  body: string;
  confirmLabel?: string;
  danger?: boolean;
  typedPhrase?: string;
}

function openDialog(
  request: Omit<DialogRequest, "id" | "resolve">,
): Promise<boolean | string | null> {
  return new Promise((resolve) => {
    // Only one dialog at a time; a second request cancels the first.
    state.confirm?.resolve(state.confirm.kind === "confirm" ? false : null);
    state.confirm = { ...request, id: nextId(), resolve };
  });
}

/** Promise-based confirm rendered by the App-level modal (replaces window.confirm). */
export function requestConfirm(options: ConfirmOptions): Promise<boolean> {
  const request: Omit<DialogRequest, "id" | "resolve"> = {
    kind: "confirm",
    title: options.title,
    body: options.body,
    confirmLabel: options.confirmLabel ?? "Confirm",
    danger: options.danger ?? false,
  };
  if (options.typedPhrase) request.typedPhrase = options.typedPhrase;
  return openDialog(request) as Promise<boolean>;
}

/** Promise-based one-line text input (replaces window.prompt). Null = cancelled. */
export function requestText(options: {
  title: string;
  body?: string;
  label?: string;
  initial?: string;
  confirmLabel?: string;
}): Promise<string | null> {
  const request: Omit<DialogRequest, "id" | "resolve"> = {
    kind: "text",
    title: options.title,
    body: options.body ?? "",
    confirmLabel: options.confirmLabel ?? "Save",
    danger: false,
  };
  if (options.label) request.inputLabel = options.label;
  if (options.initial) request.inputInitial = options.initial;
  return openDialog(request) as Promise<string | null>;
}

/** Promise-based multi-choice dialog. Resolves the chosen id, null on cancel. */
export function requestChoice(options: {
  title: string;
  body: string;
  choices: DialogChoice[];
}): Promise<string | null> {
  return openDialog({
    kind: "choice",
    title: options.title,
    body: options.body,
    confirmLabel: "",
    danger: false,
    choices: options.choices,
  }) as Promise<string | null>;
}

export function settleConfirm(result: boolean | string | null): void {
  state.confirm?.resolve(result);
  state.confirm = null;
}

export function useNotifications() {
  return readonly(state);
}

/** Test hook — clears all pending state. */
export function resetNotifications(): void {
  for (const timer of timers.values()) clearTimeout(timer);
  timers.clear();
  state.toasts.splice(0);
  state.confirm?.resolve(state.confirm.kind === "confirm" ? false : null);
  state.confirm = null;
  counter = 0;
}
