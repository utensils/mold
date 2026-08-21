/** Closed set of internal destinations encoded into native notifications. */
export type NotificationAction =
  | { kind: "gallery"; filename?: string }
  | { kind: "create" }
  | { kind: "models" }
  | { kind: "updates" };

export interface NotificationRoute {
  path: string;
  query?: Record<string, string>;
}

/** Translate native actions into allowlisted router locations. */
export function notificationRoute(action: NotificationAction): NotificationRoute {
  switch (action.kind) {
    case "gallery":
      return action.filename
        ? { path: "/library", query: { print: action.filename } }
        : { path: "/library" };
    case "create":
      return { path: "/create" };
    case "models":
      return { path: "/models" };
    case "updates":
      return { path: "/settings", query: { section: "updates" } };
  }
}
