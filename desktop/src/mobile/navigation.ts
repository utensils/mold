import type { IconName } from "@ui/icons";

export type MobileTab = "generate" | "queue" | "gallery" | "catalog" | "hosts";

/** Short phone labels; destination headings retain the desktop vocabulary. */
export const MOBILE_TABS: readonly { id: MobileTab; label: string; icon: IconName }[] = [
  { id: "generate", label: "Make", icon: "create" },
  { id: "queue", label: "Queue", icon: "list" },
  { id: "gallery", label: "Images", icon: "library" },
  { id: "catalog", label: "Styles", icon: "models" },
  { id: "hosts", label: "Machines", icon: "machines" },
];
