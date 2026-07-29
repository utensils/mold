export const CAMERA_MOTION_PRESETS = [
  { id: "dolly-in", label: "Dolly in" },
  { id: "dolly-left", label: "Dolly left" },
  { id: "dolly-out", label: "Dolly out" },
  { id: "dolly-right", label: "Dolly right" },
  { id: "jib-down", label: "Jib down" },
  { id: "jib-up", label: "Jib up" },
  { id: "static", label: "Static" },
] as const;

export function cameraMotionMode(value: string | null): string {
  if (!value) return "";
  return CAMERA_MOTION_PRESETS.some((preset) => preset.id === value)
    ? value
    : "custom";
}
