/**
 * Offline vocabulary for the `camera-control:<id>` wire form.
 *
 * This is NOT an options source — availability always comes from
 * `GET /api/capabilities/ltx2-camera-controls`, which knows what the selected
 * checkpoint can actually run. These entries exist because a clip's camera
 * choice has to be recovered from a saved print or a template with no server
 * in the loop, and because the matching there falls back to the human label.
 *
 * `ltx2_camera.rs` is the authority; `camera_motion_ts_mirror_matches_the_rust_registry`
 * pins this list to it.
 */
export const CAMERA_MOTION_PRESETS = [
  { id: "dolly-in", label: "Dolly in" },
  { id: "dolly-left", label: "Dolly left" },
  { id: "dolly-out", label: "Dolly out" },
  { id: "dolly-right", label: "Dolly right" },
  { id: "jib-down", label: "Jib down" },
  { id: "jib-up", label: "Jib up" },
  { id: "static", label: "Static" },
] as const;

export function isCameraMotionPreset(value: string): boolean {
  return CAMERA_MOTION_PRESETS.some((preset) => preset.id === value);
}

export function cameraMotionMode(value: string | null): string {
  if (value === null) return "";
  return isCameraMotionPreset(value) ? value : "custom";
}

export function cameraMotionLabel(value: string): string {
  return (
    CAMERA_MOTION_PRESETS.find((preset) => preset.id === value)?.label ??
    "Camera motion"
  );
}

/** The ordinary LoRA-stack path used by the server for a camera choice. */
export function cameraMotionLoraPath(value: string | null): string | null {
  const trimmed = value?.trim();
  if (!trimmed) return null;
  return trimmed.endsWith(".safetensors") || trimmed.includes("/")
    ? trimmed
    : `camera-control:${trimmed}`;
}

/** Recover a built-in camera choice from an ordinary LoRA-stack row. */
export function cameraMotionFromLoraPath(path: string): string | null {
  const preset = path.startsWith("camera-control:")
    ? path.slice("camera-control:".length)
    : null;
  return preset && isCameraMotionPreset(preset) ? preset : null;
}

/** User-facing fallback for auto-download rows that are not installed yet. */
export function cameraMotionLoraLabel(path: string): string {
  const preset = cameraMotionFromLoraPath(path);
  return preset ? `${cameraMotionLabel(preset)} camera control` : path;
}

export interface CameraMotionLoraRow {
  path: string;
  scale: number;
}

/** Whether the current stack can select/replace one camera adapter. */
export function cameraMotionLoraSlotAvailable(
  loras: readonly CameraMotionLoraRow[],
  currentControl: string | null,
  maxRows: number,
): boolean {
  const currentPath = cameraMotionLoraPath(currentControl);
  return (
    loras.length < maxRows ||
    loras.some(
      (row) =>
        row.path === currentPath || row.path.startsWith("camera-control:"),
    )
  );
}

/**
 * Keep the camera picker and the visible LoRA stack as one piece of state.
 *
 * A newly selected preset is inserted immediately, retaining the previous
 * camera row's editable strength when the user switches motions. Existing
 * ordinary LoRAs keep their order. The caller supplies display-only fields so
 * web, desktop, and iPhone can share the state transition without sharing
 * their surface-specific row types.
 */
export function syncCameraMotionLora<T extends CameraMotionLoraRow>(
  loras: readonly T[],
  previousControl: string | null,
  nextControl: string | null,
  create: (path: string, scale: number) => T,
  maxRows = Number.POSITIVE_INFINITY,
): T[] {
  const previousPath = cameraMotionLoraPath(previousControl);
  const nextPath = cameraMotionLoraPath(nextControl);
  const presetIndex = loras.findIndex((row) =>
    row.path.startsWith("camera-control:"),
  );
  const previousIndex = previousPath
    ? loras.findIndex((row) => row.path === previousPath)
    : -1;
  const replaceIndex = previousIndex >= 0 ? previousIndex : presetIndex;
  const existingNext = nextPath
    ? loras.find((row) => row.path === nextPath)
    : undefined;
  const previous = replaceIndex >= 0 ? loras[replaceIndex] : undefined;
  const scale = existingNext?.scale ?? previous?.scale ?? 1;
  // Custom-path inputs pass through an intentional empty editing state. Keep
  // the previous row alive so its user-edited strength can transfer to the
  // completed path; `null` remains the explicit remove action.
  if (nextControl !== null && !nextPath) return [...loras];
  if (
    nextPath &&
    replaceIndex < 0 &&
    !existingNext &&
    loras.length >= maxRows
  ) {
    return [...loras];
  }
  const withoutCamera = loras.filter(
    (row, index) =>
      index !== replaceIndex &&
      row.path !== nextPath &&
      !row.path.startsWith("camera-control:"),
  );

  if (!nextPath) return withoutCamera;
  const row = create(nextPath, scale);
  const insertion =
    replaceIndex < 0
      ? withoutCamera.length
      : Math.min(replaceIndex, withoutCamera.length);
  return [
    ...withoutCamera.slice(0, insertion),
    row,
    ...withoutCamera.slice(insertion),
  ];
}

/** Normalize restored/persisted camera state without evicting user LoRAs. */
export function normalizeCameraMotionLoraState<T extends CameraMotionLoraRow>(
  loras: readonly T[],
  cameraControl: string | null,
  create: (path: string, scale: number) => T,
  maxRows: number,
): { loras: T[]; cameraControl: string | null } {
  const inferred =
    loras
      .map((row) => cameraMotionFromLoraPath(row.path))
      .find((value): value is string => value !== null) ?? null;
  const control = cameraControl?.trim() || inferred;
  if (!control) return { loras: [...loras], cameraControl: null };
  if (!cameraMotionLoraSlotAvailable(loras, control, maxRows)) {
    // A legacy snapshot can contain four ordinary LoRAs plus a camera picker
    // value that was never represented in the stack. Preserve the four
    // visible rows and clear the unrepresentable selection rather than
    // silently evicting an adapter or dropping camera motion at submit time.
    return { loras: [...loras], cameraControl: null };
  }
  return {
    loras: syncCameraMotionLora(loras, control, control, create, maxRows),
    cameraControl: control,
  };
}

/** One camera-control preset as advertised by the selected host. */
export interface CameraControlInfo {
  id: string;
  label: string;
  size_bytes: number;
  installed: boolean;
  download_model: string;
  download_repo: string;
  download_filename: string;
  download_sha256: string;
}

/** What the host says about camera motion for the selected checkpoint. */
export interface CameraControlAvailability {
  controls: CameraControlInfo[];
  supported: boolean;
  /** The host's own reason. `null` on hosts that predate `?detail=1`. */
  unsupportedReason: string | null;
}

/**
 * Read either shape of `/api/capabilities/ltx2-camera-controls`.
 *
 * Older hosts answer with a bare array and cannot distinguish "no presets for
 * this checkpoint" from "the request failed", so every surface hard-coded its
 * own guess at the server's policy. Hosts that understand `?detail=1` send the
 * reason instead. Accept both, and never let the newer shape become a hard
 * requirement — desktop and iPhone talk to arbitrary-version remotes.
 */
export function parseCameraControlAvailability(
  body: unknown,
): CameraControlAvailability {
  if (Array.isArray(body)) {
    return {
      controls: body as CameraControlInfo[],
      supported: body.length > 0,
      unsupportedReason: null,
    };
  }
  const envelope = body as Partial<{
    controls: CameraControlInfo[];
    supported: boolean;
    unsupported_reason: string | null;
  }> | null;
  const controls = Array.isArray(envelope?.controls) ? envelope.controls : [];
  return {
    controls,
    supported: envelope?.supported ?? controls.length > 0,
    unsupportedReason: envelope?.unsupported_reason ?? null,
  };
}
