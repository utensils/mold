/**
 * Offline vocabulary for the `camera-control:<id>` wire form.
 *
 * This is NOT an options source — availability always comes from
 * `GET /api/capabilities/ltx2-camera-controls`, which knows what the selected
 * checkpoint can actually run. These entries exist because a clip's camera
 * choice has to be recovered from a chain script, a saved print, or a template
 * with no server in the loop (`sequenceForm.chainScriptToClips`,
 * `sequenceReuse.chainMetadataToClips`), and because the matching there falls
 * back to the human label.
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
