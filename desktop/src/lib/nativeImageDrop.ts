export interface NativeImageDragState {
  candidate: boolean;
  visible: boolean;
}

/**
 * Tauri reports the cursor in PHYSICAL pixels; `dropTargetAtPosition` divides
 * by `devicePixelRatio` to reach CSS pixels before hit-testing.
 */
export interface NativeDragPosition {
  x: number;
  y: number;
}

export type NativeImageDragPayload =
  | { type: "enter"; paths: string[]; position?: NativeDragPosition }
  | { type: "over"; position?: NativeDragPosition }
  | { type: "drop"; paths: string[]; position?: NativeDragPosition }
  | { type: "leave" };

/**
 * The well under an OS drag, as its `data-drop-target` value.
 *
 * This is the whole reason the desktop drop stopped being routed by model
 * capability: Tauri swallows the drag before any HTML5 `drop`, so the ONE
 * window-level bridge has to hit-test the cursor itself. A drop on chrome
 * answers `null`, which the shared router reads as "use the plan default".
 */
export function dropTargetAtPosition(
  position: NativeDragPosition | null | undefined,
): string | null {
  if (!position || typeof document === "undefined") return null;
  const ratio = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const element = document.elementFromPoint(position.x / ratio, position.y / ratio);
  const well = element?.closest("[data-drop-target]");
  return well?.getAttribute("data-drop-target") ?? null;
}

export function isSupportedDroppedImage(path: string): boolean {
  return /\.(png|jpe?g)$/i.test(path);
}

/**
 * Tauri can report ordinary WebView drags as `over`. Keep the native source
 * drop overlay inert until an `enter` proves the drag carries an image path.
 */
export function reduceNativeImageDrag(
  state: NativeImageDragState,
  payload: NativeImageDragPayload,
): NativeImageDragState {
  if (payload.type === "enter") {
    const candidate = payload.paths.some(isSupportedDroppedImage);
    return { candidate, visible: candidate };
  }
  if (payload.type === "over") {
    return { candidate: state.candidate, visible: state.candidate };
  }
  return { candidate: false, visible: false };
}
