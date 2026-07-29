export interface NativeImageDragState {
  candidate: boolean;
  visible: boolean;
}

export type NativeImageDragPayload =
  | { type: "enter"; paths: string[] }
  | { type: "over" }
  | { type: "drop"; paths: string[] }
  | { type: "leave" };

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
