export interface FittedSize {
  width: number;
  height: number;
}

/** Fit an aspect ratio inside a measured region without overflowing either axis. */
export function fitAspectRatio(
  containerWidth: number,
  containerHeight: number,
  contentWidth: number,
  contentHeight: number,
): FittedSize {
  if (containerWidth <= 0 || containerHeight <= 0 || contentWidth <= 0 || contentHeight <= 0) {
    return { width: 0, height: 0 };
  }

  const aspect = contentWidth / contentHeight;
  const containerAspect = containerWidth / containerHeight;
  if (containerAspect > aspect) {
    return {
      width: Math.floor(containerHeight * aspect),
      height: Math.floor(containerHeight),
    };
  }
  return {
    width: Math.floor(containerWidth),
    height: Math.floor(containerWidth / aspect),
  };
}
