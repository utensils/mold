export interface VirtualGridInput {
  itemCount: number;
  containerWidth: number;
  minimumItemWidth: number;
  gap: number;
  viewportStart: number;
  viewportSize: number;
  overscanRows?: number;
  minimumColumns?: number;
  maximumColumns?: number;
}

export interface VirtualGridWindow {
  columns: number;
  itemSize: number;
  rowStep: number;
  rowCount: number;
  startRow: number;
  endRow: number;
  startIndex: number;
  endIndex: number;
  offset: number;
  totalSize: number;
}

/** Pure grid window shared by browser and mobile WebView adapters. */
export function virtualGridWindow(input: VirtualGridInput): VirtualGridWindow {
  const width = Math.max(0, input.containerWidth);
  const gap = Math.max(0, input.gap);
  const minimumWidth = Math.max(1, input.minimumItemWidth);
  const minimumColumns = Math.max(1, input.minimumColumns ?? 1);
  const naturalColumns = Math.max(
    1,
    Math.floor((width + gap) / (minimumWidth + gap)),
  );
  const columns = Math.max(
    minimumColumns,
    Math.min(input.maximumColumns ?? Number.MAX_SAFE_INTEGER, naturalColumns),
  );
  const itemSize = Math.max(1, (width - gap * (columns - 1)) / columns);
  const rowStep = itemSize + gap;
  const rowCount = Math.ceil(Math.max(0, input.itemCount) / columns);
  const overscan = Math.max(0, input.overscanRows ?? 2);
  const firstVisibleRow = Math.max(
    0,
    Math.floor(Math.max(0, input.viewportStart) / rowStep),
  );
  const lastVisibleRow = Math.min(
    rowCount,
    Math.ceil(
      (Math.max(0, input.viewportStart) + Math.max(0, input.viewportSize)) /
        rowStep,
    ),
  );
  const startRow = Math.max(0, firstVisibleRow - overscan);
  const endRow = Math.min(rowCount, lastVisibleRow + overscan);
  return {
    columns,
    itemSize,
    rowStep,
    rowCount,
    startRow,
    endRow,
    startIndex: startRow * columns,
    endIndex: Math.min(input.itemCount, endRow * columns),
    offset: startRow * rowStep,
    totalSize: Math.max(0, rowCount * rowStep - gap),
  };
}
