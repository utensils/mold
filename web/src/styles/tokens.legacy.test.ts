import { describe, expect, it } from "vitest";
import { hexColours, legacyUses, literalStyles, measure } from "./tokenRatchet";

/*
 * A RATCHET over web's legacy styling debt (see tokenRatchet.ts): every file
 * keeps at most the count it has today, a file that reaches zero must leave
 * its table, and a new file starts at zero. Regenerate the tables with
 * `bun run ratchet:web` after a migration pass; never raise a number by hand.
 */

function over(
  actual: Record<string, number>,
  frozen: Record<string, number>,
): string[] {
  return Object.entries(actual)
    .filter(([file, count]) => count > (frozen[file] ?? 0))
    .map(([file, count]) => `${file}: ${count} > ${frozen[file] ?? 0}`);
}
function cleared(
  actual: Record<string, number>,
  frozen: Record<string, number>,
): string[] {
  return Object.keys(frozen).filter((file) => !actual[file]);
}

const LEGACY_FROZEN: Record<string, number> = {
  "src/components/CatalogCard.vue": 50,
  "src/components/CatalogCardGrid.vue": 17,
  "src/components/CatalogSidebar.vue": 10,
  "src/components/CatalogTopbar.vue": 33,
  "src/components/create/ActivityStrip.vue": 42,
  "src/components/create/advanced/ExtendVideoControls.vue": 12,
  "src/components/create/advanced/Ltx2VideoControls.vue": 20,
  "src/components/create/advanced/UpscaleSection.vue": 17,
  "src/components/create/AdvancedDrawer.vue": 58,
  "src/components/create/ColdStartGuide.vue": 25,
  "src/components/create/ComposerCard.vue": 15,
  "src/components/create/ControlsAside.vue": 22,
  "src/components/create/FileUnderGroup.vue": 57,
  "src/components/create/HostRoutingPicker.vue": 20,
  "src/components/create/IdentityPanel.vue": 5,
  "src/components/create/RecentGrid.vue": 10,
  "src/components/create/ResultCanvas.vue": 32,
  "src/components/create/SourceMediaPanel.vue": 25,
  "src/components/ExpandModal.vue": 39,
  "src/components/gallery/GalleryGrid.vue": 36,
  "src/components/gallery/Lightbox.vue": 64,
  "src/components/GalleryCard.vue": 19,
  "src/components/GalleryFeed.vue": 10,
  "src/components/GenerationTemplatesPanel.vue": 23,
  "src/components/ImagePickerModal.vue": 53,
  "src/components/library/CollectionPicker.vue": 14,
  "src/components/library/CollectionsShelf.vue": 41,
  "src/components/library/LibraryChipRow.vue": 34,
  "src/components/library/PrintOrganizer.vue": 28,
  "src/components/library/TagEditor.vue": 22,
  "src/components/LoraPicker.vue": 51,
  "src/components/machines/ConnectModal.vue": 40,
  "src/components/machines/HostCard.vue": 10,
  "src/components/machines/QueueCard.vue": 21,
  "src/components/MaskEditorModal.vue": 33,
  "src/components/models/InstalledModelRow.vue": 31,
  "src/components/models/ModelDetailDrawer.vue": 71,
  "src/components/models/ModelInstallTargetDialog.vue": 29,
  "src/components/PlacementPanel.vue": 2,
  "src/components/RemixModal.vue": 16,
  "src/components/shell/AppNav.vue": 43,
  "src/components/shell/DownloadsBody.vue": 31,
  "src/components/shell/DownloadsPopover.vue": 19,
  "src/components/shell/NowDevelopingPopover.vue": 10,
  "src/pages/CreatePage.vue": 44,
  "src/pages/HostDetailPage.vue": 66,
  "src/pages/LibraryPage.vue": 109,
  "src/pages/MachinesPage.vue": 16,
  "src/pages/ModelsPage.vue": 25,
  "src/pages/NotFoundPage.vue": 1,
  "src/style.css": 41,
};

const LITERAL_FROZEN: Record<string, number> = {
  "src/components/CatalogCard.vue": 2,
  "src/components/create/ActivityStrip.vue": 11,
  "src/components/create/advanced/ExtendVideoControls.vue": 7,
  "src/components/create/advanced/Ltx2VideoControls.vue": 10,
  "src/components/create/advanced/UpscaleSection.vue": 4,
  "src/components/create/AdvancedDrawer.vue": 17,
  "src/components/create/ColdStartGuide.vue": 9,
  "src/components/create/ComposerCard.vue": 6,
  "src/components/create/ControlsAside.vue": 7,
  "src/components/create/FileUnderGroup.vue": 18,
  "src/components/create/HostRoutingPicker.vue": 6,
  "src/components/create/IdentityPanel.vue": 1,
  "src/components/create/RecentGrid.vue": 3,
  "src/components/create/ResultCanvas.vue": 14,
  "src/components/create/SourceMediaPanel.vue": 11,
  "src/components/ExpandModal.vue": 10,
  "src/components/gallery/GalleryGrid.vue": 4,
  "src/components/gallery/Lightbox.vue": 1,
  "src/components/ImagePickerModal.vue": 6,
  "src/components/library/CollectionPicker.vue": 5,
  "src/components/library/CollectionsShelf.vue": 8,
  "src/components/library/LibraryChipRow.vue": 3,
  "src/components/library/PrintOrganizer.vue": 5,
  "src/components/library/TagEditor.vue": 6,
  "src/components/LoraPicker.vue": 9,
  "src/components/machines/ConnectModal.vue": 7,
  "src/components/machines/HostCard.vue": 2,
  "src/components/machines/QueueCard.vue": 8,
  "src/components/MaskEditorModal.vue": 5,
  "src/components/models/InstalledModelRow.vue": 1,
  "src/components/models/ModelInstallTargetDialog.vue": 5,
  "src/components/shell/AppNav.vue": 3,
  "src/components/shell/NowDevelopingPopover.vue": 2,
  "src/pages/HostDetailPage.vue": 6,
  "src/pages/LibraryPage.vue": 16,
  "src/pages/MachinesPage.vue": 3,
  "src/style.css": 1,
};

const HEX_FROZEN: Record<string, number> = {
  "src/components/create/RecentGrid.vue": 1,
  "src/components/gallery/Lightbox.vue": 1,
  "src/pages/LibraryPage.vue": 1,
};

describe("the guards recognise what they refuse (positive controls)", () => {
  it("legacy vocabulary", () => {
    expect(legacyUses("color: var(--rebate); background: var(--bath);")).toBe(
      2,
    );
    expect(legacyUses('<div class="bg-bench text-ink-3 rounded-card">')).toBe(
      3,
    );
    expect(legacyUses("color: var(--mold-text);")).toBe(0);
    expect(legacyUses("font-display: swap;")).toBe(0);
  });
  it("literal radii and font sizes", () => {
    expect(literalStyles("  border-radius: 9px;\n  font-size: 13.5px;")).toBe(
      2,
    );
    expect(
      literalStyles("  border-radius: var(--mold-radius-2);\n  padding: 2px;"),
    ).toBe(0);
    expect(
      literalStyles("  border-radius: 3px; /* literal: QR quiet zone */"),
    ).toBe(0);
  });
  it("hex colours", () => {
    expect(hexColours("  color: #fff;\n  background: rgba(0,0,0,0.5);")).toBe(
      1,
    );
    expect(hexColours("  /* see (#1224) */\n  color: var(--mold-text);")).toBe(
      0,
    );
  });
});

describe("web's legacy debt only ever shrinks", () => {
  const actual = measure();
  it("legacy token usages stay at or under each file's frozen count", () => {
    expect(over(actual.legacy, LEGACY_FROZEN)).toEqual([]);
    expect(cleared(actual.legacy, LEGACY_FROZEN)).toEqual([]);
  });
  it("literal radii/font sizes stay at or under each file's frozen count", () => {
    expect(over(actual.literal, LITERAL_FROZEN)).toEqual([]);
    expect(cleared(actual.literal, LITERAL_FROZEN)).toEqual([]);
  });
  it("hex colours appear only where they already did", () => {
    expect(over(actual.hex, HEX_FROZEN)).toEqual([]);
    expect(cleared(actual.hex, HEX_FROZEN)).toEqual([]);
  });
});
