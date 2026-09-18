import AppKit
import MoldClient
import Testing

@testable import Mold

/// What each Generate surface offers on a right-click: the ORDER, the GATING
/// and the shared source, not a list of titles. Every one of these is the same
/// list the inline control beside it renders.
@MainActor
struct GenerateMenusTests {
    // MARK: - The house rules

    @Test func destructiveItemsAlwaysComeLastBehindADivider() {
        let items = GenerateMenus.sourceWell(hasPicture: true, canEditMask: true, canPaste: true)
        #expect(items.dropLast().allSatisfy { !$0.isDestructive })
        #expect(items.filter(\.isDestructive).map(\.kind) == [.removeSource])
        #expect(items.last?.kind == .removeSource)
        // `RowAction.rendered` is the ONE place the split -- and the divider
        // in front of it -- happens, for every menu in the app.
        let drawn = RowAction.rendered([GenerateAction.removeSource.row,
                                        GenerateAction.chooseFile.row])
        #expect(drawn.map(\.kind) == [.chooseFile, nil, .removeSource])
        #expect(drawn.dropLast().last?.isSeparator == true)
    }

    @Test func aRowWithNothingApplicableHasNoMenuAtAll() {
        // An unpainted mask row's inline button already IS "Edit mask…".
        #expect(GenerateMenus.maskRow(hasMask: false).isEmpty)
        // An empty strip: nothing to remove, and Add is the add well's own.
        #expect(GenerateMenus.referenceStrip(count: 0).isEmpty)
    }

    // MARK: - A finished result

    @Test func aResultOffersReuseOnlyWhereTheWellExists() {
        let bare = GenerateMenus.result(canUseAsSource: false, canAddReference: false)
        #expect(bare.map(\.kind) == [.saveACopy, .copyResult, .showInLibrary])

        let both = GenerateMenus.result(canUseAsSource: true, canAddReference: true)
        #expect(both.map(\.kind) == [.saveACopy, .copyResult, .showInLibrary,
                             .useAsSourceImage, .addAsReference])

        let sourceOnly = GenerateMenus.result(canUseAsSource: true, canAddReference: false)
        #expect(!sourceOnly.map(\.kind).contains(.addAsReference))
    }

    // MARK: - The source well

    @Test func theSourceWellOffersRemoveAndTheMaskOnlyWhenTheyApply() {
        let empty = GenerateMenus.sourceWell(
            hasPicture: false, canEditMask: true, canPaste: false)
        #expect(empty.map(\.kind) == [.chooseFile, .chooseFromLibrary])

        let held = GenerateMenus.sourceWell(hasPicture: true, canEditMask: false, canPaste: true)
        #expect(held.map(\.kind) == [.chooseFile, .chooseFromLibrary, .paste, .removeSource])

        let maskable = GenerateMenus.sourceWell(
            hasPicture: true, canEditMask: true, canPaste: false)
        #expect(maskable.map(\.kind) == [.chooseFile, .chooseFromLibrary, .editMask, .removeSource])
    }

    // MARK: - The reference strip

    /// Index 0 is Qwen's edit TARGET, so moving one is a real instruction --
    /// and the ends carry no move.
    @Test func aReferenceOffersOnlyTheMovesThatExist() {
        #expect(GenerateMenus.referenceItem(index: 0, count: 3).map(\.kind)
            == [.moveRight, .replacePicture, .replaceFromLibrary, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 1, count: 3).map(\.kind)
            == [.moveLeft, .moveRight, .replacePicture, .replaceFromLibrary, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 2, count: 3).map(\.kind)
            == [.moveLeft, .replacePicture, .replaceFromLibrary, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 0, count: 1).map(\.kind)
            == [.replacePicture, .replaceFromLibrary, .removeReference])
    }

    /// The background is about the whole strip: its Add and its Paste are the
    /// add well's own, one square away (`PictureWellTests`).
    @Test func theStripBackgroundFollowsItsContents() {
        #expect(GenerateMenus.referenceStrip(count: 0).isEmpty)
        #expect(GenerateMenus.referenceStrip(count: 2).map(\.kind) == [.removeAllReferences])
        #expect(GenerateMenus.referenceStrip(count: 4).map(\.kind) == [.removeAllReferences])
    }

    /// An ABSENT `max_count` is UNBOUNDED, the way studio reads it -- treating
    /// it as one left the legacy Qwen strip holding a Target and no references
    /// at all (review 11, low).
    @Test func anAbsentReferenceCapIsUnbounded() throws {
        let qwen = try #require(
            ReferenceImagesCapability.legacy(family: "qwen-image-edit", model: "q"))
        #expect(qwen.maxCount == nil)
        #expect(qwen.hasRoom(for: 0))
        #expect(qwen.hasRoom(for: 7))

        let flux2 = try #require(
            ReferenceImagesCapability.legacy(family: "flux2", model: "flux2-dev:q4"))
        #expect(flux2.hasRoom(for: 3))
        #expect(!flux2.hasRoom(for: 4))
    }

    // MARK: - The rest

    @Test func theRemainingRowsOfferWhatTheyCan() {
        #expect(GenerateMenus.identityPhoto().map(\.kind)
            == [.replacePhoto, .replaceFromLibrary, .removePhoto])
        #expect(GenerateMenus.maskRow(hasMask: true).map(\.kind) == [.editMask, .clearMask])
        #expect(GenerateMenus.adapterRow(isAtDefaultStrength: true).map(\.kind) == [.removeAdapter])
        #expect(GenerateMenus.adapterRow(isAtDefaultStrength: false).map(\.kind)
            == [.resetStrength, .removeAdapter])
        // No per-entry delete verb exists on the wire, so none is offered.
        #expect(GenerateMenus.recentPrompt().map(\.kind) == [.usePrompt, .copyPrompt])
    }

    @Test func everyActionHasATitleAndNoTwoDestructivesHide() {
        for action in GenerateAction.allCases {
            #expect(!action.title.isEmpty)
        }
    }

    // MARK: - Bare arrow keys (finding 02#15)

    /// **Fails today**: the strip's ←/→ are window-scoped key equivalents,
    /// which AppKit checks BEFORE the first responder -- so a focused Slider
    /// or Stepper (Steps, Guidance, Length, Strength, identity Weight, every
    /// LoRA scale) could not adjust itself with the same two keys.
    @Test func aFocusedControlKeepsItsOwnArrowKeys() {
        #expect(ArrowKeyClaim.claims(NSSlider()))
        #expect(ArrowKeyClaim.claims(NSStepper()))
        #expect(ArrowKeyClaim.claims(NSTextView()))
        #expect(ArrowKeyClaim.claims(NSTextField()))
        #expect(!ArrowKeyClaim.claims(NSButton()))
        #expect(!ArrowKeyClaim.claims(nil))
    }

    @Test func theResultStripStandsDownWheneverTheArrowsAreClaimed() {
        #expect(ResultStrip.key(for: 0, selected: 1, arrowsAreClaimed: false) == .leftArrow)
        #expect(ResultStrip.key(for: 2, selected: 1, arrowsAreClaimed: false) == .rightArrow)
        // Neither neighbour, so nothing to bind -- no wrap-around.
        #expect(ResultStrip.key(for: 3, selected: 1, arrowsAreClaimed: false) == nil)
        #expect(ResultStrip.key(for: 1, selected: 1, arrowsAreClaimed: false) == nil)
        for index in 0 ... 3 {
            #expect(ResultStrip.key(for: index, selected: 1, arrowsAreClaimed: true) == nil)
        }
    }
}
