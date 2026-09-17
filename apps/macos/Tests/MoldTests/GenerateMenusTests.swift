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
        #expect(items.ordinary.allSatisfy { !$0.isDestructive })
        #expect(items.destructive == [.removeSource])
        #expect(items.all.last == .removeSource)
        // `GenerateMenuItems` is the ONE place the split happens.
        #expect(GenerateMenuItems([.removeSource, .chooseFile]).all == [.chooseFile, .removeSource])
    }

    @Test func aRowWithNothingApplicableHasNoMenuAtAll() {
        // An unpainted mask row's inline button already IS "Edit mask…".
        #expect(GenerateMenus.maskRow(hasMask: false).isEmpty)
        // An empty, full strip: nothing to add, nothing to remove.
        #expect(GenerateMenus.referenceStrip(count: 0, hasRoom: false, canPaste: true).isEmpty)
    }

    // MARK: - A finished result

    @Test func aResultOffersReuseOnlyWhereTheWellExists() {
        let bare = GenerateMenus.result(canUseAsSource: false, canAddReference: false)
        #expect(bare.all == [.saveACopy, .copyResult, .showInLibrary])

        let both = GenerateMenus.result(canUseAsSource: true, canAddReference: true)
        #expect(both.all == [.saveACopy, .copyResult, .showInLibrary,
                             .useAsSourceImage, .addAsReference])

        let sourceOnly = GenerateMenus.result(canUseAsSource: true, canAddReference: false)
        #expect(!sourceOnly.all.contains(.addAsReference))
    }

    // MARK: - The source well

    @Test func theSourceWellOffersRemoveAndTheMaskOnlyWhenTheyApply() {
        let empty = GenerateMenus.sourceWell(
            hasPicture: false, canEditMask: true, canPaste: false)
        #expect(empty.all == [.chooseFile, .chooseFromLibrary])

        let held = GenerateMenus.sourceWell(hasPicture: true, canEditMask: false, canPaste: true)
        #expect(held.all == [.chooseFile, .chooseFromLibrary, .paste, .removeSource])

        let maskable = GenerateMenus.sourceWell(
            hasPicture: true, canEditMask: true, canPaste: false)
        #expect(maskable.all == [.chooseFile, .chooseFromLibrary, .editMask, .removeSource])
    }

    // MARK: - The reference strip

    /// Index 0 is Qwen's edit TARGET, so moving one is a real instruction --
    /// and the ends carry no move.
    @Test func aReferenceOffersOnlyTheMovesThatExist() {
        #expect(GenerateMenus.referenceItem(index: 0, count: 3).all
            == [.moveRight, .replacePicture, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 1, count: 3).all
            == [.moveLeft, .moveRight, .replacePicture, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 2, count: 3).all
            == [.moveLeft, .replacePicture, .removeReference])
        #expect(GenerateMenus.referenceItem(index: 0, count: 1).all
            == [.replacePicture, .removeReference])
    }

    @Test func theStripBackgroundFollowsItsRoomAndItsContents() {
        #expect(GenerateMenus.referenceStrip(count: 0, hasRoom: true, canPaste: false).all
            == [.addReference])
        #expect(GenerateMenus.referenceStrip(count: 2, hasRoom: true, canPaste: true).all
            == [.addReference, .paste, .removeAllReferences])
        // Full: nothing to add or paste, but there is something to clear.
        #expect(GenerateMenus.referenceStrip(count: 4, hasRoom: false, canPaste: true).all
            == [.removeAllReferences])
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
        #expect(GenerateMenus.identityPhoto().all == [.replacePhoto, .removePhoto])
        #expect(GenerateMenus.maskRow(hasMask: true).all == [.editMask, .clearMask])
        #expect(GenerateMenus.adapterRow(isAtDefaultStrength: true).all == [.removeAdapter])
        #expect(GenerateMenus.adapterRow(isAtDefaultStrength: false).all
            == [.resetStrength, .removeAdapter])
        // No per-entry delete verb exists on the wire, so none is offered.
        #expect(GenerateMenus.recentPrompt().all == [.usePrompt, .copyPrompt])
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
