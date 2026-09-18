import MoldClient

/// What each Generate surface offers, resolved purely so the contextual menu
/// and the inline control it mirrors can never drift apart, and so a test can
/// pin the ORDER and the GATING rather than a list of titles.
///
/// Every gate here is a fact the caller already has to answer to draw its
/// inline control -- there is no second copy of "does this recipe take a
/// mask", only the answer being passed in.
enum GenerateMenus {
    /// One row of a Generate menu -- the app's one menu model.
    typealias Row = RowAction<GenerateAction>

    /// A finished result: the big one on the canvas and every strip tile.
    ///
    /// Save / Copy / Show in Library are `ResultBar`'s own three; the two
    /// reuse items are a composition of the source well's and the strip's own
    /// "attach these bytes", and each is offered only where that well exists
    /// and has room.
    static func result(canUseAsSource: Bool, canAddReference: Bool) -> [Row] {
        var actions: [GenerateAction] = [.saveACopy, .copyResult, .showInLibrary]
        if canUseAsSource { actions.append(.useAsSourceImage) }
        if canAddReference { actions.append(.addAsReference) }
        return Row.ordered(actions.map(\.row))
    }

    /// The three doors every well that TAKES a picture offers, in the same
    /// order, by the same names -- a file, a print already in the fleet, or
    /// whatever is on the pasteboard. The owner's ask: one selector
    /// everywhere, not four.
    ///
    /// Paste is absent, not disabled, when the pasteboard holds no picture.
    private static func doors(canPaste: Bool) -> [GenerateAction] {
        canPaste ? [.chooseFile, .chooseFromLibrary, .paste] : [.chooseFile, .chooseFromLibrary]
    }

    /// The source well. `canEditMask` is `RefineGroup.maskCapable`'s answer --
    /// the recipe's own mask path, not a guess.
    static func sourceWell(
        hasPicture: Bool, canEditMask: Bool, canPaste: Bool
    ) -> [Row] {
        var actions = doors(canPaste: canPaste)
        if hasPicture, canEditMask { actions.append(.editMask) }
        if hasPicture { actions.append(.removeSource) }
        return Row.ordered(actions.map(\.row))
    }

    /// The ControlNet still's well, which used to open a panel on a click and
    /// offer no menu, no Library and no Paste at all.
    static func controlWell(hasPicture: Bool, canPaste: Bool) -> [Row] {
        var actions = doors(canPaste: canPaste)
        if hasPicture { actions.append(.removeControl) }
        return Row.ordered(actions.map(\.row))
    }

    /// The reference strip's add well. Its `+` is a well like any other.
    static func referenceAdd(canPaste: Bool) -> [Row] {
        Row.ordered(doors(canPaste: canPaste).map(\.row))
    }

    /// The identity group's add well, which used to be an `NSOpenPanel` and
    /// nothing else -- so a photograph already in the fleet could not be used
    /// as a face without saving it to this Mac first.
    static func identityAdd(canPaste: Bool) -> [Row] {
        Row.ordered(doors(canPaste: canPaste).map(\.row))
    }

    /// One reference picture. Order MATTERS here: on a `primaryIsTarget`
    /// recipe index 0 is the picture being edited, so moving one is a real
    /// instruction and not a convenience.
    static func referenceItem(index: Int, count: Int) -> [Row] {
        var actions: [GenerateAction] = []
        if index > 0 { actions.append(.moveLeft) }
        if index < count - 1 { actions.append(.moveRight) }
        actions.append(contentsOf: [.replacePicture, .replaceFromLibrary, .removeReference])
        return Row.ordered(actions.map(\.row))
    }

    /// The strip's BACKGROUND, which is about the whole strip. Add and Paste
    /// are the add well's own -- and the add well is drawn whenever there is
    /// room -- so repeating them here would be two menus one hover apart
    /// offering the same door. An empty strip has nothing to say at all.
    static func referenceStrip(count: Int) -> [Row] {
        guard count > 0 else { return [] }
        return Row.ordered([GenerateAction.removeAllReferences.row])
    }

    /// One staged identity photograph. Replaced from either door, like every
    /// other staged picture.
    static func identityPhoto() -> [Row] {
        Row.ordered([GenerateAction.replacePhoto, .replaceFromLibrary, .removePhoto].map(\.row))
    }

    /// The mask row. Only where a mask has actually been painted -- an
    /// unpainted row's inline button IS "Edit mask…", so a menu repeating it
    /// alone would be furniture.
    static func maskRow(hasMask: Bool) -> [Row] {
        guard hasMask else { return [] }
        return Row.ordered([GenerateAction.editMask, .clearMask].map(\.row))
    }

    /// An adapter row, extending the trained-word menu that is already there.
    /// `trainedWords` are rendered ahead of these by the view, because they
    /// are the row's own vocabulary rather than an action on it.
    static func adapterRow(isAtDefaultStrength: Bool) -> [Row] {
        var actions: [GenerateAction] = []
        if !isAtDefaultStrength { actions.append(.resetStrength) }
        actions.append(.removeAdapter)
        return Row.ordered(actions.map(\.row))
    }

    /// The reference-weight slider. Its ONE verb is putting it back to the
    /// recipe's own default, which a slider already sitting there has nothing
    /// to offer -- so that row answers empty and no menu is attached at all,
    /// the same rule the unpainted mask row follows.
    static func referenceWeight(isAtDefault: Bool) -> [Row] {
        guard !isAtDefault else { return [] }
        return Row.ordered([GenerateAction.resetReferenceWeight.row])
    }

    /// The Sampler group. Its one verb puts every control back to the
    /// recipe's own value; a group nobody has touched has nothing to put
    /// back, so it answers empty and no menu is attached.
    static func sampler(touched: Bool) -> [Row] {
        guard touched else { return [] }
        return Row.ordered([GenerateAction.resetSampler.row])
    }

    /// The Fit row. Crop to fill is the intentional default for every newly
    /// attached picture on every surface (`sourceFit.ts:26-29`), so a row
    /// already there has nothing to put back.
    static func sourceFit(isAtDefault: Bool) -> [Row] {
        guard !isAtDefault else { return [] }
        return Row.ordered([GenerateAction.resetSourceFit.row])
    }

    /// A recent prompt. There is no per-entry delete verb on the wire -- the
    /// history route offers `clear` alone -- so "Remove from History" is
    /// deliberately absent rather than shown and broken.
    static func recentPrompt() -> [Row] {
        Row.ordered([GenerateAction.usePrompt, .copyPrompt].map(\.row))
    }
}
