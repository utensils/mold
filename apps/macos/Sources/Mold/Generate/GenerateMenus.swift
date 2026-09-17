import Foundation

/// What each Generate surface offers, resolved purely so the contextual menu
/// and the inline control it mirrors can never drift apart, and so a test can
/// pin the ORDER and the GATING rather than a list of titles.
///
/// Every gate here is a fact the caller already has to answer to draw its
/// inline control -- there is no second copy of "does this recipe take a
/// mask", only the answer being passed in.
enum GenerateMenus {
    /// A finished result: the big one on the canvas and every strip tile.
    ///
    /// Save / Copy / Show in Library are `ResultBar`'s own three; the two
    /// reuse items are a composition of the source well's and the strip's own
    /// "attach these bytes", and each is offered only where that well exists
    /// and has room.
    static func result(canUseAsSource: Bool, canAddReference: Bool) -> GenerateMenuItems {
        var actions: [GenerateAction] = [.saveACopy, .copyResult, .showInLibrary]
        if canUseAsSource { actions.append(.useAsSourceImage) }
        if canAddReference { actions.append(.addAsReference) }
        return GenerateMenuItems(actions)
    }

    /// The source well. `canEditMask` is `RefineGroup.maskCapable`'s answer --
    /// the recipe's own mask path, not a guess.
    static func sourceWell(
        hasPicture: Bool, canEditMask: Bool, canPaste: Bool
    ) -> GenerateMenuItems {
        var actions: [GenerateAction] = [.chooseFile, .chooseFromLibrary]
        if canPaste { actions.append(.paste) }
        if hasPicture, canEditMask { actions.append(.editMask) }
        if hasPicture { actions.append(.removeSource) }
        return GenerateMenuItems(actions)
    }

    /// One reference picture. Order MATTERS here: on a `primaryIsTarget`
    /// recipe index 0 is the picture being edited, so moving one is a real
    /// instruction and not a convenience.
    static func referenceItem(index: Int, count: Int) -> GenerateMenuItems {
        var actions: [GenerateAction] = []
        if index > 0 { actions.append(.moveLeft) }
        if index < count - 1 { actions.append(.moveRight) }
        actions.append(.replacePicture)
        actions.append(.removeReference)
        return GenerateMenuItems(actions)
    }

    /// The strip's background. Nothing applies to an empty strip with no room,
    /// which is a menu that must not appear.
    static func referenceStrip(count: Int, hasRoom: Bool, canPaste: Bool) -> GenerateMenuItems {
        var actions: [GenerateAction] = []
        if hasRoom { actions.append(.addReference) }
        if hasRoom, canPaste { actions.append(.paste) }
        if count > 0 { actions.append(.removeAllReferences) }
        return GenerateMenuItems(actions)
    }

    static func identityPhoto() -> GenerateMenuItems {
        GenerateMenuItems([.replacePhoto, .removePhoto])
    }

    /// The mask row. Only where a mask has actually been painted -- an
    /// unpainted row's inline button IS "Edit mask…", so a menu repeating it
    /// alone would be furniture.
    static func maskRow(hasMask: Bool) -> GenerateMenuItems {
        guard hasMask else { return GenerateMenuItems([]) }
        return GenerateMenuItems([.editMask, .clearMask])
    }

    /// An adapter row, extending the trained-word menu that is already there.
    /// `trainedWords` are rendered ahead of these by the view, because they
    /// are the row's own vocabulary rather than an action on it.
    static func adapterRow(isAtDefaultStrength: Bool) -> GenerateMenuItems {
        var actions: [GenerateAction] = []
        if !isAtDefaultStrength { actions.append(.resetStrength) }
        actions.append(.removeAdapter)
        return GenerateMenuItems(actions)
    }

    /// A recent prompt. There is no per-entry delete verb on the wire -- the
    /// history route offers `clear` alone -- so "Remove from History" is
    /// deliberately absent rather than shown and broken.
    static func recentPrompt() -> GenerateMenuItems {
        GenerateMenuItems([.usePrompt, .copyPrompt])
    }
}
