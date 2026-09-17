import Foundation

// Parking and restoring conditioning a recipe cannot currently take. Split
// from `DraftMedia`'s own shape purely for size.
public extension DraftMedia {
    /// Moves a value out of the way when the recipe cannot read it, and
    /// brings it back when one that can returns.
    ///
    /// A live value always wins over a parked one: parking is a rescue, not
    /// a history, so a value already sitting in `live` is never overwritten
    /// by whatever happened to be parked earlier.
    static func reconcile<T>(_ live: inout T?, _ parked: inout T?, supported: Bool) {
        if supported {
            if live == nil {
                live = parked
                parked = nil
            }
        } else if live != nil {
            parked = live
            live = nil
        }
    }

    /// Parks/restores the source image and its name TOGETHER -- the two
    /// always travel as a pair, never independently.
    mutating func reconcileSourceImage(supported: Bool) {
        if supported {
            if sourceImage == nil, let restored = parked.sourceImage {
                sourceImage = restored
                sourceImageName = parked.sourceImageName
                sourceImageOriginal = parked.sourceImageOriginal
                sourceImageOriginalName = parked.sourceImageOriginalName
                parked.sourceImage = nil
                parked.sourceImageName = nil
                parked.sourceImageOriginal = nil
                parked.sourceImageOriginalName = nil
            }
        } else if sourceImage != nil {
            parked.sourceImage = sourceImage
            parked.sourceImageName = sourceImageName
            // The UNFITTED copy travels with it, or the picture comes back
            // frozen at the old canvas's crop with nothing left to re-fit from.
            parked.sourceImageOriginal = sourceImageOriginal
            parked.sourceImageOriginalName = sourceImageOriginalName
            sourceImage = nil
            sourceImageName = nil
            sourceImageOriginal = nil
            sourceImageOriginalName = nil
        }
    }

    /// Parks/restores the reference-image list. A stack longer than
    /// `maxCount` parks its TAIL rather than dropping it, so shrinking and
    /// then growing the limit again hands the rest back.
    mutating func reconcileEditImages(supported: Bool, maxCount: Int?) {
        if supported {
            if editImages.isEmpty, !parked.editImages.isEmpty {
                editImages = parked.editImages
                parked.editImages = []
            }
        } else if !editImages.isEmpty {
            parked.editImages = editImages
            editImages = []
        }
        if let maxCount, editImages.count > maxCount {
            parked.editImages = Array(editImages[maxCount...]) + parked.editImages
            editImages = Array(editImages.prefix(maxCount))
        }
    }

    /// The mask. Its `supported` already folds in both `acceptsMask` and
    /// "the source image survived" at the call site (`reconcile(for:)`) --
    /// an orphaned mask over no source is meaningless
    /// (`validation.rs:3101-3107`).
    mutating func reconcileMask(supported: Bool) {
        Self.reconcile(&maskImage, &parked.maskImage, supported: supported)
    }

    mutating func reconcileIdentity(supported: Bool) {
        Self.reconcile(&identity, &parked.identity, supported: supported)
    }

    /// ControlNet. One value, like identity -- `ControlConditioning` already
    /// bundles the picture, the chosen adapter and the scale, so parking it
    /// is a single swap rather than a pair of fields kept in lockstep.
    mutating func reconcileControl(supported: Bool) {
        Self.reconcile(&control, &parked.control, supported: supported)
    }

    /// Parks/restores the adapter stack, truncating to `maxCount` and
    /// parking the tail rather than dropping it -- same shape as
    /// `reconcileEditImages`.
    mutating func reconcileLoras(supported: Bool, maxCount: Int?) {
        if supported {
            if loras.isEmpty, !parked.loras.isEmpty {
                loras = parked.loras
                parked.loras = []
            }
        } else if !loras.isEmpty {
            parked.loras = loras
            loras = []
        }
        if let maxCount, loras.count > maxCount {
            parked.loras = Array(loras[maxCount...]) + parked.loras
            loras = Array(loras.prefix(maxCount))
        }
    }
}
