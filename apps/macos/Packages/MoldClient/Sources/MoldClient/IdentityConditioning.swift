import Foundation

/// One identity-conditioning photograph, base64-encoded.
public struct IdentityPhoto: Codable, Hashable, Sendable, Identifiable {
    /// Base64 PNG/JPEG bytes.
    public var encoded: String
    public var name: String
    public var id: String { name + String(encoded.count) }

    public init(encoded: String, name: String) {
        self.encoded = encoded
        self.name = name
    }
}

/// Face-identity conditioning for a draft.
///
/// One value whether it ends up carrying one photograph or four -- the wire
/// shape (`id_image` vs `id_images`) is chosen at request time by
/// `wire(maxPhotos:)`, never stored here, because it depends on the HOST the
/// request is going to, not on the draft.
public struct IdentityConditioning: Codable, Hashable, Sendable {
    public var photos: [IdentityPhoto]
    /// `identity.rs:295-298`: `[0.0, ID_WEIGHT_MAX]`.
    public var weight: Double = Identity.weightDefault
    /// `identity.rs:560-566`: must stay `< steps`. See `Identity.startStepRange`.
    public var startStep: Int = Identity.startStepDefault

    public init(
        photos: [IdentityPhoto], weight: Double = Identity.weightDefault,
        startStep: Int = Identity.startStepDefault
    ) {
        self.photos = photos
        self.weight = weight
        self.startStep = startStep
    }

    /// The wire shape, chosen from what the HOST understands.
    ///
    /// `id_image` and `id_images` are the SAME field in two shapes, and
    /// supplying both is a hard validation error, never a precedence rule
    /// (`identity.rs:981`) -- so they come from this one `switch` and can
    /// never both exist. `maxPhotos` is `Capabilities.maxIdentityPhotos`,
    /// which is 1 on a host without `multi_photo` even if it advertises a
    /// larger `max_photos` (`types.rs:11563-11572`), so a host that takes
    /// one photo gets the singular form from a longer list rather than the
    /// plural form truncated to one entry.
    public enum Wire: Equatable {
        case single(IdentityPhoto)
        case several([IdentityPhoto])
    }

    public func wire(maxPhotos: Int) -> Wire? {
        let taken = Array(photos.prefix(Swift.max(maxPhotos, 0)))
        switch taken.count {
        case 0: return nil
        case 1: return .single(taken[0])
        default: return .several(taken)
        }
    }
}

public enum Identity {
    /// `identity.rs:298` (`ID_WEIGHT_MAX`).
    public static let weightRange: ClosedRange<Double> = 0 ... 3
    /// `identity.rs:295` (`ID_WEIGHT_DEFAULT`).
    public static let weightDefault = 1.0
    /// Presentation only -- not a server contract.
    public static let weightStep = 0.05
    /// `identity.rs:302` (`ID_START_STEP_DEFAULT`).
    public static let startStepDefault = 0
    /// `identity.rs:400` (`ID_IMAGES_MAX`).
    public static let maxPhotosCeiling = 4

    /// `id_start_step` must be strictly less than the run's own step count
    /// (`identity.rs:560-566`, `validate_id_start_step`), so the bound MOVES
    /// with the Steps control rather than being a fixed constant -- dragging
    /// Steps down after Start step was set must not silently arm a 422.
    public static func startStepRange(steps: Int) -> ClosedRange<Int> {
        0 ... Swift.max(steps - 1, 0)
    }
}
