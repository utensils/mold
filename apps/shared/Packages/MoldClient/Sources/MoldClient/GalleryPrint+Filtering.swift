import Foundation

/// How a picture or clip is framed, as the Library's filter names it.
public enum PrintOrientation: String, Hashable, Sendable, CaseIterable {
    case portrait, landscape, square

    /// Within 1% counts as square: 1024 x 1016 is a square picture to a
    /// person, and an exact test would file it as landscape.
    init?(width: Int?, height: Int?) {
        guard let width, let height, width > 0, height > 0 else { return nil }
        let ratio = Double(width) / Double(height)
        if abs(ratio - 1) <= 0.01 { self = .square } else { self = ratio > 1 ? .landscape : .portrait }
    }
}

public extension GalleryPrint {
    /// nil for a mesh (it has no frame) or a print that recorded no size.
    var orientation: PrintOrientation? {
        guard kind != .mesh else { return nil }
        return PrintOrientation(width: metadata.width, height: metadata.height)
    }

    /// The print's canonical family slug: the recorded family through its
    /// aliases, else the family the generated table names for its model. nil
    /// when neither says.
    var familySlug: String? {
        if let family = metadata.family, !family.isEmpty { return Model.canonicalFamily(family) }
        guard let model = metadata.model, let label = ModelPresentationTable.models[model]?.familyLabel else {
            return nil
        }
        return ModelPresentationTable.familyLabels.filter { $0.value == label }.keys.min()
    }
}
