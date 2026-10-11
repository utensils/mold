import Foundation

/// How a curated model is named to a person. The server's
/// `mold_core::model_presentation` is the one authority: `/api/models` rows
/// carry these fields, and `ModelPresentationTable` (generated from the same
/// Rust code) names an id no connected machine reports.
public struct ModelPresentation: Sendable, Hashable, Codable {
    /// The model without precision: "FLUX.1 Dev", "MiniMax H3 Frames".
    public let title: String
    /// What tells siblings apart: "Q4", "Turbo 4-step 768p".
    public let variant: String?
    /// One line on why to pick it over its siblings.
    public let summary: String?
    /// The section it sits under: "FLUX.1", "MiniMax H3".
    public let familyLabel: String?

    public init(title: String, variant: String?, summary: String?, familyLabel: String?) {
        self.title = title
        self.variant = variant
        self.summary = summary
        self.familyLabel = familyLabel
    }

    /// Title, then variant — the short alias `display_name` carries.
    public var displayName: String {
        guard let variant, !variant.isEmpty else { return title }
        return "\(title) \(variant)"
    }
}
