import Foundation

// How a model is NAMED. `mold_core::model_presentation` is the one authority:
// the server's own fields win, the generated table names a curated id the
// server did not name, and a catalog (Civitai / Hugging Face) install keeps
// its provider title. Nothing here parses a description for a name.
public extension Model {
    /// The server's fields when it sent a title, else the generated table's
    /// entry for this id; nil for a catalog or custom model.
    var presentation: ModelPresentation? {
        if let presentedTitle, !presentedTitle.isEmpty {
            return ModelPresentation(
                title: presentedTitle, variant: presentedVariant,
                summary: presentedSummary, familyLabel: presentedFamilyLabel)
        }
        return ModelPresentationTable.models[name]
    }

    /// The full short name: "FLUX.1 Dev Q4". A catalog install falls to the
    /// same order the server's `human_name` uses (`types.rs`): its provider
    /// title, then a name derived from the id itself.
    var headline: String {
        if let presentation { return presentation.displayName }
        if let displayName, !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return displayName }
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        if name.hasPrefix("cv:") { return "Civitai model #\(name.dropFirst(3))" }
        if name.hasPrefix("hf:") {
            let repo = name.dropFirst(3)
            let title = repo.split(separator: "/").last.map(String.init) ?? String(repo)
            return title.replacingOccurrences(of: "-", with: " ")
                .replacingOccurrences(of: "_", with: " ")
        }
        return name
    }

    /// The model without its variant: "FLUX.1 Dev". A group heading and the
    /// Model column read this, leaving the variant to its own cell.
    var title: String { presentation?.title ?? headline }

    /// What tells siblings apart: "Q4", "Turbo 4-step 768p". nil when the
    /// authority names none (an only child, or a catalog install).
    var variantLabel: String? {
        guard let variant = presentation?.variant, !variant.isEmpty else { return nil }
        return variant
    }

    /// One line on why to pick it. A model the authority does not name
    /// offers its description only when that says something the headline
    /// does not -- a catalog install's description IS its title.
    var summary: String? {
        if let presentation { return presentation.summary.flatMap { $0.isEmpty ? nil : $0 } }
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty || trimmed == headline ? nil : trimmed
    }

    /// The section this model sits under: "FLUX.1", "MiniMax H3".
    var familyLabel: String {
        if let label = presentation?.familyLabel, !label.isEmpty { return label }
        return Self.familyLabel(forFamily: family)
    }

    /// A bare model id's name when no machine's row for it is at hand (a
    /// download, a history entry, a queue row): the generated table's name
    /// for a curated id, else the id itself.
    static func headline(forName name: String) -> String {
        ModelPresentationTable.models[name]?.displayName ?? name
    }

    /// A bare family slug's label, for a section keyed by `family`. A family
    /// the authority has never heard of shows its own slug.
    static func familyLabel(forFamily family: String) -> String {
        ModelPresentationTable.familyLabels[family] ?? family
    }
}
