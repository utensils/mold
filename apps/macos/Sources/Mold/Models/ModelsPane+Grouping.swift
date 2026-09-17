import MoldClient
import SwiftUI

// Which machine the pane is about, which of its models match, and how those
// rows collect into groups. Split from the view for size -- `private` does not
// cross a file boundary, so what the pane reads is `internal` here.
extension ModelsPane {
    /// Deliberately not named `Group`: that shadows SwiftUI's own view inside
    /// `body`, and the resulting errors point everywhere but here.
    struct VariantGroup {
        let title: String
        let repo: String?
        let variants: [Model]

        /// A model with one untagged variant is not a group of anything.
        /// Giving it a heading plus a row leaves the row with nothing to say,
        /// which reads as a rendering bug rather than as a simple model.
        var isSolo: Bool { variants.count == 1 && variants[0].tag == nil }
    }

    var host: MoldHost? { hosts.machine(selected: selectedMachine) }

    var candidates: [Model] {
        guard let host else { return [] }
        let all = installedOnly ? models.ready(on: host.id) : models.generators(on: host.id)
        guard !query.isEmpty else { return all }
        let needle = query.lowercased()
        return all.filter {
            $0.description.lowercased().contains(needle) || $0.name.lowercased().contains(needle)
        }
    }

    var groups: [VariantGroup] {
        Dictionary(grouping: candidates, by: \.baseName)
            .map { _, variants in
                let sorted = variants.sorted { ($0.sizeGb ?? 0) < ($1.sizeGb ?? 0) }
                let lead = sorted[0]
                // The trade-off sentence describes the VARIANT, so it stays
                // on the row. Repeating it in the heading said the same thing
                // twice for every single-variant model.
                return VariantGroup(title: lead.baseTitle, repo: lead.hfRepo, variants: sorted)
            }
            .sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
    }
}
