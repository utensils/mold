import Foundation
import MoldClient

/// Search the entire machine history before limiting visible rows, so older
/// matching prompts remain reachable without expanding every preceding row.
struct RecentPromptPage {
    let visible: [HistoryEntry]
    let remaining: Int
    let showsSearch: Bool

    init(entries: [HistoryEntry], query: String, limit: Int) {
        showsSearch = entries.count > 5 || !query.isEmpty
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let matches = query.isEmpty ? entries : entries.filter {
            $0.prompt.localizedCaseInsensitiveContains(query)
                || $0.model.localizedCaseInsensitiveContains(query)
        }
        visible = Array(matches.prefix(max(0, limit)))
        remaining = matches.count - visible.count
    }
}
