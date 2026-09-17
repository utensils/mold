import AppKit
import MoldClient
import SwiftUI

// One recent prompt: the row and what it offers on a right-click. Split from
// the group's own shape purely for size.
extension RecentGroup {
    /// One recent prompt's menu. There is no per-entry delete verb on the
    /// wire -- the history route offers `clear` alone -- so "Remove from
    /// History" is absent rather than shown and broken.
    func perform(_ action: GenerateAction, on entry: HistoryEntry) {
        switch action {
        case .usePrompt:
            Self.pick(entry, into: &draft)
        case .copyPrompt:
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(entry.prompt, forType: .string)
        default:
            break
        }
    }

    func row(_ entry: HistoryEntry) -> some View {
        Button { Self.pick(entry, into: &draft) } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.prompt).lineLimit(3)
                Text("\(entry.model) · \(entry.usedAtDate, format: .relative(presentation: .named))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .help("Puts this prompt back. The model and the controls stay as they are.")
        .rowActionMenu(GenerateMenus.recentPrompt()) { perform($0, on: entry) }
    }
}
