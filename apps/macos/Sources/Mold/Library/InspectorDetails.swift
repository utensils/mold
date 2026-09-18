import MoldClient
import SwiftUI

/// Everything one print recorded about itself.
///
/// WHAT is shown and in what order is `PrintDetails`', which is pure and
/// tested away from any view; this draws it. A group appears only when the
/// print has something to put in it, so a text-to-image print has no Clip
/// heading and a picture that conditioned nothing has no Made from -- an
/// inspector of em-dashes describes the wire format rather than a picture.
struct InspectorDetails: View {
    let entry: LibraryEntry
    let scope: LibraryScope
    let actions: LibraryActions

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(PrintDetails.groups(for: entry)) { group in
                LabeledSection(group.title) {
                    VStack(alignment: .leading, spacing: 6) {
                        // By POSITION, like `RowActionMenu`: two groups can
                        // hold a row of the same name.
                        ForEach(Array(group.rows.enumerated()), id: \.offset) { _, row in
                            line(row)
                        }
                    }
                }
            }
        }
        .font(.caption)
    }

    /// Prose gets its own paragraph under its name and wraps; a figure sits on
    /// the baseline beside it. Both are selectable, because the point of
    /// showing a seed is that somebody copies it.
    @ViewBuilder private func line(_ row: PrintDetailRow) -> some View {
        Group {
            if row.isProse {
                VStack(alignment: .leading, spacing: 2) {
                    Text(row.label).foregroundStyle(.secondary)
                    value(row).font(.callout)
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(row.label)
                        .foregroundStyle(.secondary)
                        .frame(width: Self.labelWidth, alignment: .trailing)
                    value(row)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .rowActionMenu(PrintDetails.menu(for: row, offering: plan.items)) {
            perform($0, on: row)
        }
    }

    private func value(_ row: PrintDetailRow) -> some View {
        Text(row.value)
            .textSelection(.enabled)
            .monospacedDigit()
            // Long values wrap rather than being cut off with an ellipsis:
            // a truncated prompt is a prompt you cannot read.
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Wide enough for the longest label `PrintDetails` can emit, at caption
    /// size, with the value wrapping beside it. Deliberately not `private`:
    /// `LibraryInspectedTests` MEASURES every label against it, because a
    /// label added later is exactly how a column like this drifts into
    /// truncating one row nobody happens to look at. (At a larger text size
    /// the label WRAPS rather than truncating -- nothing sets `lineLimit` --
    /// so the measurement is about the default size, where it matters.)
    static let labelWidth: CGFloat = 104

    /// The Library's own offer for this one print, so the row's menu borrows
    /// Use These Settings with its one wording and its one gate rather than
    /// declaring a second copy. Only that item is taken.
    private var plan: LibraryMenuPlan {
        LibraryMenuPlan(scope: scope.menuKind, count: 1,
                        canReuse: actions.reuse != nil && !scope.isTrash)
    }

    private func perform(_ action: PrintDetailAction, on row: PrintDetailRow) {
        switch action {
        case .copy:
            Clipboard.put(row.value)
        case let .library(item):
            actions.perform(item, on: [entry], scope: scope, open: nil)
        }
    }
}
