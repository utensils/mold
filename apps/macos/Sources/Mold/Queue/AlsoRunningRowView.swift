import MoldClient
import SwiftUI

/// One line of work with no queue row of its own.
struct AlsoRunningRowView: View {
    let row: AlsoRunningRow
    let act: (AlsoRunningActions.Kind) -> Void

    private var actions: AlsoRunningActions { AlsoRunningActions(row) }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(row.title).font(.body)
                    if let subject = row.subject {
                        Text(subject)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Text(row.isStale ? "\(row.detail) — last heard" : row.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let progress = row.progress {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 180)
                        // a11y: the sentence above IS this meter's label, and
                        // reading both aloud says the same thing twice.
                        .accessibilityHidden(true)
                }
            }
            Spacer(minLength: 0)
            controls
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(row.title). \(row.detail)")
        .rowActionMenu(actions.offered(), perform: act)
    }

    @ViewBuilder private var controls: some View {
        // Inline, from the SAME list the menu draws -- a control the menu
        // does not offer would be a second opinion about what a row can do.
        ForEach(actions.offered().filter { !$0.isSeparator }, id: \.title) { action in
            if let kind = action.kind {
                Button(action.title) { act(kind) }
                    .buttonStyle(.borderless)
                    .font(.caption)
            }
        }
    }
}
