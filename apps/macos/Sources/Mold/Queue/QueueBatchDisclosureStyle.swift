import SwiftUI

/// Keep the batch's label actions independent of its disclosure toggle. Separate
/// roots let List retain a selectable native row for each expanded child.
struct QueueBatchDisclosureStyle: DisclosureGroupStyle {
    let id: String

    func makeBody(configuration: Configuration) -> some View {
        Group {
            HStack(spacing: 8) {
                QueueBatchDisclosureToggle(id: id, expanded: configuration.$isExpanded)
                configuration.label
            }
            if configuration.isExpanded { configuration.content }
        }
    }
}

private struct QueueBatchDisclosureToggle: View {
    let id: String
    @Binding var expanded: Bool
    @FocusState private var focused: Bool

    var body: some View {
        Button {
            focused = true
            expanded.toggle()
        } label: {
            Image(systemName: expanded ? "chevron.down" : "chevron.right")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .focusable()
        .focused($focused)
        .onKeyPress(keys: [.space, .return]) { press in
            guard focused, press.modifiers.isEmpty else { return .ignored }
            expanded.toggle()
            return .handled
        }
        .help(expanded ? "Hide the individual jobs in this batch" : "Show the individual jobs in this batch")
        .accessibilityLabel(expanded ? "Collapse batch" : "Expand batch")
        .accessibilityValue(expanded ? "Expanded" : "Collapsed")
        .accessibilityIdentifier("queue-batch-disclosure-" + id)
    }
}
